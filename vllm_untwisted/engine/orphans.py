"""Reclaim engines a manager did not live long enough to stop.

Shutdown is not always orderly. A SIGKILL, an OOM kill, or a crash in the manager leaves
its engine running -- and because engines are deliberately spawned into their own session
so a stray Ctrl-C cannot kill one mid-serve, nothing else is going to clean it up either.
The next start then finds the GPU already occupied, which reads as a memory bug in
whatever changed since.

Inside a container this is easy: the PID namespace is ours, so everything in it is ours,
and a restart can reap the lot. **Outside one it is not**, and untwisted runs outside one
for all of development. Killing every `vllm` on the box would take out whatever the
developer was running from their own shell, which is a worse failure than the one being
fixed.

So engines are marked, and only marked processes are candidates. The second half of the
rule is what makes it safe to run automatically:

> Reap a marked engine only when **its parent is gone**.

A manager that is alive is still the parent of its own engine, so a second manager -- the
CLI's embedded server, say, started while a serving manager is up -- cannot reap an engine
that is still being supervised. An orphan has been reparented away from its manager, and
an orphan is the only thing this touches.

The marker is the store the manager was using, so two installations on one box do not
reach into each other's engines, and a restarted manager still recognises what it left
behind.
"""

from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from pathlib import Path

#: Set in every engine's environment. Its value is the owning manager's store.
OWNER_VAR = "UNTWISTED_ENGINE_OWNER"


@dataclass(frozen=True, slots=True)
class Orphan:
    pid: int
    owner: str
    cmdline: str


def _read(pid: str, name: str) -> str:
    try:
        return (Path("/proc") / pid / name).read_bytes().decode(errors="replace")
    except OSError:
        return ""


def _environ(pid: str) -> dict[str, str]:
    raw = _read(pid, "environ")
    out: dict[str, str] = {}
    for entry in raw.split("\0"):
        key, sep, value = entry.partition("=")
        if sep:
            out[key] = value
    return out


def _ppid(pid: str) -> int | None:
    # Field 4 of /proc/pid/stat, read after the last ')' because a process name can
    # contain spaces and parentheses.
    stat = _read(pid, "stat")
    tail = stat.rpartition(")")[2].split()
    return int(tail[1]) if len(tail) > 1 else None


def find(owner: str) -> list[Orphan]:
    """Marked engines belonging to `owner` whose parent is no longer alive."""
    found: list[Orphan] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        env = _environ(entry)
        if env.get(OWNER_VAR) != owner:
            continue
        ppid = _ppid(entry)
        if ppid is None:
            continue
        # Reparented: its manager is gone. `ppid == 1` is the usual case; a subreaper
        # (a user systemd, say) can adopt it instead, so an unreadable or dead parent
        # counts too. A parent that is alive means somebody is still supervising it.
        if ppid != 1 and Path(f"/proc/{ppid}").exists():
            continue
        found.append(Orphan(int(entry), owner, _read(entry, "cmdline").replace("\0", " ")))
    return found


def reap(owner: str, dry_run: bool = False) -> list[Orphan]:
    """Terminate orphaned engines and return what was reclaimed.

    Signals the process group, not the process: an engine has children of its own -- an
    EngineCore among them -- and killing only the parent strands them still holding VRAM,
    which is the failure this exists to prevent rather than to reproduce.
    """
    reclaimed: list[Orphan] = []
    for orphan in find(owner):
        if not dry_run:
            try:
                os.killpg(os.getpgid(orphan.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(orphan.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    continue
        reclaimed.append(orphan)
    return reclaimed
