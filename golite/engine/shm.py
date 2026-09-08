"""Reclaim shared-memory regions that a crashed engine left behind.

The worst-shaped bug this project has met. vLLM's CPU offload region lives at
`/dev/shm/vllm_offload_{engine_id}.mmap` and is unlinked by `cleanup()` only on a
graceful shutdown. After a crash the file survives, and the next engine's constructor
does this:

    try:    fd = os.open(path, O_CREAT | O_EXCL | O_RDWR)   # creator
    except FileExistsError:
        fd = os.open(path, O_RDWR)                          # joiner
        _wait_for_file_size(fd, self.total_size_bytes)

The joiner branch exists so sibling workers can attach to the region the creator made.
A leftover file of the *same size* -- which is exactly what a retry of the same
configuration produces -- satisfies that wait immediately, so the new engine **mmaps a
dead process's memory as its own KV cache** and serves from it.

Nothing about that announces itself. The reported symptoms are "bad address", unrelated
podman failures, nothing in dmesg, and a reboot appearing to fix it -- because a reboot
clears tmpfs. It is the archetype of the failure this project cares about most: a crash
that seeds silent corruption in the *next* run, where the damage is a wrong number rather
than a stopped process.

So the supervisor sweeps before every start. The one rule that makes that safe is that a
region is only removed if **no live process has it mapped or open** -- deleting a region
another engine is serving from would trade one corruption for another.

`vllm_ec_{engine_id}.mmap` (the encoder-cache transfer region) has the identical
per-engine shape and is swept the same way.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SHM_ROOT = "/dev/shm"
#: Per-engine regions keyed on an engine id, unlinked only on graceful shutdown.
REGION_GLOBS = ("vllm_offload_*.mmap", "vllm_ec_*.mmap")


@dataclass(frozen=True, slots=True)
class Region:
    path: Path
    size_bytes: int
    in_use: bool


def _paths_in_use(root: str = SHM_ROOT) -> set[str]:
    """Every file some live process has mapped or open.

    Reads `/proc/*/maps` and `/proc/*/fd`. Both are readable for this user's own
    processes and not for another user's -- so a region held by a different user could
    look free. That is acceptable here because we would not be able to unlink it either,
    and unlink is the only thing this gates.
    """
    used: set[str] = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            with (proc / "maps").open() as fh:
                for line in fh:
                    if root in line:
                        used.add(line.rsplit(" ", 1)[-1].strip())
        except OSError:
            pass  # process exited, or not ours to read
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith(root):
                    used.add(target)
        except OSError:
            pass
    return used


def survey(root: str = SHM_ROOT) -> list[Region]:
    """Every engine region present, and whether anything is still using it."""
    used = _paths_in_use(root)
    out: list[Region] = []
    base = Path(root)
    for glob in REGION_GLOBS:
        for path in sorted(base.glob(glob)):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            out.append(Region(path, size, str(path) in used))
    return out


def sweep(root: str = SHM_ROOT, dry_run: bool = False) -> list[Region]:
    """Unlink regions nothing is using. Returns what was reclaimed.

    Deliberately reports rather than staying quiet: a start that silently reclaimed
    8 GB of someone else's leftovers is a fact worth having in the log when the *next*
    surprise arrives.
    """
    reclaimed: list[Region] = []
    for region in survey(root):
        if region.in_use:
            continue
        if not dry_run:
            try:
                region.path.unlink()
            except OSError:
                continue  # vanished, or not ours to remove
        reclaimed.append(region)
    return reclaimed
