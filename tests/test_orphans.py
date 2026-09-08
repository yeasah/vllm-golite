"""Reclaiming engines a manager did not live long enough to stop.

Both directions are load-bearing and the second one more: reaping too eagerly on a
machine that is not a container would kill whatever the developer is running from their
own shell, which is worse than the leak being fixed.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time

import pytest

from vllm_untwisted.engine import orphans

OWNER = "/test/store.db"
SLEEPER = "import time; time.sleep(60)"


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def orphan():
    """A marked process whose parent has exited, as a killed manager leaves behind.

    Made by a child that spawns a detached grandchild and exits at once, so the
    grandchild is reparented exactly the way a SIGKILLed manager's engine is.
    """
    spawner = (
        "import os, subprocess, sys\n"
        f"p = subprocess.Popen([sys.executable, '-c', {SLEEPER!r}],\n"
        f"    env={{**os.environ, {orphans.OWNER_VAR!r}: {OWNER!r}}},\n"
        "    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
        "    start_new_session=True)\n"
        "print(p.pid)\n"
    )
    pid = int(subprocess.run([sys.executable, "-c", spawner],
                             capture_output=True, text=True, check=True).stdout.strip())
    for _ in range(100):  # wait for reparenting
        if orphans.find(OWNER):
            break
        time.sleep(0.05)
    yield pid
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def test_an_orphaned_engine_is_found_and_reaped(orphan):
    found = orphans.find(OWNER)
    assert [o.pid for o in found] == [orphan]

    reclaimed = orphans.reap(OWNER)
    assert [o.pid for o in reclaimed] == [orphan]
    assert _wait_gone(orphan)
    assert orphans.find(OWNER) == []


def test_a_dry_run_reports_without_killing(orphan):
    assert [o.pid for o in orphans.reap(OWNER, dry_run=True)] == [orphan]
    os.kill(orphan, 0)  # still there


def test_an_engine_whose_manager_is_alive_is_left_alone():
    """The rule that makes this safe to run automatically. A second manager -- the CLI's
    embedded server started while one is serving -- must not reap a supervised engine."""
    child = subprocess.Popen(
        [sys.executable, "-c", SLEEPER],
        env={**os.environ, orphans.OWNER_VAR: OWNER}, start_new_session=True)
    try:
        time.sleep(0.3)
        assert child.pid not in [o.pid for o in orphans.find(OWNER)]
        assert orphans.reap(OWNER) == []
        assert child.poll() is None
    finally:
        child.kill()
        child.wait()


def test_another_owners_engine_is_not_ours_to_touch():
    """Two installations on one box, and the developer's own vLLM, are none of our
    business. Only our own marker counts."""
    spawner = (
        "import os, subprocess, sys\n"
        f"p = subprocess.Popen([sys.executable, '-c', {SLEEPER!r}],\n"
        f"    env={{**os.environ, {orphans.OWNER_VAR!r}: '/somebody/else.db'}},\n"
        "    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
        "    start_new_session=True)\n"
        "print(p.pid)\n"
    )
    pid = int(subprocess.run([sys.executable, "-c", spawner],
                             capture_output=True, text=True, check=True).stdout.strip())
    try:
        time.sleep(0.5)
        assert orphans.find(OWNER) == []
        assert orphans.reap(OWNER) == []
        os.kill(pid, 0)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def test_an_unmarked_process_is_never_a_candidate():
    child = subprocess.Popen([sys.executable, "-c", SLEEPER], start_new_session=True)
    try:
        time.sleep(0.2)
        assert child.pid not in [o.pid for o in orphans.find(OWNER)]
    finally:
        child.kill()
        child.wait()
