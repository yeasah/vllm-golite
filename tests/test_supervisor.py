"""Lifecycle tests against a fake engine.

Every failure path here is one that is otherwise only reachable by breaking a real
engine, which is exactly why they are worth having: a classifier nobody has watched fire
is a comment.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

import pytest

from golite.engine import EngineConfig, EngineState, Supervisor
from golite.engine.state import FailureKind
from golite.engine.supervisor import EngineStartError, running

FAKE = str(Path(__file__).parent / "fake_engine.py")


def cfg(*extra: str) -> EngineConfig:
    return EngineConfig(
        name="fake", model="a-model", args=list(extra), launcher=(sys.executable, FAKE)
    )


@pytest.fixture
def sup() -> Supervisor:
    return Supervisor(start_timeout=15.0, poll_interval=0.05)


async def test_starts_and_reports_ready(sup):
    async with running(cfg(), supervisor=sup) as s:
        assert s.state is EngineState.READY
        assert s.record.pid is not None
        # What a model change costs, recorded on every start rather than in a special run.
        assert s.record.startup_seconds is not None


async def test_ready_engine_carries_what_the_start_measured(sup):
    async with running(cfg(), supervisor=sup) as s:
        assert s.record.facts["maximum_concurrency"] == "1.03"
        assert s.record.facts["kv_cache_memory_suggestion"] == "1323302912"


async def test_oom_is_classified_not_just_an_exit_code(sup):
    await sup.start(cfg("--emit-oom", "--exit-before-ready", "1"))
    assert sup.state is EngineState.FAILED
    assert sup.record.failure.kind is FailureKind.OOM
    assert sup.record.failure.exit_code == 1
    assert sup.record.failure.log_tail  # the evidence survives
    await sup.stop()


async def test_unclassified_exit_still_reports_the_code_and_the_tail(sup):
    await sup.start(cfg("--exit-before-ready", "3"))
    assert sup.record.failure.kind is FailureKind.EXIT_BEFORE_READY
    assert sup.record.failure.exit_code == 3
    assert any("starting engine" in line for line in sup.record.failure.log_tail)
    await sup.stop()


async def test_a_hang_is_a_timeout_not_a_wait_forever():
    sup = Supervisor(start_timeout=1.0, poll_interval=0.05)
    await sup.start(cfg("--hang"))
    assert sup.record.failure.kind is FailureKind.START_TIMEOUT
    await sup.stop()


async def test_dying_while_ready_is_a_failure_not_a_stop(sup):
    await sup.start(cfg("--die-after-ready", "0.2"))
    assert sup.state is EngineState.READY
    await asyncio.sleep(1.0)
    assert sup.state is EngineState.FAILED
    assert sup.record.failure.kind is FailureKind.DIED_WHILE_READY
    await sup.stop()


async def test_stop_kills_the_whole_process_group(sup):
    """The failure this guards: vLLM's children outlive a naive kill and keep holding
    VRAM, so the *next* start looks like a code regression."""
    await sup.start(cfg("--spawn-child"))
    pgid = os.getpgid(sup.record.pid)
    await sup.stop()

    # Nothing left in the engine's group. os.killpg with signal 0 only probes.
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


async def test_will_not_start_a_second_engine_over_a_running_one(sup):
    async with running(cfg(), supervisor=sup):
        with pytest.raises(EngineStartError):
            await sup.start(cfg())


async def test_a_failed_start_leaves_nothing_behind(sup):
    await sup.start(cfg("--exit-before-ready", "1"))
    await sup.stop()
    assert sup.state is EngineState.STOPPED
    assert sup._handle is None and sup._pump is None
