"""Starting a stored configuration and keeping what the start reported."""

import sys
from pathlib import Path

import pytest

from vllm_untwisted.engine import EngineConfig, EngineState, Supervisor
from vllm_untwisted.manager import Manager
from vllm_untwisted.store import Store
from vllm_untwisted.store.db import DRAFT, KNOWN_GOOD, REGRESSED

FAKE = str(Path(__file__).parent / "fake_engine.py")
FINGERPRINT = {"gpu_models": "TestCard", "pkg.vllm": "0.28.0"}


def cfg(name="c", *extra: str) -> EngineConfig:
    return EngineConfig(name=name, model="a-model", args=list(extra),
                        launcher=(sys.executable, FAKE))


@pytest.fixture
def manager(tmp_path):
    store = Store(tmp_path / "s.db")
    sup = Supervisor(start_timeout=15.0, poll_interval=0.05, shm_root=str(tmp_path))
    yield Manager(store, sup, fingerprint=FINGERPRINT)
    store.close()


async def test_a_successful_start_is_recorded_with_what_it_measured(manager):
    cid = manager.store.add(cfg())
    started = await manager.start(cid)
    assert started.ok

    run = manager.store.get(cid).last_run
    assert run.became_ready
    assert run.startup_seconds > 0
    assert run.facts["maximum_concurrency"] == "1.00"
    assert run.compile_state in {"hit", "populating", "disabled", "unknown"}
    await manager.stop()


async def test_a_run_says_what_it_was_taken_on(manager):
    """A measurement is of one machine in one state, so it carries that state."""
    cid = manager.store.add(cfg())
    await manager.start(cid)
    assert manager.store.get(cid).last_run.fingerprint == FINGERPRINT
    await manager.stop()


async def test_starting_promotes_a_draft(manager):
    cid = manager.store.add(cfg())
    assert manager.store.get(cid).status == DRAFT
    await manager.start(cid)
    await manager.stop()
    assert manager.store.get(cid).status == KNOWN_GOOD


async def test_a_failed_start_records_why_and_leaves_it_a_draft(manager):
    cid = manager.store.add(cfg("c", "--emit-oom", "--exit-before-ready", "1"))
    started = await manager.start(cid)
    assert not started.ok

    entry = manager.store.get(cid)
    assert entry.status == DRAFT  # it has still never worked
    assert entry.last_run.failure_kind == "oom"
    assert not entry.last_run.became_ready
    await manager.stop()


async def test_a_configuration_that_stops_working_is_regressed(manager):
    cid = manager.store.add(cfg())
    await manager.start(cid)
    await manager.stop()

    manager.store.update(cid, cfg("c", "--exit-before-ready", "1"))
    await manager.start(cid)
    await manager.stop()
    assert manager.store.get(cid).status == REGRESSED


async def test_one_start_is_one_run_even_when_the_engine_dies(manager):
    """Recording the death separately would double-count every start."""
    cid = manager.store.add(cfg("c", "--die-after-ready", "0.2"))
    await manager.start(cid)
    import asyncio
    await asyncio.sleep(1.0)
    assert manager.supervisor.state is EngineState.FAILED
    await manager.stop()

    runs = manager.store.runs(cid)
    assert len(runs) == 1
    assert runs[0].outcome == "died"
    assert runs[0].became_ready  # it did serve, which is a different fact


async def test_an_unknown_configuration_is_an_error(manager):
    with pytest.raises(KeyError):
        await manager.start("nope")
