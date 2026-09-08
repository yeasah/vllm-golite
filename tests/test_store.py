"""The store, and the properties that make it better than a directory of scripts."""

import pytest

from vllm_untwisted.engine import EngineConfig
from vllm_untwisted.store import Store
from vllm_untwisted.store.db import DRAFT, KNOWN_GOOD, REGRESSED


def cfg(name="a", **kw) -> EngineConfig:
    return EngineConfig(name=name, model=kw.pop("model", "/ckpt/m"), **kw)


@pytest.fixture
def store():
    with Store() as s:
        yield s


def test_lookup_by_id_or_name(store):
    cid = store.add(cfg("qwen-long"))
    assert store.get(cid).id == store.get("qwen-long").id == cid


def test_names_are_unique(store):
    store.add(cfg("dup"))
    with pytest.raises(ValueError):
        store.add(cfg("dup"))


def test_renaming_keeps_identity_and_history(store):
    """Why identity is opaque: runs reference the configuration, and names get revised."""
    cid = store.add(cfg("old-name"))
    store.record_run(cid, outcome="ready")
    store.rename(cid, "better-name")

    entry = store.get(cid)
    assert entry.name == "better-name"
    assert entry.run_count == 1
    assert store.get("old-name") is None


def test_a_configuration_that_never_started_is_a_draft(store):
    """Most of the cure for the rot: in a shell script a commented-out invocation and a
    working one look identical."""
    cid = store.add(cfg("untried"))
    assert store.get(cid).status == DRAFT
    assert store.get(cid).is_draft


def test_status_tracks_the_latest_outcome(store):
    cid = store.add(cfg("c"))
    store.record_run(cid, outcome="ready")
    assert store.get(cid).status == KNOWN_GOOD
    store.record_run(cid, outcome="failed", failure_kind="oom")
    assert store.get(cid).status == REGRESSED
    store.record_run(cid, outcome="ready")
    assert store.get(cid).status == KNOWN_GOOD


def test_a_failure_before_any_success_is_still_a_draft(store):
    cid = store.add(cfg("c"))
    store.record_run(cid, outcome="failed", failure_kind="oom")
    assert store.get(cid).status == DRAFT


def test_runs_keep_what_the_start_measured(store):
    cid = store.add(cfg("c"))
    store.record_run(cid, outcome="ready", startup_seconds=24.1, compile_state="hit",
                     facts={"available_kv_cache_gib": "3.41"},
                     fingerprint={"gpu": "RTX 5070 Ti", "vllm": "0.28.0"})
    run = store.get(cid).last_run
    assert run.facts["available_kv_cache_gib"] == "3.41"
    # A measurement is of one machine in one state, so it carries that state itself.
    assert run.fingerprint["vllm"] == "0.28.0"
    assert run.compile_state == "hit"


def test_derived_configurations_point_at_their_parent(store):
    parent = store.add(cfg("hand-written"))
    child = store.add(cfg("tier2-result"), origin="derived", derived_by="fit-shortlist",
                      derived_from=parent)
    assert store.get(child).derived_from == parent
    assert store.get(child).origin == "derived"


def test_deleting_a_configuration_takes_its_runs(store):
    cid = store.add(cfg("c"))
    store.record_run(cid, outcome="ready")
    store.delete(cid)
    assert store.get(cid) is None
    assert store.db.execute("SELECT COUNT(*) c FROM runs").fetchone()["c"] == 0


def test_it_survives_being_closed(tmp_path):
    path = tmp_path / "sub" / "store.db"  # parent directory created on demand
    with Store(path) as s:
        cid = s.add(cfg("persistent", args=["--max-num-seqs", "1"]))
        s.record_run(cid, outcome="ready")
    with Store(path) as s:
        entry = s.get("persistent")
        assert entry.id == cid
        assert entry.status == KNOWN_GOOD
        assert entry.config.args == ["--max-num-seqs", "1"]


def test_the_invocation_round_trips(store):
    # The cases the shell scripts contain: JSON with quotes, multi-valued flags, env.
    original = cfg("round-trip",
                   args=["--kv-transfer-config", '{"kv_connector":"X","kv_role":"kv_both"}',
                         "--cudagraph-capture-sizes", "1", "2", "4"],
                   env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    cid = store.add(original)
    back = store.get(cid).config
    assert back.args == original.args
    assert back.env == original.env
    assert back.command_line() == original.command_line()


def test_unknown_reference_raises(store):
    with pytest.raises(KeyError):
        store.record_run("nope", outcome="ready")
