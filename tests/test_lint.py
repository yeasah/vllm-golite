"""Checking a stored configuration against traps already paid for.

Two rules, not a catalogue: enough to show the mechanism carries a rule that reads the
invocation alone and one that reads it against what previous runs reported. Each test
puts the trap back and watches the rule fire, because a rule nobody has seen trigger is
a comment.
"""

import pytest

from vllm_untwisted.engine import EngineConfig
from vllm_untwisted.store import Store
from vllm_untwisted.store.lint import NOTE, WARN, lint


@pytest.fixture
def store():
    with Store() as s:
        yield s


def entry(store, *args, name="c"):
    cid = store.add(EngineConfig(name=name, model="/m", args=list(args)))
    return store.get(cid)


def rules(findings):
    return {f.rule for f in findings}


def test_auto_context_with_batching_is_flagged(store):
    e = entry(store, "--max-model-len", "auto", "--max-num-seqs", "4")
    found = lint(e)
    assert rules(found) == {"overcommitted-concurrency"}
    assert found[0].severity == WARN
    assert "--max-num-seqs 4" in found[0].message


def test_auto_context_with_one_sequence_is_fine(store):
    # The shape every configuration the user verified actually uses.
    assert lint(entry(store, "--max-model-len", "auto", "--max-num-seqs", "1")) == []


def test_batching_with_a_pinned_context_is_fine(store):
    # It is `auto` that makes the concurrency figure a tautology, not batching.
    assert lint(entry(store, "--max-model-len", "262144", "--max-num-seqs", "4")) == []


def test_a_pinned_kv_cache_is_flagged(store):
    found = lint(entry(store, "--kv-cache-memory=1323302912"))
    assert rules(found) == {"pinned-kv-cache-memory"}


def test_the_pin_is_found_in_either_syntax(store):
    # Both appear in the same script, which is why nothing may assume one of them.
    a = lint(entry(store, "--kv-cache-memory=123", name="a"))
    b = lint(entry(store, "--kv-cache-memory", "123", name="b"))
    assert rules(a) == rules(b) == {"pinned-kv-cache-memory"}


def test_a_pin_is_checked_against_what_a_run_actually_measured(store):
    """The evidence half. A pin cannot be revised because it suppresses the profile run,
    so a recorded run is the only thing that can contradict it."""
    cid = store.add(EngineConfig(name="c", model="/m",
                                 args=["--kv-cache-memory", str(int(2.0 * 2**30))]))
    store.record_run(cid, outcome="ready", facts={"available_kv_cache_gib": "3.41"})
    found = lint(store.get(cid))
    notes = [f for f in found if f.severity == NOTE]
    assert notes and "3.41" in notes[0].message


def test_a_pin_matching_the_evidence_gets_no_second_finding(store):
    cid = store.add(EngineConfig(name="c", model="/m",
                                 args=["--kv-cache-memory", str(int(3.41 * 2**30))]))
    store.record_run(cid, outcome="ready", facts={"available_kv_cache_gib": "3.41"})
    assert [f.severity for f in lint(store.get(cid))] == [WARN]


def test_a_clean_configuration_reports_nothing(store):
    assert lint(entry(store, "--max-num-seqs", "1", "--enable-prefix-caching")) == []


def test_flag_reading_distinguishes_absent_from_valueless():
    c = EngineConfig(name="c", model="/m",
                     args=["--enable-prefix-caching", "--max-num-seqs", "1"])
    assert c.flag("enable-prefix-caching") == ""   # present, no value
    assert c.flag("max-num-seqs") == "1"
    assert c.flag("gpu-memory-utilization") is None  # absent
