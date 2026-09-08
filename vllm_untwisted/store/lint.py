"""Check a stored configuration against traps we have already paid for.

The knowledge layer in its cheapest possible form: no GPU, no engine start, no download.
It exists early because it is also a test of whether the store holds enough structure to
reason about -- a rule that cannot be written against an entry is a sign the entry is
still just text.

Deliberately not exhaustive. Two rules are enough to show the mechanism carries both
kinds: one that reads the invocation alone, and one that reads it against what previous
runs of it actually reported. Rules are added when a trap costs someone something, not
to fill out a list.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vllm_untwisted.store.db import ConfigEntry

WARN = "warn"
NOTE = "note"


@dataclass(frozen=True, slots=True)
class Finding:
    rule: str
    severity: str
    message: str


def overcommitted_concurrency(entry: ConfigEntry) -> Iterable[Finding]:
    """`--max-model-len auto` with `--max-num-seqs > 1` is overcommitted by construction.

    `auto` sets `max_model_len` *to* the KV capacity, so the engine reports "Maximum
    concurrency: 1.00x" as a tautology rather than an observation -- there is exactly
    enough cache for one request at the declared length. Any `max_num_seqs` above one is
    then admitting batches the configuration was never sized for, and the failure lands
    mid-session rather than at startup.
    """
    seqs = entry.config.flag("max-num-seqs")
    if entry.config.flag("max-model-len") == "auto" and seqs and int(seqs) > 1:
        yield Finding(
            "overcommitted-concurrency", WARN,
            f"--max-model-len auto sizes the context to the whole KV cache, so "
            f"--max-num-seqs {seqs} is overcommitted by construction. It will not fail "
            f"at startup. Pin a context length, or accept one sequence.")


def pinned_kv_cache_memory(entry: ConfigEntry) -> Iterable[Finding]:
    """Pinning `--kv-cache-memory` freezes a number and blinds the thing that produced it.

    The pin suppresses the profile run, and with it all memory reporting -- so the value
    can never be revised, and nothing will notice if it becomes wrong. It is worse than
    it looks because of where such values come from: vLLM prints two suggestions, and a
    start that was compiling at the same time profiles a larger transient and so
    recommends a smaller cache. A value captured from that start is permanently 14% short
    on the configuration where this was measured.
    """
    pinned = entry.config.flag("kv-cache-memory")
    if not pinned:
        return
    yield Finding(
        "pinned-kv-cache-memory", WARN,
        f"--kv-cache-memory={pinned} suppresses the profile run and all memory "
        f"reporting, so this value can never be revised or checked.")

    # The evidence half: if this configuration has ever run, say what it actually got.
    for run in _runs_with(entry, "available_kv_cache_gib"):
        measured = float(run.facts["available_kv_cache_gib"]) * 2**30
        if abs(measured - int(pinned)) / measured > 0.05:
            yield Finding(
                "pinned-kv-cache-memory", NOTE,
                f"a recorded run of this configuration had "
                f"{run.facts['available_kv_cache_gib']} GiB of KV cache available, "
                f"against the {int(pinned) / 2**30:.2f} GiB pinned here.")
        break


def _runs_with(entry: ConfigEntry, key: str):
    run = entry.last_run
    return [run] if run is not None and key in run.facts else []


#: Order is the order findings are reported in.
RULES: tuple[Callable[[ConfigEntry], Iterable[Finding]], ...] = (
    overcommitted_concurrency,
    pinned_kv_cache_memory,
)


def lint(entry: ConfigEntry) -> list[Finding]:
    return [f for rule in RULES for f in rule(entry)]
