"""Read vLLM's startup log for two things: why it failed, and what it measured.

The failure half is not optional. vLLM reports its interesting failures rather than
raising them, so an exit code alone cannot separate an OOM at profile time from a
checkpoint that was never there. It also logs `[ERROR]` for things that are not errors
at all -- undocumented processor kwargs, absent ROCm modules -- so classification has to
be specific patterns rather than a search for the word.

The measurement half is the more valuable one and easy to miss. An engine start costs a
whole process launch, and while it starts vLLM prints numbers no amount of `config.json`
arithmetic can produce: free versus total memory, where the memory actually went, the
KV cache it got, and how long each phase took. Harvesting those makes every launch a
tier-2 measurement instead of only a launch.

Patterns carry their provenance. Most are now `attested` against a real capture
(Qwen3.8-27B EXL3 3.00bpw, turboquant KV, fork v0.28.0, 2026-09-07) kept as
`tests/data/`. vLLM's log text is not an API, so `python -m golite.calibrate <log>`
re-checks the table against any capture and reports what stopped matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from golite.engine.state import FailureKind

ATTESTED = "attested"   # matched against a real capture kept in tests/data/
LIKELY = "likely"       # standard library or framework text, not seen here
GUESSED = "guessed"     # plausible shape, never checked


@dataclass(frozen=True, slots=True)
class Pattern:
    #: One key per capture group, in order.
    keys: tuple[str, ...]
    regex: re.Pattern[str]
    provenance: str


def _p(keys: str, pattern: str, provenance: str = ATTESTED) -> Pattern:
    return Pattern(tuple(keys.split()), re.compile(pattern), provenance)


#: The startup summary. One line carries most of the memory story, which is why several
#: patterns read the same sentence:
#:
#:   Free memory on device (15.28/15.51 GiB) on startup. Desired GPU memory utilization
#:   is (0.92, 14.27 GiB). Actual usage is 10.47 GiB for consumed memory (weights +
#:   non-torch), 0.79 GiB for peak activation, and 0.04 GiB for CUDAGraph memory.
#:   Replace gpu_memory_utilization config with `--kv-cache-memory=3031561913` (2.82
#:   GiB) to fit into requested memory, or `--kv-cache-memory=4121221120` (3.84 GiB) to
#:   fully utilize gpu memory. Current kv cache memory in use is 3.01 GiB.
FACTS: tuple[Pattern, ...] = (
    # The ceiling on --gpu-memory-utilization is the ratio of these two: a fraction of
    # *total* that must fit within *free*. The gap is driver and context overhead.
    _p("memory_free_gib memory_total_gib",
       r"Free memory on device \(([\d.]+)/([\d.]+)\s*GiB\)"),
    _p("gpu_memory_utilization gpu_memory_budget_gib",
       r"Desired GPU memory utilization is \(([\d.]+),\s*([\d.]+)\s*GiB\)"),
    _p("consumed_memory_gib", r"Actual usage is ([\d.]+)\s*GiB for consumed memory"),
    # The transient the profiler *does* see. It does not vary with cached context, which
    # is why a config can pass here and still OOM mid-session.
    _p("peak_activation_gib", r"([\d.]+)\s*GiB for peak activation"),
    _p("cudagraph_memory_gib", r"([\d.]+)\s*GiB for CUDAGraph memory"),
    # Two suggestions, not one, and they bracket what is actually running. Neither is
    # safe to apply blindly -- see docs/design.md.
    _p("kv_cache_memory_requested kv_cache_requested_gib",
       r"--kv-cache-memory=(\d+)`?\s*\(([\d.]+)\s*GiB\) to fit into requested"),
    _p("kv_cache_memory_full kv_cache_full_gib",
       r"--kv-cache-memory=(\d+)`?\s*\(([\d.]+)\s*GiB\) to fully utilize"),
    _p("kv_cache_in_use_gib", r"Current kv cache memory in use is ([\d.]+)\s*GiB"),

    _p("available_kv_cache_gib", r"Available KV cache memory:\s*([\d.]+)\s*GiB"),
    _p("kv_cache_size_tokens", r"GPU KV cache size:\s*([\d,]+)\s*tokens"),
    # Lazy, not [^\d]*: the label and value are separated by the request size --
    # "Maximum concurrency for 172,032 tokens per request: 1.00x".
    _p("maximum_concurrency", r"Maximum concurrency.*?([\d.]+)x"),
    # `--max-model-len auto` in action. When this fires, max_model_len was set *to* the
    # KV capacity, which makes a reported concurrency of 1.00x a tautology.
    _p("auto_fit_from auto_fit_to",
       r"Auto-fit max_model_len: reduced from (\d+) to (\d+)"),
    # Hybrid models force a large attention block so the mamba page size matches. This
    # is the floor on granularity for anything that works in blocks.
    _p("attention_block_size", r"Setting attention block size to (\d+)\s*tokens"),

    # Phase timings. Startup cost is the appliance's dominant UX cost with one engine,
    # and vLLM already breaks it down -- no instrumentation needed, just reading.
    _p("weights_gib weight_load_seconds",
       r"Model loading took ([\d.]+)\s*GiB memory and ([\d.]+)\s*seconds"),
    _p("dynamo_seconds", r"Dynamo bytecode transform time:\s*([\d.]+)\s*s"),
    _p("compile_warmup_seconds",
       r"torch\.compile and initial profiling/warmup run together took ([\d.]+)\s*s"),
    # With the cache enabled these are reported separately; disabled, they are merged
    # into the line above. A comparison across states has to handle both shapes.
    _p("compile_seconds", r"torch\.compile took ([\d.]+)\s*s in total"),
    _p("profiling_warmup_seconds", r"Initial profiling/warmup run took ([\d.]+)\s*s"),
    _p("graph_compile_seconds",
       r"Compiling a graph for compile range \([^)]*\) takes ([\d.]+)\s*s"),
    _p("init_engine_seconds compilation_seconds",
       r"init engine .*?took ([\d.]+)\s*s \(compilation: ([\d.]+)\s*s\)"),
    _p("graph_capture_seconds graph_capture_gib",
       r"Graph capturing finished in (\d+)\s*secs, took ([\d.]+)\s*GiB"),
    _p("checkpoint_gib", r"Checkpoint size:\s*([\d.]+)\s*GiB"),
)

#: Which of the three compile-cache states a start was in. vLLM says so explicitly, and
#: it matters because the profiling run that sizes the KV cache shares a process with
#: compilation -- see docs/design.md. A start that *populates* the cache also collects
#: AOT artifacts during the warmup run and profiles a larger transient.
FLAGS: tuple[Pattern, ...] = (
    _p("compile_cache_disabled", r"(vLLM's torch\.compile cache is disabled)"),
    _p("compile_cache_populating", r"Using cache directory: (\S+) for vLLM's torch\.compile"),
    _p("compile_cache_hit", r"Directly load AOT compilation from path (\S+)"),
    _p("compile_artifacts_bytes", r"collected artifacts: .*?([\d]+) bytes total"),
)

#: Failure signatures, most specific first -- the first match wins.
FAILURES: tuple[tuple[Pattern, FailureKind], ...] = (
    (_p("cuda_oom", r"torch\.(?:cuda\.)?OutOfMemoryError|CUDA out of memory", LIKELY),
     FailureKind.OOM),
    (_p("no_kv_cache", r"No available memory for the cache blocks|"
        r"to serve at least one request", GUESSED), FailureKind.NO_KV_CACHE),
    (_p("port_in_use", r"[Aa]ddress already in use|EADDRINUSE", LIKELY),
     FailureKind.PORT_IN_USE),
    (_p("model_not_found",
        r"RepositoryNotFoundError|does not appear to have a file named|"
        r"No such file or directory.*?config\.json", LIKELY), FailureKind.MODEL_NOT_FOUND),
)

#: For calibration only. Deliberately broad, and deliberately not a classifier: vLLM
#: emits [ERROR] for benign things, so this exists to show a human what a table did not
#: claim, never to decide that a start failed.
INTERESTING = re.compile(r"error|traceback|exception|fail|out of memory", re.I)


class LogScanner:
    """Fed one line at a time as the engine runs. Cheap enough to leave on always."""

    def __init__(self) -> None:
        self.facts: dict[str, str] = {}
        self.failure_kind: FailureKind | None = None
        self.failure_line: str | None = None

    @property
    def compile_state(self) -> str:
        """`hit`, `populating`, `disabled`, or `unknown`.

        Reported rather than judged. A populating start is known to profile a larger
        transient, but `disabled` is **not** therefore safe: a disabled start on a box
        where torch's own caches were also cold profiled the same inflated value. What
        the state cannot tell us is how much compiling happened below vLLM, so this
        identifies the one case we can name and leaves the rest to the measurement
        protocol.
        """
        if "compile_cache_hit" in self.facts:
            return "hit"
        if "compile_cache_populating" in self.facts:
            return "populating"
        if "compile_cache_disabled" in self.facts:
            return "disabled"
        return "unknown"

    def feed(self, line: str) -> None:
        for pat in (*FACTS, *FLAGS):
            if pat.keys[0] in self.facts:
                continue  # first occurrence wins; a restart makes a new scanner
            if m := pat.regex.search(line):
                for key, value in zip(pat.keys, m.groups(), strict=False):
                    self.facts[key] = value
        if self.failure_kind is None:
            for pat, kind in FAILURES:
                if pat.regex.search(line):
                    self.failure_kind, self.failure_line = kind, line.strip()
                    break
