"""Patterns are checked twice: against single lines, and against a real capture.

The capture is the one that matters. vLLM's log text is not an API, so a bump can
silently stop a pattern matching, and a fact that quietly goes missing is worse than one
that was never harvested.
"""

from pathlib import Path

from golite.engine.logscan import FACTS, FLAGS, LogScanner
from golite.engine.state import FailureKind

CAPTURE = Path(__file__).parent / "data" / "vllm-start-qwen3.8-27b.log"

# The startup summary: most of the memory story arrives on one line.
SUMMARY = (
    "(EngineCore pid=1) INFO [gpu_worker.py:804] Free memory on device (15.28/15.51 GiB) "
    "on startup. Desired GPU memory utilization is (0.92, 14.27 GiB). Actual usage is "
    "10.47 GiB for consumed memory (weights + non-torch), 0.79 GiB for peak activation, "
    "and 0.04 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with "
    "`--kv-cache-memory=3031561913` (2.82 GiB) to fit into requested memory, or "
    "`--kv-cache-memory=4121221120` (3.84 GiB) to fully utilize gpu memory. Current kv "
    "cache memory in use is 3.01 GiB."
)


def scan(*lines: str) -> LogScanner:
    s = LogScanner()
    for line in lines:
        s.feed(line)
    return s


def scan_capture() -> LogScanner:
    s = LogScanner()
    with CAPTURE.open() as fh:
        for line in fh:
            s.feed(line)
    return s


def test_the_utilization_ceiling_is_a_ratio_of_two_logged_numbers():
    # A fraction of *total* that must fit within *free*: 15.28/15.51 = 0.985.
    f = scan(SUMMARY).facts
    assert (f["memory_free_gib"], f["memory_total_gib"]) == ("15.28", "15.51")


def test_both_kv_cache_suggestions_are_kept_separately():
    # There are two, they differ, and they bracket what is actually running.
    f = scan(SUMMARY).facts
    assert f["kv_cache_memory_requested"] == "3031561913"
    assert f["kv_cache_memory_full"] == "4121221120"
    assert float(f["kv_cache_requested_gib"]) < float(f["kv_cache_in_use_gib"])
    assert float(f["kv_cache_full_gib"]) > float(f["kv_cache_in_use_gib"])


def test_memory_breakdown_is_harvested():
    f = scan(SUMMARY).facts
    assert f["consumed_memory_gib"] == "10.47"
    assert f["peak_activation_gib"] == "0.79"
    assert f["cudagraph_memory_gib"] == "0.04"


def test_concurrency_and_cache_size_share_one_line():
    f = scan("INFO [kv_cache_utils.py:1883] GPU KV cache size: 172,032 tokens, "
             "Maximum concurrency for 172,032 tokens per request: 1.00x").facts
    assert f["kv_cache_size_tokens"] == "172,032"
    assert f["maximum_concurrency"] == "1.00"


def test_auto_fit_is_visible_so_the_tautology_can_be_detected():
    # --max-model-len auto sets max_model_len *to* the KV capacity, which is what makes
    # a reported 1.00x concurrency a tautology rather than an observation.
    f = scan("INFO Auto-fit max_model_len: reduced from 262144 to 172032 to fit in "
             "available GPU memory (3.01 GiB available for KV cache)").facts
    assert (f["auto_fit_from"], f["auto_fit_to"]) == ("262144", "172032")


def test_every_pattern_fires_against_the_real_capture():
    facts = scan_capture().facts
    unmatched = [p.keys for p in (*FACTS, *FLAGS) if not any(k in facts for k in p.keys)]
    assert not unmatched, f"patterns stopped matching a real log: {unmatched}"


def test_the_capture_is_a_clean_start():
    # vLLM logs [ERROR] for undocumented processor kwargs and absent ROCm modules, so a
    # classifier that searched for the word would call this successful start a failure.
    assert scan_capture().failure_kind is None


def test_classifies_oom_rather_than_reporting_an_exit_code():
    assert scan("ERROR torch.cuda.OutOfMemoryError: CUDA out of memory.").failure_kind is FailureKind.OOM


def test_classifies_a_busy_port():
    assert scan("OSError: [Errno 98] Address already in use").failure_kind is FailureKind.PORT_IN_USE


def test_first_failure_wins():
    s = scan("ERROR CUDA out of memory", "OSError: Address already in use")
    assert s.failure_kind is FailureKind.OOM


def test_every_pattern_declares_its_provenance():
    assert all(p.provenance in {"attested", "likely", "guessed"} for p in (*FACTS, *FLAGS))


def test_a_warm_compile_cache_changes_what_the_profiler_measures():
    """Not a latency detail: it moves the number vLLM sizes the KV cache from.

    Same configuration, same card, cache cold in one capture and hit in the other.
    Peak activation halves and the resolved context grows by 14%, which means a fit
    measured cold understates capacity.
    """
    cold = scan_capture().facts
    warm = LogScanner()
    with (CAPTURE.parent / "vllm-start-warm-cache.log").open() as fh:
        for line in fh:
            warm.feed(line)
    warm = warm.facts

    assert float(warm["peak_activation_gib"]) < float(cold["peak_activation_gib"])
    assert float(warm["available_kv_cache_gib"]) > float(cold["available_kv_cache_gib"])
    assert int(warm["auto_fit_to"]) > int(cold["auto_fit_to"])
    # Phase-timing lines are simply not printed on a cache hit, so a pattern that does
    # not fire here is state, not a regression.
    assert float(warm["compilation_seconds"]) < 1.0
