from golite.engine.logscan import FACTS, LogScanner
from golite.engine.state import FailureKind


def scan(*lines: str) -> LogScanner:
    s = LogScanner()
    for line in lines:
        s.feed(line)
    return s


def test_harvests_the_numbers_only_a_start_can_produce():
    s = scan(
        "INFO gpu memory: 15.28 GiB is free of 15.51 GiB total",
        "INFO GPU KV cache size: 67,584 tokens",
        "INFO Maximum concurrency for 262,144 tokens per request: 1.03x",
        "INFO To fully utilize gpu memory pass --kv-cache-memory=1323302912",
    )
    assert s.facts["maximum_concurrency"] == "1.03"
    assert s.facts["kv_cache_memory_suggestion"] == "1323302912"
    assert s.facts["kv_cache_size_tokens"] == "67,584"
    assert s.facts["memory_free_total"] == "15.28/15.51"


def test_classifies_oom_rather_than_reporting_an_exit_code():
    s = scan("ERROR torch.cuda.OutOfMemoryError: CUDA out of memory.")
    assert s.failure_kind is FailureKind.OOM


def test_classifies_a_busy_port():
    assert scan("OSError: [Errno 98] Address already in use").failure_kind is FailureKind.PORT_IN_USE


def test_first_failure_wins():
    # A crash cascades; the first classified line is the one that explains it.
    s = scan("ERROR CUDA out of memory", "OSError: Address already in use")
    assert s.failure_kind is FailureKind.OOM


def test_clean_log_classifies_nothing():
    assert scan("INFO vllm: starting engine", "INFO serving on 8000").failure_kind is None


def test_every_pattern_declares_its_provenance():
    # The table is data so a real capture can settle it; an undeclared guess is the
    # thing that would quietly get trusted.
    assert all(p.provenance in {"attested", "likely", "guessed"} for p in FACTS)
