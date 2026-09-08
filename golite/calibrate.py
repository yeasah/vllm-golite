"""Re-check the log patterns against a capture.

vLLM's log text is not an API, and every fact golite harvests rides on it. This is the
loop that turns a guess into an attested pattern and, more importantly, notices when an
attested one silently stops matching after a bump. A pattern that never fires means
unverified, not absent.

    python -m golite.calibrate tests/data/vllm-start-qwen3.8-27b.log
"""

from __future__ import annotations

import sys

from golite.engine.logscan import FACTS, FLAGS, INTERESTING, LogScanner


def calibrate(path: str) -> int:
    scanner = LogScanner()
    unclaimed: list[str] = []
    with open(path, errors="replace") as fh:
        for line in fh:
            scanner.feed(line)
            if INTERESTING.search(line) and scanner.failure_line != line.strip():
                unclaimed.append(line.rstrip())

    missing = 0
    print(f"== {len(scanner.facts)} facts from {len(FACTS) + len(FLAGS)} patterns")
    for pat in (*FACTS, *FLAGS):
        got = [scanner.facts.get(k) for k in pat.keys]
        if not any(got):
            missing += 1
        mark = "  " if any(got) else "!!"
        shown = ", ".join(f"{k}={v}" for k, v in zip(pat.keys, got, strict=False) if v)
        print(f"{mark} [{pat.provenance:<8}] {shown or ' '.join(pat.keys) + ' -- no match'}")

    print(f"\n== failure: {scanner.failure_kind or 'none classified'}")
    if scanner.failure_line:
        print(f"   {scanner.failure_line}")
    if unclaimed:
        print(f"\n== {len(unclaimed)} lines matched /{INTERESTING.pattern}/ and were not"
              " classified (vLLM logs [ERROR] for benign things; this is not a failure list)")
        for line in unclaimed[:30]:
            print(f"   {line[:160]}")
    return 1 if missing else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m golite.calibrate <vllm-log-file>")
    sys.exit(calibrate(sys.argv[1]))
