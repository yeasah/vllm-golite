"""Read vLLM's startup log for two things: why it failed, and what it measured.

The failure half is not optional. vLLM reports its interesting failures rather than
raising them, so a supervisor watching only the exit code learns almost nothing about
an OOM at profile time versus a checkpoint that was never there.

The measurement half is the more valuable one and is easy to miss. An engine start
costs a whole process launch, and while it starts, vLLM prints numbers that no amount
of `config.json` arithmetic can produce -- free versus total memory, the KV cache it
actually got, what it thinks concurrency will be. Harvesting those turns every launch
into a tier-2 measurement instead of only a launch.

**Patterns here are provisional.** They are written from the strings this project has
recorded seeing, not from a survey of vLLM's logging, and vLLM's log text is not an API.
Each carries its provenance, and `python -m golite.engine.logscan <logfile>` prints what
matched and what did not so a real capture can settle them. Treat an unverified pattern
that never fires as unverified, not as an absent condition.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from golite.engine.state import FailureKind


@dataclass(frozen=True, slots=True)
class Pattern:
    key: str
    regex: re.Pattern[str]
    #: How much we actually know about this string.
    #: "attested"  -- quoted in this project's own field notes, from a real run.
    #: "likely"    -- standard library/framework text (a Python exception, an OSError).
    #: "guessed"   -- plausible shape, never checked. Calibrate before relying on it.
    provenance: str


#: Numbers vLLM prints while starting. Keys are ours and stable; the regexes are not.
FACTS: tuple[Pattern, ...] = (
    # Attested in the field notes: "the startup line prints both, so the maximum usable
    # value is literally their ratio -- 15.28 / 15.51 = 0.985". Format unconfirmed.
    Pattern("memory_free_total",
            re.compile(r"([\d.]+)\s*GiB\s*(?:is\s*)?free.*?([\d.]+)\s*GiB.*?total", re.I),
            "guessed"),
    # Attested verbatim in the field notes, including the trailing "x".
    Pattern("maximum_concurrency",
            # Lazy, not [^\d]*: the real line puts the request size between the label
            # and the value -- "Maximum concurrency for 262,144 tokens per request: 1.03x".
            re.compile(r"Maximum concurrency.*?([\d.]+)x", re.I),
            "attested"),
    # Attested: vLLM prints a `--kv-cache-memory=` line. Do not apply it blindly --
    # it subtracts graph memory and a 150 MiB redundancy buffer the profiler did not
    # count, so it lands below what the running engine is already surviving on.
    Pattern("kv_cache_memory_suggestion",
            re.compile(r"--kv-cache-memory=(\d+)"),
            "attested"),
    Pattern("kv_cache_size_tokens",
            re.compile(r"KV cache size[^\d]*([\d,]+)\s*tokens", re.I),
            "guessed"),
    # Attested that the capture cost is "reported directly at startup"; wording unknown.
    Pattern("graph_capture_memory",
            re.compile(r"[Gg]raph capturing finished.*?([\d.]+)\s*GiB", re.I),
            "guessed"),
)

#: Failure signatures, most specific first -- the first match wins.
FAILURES: tuple[tuple[Pattern, FailureKind], ...] = (
    (Pattern("cuda_oom", re.compile(r"torch\.(cuda\.)?OutOfMemoryError|CUDA out of memory", re.I),
             "likely"), FailureKind.OOM),
    (Pattern("no_kv_cache", re.compile(r"No available memory for the cache blocks|"
                                       r"initial engine.*?memory.*?not enough", re.I),
             "guessed"), FailureKind.NO_KV_CACHE),
    (Pattern("port_in_use", re.compile(r"[Aa]ddress already in use|EADDRINUSE"),
             "likely"), FailureKind.PORT_IN_USE),
    (Pattern("model_not_found",
             re.compile(r"RepositoryNotFoundError|does not appear to have a file named|"
                        r"No such file or directory.*?config\.json", re.I),
             "likely"), FailureKind.MODEL_NOT_FOUND),
)

#: Lines worth showing a human even when nothing classified them.
INTERESTING = re.compile(r"error|traceback|exception|fail|out of memory", re.I)


class LogScanner:
    """Fed one line at a time as the engine runs. Cheap enough to leave on always."""

    def __init__(self) -> None:
        self.facts: dict[str, str] = {}
        self.failure_kind: FailureKind | None = None
        self.failure_line: str | None = None
        self.matched_keys: set[str] = set()

    def feed(self, line: str) -> None:
        for pat in FACTS:
            if pat.key in self.facts:
                continue  # first occurrence wins; a restart makes a new scanner
            if m := pat.regex.search(line):
                self.facts[pat.key] = m.group(1) if m.lastindex == 1 else "/".join(m.groups())
                self.matched_keys.add(pat.key)
        if self.failure_kind is None:
            for pat, kind in FAILURES:
                if pat.regex.search(line):
                    self.failure_kind, self.failure_line = kind, line.strip()
                    self.matched_keys.add(pat.key)
                    break


def _calibrate(path: str) -> int:
    """Run the pattern table over a captured log and report what it can and cannot see.

    This is the loop that turns the guesses above into attested patterns, and the reason
    the table is data rather than inline regexes.
    """
    scanner = LogScanner()
    unclassified: list[str] = []
    with open(path, errors="replace") as fh:
        for line in fh:
            scanner.feed(line)
            if INTERESTING.search(line) and scanner.failure_line != line.strip():
                unclassified.append(line.rstrip())

    print(f"== facts ({len(scanner.facts)}/{len(FACTS)} patterns fired)")
    for pat in FACTS:
        got = scanner.facts.get(pat.key)
        mark = "  " if got else "!!"
        print(f"{mark} {pat.key:<28} [{pat.provenance:<8}] {got if got else '-- no match'}")
    print(f"\n== failure: {scanner.failure_kind or 'none classified'}")
    if scanner.failure_line:
        print(f"   {scanner.failure_line}")
    if unclassified:
        print(f"\n== {len(unclassified)} interesting lines nothing claimed")
        for line in unclassified[:40]:
            print(f"   {line}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m golite.engine.logscan <vllm-log-file>")
    sys.exit(_calibrate(sys.argv[1]))
