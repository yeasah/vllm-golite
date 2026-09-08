"""What a measurement was taken on.

A stored fit result is a measurement of one machine in one state, and this is the part of
that state we can name. It exists because of a specific finding: the same configuration
on the same card resolved `--max-model-len auto` to 172,032 tokens on some starts and
196,608 on others, and the difference was how much compiling happened in the process.
So a run's provenance is not optional decoration -- it is what lets a stored number be
believed later, or noticed as stale.

**This is not `host_survey.py`'s list, and the difference is the point.** That tool
classifies which host fields can change the *tokens a model emits* -- GPU architecture,
count, driver, VRAM, uncorrected ECC -- because it screens boxes for output-based
benchmarks. Fit asks a different question, so it keeps a different list: VRAM and GPU
count matter to both, driver to both, but the software stack matters far more here and
host CPU or PCIe width matter not at all. What transfers from that tool is its actual
innovation, which is keeping the list explicit and checkable rather than remembered.

Deliberately partial, and honest about it: `compile_state` belongs to the run rather
than the box, and nothing here can see whether torch's own caches were warm.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version

#: Packages whose version can change a measurement. The forks first, since those are the
#: ones that move without a release.
TRACKED_PACKAGES = ("vllm", "vllm-untwisted", "vllm-exl3-plugin", "vllm-gguf-plugin",
                    "vllm-virtualkv-plugin", "exllamav3", "torch")


def _gpus() -> dict[str, str]:
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return {}
    rows = [r.strip() for r in out.splitlines() if r.strip()]
    if not rows:
        return {}
    names, memories, drivers = zip(*(r.split(", ") for r in rows), strict=False)
    return {
        "gpu_count": str(len(rows)),
        # Sorted and joined so two boxes with the same cards in a different order agree.
        "gpu_models": ",".join(sorted(names)),
        "gpu_memory_total": ",".join(sorted(memories)),
        "driver_version": drivers[0],
    }


def _packages() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def collect() -> dict[str, str]:
    """The box and the software on it, as flat strings."""
    fp: dict[str, str] = {"python": platform.python_version()}
    fp |= _gpus()
    fp |= {f"pkg.{k}": v for k, v in _packages().items()}
    return fp


def digest(fingerprint: dict[str, str]) -> str:
    """A stable short hash, for grouping runs and for asking "same box?" cheaply."""
    payload = json.dumps(fingerprint, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def differences(a: dict[str, str], b: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Every field the two disagree on, missing counted as a disagreement.

    What makes a stored result stale is a question this cannot answer on its own -- it
    reports, and the fit layer decides which fields it cares about.
    """
    return {k: (a.get(k, "-"), b.get(k, "-"))
            for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
