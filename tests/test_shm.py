"""Reclaiming crashed engines' shared-memory regions.

Both directions matter and the second one more: failing to remove a stale region gives
the next engine a dead process's memory as its KV cache, and removing a *live* one takes
the cache out from under a running engine. The safety test is the one to keep working.
"""

from __future__ import annotations

import mmap
from pathlib import Path

from golite.engine import shm


def region(root: Path, name: str = "vllm_offload_abc123.mmap", size: int = 4096) -> Path:
    path = root / name
    path.write_bytes(b"\0" * size)
    return path


def test_an_abandoned_region_is_reclaimed(tmp_path):
    path = region(tmp_path)
    reclaimed = shm.sweep(str(tmp_path))
    assert [r.path for r in reclaimed] == [path]
    assert not path.exists()


def test_a_region_a_live_process_holds_open_is_left_alone(tmp_path):
    """The guard that matters. Nothing here is worth deleting a running engine's cache."""
    path = region(tmp_path)
    with path.open("r+b") as fh:  # this process now holds it open
        assert shm.survey(str(tmp_path))[0].in_use
        assert shm.sweep(str(tmp_path)) == []
        assert path.exists()


def test_a_mapped_region_is_left_alone(tmp_path):
    path = region(tmp_path)
    with path.open("r+b") as fh, mmap.mmap(fh.fileno(), 0):
        assert shm.sweep(str(tmp_path)) == []
        assert path.exists()


def test_the_encoder_cache_region_has_the_same_shape_and_is_swept_too(tmp_path):
    path = region(tmp_path, "vllm_ec_abc123.mmap")
    assert [r.path for r in shm.sweep(str(tmp_path))] == [path]


def test_unrelated_shm_files_are_never_touched(tmp_path):
    # /dev/shm is shared: podman locks, loky semaphores, other tenants.
    keep = [region(tmp_path, n) for n in
            ("libpod_rootless_lock_1000", "sem.loky-3658205-8xiin2aw", "vllm_offload.mmap")]
    assert shm.sweep(str(tmp_path)) == []
    assert all(p.exists() for p in keep)


def test_dry_run_reports_without_removing(tmp_path):
    path = region(tmp_path)
    assert [r.path for r in shm.sweep(str(tmp_path), dry_run=True)] == [path]
    assert path.exists()


def test_size_is_reported_because_it_is_the_thing_that_makes_it_dangerous(tmp_path):
    # A leftover only gets adopted when its size matches what the new engine wants,
    # which is exactly what a retry of the same configuration produces.
    region(tmp_path, size=8192)
    assert shm.survey(str(tmp_path))[0].size_bytes == 8192
