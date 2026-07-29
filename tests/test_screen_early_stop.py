"""Early-stop cross-history screening: equivalence to a full-membership
reference plus probe-count savings/no-savings under controlled data."""

import numpy as np
import pytest

from puffer.config import PufferConfig
from puffer.index import append_shard, iter_shards, read_shard_bin
from puffer.screen import screen_release


def _reference_cross_mask(band_keys: np.ndarray, index_dir, exclude_tag=None) -> np.ndarray:
    """Independent full cross-membership reference (no early-stop): a row hits
    if any of its per-band keys is present in any shard of that band. Kept as
    the correctness oracle the optimized ``screen_release`` is validated against."""
    n_docs, num_bands = band_keys.shape
    hit = np.zeros(n_docs, dtype=bool)
    for band_id in range(num_bands):
        keys = band_keys[:, band_id]
        for _meta, path in iter_shards(index_dir, band_id, exclude_tag):
            arr = read_shard_bin(path, mmap=False)
            n = len(arr)
            if n == 0:
                continue
            idx = np.searchsorted(arr, keys)
            valid = idx < n
            idx_c = np.minimum(idx, n - 1)
            hit |= valid & (np.asarray(arr[idx_c]) == keys)
    return hit


def _build_index(tmp_path, num_bands, rng, n_history_shards=5, keys_per_shard=400):
    idx = tmp_path / "hash_index"
    all_keys_per_band = [set() for _ in range(num_bands)]
    for i in range(n_history_shards):
        for band_id in range(num_bands):
            keys = rng.integers(-50_000, 50_000, size=keys_per_shard).astype(np.int64)
            keys = np.unique(keys)
            append_shard(keys, idx, band_id, f"hist_{i}")
            all_keys_per_band[band_id].update(int(v) for v in keys)
    return idx, all_keys_per_band


@pytest.mark.parametrize("probe_order", ["largest_first", "smallest_first", "newest_first"])
def test_early_stop_mask_matches_reference(tmp_path, probe_order):
    rng = np.random.default_rng(42)
    num_bands = 4
    idx, all_keys_per_band = _build_index(tmp_path, num_bands, rng)

    n_docs = 2000
    band_keys = np.empty((n_docs, num_bands), dtype=np.int64)
    for band_id in range(num_bands):
        history = np.array(sorted(all_keys_per_band[band_id]), dtype=np.int64)
        # Half overlapping with history (guarantees hits to exercise early stop).
        overlap = rng.choice(history, size=n_docs // 2, replace=True) if history.size else np.array([], dtype=np.int64)
        fresh = rng.integers(-50_000, 50_000, size=n_docs - overlap.size).astype(np.int64)
        col = np.concatenate([overlap, fresh])
        rng.shuffle(col)
        band_keys[:, band_id] = col

    cfg = PufferConfig(num_bands=num_bands, num_perm=num_bands, probe_order=probe_order, n_workers=4)
    counters: dict = {}
    got = screen_release(band_keys, idx, exclude_tag=None, cfg=cfg, counters=counters)
    expected = _reference_cross_mask(band_keys, idx)

    assert np.array_equal(got, expected)
    assert counters["probes_scheduled"] == n_docs * num_bands * len(
        [1 for _ in range(5)]
    )
    assert counters["probes_done"] < counters["probes_scheduled"]
    assert expected.any(), "test data must contain real duplicates"


def test_early_stop_no_matches_probes_done_equals_scheduled(tmp_path):
    rng = np.random.default_rng(7)
    num_bands = 3
    idx, all_keys_per_band = _build_index(tmp_path, num_bands, rng, n_history_shards=3, keys_per_shard=100)

    # Disjoint key space: no document key can ever match any history shard.
    n_docs = 500
    band_keys = np.full((n_docs, num_bands), -1, dtype=np.int64)
    for band_id in range(num_bands):
        history = all_keys_per_band[band_id]
        vals = []
        candidate = 10_000_000
        while len(vals) < n_docs:
            if candidate not in history:
                vals.append(candidate)
            candidate += 1
        band_keys[:, band_id] = np.array(vals, dtype=np.int64)

    cfg = PufferConfig(num_bands=num_bands, num_perm=num_bands, probe_order="largest_first", n_workers=2)
    counters: dict = {}
    got = screen_release(band_keys, idx, exclude_tag=None, cfg=cfg, counters=counters)

    assert not got.any()
    assert counters["probes_done"] == counters["probes_scheduled"]


def test_exclude_tag_excludes_own_shard(tmp_path):
    idx = tmp_path / "hash_index"
    append_shard(np.array([1, 2, 3], dtype=np.int64), idx, 0, "self")
    band_keys = np.array([[1], [2], [99]], dtype=np.int64)
    cfg = PufferConfig(num_bands=1, num_perm=1, n_workers=1)

    with_self = screen_release(band_keys, idx, exclude_tag=None, cfg=cfg)
    assert with_self.tolist() == [True, True, False]

    without_self = screen_release(band_keys, idx, exclude_tag="self", cfg=cfg)
    assert without_self.tolist() == [False, False, False]


def test_invalid_probe_order_raises(tmp_path):
    """PufferConfig itself rejects unknown probe_order, but screen_release's
    own guard must also reject any object exposing a bad value (defense in
    depth for callers that don't go through PufferConfig validation)."""
    from types import SimpleNamespace

    idx = tmp_path / "hash_index"
    append_shard(np.array([1], dtype=np.int64), idx, 0, "a")
    append_shard(np.array([2], dtype=np.int64), idx, 0, "b")
    band_keys = np.array([[1]], dtype=np.int64)
    cfg = SimpleNamespace(probe_order="bogus", effective_workers=1)
    with pytest.raises(ValueError):
        screen_release(band_keys, idx, exclude_tag=None, cfg=cfg)

    with pytest.raises(ValueError):
        PufferConfig(num_bands=1, num_perm=1, probe_order="bogus", n_workers=1)
