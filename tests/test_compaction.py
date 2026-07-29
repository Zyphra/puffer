"""Tiered compaction policy + streaming merge executor."""

import logging

import numpy as np
import pytest

from puffer.index import (
    append_shard,
    compact_band,
    load_manifest,
    read_band_union,
    read_shard_bin,
    set_protected_tags,
    write_shard_bin,
)
from puffer.merge import stream_chunk_keys, streaming_merge_unique


def _binary_level_counts(n: int) -> dict[int, int]:
    """Level j holds a run iff bit j of n is set — the binary-counter shape a
    T=2 tiered policy produces after n disjoint-key inserts, each followed by
    compaction to fixpoint."""
    counts: dict[int, int] = {}
    lvl = 0
    while n:
        if n & 1:
            counts[lvl] = 1
        n >>= 1
        lvl += 1
    return counts


def _runs_by_level(idx, bid=0, exclude=()):
    out: dict[int, int] = {}
    for s in load_manifest(idx, bid).get("shards", []):
        if s.get("dataset") in exclude:
            continue
        lvl = int(s.get("level", 0))
        out[lvl] = out.get(lvl, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Tiered policy
# ---------------------------------------------------------------------------

def test_tiered_fanout_binary_counter_pattern_with_protected(tmp_path):
    """T=2 ingest of 8 disjoint-key runs produces exactly the binary-counter
    level shape; a protected tag ingested alongside stays an unmerged level-0
    run forever, on top of that pattern."""
    idx = tmp_path / "hash_index"
    T = 2
    set_protected_tags(idx, ["keep"])
    append_shard(np.array([-1], dtype=np.int64), idx, 0, "keep")

    all_keys: set[int] = set()
    for k in range(8):
        keys = np.arange(k * 100, k * 100 + 10, dtype=np.int64)
        all_keys.update(int(v) for v in keys)
        append_shard(keys, idx, 0, f"rel_{k:03d}")
        compact_band(idx, 0, tier_fanout=T, ram_budget=1 << 30, protect_tag=None)

    by_level = _runs_by_level(idx, exclude={"keep"})
    assert by_level == _binary_level_counts(8), by_level
    assert all(n < T for n in by_level.values())

    manifest = load_manifest(idx, 0)
    protected_entries = [s for s in manifest["shards"] if s["dataset"] == "keep"]
    assert len(protected_entries) == 1
    assert protected_entries[0]["level"] == 0

    union = {int(v) for v in read_band_union(idx, 0)}
    assert union == all_keys | {-1}


def test_mid_life_tier_fanout_change_raises(tmp_path):
    idx = tmp_path / "hash_index"
    for k in range(3):
        append_shard(np.array([k], dtype=np.int64), idx, 0, f"r{k}")
    compact_band(idx, 0, tier_fanout=3, ram_budget=1 << 30)

    manifest = load_manifest(idx, 0)
    assert manifest["compaction_policy"] == {"name": "tiered", "T": 3}

    with pytest.raises(ValueError, match="stamped tiered T=3"):
        compact_band(idx, 0, tier_fanout=4, ram_budget=1 << 30)


def test_protected_tag_never_merged_even_under_pressure(tmp_path):
    idx = tmp_path / "hash_index"
    set_protected_tags(idx, ["keepme"])
    append_shard(np.array([1, 2], dtype=np.int64), idx, 0, "keepme")
    for k in range(9):
        append_shard(np.array([100 + k], dtype=np.int64), idx, 0, f"rel_{k}")
        compact_band(idx, 0, tier_fanout=2, ram_budget=1 << 30)
    names = {s["dataset"] for s in load_manifest(idx, 0)["shards"]}
    assert "keepme" in names
    keepme = [s for s in load_manifest(idx, 0)["shards"] if s["dataset"] == "keepme"][0]
    assert keepme["level"] == 0


def test_tiered_streaming_and_inram_paths_agree(tmp_path):
    """Same appends compacted under a tiny ram_budget (forces streaming
    merge) vs. a huge one (in-RAM) must produce identical band unions."""
    keys_per = 2000
    unions = []
    for label, budget in (("stream", 1024), ("ram", 1 << 30)):
        idx = tmp_path / f"idx_{label}"
        rng = np.random.default_rng(3)
        for k in range(9):
            keys = rng.integers(-10_000, 10_000, size=keys_per).astype(np.int64)
            append_shard(np.unique(keys), idx, 0, f"rel_{k}")
            compact_band(idx, 0, tier_fanout=3, ram_budget=budget)
        unions.append({int(v) for v in read_band_union(idx, 0)})
    assert unions[0] == unions[1]


# ---------------------------------------------------------------------------
# Streaming merge executor == in-RAM reference
# ---------------------------------------------------------------------------

def test_streaming_merge_matches_inram_reference_under_tiny_budget(tmp_path):
    rng = np.random.default_rng(0)
    arrs = [
        np.unique(rng.integers(-100_000, 100_000, size=5000).astype(np.int64))
        for _ in range(4)
    ]
    paths = []
    for i, a in enumerate(arrs):
        p = tmp_path / f"in_{i}.bin"
        write_shard_bin(a, p)
        paths.append(p)

    out = tmp_path / "out.bin"
    chunk = stream_chunk_keys(4096, len(paths))  # force tiny chunks
    n, mn, mx = streaming_merge_unique(paths, out, chunk_keys=chunk)

    ref = np.unique(np.concatenate(arrs))
    got = read_shard_bin(out, mmap=False)
    assert np.array_equal(np.asarray(got), ref)
    assert n == ref.size
    assert mn == int(ref[0]) and mx == int(ref[-1])


def test_streaming_merge_empty_and_disjoint_inputs(tmp_path):
    empty = tmp_path / "empty.bin"
    write_shard_bin(np.array([], dtype=np.int64), empty)
    lo = tmp_path / "lo.bin"
    write_shard_bin(np.arange(0, 50, dtype=np.int64), lo)
    hi = tmp_path / "hi.bin"
    write_shard_bin(np.arange(1000, 1050, dtype=np.int64), hi)

    out = tmp_path / "out.bin"
    n, mn, mx = streaming_merge_unique([empty, lo, hi], out, chunk_keys=7)
    ref = np.concatenate([np.arange(0, 50), np.arange(1000, 1050)]).astype(np.int64)
    got = np.asarray(read_shard_bin(out, mmap=False))
    assert np.array_equal(got, ref)
    assert (n, mn, mx) == (ref.size, int(ref[0]), int(ref[-1]))


def test_stream_chunk_keys_budget_model_and_floor_warning(caplog):
    assert stream_chunk_keys(4 * 1024**3, 4) == (4 * 1024**3) // (3 * 8 * 4)
    assert stream_chunk_keys(1 << 30, 8) == (1 << 30) // (3 * 8 * 8)

    with caplog.at_level(logging.WARNING, logger="puffer.merge"):
        got = stream_chunk_keys(4 * 1024**2, 8)
    assert got == 64 * 1024
    assert any("advisory" in r.message for r in caplog.records)
