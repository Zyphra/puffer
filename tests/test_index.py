"""LSM shard store: append/iter/exclude_tag roundtrip, idempotent re-append."""

import numpy as np

from puffer.index import append_shard, iter_shards, load_manifest, read_band_union


def test_append_iter_and_exclude_tag_roundtrip(tmp_path):
    idx = tmp_path / "hash_index"
    append_shard(np.array([1, 2, 3], dtype=np.int64), idx, 0, "ds_a")
    append_shard(np.array([3, 4, 5], dtype=np.int64), idx, 0, "ds_b")

    all_shards = iter_shards(idx, 0)
    assert {s["dataset"] for s, _ in all_shards} == {"ds_a", "ds_b"}

    only_b = iter_shards(idx, 0, exclude_tag="ds_a")
    assert {s["dataset"] for s, _ in only_b} == {"ds_b"}

    union = read_band_union(idx, 0)
    assert union is not None
    assert {int(v) for v in union} == {1, 2, 3, 4, 5}

    excl = read_band_union(idx, 0, exclude_tag="ds_a")
    assert {int(v) for v in excl} == {3, 4, 5}


def test_append_shard_idempotent_reappend_overwrites_by_tag(tmp_path):
    idx = tmp_path / "hash_index"
    append_shard(np.array([1, 2, 3], dtype=np.int64), idx, 0, "ds_a")
    append_shard(np.array([10, 20], dtype=np.int64), idx, 0, "ds_b")

    # Re-append ds_a with different (still sorted-unique) content, simulating
    # a resumed/retried ingest of the same dataset.
    append_shard(np.array([7, 8, 9], dtype=np.int64), idx, 0, "ds_a")

    manifest = load_manifest(idx, 0)
    a_entries = [s for s in manifest["shards"] if s["dataset"] == "ds_a"]
    assert len(a_entries) == 1, "re-append must overwrite, not duplicate"
    assert a_entries[0]["count"] == 3
    assert a_entries[0]["min"] == 7 and a_entries[0]["max"] == 9

    union = {int(v) for v in read_band_union(idx, 0)}
    assert union == {7, 8, 9, 10, 20}


def test_append_shard_empty_array_is_noop(tmp_path):
    idx = tmp_path / "hash_index"
    append_shard(np.array([], dtype=np.int64), idx, 0, "ds_empty")
    manifest = load_manifest(idx, 0)
    assert manifest["shards"] == []


def test_sanitize_tag_used_consistently_for_lookup(tmp_path):
    idx = tmp_path / "hash_index"
    append_shard(np.array([1], dtype=np.int64), idx, 0, "weird tag/name!!")
    shards = iter_shards(idx, 0)
    assert len(shards) == 1
    # exclude_tag with the same unsanitized string must still match.
    excluded = iter_shards(idx, 0, exclude_tag="weird tag/name!!")
    assert excluded == []
