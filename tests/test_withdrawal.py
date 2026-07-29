"""Dataset withdrawal: O(1) standalone drop, survivor-rebuild via sidecars,
plan/apply crash-safety, and the state-directory ``withdraw_dataset`` helper."""

import numpy as np
import pytest

from puffer.config import PufferConfig
from puffer.index import (
    append_shard,
    compact_band,
    load_manifest,
    read_band_union,
    write_shard_bin,
)
from puffer.withdraw import apply_band_withdrawal, plan_band_withdrawal, withdraw_dataset

CONTRIBS = {
    "ds_a": [1, 2, 3, 10],
    "ds_b": [3, 4, 5, 11],
    "ds_c": [5, 6, 7, 12],
}


def _write_sidecars(state_dir, num_bands, contribs):
    """Write both the index shards AND the dataset sidecar (state/datasets/
    <tag>/band_XX.bin) — the sidecar is the exact array a dataset contributed
    for that band, kept for withdrawal rebuilds."""
    index_dir = state_dir / "hash_index"
    for tag, keys in contribs.items():
        arr = np.array(sorted(set(keys)), dtype=np.int64)
        for bid in range(num_bands):
            append_shard(arr, index_dir, bid, tag)
            sidecar = state_dir / "datasets" / tag / f"band_{bid:02d}.bin"
            write_shard_bin(arr, sidecar)
        (state_dir / "datasets" / tag).mkdir(parents=True, exist_ok=True)
        (state_dir / "datasets" / tag / "meta.json").write_text("{}")


def _band_union(index_dir, bid):
    arr = read_band_union(index_dir, bid)
    return set() if arr is None else {int(v) for v in arr}


# ---------------------------------------------------------------------------
# O(1) standalone drop
# ---------------------------------------------------------------------------

def test_standalone_drop_is_o1_and_untouches_survivors(tmp_path):
    idx = tmp_path / "hash_index"
    append_shard(np.array([1, 2, 3], dtype=np.int64), idx, 0, "ds_a")
    append_shard(np.array([3, 4, 5], dtype=np.int64), idx, 0, "ds_b")

    survivor_file = idx / "band_00" / "ds_b.bin"
    before = survivor_file.read_bytes()

    plan = plan_band_withdrawal(idx, 0, "ds_a")
    assert plan["tag_files"] == ["ds_a.bin"]
    assert plan["affected"] == []

    apply_band_withdrawal(idx, 0, "ds_a", plan)

    assert not (idx / "band_00" / "ds_a.bin").exists()
    assert survivor_file.read_bytes() == before
    assert _band_union(idx, 0) == {3, 4, 5}


def test_plan_is_noop_for_never_ingested_dataset(tmp_path):
    idx = tmp_path / "hash_index"
    append_shard(np.array([1, 2], dtype=np.int64), idx, 0, "ds_a")
    plan = plan_band_withdrawal(idx, 0, "never_seen")
    assert plan == {"tag_files": [], "affected": [], "contributors": []}


# ---------------------------------------------------------------------------
# Plan/apply crash-safety: apply twice is a no-op
# ---------------------------------------------------------------------------

def test_apply_twice_is_a_noop(tmp_path):
    idx = tmp_path / "hash_index"
    append_shard(np.array([1, 2], dtype=np.int64), idx, 0, "ds_a")
    append_shard(np.array([2, 3], dtype=np.int64), idx, 0, "ds_b")

    plan = plan_band_withdrawal(idx, 0, "ds_a")
    apply_band_withdrawal(idx, 0, "ds_a", plan)
    manifest_after_first = load_manifest(idx, 0)

    # Re-applying the SAME (now-stale) plan must not resurrect/duplicate
    # anything — the tag's file is already gone.
    apply_band_withdrawal(idx, 0, "ds_a", plan)
    manifest_after_second = load_manifest(idx, 0)
    assert manifest_after_first == manifest_after_second
    assert _band_union(idx, 0) == {2, 3}

    # A fresh plan against the already-withdrawn state is also a no-op.
    plan2 = plan_band_withdrawal(idx, 0, "ds_a")
    assert plan2 == {"tag_files": [], "affected": [], "contributors": []}
    apply_band_withdrawal(idx, 0, "ds_a", plan2)
    assert load_manifest(idx, 0) == manifest_after_second


# ---------------------------------------------------------------------------
# Compacted withdrawal: rebuild from surviving sidecars
# ---------------------------------------------------------------------------

def test_compacted_withdrawal_rebuilds_from_survivor_sidecars(tmp_path):
    idx = tmp_path / "hash_index"
    for tag, keys in CONTRIBS.items():
        append_shard(np.array(sorted(set(keys)), dtype=np.int64), idx, 0, tag)
    compact_band(idx, 0, tier_fanout=3, ram_budget=1 << 30)  # force full merge

    manifest = load_manifest(idx, 0)
    merged = [s for s in manifest["shards"] if s["dataset"].startswith("cmp_")]
    assert merged, "expected a merged run"

    plan = plan_band_withdrawal(idx, 0, "ds_b")
    assert plan["tag_files"] == []  # compacted away
    assert plan["affected"]
    contributors = plan["contributors"]
    assert contributors is not None and "ds_b" not in contributors

    surv = np.array(sorted({h for t in contributors for h in CONTRIBS[t]}), dtype=np.int64)
    rebuilt = tmp_path / "_rebuilt_ds_b.bin"
    surv.tofile(rebuilt)          # a prewritten sorted-unique bounded-merge output
    apply_band_withdrawal(
        idx, 0, "ds_b", plan,
        rebuilt_path=rebuilt, rebuilt_count=int(surv.size),
        rebuilt_min=int(surv[0]), rebuilt_max=int(surv[-1]),
        rebuilt_sources=contributors,
    )

    expected = set()
    for t, hs in CONTRIBS.items():
        if t != "ds_b":
            expected.update(hs)
    assert _band_union(idx, 0) == expected


# ---------------------------------------------------------------------------
# withdraw_dataset: full state-dir helper
# ---------------------------------------------------------------------------

def test_withdraw_dataset_middle_release_rebuilds_from_sidecars(tmp_path):
    num_bands = 2
    state_dir = tmp_path / "state"
    _write_sidecars(state_dir, num_bands, CONTRIBS)
    index_dir = state_dir / "hash_index"
    for bid in range(num_bands):
        compact_band(index_dir, bid, tier_fanout=3, ram_budget=1 << 30)

    cfg = PufferConfig(num_bands=num_bands, num_perm=num_bands, ram_budget_bytes=1 << 30)
    report = withdraw_dataset(state_dir, "ds_b", cfg)

    assert report["bands_rebuilt"] == num_bands
    assert report["bands_o1"] == 0
    assert report["elapsed_s"] >= 0

    survivors = {"ds_a": CONTRIBS["ds_a"], "ds_c": CONTRIBS["ds_c"]}
    for bid in range(num_bands):
        expected = set()
        for hs in survivors.values():
            expected.update(hs)
        assert _band_union(index_dir, bid) == expected

    assert not (state_dir / "datasets" / "ds_b").exists()
    for tag in survivors:
        assert (state_dir / "datasets" / tag / f"band_00.bin").exists()


def test_withdraw_dataset_standalone_takes_o1_path(tmp_path):
    num_bands = 1
    state_dir = tmp_path / "state"
    _write_sidecars(state_dir, num_bands, CONTRIBS)
    index_dir = state_dir / "hash_index"
    # No compaction — every dataset's shard is still a standalone L0 run.

    survivor_bytes = (index_dir / "band_00" / "ds_a.bin").read_bytes()
    cfg = PufferConfig(num_bands=num_bands, num_perm=num_bands, ram_budget_bytes=1 << 30)
    report = withdraw_dataset(state_dir, "ds_b", cfg)

    assert report["bands_o1"] == num_bands
    assert report["bands_rebuilt"] == 0
    assert (index_dir / "band_00" / "ds_a.bin").read_bytes() == survivor_bytes
    assert _band_union(index_dir, 0) == set(CONTRIBS["ds_a"]) | set(CONTRIBS["ds_c"])


def test_withdraw_dataset_removes_removals_parquet(tmp_path):
    num_bands = 1
    state_dir = tmp_path / "state"
    _write_sidecars(state_dir, num_bands, CONTRIBS)
    removals_dir = state_dir / "removals"
    removals_dir.mkdir(parents=True)
    victim_removals = removals_dir / "ds_b.parquet"
    victim_removals.write_bytes(b"not-really-parquet-just-a-marker")

    cfg = PufferConfig(num_bands=num_bands, num_perm=num_bands)
    withdraw_dataset(state_dir, "ds_b", cfg)

    assert not victim_removals.exists()


def test_withdraw_dataset_rebuild_does_not_materialize_union(tmp_path, monkeypatch):
    """The survivor-rebuild path must union sidecars via the bounded merge and
    publish a file -- never read shard payloads into RAM. Guard: make
    ``read_shard_bin`` explode during the rebuild/apply and require success."""
    import puffer.index as pindex

    num_bands = 2
    state_dir = tmp_path / "state"
    _write_sidecars(state_dir, num_bands, CONTRIBS)
    index_dir = state_dir / "hash_index"
    for bid in range(num_bands):
        compact_band(index_dir, bid, tier_fanout=3, ram_budget=1 << 30)  # setup uses it

    def _boom(*a, **k):
        raise AssertionError("rebuilt union materialized via read_shard_bin")

    monkeypatch.setattr(pindex, "read_shard_bin", _boom)
    cfg = PufferConfig(num_bands=num_bands, num_perm=num_bands, ram_budget_bytes=1 << 30)
    report = withdraw_dataset(state_dir, "ds_b", cfg)
    monkeypatch.undo()  # verification below legitimately reads shards

    assert report["bands_rebuilt"] == num_bands
    survivors = set(CONTRIBS["ds_a"]) | set(CONTRIBS["ds_c"])
    for bid in range(num_bands):
        assert _band_union(index_dir, bid) == survivors


def test_legacy_l0_shard_provenance_inferred(tmp_path):
    """A legacy L0 manifest entry (no ``source_datasets``) is treated as
    singleton provenance ``[dataset]``: withdrawing an already-compacted tag
    affects only the merged shard and rebuilds from its true survivors, not the
    unrelated legacy L0 shard (which pre-fix was conservatively 'unknown')."""
    from puffer.index import load_manifest, write_manifest
    from puffer.withdraw import plan_band_withdrawal

    idx = tmp_path / "hash_index"
    append_shard(np.array([1, 2, 3], dtype=np.int64), idx, 0, "ds_a")
    append_shard(np.array([3, 4, 5], dtype=np.int64), idx, 0, "ds_b")
    compact_band(idx, 0, tier_fanout=2, ram_budget=1 << 30)   # a+b -> one merged run
    append_shard(np.array([6, 7, 8], dtype=np.int64), idx, 0, "ds_c")  # standalone L0

    man = load_manifest(idx, 0)                               # emulate a legacy manifest
    for s in man["shards"]:
        if s.get("dataset") == "ds_c":
            s.pop("source_datasets", None)
    write_manifest(idx, 0, man)

    plan = plan_band_withdrawal(idx, 0, "ds_a")
    assert plan["tag_files"] == []                            # ds_a compacted away
    assert len(plan["affected"]) == 1                         # only the merged shard
    assert plan["contributors"] == ["ds_b"]                   # not forced to None/full


# ---------------------------------------------------------------------------
# Merge-backend gate: refuse up front, never partially withdraw
# ---------------------------------------------------------------------------

def test_withdraw_refuses_up_front_without_merge_backend(tmp_path, monkeypatch):
    """When any band needs a survivor rebuild and the compiled loser-tree is
    unavailable (no C compiler), ``withdraw_dataset`` must refuse BEFORE
    mutating any band -- no partial withdrawal -- and a later retry with the
    backend available must complete normally."""
    import puffer.bounded_merge as bm

    num_bands = 2
    state_dir = tmp_path / "state"
    _write_sidecars(state_dir, num_bands, CONTRIBS)
    index_dir = state_dir / "hash_index"
    for bid in range(num_bands):
        compact_band(index_dir, bid, tier_fanout=3, ram_budget=1 << 30)

    # Victim artifacts that a completed withdrawal would delete.
    removals = state_dir / "removals" / "ds_b.parquet"
    removals.parent.mkdir(parents=True, exist_ok=True)
    removals.write_bytes(b"sentinel")

    cfg = PufferConfig(num_bands=num_bands, num_perm=num_bands, ram_budget_bytes=1 << 30)
    before_unions = [_band_union(index_dir, bid) for bid in range(num_bands)]
    before_manifests = [load_manifest(index_dir, bid) for bid in range(num_bands)]
    before_sidecars = {
        (tag, bid): (state_dir / "datasets" / tag / f"band_{bid:02d}.bin").read_bytes()
        for tag in CONTRIBS for bid in range(num_bands)
    }

    monkeypatch.setattr(bm, "_load_lib", lambda: False)
    with pytest.raises(RuntimeError, match="No band was modified"):
        withdraw_dataset(state_dir, "ds_b", cfg)

    # Nothing mutated: unions, manifests, every dataset-band sidecar
    # byte-identical, and the victim's artifacts all still present.
    assert [_band_union(index_dir, bid) for bid in range(num_bands)] == before_unions
    assert [load_manifest(index_dir, bid) for bid in range(num_bands)] == before_manifests
    for (tag, bid), payload in before_sidecars.items():
        assert (state_dir / "datasets" / tag / f"band_{bid:02d}.bin").read_bytes() == payload
    assert removals.read_bytes() == b"sentinel"
    assert (state_dir / "datasets" / "ds_b" / "meta.json").exists()

    monkeypatch.undo()
    report = withdraw_dataset(state_dir, "ds_b", cfg)  # retry completes
    assert report["bands_rebuilt"] == num_bands
    survivors = set(CONTRIBS["ds_a"]) | set(CONTRIBS["ds_c"])
    for bid in range(num_bands):
        assert _band_union(index_dir, bid) == survivors
    # Retry also completed the artifact cleanup the refusal had preserved.
    assert not (state_dir / "datasets" / "ds_b").exists()
    assert not removals.exists()
    for tag in ("ds_a", "ds_c"):
        for bid in range(num_bands):
            assert (state_dir / "datasets" / tag / f"band_{bid:02d}.bin").exists()


def test_withdraw_standalone_needs_no_merge_backend(tmp_path, monkeypatch):
    """A purely-standalone (O(1)) withdrawal has no affected bands and must
    succeed even when the compiled backend is unavailable."""
    import puffer.bounded_merge as bm

    num_bands = 1
    state_dir = tmp_path / "state"
    _write_sidecars(state_dir, num_bands, CONTRIBS)  # no compaction: all standalone
    index_dir = state_dir / "hash_index"

    monkeypatch.setattr(bm, "_load_lib", lambda: False)
    cfg = PufferConfig(num_bands=num_bands, num_perm=num_bands, ram_budget_bytes=1 << 30)
    report = withdraw_dataset(state_dir, "ds_b", cfg)
    assert report["bands_o1"] == num_bands and report["bands_rebuilt"] == 0
    assert _band_union(index_dir, 0) == set(CONTRIBS["ds_a"]) | set(CONTRIBS["ds_c"])
