"""Contract tests for the optional distributed executor.

The first two tests require no Ray installation.  The integration test is
intentionally skipped in ordinary developer environments, but exercises the
same two-release decision path when the optional extra is installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

import numpy as np
import pytest


def test_ray_is_lazy_and_has_install_hint() -> None:
    """Importing PUFFER/Ray dispatch must not make Ray a local dependency."""
    if importlib.util.find_spec("ray") is not None:
        pytest.skip("ray is installed; the missing-dependency hint cannot fire")
    sys.modules.pop("puffer.ray_exec", None)
    module = importlib.import_module("puffer.ray_exec")
    assert "ray" not in sys.modules
    with pytest.raises(ImportError, match=r"pip install puffer-dedup\[ray\]"):
        module.ray_compute_band_keys([], object())
    with pytest.raises(ImportError, match=r"pip install puffer-dedup\[ray\]"):
        module.ray_screen_chunks(np.zeros((1, 1), dtype=np.int64), ".", None, object(), chunk_rows=1)


def test_screen_chunk_combination_preserves_rows_and_sums_counters() -> None:
    """Workers may finish arbitrarily; the driver restores release row order."""
    from puffer.ray_exec import combine_screen_chunks, split_row_chunks

    assert split_row_chunks(7, 3) == [(0, 3), (3, 6), (6, 7)]
    mask, counters = combine_screen_chunks(
        7,
        [
            (3, 6, np.array([False, True, False]), {"probes_scheduled": 12, "probes_done": 7}),
            (0, 3, np.array([True, False, False]), {"probes_scheduled": 12, "probes_done": 10}),
            (6, 7, np.array([True]), {"probes_scheduled": 4, "probes_done": 1}),
        ],
    )
    assert mask.tolist() == [True, False, False, False, True, False, True]
    assert counters == {"probes_scheduled": 28, "probes_done": 18}

def test_ray_cap_defaults_to_cluster_and_honors_total_overrides(monkeypatch) -> None:
    """Auto capacity uses all advertised cluster CPUs; overrides remain totals."""
    from puffer import ray_exec

    monkeypatch.delenv("PUFFER_RAY_MAX_IN_FLIGHT", raising=False)
    monkeypatch.setattr(ray_exec, "_ensure_ray", lambda: None)
    monkeypatch.setattr(ray_exec, "cluster_cpus", lambda default=8: 4 * 64)
    assert ray_exec.resolve_max_in_flight() == 256
    assert ray_exec.resolve_max_in_flight(64) == 64

    monkeypatch.setenv("PUFFER_RAY_MAX_IN_FLIGHT", "128")
    assert ray_exec.resolve_max_in_flight() == 128
    assert ray_exec.resolve_max_in_flight(32) == 32


def test_resolve_inputs_returns_absolute_paths(tmp_path, monkeypatch) -> None:
    """Ray workers lack the driver's CWD: input paths dispatched to them must be
    absolute. A relative path silently resolves to nothing on a worker."""
    import polars as pl

    from puffer.pipeline import _resolve_inputs

    (tmp_path / "a.parquet").write_bytes(b"")
    pl.DataFrame({"text": ["x"]}).write_parquet(tmp_path / "b.parquet")
    monkeypatch.chdir(tmp_path)
    resolved = _resolve_inputs(["*.parquet"])  # relative glob
    assert resolved, "glob should match"
    assert all(p.is_absolute() for p in resolved), resolved


def test_state_dir_is_absolute_from_relative(tmp_path, monkeypatch) -> None:
    """A relative state_dir must be absolutised so index_dir handed to Ray
    screen workers is valid regardless of their CWD (else cross-history dedup
    silently returns zero hits)."""
    from puffer import Deduper, PufferConfig

    monkeypatch.chdir(tmp_path)
    dd = Deduper("relstate", PufferConfig())
    assert dd.state_dir.is_absolute()


def test_ray_two_release_matches_local_masks(tmp_path, monkeypatch) -> None:
    """Ray tasks preserve the local two-release cross-history decision mask."""
    ray = pytest.importorskip("ray")
    import polars as pl

    from puffer import Deduper, PufferConfig
    from puffer.ray_exec import ray_screen_chunks
    from puffer.screen import screen_release
    from puffer.signature import compute_band_keys

    cfg = PufferConfig(
        num_perm=8,
        num_bands=2,
        ngram_type="word",
        ngram_size=1,
        n_workers=2,
    )
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pl.DataFrame({"text": ["alpha beta gamma", "separate first record"]}).write_parquet(first)
    pl.DataFrame({"text": ["alpha beta gamma", "novel second record"]}).write_parquet(second)

    # Relative state_dir on purpose: Ray screen workers do not inherit the
    # driver CWD, so index_dir must be absolutised or cross-history dedup
    # silently returns zero hits.
    # ray.init BEFORE any executor="ray" ingest: _ensure_ray would otherwise
    # auto-join (address="auto") any reachable multi-node cluster, whose remote
    # workers cannot see this test's node-local tmp files.
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    monkeypatch.chdir(tmp_path)
    try:
        local_state = "local-state"
        ray_state = "ray-state"
        local = Deduper(local_state, cfg)
        distributed = Deduper(ray_state, cfg, executor="ray")
        local.ingest([first], dataset="r0", output_dir=tmp_path / "local-r0")
        distributed.ingest([first], dataset="r0", output_dir=tmp_path / "ray-r0")

        texts = pl.read_parquet(second).get_column("text").to_list()
        keys = compute_band_keys(texts, cfg)
        expected = screen_release(keys, local.state_dir / "hash_index", None, cfg, {})

        actual, counters = ray_screen_chunks(
            keys, distributed.state_dir / "hash_index", None, cfg,
        )
        assert actual.tolist() == expected.tolist()
        assert counters["probes_done"] <= counters["probes_scheduled"]
        local_report = local.ingest([second], dataset="r1", output_dir=tmp_path / "local-r1")
        ray_report = distributed.ingest([second], dataset="r1", output_dir=tmp_path / "ray-r1")
        assert ray_report.n_cross_removed == local_report.n_cross_removed
        assert ray_report.n_output == local_report.n_output
    finally:
        ray.shutdown()


def test_plan_release_chunks_row_group_aligned_round_robin(monkeypatch) -> None:
    """Chunks never split a row group, cover every row exactly once, and the
    dispatch order interleaves files (round-robin) so a skewed file cannot
    monopolize early scheduling waves."""
    from puffer import ray_dist, ray_exec
    from puffer.config import PufferConfig

    monkeypatch.setattr(ray_exec, "_ensure_ray", lambda: None)
    monkeypatch.setattr(ray_exec, "cluster_cpus", lambda default=8: 4)
    monkeypatch.delenv("PUFFER_RAY_MAX_IN_FLIGHT", raising=False)

    # file 0: skewed large (10 row groups x 10 rows); file 1: small (1x5);
    # file 2: zero-row.  target = max(sig_chunk_rows=8, ceil(105/12)=9) = 9
    # -> one row group (10 rows) already exceeds it, so chunks are 1 rg each.
    cfg = PufferConfig(sig_chunk_rows=8)
    rg_rows = [[10] * 10, [5], []]
    rg_bytes = [[100] * 10, [50], []]
    chunks = ray_dist.plan_release_chunks(rg_rows, rg_bytes, 105, cfg)

    per_file: dict[int, list] = {}
    for c in chunks:
        per_file.setdefault(c.file_index, []).append(c)
    assert len(per_file[0]) == 10 and all(c.n_rows == 10 for c in per_file[0])
    assert [c.row_offset for c in per_file[0]] == [i * 10 for i in range(10)]
    assert len(per_file[1]) == 1 and per_file[1][0].n_rows == 5
    assert len(per_file[2]) == 1 and per_file[2][0].n_rows == 0  # empty unit kept
    # round-robin: first wave holds every file's chunk 0
    first_wave = chunks[: len(per_file)]
    assert sorted(c.file_index for c in first_wave) == [0, 1, 2]
    assert all(c.chunk_index == 0 for c in first_wave)

    # coalescing: raise the target so several row groups pack into one chunk
    cfg2 = PufferConfig(sig_chunk_rows=35)
    packed = ray_dist.plan_release_chunks(rg_rows, rg_bytes, 105, cfg2)
    f0 = sorted((c for c in packed if c.file_index == 0), key=lambda c: c.chunk_index)
    assert [c.n_rows for c in f0] == [30, 30, 30, 10]
    assert [c.rg_lo for c in f0] == [0, 3, 6, 9]
    assert sum(c.n_rows for c in packed) == 105


def test_ray_distributed_ingest_matches_local(tmp_path) -> None:
    """The distributed path reproduces local decisions exactly --
    same removal reasons per row, same output rows, same committed shards --
    including on a skewed multi-row-group release that forces chunked spool
    and chunked (part-file) output writes."""
    ray = pytest.importorskip("ray")
    import polars as pl

    from puffer import Deduper, PufferConfig

    cfg = PufferConfig(
        num_perm=8,
        num_bands=2,
        ngram_type="word",
        ngram_size=1,
        n_workers=2,
        record_row_signatures=True,
        sig_chunk_rows=4,  # tiny target -> multi-chunk files
    )
    # Release 0: one skewed file (12 rows across small row groups -> several
    # chunks + part-file outputs) and one small file; duplicates within and
    # across files. Release 1 collides with release 0 across history.
    r0a, r0b = tmp_path / "r0_a.parquet", tmp_path / "r0_b.parquet"
    r1a, r1b = tmp_path / "r1_a.parquet", tmp_path / "r1_b.parquet"
    big = [f"row number {i} words" for i in range(9)] + ["dup dup dup"] * 3
    pl.DataFrame({"text": big}).write_parquet(r0a, row_group_size=3)
    pl.DataFrame({"text": ["dup dup dup", "other zero text"]}).write_parquet(r0b)
    pl.DataFrame({"text": ["row number 3 words", "brand new record", "same same same"]}).write_parquet(r1a, row_group_size=2)
    pl.DataFrame({"text": ["same same same", "another new one"]}).write_parquet(r1b)

    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    try:
        local = Deduper(tmp_path / "local-state", cfg)
        dist = Deduper(tmp_path / "ray-state", cfg, executor="ray")
        for tag, files in (("r0", [r0a, r0b]), ("r1", [r1a, r1b])):
            lrep = local.ingest(files, dataset=tag, output_dir=tmp_path / f"local-out-{tag}")
            rrep = dist.ingest(files, dataset=tag, output_dir=tmp_path / f"ray-out-{tag}")
            assert rrep.ray_transport == "spool"
            assert (rrep.n_input, rrep.n_output) == (lrep.n_input, lrep.n_output)
            assert rrep.n_within_removed == lrep.n_within_removed
            assert rrep.n_cross_removed == lrep.n_cross_removed
            # identical removal rows + reasons
            lrem = pl.read_parquet(local.state_dir / "removals" / f"{tag}.parquet")
            rrem = pl.read_parquet(dist.state_dir / "removals" / f"{tag}.parquet")
            assert lrem.sort(lrem.columns).rows() == rrem.sort(rrem.columns).rows()
            # identical surviving row stream: local files are 1:1 with inputs
            # (input order); dist output_files are part files in the same
            # global row order, so the concatenated streams must be equal.
            lrows = [r for f in lrep.output_files for r in pl.read_parquet(f).rows()]
            rrows = [r for f in rrep.output_files for r in pl.read_parquet(f).rows()]
            assert lrows == rrows
            # the skewed file must actually have produced multiple parts
            if tag == "r0":
                assert sum(".part" in f for f in rrep.output_files) >= 2
            # identical committed shard bytes and row-signature artifacts
            for band_dir in sorted((local.state_dir / "hash_index").glob("band_*")):
                rel = band_dir.relative_to(local.state_dir)
                for shard in sorted(band_dir.glob("*.bin")):
                    other = dist.state_dir / rel / shard.name
                    assert other.exists(), f"missing shard {other}"
                    assert shard.read_bytes() == other.read_bytes()
            lsig = local.state_dir / "datasets" / tag / "row_band_keys.i64"
            rsig = dist.state_dir / "datasets" / tag / "row_band_keys.i64"
            assert lsig.read_bytes() == rsig.read_bytes()
    finally:
        ray.shutdown()
