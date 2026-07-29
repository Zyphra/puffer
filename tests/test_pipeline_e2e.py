"""End-to-end behavioral tests for ``puffer.pipeline.Deduper`` (ingest,
withdraw, stats, CLI) over seeded synthetic parquet releases.

Documents are ~500-char strings drawn from a fixed random-word pool: long
enough that char-20-gram MinHash shingling has plenty of distinct shingles,
so a single planted edit only perturbs a small fraction of them (near-dups
reliably band-collide) while unrelated documents essentially never do.
"""

from __future__ import annotations

import json
import random
import string
import subprocess
import sys
from pathlib import Path

import polars as pl

from puffer.config import PufferConfig
from puffer.pipeline import Deduper

# ---------------------------------------------------------------------------
# Synthetic corpus generation
# ---------------------------------------------------------------------------


def _word_pool(seed: int, n: int = 6000) -> list[str]:
    rng = random.Random(seed)
    return [
        "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 9)))
        for _ in range(n)
    ]


_POOL = _word_pool(1)


def _make_doc(rng: random.Random, target_chars: int = 500) -> str:
    """A ~500-char document: whitespace-joined random words from ``_POOL``."""
    words: list[str] = []
    total = 0
    while total < target_chars:
        w = rng.choice(_POOL)
        words.append(w)
        total += len(w) + 1
    return " ".join(words)


def _one_char_edit(text: str, rng: random.Random) -> str:
    """Substitute one alphabetic character in ``text`` for a different one."""
    letters = [i for i, c in enumerate(text) if c.isalpha()]
    idx = rng.choice(letters)
    chars = list(text)
    orig = chars[idx]
    chars[idx] = rng.choice([c for c in string.ascii_lowercase if c != orig])
    return "".join(chars)


def _write_release(dir_: Path, frame: "pl.DataFrame", filename: str = "part-0.parquet") -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / filename
    frame.write_parquet(path)
    return path


def _cfg(**overrides) -> PufferConfig:
    params = {"num_bands": 8, "num_perm": 64, "tier_fanout": 4}
    params.update(overrides)
    return PufferConfig(**params)


# ---------------------------------------------------------------------------
# 1. Single-release ingest: identical dup, near dup, unrelated docs kept
# ---------------------------------------------------------------------------


def test_single_release_ingest_within_and_unrelated(tmp_path):
    rng = random.Random(101)
    docs = [_make_doc(rng) for _ in range(400)]

    dup_of = 50
    docs.append(docs[dup_of])  # planted identical dup -> "within" (fuzzy Jaccard 1.0)

    near_dup_source = 120
    docs.append(_one_char_edit(docs[near_dup_source], rng))  # planted near dup -> "within"

    unrelated_samples = [0, 30, 200, 350, 399]

    frame = pl.DataFrame(
        {
            "id": list(range(len(docs))),
            "text": docs,
            "weight": [float(i % 7) for i in range(len(docs))],
        }
    )
    in_dir = tmp_path / "in"
    _write_release(in_dir, frame)

    dd = Deduper(tmp_path / "state", _cfg())
    report = dd.ingest(
        [str(in_dir / "*.parquet")], dataset="A", output_dir=str(tmp_path / "out"),
    )

    assert report.n_input == len(docs)
    assert report.n_within_removed == 2  # identical dup + near dup, both via fuzzy LSH
    assert report.n_cross_removed == 0
    assert report.n_output == len(docs) - 2
    assert len(report.output_files) == 1

    out_frame = pl.read_parquet(report.output_files[0])
    assert out_frame.schema == frame.schema

    out_texts = set(out_frame.get_column("text").to_list())
    for i in unrelated_samples:
        assert docs[i] in out_texts

    removals = pl.read_parquet(tmp_path / "state" / "removals" / "A.parquet")
    reasons = removals.get_column("reason").to_list()
    assert reasons.count("within") == report.n_within_removed
    assert reasons.count("cross") == report.n_cross_removed
    assert len(removals) == len(docs) - report.n_output


def test_ingest_emits_phase_logs_at_info(tmp_path, caplog):
    """`-v` (INFO) surfaces phase progress: an ingest-begin line,
    per-phase timing lines, a hash_index summary, and an ingest-done line with
    the release counts. Default (WARNING) stays quiet."""
    import logging

    rng = random.Random(77)
    docs = [_make_doc(rng) for _ in range(60)]
    docs.append(docs[3])  # one exact dup so a removal count is non-trivial
    _write_release(tmp_path / "in", pl.DataFrame({"text": docs}))

    dd = Deduper(tmp_path / "state", _cfg())
    with caplog.at_level(logging.INFO, logger="puffer.pipeline"):
        dd.ingest([str(tmp_path / "in" / "*.parquet")], dataset="A",
                  output_dir=str(tmp_path / "out"))
    msgs = [r.getMessage() for r in caplog.records if r.name == "puffer.pipeline"]
    blob = "\n".join(msgs)
    assert any("ingest begin" in m and "local" in m for m in msgs)
    assert "signatures computed" in blob
    assert "within-release keep-first" in blob
    assert "cross-history screen" in blob
    assert "committed 8 band(s)" in blob
    assert "wrote 1 output file(s)" in blob
    assert "hash_index:" in blob
    assert any("ingest done" in m and "out=" in m for m in msgs)

    # default level: no INFO chatter leaks
    caplog.clear()
    _write_release(tmp_path / "in2", pl.DataFrame({"text": docs}))
    with caplog.at_level(logging.WARNING, logger="puffer.pipeline"):
        dd.ingest([str(tmp_path / "in2" / "*.parquet")], dataset="B",
                  output_dir=str(tmp_path / "out2"))
    assert not [r for r in caplog.records
                if r.name == "puffer.pipeline" and r.levelno == logging.INFO]


# ---------------------------------------------------------------------------
# 2. Two-release incremental: cross removal + early-stop probe savings
# ---------------------------------------------------------------------------


def test_incremental_second_release_cross_removes_and_early_stops(tmp_path):
    rng_a = random.Random(11)
    docs_a = [_make_doc(rng_a) for _ in range(200)]
    _write_release(tmp_path / "in" / "A", pl.DataFrame({"text": docs_a}))

    rng_b = random.Random(12)
    docs_b_fresh = [_make_doc(rng_b) for _ in range(150)]
    # near-dups of 50 of A's docs -> caught by the cross-history fuzzy screen.
    docs_b_crossdup = [_one_char_edit(d, rng_b) for d in docs_a[:50]]
    docs_b = docs_b_crossdup + docs_b_fresh
    _write_release(tmp_path / "in" / "B", pl.DataFrame({"text": docs_b}))

    cfg = _cfg()
    dd = Deduper(tmp_path / "state", cfg)
    dd.ingest([str(tmp_path / "in" / "A" / "*.parquet")], dataset="A", output_dir=str(tmp_path / "out" / "A"))
    report_b = dd.ingest(
        [str(tmp_path / "in" / "B" / "*.parquet")], dataset="B", output_dir=str(tmp_path / "out" / "B"),
    )

    assert report_b.n_cross_removed >= 40
    assert report_b.probes_scheduled > 0
    assert report_b.probes_done < report_b.probes_scheduled


# ---------------------------------------------------------------------------
# 3. Retry idempotency: re-ingesting the same dataset is a pure, byte-
#    identical replay with no self-collision
# ---------------------------------------------------------------------------


def test_reingesting_same_dataset_is_idempotent_replay(tmp_path):
    rng_a = random.Random(21)
    docs_a = [_make_doc(rng_a) for _ in range(120)]
    _write_release(tmp_path / "in" / "A", pl.DataFrame({"text": docs_a}))

    rng_b = random.Random(22)
    docs_b = [_make_doc(rng_b) for _ in range(120)]
    _write_release(tmp_path / "in" / "B", pl.DataFrame({"text": docs_b}))

    cfg = _cfg()
    dd = Deduper(tmp_path / "state", cfg)
    dd.ingest([str(tmp_path / "in" / "A" / "*.parquet")], dataset="A", output_dir=str(tmp_path / "out" / "A"))
    first = dd.ingest(
        [str(tmp_path / "in" / "B" / "*.parquet")], dataset="B", output_dir=str(tmp_path / "out" / "B"),
    )
    out_path = Path(first.output_files[0])
    bytes_after_first = out_path.read_bytes()

    # Fresh Deduper instance over the SAME state dir, re-ingest B again.
    dd2 = Deduper(tmp_path / "state", cfg)
    second = dd2.ingest(
        [str(tmp_path / "in" / "B" / "*.parquet")], dataset="B", output_dir=str(tmp_path / "out" / "B"),
    )

    assert out_path.read_bytes() == bytes_after_first  # never rewritten -> trivially byte-identical
    assert (first.n_input, first.n_within_removed, first.n_cross_removed, first.n_output, first.output_files) == (
        second.n_input, second.n_within_removed, second.n_cross_removed, second.n_output, second.output_files,
    )
    # No self-collision: replay never re-screens, so no spurious removal appears.
    assert second.n_cross_removed == first.n_cross_removed == 0


# ---------------------------------------------------------------------------
# 4. withdraw(A) then ingest A-copies as C: copies survive
# ---------------------------------------------------------------------------


def test_withdrawn_dataset_no_longer_causes_cross_collisions(tmp_path):
    rng = random.Random(31)
    docs_a = [_make_doc(rng) for _ in range(80)]
    _write_release(tmp_path / "in" / "A", pl.DataFrame({"text": docs_a}))
    _write_release(tmp_path / "in" / "C", pl.DataFrame({"text": docs_a}))  # exact copies

    cfg = _cfg()
    dd = Deduper(tmp_path / "state", cfg)
    dd.ingest([str(tmp_path / "in" / "A" / "*.parquet")], dataset="A", output_dir=str(tmp_path / "out" / "A"))
    dd.withdraw("A")

    report_c = dd.ingest(
        [str(tmp_path / "in" / "C" / "*.parquet")], dataset="C", output_dir=str(tmp_path / "out" / "C"),
    )

    assert report_c.n_output == len(docs_a)
    assert report_c.n_cross_removed == 0

    out_frame = pl.read_parquet(report_c.output_files[0])
    assert set(out_frame.get_column("text").to_list()) == set(docs_a)


# ---------------------------------------------------------------------------
# 5. Withdrawal keeps a shared key alive via a surviving presenter: a doc
#    presented by both A and B stays flagged after A is withdrawn, because
#    B's committed band keys (and sidecar) still present it.
# ---------------------------------------------------------------------------


def test_shared_doc_stays_flagged_after_withdrawing_first_presenter(tmp_path):
    rng = random.Random(41)
    shared_doc = _make_doc(rng)
    docs_a = [_make_doc(rng) for _ in range(30)] + [shared_doc]
    docs_b = [_make_doc(rng) for _ in range(30)] + [shared_doc]
    _write_release(tmp_path / "in" / "A", pl.DataFrame({"text": docs_a}))
    _write_release(tmp_path / "in" / "B", pl.DataFrame({"text": docs_b}))

    cfg = _cfg()
    dd = Deduper(tmp_path / "state", cfg)
    dd.ingest([str(tmp_path / "in" / "A" / "*.parquet")], dataset="A", output_dir=str(tmp_path / "out" / "A"))
    report_b = dd.ingest(
        [str(tmp_path / "in" / "B" / "*.parquet")], dataset="B", output_dir=str(tmp_path / "out" / "B"),
    )
    assert report_b.n_cross_removed == 1  # B's copy of shared_doc collides with A's (fuzzy)

    dd.withdraw("A")

    _write_release(tmp_path / "in" / "C", pl.DataFrame({"text": [shared_doc]}))
    report_c = dd.ingest(
        [str(tmp_path / "in" / "C" / "*.parquet")], dataset="C", output_dir=str(tmp_path / "out" / "C"),
    )
    # A is gone, but B also presented shared_doc: full-key commit + B's sidecar
    # keep that band key in the survivor-rebuilt index, so re-presenting it
    # still collides (cross) and is dropped.
    assert report_c.n_cross_removed == 1
    assert report_c.n_output == 0


def test_compacted_withdrawal_through_ingest_matches_fresh_survivor_index(tmp_path):
    """Provenance + withdrawal end-to-end through real ingest (not hand-built
    fixtures): ingest A,B,C at tier_fanout=2 so A and B compact into one
    ``source_datasets``-stamped L1 run once C's ingest unprotects them; withdraw
    B; assert every band's key union equals a fresh index built from only the
    survivors A+C. This proves stamping survives append+merge and drives the
    rebuild -- unions are compared, not run structure (the reference's A,C stay
    two unmerged L0 runs)."""
    from puffer.index import iter_shards, load_manifest, read_shard_bin

    rng = random.Random(7)
    releases = {n: [_make_doc(rng) for _ in range(60)] for n in ("A", "B", "C")}
    for name, docs in releases.items():
        _write_release(tmp_path / "in" / name, pl.DataFrame({"text": docs}))

    cfg = _cfg(tier_fanout=2)
    dd = Deduper(tmp_path / "state", cfg)
    for name in ("A", "B", "C"):
        dd.ingest([str(tmp_path / "in" / name / "*.parquet")], dataset=name,
                  output_dir=str(tmp_path / "out" / name))

    # Real ingest+merge produced a provenance-stamped A+B run (not a fixture).
    idx = tmp_path / "state" / "hash_index"
    merged = [
        s
        for b in range(cfg.num_bands)
        for s in load_manifest(idx, b)["shards"]
        if set(s.get("source_datasets") or []) >= {"A", "B"}
    ]
    assert merged, "expected a compacted A+B shard carrying stamped source_datasets"

    dd.withdraw("B")

    ref = Deduper(tmp_path / "state_ref", cfg)
    for name in ("A", "C"):
        ref.ingest([str(tmp_path / "in" / name / "*.parquet")], dataset=name,
                   output_dir=str(tmp_path / "out_ref" / name))

    def band_union(state_dir: Path, band_id: int) -> set[int]:
        keys: set[int] = set()
        for _meta, path in iter_shards(state_dir / "hash_index", band_id):
            keys |= set(read_shard_bin(path).tolist())
        return keys

    for b in range(cfg.num_bands):
        assert band_union(tmp_path / "state", b) == band_union(tmp_path / "state_ref", b), f"band {b} union mismatch after withdrawing B"

    # B's per-release artifacts are gone: removals parquet and the per-dataset
    # sidecar dir (withdraw_dataset rmtrees state/datasets/<tag>).
    assert not (tmp_path / "state" / "removals" / "B.parquet").exists()
    assert not (tmp_path / "state" / "datasets" / "B").exists()


# ---------------------------------------------------------------------------
# 6. withdraw default retains outputs; purge_outputs unlinks exactly the
#    recorded files and leaves an unrelated file untouched
# ---------------------------------------------------------------------------


def test_withdraw_retains_outputs_by_default(tmp_path):
    rng = random.Random(51)
    docs = [_make_doc(rng) for _ in range(50)]
    _write_release(tmp_path / "in" / "A", pl.DataFrame({"text": docs}))

    cfg = _cfg()
    dd = Deduper(tmp_path / "state", cfg)
    report = dd.ingest(
        [str(tmp_path / "in" / "A" / "*.parquet")], dataset="A", output_dir=str(tmp_path / "out" / "A"),
    )

    withdraw_report = dd.withdraw("A")

    assert withdraw_report.outputs_purged is False
    assert sorted(withdraw_report.outputs_retained) == sorted(report.output_files)
    assert all(Path(f).exists() for f in withdraw_report.outputs_retained)


def test_withdraw_purge_outputs_unlinks_only_recorded_files(tmp_path):
    rng = random.Random(52)
    docs = [_make_doc(rng) for _ in range(50)]
    _write_release(tmp_path / "in" / "A", pl.DataFrame({"text": docs}))

    cfg = _cfg()
    dd = Deduper(tmp_path / "state", cfg)
    report = dd.ingest(
        [str(tmp_path / "in" / "A" / "*.parquet")], dataset="A", output_dir=str(tmp_path / "out" / "A"),
    )
    out_dir = Path(report.output_files[0]).parent
    unrelated = out_dir / "unrelated.txt"
    unrelated.write_text("keep me")

    withdraw_report = dd.withdraw("A", purge_outputs=True)

    assert withdraw_report.outputs_purged is True
    assert withdraw_report.outputs_retained == []
    assert all(not Path(f).exists() for f in report.output_files)
    assert unrelated.exists()

    ledger = json.loads((tmp_path / "state" / "ledger.json").read_text())
    withdraw_events = [e for e in ledger if e["op"] == "withdraw" and e["dataset"] == "A"]
    assert withdraw_events
    assert withdraw_events[-1]["purged"] is True


# ---------------------------------------------------------------------------
# 7. CLI smoke test: ingest, stats, withdraw each exit 0
# ---------------------------------------------------------------------------


def test_cli_ingest_stats_withdraw_smoke(tmp_path):
    rng = random.Random(61)
    docs = [_make_doc(rng) for _ in range(60)]
    in_dir = tmp_path / "in"
    _write_release(in_dir, pl.DataFrame({"text": docs}))

    state_dir = tmp_path / "state"
    out_dir = tmp_path / "out"

    def run_cli(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "puffer.cli", "--state-dir", str(state_dir), *args],
            capture_output=True, text=True, timeout=120,
        )

    ingest_proc = run_cli(
        "ingest", str(in_dir / "*.parquet"), "--dataset", "A", "--output-dir", str(out_dir),
    )
    assert ingest_proc.returncode == 0, ingest_proc.stderr
    ingest_report = json.loads(ingest_proc.stdout)
    assert ingest_report["dataset"] == "A"
    assert ingest_report["n_output"] == len(docs)

    stats_proc = run_cli("stats")
    assert stats_proc.returncode == 0, stats_proc.stderr
    stats = json.loads(stats_proc.stdout)
    assert stats["datasets"] == ["A"]

    withdraw_proc = run_cli("withdraw", "--dataset", "A")
    assert withdraw_proc.returncode == 0, withdraw_proc.stderr
    withdraw_report = json.loads(withdraw_proc.stdout)
    assert withdraw_report["dataset"] == "A"


def test_faithful_rebuild_survivors_matches_fresh_ingest(tmp_path):
    """Both faithful engines match a fresh survivor-only ingest. Withdrawing
    middle release B resurrects C's exact duplicate; C's near-duplicate then
    changes from cross-removed to within-removed, while D still sees C's full
    committed keys. This exercises exact/within/cross order, not just unions.
    """
    from puffer.index import iter_shards, read_shard_bin, sanitize_tag
    from puffer.withdraw import rebuild_survivors, rebuild_survivors_suffix

    rng = random.Random(11)
    shared = _make_doc(rng)
    near = ("x" if shared[0] != "x" else "y") + shared[1:]
    docs = {
        "A": [_make_doc(rng) for _ in range(50)],
        "B": [_make_doc(rng) for _ in range(40)] + [shared],
        # Both are removed in live (exact, then cross). Without B, shared
        # survives and near becomes a within-release duplicate of shared.
        "C": [_make_doc(rng) for _ in range(40)] + [shared, near],
        # C commits all presented keys, so removing B must not resurrect D.
        "D": [_make_doc(rng) for _ in range(20)] + [shared],
    }
    for name, d in docs.items():
        _write_release(tmp_path / "in" / name, pl.DataFrame({"text": d}))

    cfg = _cfg(tier_fanout=2, record_row_signatures=True)
    live = Deduper(tmp_path / "live", cfg)
    for name in ("A", "B", "C", "D"):
        live.ingest([str(tmp_path / "in" / name / "*.parquet")], dataset=name,
                    output_dir=str(tmp_path / "liveout" / name))

    oracle = rebuild_survivors(tmp_path / "live", "B", tmp_path / "oracle", cfg)

    ref = Deduper(tmp_path / "ref", cfg)
    for name in ("A", "C", "D"):
        ref.ingest([str(tmp_path / "in" / name / "*.parquet")], dataset=name,
                   output_dir=str(tmp_path / "ref" / "outputs" / name))

    # Prefix reuse does not need A's input online. The full oracle/ref were
    # already built; deleting it before fast replay proves A is not re-ingested.
    next((tmp_path / "in" / "A").glob("*.parquet")).unlink()
    fast = rebuild_survivors_suffix(
        tmp_path / "live", "B", tmp_path / "fast", cfg,
    )

    def band_unions(state: Path) -> dict:
        idx = state / "hash_index"
        return {
            b: frozenset(k for _m, p in iter_shards(idx, b) for k in read_shard_bin(p).tolist())
            for b in range(cfg.num_bands)
        }

    def out_frames(outputs_root: Path, tag: str) -> dict:
        d = outputs_root / sanitize_tag(tag)
        return {f.name: pl.read_parquet(f) for f in sorted(d.glob("*.parquet"))}

    def resolved_removals(state: Path) -> list:
        # Sorted multiset of (dataset, uri, row, reason) -- keyed by dataset so
        # identical (uri,row,reason) across releases never collapses.
        from puffer import paths as _paths
        out: list = []
        rem_dir = state / "removals"
        for f in sorted(rem_dir.glob("*.parquet")) if rem_dir.exists() else []:
            df = pl.read_parquet(f)
            if not len(df):
                continue
            ids = df.get_column("file_id").to_list()
            uri = {e["id"]: e["uri"] for e in _paths.resolve_ids(state, sorted(set(ids)))}
            for fid, row, reason in zip(ids, df.get_column("source_row").to_list(), df.get_column("reason").to_list()):
                out.append((f.stem, uri[fid], int(row), reason))
        return sorted(out)

    for rebuilt in ("oracle", "fast"):
        rebuilt_state = tmp_path / rebuilt
        # Index membership identical.
        assert band_unions(rebuilt_state) == band_unions(tmp_path / "ref")
        # Full per-file output DataFrames identical (schema + every column +
        # order + multiplicity), per surviving dataset.
        for tag in ("A", "C", "D"):
            got = out_frames(rebuilt_state / "outputs", tag)
            expected = out_frames(tmp_path / "ref" / "outputs", tag)
            assert got.keys() == expected.keys()
            for name in got:
                assert got[name].equals(expected[name]), f"{rebuilt}/{tag}/{name} differs"
        # Removal records identical once resolved to dataset/URI/row/reason.
        assert resolved_removals(rebuilt_state) == resolved_removals(tmp_path / "ref")

    # Fast suffix releases actually used compact row artifacts, not text
    # shingling/MinHash. Prefix A was reused unchanged.
    for tag in ("C", "D"):
        meta = json.loads((tmp_path / "fast" / "datasets" / tag / "meta.json").read_text())
        assert meta["signature_source"] == "artifact"

    # B's influence on C was reversed: shared survives now; in live it was
    # exact-removed. Near remains removed, but changes cross -> within.
    fast_c = out_frames(tmp_path / "fast" / "outputs", "C")
    live_c = out_frames(tmp_path / "liveout", "C")
    assert any(shared in fr.get_column("text").to_list() for fr in fast_c.values())
    assert not any(shared in fr.get_column("text").to_list() for fr in live_c.values())
    fast_reasons = pl.read_parquet(tmp_path / "fast" / "removals" / "C.parquet")
    assert "within" in fast_reasons.get_column("reason").to_list()


def test_faithful_suffix_replay_falls_back_without_row_artifact(tmp_path):
    """Legacy releases remain correct: absent compact rows, only the suffix
    falls back to the normal text-to-signature path."""
    from puffer.withdraw import rebuild_survivors_suffix

    rng = random.Random(17)
    shared = _make_doc(rng)
    docs = {
        "A": [_make_doc(rng) for _ in range(10)],
        "B": [_make_doc(rng) for _ in range(10)] + [shared],
        "C": [_make_doc(rng) for _ in range(10)] + [shared],
    }
    for name, rows in docs.items():
        _write_release(tmp_path / "in" / name, pl.DataFrame({"text": rows}))
    cfg = _cfg(record_row_signatures=False)
    live = Deduper(tmp_path / "live", cfg)
    for name in ("A", "B", "C"):
        live.ingest([str(tmp_path / "in" / name / "*.parquet")], name,
                    tmp_path / "liveout" / name)

    rebuild_survivors_suffix(tmp_path / "live", "B", tmp_path / "fast", cfg)
    meta = json.loads((tmp_path / "fast" / "datasets" / "C" / "meta.json").read_text())
    assert meta["signature_source"] == "text"
    outputs = [
        text
        for path in (tmp_path / "fast" / "outputs" / "C").glob("*.parquet")
        for text in pl.read_parquet(path).get_column("text").to_list()
    ]
    assert shared in outputs


def test_faithful_suffix_replay_preflights_corrupt_artifact(tmp_path):
    """A malformed compact suffix artifact fails before destination creation."""
    import pytest

    from puffer.withdraw import rebuild_survivors_suffix

    rng = random.Random(19)
    for name in ("A", "B", "C"):
        _write_release(
            tmp_path / "in" / name,
            pl.DataFrame({"text": [_make_doc(rng) for _ in range(8)]}),
        )
    cfg = _cfg(record_row_signatures=True)
    live = Deduper(tmp_path / "live", cfg)
    for name in ("A", "B", "C"):
        live.ingest([str(tmp_path / "in" / name / "*.parquet")], name,
                    tmp_path / "liveout" / name)

    artifact = tmp_path / "live" / "datasets" / "C" / "row_band_keys.i64"
    artifact.write_bytes(artifact.read_bytes()[:-8])
    dest = tmp_path / "fast"
    with pytest.raises(ValueError, match="row-signature artifact"):
        rebuild_survivors_suffix(tmp_path / "live", "B", dest, cfg)
    assert not dest.exists()


def test_faithful_rebuild_refuses_on_input_drift(tmp_path):
    """Replay must refuse if a survivor's input bytes changed since ingest
    (content digest mismatch) -- faithful replay runs only against recorded
    content."""
    import pytest

    from puffer.withdraw import rebuild_survivors

    rng = random.Random(3)
    for name in ("A", "B"):
        _write_release(tmp_path / "in" / name, pl.DataFrame({"text": [_make_doc(rng) for _ in range(20)]}))
    cfg = _cfg()
    dd = Deduper(tmp_path / "live", cfg)
    for name in ("A", "B"):
        dd.ingest([str(tmp_path / "in" / name / "*.parquet")], dataset=name,
                  output_dir=str(tmp_path / "liveout" / name))

    # Mutate a surviving dataset's (A) input after ingest.
    a_file = next((tmp_path / "in" / "A").glob("*.parquet"))
    pl.DataFrame({"text": ["totally different content"]}).write_parquet(a_file)

    with pytest.raises(RuntimeError, match="drift"):
        rebuild_survivors(tmp_path / "live", "B", tmp_path / "oracle", cfg)


def test_faithful_rebuild_rejects_divergent_config(tmp_path):
    """A cfg that changes dedup semantics (immutable field) must be rejected --
    the source state's persisted config is authoritative for a faithful rebuild."""
    import pytest

    from puffer.withdraw import rebuild_survivors

    rng = random.Random(5)
    for name in ("A", "B"):
        _write_release(tmp_path / "in" / name, pl.DataFrame({"text": [_make_doc(rng) for _ in range(15)]}))
    dd = Deduper(tmp_path / "live", _cfg())  # num_bands=8
    for name in ("A", "B"):
        dd.ingest([str(tmp_path / "in" / name / "*.parquet")], dataset=name,
                  output_dir=str(tmp_path / "out" / name))

    with pytest.raises(ValueError, match="num_bands|immutable"):
        rebuild_survivors(tmp_path / "live", "B", tmp_path / "oracle", _cfg(num_bands=4))
