"""The streaming ingest bounds resident text to ``cfg.sig_chunk_rows``.

Rather than assert a flaky peak-RSS number, this instruments the per-document
signature/sha calls and asserts no single call ever receives more than
``sig_chunk_rows`` documents -- i.e. release text is processed in bounded
chunks regardless of release size. Also checks the ingest temp dir is cleaned.
"""

from __future__ import annotations

import random
import string

import polars as pl

from puffer import Deduper, PufferConfig


def _docs(seed, n):
    rng = random.Random(seed)
    pool = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9))) for _ in range(3000)]
    return [" ".join(rng.choices(pool, k=90)) for _ in range(n)]


def _write(path, docs):
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"text": docs}).write_parquet(path)


def test_ingest_text_is_chunked_to_sig_chunk_rows(tmp_path, monkeypatch):
    import concurrent.futures as cf

    chunk = 1000
    # two files, each far larger than one chunk
    _write(tmp_path / "in" / "a.parquet", _docs(1, 2500))
    _write(tmp_path / "in" / "b.parquet", _docs(2, 2500))

    # Spy on the pool: each submit gets one chunk's texts as its first arg, so
    # the max submitted text-list length is the resident-per-task bound.
    max_texts = {"n": 0}
    real_ppe = cf.ProcessPoolExecutor

    class SpyPool(real_ppe):
        def submit(self, fn, *args, **kwargs):
            if args and hasattr(args[0], "__len__"):
                max_texts["n"] = max(max_texts["n"], len(args[0]))
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(cf, "ProcessPoolExecutor", SpyPool)

    state = tmp_path / "state"
    dd = Deduper(state, PufferConfig(sig_chunk_rows=chunk))
    report = dd.ingest([str(tmp_path / "in" / "*.parquet")], dataset="A",
                       output_dir=str(tmp_path / "out"))

    assert report.n_input == 5000
    # No chunk task ever received more than one chunk of documents.
    assert 0 < max_texts["n"] <= chunk
    # ingest temp dirs are cleaned up (no .ingest_* left behind).
    assert not list(state.glob(".ingest_*"))


def test_chunk_boundary_does_not_change_decisions(tmp_path):
    """Same release, two chunk sizes -> identical removal counts and output."""
    docs = _docs(5, 1500)
    docs += [docs[3], docs[7]]  # identical dups spanning chunk boundaries
    _write(tmp_path / "in" / "r.parquet", docs)

    def run(chunk):
        st = tmp_path / f"st{chunk}"
        dd = Deduper(st, PufferConfig(sig_chunk_rows=chunk))
        r = dd.ingest([str(tmp_path / "in" / "r.parquet")], dataset="A",
                      output_dir=str(tmp_path / f"out{chunk}"))
        out = sorted(pl.read_parquet(tmp_path / f"out{chunk}" / "r.parquet")["text"].to_list())
        return (r.n_within_removed, r.n_cross_removed, r.n_output), out

    small, out_small = run(200)
    big, out_big = run(100000)
    assert small == big
    assert out_small == out_big


def test_local_ingest_creates_at_most_one_process_pool(tmp_path, monkeypatch):
    """Guards the per-chunk-pool regression: the streaming local ingest must
    reuse a single ProcessPoolExecutor, not construct one per chunk."""
    import concurrent.futures as cf

    _write(tmp_path / "in" / "a.parquet", _docs(1, 6000))  # many chunks at chunk=1000
    count = {"n": 0}
    real_ppe = cf.ProcessPoolExecutor

    def counting_ppe(*a, **k):
        count["n"] += 1
        return real_ppe(*a, **k)

    monkeypatch.setattr(cf, "ProcessPoolExecutor", counting_ppe)
    dd = Deduper(tmp_path / "state", PufferConfig(sig_chunk_rows=1000))
    dd.ingest([str(tmp_path / "in" / "a.parquet")], dataset="A",
              output_dir=str(tmp_path / "out"))
    assert count["n"] <= 1
