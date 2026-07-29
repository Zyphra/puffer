"""``Deduper``: the end-to-end PUFFER ingest/withdraw orchestrator.

One release (a set of parquet files sharing a dataset tag) goes through five
phases on ``ingest``:

  1. Read every input parquet (Polars), extract the text column, and compute
     each document's ``num_bands`` MinHash-LSH band keys
     (:mod:`puffer.signature`).
  2. Within-release fuzzy stage: for each band, group survivors by band key;
     any group with >=2 members is a same-release near-dup candidate. Each
     multi-member group is reduced keep-first (lexicographically smallest
     ``(source_file, source_row)`` survives); the union of flags across all
     bands is this stage's removal set (``"within"``).
  3. Cross-history stage: every within-release survivor is screened
     (:mod:`puffer.screen`, early-stop) against the LSM band index built by
     every *other* previously ingested release (``exclude_tag=dataset``, so
     a retried ingest never collides with its own not-yet-superseded shard).
     A hit flags ``"cross"``.
  4. Commit: for each band, the FULL release's unique band keys — including
     rows removed in stages 2-3 — are appended as one new level-0 shard
     (:func:`puffer.index.append_shard`, commit-once: idempotent by tag, one
     shard per dataset). This is deliberate: a later release's near-dup of a
     row THIS release dropped as a fuzzy duplicate must still be caught
     later; index membership records every presented (not merely output)
     document. Each band's array is also copied byte-for-byte
     to a per-dataset sidecar (``state/datasets/<tag>/band_XX.bin``) — the
     only ingredient :mod:`puffer.withdraw` needs to rebuild a merged run
     after this dataset is withdrawn, without ever re-reading or re-hashing
     text. The band is then compacted (protecting this dataset's own shard
     from the same compaction call).
  5. Write one output parquet per input file (same schema, flagged rows
     dropped), the release's removals parquet, its dataset metadata, and one
     ledger event.

Withdrawal (:func:`puffer.withdraw.withdraw_dataset`) is INDEX-STATE removal:
it restores the band index to the state it
would be in had the withdrawn dataset never been ingested. It does **not**
retroactively re-derive any other, already-written release's output
parquet — a document some earlier release dropped because it collided with
the withdrawn dataset stays dropped in that release's output. Only the
index (and hence future ingests) forgets the withdrawn dataset existed.
``Deduper.withdraw`` never implicitly deletes a dataset's already-written
output files, either: by default it only removes the index contribution and
state artifacts, reporting the surviving output files as
``WithdrawReport.outputs_retained``. Pass ``purge_outputs=True`` to also
unlink exactly those recorded files (and the now-empty output directory, if
any) — never a blind directory removal, since other datasets' files may
share the same ``output_dir``.

Idempotent retries: a full, already-completed ingest of ``dataset`` (its
``state/datasets/<tag>/meta.json`` marker exists) short-circuits into a pure
replay — the persisted removals manifest is re-read to reproduce the same
``IngestReport`` counts and it touches none of the LSM index,
or output files a second time, so re-running is trivially byte-identical and
never risks a false self-collision. A crash *mid*-ingest relies on the
underlying primitives' own idempotency (``append_shard``/``compact_band``
overwrite by tag); only a *fully completed* ingest is guaranteed replay-safe
by this short-circuit.
"""

from __future__ import annotations

import contextlib
import glob as _glob_mod
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from puffer.config import PufferConfig

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


def _peak_rss_gib() -> float:
    """Best-effort process peak RSS in GiB (Unix); 0.0 if unavailable.

    ``ru_maxrss`` is KiB on Linux, bytes on macOS -- normalize both.
    """
    try:
        import resource
        import sys

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1 if sys.platform == "darwin" else 1024
        return maxrss * scale / (1024 ** 3)
    except Exception:  # noqa: BLE001 -- logging aid, never fatal
        return 0.0


def _git_provenance() -> "str | None":
    """Best-effort ``branch@shortsha`` of the installed puffer checkout, for
    run-provenance logging (mirrors common git branch/commit logging patterns). Returns
    None outside a git checkout."""
    import subprocess

    root = Path(__file__).resolve().parents[2]

    def _git(args: list[str]) -> "str | None":
        try:
            out = subprocess.run(
                ["git", *args], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
            return out or None
        except (OSError, subprocess.CalledProcessError):
            return None

    sha = _git(["rev-parse", "--short", "HEAD"])
    if not sha:
        return None
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]) or "detached"
    return f"{branch}@{sha}"

_VALID_EXECUTORS = ("local", "ray")


@dataclass
class IngestReport:
    dataset: str
    n_input: int
    n_within_removed: int
    n_cross_removed: int
    n_output: int
    output_files: list[str]
    probes_done: int
    probes_scheduled: int
    elapsed_s: float
    ray_transport: str = "local"  # local | object_store (ray)


@dataclass
class WithdrawReport:
    dataset: str
    bands_o1: int
    bands_rebuilt: int
    outputs_retained: list[str]
    outputs_purged: bool
    elapsed_s: float


# ---------------------------------------------------------------------------
# Small persistence helpers (ledger, atomic parquet/json writes)
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _write_parquet_atomic(frame: "pl.DataFrame", path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.tmp_")
    os.close(fd)
    try:
        frame.write_parquet(tmp)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _write_parquet_filtered_stream(src: Path, dst: Path, removed_slice, batch_rows: int) -> None:
    """Copy ``src`` to ``dst`` dropping rows flagged in ``removed_slice`` (a bool
    array over the file's rows), streaming one row-batch at a time so peak RAM is
    O(batch_rows) regardless of file size. Preserves the full input schema.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(str(src))
    removed = np.asarray(removed_slice, dtype=bool)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.tmp_")
    os.close(fd)
    writer = None
    pos = 0
    try:
        writer = pq.ParquetWriter(tmp, pf.schema_arrow)
        for batch in pf.iter_batches(batch_size=batch_rows):
            m = batch.num_rows
            keep = ~removed[pos:pos + m]
            pos += m
            table = pa.Table.from_batches([batch])
            writer.write_table(table.filter(pa.array(keep)))
        writer.close()
        writer = None
        os.replace(tmp, dst)
    except BaseException:
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.close()
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _ledger_path(state_dir: Path) -> Path:
    return Path(state_dir) / "ledger.json"


def _read_ledger(state_dir: Path) -> list[dict]:
    path = _ledger_path(state_dir)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _append_ledger_event(state_dir: Path, event: dict) -> None:
    path = _ledger_path(state_dir)
    events = _read_ledger(state_dir)
    events.append(event)
    _write_json_atomic(path, events)



def _resolve_inputs(inputs: list) -> list[Path]:
    """Expand file paths and glob patterns into a deduplicated, ordered file list."""
    out: list[str] = []
    seen: set[str] = set()
    for pattern in inputs:
        pattern = str(pattern)
        matches = sorted(_glob_mod.glob(pattern))
        if not matches and Path(pattern).exists():
            matches = [pattern]
        for m in matches:
            rp = str(Path(m).resolve())
            if rp not in seen:
                seen.add(rp)
                out.append(rp)
    return [Path(p) for p in out]


# ---------------------------------------------------------------------------
# Keep-first reduction
# ---------------------------------------------------------------------------


def _keep_first_dup_map(groups: "pl.DataFrame") -> tuple[dict[tuple[str, int], tuple[str, int]], int]:
    """Per-band keep-first reduction over candidate collision groups.

    Within each candidate group the lexicographically smallest
    ``(source_file, source_row)`` survives and every other member is
    flagged. ``groups`` has one row per candidate group (>=2 members):
    ``members_file: list[str]``, ``members_row: list[i64]``. Returns
    ``(dup_map, n_groups)`` where ``dup_map`` maps every flagged
    ``(source_file, source_row)`` to its surviving canonical key.
    """
    import polars as pl

    if groups is None or groups.is_empty():
        return {}, 0

    ranked = (
        groups.with_row_index("_gid")
        .explode(["members_file", "members_row"])
        .rename({"members_file": "source_file", "members_row": "source_row"})
        .sort(["_gid", "source_file", "source_row"])
        .with_columns(pl.int_range(pl.len()).over("_gid").alias("_rn"))
    )
    n_groups = ranked.get_column("_gid").n_unique()

    canon = ranked.filter(pl.col("_rn") == 0).select(
        "_gid", pl.col("source_file").alias("_cf"), pl.col("source_row").alias("_cr"),
    )
    removed = (
        ranked.filter(pl.col("_rn") > 0)
        .join(canon, on="_gid")
        .select("source_file", "source_row", "_cf", "_cr")
    )

    dup_map: dict[tuple[str, int], tuple[str, int]] = {}
    for sf, sr, cf, cr in removed.iter_rows():
        key = (sf, sr)
        if key not in dup_map:
            dup_map[key] = (cf, cr)
    return dup_map, n_groups


# ---------------------------------------------------------------------------
# Deduper
# ---------------------------------------------------------------------------


class Deduper:
    """Stateful orchestrator over one ``state_dir``. See module docstring."""

    def __init__(
        self,
        state_dir: "str | Path",
        config: PufferConfig | None = None,
        executor: str = "local",
    ):
        if executor not in _VALID_EXECUTORS:
            raise ValueError(f"executor must be one of {_VALID_EXECUTORS}, got {executor!r}")
        # Resolve to absolute so paths derived for Ray workers (index_dir, input
        # files) are valid regardless of a worker's CWD (workers do not inherit
        # the driver's CWD; a relative index_dir silently reads an empty index).
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.executor = executor

        persisted = PufferConfig.load(self.state_dir)
        if config is None:
            self.cfg = persisted if persisted is not None else PufferConfig()
        else:
            if persisted is not None:
                config.validate_against(persisted)
            self.cfg = config

    # -- public API -----------------------------------------------------

    def ingest(
        self,
        inputs: list,
        dataset: str,
        output_dir: "str | Path",
        text_column: str | None = None,
    ) -> IngestReport:
        """Ingest one release. See the module docstring for the six phases."""
        from puffer.index import sanitize_tag

        start = time.monotonic()
        self._ensure_config_persisted()

        safe_tag = sanitize_tag(dataset)
        output_dir = Path(output_dir)
        paths = _resolve_inputs(inputs)
        column = text_column or self.cfg.text_column
        meta_path = self.state_dir / "datasets" / safe_tag / "meta.json"

        if meta_path.exists():
            logger.info("dataset %r already ingested; replaying persisted result", dataset)
            return self._replay_ingest(paths, dataset, output_dir, start)

        return self._ingest_fresh(paths, dataset, safe_tag, output_dir, column, start)

    def withdraw(self, dataset: str, purge_outputs: bool = False) -> WithdrawReport:
        """Withdraw ``dataset`` from the index. See module docstring for semantics.

        Default (``purge_outputs=False``) removes only this dataset's index
        contribution and state artifacts (sidecars, removals manifest) — the
        output parquets it already wrote under its ``output_dir`` are left
        untouched and reported back as ``outputs_retained``, since withdrawal
        does not imply the caller wants already-served files deleted. Pass
        ``purge_outputs=True`` to additionally unlink exactly the files this
        dataset's ingest recorded (never a blind directory removal — other
        datasets' files sharing the same ``output_dir`` are untouched) and,
        if that leaves the directory empty, remove the now-empty directory.
        """
        from puffer.index import sanitize_tag
        from puffer.withdraw import withdraw_dataset

        start = time.monotonic()
        self._ensure_config_persisted()
        safe_tag = sanitize_tag(dataset)

        meta_path = self.state_dir / "datasets" / safe_tag / "meta.json"
        prior_n_docs = 0
        output_files: list[str] = []
        output_dir_str: str | None = None
        if meta_path.exists():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                meta = json.loads(meta_path.read_text())
                prior_n_docs = int(meta.get("n_docs", 0))
                output_files = list(meta.get("output_files", []))
                output_dir_str = meta.get("output_dir")

        plan = withdraw_dataset(self.state_dir, dataset, self.cfg)


        outputs_purged = False
        if purge_outputs and output_files:
            for f in output_files:
                with contextlib.suppress(OSError):
                    Path(f).unlink(missing_ok=True)
            outputs_purged = True
            if output_dir_str:
                out_dir = Path(output_dir_str)
                with contextlib.suppress(OSError):
                    if out_dir.is_dir() and not any(out_dir.iterdir()):
                        out_dir.rmdir()

        _append_ledger_event(
            self.state_dir,
            {
                "op": "withdraw",
                "dataset": dataset,
                "utc": _utcnow_iso(),
                "n_docs": prior_n_docs,
                "output_dir": None,
                "purged": outputs_purged,
            },
        )

        return WithdrawReport(
            dataset=dataset,
            bands_o1=int(plan["bands_o1"]),
            bands_rebuilt=int(plan["bands_rebuilt"]),
            outputs_retained=[] if outputs_purged else output_files,
            outputs_purged=outputs_purged,
            elapsed_s=time.monotonic() - start,
        )

    def datasets(self) -> list[str]:
        """Currently active dataset tags (ingested and not withdrawn), in ingest order."""
        from puffer.index import sanitize_tag

        order: list[str] = []
        active: set[str] = set()
        original: dict[str, str] = {}
        for event in _read_ledger(self.state_dir):
            tag = sanitize_tag(event["dataset"])
            if event.get("op") == "ingest":
                if tag not in active:
                    order.append(tag)
                active.add(tag)
                original[tag] = event["dataset"]
            elif event.get("op") == "withdraw":
                active.discard(tag)
        return [original[tag] for tag in order if tag in active]

    def stats(self) -> dict:
        """Summary of index + dataset state (band shard counts, active datasets, config)."""
        from puffer.index import load_manifest

        index_dir = self.state_dir / "hash_index"
        bands = []
        for band_id in range(self.cfg.num_bands):
            manifest = load_manifest(index_dir, band_id)
            shards = manifest.get("shards", [])
            bands.append({
                "band": band_id,
                "n_shards": len(shards),
                "total_keys": sum(int(s.get("count", 0)) for s in shards),
                "compaction_policy": manifest.get("compaction_policy"),
            })
        return {
            "state_dir": str(self.state_dir),
            "datasets": self.datasets(),
            "bands": bands,
            "config": self.cfg.to_dict(),
        }

    # -- ingest internals -------------------------------------------------

    def _ensure_config_persisted(self) -> None:
        persisted = PufferConfig.load(self.state_dir)
        if persisted is None:
            self.cfg.save(self.state_dir)
        else:
            self.cfg.validate_against(persisted)

    def _replay_ingest(
        self, paths: list[Path], dataset: str, output_dir: Path, start: float,
    ) -> IngestReport:
        import polars as pl

        from puffer.index import sanitize_tag

        safe_tag = sanitize_tag(dataset)
        meta = json.loads((self.state_dir / "datasets" / safe_tag / "meta.json").read_text())
        removals_path = self.state_dir / "removals" / f"{safe_tag}.parquet"
        reasons: list[str] = []
        if removals_path.exists():
            reasons = pl.read_parquet(removals_path).get_column("reason").to_list()

        n_input = int(meta["n_docs"])
        n_within = reasons.count("within")
        n_cross = reasons.count("cross")
        output_files = list(meta.get("output_files") or [str(Path(meta["output_dir"]) / p.name) for p in paths])

        return IngestReport(
            dataset=dataset,
            n_input=n_input,
            n_within_removed=n_within,
            n_cross_removed=n_cross,
            n_output=n_input - len(reasons),
            output_files=output_files,
            probes_done=0,
            probes_scheduled=0,
            elapsed_s=time.monotonic() - start,
        )

    def _ingest_fresh(
        self,
        paths: list[Path],
        dataset: str,
        safe_tag: str,
        output_dir: Path,
        column: str,
        start: float,
        replay_artifacts_dir: Path | None = None,
        replay_file_ids: list[int] | None = None,
    ) -> IngestReport:
        """Bounded-memory streaming ingest.

        Release text is never fully resident: it is streamed in row-chunks of
        ``cfg.sig_chunk_rows`` (read -> sha + band keys -> exact add -> discard
        text). Per-doc band keys land in a disk-backed ``np.memmap`` so resident
        RAM is O(chunk x workers) + compact per-doc bytes, independent of release
        size. Within-release keep-first, cross screen, commit and output all run
        off the memmap in bounded batches. Decisions match a whole-release
        implementation: signatures/sha are per-document, the exact Bloom is fed
        in file/presentation order (ordered even under the bounded Ray producer),
        and the within-release canonical is the lexicographically smallest
        ``(source_file, source_row)`` via a per-band ``lexsort`` on
        ``(band_key, file_rank, source_row)``.
        """
        import shutil

        import numpy as np
        import polars as pl
        import pyarrow.parquet as pq

        from puffer.index import append_shard, compact_band, write_shard_bin

        num_bands = self.cfg.num_bands
        index_dir = self.state_dir / "hash_index"
        chunk_rows = self.cfg.sig_chunk_rows
        # Replay manifest: normal ingest interns resolved inputs; faithful replay
        # can reuse ids from its copied content-addressed path table after it has
        # verified those ids against the source, avoiding a second full-file hash.
        file_ids: list[int] | None = None
        if self.cfg.record_replay_manifest:
            if replay_file_ids is not None:
                if len(replay_file_ids) != len(paths):
                    raise ValueError("replay_file_ids must align one-for-one with input paths")
                file_ids = [int(i) for i in replay_file_ids]
            else:
                from puffer import paths as _paths
                file_ids = _paths.intern_paths(self.state_dir, [str(p) for p in paths])

        # -- Pass 0: footer-only row counts, file bounds, lexicographic ranks --
        # The distributed path also collects per-row-group geometry here (the
        # footer is already parsed) so chunks can be planned without a second
        # metadata pass. Suffix replay feeds the local memmap path, so it is
        # excluded from distribution.
        want_dist = self.executor == "ray" and replay_artifacts_dir is None
        n_rows: list[int] = []
        rg_rows: list[list[int]] = []
        rg_bytes: list[list[int]] = []
        for p in paths:
            pf = pq.ParquetFile(str(p))
            if column not in pf.schema_arrow.names:
                raise ValueError(f"text column {column!r} is absent from {p}")
            md = pf.metadata
            n_rows.append(md.num_rows)
            if want_dist:
                rg_rows.append([md.row_group(i).num_rows for i in range(md.num_row_groups)])
                rg_bytes.append([md.row_group(i).total_byte_size for i in range(md.num_row_groups)])
        file_bounds: list[tuple[int, int]] = []
        off = 0
        for n in n_rows:
            file_bounds.append((off, off + n))
            off += n
        n_total = off
        starts = np.array([a for (a, _b) in file_bounds] or [0], dtype=np.int64)

        # lexicographic file rank (by str(path)) drives the within-release canonical
        lex = sorted(range(len(paths)), key=lambda i: str(paths[i]))
        file_rank = [0] * len(paths)
        for rank, i in enumerate(lex):
            file_rank[i] = rank
        file_rank_arr = np.array(file_rank or [0], dtype=np.int64)

        within_flag = np.zeros(n_total, dtype=bool)
        cross_flag = np.zeros(n_total, dtype=bool)
        ray_transport = "local"
        signature_source = "text"
        # Distributed path (row-group chunk tasks spool signatures
        # to shared FS, one reduce task per band, chunked output writes).
        use_dist = want_dist and n_total > 0
        _path_label = (
            "distributed-spool" if use_dist
            else ("ray-object-store" if self.executor == "ray" else "local")
        )
        _prov = _git_provenance()
        logger.info(
            "[%s] ingest begin: %d file(s), %d doc(s), executor=%s, path=%s%s",
            dataset, len(paths), n_total, self.executor, _path_label,
            f", puffer={_prov}" if _prov else "",
        )
        _phase_t = {"last": time.monotonic()}

        def _phase(msg: str) -> None:
            now = time.monotonic()
            logger.info(
                "[%s] %s (+%.1fs, %.1fs total, peak RSS %.1f GiB)",
                dataset, msg, now - _phase_t["last"], now - start, _peak_rss_gib(),
            )
            _phase_t["last"] = now

        tmp_dir = Path(tempfile.mkdtemp(prefix=f".ingest_{safe_tag}_", dir=self.state_dir))
        band_keys = None
        try:
            if n_total and not use_dist:
                band_keys = np.memmap(
                    tmp_dir / "band_keys.i64", dtype=np.int64, mode="w+",
                    shape=(n_total, num_bands),
                )
            elif not use_dist:
                band_keys = np.empty((0, num_bands), dtype=np.int64)


            # -- Pass 1: row-ordered band keys in file order ----------------
            replay_keys = (
                Path(replay_artifacts_dir) / "row_band_keys.i64"
                if replay_artifacts_dir is not None else None
            )
            if replay_keys is not None and replay_keys.exists():
                expected = n_total * num_bands * np.dtype(np.int64).itemsize
                actual = replay_keys.stat().st_size
                if actual != expected:
                    raise ValueError(
                        f"row-signature artifact {replay_keys} has {actual} bytes; "
                        f"expected {expected} for {n_total} rows x {num_bands} bands"
                    )
                signature_source = "artifact"
                if n_total:
                    source_keys = np.memmap(
                        replay_keys, dtype=np.int64, mode="r",
                        shape=(n_total, num_bands),
                    )
                    for lo in range(0, n_total, chunk_rows):
                        hi = min(n_total, lo + chunk_rows)
                        band_keys[lo:hi, :] = source_keys[lo:hi, :]
                    del source_keys
            elif use_dist:
                pass  # signatures are spooled by the distributed stages below
            elif self.executor == "ray" and n_total:
                # Ray transport is the object-store return path: workers stream
                # their file's text (bounded) and return compact band keys/sha.
                from puffer import ray_exec

                ray_transport = "object_store"
                for fi, res in ray_exec.ray_iter_band_keys(
                    paths, self.cfg, text_column=column,
                    max_in_flight=self.cfg.ray_max_in_flight,
                ):
                    a, b = file_bounds[fi]
                    if b > a:
                        band_keys[a:b, :] = res["band_keys"]
                    res["band_keys"] = None
            elif n_total:
                from collections import deque
                from concurrent.futures import ProcessPoolExecutor

                from puffer.signature import chunk_signature

                # One persistent pool for the whole ingest (never per chunk).
                # In-flight chunks are capped by ram_budget_bytes (not worker
                # count) so resident text stays bounded regardless of cores:
                # ~= cap x sig_chunk_rows documents.
                est_row_bytes = 8192
                cap = self.cfg.ram_budget_bytes // max(1, chunk_rows * est_row_bytes)
                inflight = max(1, min(self.cfg.effective_workers, int(cap), 64))
                workers = max(1, min(self.cfg.effective_workers, inflight))

                def _chunk_stream():
                    for fi, path in enumerate(paths):
                        pos = file_bounds[fi][0]
                        for batch in pq.ParquetFile(str(path)).iter_batches(
                            batch_size=chunk_rows, columns=[column]
                        ):
                            texts = ["" if v is None else str(v)
                                     for v in batch.column(0).to_pylist()]
                            m = len(texts)
                            if m:
                                yield pos, m, texts
                            pos += m

                gen = _chunk_stream()
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    pending: deque = deque()
                    for _ in range(inflight):
                        try:
                            pos, m, texts = next(gen)
                        except StopIteration:
                            break
                        pending.append((pos, m, pool.submit(chunk_signature, texts, self.cfg)))
                        del texts
                    while pending:
                        pos, m, fut = pending.popleft()
                        bk = fut.result()
                        band_keys[pos:pos + m, :] = bk
                        try:
                            npos, nm, ntexts = next(gen)
                            pending.append(
                                (npos, nm, pool.submit(chunk_signature, ntexts, self.cfg))
                            )
                            del ntexts
                        except StopIteration:
                            pass

            if n_total and not use_dist:
                band_keys.flush()

            if self.cfg.record_row_signatures and not use_dist:
                # The ingest already paid to materialize this matrix; retaining
                # its raw C-order bytes makes later suffix replay skip shingling
                # and MinHash without changing any decision path.
                artifact = self.state_dir / "datasets" / safe_tag / "row_band_keys.i64"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                afd, artifact_tmp = tempfile.mkstemp(
                    dir=artifact.parent, prefix=".row_band_keys_tmp_"
                )
                try:
                    with os.fdopen(afd, "wb") as out:
                        for lo in range(0, n_total, chunk_rows):
                            hi = min(n_total, lo + chunk_rows)
                            out.write(np.ascontiguousarray(band_keys[lo:hi, :]).tobytes())
                        out.flush()
                        os.fsync(out.fileno())
                    os.replace(artifact_tmp, artifact)
                    artifact_tmp = None
                finally:
                    if artifact_tmp is not None and os.path.exists(artifact_tmp):
                        os.unlink(artifact_tmp)

            if not use_dist:
                _phase(f"signatures computed ({n_total} doc(s), source={signature_source})")
            # -- Phase 3: within-release keep-first (bounded, per band) ------
            active1 = np.ones(n_total, dtype=bool)
            if not use_dist and n_total and active1.any():
                active_idx = np.nonzero(active1)[0]
                fidx = np.searchsorted(starts, active_idx, side="right") - 1
                rw = active_idx - starts[fidx]
                fr = file_rank_arr[fidx]
                for band_id in range(num_bands):
                    kb = np.asarray(band_keys[active_idx, band_id])
                    ob = np.lexsort((rw, fr, kb))
                    ks = kb[ob]
                    if ks.size == 0:
                        continue
                    new_run = np.ones(ks.size, dtype=bool)
                    np.not_equal(ks[1:], ks[:-1], out=new_run[1:])
                    run_id = np.cumsum(new_run) - 1
                    counts = np.bincount(run_id)
                    flagged = (~new_run) & (counts[run_id] >= 2)
                    within_flag[active_idx[ob[flagged]]] = True

            # -- Phase 4: cross-history screen (bounded, chunked) ------------
            active2 = active1 & ~within_flag
            if not use_dist:
                _phase(f"within-release keep-first: {int(within_flag.sum())} flagged")
            probes_done = 0
            probes_scheduled = 0
            if not use_dist and n_total and active2.any():
                active_idx2 = np.nonzero(active2)[0]
                ray_exec = None
                if self.executor == "ray":
                    from puffer import ray_exec as _rx
                    ray_exec = _rx
                else:
                    from puffer.screen import screen_release
                screen_batch = self.cfg.screen_batch_rows
                for cs in range(0, active_idx2.size, screen_batch):
                    cidx = active_idx2[cs:cs + screen_batch]
                    sub = np.asarray(band_keys[cidx])
                    cc: dict = {}
                    if ray_exec is not None:
                        mm, cc = ray_exec.ray_screen_chunks(sub, index_dir, dataset, self.cfg)
                        mm = np.asarray(mm, dtype=bool)
                    else:
                        mm = np.asarray(
                            screen_release(sub, index_dir, dataset, self.cfg, cc), dtype=bool
                        )
                    cross_flag[cidx[mm]] = True
                    probes_done += int(cc.get("probes_done", 0))
                    probes_scheduled += int(cc.get("probes_scheduled", 0))

            if not use_dist and n_total:
                _phase(
                    f"cross-history screen: {int(cross_flag.sum())} flagged, "
                    f"{probes_done}/{probes_scheduled} probes"
                )
            # -- Distributed stages: chunk plan -> spool -> band reduce --
            if use_dist:
                from puffer import ray_dist

                ray_transport = "spool"
                chunk_plan = ray_dist.plan_release_chunks(
                    rg_rows, rg_bytes, n_total, self.cfg,
                )
                _phase(f"planned {len(chunk_plan)} row-group chunk(s) over {len(paths)} file(s)")
                metas = ray_dist.ray_spool_band_keys(
                    paths, chunk_plan, self.cfg, tmp_dir, text_column=column,
                )
                spooled = [0] * len(paths)
                for m in metas:
                    spooled[int(m["file_index"])] += int(m["n_rows"])
                if spooled != n_rows:
                    raise RuntimeError(
                        f"spooled row counts {sum(spooled)} diverge from "
                        f"parquet footers {n_total}"
                    )
                _phase(f"spooled {len(metas)} chunk(s) ({n_total} doc(s))")
                band_results = ray_dist.ray_band_reduce(
                    metas, file_rank, index_dir, dataset, self.cfg, tmp_dir,
                )
                for res in band_results:
                    within_flag[res["within_rows"]] = True
                for res in band_results:
                    cross_flag[res["cross_rows"]] = True
                    probes_scheduled += int(res["probes_scheduled"])
                    probes_done += int(res["probes_done"])
                # The local path never cross-probes within-removed rows;
                # removal priority is within > cross either way.
                cross_flag &= ~within_flag
                _phase(
                    f"band reduce: within={int(within_flag.sum())} "
                    f"cross={int(cross_flag.sum())}, {probes_done} probes"
                )
                for band_id in range(num_bands):
                    uniq = np.fromfile(band_results[band_id]["uniq_path"], dtype=np.int64)
                    append_shard(uniq, index_dir, band_id, dataset)
                    write_shard_bin(
                        uniq, self.state_dir / "datasets" / safe_tag / f"band_{band_id:02d}.bin"
                    )
                    compact_band(
                        index_dir, band_id, self.cfg.tier_fanout, self.cfg.ram_budget_bytes,
                        protect_tag=dataset,
                    )
                _phase(f"committed {num_bands} band(s) to index")
                if self.cfg.record_row_signatures:
                    # Assemble the row-major artifact from the band-major spool.
                    artifact = self.state_dir / "datasets" / safe_tag / "row_band_keys.i64"
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    afd, artifact_tmp = tempfile.mkstemp(
                        dir=artifact.parent, prefix=".row_band_keys_tmp_"
                    )
                    try:
                        with os.fdopen(afd, "wb") as out:
                            for m in metas:
                                nrf = int(m["n_rows"])
                                if not nrf:
                                    continue
                                bm = np.fromfile(m["path"], dtype=np.int64).reshape(num_bands, nrf)
                                out.write(np.ascontiguousarray(bm.T).tobytes())
                            out.flush()
                            os.fsync(out.fileno())
                        os.replace(artifact_tmp, artifact)
                        artifact_tmp = None
                    finally:
                        if artifact_tmp is not None and os.path.exists(artifact_tmp):
                            os.unlink(artifact_tmp)

            # -- Phase 5: commit full release band keys to the index ---------
            if not use_dist:
                for band_id in range(num_bands):
                    if n_total:
                        unique_keys = np.unique(np.asarray(band_keys[:, band_id]))
                    else:
                        unique_keys = np.empty(0, dtype=np.int64)
                    append_shard(unique_keys, index_dir, band_id, dataset)
                    write_shard_bin(
                        unique_keys, self.state_dir / "datasets" / safe_tag / f"band_{band_id:02d}.bin"
                    )
                    compact_band(
                        index_dir, band_id, self.cfg.tier_fanout, self.cfg.ram_budget_bytes,
                        protect_tag=dataset,
                    )

            if not use_dist and n_total:
                _phase(f"committed {num_bands} band(s) to index")
            # -- Phase 6: outputs (streamed per file), removals, meta, ledger --
            final_removed = within_flag | cross_flag
            output_dir.mkdir(parents=True, exist_ok=True)
            output_files: list[str] = []
            if use_dist:
                from puffer import ray_dist

                output_files = ray_dist.ray_write_outputs(
                    paths, chunk_plan, output_dir, final_removed, file_bounds, self.cfg,
                )
            else:
                for fi, path in enumerate(paths):
                    a, b = file_bounds[fi]
                    out_path = output_dir / path.name
                    _write_parquet_filtered_stream(path, out_path, final_removed[a:b], chunk_rows)
                    output_files.append(str(out_path))

            _phase(f"wrote {len(output_files)} output file(s)")
            removed_idx = np.nonzero(final_removed)[0]
            if removed_idx.size:
                rfidx = np.searchsorted(starts, removed_idx, side="right") - 1
                rrow = [int(x) for x in (removed_idx - starts[rfidx])]
                reasons = [
                    "within" if within_flag[i] else "cross"
                    for i in removed_idx
                ]
            else:
                rfidx, rrow, reasons = np.empty(0, dtype=int), [], []
            if file_ids is not None:
                # Interned: store a compact int32 file id (into the path table)
                # instead of the full source path per removed row.
                fid = [int(file_ids[int(fi)]) for fi in rfidx]
                removals_df = pl.DataFrame(
                    {"file_id": fid, "source_row": rrow, "reason": reasons},
                    schema={"file_id": pl.Int32, "source_row": pl.Int64, "reason": pl.Utf8},
                )
            else:
                rsf = [str(paths[int(fi)]) for fi in rfidx]
                removals_df = pl.DataFrame(
                    {"source_file": rsf, "source_row": rrow, "reason": reasons},
                    schema={"source_file": pl.Utf8, "source_row": pl.Int64, "reason": pl.Utf8},
                )
            _write_parquet_atomic(removals_df, self.state_dir / "removals" / f"{safe_tag}.parquet")

            _write_json_atomic(
                self.state_dir / "datasets" / safe_tag / "meta.json",
                {
                    "dataset": dataset,
                    "n_docs": n_total,
                    "output_dir": str(output_dir),
                    "output_files": output_files,
                    "ray_transport": ray_transport,
                    "signature_source": signature_source,
                },
            )
            event = {
                "op": "ingest",
                "dataset": dataset,
                "utc": _utcnow_iso(),
                "n_docs": n_total,
                "output_dir": str(output_dir),
                "ray_transport": ray_transport,
                "signature_source": signature_source,
            }
            if file_ids is not None:
                # Ordered, resolved input ids (into the path table) + effective
                # text column: the record faithful withdrawal replays from.
                event["inputs"] = file_ids
                event["column"] = column
            _append_ledger_event(self.state_dir, event)

            if logger.isEnabledFor(logging.INFO):
                from puffer.index import iter_shards

                tot_shards = tot_keys = 0
                for band_id in range(num_bands):
                    ents = iter_shards(index_dir, band_id, None)
                    tot_shards += len(ents)
                    tot_keys += sum(int(meta.get("count", 0)) for meta, _p in ents)
                logger.info(
                    "[%s] hash_index: %d shard(s), ~%d key(s) across %d band(s)",
                    dataset, tot_shards, tot_keys, num_bands,
                )
            logger.info(
                "[%s] ingest done: in=%d out=%d within=%d cross=%d "
                "in %.1fs (peak RSS %.1f GiB, transport=%s)",
                dataset, n_total, int((~final_removed).sum()),
                int(within_flag.sum()), int(cross_flag.sum()), time.monotonic() - start,
                _peak_rss_gib(), ray_transport,
            )
            return IngestReport(
                dataset=dataset,
                n_input=n_total,
                n_within_removed=int(within_flag.sum()),
                n_cross_removed=int(cross_flag.sum()),
                n_output=int((~final_removed).sum()),
                output_files=output_files,
                probes_done=probes_done,
                probes_scheduled=probes_scheduled,
                elapsed_s=time.monotonic() - start,
                ray_transport=ray_transport,
            )
        finally:
            if band_keys is not None:
                del band_keys
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # -- withdraw internals -------------------------------------------------

