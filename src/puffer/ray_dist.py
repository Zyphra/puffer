"""Distributed ingest stages for the Ray executor.

Implements a parallelization model for PUFFER's distributed path while
preserving PUFFER's decision semantics and single-writer commit:

  * **Row-group chunk planning** (per-chunk tasks, not per-file). Each input
    parquet is split into chunks of consecutive row
    groups sized toward ``~3`` scheduling waves per usable CPU slot, and the
    chunk list is dispatched **round-robin across files**, so parallelism and
    tail latency are set by total rows -- independent of the input's file
    count or file-size distribution. A file with a single giant row group
    cannot be split further (row groups are the smallest independently
    readable parquet unit); such inputs degrade gracefully to per-file
    granularity.

  * **Signature spool** -- each chunk is one Ray task that writes its band
    keys to a *shared-filesystem* spool file (band-major ``int64``, shape
    ``(num_bands, n_rows)``) via tmp+rename, returning only tiny metadata.
    This replaces returning band-key matrices through the object store to a
    single draining driver. Files are opened by readers only after the
    atomic rename, so plain read() syscalls see full contents under the
    shared filesystem's close-to-open coherence.

  * **Per-band reduce** -- one Ray task per band, since each band is fully
    independent. The task reads ONLY its band's row from every
    spool chunk (in global row order), computes the within-release
    keep-first flags (identical lexsort tie-break as the local path),
    screens the full presented column against the band's historical shards
    (union over shards == the local early-stop result), and writes the
    band's sorted-unique key set to the spool. The driver merges the tiny
    row-id sets and performs the atomic manifest commit + compaction itself,
    keeping the index single-writer.

  * **Distributed output writes** -- one Ray task per chunk applies the
    release's removed-row mask (shipped bit-packed, ~n_rows/8 bytes) and
    streams its row groups to a part file at row-group granularity.
    Single-chunk files keep their original output
    name; multi-chunk files emit ``<stem>.partNNNNN<suffix>`` parts in row
    order. Each task carries a RAM reservation derived from its row groups'
    uncompressed size so Ray packs nodes by memory when that binds first
    (reserving RAM per finalize task so Ray packs by memory, not CPU).

  * **Target-utilization reservation**: one-CPU chunk tasks reserve
    ``1/target_util`` CPUs so a "fully packed"
    node sits at ``~target_util`` of its slots, leaving headroom for the
    raylet/OS. ``PUFFER_RAY_TARGET_UTIL`` (default ``0.8``); set ``0`` to
    disable and pack every slot.

Structures deliberately not used here:

  * removal spill buckets: spilling Python-object removal maps to disk is
    needed when they reach hundreds of GB, but
    PUFFER's removal state is two release-length boolean numpy arrays on
    the driver plus bit-packed per-chunk slices in flight -- bounded and
    small at any release size that fits the band-key index model.
  * chunk-filter/shard-merge pipelining (overlapping the two stages):
    PUFFER has no merge stage; the band reduce needs every spool chunk, so
    there is nothing to overlap.

Decision equivalence with the local executor:

  * within: same per-band ``lexsort((row_in_file, file_rank, band_key))``
    keep-first reduction over all rows (no exact pre-stage).
  * cross: screening is a monotone OR across bands/shards; per-band union
    equals the early-stopping multi-band mask (only probe counts differ).
    Rows the local path never probes (already within-flagged) may be probed
    here; the final removed set is identical because within takes priority.
  * commit: identical ``np.unique`` over the full presented column per band.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from puffer.config import PufferConfig

logger = logging.getLogger(__name__)

#: Scheduling waves per usable CPU slot the chunk planner aims for: enough
#: tasks for load balance, not so many that per-task dispatch overhead dominates.
_WAVES_PER_SLOT = 3

#: Floor for the per-output-task RAM reservation.
_MIN_OUTPUT_TASK_MEM = 256 * 1024 * 1024


def _target_util() -> float:
    """Target node utilization for one-CPU chunk tasks."""
    try:
        util = float(os.environ.get("PUFFER_RAY_TARGET_UTIL", "0.8"))
    except ValueError:
        util = 0.8
    return util


def _chunk_task_cpus() -> float:
    """CPU reservation for a nominal one-CPU chunk task.

    Reserving ``1/target_util`` CPUs makes Ray's "node full" equal
    ``~target_util`` of the advertised slots (headroom for raylet/OS work),
    a target-utilization reservation mechanism.
    """
    util = _target_util()
    if 0.0 < util < 1.0:
        return 1.0 / util
    return 1.0


@dataclass(frozen=True)
class ReleaseChunk:
    """One dispatchable unit: consecutive row groups of one input file."""

    file_index: int
    chunk_index: int  # per-file ordinal
    rg_lo: int  # first row group (inclusive)
    rg_hi: int  # last row group (exclusive)
    row_offset: int  # file-local starting row
    n_rows: int
    est_bytes: int  # uncompressed row-group bytes (footer metadata)


def plan_release_chunks(
    rg_rows: Sequence[Sequence[int]],
    rg_bytes: Sequence[Sequence[int]],
    n_total: int,
    cfg: PufferConfig,
) -> list[ReleaseChunk]:
    """Split every file into row-group chunks and order them round-robin.

    ``rg_rows[fi]``/``rg_bytes[fi]`` are per-row-group row counts and
    uncompressed byte sizes from file ``fi``'s parquet footer. The target
    chunk size aims at ``_WAVES_PER_SLOT`` chunks per usable CPU slot with a
    floor of ``cfg.sig_chunk_rows`` (micro-chunks only inflate dispatch and
    spool-file overhead). Chunks never split a row group -- the smallest
    independently readable parquet unit.

    The returned order interleaves files (all files' chunk 0, then chunk 1,
    ...) so early scheduling waves spread across distinct inputs regardless
    of per-file size skew -- the "round robin" dispatch.
    """
    from puffer.ray_exec import cluster_cpus, resolve_max_in_flight

    cap = resolve_max_in_flight(cfg.ray_max_in_flight)
    slots = max(1, min(cap, cluster_cpus()))
    target = max(int(cfg.sig_chunk_rows), -(-int(n_total) // (slots * _WAVES_PER_SLOT)))

    per_file: list[list[ReleaseChunk]] = []
    for fi, groups in enumerate(rg_rows):
        sizes = rg_bytes[fi]
        chunks: list[ReleaseChunk] = []
        lo = 0
        row_off = 0
        while lo < len(groups):
            hi = lo
            rows = 0
            est = 0
            while hi < len(groups) and (rows == 0 or rows + int(groups[hi]) <= target):
                rows += int(groups[hi])
                est += int(sizes[hi])
                hi += 1
            chunks.append(ReleaseChunk(fi, len(chunks), lo, hi, row_off, rows, est))
            row_off += rows
            lo = hi
        if not chunks:  # zero-row (but valid) parquet: keep one empty unit
            chunks.append(ReleaseChunk(fi, 0, 0, 0, 0, 0, 0))
        per_file.append(chunks)

    ordered: list[ReleaseChunk] = []
    layer = 0
    while True:
        wave = [cf[layer] for cf in per_file if layer < len(cf)]
        if not wave:
            break
        ordered.extend(wave)
        layer += 1
    return ordered


# -- signature spool ---------------------------------------------------------

def _spool_path(spool_dir: str | Path, chunk: ReleaseChunk) -> Path:
    return Path(spool_dir) / f"bk_{chunk.file_index:06d}_{chunk.chunk_index:05d}.i64"


def _signature_spool_task(
    pq_file: str,
    cfg: PufferConfig,
    text_column: str,
    spool_dir: str,
    chunk: ReleaseChunk,
) -> dict[str, Any]:
    """Compute one chunk's band keys and spool them band-major to shared FS.

    Band-major layout means a band-reduce task can read its row with one
    contiguous ``np.fromfile(offset=band_id * n_rows * 8, count=n_rows)``.
    Written to a tmp name then renamed, so any reader that can see the final
    name sees complete bytes (close-to-open coherence). The rename is
    idempotent: a Ray retry after a worker crash rewrites the same content.
    """
    from dataclasses import replace

    import numpy as np
    import pyarrow.parquet as pq

    from puffer.signature import compute_band_keys

    pf = pq.ParquetFile(str(pq_file))
    if text_column not in pf.schema_arrow.names:
        raise ValueError(f"text column {text_column!r} is absent from {pq_file}")
    scfg = replace(cfg, n_workers=1)  # one thread per chunk task; Ray owns parallelism
    parts: list = []
    n = 0
    if chunk.rg_hi > chunk.rg_lo:
        for batch in pf.iter_batches(
            batch_size=max(1, cfg.sig_chunk_rows),
            row_groups=list(range(chunk.rg_lo, chunk.rg_hi)),
            columns=[text_column],
        ):
            texts = ["" if v is None else str(v) for v in batch.column(0).to_pylist()]
            if texts:
                parts.append(compute_band_keys(texts, scfg))
                n += len(texts)
            del texts
    if n != chunk.n_rows:
        raise ValueError(
            f"chunk {chunk.file_index}:{chunk.chunk_index} of {pq_file} read {n} rows; "
            f"footer promised {chunk.n_rows}"
        )
    band_keys = (
        np.concatenate(parts, axis=0) if parts
        else np.empty((0, cfg.num_bands), dtype=np.int64)
    )
    dest = _spool_path(spool_dir, chunk)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.tmp_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(np.ascontiguousarray(band_keys.T).tobytes())  # band-major
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return {
        "file_index": chunk.file_index,
        "chunk_index": chunk.chunk_index,
        "row_offset": chunk.row_offset,
        "n_rows": n,
        "path": str(dest),
    }


def _spool_job(job: tuple) -> dict[str, Any]:
    return _signature_spool_task(*job)


def ray_spool_band_keys(
    pq_files: Sequence[str | Path],
    chunks: Sequence[ReleaseChunk],
    cfg: PufferConfig,
    spool_dir: str | Path,
    *,
    text_column: str | None = None,
) -> list[dict[str, Any]]:
    """Spool every chunk's band keys to shared FS; returns metadata in global
    row order (sorted by ``(file_index, chunk_index)``).

    Dispatch follows the planner's round-robin chunk order (no Bloom to feed
    in presentation order on this path); each returned payload is a tiny
    dict, so the driver never accumulates signature bytes. In-flight is the
    cluster-wide :func:`puffer.ray_exec.resolve_max_in_flight` cap.
    """
    from puffer.ray_exec import ray_map_unordered, resolve_max_in_flight

    column = text_column or cfg.text_column
    paths = [str(Path(p)) for p in pq_files]
    if not chunks:
        return []
    jobs = [(paths[c.file_index], cfg, column, str(spool_dir), c) for c in chunks]
    cap = resolve_max_in_flight(cfg.ray_max_in_flight)
    logger.info(
        "spool: %d chunk(s) over %d file(s), %.2f CPU/task, cluster cap=%d",
        len(chunks), len({c.file_index for c in chunks}), _chunk_task_cpus(), cap,
    )
    metas: dict[tuple[int, int], dict[str, Any]] = {}
    for _job, res in ray_map_unordered(
        _spool_job, jobs, num_cpus=_chunk_task_cpus(), scheduling_strategy="SPREAD",
        label="signature-spool", max_in_flight=cap,
    ):
        if isinstance(res, BaseException):
            raise res
        metas[(res["file_index"], res["chunk_index"])] = res
    if len(metas) != len(chunks):
        raise RuntimeError(f"signature spool returned {len(metas)} of {len(chunks)} chunks")
    return [metas[(c.file_index, c.chunk_index)] for c in sorted(
        chunks, key=lambda c: (c.file_index, c.chunk_index)
    )]


# -- per-band reduce ---------------------------------------------------------

def _band_reduce_task(
    band_id: int,
    spool_metas: Sequence[dict[str, Any]],
    file_rank: Sequence[int],
    index_dir: str,
    exclude_tag: str | None,
    cfg: PufferConfig,
    uniq_dir: str,
    n_threads: int,
) -> dict[str, Any]:
    """Within-release keep-first + cross-history screen + unique for one band.

    ``spool_metas`` is in global row order. Reads only this band's row from
    every spool chunk (contiguous), reproduces the local within-release
    reduction (identical lexsort ordering), screens the full presented column
    against the band's shards (probe order per ``cfg.probe_order``; union
    over shards), and writes the band's sorted unique keys for the driver to
    commit.
    """
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from puffer.index import iter_shards, read_shard_bin
    from puffer.screen import _order_shard_entries, _screen_chunk

    total = int(sum(int(m["n_rows"]) for m in spool_metas))
    kb = np.empty(total, dtype=np.int64)
    rw = np.empty(total, dtype=np.int64)
    fr = np.empty(total, dtype=np.int64)
    off = 0
    for m in spool_metas:
        n = int(m["n_rows"])
        if not n:
            continue
        kb[off:off + n] = np.fromfile(
            m["path"], dtype=np.int64, count=n, offset=band_id * n * 8
        )
        base = int(m["row_offset"])
        rw[off:off + n] = np.arange(base, base + n, dtype=np.int64)
        fr[off:off + n] = int(file_rank[int(m["file_index"])])
        off += n

    # within-release keep-first: identical ordering to the local path
    within_rows = np.empty(0, dtype=np.int64)
    if total:
        ob = np.lexsort((rw, fr, kb))
        ks = kb[ob]
        new_run = np.ones(ks.size, dtype=bool)
        np.not_equal(ks[1:], ks[:-1], out=new_run[1:])
        run_id = np.cumsum(new_run) - 1
        counts = np.bincount(run_id)
        flagged = (~new_run) & (counts[run_id] >= 2)
        within_rows = np.asarray(ob[flagged], dtype=np.int64)

    # cross-history screen for this band (union over shards == local result)
    probes_scheduled = 0
    probes_done = 0
    cross_rows = np.empty(0, dtype=np.int64)
    entries = iter_shards(Path(index_dir), band_id, exclude_tag)
    probes_scheduled = total * len(entries)
    if total and entries:
        ordered = _order_shard_entries(entries, getattr(cfg, "probe_order", "largest_first"))
        arrays = [read_shard_bin(path, mmap=True) for _meta, path in ordered]
        n_chunks = max(1, min(int(n_threads), total))
        bounds = np.linspace(0, total, n_chunks + 1, dtype=np.int64)
        slices = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]

        def _work(interval):
            a, b = interval
            hit_chunk, probes = _screen_chunk(kb[a:b], arrays)
            return a, hit_chunk, probes

        if len(slices) <= 1:
            results = [_work(s) for s in slices]
        else:
            with ThreadPoolExecutor(max_workers=len(slices)) as ex:
                results = list(ex.map(_work, slices))
        hits: list = []
        for a, hit_chunk, probes in results:
            probes_done += int(probes)
            rows = np.nonzero(hit_chunk)[0]
            if rows.size:
                hits.append(rows + a)
        if hits:
            cross_rows = np.concatenate(hits)

    # full presented unique keys for the commit (identical to local Phase 5)
    uniq = np.unique(kb) if total else np.empty(0, dtype=np.int64)
    uniq_dest = Path(uniq_dir) / f"uniq_band_{band_id:02d}.i64"
    fd, tmp = tempfile.mkstemp(dir=uniq_dest.parent, prefix=f".{uniq_dest.name}.tmp_")
    with os.fdopen(fd, "wb") as fh:
        fh.write(uniq.tobytes())
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, uniq_dest)

    return {
        "band_id": band_id,
        "within_rows": within_rows,
        "cross_rows": cross_rows,
        "uniq_path": str(uniq_dest),
        "n_unique": int(uniq.size),
        "probes_scheduled": int(probes_scheduled),
        "probes_done": int(probes_done),
    }


def _band_reduce_job(job: tuple) -> dict[str, Any]:
    return _band_reduce_task(*job)


def ray_band_reduce(
    spool_metas: Sequence[dict[str, Any]],
    file_rank: Sequence[int],
    index_dir: str | Path,
    exclude_tag: str | None,
    cfg: PufferConfig,
    uniq_dir: str | Path,
) -> list[dict[str, Any]]:
    """Fan one reduce task per band across the cluster; return per-band results.

    ``spool_metas`` must be in global row order (as returned by
    :func:`ray_spool_band_keys`). The rows a band task sees are therefore
    positionally identical to the local path's global row order. Each task
    reserves ``cluster_cpus // num_bands`` CPUs (floor 1, capped to one node)
    -- one band per node sizing -- threads its shard probing to
    that width, and reserves RAM for its release-length working set
    (three int64 columns plus sort workspace, ~64 bytes/row) so Ray packs
    band tasks by memory, never co-locating more than a node can hold.
    """
    from puffer.ray_exec import _ensure_ray, cluster_cpus, num_nodes, ray_map_unordered

    _ensure_ray()  # cluster_cpus()/num_nodes() on uninitialized Ray return tiny defaults
    ccpu = max(1, cluster_cpus())
    per_node = max(1, ccpu // max(1, num_nodes()))
    band_cpus = max(1, min(per_node, ccpu // max(1, cfg.num_bands)))
    metas = list(spool_metas)
    total_rows = sum(int(m["n_rows"]) for m in metas)
    # kb/rw/fr (24 B/row) + lexsort permutation, sorted copy, run bookkeeping
    # (~40 B/row peak) -- reserve 64 B/row with a modest floor.
    band_task_mem = float(max(_MIN_OUTPUT_TASK_MEM, 64 * total_rows))
    logger.info(
        "band-reduce: %d band(s), %d CPU/band, %.2f GiB/band reserved (%d rows)",
        cfg.num_bands, band_cpus, band_task_mem / 1024 ** 3, total_rows,
    )
    jobs = [
        (
            band_id, metas, list(file_rank), str(Path(index_dir).resolve()),
            exclude_tag, cfg, str(uniq_dir), band_cpus,
        )
        for band_id in range(cfg.num_bands)
    ]
    results: list[dict[str, Any] | None] = [None] * cfg.num_bands
    for _job, res in ray_map_unordered(
        _band_reduce_job, jobs, num_cpus=band_cpus, memory_bytes=band_task_mem,
        scheduling_strategy="SPREAD", label="band-reduce", max_in_flight=cfg.num_bands,
    ):
        if isinstance(res, BaseException):
            raise res
        results[res["band_id"]] = res
    missing = [i for i, r in enumerate(results) if r is None]
    if missing:
        raise RuntimeError(f"band reduce lost bands {missing}")
    return results  # type: ignore[return-value]


# -- distributed output writes ------------------------------------------------

def _part_name(src_name: str, chunk: ReleaseChunk, n_chunks_for_file: int) -> str:
    """Single-chunk files keep the input name (1:1 like the local path);
    multi-chunk files emit ordered part files."""
    if n_chunks_for_file <= 1:
        return src_name
    p = Path(src_name)
    return f"{p.stem}.part{chunk.chunk_index:05d}{p.suffix}"


def _output_write_task(
    src: str,
    dst: str,
    rg_lo: int,
    rg_hi: int,
    packed_removed: bytes,
    n_rows: int,
    batch_rows: int,
) -> str:
    """Stream one chunk's row groups to a filtered part file on a worker.

    Mirrors ``pipeline._write_parquet_filtered_stream`` (atomic tmp+rename,
    full schema preserved, O(batch) resident rows) restricted to
    ``[rg_lo, rg_hi)``.
    """
    import numpy as np
    import pyarrow.parquet as pq

    removed = np.unpackbits(
        np.frombuffer(packed_removed, dtype=np.uint8), count=n_rows
    ).astype(bool)
    src_path, dst_path = Path(src), Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(str(src_path))
    fd, tmp = tempfile.mkstemp(dir=dst_path.parent, prefix=f".{dst_path.name}.tmp_")
    os.close(fd)
    writer = None
    try:
        pos = 0
        if rg_hi > rg_lo:
            for batch in pf.iter_batches(
                batch_size=max(1, batch_rows), row_groups=list(range(rg_lo, rg_hi))
            ):
                m = batch.num_rows
                keep = ~removed[pos:pos + m]
                pos += m
                import pyarrow as pa

                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema)
                writer.write_table(table.filter(pa.array(keep)))
        if writer is None:  # empty chunk: still emit schema-only parquet
            import pyarrow as pa

            writer = pq.ParquetWriter(tmp, pf.schema_arrow)
        writer.close()
        writer = None
        if pos != n_rows:
            raise ValueError(f"{src}: chunk read {pos} rows, mask has {n_rows}")
        os.replace(tmp, dst_path)
    except BaseException:
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.close()
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return dst


def _output_write_job(job: tuple) -> str:
    return _output_write_task(*job)


def ray_write_outputs(
    paths: Sequence[Path],
    chunks: Sequence[ReleaseChunk],
    output_dir: Path,
    final_removed,
    file_bounds: Sequence[tuple[int, int]],
    cfg: PufferConfig,
) -> list[str]:
    """Write filtered outputs as one Ray task per chunk.

    The removed mask is shipped bit-packed per chunk (n_rows/8 bytes) and
    jobs are built lazily, so resident mask bytes are bounded by the
    in-flight cap rather than the release size. Each task reserves RAM
    proportional to the largest chunk's uncompressed footer size (floor
    256 MiB) so Ray packs nodes by memory when that binds first. The
    returned list is ordered by ``(file_index, chunk_index)`` -- global row
    order -- regardless of completion order.
    """
    import numpy as np

    from puffer.ray_exec import ray_map_unordered, resolve_max_in_flight

    ordered = sorted(chunks, key=lambda c: (c.file_index, c.chunk_index))
    n_chunks_per_file: dict[int, int] = {}
    for c in ordered:
        n_chunks_per_file[c.file_index] = n_chunks_per_file.get(c.file_index, 0) + 1

    dests = [
        str(Path(output_dir) / _part_name(
            Path(paths[c.file_index]).name, c, n_chunks_per_file[c.file_index]
        ))
        for c in ordered
    ]

    def _jobs():
        for c, dst in zip(ordered, dests):
            a, _b = file_bounds[c.file_index]
            lo = a + c.row_offset
            packed = np.packbits(
                np.asarray(final_removed[lo:lo + c.n_rows], dtype=bool)
            ).tobytes()
            yield (
                str(paths[c.file_index]), dst, c.rg_lo, c.rg_hi, packed, c.n_rows,
                cfg.sig_chunk_rows,
            )

    max_est = max((c.est_bytes for c in ordered), default=0)
    task_mem = float(max(_MIN_OUTPUT_TASK_MEM, 2 * max_est))
    cap = resolve_max_in_flight(cfg.ray_max_in_flight)
    logger.info(
        "output-write: %d chunk task(s), %.0f MiB/task reserved, cluster cap=%d",
        len(dests), task_mem / 1024 ** 2, cap,
    )
    for _job, res in ray_map_unordered(
        _output_write_job, _jobs(), num_cpus=_chunk_task_cpus(), memory_bytes=task_mem,
        scheduling_strategy="SPREAD", label="output-write", max_in_flight=cap,
    ):
        if isinstance(res, BaseException):
            raise res
    return dests
