"""Cross-history screening: is a document's band key already in the index?

Screening a release's band keys against a band's shard history is a boolean
OR: a row is a hit if it matches ANY shard in ANY band. The reference meaning
is the full cross-membership OR -- each row checked against every shard of a
band via ``searchsorted`` -- and the optimized path below returns exactly that
result while performing fewer probes.

``screen_release`` is the optimized, early-stopping path (invented for
PUFFER). It exploits that the OR is over a
MONOTONE sequence of per-probe hit indicators: once a row is known to match
some shard, no later probe (a different shard in the same band, or any shard
in a later band) can change its membership in the union. So it is always
safe to:

  * stop probing a row entirely once it has hit (skip its remaining bands),
  * within a band, hand each shard a shrinking "active" subset — the rows
    that are STILL unmatched by every shard probed so far in that band —
    instead of re-probing rows the band has already resolved.

Both are pure "skip a row whose answer is already `True`" optimizations
over a monotone OR, so the RESULT is provably identical to the full
cross-membership OR over every band/shard; only the number of probes
performed changes. Bands are still visited in a fixed (sequential) order so
the "active rows so far" carries forward correctly, but which shards are
probed FIRST within a band is a pure performance heuristic (``probe_order``)
with no effect on the result.

Within a band, the still-active row set is split into contiguous slices and
handed to a thread pool: ``np.searchsorted`` releases the GIL, so threads
usefully overlap CPU-bound probing (unlike a process pool, no serialization
of the mmap'd shard arrays is needed).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

_PROBE_ORDERS = ("largest_first", "smallest_first", "newest_first")


def _order_shard_entries(entries: list[tuple[dict, Path]], probe_order: str):
    if probe_order == "largest_first":
        return sorted(entries, key=lambda e: e[0].get("count", 0), reverse=True)
    if probe_order == "smallest_first":
        return sorted(entries, key=lambda e: e[0].get("count", 0))
    if probe_order == "newest_first":
        return list(reversed(entries))
    raise ValueError(
        f"probe_order must be one of {_PROBE_ORDERS}, got {probe_order!r}"
    )


def _screen_chunk(keys_chunk, shard_arrays_ordered) -> tuple["object", int]:
    """Sequentially probe one contiguous chunk's keys against ordered shards,
    shrinking the active subset after each shard. Returns (hit_mask_over_chunk,
    n_probes_performed)."""
    import numpy as np

    n = keys_chunk.shape[0]
    hit_chunk = np.zeros(n, dtype=bool)
    active = np.arange(n)
    probes = 0
    for arr in shard_arrays_ordered:
        if active.size == 0:
            break
        n_arr = len(arr)
        if n_arr == 0:
            continue
        probes += int(active.size)
        sub_keys = keys_chunk[active]
        idx = np.searchsorted(arr, sub_keys)
        valid = idx < n_arr
        idx_c = np.minimum(idx, n_arr - 1)
        m = valid & (np.asarray(arr[idx_c]) == sub_keys)
        hit_positions = active[m]
        if hit_positions.size:
            hit_chunk[hit_positions] = True
        active = active[~m]
    return hit_chunk, probes


def screen_release(band_keys, index_dir: Path, exclude_tag: str | None, cfg, counters: dict | None = None):
    """Early-stopping cross-history screen over ``band_keys`` (n_docs x
    num_bands int64). Returns a boolean mask over rows -- identical to the
    full cross-membership OR across every band's shards.

    Bands are probed sequentially; a row already hit by an earlier band is
    excluded from all later probing. Within a band, shards are probed in
    ``cfg.probe_order`` and the still-active row subset shrinks after each
    shard. The still-active set is split into contiguous slices and probed
    in a thread pool sized by ``cfg.effective_workers``.

    When ``counters`` is given, it is populated with ``probes_scheduled``
    (the full probe schedule, Sum over bands of n_docs x n_shards -- a fixed
    upper bound independent of hit patterns) and ``probes_done`` (the actual
    number of row x shard searches performed).
    """
    import numpy as np

    from puffer.index import iter_shards, read_shard_bin

    band_keys = np.asarray(band_keys)
    n_docs, num_bands = band_keys.shape
    hit = np.zeros(n_docs, dtype=bool)
    probes_scheduled = 0
    probes_done = 0
    n_workers = max(1, int(getattr(cfg, "effective_workers", 1)))
    probe_order = getattr(cfg, "probe_order", "largest_first")

    for band_id in range(num_bands):
        entries = iter_shards(index_dir, band_id, exclude_tag)
        n_shards = len(entries)
        probes_scheduled += n_docs * n_shards
        if n_shards == 0:
            continue
        active_idx = np.nonzero(~hit)[0]
        if active_idx.size == 0:
            continue
        ordered = _order_shard_entries(entries, probe_order)
        arrays = [read_shard_bin(path, mmap=True) for _meta, path in ordered]
        keys_col = band_keys[:, band_id]

        n_chunks = min(n_workers, int(active_idx.size)) or 1
        chunks = [c for c in np.array_split(active_idx, n_chunks) if c.size]

        def _work(idx_chunk):
            hit_chunk, probes = _screen_chunk(keys_col[idx_chunk], arrays)
            return idx_chunk[hit_chunk], probes

        if len(chunks) <= 1:
            results = [_work(c) for c in chunks]
        else:
            with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
                results = list(ex.map(_work, chunks))

        for hit_rows, probes in results:
            if hit_rows.size:
                hit[hit_rows] = True
            probes_done += probes

    if counters is not None:
        counters["probes_scheduled"] = int(probes_scheduled)
        counters["probes_done"] = int(probes_done)
    return hit
