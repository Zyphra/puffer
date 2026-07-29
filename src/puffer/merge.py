"""Merge executor for the LSM band-key index: in-RAM or streaming k-way unique.

``merge_runs`` is the single place that actually rewrites shards (used by
both tiered compaction and withdrawal's survivor-rebuild path). Two
execution modes share one contract — same output, same crash-safety, same
provenance bookkeeping:

  * **In-RAM**: when the selected inputs fit ``ram_budget``, concatenate and
    ``np.unique`` in one shot. Simple and fast for the common case.
  * **Streaming** (``streaming_merge_unique``): a bounded-RAM k-way merge of
    sorted-unique int64 files. Each round advances every active input by up
    to ``chunk_keys`` keys, picks ``cut`` = the minimum chunk-end value across
    inputs, merges everything <= ``cut`` in RAM, and appends it (deduped
    against the previous round's boundary) to the output. Because ``cut`` is
    the min of the chunk-end values, every input has consumed at least up to
    ``cut`` by the time the round closes, so no smaller value is ever left
    behind — the round boundary is always safe to flush. ``chunk_keys`` is
    sized by ``stream_chunk_keys`` from a ~3x working-set model (inputs +
    concat copy + unique output).

Crash safety: the merged shard is written to a temp file and swapped into
place, the manifest is written (new entry in, selected entries out) via
write-tmp-then-``os.replace``, and ONLY THEN are the input shard files
unlinked. A crash at any point before the manifest swap leaves the old
shards + old manifest intact (merge simply re-runs); a crash after leaves
the new shard extraneous but harmless (next compaction call re-derives
selection from the current manifest, which no longer references the stale
inputs).

Provenance: each input shard's ``source_datasets`` (the L0 dataset tags
whose keys it carries, transitively) are unioned into the merged shard's
``source_datasets``. If ANY input shard's provenance is unknown (``None`` —
e.g. a hand-rolled index without provenance), the union is poisoned to
``None`` rather than silently under-reporting; downstream (withdrawal)
falls back to the conservative "rebuild from all remaining datasets" path
whenever it sees ``None``.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def stream_chunk_keys(ram_budget: int, k_inputs: int) -> int:
    """Per-input chunk size (keys) for a streaming-merge round.

    Derived from ``ram_budget`` via a working-set MODEL (not a measured
    peak): roughly 3x the resident input keys per round (inputs + the
    concatenate copy + the unique output), i.e. ``k_inputs * chunk * 8 * 3``
    bytes.

    A 64Ki-key floor keeps I/O in efficient sequential blocks. When the
    requested budget is below the operational minimum for that floor
    (``k_inputs * 64Ki * 24`` bytes, ~12 MiB at k=8), the effective working
    set EXCEEDS the request and a warning is logged — the budget is
    advisory below that point, never silently honored.
    """
    per_round_factor = 3 * 8 * max(1, k_inputs)
    derived = int(ram_budget // per_round_factor)
    floor = 64 * 1024
    if derived < floor:
        logger.warning(
            "streaming merge: requested ram_budget %d B is below the "
            "operational minimum for %d inputs (~%d B working set at the "
            "64Ki-key I/O floor); proceeding with the minimum — the budget "
            "is advisory here",
            ram_budget, k_inputs, floor * per_round_factor,
        )
        return floor
    return derived


def streaming_merge_unique(
    paths: list[Path],
    out_path: Path,
    chunk_keys: int = 16 * 1024 * 1024,
) -> tuple[int, int, int]:
    """K-way merge of sorted-unique int64 ``.bin`` files into one sorted-unique
    output, with a modeled RAM working set and sequential I/O throughout.

    Working-set model (not a measured guarantee): <= ``k * chunk_keys`` input
    keys resident per round plus the concatenate/unique copies (~3x total);
    size ``chunk_keys`` via :func:`stream_chunk_keys`.

    Per round: ``cut`` = the minimum over active inputs of that input's
    chunk-end value (the last key in its next up-to-``chunk_keys`` window).
    Every value <= ``cut`` in every input is consumed this round (each input
    contributes at most ``chunk_keys`` of them, since its own chunk-end value
    is >= ``cut``), merged in RAM, deduped against the previous round's
    boundary via ``carry``, and appended to the output.

    Returns ``(count, min, max)``; ``(0, 0, 0)`` when all inputs are empty.
    """
    import numpy as np

    mms = [np.memmap(p, dtype=np.int64, mode="r") for p in paths if Path(p).stat().st_size > 0]
    ptrs = [0] * len(mms)
    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.name + ".tmp")
    n_out = 0
    mn: int | None = None
    mx: int | None = None
    carry: int | None = None
    with open(tmp, "wb") as f:
        while True:
            active = [i for i in range(len(mms)) if ptrs[i] < mms[i].shape[0]]
            if not active:
                break
            cut = min(
                int(mms[i][min(ptrs[i] + chunk_keys, mms[i].shape[0]) - 1])
                for i in active
            )
            parts = []
            for i in active:
                end = min(ptrs[i] + chunk_keys, mms[i].shape[0])
                window = mms[i][ptrs[i]:end]
                hi = ptrs[i] + int(np.searchsorted(window, cut, side="right"))
                if hi > ptrs[i]:
                    parts.append(np.asarray(mms[i][ptrs[i]:hi]))
                    ptrs[i] = hi
            batch = np.unique(np.concatenate(parts)) if parts else None
            del parts
            if batch is None or batch.size == 0:
                continue
            if carry is not None:
                batch = batch[batch > carry]
            if batch.size:
                batch.tofile(f)
                n_out += int(batch.size)
                if mn is None:
                    mn = int(batch[0])
                carry = mx = int(batch[-1])
            del batch
    del mms
    tmp.replace(out_path)
    if n_out == 0:
        return 0, 0, 0
    return n_out, int(mn), int(mx)


def _shard_source_datasets(entry: dict) -> list[str] | None:
    v = entry.get("source_datasets")
    return list(v) if v is not None else None


def merge_runs(
    index_dir: Path,
    band_id: int,
    manifest: dict,
    selected: list[dict],
    new_level: int,
    ram_budget: int,
) -> int:
    """Merge ``selected`` shard entries into one new run at ``new_level``.

    In-RAM when the inputs fit ``ram_budget``, bounded-RAM streaming k-way
    merge otherwise. Writes the new run, swaps the manifest, unlinks the
    input shard files (crash-safe: new file + manifest first, unlinks last).
    Returns ``len(selected)``.
    """
    import numpy as np

    from puffer.index import band_dir, read_shard_bin, write_manifest, write_shard_bin

    bdir = band_dir(index_dir, band_id)
    seq = int(manifest.get("seq", 0)) + 1
    new_name = f"cmp_{seq:06d}.bin"
    total_bytes = sum(int(s.get("count", 0)) for s in selected) * 8

    if total_bytes <= ram_budget:
        arrs = [np.asarray(read_shard_bin(bdir / s["file"], mmap=False)) for s in selected]
        merged = np.unique(np.concatenate(arrs))
        del arrs
        write_shard_bin(merged, bdir / new_name)
        n_out, mn, mx = int(merged.size), int(merged[0]), int(merged[-1])
        del merged
    else:
        n_out, mn, mx = streaming_merge_unique(
            [bdir / s["file"] for s in selected], bdir / new_name,
            chunk_keys=stream_chunk_keys(ram_budget, len(selected)),
        )

    src_lists = [_shard_source_datasets(s) for s in selected]
    if any(v is None for v in src_lists):
        merged_sources: list[str] | None = None
    else:
        merged_sources = sorted({t for v in src_lists for t in v})
    new_entry = {
        "file": new_name,
        "level": int(new_level),
        "dataset": new_name[:-4],
        "source_datasets": merged_sources,
        "count": n_out,
        "min": mn,
        "max": mx,
    }
    selected_files = {s["file"] for s in selected}
    manifest["shards"] = [
        s for s in manifest.get("shards", []) if s["file"] not in selected_files
    ] + [new_entry]
    manifest["seq"] = seq
    write_manifest(index_dir, band_id, manifest)
    for f in selected_files:
        try:
            (bdir / f).unlink()
        except FileNotFoundError:
            pass
    return len(selected)
