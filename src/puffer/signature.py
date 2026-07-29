"""MinHash signatures and LSH band keys.

Per document: shingle the text (:mod:`puffer.shingle`), compute a rensa
``RMinHash(num_perm, seed)`` signature over the shingles, split the
signature into ``num_bands`` contiguous rows of ``num_perm // num_bands``
permutation values each, and hash each row's packed bytes with xxh64 to
get one band key per band (same packing: ``struct.pack(f"{rows}I",
*band_sig)`` over the raw uint32 MinHash rows), but returns a vectorized
``(n_docs, num_bands)`` ``int64`` array instead of per-doc byte blobs.

Signedness: xxh64 produces an *unsigned* 64-bit digest, but every persisted
band-key shard is a sorted ``int64`` array (see ``puffer.index``), and
``np.uint64 -> np.int64`` is a reinterpret-the-bits view, not a numeric
cast. The reinterpret MUST happen before any sort — sorting the uint64
values first and then viewing as int64 would silently permute the array
(large unsigned values with the top bit set become negative int64s, which
sort *before* every non-negative value). Every band-key array this module
produces is already int64 by construction, so callers can sort/search it
directly with no further conversion.

Parallelism: rensa's ``RMinHash.update``/``digest`` hold the GIL for the
whole call (measured: ~1.0x wall time across 1/2/4/8 threads, i.e. no
threading speedup), so this fans out via :func:`puffer.parallel.map_chunks`
with ``mode="process"``, despite ``cfg.effective_workers`` being a plain
worker count used uniformly elsewhere in puffer.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from puffer.parallel import map_chunks
from puffer.shingle import make_shingles

if TYPE_CHECKING:
    import numpy as np

    from puffer.config import PufferConfig


def _band_keys_for_text(text: str, cfg: "PufferConfig") -> list[int]:
    """Return ``cfg.num_bands`` unsigned 64-bit band hashes for one document."""
    import struct

    import xxhash
    from rensa import RMinHash

    shingles = make_shingles(text, cfg.ngram_type, cfg.ngram_size)
    m = RMinHash(num_perm=cfg.num_perm, seed=cfg.seed)
    m.update(shingles)
    sig = m.digest()  # list[int], len == num_perm, each a uint32

    rows = cfg.rows_per_band
    band_hashes = []
    for band in range(cfg.num_bands):
        band_sig = sig[band * rows : (band + 1) * rows]
        h = xxhash.xxh64(struct.pack(f"{rows}I", *band_sig)).intdigest()
        band_hashes.append(h)
    return band_hashes


def _compute_chunk(texts: list[str], cfg: "PufferConfig") -> "np.ndarray":
    """Compute band keys for a contiguous slice of documents (one worker's share).

    Module-level (not a closure) and picklable together with ``cfg`` so it
    can be shipped to a ``ProcessPoolExecutor`` worker by
    :func:`puffer.parallel.map_chunks`.
    """
    import numpy as np

    if not texts:
        return np.empty((0, cfg.num_bands), dtype=np.int64)
    rows = [_band_keys_for_text(t, cfg) for t in texts]
    # xxh64 digests are unsigned; reinterpret (not cast) to int64 for storage/sort.
    arr_u64 = np.array(rows, dtype=np.uint64)
    return arr_u64.view(np.int64)


def compute_band_keys(texts: list[str], cfg: "PufferConfig") -> "np.ndarray":
    """Compute LSH band keys for every document in ``texts``.

    Returns an ``(len(texts), cfg.num_bands)`` ``int64`` array; row ``i``
    holds the band keys for ``texts[i]``. Deterministic across worker
    counts and repeated calls given the same ``cfg`` (same ``num_perm``,
    ``num_bands``, ``seed``, ``ngram_type``, ``ngram_size``) — chunking
    only changes which worker computes a given row, never its value.
    """
    import numpy as np

    n = len(texts)
    if n == 0:
        return np.empty((0, cfg.num_bands), dtype=np.int64)

    fn = functools.partial(_compute_chunk, cfg=cfg)
    results = map_chunks(fn, texts, cfg.effective_workers, mode="process")
    if not results:
        return np.empty((0, cfg.num_bands), dtype=np.int64)
    return np.concatenate(results, axis=0)


def chunk_signature(texts: list[str], cfg: "PufferConfig"):
    """Band keys for one chunk, computed inline.

    Runs single-process (no nested pool): this is the unit a persistent pool
    schedules across chunks in the streaming ingest, so it must never spawn its
    own workers. Returns the ``band_keys`` ndarray.
    """
    return _compute_chunk(texts, cfg)
