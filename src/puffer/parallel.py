"""Chunked map helper shared by the signature and pipeline stages.

A single tiny wrapper around ``ThreadPoolExecutor``/``ProcessPoolExecutor``
so callers don't hand-roll chunk splitting + pool selection at every call
site. Order is always preserved (results come back in the same order as
``items`` regardless of completion order), since downstream row indices
(``source_row``) depend on it.

``mode="process"`` requires ``fn`` to be picklable (a module-level function,
not a closure/lambda) — the standard ``ProcessPoolExecutor`` constraint.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence


def _split_contiguous(n: int, n_chunks: int) -> list[tuple[int, int]]:
    if n_chunks <= 0:
        n_chunks = 1
    n_chunks = min(n_chunks, n) or 1
    base, rem = divmod(n, n_chunks)
    bounds = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        stop = start + size
        if size:
            bounds.append((start, stop))
        start = stop
    return bounds


def map_chunks(
    fn: Callable[[list], Any],
    items: Sequence[Any],
    n_workers: int,
    mode: str = "thread",
) -> list[Any]:
    """Split ``items`` into ``n_workers`` contiguous chunks, call ``fn`` on each
    chunk (a plain ``list``) in parallel, and return the per-chunk results in
    input order (one result per chunk, not flattened — callers concatenate
    themselves since a chunk's result shape depends on ``fn``).

    ``n_workers <= 1`` or fewer than 2 items runs ``fn`` inline with no pool.
    """
    n = len(items)
    if n == 0:
        return []
    bounds = _split_contiguous(n, max(1, int(n_workers)))
    chunks = [list(items[start:stop]) for start, stop in bounds]
    if len(chunks) <= 1:
        return [fn(chunks[0])] if chunks else []

    if mode == "thread":
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
            return list(ex.map(fn, chunks))
    elif mode == "process":
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
            return list(ex.map(fn, chunks))
    else:
        raise ValueError(f"mode must be 'thread' or 'process', got {mode!r}")
