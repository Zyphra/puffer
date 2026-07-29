"""Bounded-memory external k-way merge of sorted-unique int64 ``.bin`` files.

Withdrawal's survivor-rebuild re-unions the surviving contributor sidecars.
This module does that union with a bounded merge WORKING SET: ``ram_budget``
bytes cap the accounted merge buffers (and per-round temporaries), not the
whole process. Inputs are read in explicit bounded blocks -- ``fread`` in the
compiled kernel, ``readinto`` in the batch backend -- never ``np.memmap``,
whose mapped pages count toward process RSS as they are touched, and which numpy
ops can silently materialize to full anonymous RSS. The key
guarantee is size-INDEPENDENCE: the merge's resident storage is fixed by
``ram_budget`` and does not grow with the number of input keys N or the union
size, regardless of key distribution. Total process RSS is that budget plus a
fixed interpreter/library baseline (tens of MB), independent of N -- for a hard
per-process ceiling, run the kernel in a separately constrained process.

Backends (same sorted-unique output and ``(count, min, max)`` return):

  * ``losertree`` (default / ``auto``): the compiled C tournament loser tree
    (``_kway.c`` -> ``_kway.so`` via gcc), O(N log K), one element consumed per
    comparison round. K input buffers + 1 output buffer are the only heap
    storage (streams are ``_IONBF``); budget = ``(K+1)*bufcap*8 + O(K)``. This
    is the intended algorithm and the only one used for measured runs.

  * ``batch`` (explicit opt-in for hosts without a C compiler): bounded
    ``readinto`` + per-round ``np.unique`` over the ``<= cut`` portion,
    O(N log B). Its per-round ``parts``/``concatenate``/``unique`` temporaries
    are budgeted (a ~6x-per-key reserve), so it is also hard-bounded, but it is
    NOT selected automatically: ``auto`` raises if the C kernel cannot build,
    rather than silently substituting a different algorithm.

If the fan-in K exceeds what the budget serves at the minimum buffer, the merge
runs in STAGED passes of bounded fan-in (bounded RAM; intermediates spill to
disk beside the output). A budget too small for a 2-way merge is rejected.
"""
from __future__ import annotations

import logging
import os
import struct
import tempfile
import threading
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_ITEMSIZE = 8
_MIN_BUF_KEYS = 256           # per-input floor; below this the fan-in is staged
_CURSOR_OVERHEAD = 512        # bytes/input for python/C bookkeeping (+ FILE struct)
_BATCH_RESERVE = 6            # per-key working-set multiple for the batch backend
_COPY_BLOCK_MAX = 1 << 21     # cap the K==1 copy buffer (keys) even if budget is huge
_HERE = Path(__file__).resolve().parent
_LIB = None                   # cached ctypes fn, or False (unavailable)
_LIB_LOCK = threading.Lock()

last_algo: str | None = None  # backend used by the most recent top-level call


class _NeedStage(Exception):
    """Fan-in too large for the budget at this level; caller must stage."""


# --------------------------------------------------------------------------- #
# Budget math (single source of truth)
# --------------------------------------------------------------------------- #
def _bufcap_lt(ram_budget: int, k: int) -> int:
    """Per-input buffer keys for a compiled K-way pass: K input + 1 output
    buffer + O(K) cursors within budget. -1 if k can't be served at the floor."""
    avail = ram_budget - k * _CURSOR_OVERHEAD
    if avail <= 0:
        return -1
    cap = int(avail // ((k + 1) * _ITEMSIZE))
    return cap if cap >= _MIN_BUF_KEYS else -1


def _bufcap_batch(ram_budget: int, k: int) -> int:
    """Per-input buffer keys for the batch pass. Peak working set per round is
    the K resident buffers plus ``parts``/``concatenate``/``unique`` copies of
    the ``<= cut`` portion; ``_BATCH_RESERVE`` bounds that whole set."""
    avail = ram_budget - k * _CURSOR_OVERHEAD
    if avail <= 0:
        return -1
    cap = int(avail // ((_BATCH_RESERVE * k + 1) * _ITEMSIZE))
    return cap if cap >= _MIN_BUF_KEYS else -1


def _max_fanin(cap_fn, ram_budget: int) -> int:
    """Largest K serviceable at the minimum buffer size for ``cap_fn`` (>=1)."""
    k = 1
    while cap_fn(ram_budget, max(2, k * 2)) >= 0:
        k *= 2
    lo, hi = max(1, k), max(2, k * 2)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if cap_fn(ram_budget, mid) >= 0:
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------- #
# Compiled loser-tree backend (gcc + ctypes)
# --------------------------------------------------------------------------- #
def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        try:
            import ctypes
            so, src = _HERE / "_kway.so", _HERE / "_kway.c"
            if (not so.exists()) or src.stat().st_mtime > so.stat().st_mtime:
                # Build to a unique temp, then atomically publish, so concurrent
                # workers never load a half-written or truncated .so.
                fd, tmp_so = tempfile.mkstemp(dir=str(_HERE), prefix="_kway.",
                                              suffix=".so.tmp")
                os.close(fd)
                try:
                    subprocess.run(
                        ["gcc", "-O3", "-shared", "-fPIC", "-o", tmp_so, str(src)],
                        check=True, capture_output=True, timeout=180,
                    )
                    os.replace(tmp_so, so)
                finally:
                    if os.path.exists(tmp_so):
                        os.unlink(tmp_so)
            lib = ctypes.CDLL(str(so))
            fn = lib.kway_merge_unique
            fn.restype = ctypes.c_int64
            fn.argtypes = [
                ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_char_p,
                ctypes.c_int64, ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
            ]
            _LIB = fn
        except Exception as e:  # noqa: BLE001
            logger.warning("bounded_merge: compiled loser-tree unavailable (%s)", e)
            _LIB = False
    return _LIB


def _pass_losertree(paths, out_path, ram_budget) -> tuple[int, int, int]:
    import ctypes
    fn = _load_lib()
    if not fn:
        raise RuntimeError("losertree backend unavailable")
    K = len(paths)
    cap = _bufcap_lt(ram_budget, K)
    if cap < 0:
        raise _NeedStage()
    arr = (ctypes.c_char_p * K)(*[str(p).encode() for p in paths])
    mn, mx = ctypes.c_int64(0), ctypes.c_int64(0)
    r = fn(arr, K, str(out_path).encode(), ctypes.c_int64(cap),
           ctypes.byref(mn), ctypes.byref(mx))
    if r == -2:
        raise _NeedStage()
    if r < 0:
        raise RuntimeError(f"kway_merge_unique failed (code {r})")
    return int(r), (int(mn.value) if r else 0), (int(mx.value) if r else 0)


# --------------------------------------------------------------------------- #
# Batch backend (readinto + per-round np.unique); explicit opt-in only
# --------------------------------------------------------------------------- #
def _pass_batch(paths, out_path, ram_budget) -> tuple[int, int, int]:
    import numpy as np
    K = len(paths)
    cap = _bufcap_batch(ram_budget, K)
    if cap < 0:
        raise _NeedStage()
    files = [open(p, "rb", buffering=0) for p in paths]
    bufs = [np.empty(cap, dtype=np.int64) for _ in range(K)]
    n = [0] * K
    eof = [False] * K

    def topup(i):
        if eof[i] or n[i] >= cap:
            return
        mv = memoryview(bufs[i]).cast("B")
        want = (cap - n[i]) * _ITEMSIZE
        got = files[i].readinto(mv[n[i] * _ITEMSIZE: n[i] * _ITEMSIZE + want])
        if not got:
            eof[i] = True
        else:
            n[i] += got // _ITEMSIZE

    total, mn, mx, carry = 0, None, None, None
    tmp = Path(out_path).with_name(Path(out_path).name + ".tmp")
    try:
        for i in range(K):
            topup(i)
        with open(tmp, "wb", buffering=0) as fo:
            while True:
                active = [i for i in range(K) if n[i] > 0]
                if not active:
                    break
                cut = min(int(bufs[i][n[i] - 1]) for i in active)
                parts = []
                for i in active:
                    h = int(np.searchsorted(bufs[i][:n[i]], cut, side="right"))
                    if h > 0:
                        parts.append(bufs[i][:h].copy())
                        rem = n[i] - h
                        if rem:
                            bufs[i][:rem] = bufs[i][h:n[i]]
                        n[i] = rem
                batch = np.unique(np.concatenate(parts)) if len(parts) > 1 else parts[0]
                if carry is not None:
                    batch = batch[batch > carry]
                if batch.size:
                    if mn is None:
                        mn = int(batch[0])
                    carry = mx = int(batch[-1])
                    _write_all(fo, np.ascontiguousarray(batch))
                    total += int(batch.size)
                for i in active:
                    topup(i)
        os.replace(tmp, out_path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        for f in files:
            f.close()
    return (0, 0, 0) if total == 0 else (total, mn, mx)


# --------------------------------------------------------------------------- #
# Helpers + staged driver + public entry
# --------------------------------------------------------------------------- #
def _write_all(fo, data) -> None:
    """Write every byte of ``data`` to a raw (unbuffered) file, looping over
    short writes; raise if a write makes no progress."""
    mv = memoryview(data).cast("B")
    off, total = 0, mv.nbytes
    while off < total:
        w = fo.write(mv[off:])
        if not w:
            raise OSError("short write: no progress")
        off += w


def _write_empty(out_path: Path) -> None:
    tmp = out_path.with_name(out_path.name + ".tmp")
    open(tmp, "wb").close()
    os.replace(tmp, out_path)


def _copy_stream(src: Path, out_path: Path, ram_budget: int) -> tuple[int, int, int]:
    """Publish an already sorted-unique single input: bounded, unbuffered raw
    byte copy (buffer <= ram_budget); min/max read directly from the ends."""
    size = src.stat().st_size
    if size < _ITEMSIZE:
        _write_empty(out_path)
        return 0, 0, 0
    block_keys = max(1, min(_COPY_BLOCK_MAX, ram_budget // _ITEMSIZE))
    ba = bytearray(block_keys * _ITEMSIZE)
    mv = memoryview(ba)
    tmp = out_path.with_name(out_path.name + ".tmp")
    try:
        with open(src, "rb", buffering=0) as fi, open(tmp, "wb", buffering=0) as fo:
            mn = struct.unpack("<q", fi.read(_ITEMSIZE))[0]
            fi.seek(0)
            while True:
                got = fi.readinto(mv)
                if not got:
                    break
                _write_all(fo, mv[:got])
            fi.seek(-_ITEMSIZE, os.SEEK_END)
            mx = struct.unpack("<q", fi.read(_ITEMSIZE))[0]
        os.replace(tmp, out_path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return size // _ITEMSIZE, mn, mx


def _staged(paths, out_path, ram_budget, fmax, pass_fn) -> tuple[int, int, int]:
    runs = [Path(p) for p in paths]
    scratch = Path(out_path).parent
    tmps: list[Path] = []
    try:
        while len(runs) > fmax:
            nxt: list[Path] = []
            for g in range(0, len(runs), fmax):
                group = runs[g:g + fmax]
                if len(group) == 1:
                    nxt.append(group[0])
                    continue
                fd, name = tempfile.mkstemp(dir=scratch, prefix="_stage_", suffix=".bin")
                os.close(fd)
                t = Path(name)
                pass_fn(group, t, ram_budget)
                nxt.append(t)
                tmps.append(t)
            runs = nxt
        return pass_fn(runs, out_path, ram_budget)
    finally:
        for t in tmps:
            try:
                t.unlink()
            except FileNotFoundError:
                pass


def bounded_merge_unique(paths, out_path, ram_budget, *, algo="auto") -> tuple[int, int, int]:
    """Union K sorted-unique int64 ``.bin`` files into one sorted-unique file
    under a hard ``ram_budget`` (bytes). ``algo``: 'auto' (compiled loser-tree;
    raises if the C kernel cannot build), 'losertree', or 'batch' (opt-in).
    Returns ``(count, min, max)``; sets module ``last_algo`` to the backend."""
    global last_algo
    paths = [Path(p) for p in paths if Path(p).exists() and Path(p).stat().st_size > 0]
    out_path = Path(out_path)
    if not paths:
        _write_empty(out_path)
        last_algo = "empty"
        return 0, 0, 0

    floor = _MIN_BUF_KEYS * _ITEMSIZE
    if ram_budget < floor:
        raise ValueError(f"ram_budget={ram_budget} B below minimum {floor} B (one buffer)")

    if len(paths) == 1:
        last_algo = "copy"
        return _copy_stream(paths[0], out_path, ram_budget)

    if algo == "batch":
        pass_fn, cap_fn, used = _pass_batch, _bufcap_batch, "batch"
    elif algo in ("auto", "losertree"):
        if not _load_lib():
            raise RuntimeError(
                "compiled loser-tree kernel (_kway.c) unavailable: a C compiler "
                "(gcc) is required. Install one, or pass algo='batch' explicitly."
            )
        pass_fn, cap_fn, used = _pass_losertree, _bufcap_lt, "losertree"
    else:
        raise ValueError(f"unknown algo {algo!r}")

    fmax = _max_fanin(cap_fn, ram_budget)
    if fmax < 2:
        raise ValueError(f"ram_budget={ram_budget} B too small for a 2-way {used} merge")

    last_algo = used
    logger.info("bounded_merge_unique: backend=%s K=%d fmax=%d budget=%d",
                used, len(paths), fmax, ram_budget)

    if len(paths) > fmax:
        return _staged(paths, out_path, ram_budget, fmax, pass_fn)
    return pass_fn(paths, out_path, ram_budget)
