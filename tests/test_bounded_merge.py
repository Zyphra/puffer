"""Correctness + bounded-memory behaviour of the external k-way merge.

The compiled loser-tree backend is the default; the batch backend is the
explicit fallback. Both must produce the exact ``np.unique(np.concatenate(...))``
of the inputs, single-pass or staged, across dedup / skew / extreme-value
distributions and tiny budgets that force refills and staging.
"""
import numpy as np
import pytest

import puffer.bounded_merge as bm
from puffer.bounded_merge import (
    _bufcap_batch,
    _bufcap_lt,
    _max_fanin,
    bounded_merge_unique,
)

_ALGOS = ["losertree", "batch"]
# Per-algo budget (bytes) that serves a 2-way merge but not 6-way -> forces
# staging. The batch backend reserves ~6x working set, so it needs more.
_STAGE_BUDGET = {"losertree": 1 << 13, "batch": 30000}
_BIG = 1 << 30


def _write_sorted_unique(path, arr):
    a = np.unique(np.asarray(arr, dtype=np.int64))
    a.tofile(path)
    return a


def _read(path):
    return np.fromfile(path, dtype=np.int64)


def _has(algo):
    if algo == "losertree" and not bm._load_lib():
        pytest.skip("no C compiler for the loser-tree backend")


def _check(tmp_path, inputs, algo, budget=_BIG):
    _has(algo)
    paths = []
    for i, arr in enumerate(inputs):
        p = tmp_path / f"in_{i}.bin"
        _write_sorted_unique(p, arr)
        paths.append(p)
    ref = np.unique(np.concatenate([np.asarray(a, dtype=np.int64) for a in inputs])) \
        if inputs else np.empty(0, dtype=np.int64)
    out = tmp_path / "out.bin"
    n, mn, mx = bounded_merge_unique(paths, out, budget, algo=algo)
    got = _read(out)
    assert np.array_equal(got, ref), f"{algo}: merged output != reference"
    assert n == ref.size
    if ref.size:
        assert (mn, mx) == (int(ref[0]), int(ref[-1]))
    return n, mn, mx


@pytest.mark.parametrize("algo", _ALGOS)
def test_disjoint(tmp_path, algo):
    _check(tmp_path, [[1, 2, 3], [4, 5, 6], [7, 8, 9]], algo)


@pytest.mark.parametrize("algo", _ALGOS)
def test_overlap_dedup(tmp_path, algo):
    _check(tmp_path, [[1, 4, 7, 10], [2, 4, 6, 10], [3, 4, 9, 10]], algo)


@pytest.mark.parametrize("algo", _ALGOS)
def test_all_identical(tmp_path, algo):
    keys = list(range(500))
    _check(tmp_path, [keys, keys, keys, keys], algo)


@pytest.mark.parametrize("algo", _ALGOS)
def test_random_many(tmp_path, algo):
    rng = np.random.default_rng(42)
    inputs = [rng.integers(0, 5000, size=rng.integers(200, 1200)) for _ in range(7)]
    _check(tmp_path, inputs, algo)


@pytest.mark.parametrize("algo", _ALGOS)
def test_skew(tmp_path, algo):
    big = np.arange(0, 6000, 3)
    _check(tmp_path, [big, [1, 40000], [2, 50000]], algo)


@pytest.mark.parametrize("algo", _ALGOS)
def test_extreme_values(tmp_path, algo):
    lo, hi = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    _check(tmp_path, [[lo, -1, 0, hi], [lo, 1, hi], [0, 7]], algo)


@pytest.mark.parametrize("algo", _ALGOS)
def test_empty_inputs_mixed(tmp_path, algo):
    _check(tmp_path, [[], [1, 2], [], [3, 4], []], algo)


@pytest.mark.parametrize("algo", _ALGOS)
def test_tiny_budget_forces_refills(tmp_path, algo):
    # A budget serving the fan-in but far smaller than the data -> many rounds.
    rng = np.random.default_rng(7)
    inputs = [rng.integers(0, 100000, size=4000) for _ in range(3)]
    _check(tmp_path, inputs, algo, budget=1 << 16)


@pytest.mark.parametrize("algo", _ALGOS)
def test_staged_when_fanin_exceeds_budget(tmp_path, algo):
    _has(algo)
    budget = _STAGE_BUDGET[algo]
    cap_fn = _bufcap_lt if algo == "losertree" else _bufcap_batch
    inputs = [list(range(i, 600, 6)) for i in range(6)]      # K=6, overlapping ranges
    assert _max_fanin(cap_fn, budget) < len(inputs), "budget should force staging"
    _check(tmp_path, inputs, algo, budget=budget)


@pytest.mark.parametrize("algo", _ALGOS)
def test_single_input_is_copy(tmp_path, algo):
    _has(algo)
    p = tmp_path / "solo.bin"
    a = _write_sorted_unique(p, [3, 1, 4, 1, 5, 9, 2, 6])
    out = tmp_path / "out.bin"
    n, mn, mx = bounded_merge_unique([p], out, _BIG, algo=algo)
    assert np.array_equal(_read(out), a)
    assert (n, mn, mx) == (a.size, int(a[0]), int(a[-1]))
    assert bm.last_algo == "copy"


@pytest.mark.parametrize("algo", _ALGOS)
def test_all_empty_returns_zero(tmp_path, algo):
    _has(algo)
    ps = []
    for i in range(3):
        p = tmp_path / f"e_{i}.bin"
        p.write_bytes(b"")
        ps.append(p)
    out = tmp_path / "out.bin"
    assert bounded_merge_unique(ps, out, _BIG, algo=algo) == (0, 0, 0)
    assert _read(out).size == 0


def test_budget_below_floor_raises(tmp_path):
    p1 = _write_and_path(tmp_path, "a.bin", [1, 2, 3])
    p2 = _write_and_path(tmp_path, "b.bin", [4, 5, 6])
    with pytest.raises(ValueError):
        bounded_merge_unique([p1, p2], tmp_path / "o.bin", 8, algo="batch")


def test_losertree_and_batch_agree(tmp_path):
    _has("losertree")
    rng = np.random.default_rng(123)
    inputs = [rng.integers(0, 20000, size=rng.integers(500, 2500)) for _ in range(5)]
    paths = [_write_and_path(tmp_path, f"in_{i}.bin", a) for i, a in enumerate(inputs)]
    o_lt, o_b = tmp_path / "lt.bin", tmp_path / "b.bin"
    r_lt = bounded_merge_unique(paths, o_lt, _BIG, algo="losertree")
    r_b = bounded_merge_unique(paths, o_b, _BIG, algo="batch")
    assert r_lt == r_b
    assert np.array_equal(_read(o_lt), _read(o_b))


def test_auto_prefers_losertree_when_available(tmp_path):
    p1 = _write_and_path(tmp_path, "a.bin", [1, 2, 3])
    p2 = _write_and_path(tmp_path, "b.bin", [3, 4, 5])
    out = tmp_path / "o.bin"
    if bm._load_lib():
        bounded_merge_unique([p1, p2], out, _BIG, algo="auto")
        assert bm.last_algo == "losertree"
    else:
        with pytest.raises(RuntimeError):
            bounded_merge_unique([p1, p2], out, _BIG, algo="auto")


def _write_and_path(tmp_path, name, arr):
    p = tmp_path / name
    _write_sorted_unique(p, arr)
    return p
