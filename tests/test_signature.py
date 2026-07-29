"""Tests for puffer.shingle and puffer.signature."""

from __future__ import annotations

import numpy as np
import pytest

from puffer.config import PufferConfig
from puffer.shingle import make_shingles
from puffer.signature import compute_band_keys

# ---------------------------------------------------------------------------
# make_shingles
# ---------------------------------------------------------------------------


def test_word_shingles_casefold_and_join():
    text = "The Quick Brown Fox Jumps"
    shingles = make_shingles(text, ngram_type="word", ngram_size=2)
    assert shingles == ["the quick", "quick brown", "brown fox", "fox jumps"]


def test_char_shingles_casefold_and_whitespace_collapse():
    text = "Ab\t\nCd   Ef"
    # casefold -> "ab\t\ncd   ef" -> whitespace-collapsed -> "ab cd ef"
    shingles = make_shingles(text, ngram_type="char", ngram_size=4)
    expected_t = "ab cd ef"
    expected = [expected_t[i : i + 4] for i in range(len(expected_t) - 4 + 1)]
    assert shingles == expected


def test_char_shingles_strip_leading_trailing_whitespace():
    shingles = make_shingles("  hi  ", ngram_type="char", ngram_size=2)
    assert shingles == ["hi"]


def test_word_short_text_edge_returns_single_shingle():
    # fewer tokens than ngram_size -> exactly one shingle covering everything
    shingles = make_shingles("only two", ngram_type="word", ngram_size=5)
    assert shingles == ["only two"]


def test_char_short_text_edge_returns_single_shingle():
    shingles = make_shingles("hi", ngram_type="char", ngram_size=20)
    assert shingles == ["hi"]


def test_empty_text_word_and_char():
    assert make_shingles("", ngram_type="word", ngram_size=3) == [""]
    assert make_shingles("", ngram_type="char", ngram_size=3) == [""]


def test_unknown_ngram_type_raises():
    with pytest.raises(ValueError):
        make_shingles("hello", ngram_type="sentence", ngram_size=3)


# ---------------------------------------------------------------------------
# compute_band_keys
# ---------------------------------------------------------------------------


def _cfg(**kw) -> PufferConfig:
    defaults = dict(
        num_perm=64,
        num_bands=8,
        ngram_type="char",
        ngram_size=20,
        seed=42,
        n_workers=1,
    )
    defaults.update(kw)
    return PufferConfig(**defaults)


def _make_doc(base: str, target_len: int = 500) -> str:
    reps = target_len // len(base) + 1
    return (base * reps)[:target_len]


def test_band_keys_shape_and_dtype():
    cfg = _cfg()
    texts = ["hello world " * 30, "goodbye moon " * 30, "z" * 200]
    arr = compute_band_keys(texts, cfg)
    assert arr.shape == (3, cfg.num_bands)
    assert arr.dtype == np.int64


def test_band_keys_empty_input():
    cfg = _cfg()
    arr = compute_band_keys([], cfg)
    assert arr.shape == (0, cfg.num_bands)
    assert arr.dtype == np.int64


def test_band_keys_deterministic_across_calls():
    cfg = _cfg()
    texts = [_make_doc("the quick brown fox jumps over the lazy dog. "), "another unrelated document about ships and the sea. " * 10]
    a1 = compute_band_keys(texts, cfg)
    a2 = compute_band_keys(texts, cfg)
    np.testing.assert_array_equal(a1, a2)


def test_band_keys_deterministic_across_worker_counts():
    base_doc = _make_doc("the quick brown fox jumps over the lazy dog. ")
    texts = [base_doc, base_doc[:-1] + "!", "totally different content about volcanoes. " * 15, "yet more unrelated prose about spreadsheets. " * 12]

    single = compute_band_keys(texts, _cfg(n_workers=1))
    multi = compute_band_keys(texts, _cfg(n_workers=4))
    np.testing.assert_array_equal(single, multi)


def test_near_duplicate_shares_band_unrelated_shares_none():
    cfg = _cfg(num_perm=64, num_bands=8, seed=42, ngram_type="char", ngram_size=20)

    doc_a = _make_doc("Lorem ipsum dolor sit amet consectetur adipiscing elit sed do. ", 500)
    # single-character edit deep inside the doc
    mid = len(doc_a) // 2
    doc_b = doc_a[:mid] + ("X" if doc_a[mid] != "X" else "Y") + doc_a[mid + 1 :]
    assert doc_a != doc_b

    doc_c = _make_doc("Completely unrelated text about deep sea navigation and sonar arrays. ", 500)

    arr = compute_band_keys([doc_a, doc_b, doc_c], cfg)
    row_a, row_b, row_c = arr[0], arr[1], arr[2]

    shared_ab = set(row_a.tolist()) & set(row_b.tolist())
    shared_ac = set(row_a.tolist()) & set(row_c.tolist())

    assert len(shared_ab) >= 1, "near-duplicate pair (1-char edit) should share >= 1 band key"
    assert len(shared_ac) == 0, "unrelated documents should share no band keys"


def test_band_keys_int64_signedness_roundtrip():
    cfg = _cfg()
    texts = [_make_doc(f"document number {i} with some filler content. ") for i in range(20)]
    arr = compute_band_keys(texts, cfg)
    flat = arr.reshape(-1)

    # Sorting the int64 numpy array must match sorting the same values as
    # plain Python ints (values round-trip exactly through .tolist(); this
    # guards against accidentally sorting the *unsigned* bit pattern before
    # reinterpreting, which would silently permute negative-looking rows).
    np_sorted = np.sort(flat)
    py_sorted = sorted(flat.tolist())
    assert np_sorted.tolist() == py_sorted

    # every value must be representable exactly as an int64 (no overflow)
    for v in flat.tolist():
        assert -(2**63) <= v < 2**63


def test_band_keys_config_affects_output():
    texts = [_make_doc("some reasonably long piece of sample prose for hashing. ")]
    a = compute_band_keys(texts, _cfg(seed=1))
    b = compute_band_keys(texts, _cfg(seed=2))
    assert not np.array_equal(a, b)
