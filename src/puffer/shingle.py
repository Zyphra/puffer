"""Text shingling for MinHash signatures.

Two shingle kinds:

  word: casefold + whitespace-split, then join sliding windows of
        ``ngram_size`` tokens back into space-separated strings.
  char: casefold, then collapse ALL whitespace runs to a single space
        (so indentation/newline differences don't inflate similarity
        between structurally similar but semantically unrelated docs),
        then slide a window of ``ngram_size`` characters.

Both fall back to a single shingle covering the whole (short) text when
there are fewer tokens/chars than ``ngram_size`` — ``max(1, n - size + 1)``
windows, never zero, so short docs still get a signature instead of an
empty shingle set.
"""

from __future__ import annotations

import re


def make_shingles(text: str, ngram_type: str = "word", ngram_size: int = 5) -> list[str]:
    """Generate n-gram shingles from ``text``.

    For char n-grams, whitespace is collapsed before shingling so that
    indentation/newline patterns do not become high-frequency "stop
    shingles" that inflate Jaccard similarity between structurally
    similar but semantically unrelated documents.
    """
    if ngram_type == "word":
        tokens = text.casefold().split()
        return [
            " ".join(tokens[i : i + ngram_size])
            for i in range(max(1, len(tokens) - ngram_size + 1))
        ]
    elif ngram_type == "char":
        t = re.sub(r"\s+", " ", text.casefold()).strip()
        return [t[i : i + ngram_size] for i in range(max(1, len(t) - ngram_size + 1))]
    else:
        raise ValueError(f"Unknown ngram_type: {ngram_type}")
