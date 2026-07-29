"""Generate seeded synthetic parquet releases with planted duplicates.

Writes three "releases" (`release0`, `release1`, `release2`) of ~700 rows
each (~2k rows total) to `examples/toy_data/<release>/part-*.parquet`, each
with a single `text` column. Duplicates are planted deliberately so the
other examples have something to dedup:

- **Exact duplicates**: identical text repeated, both within a release (so
  the within-release exact/fuzzy stage has something to catch) and across
  releases (so the cross-history screen has something to catch).
- **Fuzzy duplicates**: a base sentence lightly edited (one word swapped for
  a synonym-shaped filler, or a trailing clause appended) so it shares most
  MinHash shingles with the original but is not byte-identical. Planted both
  within a release and across releases.
- **Unrelated documents**: the majority of rows, generated from an
  independent random sentence template so they should survive dedup
  untouched.

Deterministic: reruns with the same seed regenerate byte-identical data.
"""

from __future__ import annotations

import random
from pathlib import Path

OUT_DIR = Path(__file__).parent / "toy_data"
SEED = 1234

_SUBJECTS = [
    "the otter", "a lighthouse keeper", "the quiet valley", "an old compiler",
    "the harbor market", "a traveling cartographer", "the copper kettle",
    "a mountain pass", "the night train", "a paper factory", "the river guild",
    "a clockwork bird", "the salt flats", "a wandering violinist",
    "the frost garden", "a coastal archive",
]
_VERBS = [
    "quietly rebuilt", "carefully mapped", "slowly restored", "eventually catalogued",
    "briefly abandoned", "steadily expanded", "unexpectedly rerouted", "patiently repaired",
]
_OBJECTS = [
    "the old harbor road", "a forgotten storage index", "the eastern archive",
    "a tangled supply chain", "the winter schedule", "an unlabeled ledger",
    "the coastal survey", "a half-finished map",
]
_TAILS = [
    "before the season changed.", "without telling the others.",
    "after three failed attempts.", "while the storm passed overhead.",
    "long before anyone noticed.", "just after the equinox.",
]

_FUZZY_FILLER = " It took longer than expected."
_SYNONYM_SWAPS = {
    "quietly": "silently",
    "carefully": "attentively",
    "slowly": "gradually",
    "briefly": "momentarily",
    "steadily": "gradually",
    "patiently": "calmly",
}


def _base_sentence(rng: random.Random) -> str:
    return "{} {} {} {}".format(
        rng.choice(_SUBJECTS).capitalize(),
        rng.choice(_VERBS),
        rng.choice(_OBJECTS),
        rng.choice(_TAILS),
    )


def _fuzzify(text: str, rng: random.Random) -> str:
    """Lightly edit `text`: swap a word for a near-synonym and/or append a clause."""
    words = text.split()
    for i, w in enumerate(words):
        bare = w.strip(".")
        if bare in _SYNONYM_SWAPS:
            words[i] = w.replace(bare, _SYNONYM_SWAPS[bare])
            break
    edited = " ".join(words)
    if rng.random() < 0.5:
        edited = edited.rstrip(".") + "." + _FUZZY_FILLER
    return edited


def _write_release(rows: list[str], release_dir: Path, rows_per_file: int = 250) -> list[Path]:
    """Write `rows` as one or more single-column ("text") parquet files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    release_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(rows), rows_per_file):
        chunk = rows[start : start + rows_per_file]
        table = pa.table({"text": chunk})
        path = release_dir / f"part-{start // rows_per_file:03d}.parquet"
        pq.write_table(table, path)
        paths.append(path)
    return paths


def generate(out_dir: Path = OUT_DIR, seed: int = SEED) -> dict[str, list[Path]]:
    """Generate release0/release1/release2 under `out_dir`. Returns {release: [paths]}."""
    rng = random.Random(seed)

    # release0: ~700 unique base docs, with some in-release exact and fuzzy dups planted.
    release0_unique = [_base_sentence(rng) for _ in range(600)]
    release0 = list(release0_unique)
    # in-release exact dups: repeat 40 existing rows verbatim.
    release0 += rng.sample(release0_unique, 40)
    # in-release fuzzy dups: lightly edit 40 existing rows.
    release0 += [_fuzzify(t, rng) for t in rng.sample(release0_unique, 40)]
    rng.shuffle(release0)

    # release1: ~650 new unique docs, plus cross-release exact + fuzzy dups of release0.
    release1_unique = [_base_sentence(rng) for _ in range(560)]
    release1 = list(release1_unique)
    release1 += rng.sample(release0_unique, 45)  # cross exact dup of release0
    release1 += [_fuzzify(t, rng) for t in rng.sample(release0_unique, 45)]  # cross fuzzy dup
    rng.shuffle(release1)

    # release2: ~650 new unique docs, plus cross-release dups spanning release0 AND release1.
    release2_unique = [_base_sentence(rng) for _ in range(560)]
    release2 = list(release2_unique)
    release2 += rng.sample(release0_unique, 25)
    release2 += rng.sample(release1_unique, 25)
    release2 += [_fuzzify(t, rng) for t in rng.sample(release1_unique, 30)]
    rng.shuffle(release2)

    out_dir = Path(out_dir)
    return {
        "release0": _write_release(release0, out_dir / "release0"),
        "release1": _write_release(release1, out_dir / "release1"),
        "release2": _write_release(release2, out_dir / "release2"),
    }


def main() -> None:
    written = generate()
    total = 0
    for release, paths in written.items():
        n_rows = sum(1 for _ in _iter_rows(paths))
        total += n_rows
        print(f"{release}: {len(paths)} file(s), {n_rows} rows -> {paths[0].parent}")
    print(f"total rows written: {total}")


def _iter_rows(paths: list[Path]):
    import pyarrow.parquet as pq

    for p in paths:
        table = pq.read_table(p, columns=["text"])
        yield from table.column("text").to_pylist()


if __name__ == "__main__":
    main()
