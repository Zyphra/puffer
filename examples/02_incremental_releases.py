"""Incremental releases: ingest release0, release1, release2 in sequence.

Demonstrates that cross-history removals grow as index history accumulates:
release1 is screened against release0's contribution; release2 is screened
against both. Run `make_toy_data.py`'s planted cross-release duplicates make
this visible in `n_cross_removed`.
"""

from __future__ import annotations

from pathlib import Path

from make_toy_data import OUT_DIR, generate

from puffer import Deduper, PufferConfig

HERE = Path(__file__).parent
STATE_DIR = HERE / "toy_state" / "02_incremental_releases"
OUT_ROOT = HERE / "toy_out" / "02_incremental_releases"

RELEASES = ["release0", "release1", "release2"]


def main() -> None:
    if not OUT_DIR.exists():
        generate()

    dd = Deduper(STATE_DIR, PufferConfig(num_bands=8, tier_fanout=4))

    for release in RELEASES:
        report = dd.ingest(
            [str(OUT_DIR / release / "*.parquet")],
            dataset=release,
            output_dir=str(OUT_ROOT / release),
        )
        print(
            f"{report.dataset:>10}: n_input={report.n_input:4d}  "
            f"exact={report.n_exact_removed:3d}  within={report.n_within_removed:3d}  "
            f"cross={report.n_cross_removed:3d}  n_output={report.n_output:4d}  "
            f"probes={report.probes_done}/{report.probes_scheduled}"
        )

    print()
    print(f"datasets in index: {dd.datasets()}")
    print(f"stats: {dd.stats()}")


if __name__ == "__main__":
    main()
