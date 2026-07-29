"""Quickstart: ingest one release, print the IngestReport.

Run `python examples/make_toy_data.py` first (or just run this script; it
regenerates the toy data on demand). State and outputs land under
`examples/toy_state/` and `examples/toy_out/` (gitignored scratch dirs).
"""

from __future__ import annotations

from pathlib import Path

from make_toy_data import OUT_DIR, generate

from puffer import Deduper, PufferConfig

HERE = Path(__file__).parent
STATE_DIR = HERE / "toy_state" / "01_quickstart"
OUTPUT_DIR = HERE / "toy_out" / "01_quickstart" / "release0"


def main() -> None:
    if not OUT_DIR.exists():
        generate()

    dd = Deduper(STATE_DIR, PufferConfig(num_bands=8, tier_fanout=4))

    report = dd.ingest(
        [str(OUT_DIR / "release0" / "*.parquet")],
        dataset="release0",
        output_dir=str(OUTPUT_DIR),
    )

    print(f"dataset:          {report.dataset}")
    print(f"n_input:          {report.n_input}")
    print(f"n_exact_removed:  {report.n_exact_removed}")
    print(f"n_within_removed: {report.n_within_removed}")
    print(f"n_cross_removed:  {report.n_cross_removed}")
    print(f"n_output:         {report.n_output}")
    print(f"output_files:     {len(report.output_files)} file(s) -> {OUTPUT_DIR}")
    print(f"probes_done:      {report.probes_done} / probes_scheduled: {report.probes_scheduled}")
    print(f"elapsed_s:        {report.elapsed_s:.3f}")


if __name__ == "__main__":
    main()
