"""Ray executor: same ingest as 01_quickstart.py, routed through Ray.

Signature computation parallelizes over input files and screening
parallelizes over row-chunks as Ray tasks; index appends/compaction still run
on the driver. Requires the optional `ray` extra (`pip install -e ".[ray]"`).
"""

from __future__ import annotations

from pathlib import Path

try:
    import ray  # noqa: F401
except ImportError:
    ray = None

from make_toy_data import OUT_DIR, generate

from puffer import Deduper, PufferConfig

HERE = Path(__file__).parent
STATE_DIR = HERE / "toy_state" / "04_ray_cluster"
OUTPUT_DIR = HERE / "toy_out" / "04_ray_cluster" / "release0"


def main() -> None:
    if ray is None:
        print(
            "ray is not installed -- this example needs the optional Ray executor.\n"
            'Install it with: pip install -e ".[ray]"'
        )
        return

    if not OUT_DIR.exists():
        generate()

    dd = Deduper(STATE_DIR, PufferConfig(num_bands=8, tier_fanout=4), executor="ray")

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
    print(f"probes_done:      {report.probes_done} / probes_scheduled: {report.probes_scheduled}")
    print(f"elapsed_s:        {report.elapsed_s:.3f}")
    print("(screening result under the Ray executor is identical to the local executor --")
    print(" each screening task still runs the full early-stop sequence per chunk.)")


if __name__ == "__main__":
    main()
