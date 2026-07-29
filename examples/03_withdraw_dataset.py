"""Withdraw a dataset: ingest three releases, withdraw the middle one.

Shows that withdrawing `release1` removes its own artifacts and index
contribution (WithdrawReport), while `release0` and `release2` outputs are
left exactly as they were written — withdrawal does not retroactively
re-derive other releases' outputs (see README "Withdrawal").
"""

from __future__ import annotations

from pathlib import Path

from make_toy_data import OUT_DIR, generate

from puffer import Deduper, PufferConfig

HERE = Path(__file__).parent
STATE_DIR = HERE / "toy_state" / "03_withdraw_dataset"
OUT_ROOT = HERE / "toy_out" / "03_withdraw_dataset"


def _output_file_snapshot(out_root: Path) -> dict[str, set[str]]:
    return {
        release_dir.name: {p.name for p in release_dir.glob("*.parquet")}
        for release_dir in out_root.iterdir()
        if release_dir.is_dir()
    }


def main() -> None:
    if not OUT_DIR.exists():
        generate()

    dd = Deduper(STATE_DIR, PufferConfig(num_bands=8, tier_fanout=4))

    for release in ("release0", "release1", "release2"):
        report = dd.ingest(
            [str(OUT_DIR / release / "*.parquet")],
            dataset=release,
            output_dir=str(OUT_ROOT / release),
        )
        print(f"ingested {release}: n_output={report.n_output}")

    before = _output_file_snapshot(OUT_ROOT)
    print(f"\ndatasets before withdrawal: {dd.datasets()}")

    wreport = dd.withdraw("release1")
    print("\nWithdrawReport:")
    print(f"  dataset:        {wreport.dataset}")
    print(f"  bands_o1:       {wreport.bands_o1}")
    print(f"  bands_rebuilt:  {wreport.bands_rebuilt}")
    print(f"  bloom_rebuilt:  {wreport.bloom_rebuilt}")
    print(f"  elapsed_s:      {wreport.elapsed_s:.3f}")

    print(f"\ndatasets after withdrawal: {dd.datasets()}")

    after = _output_file_snapshot(OUT_ROOT)
    for release in ("release0", "release2"):
        assert before[release] == after.get(release), (
            f"{release} output files changed after withdrawing release1 -- "
            "they should be untouched"
        )
        print(f"{release} output files unchanged: {sorted(after[release])}")

    release1_dir = OUT_ROOT / "release1"
    still_has_state = (STATE_DIR / "datasets" / "release1").exists()
    print(
        f"release1 own outputs still on disk (withdraw does not delete them, "
        f"only index state): {release1_dir.exists()}"
    )
    print(f"state/datasets/release1 deleted by withdraw: {not still_has_state}")


if __name__ == "__main__":
    main()
