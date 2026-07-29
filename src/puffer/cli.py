"""Command-line entry point: ``puffer ingest|withdraw|stats``.

A thin argparse wrapper over :class:`puffer.pipeline.Deduper` — no logic
lives here beyond argument parsing and result formatting.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="puffer", description=__doc__)
    parser.add_argument(
        "--state-dir", required=True, type=Path, help="PUFFER state directory",
    )
    parser.add_argument(
        "--executor", choices=("local", "ray"), default="local",
        help="execution backend for signature computation + screening",
    )
    parser.add_argument(
        "--ray-max-in-flight", type=int, default=None, metavar="N",
        help=(
            "cluster-wide cap for one-CPU Ray tasks; 0 or omitted uses the "
            "live cluster CPU slots"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable INFO logging")

    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="ingest one release")
    ingest_p.add_argument(
        "inputs", nargs="+", help="input parquet files or glob patterns",
    )
    ingest_p.add_argument("--dataset", required=True, help="dataset tag for this release")
    ingest_p.add_argument("--output-dir", required=True, type=Path, help="deduplicated output directory")
    ingest_p.add_argument("--text-column", default=None, help="text column name (default: config default)")

    withdraw_p = sub.add_parser("withdraw", help="withdraw a previously ingested dataset")
    withdraw_p.add_argument("--dataset", required=True, help="dataset tag to withdraw")
    withdraw_p.add_argument(
        "--purge-outputs", action="store_true",
        help="also delete the withdrawn dataset's already-written output files",
    )

    sub.add_parser("stats", help="print index + dataset summary")

    return parser


def _dataclass_to_dict(obj) -> dict:
    import dataclasses

    return dataclasses.asdict(obj)


def print_logo() -> None:
    logo_path = Path(__file__).with_name("logo.txt")
    try:
        print(logo_path.read_text(encoding="utf-8"), file=sys.stderr)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    print_logo()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    from dataclasses import replace

    from puffer.config import PufferConfig
    from puffer.pipeline import Deduper

    if args.ray_max_in_flight is not None and args.ray_max_in_flight < 0:
        parser.error("--ray-max-in-flight must be >= 0")
    config = None
    if args.ray_max_in_flight is not None:
        base = PufferConfig.load(args.state_dir) or PufferConfig()
        config = replace(base, ray_max_in_flight=args.ray_max_in_flight)
    dd = Deduper(args.state_dir, config=config, executor=args.executor)

    if args.command == "ingest":
        report = dd.ingest(
            args.inputs, dataset=args.dataset, output_dir=args.output_dir,
            text_column=args.text_column,
        )
        json.dump(_dataclass_to_dict(report), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.command == "withdraw":
        report = dd.withdraw(args.dataset, purge_outputs=args.purge_outputs)
        json.dump(_dataclass_to_dict(report), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.command == "stats":
        json.dump(dd.stats(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
