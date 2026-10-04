"""CLI for deterministic, offline backtest preparation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .preparation_pipeline import run_preparation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("core", "brinson", "all"),
        help="Preparation stage to build or validate.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate existing artifacts without rewriting them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_preparation(args.stage, check=args.check)


if __name__ == "__main__":
    main()
