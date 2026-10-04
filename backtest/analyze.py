"""CLI for prepared-only backtest analysis and attribution."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from portfolio_core.strategies import (
    available_strategy_ids,
    build_strategy_from_file,
)

from .analysis import run_analysis
from .config import DEFAULT_CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run backtest analysis from prepared CSV inputs only. "
            "This command never downloads or prepares data."
        )
    )
    parser.add_argument(
        "stage",
        choices=("strategy", "brinson", "all"),
        help="Analysis stage to execute.",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=available_strategy_ids(),
        help="Registered strategy family to analyze.",
    )
    parser.add_argument(
        "--params-file",
        type=Path,
        help="Complete JSON parameter packet for strategy or all.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stage == "brinson" and args.params_file is not None:
        parser.error("--params-file is not accepted for the brinson stage")
    strategy = (
        None if args.stage == "brinson"
        else build_strategy_from_file(args.strategy, args.params_file)
    )
    config = replace(
        DEFAULT_CONFIG,
        strategy=strategy,
        paths=DEFAULT_CONFIG.paths.for_strategy(args.strategy),
    )
    run_analysis(args.stage, config=config)


if __name__ == "__main__":
    main()
