"""Single-strategy expanding-validation research search CLI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from portfolio_core.runtime_config import apply_runtime_settings
from portfolio_core.strategies import available_strategy_ids

from .config import DEFAULT_CONFIG
from .data_loading import load_backtest_data
from .paths import GRIDS_DIRECTORY


_DEFAULT_WORKERS = 6


def _positive_workers(value: str) -> int:
    workers = int(value)
    if workers < 1:
        raise argparse.ArgumentTypeError("workers must be a positive integer")
    return workers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search one registered strategy on expanding validation folds. "
            "The test window is never evaluated."
        )
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=available_strategy_ids(),
        help="Registered strategy family to search.",
    )
    parser.add_argument(
        "--grid-file",
        type=Path,
        help="Optional complete JSON grid; otherwise use the selected strategy's grid.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_workers,
        default=_DEFAULT_WORKERS,
        help="Candidate worker processes (default: %(default)s); use 1 for serial execution.",
    )
    parser.add_argument("--preflight-only", action="store_true", help="Generate research coverage diagnostics without evaluating candidates.")
    parser.add_argument("--output-dir", type=Path, help="Research resumable output/checkpoint directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    from .research_grid import read_grid
    from .research_search import run_research_search
    grid_path = args.grid_file or GRIDS_DIRECTORY / f"{args.strategy}_grid.json"
    if not grid_path.is_file():
        parser.error(f"Missing complete research grid: {grid_path}")
    grid = read_grid(grid_path)
    if grid["strategy_id"] != args.strategy:
        parser.error("grid strategy_id must match --strategy")
    apply_runtime_settings()
    data = load_backtest_data(
        DEFAULT_CONFIG.market, DEFAULT_CONFIG.paths, development_only=True
    )
    run_research_search(
        data, grid,
        args.output_dir or Path("outputs/backtest/searches") / args.strategy,
        workers=args.workers, preflight_only=args.preflight_only,
    )


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "main"]
