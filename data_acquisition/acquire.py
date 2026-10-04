"""Root-only command line interface for explicit market-data acquisition."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from . import contracts


SCOPES = ("backtest", "live")
DATASETS = ("prices", "benchmark", "sectors", "shares", "all")


def _handler_mapping() -> dict[
    tuple[str, str], Callable[[contracts.CommandRequest], int]
]:
    """Return the fixed root dispatch matrix without importing domains eagerly."""

    from backtest.acquisition_handlers import acquisition_handlers as backtest_handlers
    from live.acquisition_handlers import acquisition_handlers as live_handlers

    handlers = {
        **{
            ("backtest", dataset): handler
            for dataset, handler in backtest_handlers().items()
        },
        **{
            ("live", dataset): handler
            for dataset, handler in live_handlers().items()
        },
    }
    expected = {(scope, dataset) for scope in SCOPES for dataset in DATASETS}
    if set(handlers) != expected:
        raise RuntimeError(
            f"Acquisition handler map contains {sorted(handlers)}, expected "
            f"{sorted(expected)}"
        )
    return handlers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire or validate raw market data only. This command never "
            "prepares data or runs an analysis."
        )
    )
    parser.add_argument("scope", choices=SCOPES)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument(
        "--tickers-file",
        type=Path,
        help="Optional one-column ticker/asset request file for shares or all.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-request terminal provider identities and update canonical state.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact provider plan without contacting any provider.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if args.tickers_file is not None and args.dataset not in {"shares", "all"}:
        parser.error("--tickers-file is valid only for shares or all")
    request = contracts.CommandRequest(
        scope=args.scope,
        dataset=args.dataset,
        tickers_file=(
            args.tickers_file.expanduser()
            if args.tickers_file is not None
            else None
        ),
        refresh=bool(args.refresh),
        dry_run=bool(args.dry_run),
        project_root=project_root,
    )
    return _handler_mapping()[(request.scope, request.dataset)](request)


def _entrypoint() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print("ERROR: acquisition interrupted; completed checkpoints were preserved.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess use.
    raise SystemExit(_entrypoint())


__all__ = [
    "DATASETS",
    "SCOPES",
    "build_parser",
    "main",
]
