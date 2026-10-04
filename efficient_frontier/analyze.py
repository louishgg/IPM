"""Diagnose the first formation portfolio of an explicit saved live replay."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .analysis import run_analysis
from .contracts import DiagnosticSettings


def build_parser() -> argparse.ArgumentParser:
    defaults = DiagnosticSettings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Saved live strategy run directory")
    parser.add_argument("--lookback", type=int, default=defaults.lookback,
                        help="Calendar-month lookback (default: %(default)s; minimum: 12)")
    parser.add_argument("--floor-multiplier", type=float, default=defaults.floor_multiplier,
                        help="Default: %(default)s times original absolute weight")
    parser.add_argument("--cap-multiplier", type=float, default=defaults.cap_multiplier,
                        help="Default: %(default)s times original absolute weight")
    parser.add_argument("--brti", type=float, default=defaults.brti,
                        help="Behavioral risk tolerance in [1, 7] (default: %(default)s)")
    parser.add_argument("--output-root", type=Path, help="Default: outputs/efficient_frontier; replaces the current diagnostic per strategy")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = DiagnosticSettings(args.lookback, args.floor_multiplier, args.cap_multiplier, args.brti)
        return run_analysis(args.run_dir, settings, args.output_root)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
