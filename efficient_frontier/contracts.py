"""Typed contracts for diagnostics; weights never enter the trading pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

import numpy as np
import pandas as pd


# Explicit schemas also apply when a diagnostic stops before optimization.
METRIC_COLUMNS = (
    "return", "volatility", "utility", "long_count", "short_count", "gross", "net",
    "largest_position", "concentration_hhi", "binding_lower", "binding_upper",
    "below_historical_position_counts",
)
WEIGHT_COLUMNS = ("case", "comparison", "portfolio", "Asset_ID", "weight", "lower_bound", "upper_bound")
SENSITIVITY_COLUMNS = (
    "case", "comparison", "status", "observations", "reference_return", "reference_volatility", "reference_utility",
    *(f"optimized_{name}" for name in METRIC_COLUMNS),
    "utility_gain", "weight_l1_from_reference", "weight_l1_from_primary_optimum",
    "volatility_reduction_at_reference_return",
)


def allocation_rows(
    assets: list[str], weights: np.ndarray, lower: np.ndarray, upper: np.ndarray,
    *, case: str, comparison: str, portfolio: str,
) -> list[dict[str, object]]:
    """Apply the allocation schema without filtering or reordering holdings."""
    return [
        {"case": case, "comparison": comparison, "portfolio": portfolio, "Asset_ID": asset,
         "weight": float(weight), "lower_bound": float(lo), "upper_bound": float(hi)}
        for asset, weight, lo, hi in zip(assets, weights, lower, upper)
    ]


@dataclass(frozen=True)
class DiagnosticSettings:
    lookback: int = 60
    floor_multiplier: float = 0.5
    cap_multiplier: float = 2.0
    brti: float = 5.2

    def __post_init__(self):
        if isinstance(self.lookback, bool) or not isinstance(self.lookback, int) or self.lookback < 12:
            raise ValueError("lookback must be an integer of at least 12 calendar months")
        if not math.isfinite(self.floor_multiplier) or not 0 < self.floor_multiplier <= 1:
            raise ValueError("floor multiplier must be positive and no greater than 1")
        if not math.isfinite(self.cap_multiplier) or self.cap_multiplier < 1:
            raise ValueError("cap multiplier must be finite and no smaller than 1")
        if not math.isfinite(self.brti) or not 1 <= self.brti <= 7:
            raise ValueError("BRTI must lie in [1, 7]")

    @property
    def q(self) -> float:
        return (self.brti - 1) / 6

    @property
    def risk_aversion(self) -> float:
        return 12.5 * math.exp(-0.35 * (self.brti - 1))


@dataclass(frozen=True)
class PortfolioSnapshot:
    strategy_id: str
    strategy_version: str
    rebalance_id: str
    formation_date: pd.Timestamp
    execution_date: pd.Timestamp
    information_cutoff: pd.Timestamp
    sizing_date: pd.Timestamp
    sizing_nav: float
    positions: pd.DataFrame
    sector_neutral: bool
    parameters: dict
    assumptions: dict
    run_dir: Path

    @property
    def assets(self) -> list[str]:
        return self.positions.Asset_ID.tolist()

    @property
    def weights(self) -> np.ndarray:
        return self.positions.Signal_Raw_Target_Weight.to_numpy(dtype=float)


@dataclass(frozen=True)
class HistorySample:
    returns: pd.DataFrame
    missing: pd.DataFrame
    coverage: pd.DataFrame
    calendar_end: pd.Timestamp


@dataclass(frozen=True)
class Moments:
    assets: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray
    diagnostics: dict


@dataclass
class Solution:
    accepted: bool
    weights: np.ndarray | None
    diagnostics: dict
    target_return: float | None = None


@dataclass
class Comparison:
    moments: Moments
    lower: np.ndarray
    upper: np.ndarray
    solutions: dict[str, Solution]
    frontier: list[Solution] = field(default_factory=list)


@dataclass
class DiagnosticResult:
    snapshot: PortfolioSnapshot
    settings: DiagnosticSettings
    history: HistorySample
    comparisons: dict[str, Comparison]
    estimates: dict[str, Moments]
    sensitivity: pd.DataFrame
    sensitivity_weights: pd.DataFrame
    sensitivity_evidence: list[dict]
    provenance: dict
    blockers: list[str] = field(default_factory=list)
