"""Build strategy-independent backtest sector-consumer requirements."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from portfolio_core.dates import month_end_index
from portfolio_core.portfolio_lifecycle import eligible_assets_for_interval
from portfolio_core.sector_assignments import (
    ASSIGNMENT_REQUIREMENT_COLUMNS,
    validate_sector_assignment_requirements,
)

from .config import BACKTEST_WARMUP_MONTHS, DEFAULT_CONFIG, BacktestMarketConfig
from .membership_resolution import build_membership_resolution_outputs
from .paths import BacktestPaths


_MEMBERSHIP_ELIGIBILITY_STATUSES = (
    "eligible_priced",
    "eligible_unavailable",
)


def _validated_resolution(resolution: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "Date",
        "Asset_ID",
        "Source_Ticker",
        "Eligibility_Status",
    }
    missing = sorted(required_columns - set(resolution.columns))
    if missing:
        raise ValueError(f"Membership resolution is missing columns {missing}")
    result = resolution.copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="raise").dt.normalize()
    return result


def active_sector_requirements_from_resolution(
    resolution: pd.DataFrame,
    market_config: BacktestMarketConfig = DEFAULT_CONFIG.market,
) -> pd.DataFrame:
    """Return the conservative active-member snapshot-acquisition superset.

    This membership-only graph is used before prepared prices and lifecycle
    tables necessarily exist.  Final prepared assignments use
    :func:`consumer_sector_requirements_from_resolution` instead.
    """

    resolution = _validated_resolution(resolution)
    dates = month_end_index(
        market_config.start_date,
        market_config.end_date,
    )[BACKTEST_WARMUP_MONTHS:-1]
    active = resolution.loc[
        resolution["Date"].isin(dates)
        & resolution["Asset_ID"].astype(str).ne("")
        & resolution["Eligibility_Status"].isin(
            _MEMBERSHIP_ELIGIBILITY_STATUSES
        ),
        ["Date", "Asset_ID", "Source_Ticker"],
    ].rename(columns={"Date": "As_Of_Date"})
    return validate_sector_assignment_requirements(
        active.loc[:, ASSIGNMENT_REQUIREMENT_COLUMNS]
    )


def consumer_sector_requirements_from_resolution(
    resolution: pd.DataFrame,
    data_close: pd.DataFrame,
    *,
    security_events: pd.DataFrame | None = None,
    security_event_legs: pd.DataFrame | None = None,
    security_event_sources: pd.DataFrame | None = None,
    market_config: BacktestMarketConfig = DEFAULT_CONFIG.market,
) -> pd.DataFrame:
    """Return exact-date rows that a strategy or Brinson can truly consume.

    The filter is independent of signal formulas and selected parameters.  A
    nominal member is required only when it has a valid start price, has not
    already been extinguished, and can be valued through the next period by a
    price or approved corporate action.
    """

    resolution = _validated_resolution(resolution)
    dates = month_end_index(
        market_config.start_date,
        market_config.end_date,
    )[BACKTEST_WARMUP_MONTHS:]
    lifecycle = SimpleNamespace(
        data_close=data_close,
        security_events=(
            pd.DataFrame() if security_events is None else security_events
        ),
        security_event_legs=(
            pd.DataFrame()
            if security_event_legs is None
            else security_event_legs
        ),
        security_event_sources=(
            pd.DataFrame()
            if security_event_sources is None
            else security_event_sources
        ),
    )
    rows: list[dict[str, object]] = []
    for start, end in zip(dates[:-1], dates[1:]):
        candidates = resolution.loc[
            resolution["Date"].eq(start)
            & resolution["Asset_ID"].astype(str).ne("")
            & resolution["Eligibility_Status"].isin(
                _MEMBERSHIP_ELIGIBILITY_STATUSES
            ),
            ["Asset_ID", "Source_Ticker"],
        ]
        eligible = eligible_assets_for_interval(
            lifecycle,
            candidates["Asset_ID"].astype(str),
            pd.Timestamp(start),
            pd.Timestamp(end),
        )
        for candidate in candidates.itertuples(index=False):
            asset_id = str(candidate.Asset_ID)
            if asset_id not in eligible:
                continue
            rows.append(
                {
                    "As_Of_Date": pd.Timestamp(start),
                    "Asset_ID": asset_id,
                    "Source_Ticker": str(candidate.Source_Ticker),
                }
            )
    return validate_sector_assignment_requirements(
        pd.DataFrame(rows, columns=ASSIGNMENT_REQUIREMENT_COLUMNS)
    )


def build_active_sector_requirements(
    market_config: BacktestMarketConfig = DEFAULT_CONFIG.market,
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> pd.DataFrame:
    """Build the conservative snapshot-acquisition graph from raw membership."""

    _, _, resolution, _, _ = build_membership_resolution_outputs(
        market_config,
        paths,
    )
    return active_sector_requirements_from_resolution(resolution, market_config)


__all__ = [
    "active_sector_requirements_from_resolution",
    "build_active_sector_requirements",
    "consumer_sector_requirements_from_resolution",
]
