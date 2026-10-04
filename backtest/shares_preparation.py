"""Offline preparation and loading of reviewed Brinson shares."""
from __future__ import annotations
import pandas as pd

from portfolio_core.io import atomic_write_dataframe
from portfolio_core.shares import load_raw_shares, RAW_SHARES_COLUMNS
from . import acquisition_planning as _acquisition_planning
from .config import BacktestConfig, BacktestBrinsonConfig, DEFAULT_CONFIG
from .data_loading import BacktestDataset, active_pit_asset_ids
from .paths import BacktestSharesPaths
from .share_resolution import required_pairs, validate_prepared_shares


def validate_share_coverage(backtest_data: BacktestDataset, shares_monthly: pd.DataFrame,
                            config: BacktestBrinsonConfig = DEFAULT_CONFIG.brinson) -> list[str]:
    shares = validate_prepared_shares(shares_monthly)
    dates = _acquisition_planning.get_share_month_end_dates(backtest_data, config)
    _, _, membership = _acquisition_planning._build_share_identity_context(backtest_data, dates)
    expected = required_pairs(membership)
    if not shares[["Date", "Asset_ID"]].reset_index(drop=True).equals(expected):
        raise RuntimeError("Prepared shares do not cover the exact required security/month pairs")
    active = set().union(*(set(active_pit_asset_ids(backtest_data, date)) for date in dates))
    unavailable = active - set(backtest_data.data_close.columns)
    declared = set(map(str, backtest_data.unavailable_members))
    if unavailable - declared:
        raise RuntimeError("Active members lack prices without reviewed unavailable identities")
    return sorted(declared)


def prepare_shares_dataset(backtest_data: BacktestDataset,
                           config: BacktestConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Resolve once, validate everything, then save one provenance-bearing table."""
    plan = _acquisition_planning.plan_share_sources(backtest_data, config=config)
    identities = plan.canonical_identities
    raw = None
    if identities:
        path = config.paths.shares.raw_shares_csv
        raw = load_raw_shares(path) if path.exists() else pd.DataFrame(columns=RAW_SHARES_COLUMNS)
    final, unresolved = plan.resolver.resolve(plan.pairs, raw, identities)
    if len(unresolved):
        raise RuntimeError(f"Unresolved reviewed shares ({len(unresolved)}): "
                           f"{unresolved.head(10).to_dict('records')}")
    validate_share_coverage(backtest_data, final, config.brinson)
    atomic_write_dataframe(final, config.paths.shares.final_shares_csv,
                           index=False, date_format="%Y-%m-%d", float_format="%.17g",
                           lineterminator="\n")
    print(f"Saved {len(final)} reviewed share rows: {final.Source.value_counts().to_dict()}")
    return final


def load_prepared_shares(paths: BacktestSharesPaths = DEFAULT_CONFIG.paths.shares,
                         config: BacktestBrinsonConfig = DEFAULT_CONFIG.brinson) -> pd.DataFrame:
    """Analysis reads only the validated prepared table, never raw audit inputs."""
    if not paths.final_shares_csv.exists():
        raise FileNotFoundError("Run `python -m backtest.prepare brinson` first")
    frame = validate_prepared_shares(pd.read_csv(paths.final_shares_csv,
                                               keep_default_na=False, float_precision="round_trip"))
    from portfolio_core.dates import month_end_index
    expected = month_end_index(config.start_date, config.end_date)
    if not pd.DatetimeIndex(frame.Date.unique()).equals(expected):
        raise ValueError("Prepared shares dates differ from the configured window")
    return frame
