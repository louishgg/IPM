"""Deterministic, offline preparation for the historical live strategy.

This module deliberately has no network imports. Raw Yahoo observations,
vendor constituent files, and dated sector-history snapshots must already exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_core.io import atomic_write_dataframe
from portfolio_core.price_basis import price_basis_frame
from data_acquisition.contracts import (
    AcquisitionStatus,
    ReadinessStatus,
    read_acquisition_statuses,
    read_readiness,
)
from data_acquisition.share_checkpoints import validate_share_status_reconciliation
from portfolio_core.provider_identity import YahooIdentityResolver
from portfolio_core.sector_resolution import (
    build_sector_assignments_from_evidence,
)
from portfolio_core.security_identity import load_security_identity_bundle
from portfolio_core.shares import (
    load_raw_shares,
    select_canonical_share_observations_asof,
)

from .config import DEFAULT_CONFIG
from .monthly_history import compose_monthly_history, load_monthly_evidence
from .corporate_action_policy import (
    LiveCorporateActionBundle,
    load_live_corporate_action_bundle,
)
from .price_coverage import (
    PRICE_REQUIREMENT_SET,
    YAHOO_SUPPLEMENT_SOURCE,
    read_yahoo_supplement,
)
from .strategy_universe import (
    DOWNLOAD_BENCHMARK_COMMAND,
    DOWNLOAD_PRICES_COMMAND,
    DOWNLOAD_SHARES_COMMAND,
    execution_valuation_dates,
    execution_valuation_membership_requirements,
    historical_monthly_dates,
    load_validated_strategy_universe,
    sector_assignment_requirements,
)


def prepare_sector_assignments(
    config=DEFAULT_CONFIG,
    *,
    schedule: pd.DataFrame,
    membership: pd.DataFrame,
    actions: LiveCorporateActionBundle,
    history_dates=(),
) -> pd.DataFrame:
    """Prepare exact-date GICS rows for every live sector consumer.

    Every requirement resolves against the selected Wikipedia revision or an
    approved, causal, fill-only S&P notice.  Missing evidence fails closed; live
    preparation never carries an earlier classification forward locally.
    """
    paths = config.paths
    requirements = sector_assignment_requirements(
        schedule,
        membership,
        evaluation_start=config.market.competition_start,
        corporate_action_events=actions.events,
        corporate_action_legs=actions.legs,
        history_dates=history_dates,
    )
    assignments = build_sector_assignments_from_evidence(
        requirements,
        paths=paths.sector_history,
        repository_root=paths.project_root,
    )
    return assignments


def _load_status(
    path: Path,
    required_asset_ids: set[str],
    command: str,
) -> list[AcquisitionStatus]:
    if not path.is_file():
        raise RuntimeError(f"Missing acquisition status {path}. Run `{command}`.")
    records = read_acquisition_statuses(path)
    by_asset: dict[str, list] = {}
    for item in records:
        by_asset.setdefault(item.identity.asset_id, []).append(item)
    missing = sorted(set(required_asset_ids) - set(by_asset))
    if missing:
        raise RuntimeError(
            f"Missing acquisition identities for {missing}. Run `{command}`."
        )
    return records


def _load_readiness(
    path: Path,
    required_asset_ids: set[str],
    command: str,
    *,
    requirement_set: str | None = None,
) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing acquisition readiness {path}. Run `{command}`.")
    records = read_readiness(path)
    if requirement_set is not None:
        records = [
            item for item in records if item.requirement_set == requirement_set
        ]
    by_asset: dict[str, list] = {}
    for item in records:
        by_asset.setdefault(item.asset_id, []).append(item)
    missing = sorted(set(required_asset_ids) - set(by_asset))
    incomplete = sorted(
        asset_id
        for asset_id in required_asset_ids
        if asset_id in by_asset
        and not any(
            item.status is ReadinessStatus.COMPLETE for item in by_asset[asset_id]
        )
    )
    if missing or incomplete:
        problem = sorted(set(missing) | set(incomplete))
        raise RuntimeError(
            f"Incomplete downstream readiness for {problem}. Run `{command}`."
        )


def _read_wide_market(path: Path, field: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, index_col=0, float_precision="round_trip")
    frame.index = pd.to_datetime(frame.index, errors="raise")
    frame.index.name = "Date"
    if frame.index.duplicated().any() or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path} dates must be unique and ordered")
    if frame.columns.duplicated().any():
        raise ValueError(f"{path} ticker columns must be unique")
    frame = frame.apply(pd.to_numeric, errors="raise")
    values = frame.to_numpy(dtype=float)
    finite = np.isfinite(values) | np.isnan(values)
    if not finite.all():
        raise ValueError(f"{path} contains infinite {field} values")
    invalid = values < 0.0 if field == "Volume" else values <= 0.0
    invalid &= ~np.isnan(values)
    if invalid.any():
        raise ValueError(f"{path} contains invalid {field} values")
    return frame


def prepare_core_data(config=DEFAULT_CONFIG) -> pd.DataFrame:
    """Consolidate canonical Yahoo data with the authoritative supplement."""
    paths = config.paths
    schedule, membership, metadata = load_validated_strategy_universe(config)
    required_assets = set(membership["Asset_ID"])
    competition_assets = {
        asset_id for _, asset_id in execution_valuation_membership_requirements(
            schedule, membership, evaluation_start=config.market.competition_start,
        )
    }
    _load_status(
        paths.raw_price_status_csv,
        required_assets,
        DOWNLOAD_PRICES_COMMAND,
    )
    _load_readiness(
        paths.raw_price_readiness_csv,
        competition_assets,
        DOWNLOAD_PRICES_COMMAND,
        requirement_set=PRICE_REQUIREMENT_SET,
    )

    close = _read_wide_market(paths.raw_price_close_csv, "Close")
    opened = _read_wide_market(paths.raw_price_open_csv, "Open")
    volume = _read_wide_market(paths.raw_price_volume_csv, "Volume")
    if not close.index.equals(opened.index) or not close.index.equals(volume.index):
        raise ValueError("Canonical price field dates do not match")
    if list(close.columns) != list(opened.columns) or list(close.columns) != list(volume.columns):
        raise ValueError("Canonical price field tickers do not match")
    market = pd.concat(
        {
            "Open": opened.stack(future_stack=True),
            "Close": close.stack(future_stack=True),
            "Volume": volume.stack(future_stack=True),
        },
        axis=1,
    ).dropna(how="all")
    market.index.names = ["Date", "Yahoo_Ticker"]
    market = market.reset_index()
    # Multiple historical labels may refer to the same reviewed Yahoo series.
    # Membership validation rejects simultaneous duplicates in an index basket.
    market = market.merge(
        metadata[["Yahoo_Ticker", "Asset_ID"]],
        on="Yahoo_Ticker", how="inner", validate="many_to_many",
    )
    market["Price_Source"] = "yahoo"

    supplemental_path = paths.raw_price_supplemental_csv
    supplemental_manifest = paths.raw_price_supplemental_manifest_csv
    if not supplemental_path.is_file() or not supplemental_manifest.is_file():
        raise RuntimeError(
            "The authoritative Yahoo supplement or its manifest is missing."
        )
    supplemental = read_yahoo_supplement(paths).rename(
        columns={"Provider_Symbol": "Yahoo_Ticker"}
    )
    supplemental = supplemental.loc[
        supplemental["Asset_ID"].isin(required_assets)
    ].copy()
    supplemental["Price_Source"] = YAHOO_SUPPLEMENT_SOURCE
    supplemental = supplemental[
        [
            "Date",
            "Asset_ID",
            "Yahoo_Ticker",
            "Open",
            "Close",
            "Volume",
            "Price_Source",
        ]
    ]
    # Downloaded Yahoo observations retain precedence. The authoritative
    # supplement fills only otherwise absent asset-date rows.
    market = pd.concat([market, supplemental], ignore_index=True)
    market = market.drop_duplicates(["Date", "Asset_ID"], keep="first")
    market = market.merge(
        metadata[["Asset_ID", "Source_Ticker"]], on="Asset_ID", how="left", validate="many_to_one"
    )
    market = market[
        ["Price_Source", "Date", "Asset_ID", "Source_Ticker", "Yahoo_Ticker", "Open", "Close", "Volume"]
    ].sort_values(["Date", "Asset_ID"], kind="stable").reset_index(drop=True)
    if market.duplicated(["Date", "Asset_ID"]).any():
        raise ValueError("Prepared market data is not unique at Date,Asset_ID")
    observations, dividends = load_monthly_evidence(paths, market)
    monthly = compose_monthly_history(
        market, observations, dividends, through=schedule.Signal_Cutoff.max(),
    )
    action_bundle = load_live_corporate_action_bundle(paths)
    sector_assignments = prepare_sector_assignments(
        config,
        schedule=schedule,
        membership=membership,
        actions=action_bundle,
        history_dates=historical_monthly_dates(
            market.Date, through=schedule.Signal_Cutoff.max(),
        ),
    )
    for frame, path in (
        (schedule, paths.decision_schedule_csv),
        (membership, paths.pit_membership_csv),
        (metadata, paths.asset_metadata_csv),
        (sector_assignments, paths.sector_assignments_csv),
        (market, paths.market_daily_csv),
        (monthly, paths.market_monthly_csv),
        (dividends, paths.monthly_dividends_csv),
        (price_basis_frame("live"), paths.price_basis_csv),
        (action_bundle.events, paths.prepared_corporate_action_events_csv),
        (action_bundle.legs, paths.prepared_corporate_action_legs_csv),
        (action_bundle.sources, paths.prepared_corporate_action_sources_csv),
        (action_bundle.policy, paths.prepared_corporate_action_policy_csv),
    ):
        atomic_write_dataframe(
            frame,
            path,
            index=False,
            date_format=None,
            float_format=None,
            lineterminator="\n",
        )
    return market


def _load_raw_shares(
    path: Path,
    statuses: list[AcquisitionStatus],
    metadata: pd.DataFrame,
    *,
    identity_root: Path,
) -> pd.DataFrame:
    raw = load_raw_shares(path)
    validate_share_status_reconciliation(
        statuses,
        raw,
    )

    unexpected_statuses = [
        status.identity.key
        for status in statuses
        if status.identity.scope != "live"
        or status.identity.dataset != "shares"
        or status.identity.provider != "yahoo"
    ]
    if unexpected_statuses:
        raise ValueError(
            "Raw shares status contains identities outside the live Yahoo "
            f"shares ledger: {unexpected_statuses}"
        )
    yahoo_resolver = YahooIdentityResolver(
        load_security_identity_bundle(identity_root),
        scope="live",
    )
    expected_symbols = {
        asset_id: yahoo_resolver.resolve(
            asset_id,
            purpose="effective_security",
        ).provider_symbol
        for asset_id in metadata["Asset_ID"].astype(str)
    }
    unknown = sorted(set(raw["Asset_ID"]) - set(expected_symbols))
    if unknown:
        raise ValueError(f"Raw shares contains unknown live assets: {unknown}")
    mismatched = raw.loc[
        raw["Provider_Symbol"].ne(raw["Asset_ID"].map(expected_symbols)),
        "Asset_ID",
    ].unique()
    if len(mismatched):
        raise ValueError(
            f"Raw shares contains mismatched provider symbols: {sorted(mismatched)}"
        )
    return raw


def prepare_brinson_data(config=DEFAULT_CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare as-of shares and total-return benchmark without future fill."""
    paths = config.paths
    schedule, membership, metadata = load_validated_strategy_universe(config)
    share_requirements = execution_valuation_membership_requirements(
        schedule,
        membership,
        evaluation_start=config.market.competition_start,
    )
    required_assets = {asset_id for _, asset_id in share_requirements}
    share_statuses = _load_status(
        paths.shares.acquisition_status_csv, required_assets, DOWNLOAD_SHARES_COMMAND
    )
    _load_readiness(
        paths.shares.readiness_csv,
        required_assets,
        DOWNLOAD_SHARES_COMMAND,
    )
    raw = _load_raw_shares(
        paths.shares.raw_shares_csv,
        share_statuses,
        metadata,
        identity_root=paths.project_root,
    )
    shares = select_canonical_share_observations_asof(raw, share_requirements)
    covered = {
        (row.Date, row.Asset_ID)
        for row in shares.itertuples(index=False)
    }
    for boundary, asset_id in share_requirements:
        if (boundary, asset_id) not in covered:
            raise RuntimeError(
                f"No shares observation on or before {boundary.date()} for {asset_id}. "
                f"Run `{DOWNLOAD_SHARES_COMMAND}`."
            )

    benchmark_path = paths.raw_benchmark_csv
    benchmark_assets = {config.benchmark.ticker}
    _load_status(
        paths.raw_benchmark_status_csv,
        benchmark_assets,
        DOWNLOAD_BENCHMARK_COMMAND,
    )
    _load_readiness(
        paths.raw_benchmark_readiness_csv,
        benchmark_assets,
        DOWNLOAD_BENCHMARK_COMMAND,
    )
    if not benchmark_path.is_file():
        raise RuntimeError(f"Missing benchmark raw data. Run `{DOWNLOAD_BENCHMARK_COMMAND}`.")
    benchmark = pd.read_csv(benchmark_path, keep_default_na=False)
    if not {"Date", "Open", "Close"}.issubset(benchmark.columns):
        raise RuntimeError(
            f"Raw benchmark requires Date,Open,Close. Run `{DOWNLOAD_BENCHMARK_COMMAND}`."
        )
    benchmark = benchmark[["Date", "Open", "Close"]].copy()
    benchmark["Date"] = pd.to_datetime(benchmark["Date"], errors="raise")
    benchmark[["Open", "Close"]] = benchmark[["Open", "Close"]].apply(
        pd.to_numeric, errors="raise"
    )
    values = benchmark[["Open", "Close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("Benchmark Open/Close must be finite and positive")
    if benchmark["Date"].duplicated().any():
        raise ValueError("Benchmark dates must be unique")
    for boundary in execution_valuation_dates(
        schedule, evaluation_start=config.market.competition_start,
    ):
        if not benchmark["Date"].eq(boundary).any():
            raise RuntimeError(
                f"Benchmark lacks prices for {boundary.date()}. "
                f"Run `{DOWNLOAD_BENCHMARK_COMMAND}`."
            )
    atomic_write_dataframe(
        shares,
        paths.shares.prepared_shares_csv,
        index=False,
        date_format=None,
        float_format=None,
        lineterminator="\n",
    )
    atomic_write_dataframe(
        benchmark.sort_values("Date"),
        paths.prepared_benchmark_csv,
        index=False,
        date_format=None,
        float_format=None,
        lineterminator="\n",
    )
    return shares, benchmark


__all__ = [
    "prepare_brinson_data",
    "prepare_core_data",
    "prepare_sector_assignments",
]
