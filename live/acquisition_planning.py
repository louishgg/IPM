"""Live-owned planning and readiness boundaries for shared acquisition.

This module contains no provider client and performs no network access.  The
root :mod:`data_acquisition.acquire` command dispatches to
``live.acquisition_execution`` for transport and checkpoint persistence; these
helpers define the exact live identities, requested windows, and downstream
requirement sets.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from data_acquisition.contracts import (
    AcquisitionIdentity,
    AcquisitionRequest,
    AcquisitionStatus,
    CommandRequest,
    ProviderStatus,
    ReadinessStatus,
    read_asset_selection,
    read_acquisition_statuses,
    read_readiness,
)
from data_acquisition.engine import build_acquisition_plan
from data_acquisition.sector_acquisition_planning import (
    dry_run_sector_acquisition,
    project_sector_acquisition_requirements,
)
from data_acquisition.share_checkpoints import project_share_identity_interval
from portfolio_core.provider_identity import YahooIdentityResolver
from portfolio_core.security_identity import load_security_identity_bundle

from .corporate_action_policy import load_live_corporate_action_bundle
from .config import DEFAULT_CONFIG
from .price_coverage import PRICE_REQUIREMENT_SET
from .strategy_universe import (
    decision_schedule,
    execution_valuation_dates,
    execution_valuation_membership_requirements,
    historical_monthly_dates,
    load_validated_membership,
    load_validated_strategy_universe,
    sector_assignment_requirements,
)


SCOPE = "live"
PROVIDER = "yahoo"
PRICE_DATASET = "prices"
SHARES_DATASET = "shares"
BENCHMARK_DATASET = "benchmark"


def _exclusive_end_as_inclusive(value: date) -> str:
    return (value - timedelta(days=1)).isoformat()


def _subtract_months(value: date, months: int) -> str:
    return (pd.Timestamp(value) - pd.DateOffset(months=months)).date().isoformat()


def build_requests(
    dataset: str,
    *,
    config=DEFAULT_CONFIG,
    asset_ids: Iterable[str] | None = None,
) -> tuple[AcquisitionRequest, ...]:
    """Build the provider-neutral request plan consumed by the root CLI."""

    dataset = str(dataset).strip().lower()
    if dataset not in {PRICE_DATASET, SHARES_DATASET, BENCHMARK_DATASET}:
        raise ValueError(f"Unsupported live acquisition dataset: {dataset!r}")
    if dataset == BENCHMARK_DATASET:
        return (
            AcquisitionRequest(
                identity=AcquisitionIdentity(
                    scope=SCOPE,
                    dataset=dataset,
                    asset_id=config.benchmark.ticker,
                    provider=PROVIDER,
                    provider_symbol=config.benchmark.ticker,
                ),
                requested_start=config.benchmark.download_start.isoformat(),
                requested_end=_exclusive_end_as_inclusive(
                    config.benchmark.download_end
                ),
            ),
        )

    schedule, membership, metadata = load_validated_strategy_universe(config)
    yahoo_by_asset = metadata.set_index("Asset_ID")["Yahoo_Ticker"].to_dict()
    requested = (
        tuple(
            sorted({
                asset_id
                for _, asset_id in execution_valuation_membership_requirements(
                    schedule,
                    membership,
                    evaluation_start=config.market.competition_start,
                )
            })
        )
        if dataset == SHARES_DATASET
        else tuple(sorted(membership["Asset_ID"].astype(str).unique()))
    )
    if asset_ids is None:
        selected = requested
    else:
        selected = tuple(
            sorted({str(value).strip().upper() for value in asset_ids})
        )
        unknown = sorted(set(selected) - set(requested))
        if unknown:
            raise ValueError(
                f"Live request contains assets outside membership: {unknown}"
            )
    if dataset == PRICE_DATASET:
        requested_start = config.market.download_start.isoformat()
        requested_end = _exclusive_end_as_inclusive(config.market.download_end)
        yahoo_resolver = None
    else:
        requested_start = _subtract_months(
            config.market.competition_start,
            config.brinson.yahoo_lookback_months,
        )
        requested_end = config.market.competition_end.isoformat()
        yahoo_resolver = YahooIdentityResolver(
            load_security_identity_bundle(config.paths.project_root),
            scope="live",
        )

    requests = []
    for asset_id in selected:
        provider_identity = None
        if dataset == SHARES_DATASET:
            assert yahoo_resolver is not None
            provider_identity = yahoo_resolver.resolve(
                asset_id,
                purpose="effective_security",
            )
            provider_symbol = provider_identity.provider_symbol
        else:
            provider_symbol = yahoo_by_asset[asset_id]
        item_requested_start = requested_start
        item_requested_end = requested_end
        effective_start = ""
        effective_end = ""
        if provider_identity is not None:
            canonical_identity, item_requested_start, item_requested_end = (
                project_share_identity_interval(
                    asset_id=asset_id,
                    provider_symbol=provider_symbol,
                    requested_start=requested_start,
                    requested_end=requested_end,
                    effective_start=provider_identity.effective_start,
                    effective_end_exclusive=provider_identity.effective_end,
                )
            )
            effective_start = canonical_identity.effective_start
            effective_end = canonical_identity.effective_end
        requests.append(
            AcquisitionRequest(
                identity=AcquisitionIdentity(
                    scope=SCOPE,
                    dataset=dataset,
                    asset_id=asset_id,
                    provider=PROVIDER,
                    provider_symbol=provider_symbol,
                    effective_start=effective_start,
                    effective_end=effective_end,
                ),
                requested_start=item_requested_start,
                requested_end=item_requested_end,
            )
        )
    return tuple(requests)


def build_command_requests(
    command_request: CommandRequest,
    *,
    config=DEFAULT_CONFIG,
) -> tuple[AcquisitionRequest, ...]:
    """Translate one live command into its canonical provider requests."""

    selected = (
        read_asset_selection(command_request.tickers_file)
        if command_request.dataset == SHARES_DATASET
        and command_request.tickers_file is not None
        else None
    )
    return build_requests(
        command_request.dataset,
        config=config,
        asset_ids=selected,
    )


def shares_requirement_dates(config=DEFAULT_CONFIG) -> dict[str, tuple[str, ...]]:
    schedule, membership = load_validated_membership(config)
    required: dict[str, list[str]] = {}
    for boundary, asset_id in execution_valuation_membership_requirements(
        schedule,
        membership,
        evaluation_start=config.market.competition_start,
    ):
        required.setdefault(asset_id, []).append(boundary.date().isoformat())
    return {
        asset_id: tuple(required[asset_id])
        for asset_id in sorted(required)
    }


def benchmark_requirement_dates(config=DEFAULT_CONFIG) -> tuple[str, ...]:
    schedule = decision_schedule(config)
    return tuple(
        value.date().isoformat()
        for value in execution_valuation_dates(
            schedule, evaluation_start=config.market.competition_start
        )
    )


def live_sector_requirements(config=DEFAULT_CONFIG) -> pd.DataFrame:
    """Return the conservative set of dated live-sector consumers.

    Signal cutoffs require every member considered by the ranking rule.
    Execution dates require current and prior execution members plus the
    current signal members because any of them may still need a trade or
    holding classification. Keeping the potential exits here avoids making
    acquisition depend on strategy results that themselves depend on the
    acquired sectors.
    """
    schedule, membership = load_validated_membership(config)
    actions = load_live_corporate_action_bundle(config.paths)
    price_dates = pd.read_csv(
        config.paths.raw_price_close_csv, usecols=["Date"],
    )["Date"]
    requirements = sector_assignment_requirements(
        schedule,
        membership,
        evaluation_start=config.market.competition_start,
        corporate_action_events=actions.events,
        corporate_action_legs=actions.legs,
        history_dates=historical_monthly_dates(
            price_dates, through=schedule.Signal_Cutoff.max(),
        ),
    )
    return project_sector_acquisition_requirements(requirements, scope=SCOPE)


def status_path(dataset: str, config=DEFAULT_CONFIG) -> Path:
    """Return the canonical live provider-status path for one dataset."""
    paths = config.paths
    return {
        PRICE_DATASET: paths.raw_price_status_csv,
        SHARES_DATASET: paths.shares.acquisition_status_csv,
        BENCHMARK_DATASET: paths.raw_benchmark_status_csv,
    }[dataset]


def dry_run_requests(command_request, *, config=DEFAULT_CONFIG) -> tuple[AcquisitionRequest, ...]:
    """Return and print the exact unresolved live provider plan."""
    dataset = command_request.dataset
    if dataset == "sectors":
        return tuple(
            dry_run_sector_acquisition(
                command_request,
                lambda: live_sector_requirements(config),
            )
        )
    requests = build_command_requests(
        command_request,
        config=config,
    )
    provider_status_path = status_path(dataset, config)
    statuses = (
        read_acquisition_statuses(provider_status_path)
        if provider_status_path.is_file()
        else []
    )
    status_by_identity = {
        item.identity.key: item
        for item in statuses
        if item.identity.dataset == dataset
    }
    source_covered_assets: set[str] = set()
    if dataset == PRICE_DATASET and not command_request.refresh:
        readiness_path = config.paths.raw_price_readiness_csv
        if readiness_path.is_file():
            source_covered_assets = {
                item.asset_id
                for item in read_readiness(readiness_path)
                if item.requirement_set == PRICE_REQUIREMENT_SET
                and item.status is ReadinessStatus.COMPLETE
            }
    eligible_requests = tuple(
        request
        for request in requests
        if command_request.refresh
        or request.identity.asset_id not in source_covered_assets
        or request.identity.key not in status_by_identity
        or not price_request_window_matches(
            request, status_by_identity[request.identity.key],
        )
    )
    plan = build_acquisition_plan(
        eligible_requests,
        statuses=statuses,
        refresh=command_request.refresh,
    )
    pending = plan.execution_order
    print(
        f"live {dataset}: {len(pending)} provider request(s) "
        f"({len(requests) - len(pending)} terminal or source-covered "
        "checkpoint(s))."
    )
    for request in pending:
        identity = request.identity
        print(
            f"  {identity.asset_id} -> {identity.provider}:"
            f"{identity.provider_symbol} "
            f"[{request.requested_start},{request.requested_end}]"
        )
    return tuple(pending)


def price_request_window_matches(
    request: AcquisitionRequest, status: AcquisitionStatus,
) -> bool:
    """A usable competition quote does not prove an earlier history request ran."""
    return (
        status.status in {ProviderStatus.OK, ProviderStatus.NO_DATA}
        and status.requested_start == request.requested_start
        and status.requested_end == request.requested_end
    )


__all__ = [
    "BENCHMARK_DATASET",
    "PRICE_DATASET",
    "SHARES_DATASET",
    "benchmark_requirement_dates",
    "build_command_requests",
    "build_requests",
    "dry_run_requests",
    "live_sector_requirements",
    "price_request_window_matches",
    "shares_requirement_dates",
    "status_path",
]
