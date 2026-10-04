"""Neutral live schedule and membership-source contracts.

Acquisition and preparation both depend on these read-only helpers.  Keeping
them outside either stage prevents planning from importing preparation code.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from portfolio_core.portfolio_lifecycle import (
    asset_is_extinguished_from_events,
    lifecycle_eligible_assets,
)
from portfolio_core.provider_identity import YahooIdentityResolver
from portfolio_core.security_identity import load_security_identity_bundle
from portfolio_core.sp500_membership import (
    membership_asof,
    validate_membership_sources,
)
from portfolio_core.sector_assignments import ASSIGNMENT_REQUIREMENT_COLUMNS

from .config import DEFAULT_CONFIG, LiveConfig


DOWNLOAD_PRICES_COMMAND = "python -m data_acquisition.acquire live prices"
DOWNLOAD_BENCHMARK_COMMAND = "python -m data_acquisition.acquire live benchmark"
DOWNLOAD_SHARES_COMMAND = "python -m data_acquisition.acquire live shares"
METADATA_COLUMNS = ("Asset_ID", "Source_Ticker", "Yahoo_Ticker")
SCHEDULE_COLUMNS = [
    "Rebalance_ID",
    "Membership_Effective_Date",
    "Signal_Cutoff",
    "Sizing_Date",
    "Sizing_Field",
    "Execution_Date",
    "Execution_Field",
    "Valuation_End",
    "Valuation_Field",
]
SCHEDULE_DATE_COLUMNS = (
    "Membership_Effective_Date",
    "Signal_Cutoff",
    "Sizing_Date",
    "Execution_Date",
    "Valuation_End",
)
EVALUATION_PERIOD_COLUMNS = (
    "Rebalance_ID", "Period_Type", "Period_Start", "Start_Field",
    "Period_End", "End_Field",
)


def evaluation_periods(
    schedule: pd.DataFrame,
    *,
    evaluation_start: object,
    evaluation_end: object,
) -> pd.DataFrame:
    """Project decision periods onto the complete window from account inception.

    The cash interval has no rebalance ID. It is an account interval, never a
    synthetic decision, order, or security position.
    """
    if schedule.empty:
        raise ValueError("Live decision schedule cannot be empty")
    start, end = pd.Timestamp(evaluation_start), pd.Timestamp(evaluation_end)
    first, last = schedule.iloc[0], schedule.iloc[-1]
    execution = pd.Timestamp(first["Execution_Date"])
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("Evaluation bounds must be finite and increasing")
    if start > pd.Timestamp(first["Sizing_Date"]) or execution < start:
        raise ValueError("Evaluation start must precede sizing and execution")
    if pd.Timestamp(last["Valuation_End"]) != end:
        raise ValueError("Final valuation must equal evaluation end")
    rows = []
    if start < execution or first["Execution_Field"] != "Open":
        rows.append({
            "Rebalance_ID": "", "Period_Type": "cash",
            "Period_Start": start, "Start_Field": "Open",
            "Period_End": execution, "End_Field": first["Execution_Field"],
        })
    rows.extend({
        "Rebalance_ID": str(row.Rebalance_ID), "Period_Type": "invested",
        "Period_Start": pd.Timestamp(row.Execution_Date),
        "Start_Field": str(row.Execution_Field),
        "Period_End": pd.Timestamp(row.Valuation_End),
        "End_Field": str(row.Valuation_Field),
    } for row in schedule.itertuples(index=False))
    return pd.DataFrame(rows, columns=EVALUATION_PERIOD_COLUMNS)


def decision_schedule(config: LiveConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Return the reviewed causal schedule from immutable configuration."""
    records = [
        {
            "Rebalance_ID": period.rebalance_id,
            "Membership_Effective_Date": period.membership_effective_date,
            "Signal_Cutoff": period.signal_cutoff,
            "Sizing_Date": period.sizing_date,
            "Sizing_Field": period.sizing_field,
            "Execution_Date": period.execution_date,
            "Execution_Field": period.execution_field,
            "Valuation_End": period.valuation_end,
            "Valuation_Field": period.valuation_field,
        }
        for period in config.schedule
    ]
    frame = pd.DataFrame(records, columns=SCHEDULE_COLUMNS)
    for column in SCHEDULE_DATE_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    return frame


def load_validated_membership(
    config: LiveConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build dated memberships for shared history and the competition schedule."""
    history = validate_membership_sources(
        config.paths.membership,
        required_through=config.market.competition_end,
    )
    schedule = decision_schedule(config)
    requested_dates = set(schedule["Membership_Effective_Date"])
    requested_dates.update(schedule["Execution_Date"])
    requested_dates.update(schedule["Valuation_End"])
    requested_dates.add(pd.Timestamp(config.market.competition_start))
    history_start = pd.Timestamp(config.market.download_start)
    requested_dates.add(history_start)
    requested_dates.update(history.loc[
        history["date"].between(history_start, pd.Timestamp(config.market.competition_end)),
        "date",
    ])

    rows: list[dict[str, object]] = []
    seen_effective_dates: set[pd.Timestamp] = set()
    for requested_date in sorted(requested_dates):
        effective_date, members = membership_asof(history, requested_date)
        if effective_date in seen_effective_dates:
            continue
        seen_effective_dates.add(effective_date)
        rows.extend(
            {"Effective_Date": effective_date, "Asset_ID": asset_id}
            for asset_id in members
        )
    membership = pd.DataFrame(rows).sort_values(
        ["Effective_Date", "Asset_ID"],
        kind="stable",
    ).reset_index(drop=True)

    expected_signal_effective = [
        membership_asof(history, signal_date)[0]
        for signal_date in schedule["Signal_Cutoff"]
    ]
    if list(schedule["Membership_Effective_Date"]) != expected_signal_effective:
        raise ValueError(
            "Reviewed schedule membership dates do not match source as-of rows"
        )
    return schedule, membership


def load_validated_strategy_universe(
    config: LiveConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the validated schedule, membership, and Yahoo metadata table."""
    schedule, membership = load_validated_membership(config)
    identity_bundle = load_security_identity_bundle(config.paths.project_root)
    yahoo_resolver = YahooIdentityResolver(identity_bundle, scope="live")
    reviewed = identity_bundle.provider_mappings
    dated_sources = set(reviewed.loc[
        reviewed.Scope.eq("live") & reviewed.Provider.eq("yahoo"), "Source_Ticker",
    ])
    rows = []
    for asset_id, active in membership.groupby("Asset_ID", sort=True):
        # The first retained source basket can predate the requested history.
        dates = pd.to_datetime(active["Effective_Date"]).clip(
            lower=pd.Timestamp(config.market.download_start)
        )
        if asset_id not in dated_sources:
            # Ordinary symbols have no reviewed date intervals to distinguish.
            dates = dates.iloc[:1]
        symbols = {
            yahoo_resolver.resolve(
                asset_id, purpose="historical_prices", as_of=value,
            ).provider_symbol for value in dates
        }
        if len(symbols) != 1:
            raise ValueError(
                f"Live history needs multiple dated Yahoo identities for "
                f"{asset_id}: {sorted(symbols)}"
            )
        rows.append((asset_id, asset_id, symbols.pop()))
    metadata = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    validate_membership_provider_identities(membership, metadata)
    return schedule, membership, metadata


def validate_membership_provider_identities(
    membership: pd.DataFrame, metadata: pd.DataFrame,
) -> None:
    """Allow reviewed relabel aliases, but never count one series twice in a basket."""
    joined = membership.merge(
        metadata[["Asset_ID", "Yahoo_Ticker"]],
        on="Asset_ID", validate="many_to_one",
    )
    duplicates = joined.duplicated(["Effective_Date", "Yahoo_Ticker"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Simultaneous constituent Yahoo identities overlap: "
            f"{joined.loc[duplicates].to_dict('records')[:10]}"
        )


def historical_monthly_dates(
    dates: Iterable, *, through: object,
) -> tuple[pd.Timestamp, ...]:
    """Use observed final sessions, retaining labels separately in research code."""
    values = pd.DatetimeIndex(pd.to_datetime(dates)).unique().sort_values()
    values = values[values <= pd.Timestamp(through)]
    return tuple(pd.Series(values, index=values.to_period("M")).groupby(level=0).max())


def historical_sector_date(observation_date: object, *, evaluation_start: object) -> pd.Timestamp:
    """Reuse calendar-month history before the existing live warm-up month.

    The original live evidence starts with the monthly close immediately before
    account inception (January 30 for the February 2026 competition). Preserve
    that observation and the competition's dated snapshots.
    """
    date = pd.Timestamp(observation_date).normalize()
    warmup_month = (pd.Timestamp(evaluation_start).to_period("M") - 1).start_time
    return date + pd.offsets.MonthEnd(0) if date < warmup_month else date


def prepared_membership_asof(
    membership: pd.DataFrame,
    cutoff: object,
) -> set[str]:
    """Return the latest prepared membership at or before ``cutoff``."""
    required = {"Effective_Date", "Asset_ID"}
    missing = sorted(required - set(membership.columns))
    if missing:
        raise ValueError(f"Prepared membership is missing columns: {missing}")
    dates = pd.to_datetime(membership["Effective_Date"], errors="raise")
    cutoff_date = pd.Timestamp(cutoff).normalize()
    eligible = dates.loc[dates.le(cutoff_date)]
    if eligible.empty:
        raise ValueError(
            "No prepared constituent membership exists on or before "
            f"{cutoff_date.date()}"
        )
    effective_date = eligible.max()
    members = membership.loc[dates.eq(effective_date), "Asset_ID"].astype(str).str.strip()
    if members.eq("").any() or members.duplicated().any():
        raise ValueError(
            f"Prepared membership is invalid on {effective_date.date()}"
        )
    return set(members)


def execution_valuation_dates(
    schedule: pd.DataFrame,
    *,
    evaluation_start: object | None = None,
) -> tuple[pd.Timestamp, ...]:
    """Return normalized account inception, execution and valuation boundaries."""
    required = {"Execution_Date", "Valuation_End"}
    missing = sorted(required - set(schedule.columns))
    if missing:
        raise ValueError(f"Decision schedule is missing columns: {missing}")
    dates = pd.to_datetime(
        pd.concat(
            [schedule["Execution_Date"], schedule["Valuation_End"]],
            ignore_index=True,
        ),
        errors="raise",
    ).dt.normalize()
    boundaries = set(dates)
    if evaluation_start is not None:
        boundaries.add(pd.Timestamp(evaluation_start).normalize())
    return tuple(sorted(boundaries))


def execution_valuation_membership_requirements(
    schedule: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    evaluation_start: object | None = None,
) -> tuple[tuple[pd.Timestamp, str], ...]:
    """Return active member requirements at every live boundary."""
    return tuple(
        (boundary, asset_id)
        for boundary in execution_valuation_dates(
            schedule, evaluation_start=evaluation_start
        )
        for asset_id in sorted(prepared_membership_asof(membership, boundary))
    )


def live_interval_eligible_assets(
    market: pd.DataFrame,
    member_asset_ids: Iterable[str],
    *,
    execution_date: object,
    execution_field: str,
    valuation_end: object,
    valuation_field: str,
    corporate_action_events: pd.DataFrame,
    corporate_action_legs: pd.DataFrame,
    corporate_action_sources: pd.DataFrame,
) -> set[str]:
    """Return members that can be opened or retained for one live interval."""

    start = pd.Timestamp(execution_date).normalize()
    end = pd.Timestamp(valuation_end).normalize()
    market_dates = pd.to_datetime(market["Date"], errors="raise").dt.normalize()

    def _prices(date: pd.Timestamp, field: str) -> dict[str, object]:
        rows = market.loc[
            market_dates.eq(date),
            ["Asset_ID", field],
        ]
        return dict(
            zip(
                rows["Asset_ID"].astype(str),
                rows[field],
                strict=True,
            )
        )

    return set(
        lifecycle_eligible_assets(
            member_asset_ids,
            start=start,
            end=end,
            start_prices=_prices(start, execution_field),
            end_prices=_prices(end, valuation_field),
            events=corporate_action_events,
            legs=corporate_action_legs,
            sources=corporate_action_sources,
        )
    )


def live_non_extinguished_assets(
    member_asset_ids: Iterable[str],
    *,
    as_of_date: object,
    corporate_action_events: pd.DataFrame,
    corporate_action_legs: pd.DataFrame,
) -> set[str]:
    """Return nominal members that remain valid Brinson constituents."""

    date = pd.Timestamp(as_of_date).normalize()
    return {
        str(asset_id)
        for asset_id in member_asset_ids
        if not asset_is_extinguished_from_events(
            corporate_action_events,
            corporate_action_legs,
            str(asset_id),
            date,
        )
    }


def sector_assignment_requirements(
    schedule: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    evaluation_start: object | None = None,
    corporate_action_events: pd.DataFrame | None = None,
    corporate_action_legs: pd.DataFrame | None = None,
    history_dates: Iterable = (),
) -> pd.DataFrame:
    """Return sector requirements for signal members and surviving constituents.

    Execution-date requirements cover the benchmark's surviving members,
    regardless of whether the strategy can trade them.
    """
    required_schedule = {
        "Membership_Effective_Date",
        "Signal_Cutoff",
        "Execution_Date",
    }
    missing = sorted(required_schedule - set(schedule.columns))
    if missing:
        raise ValueError(f"Decision schedule is missing columns: {missing}")
    if (corporate_action_events is None) != (corporate_action_legs is None):
        raise ValueError(
            "corporate-action events and legs must be supplied together"
        )
    action_events = (
        pd.DataFrame() if corporate_action_events is None else corporate_action_events
    )
    action_legs = (
        pd.DataFrame() if corporate_action_legs is None else corporate_action_legs
    )

    ordered = schedule.copy()
    for column in (
        "Membership_Effective_Date",
        "Signal_Cutoff",
        "Execution_Date",
    ):
        ordered[column] = pd.to_datetime(ordered[column], errors="raise").dt.normalize()
    ordered = ordered.sort_values("Execution_Date", kind="stable").reset_index(drop=True)

    by_pair: dict[tuple[pd.Timestamp, str], str] = {}
    for value in history_dates:
        date = pd.Timestamp(value).normalize()
        # Historical monthly decisions reuse the shared calendar-month evidence.
        # Keep membership tied to the observed close and keep competition signal
        # and execution dates exact; do not acquire duplicate weekend snapshots.
        sector_date = historical_sector_date(
            date, evaluation_start=(
                evaluation_start if evaluation_start is not None else ordered.Signal_Cutoff.min()
            ),
        )
        for asset_id in prepared_membership_asof(membership, date):
            by_pair[(sector_date, asset_id)] = asset_id
    if evaluation_start is not None:
        start = pd.Timestamp(evaluation_start).normalize()
        initial_members = live_non_extinguished_assets(
            prepared_membership_asof(membership, start),
            as_of_date=start,
            corporate_action_events=action_events,
            corporate_action_legs=action_legs,
        )
        for asset_id in initial_members:
            by_pair[(start, asset_id)] = asset_id
    for row in ordered.itertuples(index=False):
        signal_date = pd.Timestamp(row.Signal_Cutoff).normalize()
        execution_date = pd.Timestamp(row.Execution_Date).normalize()
        signal_members = prepared_membership_asof(membership, signal_date)
        execution_members = prepared_membership_asof(membership, execution_date)

        scheduled_effective = pd.Timestamp(row.Membership_Effective_Date).normalize()
        effective_rows = membership.loc[
            pd.to_datetime(membership["Effective_Date"], errors="raise")
            .dt.normalize()
            .eq(scheduled_effective),
            "Asset_ID",
        ]
        if set(effective_rows.astype(str).str.strip()) != signal_members:
            raise ValueError(
                "Decision-schedule membership disagrees with prepared membership "
                f"at signal cutoff {signal_date.date()}"
            )

        for asset_id in signal_members:
            by_pair[(signal_date, asset_id)] = asset_id
        execution_required = live_non_extinguished_assets(
            execution_members,
            as_of_date=execution_date,
            corporate_action_events=action_events,
            corporate_action_legs=action_legs,
        )
        for asset_id in execution_required:
            by_pair[(execution_date, asset_id)] = asset_id

    rows = [
        {
            "As_Of_Date": requirement_date,
            "Asset_ID": asset_id,
            "Source_Ticker": source_ticker,
        }
        for (requirement_date, asset_id), source_ticker in sorted(by_pair.items())
    ]
    return pd.DataFrame(rows, columns=ASSIGNMENT_REQUIREMENT_COLUMNS)


__all__ = [
    "DOWNLOAD_BENCHMARK_COMMAND",
    "DOWNLOAD_PRICES_COMMAND",
    "DOWNLOAD_SHARES_COMMAND",
    "METADATA_COLUMNS",
    "SCHEDULE_COLUMNS",
    "SCHEDULE_DATE_COLUMNS",
    "EVALUATION_PERIOD_COLUMNS",
    "decision_schedule",
    "evaluation_periods",
    "execution_valuation_dates",
    "execution_valuation_membership_requirements",
    "historical_monthly_dates",
    "historical_sector_date",
    "live_interval_eligible_assets",
    "live_non_extinguished_assets",
    "load_validated_membership",
    "load_validated_strategy_universe",
    "prepared_membership_asof",
    "sector_assignment_requirements",
    "validate_membership_provider_identities",
]
