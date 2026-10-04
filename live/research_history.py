"""Causal historical calculations over shared prepared monthly inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_core.interval_valuation import has_interval_event, value_one_share_through_events
from portfolio_core.sector_assignments import sector_rows_asof
from portfolio_core.strategies import StrategyDecisionContext
from portfolio_core.strategies.research_strategy import ResearchStrategy
from portfolio_core.strategies.research_state import SelectedSectors

from .analysis_data import LiveAnalysisInputs, membership_as_of
from .monthly_history import monthly_matrices, causal_monthly_rows
from .preparation_artifacts import PREPARE_COMMAND
from .strategy_universe import historical_sector_date


def month_label(value: object) -> pd.Timestamp:
    return pd.Timestamp(value) + pd.offsets.MonthEnd(0)


@dataclass(frozen=True)
class LiveResearchRequirements:
    """Independent stock-estimator and sector-signal history bounds."""

    last_signal: pd.Timestamp
    earliest_price_label: pd.Timestamp
    earliest_sector_label: pd.Timestamp | None


def build_live_research_requirements(strategy, schedule) -> LiveResearchRequirements:
    """The retained 60/24 estimator requires 24 returns, not 60."""
    if not isinstance(strategy, ResearchStrategy):
        raise TypeError("Research requirements need a resolved research strategy")
    p = strategy.parameters
    first = month_label(pd.to_datetime(schedule["Signal_Cutoff"]).min())
    last = month_label(pd.to_datetime(schedule["Signal_Cutoff"]).max())
    if p.signal.family == "low_volatility":
        signal_returns = p.signal.selection_volatility.minimum_observations
    elif p.signal.family == "monthly_trend":
        signal_returns = p.signal.moving_average_months - 1
    else:
        signal_returns = p.signal.formation_months + p.signal.skip_months
    sizing_returns = (
        p.sizing.volatility.minimum_observations
        if p.sizing.method == "inverse_volatility" else 0
    )
    needed = max(signal_returns, sizing_returns)
    sector_start = (
        first - pd.offsets.MonthEnd(signal_returns)
        if strategy.strategy_id == "sector_momentum" else None
    )
    return LiveResearchRequirements(
        last, first - pd.offsets.MonthEnd(needed), sector_start,
    )


def _members(inputs: LiveAnalysisInputs, date: pd.Timestamp) -> set[str]:
    try:
        return set(membership_as_of(inputs.membership, date))
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"Historical membership missing at {date.date()}: {exc}") from exc


def _sectors(inputs: LiveAnalysisInputs, date: pd.Timestamp, members: set[str]) -> dict[str, str]:
    date = historical_sector_date(
        date, evaluation_start=inputs.evaluation_periods.Period_Start.min(),
    )
    try:
        rows = sector_rows_asof(inputs.sector_assignments, date)
    except KeyError as exc:
        raise ValueError(
            f"Historical GICS sectors missing at {date.date()}; run `{PREPARE_COMMAND}`"
        ) from exc
    missing = sorted(members - set(rows.index))
    if missing:
        raise ValueError(f"Historical GICS sectors missing for {missing} at {date.date()}")
    return rows.GICS_Sector_Code.astype(str).to_dict()


def interval_holding_returns(
    inputs: LiveAnalysisInputs, assets, start: pd.Timestamp, end: pd.Timestamp,
    *, as_of: pd.Timestamp, monthly_index: pd.DataFrame | None = None,
    calendar: dict | None = None,
) -> pd.Series:
    """Value every starting member, including departures, using one-share economics.

    Ordinary intervals use adjusted close ratios. Extinguishing merger intervals
    use nominal legal consideration and separate rights. Dividend factors bridge
    the predecessor up to conversion and successor after conversion exactly once.
    Divisible entitlements and face-value cash here are independent of the ledger.
    """
    start, end, as_of = month_label(start), month_label(end), pd.Timestamp(as_of)
    if start + pd.offsets.MonthEnd(1) != end:
        raise ValueError(f"Nonconsecutive monthly interval {start.date()} -> {end.date()}")
    monthly = (inputs.market_monthly.set_index(["Month", "Asset_ID"])
               if monthly_index is None else monthly_index)
    calendar = (inputs.market_monthly.groupby("Month").Observation_Date.max().to_dict()
                if calendar is None else calendar)
    if start not in calendar or end not in calendar:
        raise ValueError(f"Missing monthly calendar {start.date()} -> {end.date()}")
    actual_start, actual_end = calendar[start], calendar[end]
    if actual_end > as_of:
        raise ValueError(f"Future interval endpoint {actual_end.date()} after {as_of.date()}")

    def price(asset, label, field):
        row = monthly.loc[(label, asset)]
        if row.Observation_Date != calendar[label]:
            raise ValueError(f"stale observation {row.Observation_Date} for {asset}")
        if pd.isna(row.Available_Date) or row.Available_Date > as_of:
            raise ValueError(f"future or undated monthly evidence for {asset} at {label.date()}")
        value = float(row[field])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"missing positive {field} for {asset} at {label.date()}")
        return value

    def factor(asset, left, right):
        rows = inputs.monthly_dividends.loc[
            inputs.monthly_dividends.Asset_ID.eq(asset)
            & inputs.monthly_dividends.Ex_Date.gt(left)
            & inputs.monthly_dividends.Ex_Date.le(right)
        ]
        if rows.Available_Date.gt(as_of).any():
            raise ValueError(f"future dividend evidence for {asset}")
        return float(rows.Factor.prod())

    actions = inputs.corporate_actions
    returns = {}
    for asset in sorted(assets):
        try:
            if not has_interval_event(actions.events, actions.legs, asset, actual_start, actual_end):
                returns[asset] = price(asset, end, "Close") / price(asset, start, "Close") - 1
                continue
            ids = set(actions.legs.loc[actions.legs.From_Asset_ID.eq(asset), "Event_ID"])
            events = actions.events.loc[
                actions.events.Event_ID.isin(ids)
                & actions.events.Effective_Date.gt(actual_start)
                & actions.events.Effective_Date.le(actual_end)
            ]
            if len(events) != 1 or events.iloc[0].Continuity_Class != "predecessor_extinguished":
                raise ValueError("monthly event interval needs a reviewed single extinguishing event")
            event = events.iloc[0]
            policy = actions.policy.set_index("Event_ID").loc[event.Event_ID]
            if float(policy.CVR_Base_Value_Per_Unit) > 0:
                if pd.Timestamp(policy.Valuation_As_Of_Date) != end:
                    raise ValueError(f"CVR mark is not dated for {end.date()}")
                if pd.Timestamp(policy.Valuation_Available_Date) > as_of:
                    raise ValueError("future CVR valuation evidence")
            result = value_one_share_through_events(
                actions.events, actions.legs, actions.sources,
                asset, actual_start, actual_end,
                lambda successor: price(successor, end, "Nominal_Close")
                / factor(successor, event.Effective_Date, actual_end),
            )
            if result is None:
                raise ValueError("missing successor valuation")
            settled, value = result
            applied = set(settled.audit.loc[settled.audit.Applied_To_Position, "Event_ID"])
            if applied != {event.Event_ID}:
                raise ValueError("chained monthly actions require separately reviewed dividend timing")
            denominator = price(asset, start, "Nominal_Close") * factor(
                asset, actual_start, event.Effective_Date,
            )
            returns[asset] = float(value) / denominator - 1
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"Unvalueable starting constituent {asset}, {start.date()} -> {end.date()}: {exc}"
            ) from exc
    return pd.Series(returns, dtype=float)


def _sector_returns(inputs, start_at, through):
    records = {}
    labels = pd.date_range(start_at, through, freq="ME")
    monthly = inputs.market_monthly.set_index(["Month", "Asset_ID"])
    calendar = inputs.market_monthly.groupby("Month").Observation_Date.max().to_dict()
    first_cutoff = pd.Timestamp(inputs.schedule.Signal_Cutoff.min())
    for start, end in zip(labels, labels[1:]):
        if start not in calendar or end not in calendar:
            raise ValueError(f"Missing monthly calendar {start.date()} -> {end.date()}")
        members = _members(inputs, calendar[start])
        sectors = _sectors(inputs, calendar[start], members)
        returns = interval_holding_returns(
            inputs, members, start, end, as_of=max(first_cutoff, calendar[end]),
            monthly_index=monthly, calendar=calendar,
        )
        records[end] = returns.groupby(pd.Series(sectors).reindex(returns.index)).mean()
    return pd.DataFrame.from_dict(records, orient="index").sort_index()


@dataclass(frozen=True)
class LiveResearchHistory:
    sector_returns: pd.DataFrame

    def context(self, inputs, cutoff, candidates, sectors, previous, selected: SelectedSectors):
        label = month_label(cutoff)
        close, volume, calendar = monthly_matrices(inputs.market_monthly, cutoff=cutoff)
        if label not in calendar:
            raise ValueError(f"Missing prepared monthly close for {label.date()}; run `{PREPARE_COMMAND}`")
        return StrategyDecisionContext(
            close, volume, tuple(sorted(candidates)), sectors, previous,
            label, calendar[label], previous_selected_sectors=selected,
            sector_returns=self.sector_returns.loc[:label],
        )


def prepare_live_research_history(inputs, strategy: ResearchStrategy) -> LiveResearchHistory:
    requirements = build_live_research_requirements(strategy, inputs.schedule)
    cutoff = pd.Timestamp(inputs.schedule.Signal_Cutoff.max())
    monthly = causal_monthly_rows(inputs.market_monthly, cutoff=cutoff)
    calendar = monthly.groupby("Month").Observation_Date.max().to_dict()
    for signal_cutoff in pd.to_datetime(inputs.schedule.Signal_Cutoff):
        label = month_label(signal_cutoff)
        if calendar.get(label) != signal_cutoff or (label - signal_cutoff).days > 4:
            raise ValueError(f"Signal month {label.date()} was unfinished at {signal_cutoff.date()}")
        causal_monthly_rows(monthly, cutoff=signal_cutoff)
    for label in pd.date_range(requirements.earliest_price_label, requirements.last_signal, freq="ME"):
        if label not in calendar:
            raise ValueError(
                f"{strategy.strategy_id} requires prepared monthly history at {label.date()} "
                f"for its signal/sizing windows; run `{PREPARE_COMMAND}`"
            )
    sectors = (
        _sector_returns(inputs, requirements.earliest_sector_label, requirements.last_signal)
        if requirements.earliest_sector_label is not None else pd.DataFrame()
    )
    return LiveResearchHistory(sectors)
