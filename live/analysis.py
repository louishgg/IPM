"""Prepared-only causal strategy for the completed portfolio competition.

This module deliberately contains no acquisition client.  It consumes the
validated CSV contracts produced by :mod:`live.prepare`, keeps signal,
sizing, execution, and valuation timestamps distinct, and simulates the
historical live strategy using Yahoo observations with their genuine dates.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from portfolio_core.io import atomic_write_dataframe

from portfolio_core.corporate_actions import (
    CorporateActionResult,
    apply_corporate_actions,
)
from portfolio_core.accounting_config import (
    PortfolioAccountingConfig,
    spread_sensitivity_configs,
)
from portfolio_core.interval_valuation import value_one_share_through_events
from portfolio_core.portfolio_lifecycle import (
    LedgerAdvanceResult,
    advance_ledger,
)
from portfolio_core.rebalance_planner import (
    plan_rebalance_targets,
    round_half_away_from_zero,
)
from portfolio_core.accounting_ledger import (
    InfeasibleRebalanceError,
    InvalidRebalanceError,
    LedgerRebalancePlan,
    LedgerState,
    RebalanceResult as LedgerRebalanceResult,
    account_snapshot,
    execute_rebalance,
)
from portfolio_core.runtime_config import apply_runtime_settings
from portfolio_core.price_basis import price_basis_spec
from portfolio_core.simulation_assumptions import (
    build_simulation_assumptions,
    write_simulation_assumptions,
)
from portfolio_core.sector_assignments import (
    SECTOR_AUDIT_COLUMNS,
    sector_audit_fields,
    sector_rows_asof,
    trade_sector_audit_fields,
)
from portfolio_core.strategy_audit import (
    STRATEGY_SELECTION_AUDIT_COLUMNS,
    STRATEGY_SIGNAL_AUDIT_COLUMNS,
    TRADE_EXECUTION_AUDIT_COLUMNS,
    strategy_selection_fields,
    strategy_signal_fields,
    trade_execution_audit_fields,
)
from portfolio_core.strategies import (
    ExecutionDecision,
    Strategy,
    StrategyDecision,
    read_saved_strategy_identity,
    save_strategy_parameters,
)
from portfolio_core.strategies.research_hold import validate_hold, cap_carried_targets
from portfolio_core.strategies.research_execution import (
    require_positive_equity,
    research_accounting_config,
    sector_execution_residuals,
)
from portfolio_core.strategies.research_state import SelectedSectors

from . import analysis_data
from .research_history import LiveResearchHistory, prepare_live_research_history
from .config import LiveConfig
from .performance import DailyValuationData, build_daily_nav, performance_summary
from .performance_plotting import save_performance_plots
from .preparation_artifacts import PREPARE_COMMAND
from .strategy_universe import (
    DOWNLOAD_BENCHMARK_COMMAND,
    DOWNLOAD_PRICES_COMMAND,
    EVALUATION_PERIOD_COLUMNS,
    live_interval_eligible_assets,
    live_non_extinguished_assets,
)


CORPORATE_ACTION_AUDIT_COLUMNS = (
    "Rebalance_ID",
    "Phase",
    "Event_ID",
    "Event_Date",
    "Event_Type",
    "Continuity_Class",
    "Asset_ID",
    "Shares_Before",
    "Shares_After",
    "Successor_Units_JSON",
    "Cash_Per_From_Share",
    "Cash_Effect",
    "Interest_Effect",
    "Currency",
    "Restricted_Short_Proceeds_Transferred",
    "Restricted_Short_Proceeds_Released",
    "CVR_Units",
    "CVR_Units_Per_From_Share",
    "CVR_Base_Value",
    "CVR_Base_Value_Per_From_Share",
    "CVR_Max_Value",
    "CVR_Max_Value_Per_From_Share",
    "Fixed_Fee",
    "Spread_Cost",
    "Applied_To_Position",
    "Source_URLs_JSON",
)

_DECISION_AUDIT_PREFIX = (
    "Rebalance_ID",
    "Membership_Effective_Date",
    "Signal_Cutoff",
    "Signal_Source_Max_Date",
    "Sizing_Date",
    "Execution_Date",
    "Valuation_End",
    "Asset_ID",
    "Source_Ticker",
    "Yahoo_Ticker",
    *SECTOR_AUDIT_COLUMNS,
    "Strategy_ID",
    "Strategy_Version",
    *STRATEGY_SIGNAL_AUDIT_COLUMNS,
)

_DECISION_AUDIT_SUFFIX = (
    *STRATEGY_SELECTION_AUDIT_COLUMNS,
    "Applied_Rule",
)

_TRADE_AUDIT_PREFIX = (
    "Rebalance_ID",
    "Membership_Effective_Date",
    "Signal_Cutoff",
    "Sizing_Date",
    "Execution_Date",
    "Valuation_End",
    "Asset_ID",
    "Source_Ticker",
    "Yahoo_Ticker",
    *SECTOR_AUDIT_COLUMNS,
    "Strategy_ID",
    "Strategy_Version",
    *STRATEGY_SIGNAL_AUDIT_COLUMNS,
)

_TRADE_AUDIT_SUFFIX = (
    *STRATEGY_SELECTION_AUDIT_COLUMNS,
    "Sizing_Price",
    "Prior_Position_Value_At_Sizing",
    "Target_Position_Value_At_Sizing",
    "Target_Weight",
    "Sizing_Weight_Drift",
    "Overnight_Price_Return",
    "Sizing_Rounding_Share_Delta",
    "Turnover_Suppression_Share_Delta",
    "Feasibility_Share_Delta",
    "Constraint_Override",
    *TRADE_EXECUTION_AUDIT_COLUMNS,
)


def live_decision_audit_columns(strategy: Strategy) -> list[str]:
    return [*_DECISION_AUDIT_PREFIX, *strategy.signal_columns, *_DECISION_AUDIT_SUFFIX]


def live_trade_audit_columns(strategy: Strategy) -> list[str]:
    return [*_TRADE_AUDIT_PREFIX, *strategy.signal_columns, *_TRADE_AUDIT_SUFFIX]


def _causal_trailing_adv(
    market: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.Series:
    """Return each security's own trailing three-month median daily dollar volume."""

    selected = market.loc[market["Date"] <= pd.Timestamp(cutoff)]
    if selected.empty:
        raise RuntimeError(f"No prepared prices through {cutoff.date()}")
    selected = selected.loc[
        selected["Close"].notna()
        & selected["Volume"].notna()
        & selected["Close"].gt(0.0)
        & selected["Volume"].gt(0.0)
    ].copy()
    if selected.empty:
        return pd.Series(dtype=float)
    selected["Month"] = selected["Date"].dt.to_period("M")
    selected["Dollar_Volume"] = (
        selected["Close"].astype(float) * selected["Volume"].astype(float)
    )
    monthly = (
        selected.groupby(["Month", "Asset_ID"], sort=True)["Dollar_Volume"]
        .median()
        .unstack("Asset_ID")
        .sort_index()
    )
    return monthly.tail(3).median(axis=0, skipna=True).astype(float)


def _corporate_action_audit_rows(
    result: CorporateActionResult,
    event_interest_effects: Mapping[str, float],
    *,
    rebalance_id: str,
    phase: str,
) -> list[dict[str, Any]]:
    """Format one non-overlapping lifecycle result for the live audit."""

    audit_rows: list[dict[str, Any]] = []
    attributed_event_ids: set[str] = set()
    for shared_row in result.audit.to_dict("records"):
        event_id = str(shared_row["Event_ID"])
        from_asset_id = str(shared_row["From_Asset_ID"])
        interest_effect = (
            event_interest_effects.get(event_id, 0.0)
            if event_id not in attributed_event_ids
            else 0.0
        )
        attributed_event_ids.add(event_id)
        audit_rows.append({
            "Rebalance_ID": rebalance_id,
            "Phase": phase,
            "Event_ID": event_id,
            "Event_Date": pd.Timestamp(shared_row["Effective_Date"]),
            "Event_Type": str(shared_row["Event_Type"]),
            "Continuity_Class": str(shared_row["Continuity_Class"]),
            "Asset_ID": from_asset_id,
            "Shares_Before": float(shared_row["Shares_Before"]),
            "Shares_After": float(shared_row["Shares_After"]),
            "Successor_Units_JSON": str(shared_row["Successor_Units_JSON"]),
            "Cash_Per_From_Share": float(shared_row["Cash_Per_From_Share"]),
            "Cash_Effect": float(shared_row["Cash_Effect"]),
            "Interest_Effect": float(interest_effect),
            "Currency": str(shared_row["Currency"]),
            "Restricted_Short_Proceeds_Transferred": float(
                shared_row["Restricted_Proceeds_Transferred"]
            ),
            "Restricted_Short_Proceeds_Released": float(
                shared_row["Restricted_Proceeds_Released"]
            ),
            "CVR_Units": float(shared_row["CVR_Units"]),
            "CVR_Units_Per_From_Share": float(
                shared_row["CVR_Units_Per_From_Share"]
            ),
            "CVR_Base_Value": float(shared_row["CVR_Base_Value"]),
            "CVR_Base_Value_Per_From_Share": float(
                shared_row["CVR_Base_Value_Per_From_Share"]
            ),
            "CVR_Max_Value": float(shared_row["CVR_Max_Value"]),
            "CVR_Max_Value_Per_From_Share": float(
                shared_row["CVR_Max_Value_Per_From_Share"]
            ),
            "Fixed_Fee": float(shared_row["Fixed_Fee"]),
            "Spread_Cost": float(shared_row["Spread_Cost"]),
            "Applied_To_Position": bool(shared_row["Applied_To_Position"]),
            "Source_URLs_JSON": str(shared_row["Source_URLs_JSON"]),
        })
    return audit_rows


def _benchmark_price(
    benchmark: pd.DataFrame, date: pd.Timestamp, field: str
) -> float:
    rows = benchmark.loc[benchmark["Date"].eq(pd.Timestamp(date)), field]
    if len(rows) != 1:
        raise RuntimeError(
            f"Prepared benchmark is missing {field} on {pd.Timestamp(date).date()}. "
            f"Run `{DOWNLOAD_BENCHMARK_COMMAND}` then "
            f"`{PREPARE_COMMAND}`."
        )
    return float(rows.iloc[0])


@dataclass(frozen=True)
class _StrategyContext:
    inputs: analysis_data.LiveAnalysisInputs
    metadata: pd.DataFrame
    strategy: Strategy
    accounting: PortfolioAccountingConfig
    simulation_label: str
    simulation_fingerprint: str
    research_history: LiveResearchHistory


@dataclass
class _StrategyState:
    ledger: LedgerState
    previous_target_weights: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )
    selected_sectors: SelectedSectors = field(default_factory=SelectedSectors)


@dataclass
class _StrategyRecords:
    nav: list[dict[str, Any]] = field(default_factory=list)
    holdings: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    corporate_actions: list[dict[str, Any]] = field(default_factory=list)
    research: list[dict[str, Any]] = field(default_factory=list)
    sector_residuals: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _StrategyPeriodPlan:
    row: Any
    rebalance_id: str
    signal_cutoff: pd.Timestamp
    execution_date: pd.Timestamp
    sizing_date: pd.Timestamp
    end_date: pd.Timestamp
    dvol: pd.Series
    decision: StrategyDecision
    execution_decision: ExecutionDecision
    execution_members: set[str]
    execution_eligible: set[str]
    execution_sector_rows: pd.DataFrame


@dataclass(frozen=True)
class _LiveRebalance:
    plan: _StrategyPeriodPlan
    ledger_result: LedgerRebalanceResult
    execution_prices: pd.Series
    sizing_nav: float


@dataclass
class _PendingLivePeriod:
    period: Any
    start_state: LedgerState
    rebalance: _LiveRebalance | None = None
    interest_amount: float = 0.0
    cash_interest_credit: float = 0.0
    loan_interest_charge: float = 0.0
    corporate_actions: list[dict[str, Any]] = field(default_factory=list)

    def add_advance(self, advance: LedgerAdvanceResult) -> None:
        self.interest_amount += advance.interest_amount
        self.cash_interest_credit += advance.cash_interest_credit
        self.loan_interest_charge += advance.loan_interest_charge
        self.corporate_actions.extend(
            _corporate_action_audit_rows(
                advance.corporate_actions,
                advance.event_interest_effects,
                rebalance_id=str(self.period.Rebalance_ID),
                phase="holding_period" if self.rebalance is not None else "cash_interval",
            )
        )


def _advance_strategy_state(
    context: _StrategyContext,
    state: _StrategyState,
    end_date: pd.Timestamp,
    pending: _PendingLivePeriod | None,
) -> LedgerState:
    """Advance the sole live ledger cursor and attribute it to one period."""

    if state.ledger.state_date is None:
        raise RuntimeError("A live holding-period ledger must have a state date")
    if pending is None and pd.Timestamp(end_date) != state.ledger.state_date:
        raise RuntimeError("Ledger advancement requires an evaluation interval")
    advance = advance_ledger(
        state.ledger,
        pd.Timestamp(end_date),
        context.inputs.corporate_actions.events,
        context.inputs.corporate_actions.legs,
        context.inputs.corporate_actions.sources,
        accounting_config=context.accounting,
    )
    state.ledger = advance.state
    if pending is not None:
        pending.add_advance(advance)
    return state.ledger


def _required_sector_rows(
    inputs: analysis_data.LiveAnalysisInputs,
    date: pd.Timestamp,
    asset_ids: Iterable[str],
    *,
    context: str,
) -> pd.DataFrame:
    """Return exact-date sector rows and fail closed on incomplete coverage."""
    rows = sector_rows_asof(inputs.sector_assignments, date)
    required = {str(asset_id) for asset_id in asset_ids}
    missing = sorted(required - set(rows.index.astype(str)))
    if missing:
        raise RuntimeError(
            f"Missing point-in-time sector assignments for {context} on "
            f"{pd.Timestamp(date).date()}: {missing}"
        )
    return rows


def _execution_adjustment(
    plan: _StrategyPeriodPlan,
    asset_id: str,
) -> str:
    reason = str(
        plan.execution_decision.eligibility_reasons.get(asset_id, "")
    )
    if reason == "departed_before_execution" and asset_id in plan.execution_members:
        return "lifecycle_or_valuation_ineligible"
    if reason:
        return reason
    if asset_id not in plan.execution_eligible:
        return (
            "departed_before_execution"
            if asset_id not in plan.execution_members
            else "lifecycle_or_valuation_ineligible"
        )
    return ""


def _prepare_strategy_period(
    context: _StrategyContext,
    state: _StrategyState,
    records: _StrategyRecords,
    row: Any,
) -> _StrategyPeriodPlan:
    rebalance_id = str(row.Rebalance_ID)
    signal_cutoff = pd.Timestamp(row.Signal_Cutoff)
    execution_date = pd.Timestamp(row.Execution_Date)
    sizing_date = pd.Timestamp(row.Sizing_Date)
    end_date = pd.Timestamp(row.Valuation_End)
    dvol = _causal_trailing_adv(context.inputs.market_daily, signal_cutoff)
    signal_members = analysis_data.membership_as_of(
        context.inputs.membership, pd.Timestamp(row.Membership_Effective_Date)
    )
    signal_sector_rows = _required_sector_rows(
        context.inputs,
        signal_cutoff,
        signal_members,
        context=f"{rebalance_id} signal membership",
    )
    sector_codes = signal_sector_rows["GICS_Sector_Code"].astype(str).to_dict()
    decision_context = context.research_history.context(
        context.inputs, signal_cutoff, set(signal_members), sector_codes,
        state.previous_target_weights, state.selected_sectors,
    )
    decision = context.strategy.decide(decision_context)

    execution_members = set(
        analysis_data.membership_as_of(context.inputs.membership, execution_date)
    )
    lifecycle_eligible = live_interval_eligible_assets(
        context.inputs.market_daily,
        execution_members,
        execution_date=execution_date,
        execution_field=str(row.Execution_Field),
        valuation_end=end_date,
        valuation_field=str(row.Valuation_Field),
        corporate_action_events=context.inputs.corporate_actions.events,
        corporate_action_legs=context.inputs.corporate_actions.legs,
        corporate_action_sources=context.inputs.corporate_actions.sources,
    )
    required_candidates = set(signal_members) & live_non_extinguished_assets(
        execution_members,
        as_of_date=execution_date,
        corporate_action_events=context.inputs.corporate_actions.events,
        corporate_action_legs=context.inputs.corporate_actions.legs,
    )
    uncovered = sorted(required_candidates - set(lifecycle_eligible))
    if uncovered:
        raise RuntimeError(
            f"Missing required interval prices for {rebalance_id} "
            f"({execution_date.date()} -> {end_date.date()}): {uncovered}. "
            f"Run `{DOWNLOAD_PRICES_COMMAND}` then `{PREPARE_COMMAND}`."
        )
    hold_audit = {}
    if not decision.is_complete:
        if state.previous_target_weights.empty:
            raise RuntimeError(
                f"{context.strategy.strategy_id} signal/sizing and eligible holdings "
                f"unready for {rebalance_id} on {signal_cutoff.date()}; prepared monthly coverage cannot "
                f"form the requested portfolio. Run `{DOWNLOAD_PRICES_COMMAND}` "
                f"explicitly, then `{PREPARE_COMMAND}`."
            )
        eligible_hold = set(lifecycle_eligible) & set(signal_members)
        held = validate_hold(
            state.previous_target_weights, eligible_hold, sector_codes,
            context.strategy, signal_cutoff, context=decision_context,
            selected=state.selected_sectors,
        )
        held, hold_audit = cap_carried_targets(held, context.strategy, context.accounting)
        execution_decision = ExecutionDecision(
            held,
            tuple(held.index[held.gt(0)]),
            tuple(held.index[held.lt(0)]),
        )
    else:
        execution_decision = context.strategy.finalize_for_execution(
            decision, lifecycle_eligible,
        )
    basket = execution_decision.sector_basket
    selected_sectors = basket.selected if basket is not None else state.selected_sectors
    records.research.append({
        "Rebalance_ID": rebalance_id,
        "Signal_Cutoff": signal_cutoff,
        "Signal_Source_Date": decision_context.signal_source_max_date,
        "Status": "complete" if decision.is_complete else "eligible_prior_target_hold",
        "Available_Candidates": len(signal_members),
        "Eligible_Signal_Count": int(decision.signal_audit.Strategy_Eligible.sum()),
        "Final_Long_Count": len(execution_decision.final_long_asset_ids),
        "Final_Short_Count": len(execution_decision.final_short_asset_ids),
        "Selected_Long_Sectors_JSON": json.dumps(selected_sectors.longs),
        "Selected_Short_Sectors_JSON": json.dumps(selected_sectors.shorts),
        **decision.summary_metrics,
        **hold_audit,
    })
    execution_eligible = set(lifecycle_eligible)
    execution_sector_rows = _required_sector_rows(
        context.inputs,
        execution_date,
        lifecycle_eligible,
        context=f"{rebalance_id} eligible execution universe",
    )
    plan = _StrategyPeriodPlan(
        row=row,
        rebalance_id=rebalance_id,
        signal_cutoff=signal_cutoff,
        execution_date=execution_date,
        sizing_date=sizing_date,
        end_date=end_date,
        dvol=dvol,
        decision=decision,
        execution_decision=execution_decision,
        execution_members=execution_members,
        execution_eligible=execution_eligible,
        execution_sector_rows=execution_sector_rows,
    )
    for asset_id in decision.signal_audit.index.astype(str):
        selection_fields = strategy_selection_fields(
            decision,
            execution_decision,
            asset_id,
            execution_eligible=asset_id in execution_eligible,
            execution_adjustment=_execution_adjustment(plan, asset_id),
        )
        signal_side = str(selection_fields["Signal_Selected_Side"])
        final_side = str(selection_fields["Final_Selected_Side"])
        strategy_fields = strategy_signal_fields(
            context.strategy,
            decision,
            asset_id,
        )
        if not strategy_fields["Strategy_Eligible"]:
            applied_rule = "excluded_by_strategy_rule"
        elif final_side and signal_side == final_side:
            applied_rule = f"selected_{final_side.casefold()}"
        elif final_side:
            applied_rule = f"execution_refill_{final_side.casefold()}"
        elif signal_side:
            applied_rule = "removed_before_execution"
        else:
            applied_rule = "not_selected_by_strategy_rule"
        records.decisions.append({
            "Rebalance_ID": rebalance_id,
            "Membership_Effective_Date": pd.Timestamp(row.Membership_Effective_Date),
            "Signal_Cutoff": signal_cutoff,
            "Signal_Source_Max_Date": decision_context.signal_source_max_date,
            "Sizing_Date": sizing_date,
            "Execution_Date": execution_date,
            "Valuation_End": end_date,
            "Asset_ID": str(asset_id),
            "Source_Ticker": context.metadata.at[asset_id, "Source_Ticker"],
            "Yahoo_Ticker": context.metadata.at[asset_id, "Yahoo_Ticker"],
            **sector_audit_fields(
                signal_sector_rows,
                asset_id,
                signal_cutoff,
                context="decision-date",
            ),
            "Strategy_ID": context.strategy.strategy_id,
            "Strategy_Version": context.strategy.strategy_version,
            **strategy_fields,
            **selection_fields,
            "Applied_Rule": applied_rule,
        })
    return plan


def _execute_strategy_trades(
    context: _StrategyContext,
    records: _StrategyRecords,
    plan: _StrategyPeriodPlan,
    sizing_state: LedgerState,
    execution_state: LedgerState,
) -> _LiveRebalance:
    """Size at the sizing checkpoint and trade at the execution checkpoint.

    Research targets retry without turnover suppression when hard constraints
    require it; losing any selected long/short holding raises infeasibility.
    Neutrality uses sizing prices, with overnight price effects audited separately.
    """

    row = plan.row
    sizing_assets = sorted(
        set(plan.execution_decision.final_target_weights.index)
        | set(sizing_state.shares.index)
    )
    sizing_prices = analysis_data.price_series(
        context.inputs.market_daily,
        plan.sizing_date,
        str(row.Sizing_Field),
        sizing_assets,
        context=f"{plan.rebalance_id} sizing",
    )
    sizing_snapshot = account_snapshot(
        sizing_state,
        sizing_prices,
        date=plan.sizing_date,
    )
    require_positive_equity(sizing_snapshot.equity, plan.sizing_date)
    weights = plan.execution_decision.final_target_weights
    accounting = context.accounting
    long_count = int(weights.gt(0).sum())
    short_count = int(weights.lt(0).sum())
    accounting = research_accounting_config(accounting, long_count, short_count)
    if context.strategy.parameters.sector_neutral:
        target_sectors = plan.execution_sector_rows.GICS_Sector_Code.reindex(weights.index)
        if target_sectors.isna().any():
            raise ValueError("Research targets lack execution-date sectors")
        if not np.allclose(weights.groupby(target_sectors).sum(), 0, atol=1e-12, rtol=0):
            raise InfeasibleRebalanceError(f"{plan.execution_date}: ideal sector targets are not neutral")

    def make_targets(suppress: bool):
        return plan_rebalance_targets(
            sizing_state.shares, weights, sizing_prices, sizing_snapshot.equity,
            turnover_threshold=context.strategy.turnover_threshold,
            apply_turnover_threshold=suppress,
            whole_share_orders=accounting.whole_share_orders,
        )

    targets = make_targets(True)
    unsuppressed = make_targets(False)

    execution_assets = sorted(
        set(targets.applied_shares.index) | set(execution_state.shares.index)
    )
    execution_prices = analysis_data.price_series(
        context.inputs.market_daily,
        plan.execution_date,
        str(row.Execution_Field),
        execution_assets,
        context=f"{plan.rebalance_id} execution",
    )
    sizing_prices_for_audit = sizing_prices.reindex(execution_assets).fillna(execution_prices)
    sectors = plan.execution_sector_rows.GICS_Sector_Code.reindex(execution_assets).fillna("departed")
    require_positive_equity(
        account_snapshot(execution_state, execution_prices, date=plan.execution_date).equity,
        plan.execution_date,
    )
    def execute(selected):
        ledger_plan = LedgerRebalancePlan.from_series(
            selected.applied_shares.reindex(execution_assets).fillna(0.0),
            selected.target_weights.reindex(execution_assets).fillna(0.0),
            execution_prices, plan.dvol.reindex(execution_assets),
            plan.execution_eligible, plan.execution_date,
        )
        return execute_rebalance(
            execution_state, ledger_plan, accounting,
            apply_fees=True, apply_spread=True,
        )

    override = ""
    try:
        ledger_result = execute(targets)
        if ledger_result.feasibility_scale < 1 and not (
            targets.applied_shares.equals(unsuppressed.applied_shares)
        ):
            override = "hard_constraint_suppression_override"
        if context.strategy.parameters.sector_neutral:
            residuals = sector_execution_residuals(
                unsuppressed.applied_shares, execution_state.shares,
                ledger_result.applied_shares, ledger_result.feasibility_scale,
                sizing_prices_for_audit, sectors,
                whole_share_orders=accounting.whole_share_orders,
            )
            if not np.allclose(residuals.Suppression_Net_Dollars, 0, atol=1e-7, rtol=0):
                override = "neutrality_suppression_override"
    except (InfeasibleRebalanceError, InvalidRebalanceError):
        override = "hard_constraint_suppression_override"
    if override:
        targets = unsuppressed
        try:
            ledger_result = execute(targets)
        except (InfeasibleRebalanceError, InvalidRebalanceError) as exc:
            raise InfeasibleRebalanceError(
                f"{plan.execution_date}: {plan.rebalance_id} unsuppressed execution infeasible: {exc}"
            ) from exc
    applied = ledger_result.applied_shares.reindex(execution_assets).fillna(0)
    if set(applied.index[applied.gt(0)]) != set(weights.index[weights.gt(0)]) or (
        set(applied.index[applied.lt(0)]) != set(weights.index[weights.lt(0)])
    ):
        raise InfeasibleRebalanceError(
            f"{plan.rebalance_id}: live execution lost configured holdings"
        )
    retained_ineligible = sorted(
        set(ledger_result.state.shares.index.astype(str))
        - set(plan.execution_eligible)
    )
    if retained_ineligible:
        raise RuntimeError(
            "Live strategy retained target-ineligible assets "
            f"on {plan.execution_date.date()}: {retained_ineligible}"
        )

    execution_by_asset = {
        item.asset_id: item for item in ledger_result.executions
    }
    target_weights_all = targets.target_weights.reindex(execution_assets).fillna(0.0)
    weight_drift = targets.weight_drift.reindex(execution_assets).fillna(0.0)
    current_sizing_values = sizing_snapshot.position_values.reindex(
        execution_assets
    ).fillna(0.0)
    ideal_shares = (
        target_weights_all * sizing_snapshot.equity / sizing_prices_for_audit
    )
    rounded_shares = (
        round_half_away_from_zero(ideal_shares).astype(float)
        if accounting.whole_share_orders else ideal_shares
    )
    unsuppressed_shares = unsuppressed.applied_shares.reindex(execution_assets).fillna(0.0)
    requested_shares = targets.applied_shares.reindex(execution_assets).fillna(0.0)
    applied_shares = ledger_result.applied_shares.reindex(execution_assets).fillna(0.0)

    residuals = sector_execution_residuals(
        unsuppressed.applied_shares, execution_state.shares,
        applied_shares, ledger_result.feasibility_scale,
        sizing_prices_for_audit, sectors,
        whole_share_orders=accounting.whole_share_orders,
    )
    if context.strategy.parameters.sector_neutral and not np.allclose(
        residuals.Suppression_Net_Dollars, 0, atol=1e-7, rtol=0,
    ):
        raise RuntimeError("Neutral residual does not reconcile to mechanical rounding")
    # Sector totals include unchanged positions absent from the trade log.
    for code in sorted(set(sectors[target_weights_all.ne(0)])):
        group = sectors.eq(code)
        ideal = float((target_weights_all * sizing_snapshot.equity)[group].sum())
        rounded = float((unsuppressed_shares * sizing_prices_for_audit)[group].sum())
        requested = float((requested_shares * sizing_prices_for_audit)[group].sum())
        applied_sizing = float(residuals.loc[code, "Applied_Net_Dollars"])
        applied_execution = float((applied_shares * execution_prices)[group].sum())
        records.sector_residuals.append({
            "Rebalance_ID": plan.rebalance_id,
            "Execution_Date": plan.execution_date,
            "GICS_Sector_Code": str(code),
            "Sector_Neutral_Requested": bool(context.strategy.parameters.sector_neutral),
            "Target_Net_Dollars": ideal,
            "Unsuppressed_Rounded_Net_Dollars": rounded,
            "Requested_Net_Dollars": requested,
            "Applied_Net_At_Sizing_Dollars": applied_sizing,
            "Mechanical_Net_Dollars": float(residuals.loc[code, "Mechanical_Net_Dollars"]),
            "Suppression_Net_Dollars": float(residuals.loc[code, "Suppression_Net_Dollars"]),
            "Sizing_Rounding_Effect_Dollars": rounded - ideal,
            "Turnover_Suppression_Effect_Dollars": requested - rounded,
            "Feasibility_Effect_Dollars": applied_sizing - requested,
            "Overnight_Price_Effect_Dollars": applied_execution - applied_sizing,
            "Applied_Net_Dollars": applied_execution,
            "Applied_Net_Bps": 10000.0 * applied_execution / ledger_result.after.equity,
            "Feasibility_Scale": ledger_result.feasibility_scale,
            "Constraint_Override": override,
        })

    for asset_id, execution in execution_by_asset.items():
        before = execution.current_shares
        after = execution.applied_target_shares
        records.trades.append({
            "Rebalance_ID": plan.rebalance_id,
            "Membership_Effective_Date": pd.Timestamp(row.Membership_Effective_Date),
            "Signal_Cutoff": plan.signal_cutoff,
            "Sizing_Date": plan.sizing_date,
            "Execution_Date": plan.execution_date,
            "Valuation_End": plan.end_date,
            "Asset_ID": asset_id,
            "Source_Ticker": context.metadata.at[asset_id, "Source_Ticker"],
            "Yahoo_Ticker": context.metadata.at[asset_id, "Yahoo_Ticker"],
            **trade_sector_audit_fields(
                assignments=context.inputs.sector_assignments,
                execution_rows=plan.execution_sector_rows,
                execution_date=plan.execution_date,
                execution_eligible=asset_id in plan.execution_eligible,
                asset_id=asset_id,
                current_value=before,
                target_value=after,
                domain="live",
            ),
            "Strategy_ID": context.strategy.strategy_id,
            "Strategy_Version": context.strategy.strategy_version,
            **strategy_signal_fields(
                context.strategy,
                plan.decision,
                asset_id,
                require_ranked=True,
            ),
            **strategy_selection_fields(
                plan.decision,
                plan.execution_decision,
                asset_id,
                execution_eligible=asset_id in plan.execution_eligible,
                execution_adjustment=_execution_adjustment(plan, asset_id),
            ),
            "Sizing_Price": float(sizing_prices_for_audit.loc[asset_id]),
            "Prior_Position_Value_At_Sizing": float(
                current_sizing_values.loc[asset_id]
            ),
            "Target_Position_Value_At_Sizing": float(
                target_weights_all.loc[asset_id] * sizing_snapshot.equity
            ),
            "Target_Weight": float(target_weights_all.loc[asset_id]),
            "Sizing_Weight_Drift": float(weight_drift.loc[asset_id]),
            "Overnight_Price_Return": float(
                execution_prices.loc[asset_id] / sizing_prices_for_audit.loc[asset_id] - 1
            ),
            "Sizing_Rounding_Share_Delta": float(
                rounded_shares.loc[asset_id] - ideal_shares.loc[asset_id]
            ),
            "Turnover_Suppression_Share_Delta": float(
                requested_shares.loc[asset_id] - unsuppressed_shares.loc[asset_id]
            ),
            "Feasibility_Share_Delta": float(
                applied_shares.loc[asset_id] - requested_shares.loc[asset_id]
            ),
            "Constraint_Override": override,
            **trade_execution_audit_fields(execution),
        })

    return _LiveRebalance(
        plan=plan,
        ledger_result=ledger_result,
        execution_prices=execution_prices,
        sizing_nav=sizing_snapshot.equity,
    )


def _record_strategy_period(
    context: _StrategyContext,
    records: _StrategyRecords,
    pending: _PendingLivePeriod,
    end_state: LedgerState,
) -> None:
    outcome = pending.rebalance
    period = pending.period
    if end_state.state_date != period.Period_End:
        raise RuntimeError(
            f"Live ledger ended an interval on {end_state.state_date}, "
            f"expected {period.Period_End}"
        )
    records.corporate_actions.extend(pending.corporate_actions)
    end_prices = analysis_data.price_series(
        context.inputs.market_daily,
        period.Period_End,
        str(period.End_Field),
        end_state.shares.index,
        context=f"{period.Rebalance_ID or 'initial cash'} valuation",
    )
    end_snapshot = account_snapshot(end_state, end_prices, date=period.Period_End)
    require_positive_equity(end_snapshot.equity, period.Period_End)
    end_nav = end_snapshot.equity
    ledger_result = outcome.ledger_result if outcome is not None else None
    post_trade = (
        ledger_result.after if ledger_result is not None
        else account_snapshot(pending.start_state, pd.Series(dtype=float), date=period.Period_Start)
    )
    start_nav = ledger_result.before.equity if ledger_result is not None else post_trade.equity
    period_return = end_nav / start_nav - 1.0
    benchmark_start = _benchmark_price(
        context.inputs.benchmark_daily,
        period.Period_Start,
        str(period.Start_Field),
    )
    benchmark_end = _benchmark_price(
        context.inputs.benchmark_daily,
        period.Period_End,
        str(period.End_Field),
    )
    benchmark_return = benchmark_end / benchmark_start - 1.0
    records.nav.append({
        **{column: getattr(period, column) for column in EVALUATION_PERIOD_COLUMNS},
        "Sizing_NAV": outcome.sizing_nav if outcome is not None else np.nan,
        "Start_NAV": start_nav,
        "Post_Trade_NAV": post_trade.equity,
        "End_NAV": end_nav,
        "Period_Return": period_return,
        "Benchmark_Return": benchmark_return,
        "Order_Count": ledger_result.order_count if ledger_result is not None else 0,
        "Fixed_Fees": ledger_result.fixed_fees if ledger_result is not None else 0.0,
        "Spread_Cost": ledger_result.spread_cost if ledger_result is not None else 0.0,
        "Turnover": ledger_result.turnover if ledger_result is not None else 0.0,
        "Interest": pending.interest_amount,
        "Cash_Interest_Credit": pending.cash_interest_credit,
        "Loan_Interest_Charge": pending.loan_interest_charge,
        "Post_Trade_Cash": pending.start_state.cash,
        "Post_Trade_Signed_Market_Value": float(
            post_trade.position_values.sum()
        ),
        "Post_Trade_Gross_Market_Value": float(
            post_trade.position_values.abs().sum()
        ),
        "Restricted_Short_Proceeds": (
            post_trade.restricted_short_proceeds
        ),
        "Free_Cash": post_trade.free_cash,
        "Loan": post_trade.loan,
        "Gross_Exposure": post_trade.gross_exposure,
        "Maximum_Position_Weight": (
            post_trade.maximum_position_weight
        ),
        "Position_Count": post_trade.position_count,
        "Long_Count": post_trade.long_count,
        "Short_Count": post_trade.short_count,
        "End_Cash": end_snapshot.cash,
        "End_Signed_Market_Value": float(end_snapshot.position_values.sum()),
        "End_Gross_Market_Value": float(
            end_snapshot.position_values.abs().sum()
        ),
        "End_Restricted_Short_Proceeds": (
            end_snapshot.restricted_short_proceeds
        ),
        "End_Free_Cash": end_snapshot.free_cash,
        "End_Loan": end_snapshot.loan,
        "End_Position_Count": end_snapshot.position_count,
        "End_Long_Count": end_snapshot.long_count,
        "End_Short_Count": end_snapshot.short_count,
        "Feasibility_Scale": ledger_result.feasibility_scale if ledger_result is not None else 1.0,
        "Feasibility_Adjustment": ledger_result.adjustment_reason if ledger_result is not None else "",
        "Label": context.simulation_label,
    })
    if outcome is None:
        return
    plan = outcome.plan
    row = plan.row
    post_trade_shares = ledger_result.state.shares
    post_trade_values = ledger_result.after.position_values
    actual_weights = post_trade_values / ledger_result.after.equity
    for asset_id in post_trade_shares.index:
        interval_value = value_one_share_through_events(
            context.inputs.corporate_actions.events,
            context.inputs.corporate_actions.legs,
            context.inputs.corporate_actions.sources,
            asset_id,
            plan.execution_date,
            plan.end_date,
            lambda successor: analysis_data.price_series(
                context.inputs.market_daily,
                plan.end_date,
                str(row.Valuation_Field),
                [successor],
                context=(
                    f"{plan.rebalance_id} corporate-action settlement for "
                    f"predecessor {asset_id}"
                ),
            ).loc[successor],
        )
        if interval_value is not None:
            _, exact_end_value = interval_value
            end_value_per_share = float(exact_end_value)
            valuation_source = "corporate_action_settlement"
        else:
            end_value_per_share = float(end_prices.loc[asset_id])
            valuation_source = "yahoo"
        stock_return = (
            end_value_per_share / float(outcome.execution_prices.loc[asset_id]) - 1.0
        )
        records.holdings.append({
            "Rebalance_ID": plan.rebalance_id,
            "Date": plan.execution_date,
            "Next_Date": plan.end_date,
            "Asset_ID": asset_id,
            "Source_Ticker": context.metadata.at[asset_id, "Source_Ticker"],
            "Yahoo_Ticker": context.metadata.at[asset_id, "Yahoo_Ticker"],
            **sector_audit_fields(
                plan.execution_sector_rows,
                asset_id,
                plan.execution_date,
                context="execution-date",
            ),
            "Shares": float(post_trade_shares.loc[asset_id]),
            "Execution_Price": float(outcome.execution_prices.loc[asset_id]),
            "End_Price": end_value_per_share,
            "End_Value_Source": valuation_source,
            "Position_Value": float(post_trade_values.loc[asset_id]),
            "Weight": float(actual_weights.loc[asset_id]),
            "Stock_Return": stock_return,
        })

def _record_outside_position_actions(
    context: _StrategyContext,
    records: _StrategyRecords,
) -> None:
    if context.inputs.corporate_actions.events.empty:
        return
    first_event = pd.to_datetime(
        context.inputs.corporate_actions.events["Effective_Date"]
    ).min()
    last_event = pd.to_datetime(
        context.inputs.corporate_actions.events["Effective_Date"]
    ).max()
    outside_result = apply_corporate_actions(
        {},
        {},
        context.inputs.corporate_actions.events,
        context.inputs.corporate_actions.legs,
        context.inputs.corporate_actions.sources,
        start_exclusive=first_event - pd.Timedelta(days=1),
        end_inclusive=last_event,
    )
    recorded_keys = {
        (str(row["Event_ID"]), str(row["Asset_ID"]))
        for row in records.corporate_actions
    }
    outside_rows = _corporate_action_audit_rows(
        outside_result,
        {},
        rebalance_id="",
        phase="outside_position_lifecycle",
    )
    records.corporate_actions.extend(
        row
        for row in outside_rows
        if (str(row["Event_ID"]), str(row["Asset_ID"])) not in recorded_keys
    )


def _assemble_strategy_result(
    context: _StrategyContext,
    records: _StrategyRecords,
) -> analysis_data.LiveAnalysisResult:
    nav = pd.DataFrame(records.nav)
    nav["Period_Start"] = pd.to_datetime(nav["Period_Start"])
    nav["Period_End"] = pd.to_datetime(nav["Period_End"])
    holdings = pd.DataFrame(records.holdings).sort_values(
        ["Date", "Asset_ID"], kind="stable"
    ).reset_index(drop=True)
    trade_columns = live_trade_audit_columns(context.strategy)
    decision_columns = live_decision_audit_columns(context.strategy)
    for label, audit_records, columns in (
        ("live trade audit", records.trades, trade_columns),
        ("live decision audit", records.decisions, decision_columns),
    ):
        expected = set(columns)
        for record in audit_records:
            if set(record) != expected:
                raise RuntimeError(
                    f"{label} record does not match its strategy-derived schema: "
                    f"missing={sorted(expected - set(record))}, "
                    f"extra={sorted(set(record) - expected)}"
                )
    trades = pd.DataFrame(records.trades, columns=trade_columns).sort_values(
        ["Execution_Date", "Asset_ID"], kind="stable"
    ).reset_index(drop=True)
    decisions = pd.DataFrame(
        records.decisions,
        columns=decision_columns,
    ).sort_values(
        ["Execution_Date", "Asset_ID"], kind="stable"
    ).reset_index(drop=True)
    valuation_data = DailyValuationData.from_frames(
        context.inputs.market_daily, context.inputs.benchmark_daily,
        context.inputs.corporate_actions,
        start=nav.Period_Start.iloc[0], end=nav.Period_End.iloc[-1],
    )
    daily_nav = build_daily_nav(nav, holdings, trades, valuation_data, context.accounting)
    performance = performance_summary(
        nav, daily_nav, context.accounting, context.simulation_label,
        context.inputs.price_basis, context.simulation_fingerprint,
    )
    corporate_actions = pd.DataFrame(
        records.corporate_actions,
        columns=CORPORATE_ACTION_AUDIT_COLUMNS,
    ).sort_values(["Event_Date", "Asset_ID"], kind="stable").reset_index(drop=True)
    return analysis_data.LiveAnalysisResult(
        nav,
        holdings,
        trades,
        decisions,
        performance,
        corporate_actions,
        daily_nav=daily_nav,
        research_diagnostics=pd.DataFrame(records.research),
        sector_residuals=pd.DataFrame(records.sector_residuals),
    )


def run_strategy_analysis(
    inputs: analysis_data.LiveAnalysisInputs,
    config: LiveConfig,
    *,
    research_history: LiveResearchHistory | None = None,
) -> analysis_data.LiveAnalysisResult:
    """Evaluate one continuous account across cash and invested intervals."""
    accounting = config.accounting
    assumptions = build_simulation_assumptions(
        accounting,
        inputs.price_basis,
    )
    context = _StrategyContext(
        inputs=inputs,
        metadata=inputs.metadata.set_index("Asset_ID", drop=False),
        strategy=config.strategy,
        accounting=accounting,
        simulation_label=(
            "canonical monthly replay; "
            "not actual competition-account performance"
        ),
        simulation_fingerprint=str(assumptions["simulation_fingerprint"]),
        research_history=(
            research_history or prepare_live_research_history(
                inputs, config.strategy,
            )
        ),
    )
    periods = list(inputs.evaluation_periods.itertuples(index=False))
    state = _StrategyState(ledger=LedgerState.initial(
        accounting, state_date=periods[0].Period_Start,
    ))
    records = _StrategyRecords()
    invested = {row.Rebalance_ID: row for row in periods if row.Period_Type == "invested"}
    pending = (
        _PendingLivePeriod(periods[0], state.ledger)
        if periods[0].Period_Type == "cash" else None
    )
    for row in inputs.schedule.itertuples(index=False):
        plan = _prepare_strategy_period(context, state, records, row)
        if pending is not None:
            if pending.period.Period_End != plan.execution_date:
                raise RuntimeError(
                    "Adjacent live periods do not share an execution boundary"
                )
        sizing_state = _advance_strategy_state(
            context, state, plan.sizing_date, pending,
        )
        execution_state = _advance_strategy_state(
            context, state, plan.execution_date, pending,
        )
        if pending is not None:
            _record_strategy_period(context, records, pending, execution_state)

        rebalance = _execute_strategy_trades(
            context,
            records,
            plan,
            sizing_state,
            execution_state,
        )
        state.ledger = rebalance.ledger_result.state
        state.previous_target_weights = (
            plan.execution_decision.final_target_weights.copy()
        )
        if plan.execution_decision.sector_basket is not None:
            state.selected_sectors = plan.execution_decision.sector_basket.selected
        pending = _PendingLivePeriod(
            period=invested[plan.rebalance_id],
            start_state=state.ledger,
            rebalance=rebalance,
        )

    if pending is None:
        raise RuntimeError("Live decision schedule cannot be empty")
    end_state = _advance_strategy_state(
        context,
        state,
        pending.period.Period_End,
        pending,
    )
    _record_strategy_period(context, records, pending, end_state)
    _record_outside_position_actions(context, records)
    return _assemble_strategy_result(context, records)


def run_strategy(
    config: LiveConfig,
) -> analysis_data.LiveAnalysisResult:
    """Load prepared data and calculate a complete strategy result."""
    inputs = analysis_data.load_analysis_inputs(config.paths, config=config)
    history = prepare_live_research_history(inputs, config.strategy)
    base = run_strategy_analysis(inputs, config, research_history=history)
    case_results = {"base": base}
    sensitivity_configs = spread_sensitivity_configs(config.accounting)
    for case, accounting in sensitivity_configs.items():
        if case == "base":
            continue
        case_results[case] = run_strategy_analysis(
            inputs, replace(config, accounting=accounting), research_history=history,
        )
    sensitivity_rows = []
    for case, accounting in sensitivity_configs.items():
        case_result = case_results[case]
        strategy_row = case_result.performance.loc[
            case_result.performance["Series"].eq("Live_Strategy")
        ].iloc[0]
        sensitivity_rows.append({
            "Sensitivity_Case": case,
            "Liquidity_Coefficient_Bps": float(
                accounting.transaction_costs.liquidity_bps
            ),
            "Live_PctReturn": 100.0 * float(strategy_row["Cumulative_Return"]),
            "Price_Basis": str(strategy_row["Price_Basis"]),
            "Simulation_Fingerprint": str(strategy_row["Simulation_Fingerprint"]),
            "Drives_Official_Ranking": case == "base",
        })
    return replace(
        base,
        spread_sensitivity=pd.DataFrame(sensitivity_rows),
    )


def save_strategy_result(
    result: analysis_data.LiveAnalysisResult,
    config: LiveConfig,
) -> None:
    """Persist one fully calculated live strategy result."""
    if not isinstance(config.strategy, Strategy):
        raise ValueError("Live strategy saving requires a resolved strategy")
    analysis_data.save_analysis_result(result, config.paths)
    save_performance_plots(result.daily_nav, config.paths)
    for frame, path in (
        (result.research_diagnostics, config.paths.strategy_research_diagnostics_csv),
        (result.sector_residuals, config.paths.strategy_sector_residuals_csv),
    ):
        atomic_write_dataframe(
            frame, path, index=False, date_format="%Y-%m-%d",
            float_format="%.17g", lineterminator="\n",
        )
    save_strategy_parameters(
        config.strategy,
        config.paths.strategy_parameters_json,
    )
    write_simulation_assumptions(
        config.paths.simulation_assumptions_json,
        config.accounting,
        price_basis_spec("live"),
    )


def run_analysis(stage: str, *, config: LiveConfig):
    """Execute one prepared-only live analysis stage."""
    from .brinson_attribution import (
        remove_saved_brinson_result,
        run_brinson,
        save_brinson_result,
    )

    if stage not in {"strategy", "brinson", "all"}:
        raise ValueError(f"Unsupported live analysis stage: {stage!r}")
    if stage != "brinson" and not isinstance(config.strategy, Strategy):
        raise ValueError("Live analysis requires an explicitly resolved strategy")
    apply_runtime_settings()
    if stage == "strategy":
        strategy_result = run_strategy(config)
        remove_saved_brinson_result(config.paths)
        save_strategy_result(strategy_result, config)
        return strategy_result
    if stage == "brinson":
        read_saved_strategy_identity(
            config.paths.strategy_id, config.paths.strategy_parameters_json,
        )
        attribution_result = run_brinson(config)
        save_brinson_result(attribution_result, config.paths)
        return attribution_result
    if stage == "all":
        strategy_result = run_strategy(config)
        attribution_result = run_brinson(
            config,
            strategy=strategy_result,
        )
        save_strategy_result(strategy_result, config)
        save_brinson_result(attribution_result, config.paths)
        return strategy_result, attribution_result


__all__ = [
    "live_decision_audit_columns",
    "live_trade_audit_columns",
    "run_analysis",
    "run_strategy",
    "run_strategy_analysis",
    "save_strategy_result",
]
