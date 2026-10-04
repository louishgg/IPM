"""Backtest accounting engine for an explicitly supplied portfolio strategy."""
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import pandas as pd
from portfolio_core.corporate_actions import parse_exact_number
from portfolio_core.sector_assignments import (
    SECTOR_AUDIT_COLUMNS,
    prior_sector_audit_fields,
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
from portfolio_core.accounting_config import (
    DEFAULT_ACCOUNTING_CONFIG,
    PortfolioAccountingConfig,
)
from portfolio_core.accounting_ledger import (
    AccountSnapshot,
    LedgerState,
    RebalanceResult as LedgerRebalanceResult,
    TradeExecution,
    account_snapshot,
    execute_forced_liquidation,
)
from portfolio_core.strategies.strategy_contract import (
    ExecutionDecision,
    Strategy,
    StrategyDecision,
    StrategyDecisionContext,
)
from portfolio_core.strategies.research_state import SelectedSectors

from .data_loading import BacktestDataset


from portfolio_core.portfolio_lifecycle import (
    advance_ledger,
    asset_has_interval_event,
    event_delivery_execution_candidates,
    event_valuation_result,
    eligible_assets_for_interval,
    finite_positive_price,
    period_has_position_event,
    prepared_event_tables,
)


_HOLDINGS_COLUMNS = [
    "Date",
    "Next_Date",
    "Asset_ID",
    "Ticker",
    *SECTOR_AUDIT_COLUMNS,
    "Weight",
    "Stock_Return",
]

_AUDIT_PREFIX = [
    "Signal_Cutoff",
    "Execution_Date",
    "Valuation_End",
    "Asset_ID",
    "Ticker",
    *SECTOR_AUDIT_COLUMNS,
    "Strategy_ID",
    "Strategy_Version",
    *STRATEGY_SIGNAL_AUDIT_COLUMNS,
]

_DECISION_AUDIT_SUFFIX = [
    *STRATEGY_SELECTION_AUDIT_COLUMNS,
    "Applied_Target_Weight",
    "Decision_Complete",
    "Used_Hold_Logic",
    "Applied_Rule",
]

_TRADE_AUDIT_SUFFIX = [
    *STRATEGY_SELECTION_AUDIT_COLUMNS,
    "Current_Capital",
    "Current_Weight",
    "Target_Weight",
    "Weight_Drift",
    "Turnover_Threshold",
    "Current_Position_Value",
    "Requested_Target_Position_Value",
    "Target_Position_Value",
    *TRADE_EXECUTION_AUDIT_COLUMNS,
    "Turnover_Contribution",
    "Execution_Reason",
    "Trigger_Event_ID",
]


def decision_audit_columns(strategy: Strategy) -> list[str]:
    """Return the ordered decision-audit contract for one strategy family."""
    return [
        *_AUDIT_PREFIX,
        *strategy.signal_columns,
        *_DECISION_AUDIT_SUFFIX,
    ]


def trade_audit_columns(strategy: Strategy) -> list[str]:
    """Return the ordered trade-audit contract for one strategy family."""
    return [
        *_AUDIT_PREFIX,
        *strategy.signal_columns,
        *_TRADE_AUDIT_SUFFIX,
    ]


@dataclass
class _BacktestAuditCollector:
    """Internal selected-run records materialized by the analysis layer."""

    strategy: Strategy
    decision_records: list[dict[str, object]] = field(default_factory=list)
    trade_records: list[dict[str, object]] = field(default_factory=list)
    research_records: list[dict[str, object]] = field(default_factory=list)
    sector_records: list[dict[str, object]] = field(default_factory=list)

    def decisions_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            self.decision_records,
            columns=decision_audit_columns(self.strategy),
        )
        if frame.empty:
            return frame
        return frame.sort_values(
            ["Signal_Cutoff", "Asset_ID"], kind="stable"
        ).reset_index(drop=True)

    def trades_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            self.trade_records,
            columns=trade_audit_columns(self.strategy),
        )
        if frame.empty:
            return frame
        return frame.sort_values(
            ["Execution_Date", "Asset_ID"], kind="stable"
        ).reset_index(drop=True)


@dataclass(frozen=True)
class _BacktestContext:
    backtest_data: BacktestDataset
    strategy: Strategy
    apply_turnover_threshold: bool
    apply_fees: bool
    apply_spread: bool
    return_holdings: bool
    return_diagnostics: bool
    debug: bool
    accounting_config: PortfolioAccountingConfig
    corporate_action_audit_records: list[dict[str, object]] | None
    strategy_audit: _BacktestAuditCollector | None
    sector_returns: pd.DataFrame | None = None


@dataclass
class _BacktestRunState:
    ledger: LedgerState
    current_capital: float
    portfolio_values: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )
    target_weights: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )
    selected_sectors: SelectedSectors = field(default_factory=SelectedSectors)
    started: bool = False
    diagnostics: list[dict[str, object]] = field(default_factory=list)
    holdings_records: list[dict[str, object]] | None = None


@dataclass(frozen=True)
class _PeriodDecision:
    t_0: pd.Timestamp
    t_1: pd.Timestamp
    t_0_eff: pd.Timestamp
    dollar_volume: pd.Series
    decision: StrategyDecision
    execution_decision: ExecutionDecision
    signal_sector_rows: pd.DataFrame
    execution_sector_rows: pd.DataFrame
    interval_eligible_asset_ids: frozenset[str]
    used_hold_logic: bool


@dataclass(frozen=True)
class _BacktestRebalance:
    ledger: LedgerRebalanceResult
    execution_prices: pd.Series
    current_values_t0: pd.Series
    requested_values_t0: pd.Series
    applied_values_t0: pd.Series
    current_weights_t0: pd.Series
    target_weights_t0: pd.Series
    trade_values: pd.Series


@dataclass(frozen=True)
class _LifecycleResult:
    ledger_state: LedgerState
    end_snapshot: AccountSnapshot
    current_position_values: pd.Series
    interest_amount: float
    cash_interest_credit: float
    loan_interest_charge: float
    stock_returns_t1: pd.Series
    has_position_event: bool
    forced_exit_order_count: int = 0
    forced_exit_fixed_fees: float = 0.0
    forced_exit_spread_cost: float = 0.0
    forced_exit_turnover: float = 0.0


def _prepare_period_decision(
    context: _BacktestContext,
    state: _BacktestRunState,
    t_0: pd.Timestamp,
    t_1: pd.Timestamp,
) -> _PeriodDecision | None:
    """Construct one period's causal universe and portfolio decision."""
    backtest_data = context.backtest_data

    if t_0 not in backtest_data.data_close.index:
        raise ValueError(f"Missing momentum decision endpoint {t_0}")
    t_0_eff = t_0

    dvol_slice = backtest_data.rolling_dollar_vol.loc[t_0_eff]

    closest_dt = backtest_data.pit_matrix.index[
        backtest_data.pit_matrix.index <= t_0_eff
    ][-1]
    active_asset_ids = backtest_data.pit_matrix.loc[closest_dt][
        backtest_data.pit_matrix.loc[closest_dt]
    ].index.tolist()

    membership_asset_ids = sorted(
        asset_id
        for asset_id in set(active_asset_ids) & set(backtest_data.data_close.columns)
    )
    interval_eligible_asset_ids = eligible_assets_for_interval(
        backtest_data,
        membership_asset_ids,
        pd.Timestamp(t_0_eff),
        pd.Timestamp(t_1),
    )

    signal_sector_rows = sector_rows_asof(
        backtest_data.sector_assignments,
        t_0_eff,
    )
    missing_signal_sectors = sorted(
        set(interval_eligible_asset_ids)
        - set(signal_sector_rows.index.astype(str))
    )
    if missing_signal_sectors:
        raise RuntimeError(
            "Missing point-in-time sector assignments for signal candidates "
            f"on {pd.Timestamp(t_0_eff).date()}: {missing_signal_sectors}"
        )
    strategy_context = StrategyDecisionContext(
        close_history=backtest_data.data_close.loc[:t_0_eff],
        volume_history=backtest_data.data_volume.loc[:t_0_eff],
        candidate_asset_ids=tuple(sorted(interval_eligible_asset_ids)),
        sector_code_by_asset_id=signal_sector_rows["GICS_Sector_Code"].astype(str).to_dict(),
        previous_target_weights=state.target_weights,
        signal_cutoff=t_0_eff,
        signal_source_max_date=t_0_eff,
        previous_selected_sectors=state.selected_sectors,
        sector_returns=context.sector_returns.loc[:t_0_eff] if context.sector_returns is not None else pd.DataFrame(),
    )
    decision = context.strategy.decide(strategy_context)
    execution_decision = context.strategy.finalize_for_execution(
        decision,
        interval_eligible_asset_ids,
    )

    if context.strategy_audit is not None:
        context.strategy_audit.research_records.append({
            "Date": t_0, "Status": "complete" if decision.is_complete else "incomplete",
            "Actual_Started": state.started,
            **decision.summary_metrics,
        })

    used_hold_logic = False
    if not decision.is_complete:
        if state.target_weights.empty:
            if state.started:
                from portfolio_core.accounting_ledger import InfeasibleRebalanceError
                raise InfeasibleRebalanceError(f"{t_0}: activated momentum account has no feasible prior target")
            return None
        used_hold_logic = True
    else:
        state.target_weights = execution_decision.final_target_weights
        if execution_decision.sector_basket is not None:
            state.selected_sectors = execution_decision.sector_basket.selected
        if not state.started:
            state.portfolio_values.loc[t_0] = state.current_capital
            state.started = True

    # Holding fallback never overrides lifecycle or valuation eligibility.
    # Any former target outside this interval's canonical consumer set is
    # forced to zero at the current rebalance.
    state.target_weights = state.target_weights.loc[
        state.target_weights.index.astype(str).isin(
            interval_eligible_asset_ids
        )
    ]
    if used_hold_logic:
        from portfolio_core.strategies.research_hold import validate_hold, cap_carried_targets
        state.target_weights = validate_hold(
            state.target_weights, interval_eligible_asset_ids,
            strategy_context.sector_code_by_asset_id, context.strategy, t_0,
            context=strategy_context, selected=state.selected_sectors,
        )
        state.target_weights, hold_audit = cap_carried_targets(
            state.target_weights, context.strategy, context.accounting_config,
        )
        if context.strategy_audit is not None:
            context.strategy_audit.research_records.append({
                "Date": t_0, "Status": "eligible_prior_target_hold",
                "Used_Hold_Logic": True,
                **hold_audit,
            })

    execution_sector_rows = sector_rows_asof(
        backtest_data.sector_assignments,
        t_0,
    )
    return _PeriodDecision(
        t_0=pd.Timestamp(t_0),
        t_1=pd.Timestamp(t_1),
        t_0_eff=pd.Timestamp(t_0_eff),
        dollar_volume=dvol_slice,
        decision=decision,
        execution_decision=execution_decision,
        signal_sector_rows=signal_sector_rows,
        execution_sector_rows=execution_sector_rows,
        interval_eligible_asset_ids=interval_eligible_asset_ids,
        used_hold_logic=used_hold_logic,
    )


def _process_rebalance_and_costs(
    context: _BacktestContext,
    state: _BacktestRunState,
    period: _PeriodDecision,
) -> _BacktestRebalance:
    """Plan strategy-owned turnover and execute through the shared ledger."""
    asset_ids = sorted(
        set(state.target_weights.index).union(
            state.ledger.shares.index
        )
    )
    execution_prices = context.backtest_data.data_close.loc[
        period.t_0, asset_ids
    ].astype(float)
    invalid_prices = (
        execution_prices.isna()
        | ~np.isfinite(execution_prices)
        | execution_prices.le(0.0)
    )
    if invalid_prices.any():
        raise RuntimeError(
            "Cannot rebalance with missing execution prices on "
            f"{period.t_0.date()}: "
            f"{execution_prices.index[invalid_prices].tolist()}"
        )
    ledger_state = state.ledger
    if ledger_state.state_date is None:
        ledger_state = LedgerState.initial(
            context.accounting_config,
            state_date=period.t_0,
        )
    elif ledger_state.state_date != period.t_0:
        raise RuntimeError(
            "Backtest ledger date does not match the rebalance date: "
            f"{ledger_state.state_date} != {period.t_0}"
        )
    before = account_snapshot(ledger_state, execution_prices, date=period.t_0)
    from .research_execution import execute_research_rebalance
    targets, ledger_result = execute_research_rebalance(
        context, state, period, ledger_state, execution_prices, before,
    )
    assets = sorted(
        set(execution_prices.index)
        | set(ledger_result.requested_shares.index)
        | set(ledger_result.applied_shares.index)
    )
    execution_prices = execution_prices.reindex(assets)
    current_shares = ledger_state.shares.reindex(assets).fillna(0.0)
    requested_shares = ledger_result.requested_shares.reindex(assets).fillna(0.0)
    applied_shares = ledger_result.applied_shares.reindex(assets).fillna(0.0)
    curr_vals_t0 = current_shares * execution_prices
    requested_vals_t0 = requested_shares * execution_prices
    applied_vals_t0 = applied_shares * execution_prices
    trade_shares = applied_shares - current_shares
    trade_vals = trade_shares * execution_prices
    curr_w_aligned = targets.current_weights.reindex(assets).fillna(0.0)
    targ_w_aligned = targets.target_weights.reindex(assets).fillna(0.0)
    return _BacktestRebalance(
        ledger=ledger_result,
        execution_prices=execution_prices,
        current_values_t0=curr_vals_t0,
        requested_values_t0=requested_vals_t0,
        applied_values_t0=applied_vals_t0,
        current_weights_t0=curr_w_aligned,
        target_weights_t0=targ_w_aligned,
        trade_values=trade_vals,
    )


def _forced_exit_sector_fields(
    context: _BacktestContext,
    candidate: pd.Series,
) -> dict[str, object]:
    """Use the latest causal child classification, then its delivering parent."""

    execution_date = pd.Timestamp(candidate["Execution_Date"])
    child = str(candidate["Asset_ID"])
    raw_parents = candidate["From_Asset_IDs"]
    parents = (
        tuple(str(value) for value in raw_parents)
        if isinstance(raw_parents, (list, tuple))
        else (str(raw_parents),)
    )
    failures: list[str] = []
    for asset_id in dict.fromkeys((child, *parents)):
        try:
            return prior_sector_audit_fields(
                context.backtest_data.sector_assignments,
                asset_id,
                execution_date,
                context="backtest event exit",
            )
        except RuntimeError as error:
            failures.append(str(error))
    raise RuntimeError(
        "Event-delivered forced exit lacks causal sector provenance for "
        f"{child} from {parents} on {execution_date.date()}: {failures}"
    )


def _delivery_was_applied(
    advances: list,
    candidate: pd.Series,
) -> bool:
    """Confirm that the action result contains this exact signed delivery."""

    event_id = str(candidate["Event_ID"])
    asset_id = str(candidate["Asset_ID"])
    for advance in advances:
        deliveries = advance.corporate_actions.position_deliveries
        rows = deliveries.loc[
            deliveries["Event_ID"].astype(str).eq(event_id)
            & deliveries["To_Asset_ID"].astype(str).eq(asset_id)
        ]
        if rows["Units"].map(lambda value: value != Fraction(0)).any():
            return True
    return False


def _trade_audit_row(
    context: _BacktestContext,
    period: _PeriodDecision,
    execution: TradeExecution,
    sector_fields: dict[str, object],
    *,
    execution_date: pd.Timestamp,
    execution_eligible: bool,
    execution_adjustment: str,
    current_capital: float,
    current_weight: float,
    target_weight: float,
    current_value: float,
    requested_target_value: float,
    target_value: float,
    execution_reason: str,
    trigger_event_id: str,
) -> dict[str, object]:
    """Build the one canonical audit row for any executed backtest trade."""

    asset_id = execution.asset_id
    return {
        "Signal_Cutoff": period.t_0_eff,
        "Execution_Date": pd.Timestamp(execution_date),
        "Valuation_End": period.t_1,
        "Asset_ID": asset_id,
        "Ticker": context.backtest_data.audit_ticker(
            asset_id, execution_date
        ),
        **sector_fields,
        "Strategy_ID": context.strategy.strategy_id,
        "Strategy_Version": context.strategy.strategy_version,
        **strategy_signal_fields(
            context.strategy,
            period.decision,
            asset_id,
            require_ranked=True,
        ),
        **strategy_selection_fields(
            period.decision,
            period.execution_decision,
            asset_id,
            execution_eligible=execution_eligible,
            execution_adjustment=execution_adjustment,
        ),
        "Current_Capital": float(current_capital),
        "Current_Weight": float(current_weight),
        "Target_Weight": float(target_weight),
        "Weight_Drift": float(abs(target_weight - current_weight)),
        "Turnover_Threshold": float(context.strategy.turnover_threshold),
        "Current_Position_Value": float(current_value),
        "Requested_Target_Position_Value": float(requested_target_value),
        "Target_Position_Value": float(target_value),
        **trade_execution_audit_fields(execution),
        "Turnover_Contribution": (
            abs(float(execution.reference_notional)) / float(current_capital)
            if current_capital
            else 0.0
        ),
        "Execution_Reason": execution_reason,
        "Trigger_Event_ID": trigger_event_id,
    }


def _record_forced_exit_trade(
    context: _BacktestContext,
    period: _PeriodDecision,
    rebalance: _BacktestRebalance,
    candidate: pd.Series,
    execution: TradeExecution,
) -> None:
    """Append one event-driven non-discretionary exit to the trade audit."""

    collector = context.strategy_audit
    if collector is None:
        return
    sector_fields = _forced_exit_sector_fields(context, candidate)
    current_value = float(execution.current_shares * execution.reference_price)
    capital = float(rebalance.ledger.after.equity)
    collector.trade_records.append(
        _trade_audit_row(
            context,
            period,
            execution,
            sector_fields,
            execution_date=pd.Timestamp(candidate["Execution_Date"]),
            execution_eligible=False,
            execution_adjustment="mandatory_off_universe_liquidation",
            current_capital=capital,
            current_weight=current_value / capital if capital else 0.0,
            target_weight=0.0,
            current_value=current_value,
            requested_target_value=0.0,
            target_value=0.0,
            execution_reason="mandatory_off_universe_liquidation",
            trigger_event_id=str(candidate["Event_ID"]),
        )
    )


def _value_lifecycle_period(
    context: _BacktestContext,
    period: _PeriodDecision,
    rebalance: _BacktestRebalance,
) -> _LifecycleResult:
    """Advance one ledger through actions and event-driven forced exits."""

    backtest_data = context.backtest_data
    t_0 = period.t_0
    t_1 = period.t_1
    trade_state = rebalance.ledger.state
    held_shares = trade_state.shares
    held_asset_ids = list(held_shares.index.astype(str))
    close_prices_t0 = rebalance.execution_prices.reindex(held_asset_ids)
    has_position_event = period_has_position_event(
        backtest_data,
        held_shares,
        pd.Timestamp(t_0),
        pd.Timestamp(t_1),
    )

    events, event_legs, event_sources = prepared_event_tables(backtest_data)
    execution_candidates = event_delivery_execution_candidates(
        backtest_data,
        pd.Timestamp(t_0),
        pd.Timestamp(t_1),
    )

    advances = []
    cursor = trade_state
    forced_exit_executions: list[TradeExecution] = []
    valuation_overrides: dict[str, float] = {}

    for candidate_row in execution_candidates.to_dict("records"):
        candidate = pd.Series(candidate_row)
        execution_date = pd.Timestamp(candidate["Execution_Date"])
        if cursor.state_date is None or execution_date < cursor.state_date:
            raise RuntimeError("Event-driven forced-exit dates are not monotonic")
        if execution_date > cursor.state_date:
            advance = advance_ledger(
                cursor,
                execution_date,
                events,
                event_legs,
                event_sources,
                context.accounting_config,
            )
            advances.append(advance)
            cursor = advance.state
        asset_id = str(candidate["Asset_ID"])
        if not _delivery_was_applied(advances, candidate):
            continue
        if asset_id not in cursor.shares.index:
            continue
        cursor, execution = execute_forced_liquidation(
            cursor,
            asset_id=asset_id,
            reference_price=float(candidate["Reference_Close"]),
            own_liquidity_usd=(
                float(candidate["Reference_Close"])
                * float(candidate["Volume"])
            ),
            execution_date=execution_date,
            config=context.accounting_config,
            apply_fees=context.apply_fees,
            apply_spread=context.apply_spread,
        )
        forced_exit_executions.append(execution)
        valuation_overrides[asset_id] = float(candidate["Reference_Close"])
        _record_forced_exit_trade(
            context,
            period,
            rebalance,
            candidate,
            execution,
        )

    if cursor.state_date != t_1:
        advance = advance_ledger(
            cursor,
            t_1,
            events,
            event_legs,
            event_sources,
            context.accounting_config,
        )
        advances.append(advance)
        cursor = advance.state
    if not advances:
        advances.append(
            advance_ledger(
                cursor,
                t_1,
                events,
                event_legs,
                event_sources,
                context.accounting_config,
            )
        )
        cursor = advances[-1].state

    end_state = cursor
    interest_amount = float(sum(item.interest_amount for item in advances))
    cash_interest_credit = float(
        sum(item.cash_interest_credit for item in advances)
    )
    loan_interest_charge = float(
        sum(item.loan_interest_charge for item in advances)
    )
    event_interest_effects: dict[str, float] = {}
    for advance in advances:
        for event_id, effect in advance.event_interest_effects.items():
            event_interest_effects[event_id] = (
                event_interest_effects.get(event_id, 0.0) + float(effect)
            )

    invalid_start = (
        close_prices_t0.isna()
        | ~np.isfinite(close_prices_t0)
        | close_prices_t0.le(0.0)
    )
    if invalid_start.any():
        raise RuntimeError(
            "Cannot establish signed shares on "
            f"{t_0.date()}: {close_prices_t0.index[invalid_start].tolist()}"
        )
    for advance in advances:
        rights = advance.corporate_actions.nontradable_rights
        nonzero_rights = rights.loc[
            rights["Base_Value"].map(
                lambda value: parse_exact_number(value) != Fraction(0)
            )
        ]
        if not nonzero_rights.empty:
            raise RuntimeError(
                "Backtest cannot carry a non-tradable right with a nonzero "
                "base value between rebalance dates"
            )

    end_prices = pd.Series(
        {
            str(asset_id): finite_positive_price(
                backtest_data.data_close,
                pd.Timestamp(t_1),
                str(asset_id),
            )
            for asset_id in end_state.shares.index
        },
        dtype=float,
    )
    invalid_end = end_prices.isna() | ~np.isfinite(end_prices) | end_prices.le(0.0)
    if invalid_end.any():
        raise RuntimeError(
            "Cannot mark lifecycle positions on "
            f"{t_1.date()}: {end_prices.index[invalid_end].tolist()}"
        )
    current_position_values = end_state.shares * end_prices

    if not has_position_event:
        close_prices_t1 = end_prices.reindex(held_asset_ids)
        price_ratios = (close_prices_t1 / close_prices_t0).replace(
            [np.inf, -np.inf], np.nan
        )
        invalid = price_ratios.isna() & held_shares.ne(0.0)
        if invalid.any():
            raise RuntimeError(
                "Cannot mark active positions with missing prices from "
                f"{t_0.date()} to {t_1.date()}: "
                f"{price_ratios.index[invalid].tolist()}"
            )
        stock_returns_t1 = (price_ratios.fillna(1.0) - 1.0).astype(float)
    else:
        stock_returns: dict[str, float] = {}
        for asset_id in held_shares.index:
            has_asset_event = asset_has_interval_event(
                backtest_data,
                str(asset_id),
                t_0,
                t_1,
            )
            valuation = (
                event_valuation_result(
                    backtest_data,
                    str(asset_id),
                    pd.Timestamp(t_0),
                    pd.Timestamp(t_1),
                    successor_price_overrides=valuation_overrides,
                )
                if has_asset_event
                else None
            )
            if has_asset_event:
                if valuation is None:
                    raise RuntimeError(
                        "Cannot value explicit corporate action for "
                        f"{asset_id} from {t_0.date()} to {t_1.date()}"
                    )
                _, end_value_per_share = valuation
                stock_returns[str(asset_id)] = float(
                    end_value_per_share
                    / parse_exact_number(close_prices_t0.loc[asset_id])
                    - Fraction(1)
                )
            else:
                end_price = finite_positive_price(
                    backtest_data.data_close,
                    pd.Timestamp(t_1),
                    str(asset_id),
                )
                if end_price is None:
                    raise RuntimeError(
                        "Cannot value unaffected event-period position "
                        f"{asset_id} on {t_1.date()}"
                    )
                stock_returns[str(asset_id)] = (
                    float(end_price) / float(close_prices_t0.loc[asset_id]) - 1.0
                )
        stock_returns_t1 = pd.Series(stock_returns, dtype=float)

    if context.corporate_action_audit_records is not None:
        attributed_event_ids: set[str] = set()
        event_audit_present = False
        for advance in advances:
            for row in advance.corporate_actions.audit.to_dict("records"):
                event_audit_present = True
                event_id = str(row["Event_ID"])
                interest_effect = (
                    event_interest_effects.get(event_id, 0.0)
                    if event_id not in attributed_event_ids
                    else 0.0
                )
                attributed_event_ids.add(event_id)
                context.corporate_action_audit_records.append({
                    "Record_Type": "corporate_action",
                    "Period_Start": pd.Timestamp(t_0),
                    "Period_End": pd.Timestamp(t_1),
                    "Interest_Effect": float(interest_effect),
                    **row,
                })
        if event_audit_present:
            counterfactual_interest = float(
                interest_amount - sum(event_interest_effects.values())
            )
            context.corporate_action_audit_records.append({
                "Record_Type": "financing_summary",
                "Period_Start": pd.Timestamp(t_0),
                "Period_End": pd.Timestamp(t_1),
                "Event_ID": "",
                "Interest_Amount": float(interest_amount),
                "Cash_Interest_Credit": float(cash_interest_credit),
                "Loan_Interest_Charge": float(loan_interest_charge),
                "Counterfactual_Interest_Without_Actions": (
                    counterfactual_interest
                ),
                "Interest_Effect": float(
                    interest_amount - counterfactual_interest
                ),
                "Fixed_Fee": 0.0,
                "Spread_Cost": 0.0,
            })

    end_snapshot = account_snapshot(end_state, end_prices, date=t_1)
    from portfolio_core.strategies.research_execution import require_positive_equity
    require_positive_equity(end_snapshot.equity, t_1)
    if not np.isclose(
        end_snapshot.equity,
        float(end_state.cash + current_position_values.sum()),
        atol=1e-7,
        rtol=0.0,
    ):
        raise RuntimeError("Backtest lifecycle values do not reconcile to the ledger")
    forced_exit_fees = float(
        sum(item.fixed_fee for item in forced_exit_executions)
    )
    forced_exit_spread = float(
        sum(item.spread_cost for item in forced_exit_executions)
    )
    forced_exit_orders = int(
        sum(item.order_count for item in forced_exit_executions)
    )
    turnover_denominator = float(rebalance.ledger.after.equity)
    forced_exit_turnover = float(
        sum(abs(item.reference_notional) for item in forced_exit_executions)
        / turnover_denominator
        if turnover_denominator
        else 0.0
    )
    return _LifecycleResult(
        ledger_state=end_state,
        end_snapshot=end_snapshot,
        current_position_values=current_position_values,
        interest_amount=float(interest_amount),
        cash_interest_credit=float(cash_interest_credit),
        loan_interest_charge=float(loan_interest_charge),
        stock_returns_t1=stock_returns_t1,
        has_position_event=has_position_event,
        forced_exit_order_count=forced_exit_orders,
        forced_exit_fixed_fees=forced_exit_fees,
        forced_exit_spread_cost=forced_exit_spread,
        forced_exit_turnover=forced_exit_turnover,
    )


def _execution_adjustment(
    period: _PeriodDecision,
    asset_id: str,
) -> str:
    return str(
        period.execution_decision.eligibility_reasons.get(asset_id, "")
    )


def _validate_rebalance_sector_contract(
    context: _BacktestContext,
    period: _PeriodDecision,
    rebalance: _BacktestRebalance,
) -> None:
    """Fail before valuation if an ineligible position is not fully exited."""

    retained_ineligible = sorted(
        str(asset_id)
        for asset_id, value in rebalance.applied_values_t0.items()
        if float(value) != 0.0
        and str(asset_id) not in period.interval_eligible_asset_ids
    )
    if retained_ineligible:
        raise RuntimeError(
            "Backtest retained target-ineligible assets on "
            f"{period.t_0.date()}: {retained_ineligible}"
        )
    for asset_id in rebalance.trade_values.index[
        rebalance.trade_values.ne(0.0)
    ]:
        asset = str(asset_id)
        trade_sector_audit_fields(
            assignments=context.backtest_data.sector_assignments,
            execution_rows=period.execution_sector_rows,
            execution_date=period.t_0,
            execution_eligible=asset in period.interval_eligible_asset_ids,
            asset_id=asset,
            current_value=float(rebalance.current_values_t0.loc[asset_id]),
            target_value=float(rebalance.applied_values_t0.loc[asset_id]),
            domain="backtest",
        )


def _record_strategy_audit(
    context: _BacktestContext,
    state: _BacktestRunState,
    period: _PeriodDecision,
    rebalance: _BacktestRebalance,
) -> None:
    """Record selected-run decisions and nonzero rebalance trades."""
    collector = context.strategy_audit
    if collector is None:
        return

    decision = period.decision
    execution_decision = period.execution_decision
    for asset_id in decision.signal_audit.index.astype(str):
        selection_fields = strategy_selection_fields(
            decision,
            execution_decision,
            asset_id,
            execution_eligible=(
                asset_id in period.interval_eligible_asset_ids
            ),
            execution_adjustment=_execution_adjustment(period, asset_id),
        )
        signal_side = str(selection_fields["Signal_Selected_Side"])
        strategy_fields = strategy_signal_fields(
            context.strategy,
            decision,
            asset_id,
        )
        if period.used_hold_logic:
            applied_rule = "hold_prior_target_incomplete_cross_section"
        elif not strategy_fields["Strategy_Eligible"]:
            applied_rule = "excluded_by_strategy_rule"
        elif signal_side:
            applied_rule = f"selected_{signal_side.casefold()}"
        else:
            applied_rule = "not_selected_by_strategy_rule"
        collector.decision_records.append({
            "Signal_Cutoff": period.t_0_eff,
            "Execution_Date": period.t_0,
            "Valuation_End": period.t_1,
            "Asset_ID": asset_id,
            "Ticker": context.backtest_data.audit_ticker(asset_id, period.t_0),
            **sector_audit_fields(
                period.signal_sector_rows,
                asset_id,
                period.t_0_eff,
                context="decision-date",
            ),
            "Strategy_ID": context.strategy.strategy_id,
            "Strategy_Version": context.strategy.strategy_version,
            **strategy_fields,
            **selection_fields,
            "Applied_Target_Weight": float(
                state.target_weights.get(asset_id, 0.0)
            ),
            "Decision_Complete": bool(decision.is_complete),
            "Used_Hold_Logic": bool(period.used_hold_logic),
            "Applied_Rule": applied_rule,
        })

    trade_rows: list[dict[str, object]] = []
    for execution in rebalance.ledger.executions:
        asset_id = execution.asset_id
        current_value = float(rebalance.current_values_t0.loc[asset_id])
        target_value = float(rebalance.applied_values_t0.loc[asset_id])
        trade_rows.append(
            _trade_audit_row(
                context,
                period,
                execution,
                trade_sector_audit_fields(
                    assignments=context.backtest_data.sector_assignments,
                    execution_rows=period.execution_sector_rows,
                    execution_date=period.t_0,
                    execution_eligible=(
                        asset_id in period.interval_eligible_asset_ids
                    ),
                    asset_id=asset_id,
                    current_value=current_value,
                    target_value=target_value,
                    domain="backtest",
                ),
                execution_date=period.t_0,
                execution_eligible=(
                    asset_id in period.interval_eligible_asset_ids
                ),
                execution_adjustment=_execution_adjustment(period, asset_id),
                current_capital=float(state.current_capital),
                current_weight=float(
                    rebalance.current_weights_t0.loc[asset_id]
                ),
                target_weight=float(rebalance.target_weights_t0.loc[asset_id]),
                current_value=current_value,
                requested_target_value=float(
                    rebalance.requested_values_t0.loc[asset_id]
                ),
                target_value=target_value,
                execution_reason="strategy_rebalance",
                trigger_event_id="",
            )
        )

    if (
        sum(int(row["Order_Count"]) for row in trade_rows)
        != rebalance.ledger.order_count
    ):
        raise RuntimeError("Trade audit order counts do not reconcile")
    if not np.isclose(
        sum(float(row["Fixed_Fee"]) for row in trade_rows),
        rebalance.ledger.fixed_fees,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError("Trade audit fixed fees do not reconcile")
    if not np.isclose(
        sum(float(row["Spread_Cost"]) for row in trade_rows),
        rebalance.ledger.spread_cost,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError("Trade audit spread costs do not reconcile")
    collector.trade_records.extend(trade_rows)


def _record_period_results(
    context: _BacktestContext,
    state: _BacktestRunState,
    period: _PeriodDecision,
    rebalance: _BacktestRebalance,
    lifecycle: _LifecycleResult,
) -> None:
    """Accumulate holdings, NAV, diagnostics, and next-period state."""
    backtest_data = context.backtest_data
    _record_strategy_audit(context, state, period, rebalance)
    if context.return_holdings:
        actual_weights_t0 = (
            rebalance.applied_values_t0 / rebalance.ledger.after.equity
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        held_asset_ids = sorted(
            actual_weights_t0[actual_weights_t0 != 0.0].index
        )
        assert state.holdings_records is not None
        for asset_id in held_asset_ids:
            if asset_id not in period.execution_sector_rows.index:
                raise RuntimeError(
                    "Missing execution-date sector assignment for held asset "
                    f"{asset_id} on {period.t_0.date()}"
                )
            sector = period.execution_sector_rows.loc[asset_id]
            state.holdings_records.append(
                {
                    "Date": period.t_0,
                    "Next_Date": period.t_1,
                    "Asset_ID": asset_id,
                    "Ticker": backtest_data.audit_ticker(asset_id, period.t_0),
                    "GICS_Sector_Code": str(sector["GICS_Sector_Code"]),
                    "Sector": str(sector["Sector"]),
                    "Sector_As_Of_Date": pd.Timestamp(sector["As_Of_Date"]),
                    "Sector_Source_Type": str(sector["Source_Type"]),
                    "Sector_Source_Reference": str(
                        sector["Source_Reference"]
                    ),
                    "Weight": float(actual_weights_t0.loc[asset_id]),
                    "Stock_Return": float(
                        lifecycle.stock_returns_t1.loc[asset_id]
                    ),
                }
            )

    port_value = lifecycle.end_snapshot.equity
    state.ledger = lifecycle.ledger_state

    if lifecycle.has_position_event:
        state.target_weights = (
            lifecycle.current_position_values / port_value
            if port_value != 0.0
            else pd.Series(dtype=float)
        )

    state.portfolio_values.loc[period.t_1] = port_value
    state.current_capital = port_value

    decision = period.decision
    if context.return_diagnostics:
        state.diagnostics.append(
            {
                "Date": period.t_1,
                "NAV": float(port_value),
                "Num_Trades": int(
                    rebalance.ledger.order_count + lifecycle.forced_exit_order_count
                ),
                "Fees": float(
                    rebalance.ledger.fixed_fees + lifecycle.forced_exit_fixed_fees
                ),
                "Spread_Cost": float(
                    rebalance.ledger.spread_cost
                    + lifecycle.forced_exit_spread_cost
                ),
                "Turnover": float(
                    rebalance.ledger.turnover + lifecycle.forced_exit_turnover
                ),
                "Turnover_Threshold": float(context.strategy.turnover_threshold),
                "Interest": float(lifecycle.interest_amount),
                "Cash_Interest_Credit": float(lifecycle.cash_interest_credit),
                "Loan_Interest_Charge": float(lifecycle.loan_interest_charge),
                "Cash": lifecycle.end_snapshot.cash,
                "Signed_Market_Value": float(
                    lifecycle.end_snapshot.position_values.sum()
                ),
                "Gross_Market_Value": float(
                    lifecycle.end_snapshot.position_values.abs().sum()
                ),
                "Restricted_Short_Proceeds": (
                    lifecycle.end_snapshot.restricted_short_proceeds
                ),
                "Free_Cash": lifecycle.end_snapshot.free_cash,
                "Loan": lifecycle.end_snapshot.loan,
                "Gross_Exposure": lifecycle.end_snapshot.gross_exposure,
                "Maximum_Position_Weight": (
                    lifecycle.end_snapshot.maximum_position_weight
                ),
                "Position_Count": lifecycle.end_snapshot.position_count,
                "Long_Count": lifecycle.end_snapshot.long_count,
                "Short_Count": lifecycle.end_snapshot.short_count,
                "Feasibility_Scale": rebalance.ledger.feasibility_scale,
                "Feasibility_Adjustment": rebalance.ledger.adjustment_reason,
                "Used_Hold_Logic": bool(period.used_hold_logic),
                "Net_Exposure": float(lifecycle.end_snapshot.position_values.sum() / port_value),
                "Post_Trade_Gross_Exposure": rebalance.ledger.after.gross_exposure,
                "Post_Trade_Net_Exposure": float(rebalance.applied_values_t0.sum() / rebalance.ledger.after.equity),
                **{
                    column: float(decision.summary_metrics[column])
                    for column in context.strategy.summary_columns
                },
            }
        )

    if context.debug:
        summary_text = " | ".join(
            f"{column}={float(decision.summary_metrics[column]): .4f}"
            for column in context.strategy.summary_columns
        )
        print(
            f"{period.t_1.date()} | NAV={port_value:,.0f} | "
            f"trades={rebalance.ledger.order_count + lifecycle.forced_exit_order_count:2d} | "
            f"turnover={(rebalance.ledger.turnover + lifecycle.forced_exit_turnover)*100:5.1f}% | "
            f"fees=${rebalance.ledger.fixed_fees + lifecycle.forced_exit_fixed_fees:,.0f} | "
            f"spread=${rebalance.ledger.spread_cost + lifecycle.forced_exit_spread_cost:,.0f} | "
            f"{summary_text} | "
            f"hold={period.used_hold_logic}"
        )


def _assemble_backtest_outputs(
    state: _BacktestRunState,
    *,
    return_diagnostics: bool,
    return_holdings: bool,
):
    """Build the public return shape without changing its empty-run contract."""
    if not state.started:
        empty_results: list[pd.Series | pd.DataFrame] = [
            pd.Series(dtype=float)
        ]
        if return_diagnostics:
            empty_results.append(pd.DataFrame())
        if return_holdings:
            empty_results.append(pd.DataFrame(columns=_HOLDINGS_COLUMNS))
        return (
            tuple(empty_results)
            if len(empty_results) > 1
            else empty_results[0]
        )

    outputs: list[pd.Series | pd.DataFrame] = [state.portfolio_values]
    if return_diagnostics:
        diag_df = (
            pd.DataFrame(state.diagnostics).set_index("Date")
            if state.diagnostics
            else pd.DataFrame()
        )
        outputs.append(diag_df)
    if return_holdings:
        outputs.append(
            pd.DataFrame(state.holdings_records, columns=_HOLDINGS_COLUMNS)
        )
    return tuple(outputs) if len(outputs) > 1 else outputs[0]


def _run_backtest_impl(
    backtest_data: BacktestDataset,
    dates,
    strategy: Strategy,
    *,
    accounting_config: PortfolioAccountingConfig = DEFAULT_ACCOUNTING_CONFIG,
    apply_turnover_threshold=True,
    apply_fees=True,
    apply_spread=True,
    return_diagnostics=False,
    return_holdings=False,
    debug=False,
    corporate_action_audit_records: list[dict[str, object]] | None = None,
    strategy_audit: _BacktestAuditCollector | None = None,
):
    """Run the monthly backtest while delegating each orchestration phase."""
    if not isinstance(strategy, Strategy):
        raise TypeError("strategy must implement the canonical Strategy contract")
    if strategy_audit is not None and strategy_audit.strategy is not strategy:
        raise ValueError("Strategy audit collector must own the selected strategy object")
    state = _BacktestRunState(
        ledger=LedgerState.initial(accounting_config),
        current_capital=accounting_config.initial_capital,
        holdings_records=[] if return_holdings else None,
    )

    sector_returns = None
    if strategy.strategy_id == "sector_momentum":
        from .sector_returns import build_sector_returns
        history = backtest_data.sector_return_history
        if history is None or history.end < pd.Timestamp(dates[-1]):
            history = build_sector_returns(backtest_data, dates[-1])
        sector_returns = history.returns
    context = _BacktestContext(
        backtest_data=backtest_data,
        strategy=strategy,
        apply_turnover_threshold=apply_turnover_threshold,
        apply_fees=apply_fees,
        apply_spread=apply_spread,
        return_holdings=return_holdings,
        return_diagnostics=return_diagnostics,
        debug=debug,
        accounting_config=accounting_config,
        corporate_action_audit_records=corporate_action_audit_records,
        strategy_audit=strategy_audit,
        sector_returns=sector_returns,
    )

    for i in range(len(dates) - 1):
        period = _prepare_period_decision(
            context,
            state,
            dates[i],
            dates[i + 1],
        )
        if period is None:
            continue
        rebalance = _process_rebalance_and_costs(context, state, period)
        _validate_rebalance_sector_contract(context, period, rebalance)
        lifecycle = _value_lifecycle_period(context, period, rebalance)
        _record_period_results(
            context,
            state,
            period,
            rebalance,
            lifecycle,
        )

    return _assemble_backtest_outputs(
        state,
        return_diagnostics=return_diagnostics,
        return_holdings=return_holdings,
    )
def run_backtest(
    backtest_data: BacktestDataset,
    dates,
    strategy: Strategy,
    *,
    accounting_config: PortfolioAccountingConfig = DEFAULT_ACCOUNTING_CONFIG,
    apply_turnover_threshold=True,
    apply_fees=True,
    apply_spread=True,
    return_diagnostics=False,
    return_holdings=False,
    debug=False,
):
    """Return a NAV Series, appending requested diagnostics then holdings in a tuple.

    A run that never invests returns empty outputs with the same ordering and
    holdings schema. Disabling fees/spread removes trading costs, while cash
    interest and loan charges still accrue.
    """
    return _run_backtest_impl(
        backtest_data,
        dates,
        strategy,
        accounting_config=accounting_config,
        apply_turnover_threshold=apply_turnover_threshold,
        apply_fees=apply_fees,
        apply_spread=apply_spread,
        return_diagnostics=return_diagnostics,
        return_holdings=return_holdings,
        debug=debug,
    )
