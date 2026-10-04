"""Focused invariants for the unified portfolio-accounting ledger."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import backtest.engine as backtest_adapter
import live.analysis as live_adapter
import portfolio_core.accounting_ledger as ledger
from portfolio_core.accounting_config import (
    DEFAULT_ACCOUNTING_CONFIG,
    make_tc_rate_from_dollar_volume,
    spread_sensitivity_configs,
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
    account_snapshot,
    execute_forced_liquidation,
    execute_rebalance,
    project_financing_amounts,
)
from portfolio_core.strategies import (
    ExecutionDecision,
    StrategyDecision,
)

def _rules(**overrides):
    values = {
        "minimum_positions": 0,
        "minimum_long_positions": 0,
        "minimum_short_positions": 0,
    }
    values.update(overrides)
    return replace(DEFAULT_ACCOUNTING_CONFIG, **values)


def _plan(targets, weights, prices, liquidity=None, eligible=None, date="2026-01-02"):
    price_series = pd.Series(prices, dtype=float)
    return LedgerRebalancePlan.from_series(
        pd.Series(targets, dtype=float),
        pd.Series(weights, dtype=float),
        price_series,
        pd.Series(
            liquidity or {asset: 10_000_000.0 for asset in price_series.index},
            dtype=float,
        ),
        eligible or set(price_series.index),
        pd.Timestamp(date),
    )


def test_ledger_state_rejects_arbitrary_public_component_construction():
    with pytest.raises(TypeError):
        LedgerState(
            (("A", 1.0),),
            100.0,
            (),
            pd.Timestamp("2026-01-01"),
        )


def test_spread_uses_only_own_adv_without_nominal_price_rules():
    liquidity = pd.Series({"A": 10_000_000.0, "B": np.nan, "PENNY": 1e30})
    rates = make_tc_rate_from_dollar_volume(liquidity)

    assert rates["A"] == pytest.approx(8.0 / 10_000.0)
    assert rates["B"] == pytest.approx(15.0 / 10_000.0)
    assert rates["PENNY"] == pytest.approx(1.0 / 10_000.0)

    expanded = make_tc_rate_from_dollar_volume(
        pd.concat([liquidity, pd.Series({"X": 1.0, "Y": 1e20})]),
    )
    assert expanded["A"] == rates["A"]

    monotonic = make_tc_rate_from_dollar_volume(
        pd.Series({"LOW": 1_000_000.0, "MID": 10_000_000.0, "HIGH": 1e12}),
    )
    assert monotonic["LOW"] > monotonic["MID"] > monotonic["HIGH"]


def test_share_rounding_uses_halves_away_from_zero():
    rounded = round_half_away_from_zero(
        pd.Series({"P": 2.5, "N": -2.5, "LOW_P": 0.49, "LOW_N": -0.49})
    )
    assert rounded.to_dict() == {"P": 3, "N": -3, "LOW_P": 0, "LOW_N": 0}


@pytest.mark.parametrize("signed_units", (2.5, -2.5))
def test_forced_liquidation_closes_complete_fraction_and_cannot_repeat(
    signed_units,
):
    fractional_rules = _rules(whole_share_orders=False)
    opened = execute_rebalance(
        LedgerState.initial(
            fractional_rules, state_date=pd.Timestamp("2025-02-24")
        ),
        _plan(
            {"CHILD": signed_units},
            {"CHILD": signed_units * 100.0 / 1_000_000.0},
            {"CHILD": 100.0},
            date="2025-02-24",
        ),
        fractional_rules,
        apply_fees=False,
        apply_spread=False,
    )
    forced_rules = replace(fractional_rules, whole_share_orders=True)

    closed, execution = execute_forced_liquidation(
        opened.state,
        asset_id="CHILD",
        reference_price=110.0,
        own_liquidity_usd=11_000_000.0,
        execution_date="2025-02-24",
        config=forced_rules,
    )

    assert closed.shares.empty
    assert execution.applied_rule == "mandatory_off_universe_liquidation"
    assert execution.current_shares == signed_units
    assert execution.trade_shares == -signed_units
    assert execution.applied_target_shares == 0.0
    assert execution.order_count == 1
    assert execution.fixed_fee == 2.0
    assert execution.spread_cost > 0.0
    if signed_units > 0:
        assert execution.cash_effect > 0.0
        assert execution.restricted_proceeds_change == 0.0
    else:
        assert execution.cash_effect < 0.0
        assert execution.restricted_proceeds_change < 0.0
        assert closed.restricted_short_proceeds.empty
    with pytest.raises(ValueError, match="no open position"):
        execute_forced_liquidation(
            closed,
            asset_id="CHILD",
            reference_price=110.0,
            own_liquidity_usd=11_000_000.0,
            execution_date="2025-02-24",
            config=forced_rules,
        )


def test_spread_sensitivity_coefficients_are_canonical():
    cases = spread_sensitivity_configs(DEFAULT_ACCOUNTING_CONFIG)
    assert {
        name: case.transaction_costs.liquidity_bps
        for name, case in cases.items()
    } == {"low": 3.5, "base": 7.0, "high": 10.5}
    assert cases["base"] is DEFAULT_ACCOUNTING_CONFIG
    liquidity = pd.Series({"A": 10_000_000.0})
    rates = []
    for case in cases.values():
        rates.append(
            make_tc_rate_from_dollar_volume(
                liquidity,
                config=case.transaction_costs,
            )["A"]
        )
    assert rates == sorted(rates)

    custom = replace(
        DEFAULT_ACCOUNTING_CONFIG,
        transaction_costs=replace(
            DEFAULT_ACCOUNTING_CONFIG.transaction_costs,
            liquidity_bps=9.0,
        ),
    )
    assert {
        name: case.transaction_costs.liquidity_bps
        for name, case in spread_sensitivity_configs(custom).items()
    } == {"low": 4.5, "base": 9.0, "high": 13.5}


def test_live_liquidity_is_causal_monthly_then_trailing_three_month_median():
    market = pd.DataFrame([
        {"Date": "2026-01-10", "Asset_ID": "A", "Close": 10.0, "Volume": 10.0},
        {"Date": "2026-01-20", "Asset_ID": "A", "Close": 10.0, "Volume": 30.0},
        {"Date": "2026-02-10", "Asset_ID": "A", "Close": 10.0, "Volume": 50.0},
        {"Date": "2026-03-10", "Asset_ID": "A", "Close": 10.0, "Volume": 70.0},
        {"Date": "2026-04-10", "Asset_ID": "A", "Close": 10.0, "Volume": 1e9},
        {"Date": "2026-03-10", "Asset_ID": "B", "Close": 20.0, "Volume": 5.0},
    ])
    market["Date"] = pd.to_datetime(market["Date"])

    liquidity = live_adapter._causal_trailing_adv(
        market,
        pd.Timestamp("2026-03-31"),
    )

    assert liquidity["A"] == 500.0
    assert liquidity["B"] == 100.0


def test_turnover_filter_is_strategy_owned_and_never_suppresses_flip_exit_or_entry():
    current = pd.Series({"RESIZE": 100.0, "EXIT": 100.0, "FLIP": 100.0})
    prices = pd.Series({"RESIZE": 100.0, "EXIT": 100.0, "FLIP": 100.0, "ENTRY": 100.0})
    targets = pd.Series({
        "RESIZE": 0.0105,
        "EXIT": 0.0,
        "FLIP": -0.01,
        "ENTRY": 0.01,
    })
    result = plan_rebalance_targets(
        current,
        targets,
        prices,
        1_000_000.0,
        turnover_threshold=0.015,
    )
    applied = result.applied_shares
    assert applied["RESIZE"] == current["RESIZE"]
    assert applied["EXIT"] == 0.0
    assert applied["FLIP"] < 0.0
    assert applied["ENTRY"] > 0.0

    zero = plan_rebalance_targets(
        current,
        targets,
        prices,
        1_000_000.0,
        turnover_threshold=0.0,
    )
    assert zero.applied_shares["RESIZE"] != current["RESIZE"]


def test_short_proceeds_are_transactional_and_release_average_history():
    rules = _rules()
    state = LedgerState.initial(rules)
    opened = execute_rebalance(
        state,
        _plan({"S": -10.0}, {"S": -0.001}, {"S": 100.0}),
        rules,
    )
    assert opened.order_count == 1
    assert opened.state.restricted_short_proceeds["S"] == pytest.approx(999.2)
    assert opened.state.cash == pytest.approx(1_000_997.2)

    covered = execute_rebalance(
        opened.state,
        _plan(
            {"S": -6.0},
            {"S": -0.00066},
            {"S": 110.0},
            date="2026-01-02",
        ),
        rules,
    )
    assert covered.state.restricted_short_proceeds["S"] == pytest.approx(
        999.2 * 0.6
    )
    assert covered.executions[0].restricted_proceeds_change == pytest.approx(
        -(999.2 * 0.4)
    )

    closed = execute_rebalance(
        covered.state,
        _plan(
            {"S": 0.0},
            {"S": 0.0},
            {"S": 90.0},
            date="2026-01-02",
        ),
        rules,
    )
    assert closed.state.restricted_short_proceeds.empty


def test_sign_flip_is_two_orders_and_releases_old_short_before_long():
    rules = _rules()
    short = execute_rebalance(
        LedgerState.initial(rules),
        _plan({"A": -5.0}, {"A": -0.00045}, {"A": 90.0}),
        rules,
        apply_fees=False,
        apply_spread=False,
    )
    result = execute_rebalance(
        short.state,
        _plan(
            {"A": 5.0},
            {"A": 0.0005},
            {"A": 100.0},
            date="2026-01-02",
        ),
        rules,
    )
    execution = result.executions[0]
    assert execution.order_count == 2
    assert execution.fixed_fee == 4.0
    assert tuple(leg.leg_type for leg in execution.legs) == (
        "cover_short",
        "open_long",
    )
    assert execution.restricted_proceeds_change == -450.0
    assert result.state.restricted_short_proceeds.empty


def test_long_to_short_flip_closes_then_opens_and_restricts_new_bid_proceeds():
    rules = _rules()
    long_state = execute_rebalance(
        LedgerState.initial(rules),
        _plan({"A": 5.0}, {"A": 0.0005}, {"A": 100.0}),
        rules,
        apply_fees=False,
        apply_spread=False,
    )
    result = execute_rebalance(
        long_state.state,
        _plan(
            {"A": -5.0},
            {"A": -0.0005},
            {"A": 100.0},
            date="2026-01-02",
        ),
        rules,
    )
    execution = result.executions[0]
    assert tuple(leg.leg_type for leg in execution.legs) == (
        "close_long",
        "open_short",
    )
    assert execution.fixed_fee == 4.0
    assert result.state.restricted_short_proceeds["A"] == pytest.approx(499.6)


def test_short_proceeds_do_not_remark_and_increases_add_current_bid_proceeds():
    rules = _rules()
    opened = execute_rebalance(
        LedgerState.initial(rules),
        _plan({"S": -5.0}, {"S": -0.0005}, {"S": 100.0}),
        rules,
    )
    historical = float(opened.state.restricted_short_proceeds["S"])
    moved_snapshot = account_snapshot(
        opened.state,
        pd.Series({"S": 200.0}),
        date="2026-01-02",
    )
    assert moved_snapshot.restricted_short_proceeds == historical

    increased = execute_rebalance(
        opened.state,
        _plan(
            {"S": -7.0},
            {"S": -0.0014},
            {"S": 200.0},
            date="2026-01-02",
        ),
        rules,
    )
    assert increased.state.restricted_short_proceeds["S"] == pytest.approx(
        historical + 2.0 * 199.84
    )


def test_financing_projection_uses_positive_free_cash_and_negative_loan_rates():
    rules = _rules()
    cash_rules = replace(
        rules,
        initial_capital=100.0,
        maximum_gross_exposure=3.0,
    )
    cash_state = LedgerState.initial(
        cash_rules,
        state_date="2026-01-01",
    )
    cash_end, cash_interest, cash_credit, cash_loan_charge = (
        project_financing_amounts(
            cash_state.cash,
            cash_state.restricted_total,
            cash_state.state_date,
            "2027-01-01",
            cash_rules,
        )
    )
    expected_cash = 100.0 * ((1.0 + 0.02 / 365.0) ** 365 - 1.0)
    assert cash_end == pytest.approx(100.0 + expected_cash)
    assert cash_interest == pytest.approx(expected_cash)
    assert cash_credit == pytest.approx(expected_cash)
    assert cash_loan_charge == 0.0

    loan_state = execute_rebalance(
        LedgerState.initial(cash_rules),
        _plan(
            {"L1": 1.0, "L2": 1.0, "S1": -1.0, "S2": -1.0},
            {"L1": 0.6, "L2": 0.6, "S1": -0.6, "S2": -0.6},
            {"L1": 60.0, "L2": 60.0, "S1": 60.0, "S2": 60.0},
            date="2026-01-01",
        ),
        cash_rules,
        apply_fees=True,
        apply_spread=False,
    )
    loan_end, loan_interest, cash_credit, loan_charge = project_financing_amounts(
        loan_state.state.cash,
        loan_state.state.restricted_total,
        loan_state.state.state_date,
        "2027-01-01",
        cash_rules,
    )
    expected_loan = loan_state.state.free_cash * (
        (1.0 + 0.08 / 365.0) ** 365 - 1.0
    )
    assert loan_end == pytest.approx(loan_state.state.cash + expected_loan)
    assert loan_interest == pytest.approx(expected_loan)
    assert cash_credit == 0.0
    assert loan_charge == pytest.approx(-expected_loan)


def test_low_adjusted_unit_price_does_not_block_entry_or_apply_tick_floor():
    rules = _rules()
    result = execute_rebalance(
        LedgerState.initial(rules),
        _plan({"A": 40.0}, {"A": 0.00003296}, {"A": 0.824}),
        rules,
    )
    assert result.state.shares["A"] == 40.0
    assert result.executions[0].spread_rate == pytest.approx(8.0 / 10_000.0)


@pytest.mark.parametrize(
    "whole_share_orders, expected_scale, expected_units",
    (
        (True, 0.9983499999999998, 9983.0),
        (False, 0.9983985623501598, 9983.985623501598),
    ),
)
def test_exact_two_hundred_percent_target_scales_for_execution_costs(
    monkeypatch, whole_share_orders, expected_scale, expected_units,
):
    rules = _rules(
        minimum_positions=2,
        minimum_long_positions=1,
        minimum_short_positions=1,
        whole_share_orders=whole_share_orders,
    )
    scales = []
    executed_baskets = []
    original_scale = ledger._scaled_target_shares
    original_execute = ledger._execute_exact

    def record_scale(requested, current, scale, **kwargs):
        scales.append(scale)
        return original_scale(requested, current, scale, **kwargs)

    def record_execution(state, target, *args, **kwargs):
        executed_baskets.append(tuple(target.items()))
        return original_execute(state, target, *args, **kwargs)

    monkeypatch.setattr(ledger, "_scaled_target_shares", record_scale)
    monkeypatch.setattr(ledger, "_execute_exact", record_execution)
    result = execute_rebalance(
        LedgerState.initial(rules),
        _plan(
            {"L": 10_000.0, "S": -10_000.0},
            {"L": 1.0, "S": -1.0},
            {"L": 100.0, "S": 100.0},
        ),
        rules,
    )
    # Preserve the exact boundary returned by the original 80-iteration search.
    assert result.feasibility_scale == expected_scale
    assert result.applied_items == (("L", expected_units), ("S", -expected_units))
    assert result.adjustment_reason == "execution_feasibility_scaling"
    assert result.after.gross_exposure <= 2.0 + 1e-10
    assert result.after.long_count == result.after.short_count == 1
    assert len(scales) == len(set(scales))
    assert len(executed_baskets) == len(set(executed_baskets))
    if whole_share_orders:
        assert len(executed_baskets) < len(scales)


def test_basket_results_are_not_reused_across_rebalances():
    rules = _rules()
    state = LedgerState.initial(rules)
    plan = _plan(
        {"L": 10_000.0, "S": -10_000.0},
        {"L": 1.0, "S": -1.0},
        {"L": 100.0, "S": 100.0},
    )
    with_costs = execute_rebalance(state, plan, rules)
    without_costs = execute_rebalance(
        state, plan, rules, apply_fees=False, apply_spread=False,
    )

    assert with_costs.feasibility_scale < 1.0
    assert without_costs.feasibility_scale == 1.0
    assert without_costs.applied_items == plan.target_items
    assert without_costs.fixed_fees == without_costs.spread_cost == 0.0
    assert without_costs.after.equity == rules.initial_capital
    assert execute_rebalance(state, plan, rules) == with_costs


def test_intrinsically_invalid_counts_are_rejected_not_scaled():
    rules = _rules(
        minimum_positions=2,
        minimum_long_positions=1,
        minimum_short_positions=1,
    )
    with pytest.raises(InvalidRebalanceError, match="position counts"):
        execute_rebalance(
            LedgerState.initial(rules),
            _plan({"L": 100.0}, {"L": 0.01}, {"L": 100.0}),
            rules,
        )


def test_intrinsic_gross_exposure_violation_is_rejected():
    rules = _rules(
        minimum_positions=2,
        minimum_long_positions=1,
        minimum_short_positions=1,
    )
    with pytest.raises(InvalidRebalanceError, match="gross exposure"):
        execute_rebalance(
            LedgerState.initial(rules),
            _plan(
                {"L": 1_000.0, "S": -1_000.0},
                {"L": 1.01, "S": -1.0},
                {"L": 100.0, "S": 100.0},
            ),
            rules,
        )


@pytest.mark.parametrize("dominant_weight", (0.30, -0.30, 1.20, -1.20))
def test_concentrated_positions_execute_without_a_single_position_cap(
    dominant_weight,
):
    rules = DEFAULT_ACCOUNTING_CONFIG
    weights = {f"L{i}": 0.02 for i in range(10)}
    weights.update({f"S{i}": -0.02 for i in range(10)})
    dominant_asset = "L0" if dominant_weight > 0.0 else "S0"
    weights[dominant_asset] = dominant_weight
    prices = {asset: 100.0 for asset in weights}
    targets = {
        asset: weight * rules.initial_capital / prices[asset]
        for asset, weight in weights.items()
    }

    result = execute_rebalance(
        LedgerState.initial(rules),
        _plan(targets, weights, prices),
        rules,
    )

    assert result.state.shares.to_dict() == targets
    assert result.feasibility_scale == 1.0
    assert result.after.long_count == result.after.short_count == 10
    assert result.after.gross_exposure < rules.maximum_gross_exposure
    assert result.after.maximum_position_weight == pytest.approx(
        abs(dominant_weight) * rules.initial_capital / result.after.equity
    )


def test_rounding_failure_is_execution_infeasibility_not_intrinsic_invalidity():
    rules = _rules(
        minimum_positions=2,
        minimum_long_positions=1,
        minimum_short_positions=1,
    )
    with pytest.raises(InfeasibleRebalanceError, match="position counts"):
        execute_rebalance(
            LedgerState.initial(rules),
            _plan(
                {"L": 1.0, "S": 0.0},
                {"L": 0.0001, "S": -0.000001},
                {"L": 100.0, "S": 100.0},
            ),
            rules,
        )


def test_rebalance_is_independent_of_input_series_order():
    rules = _rules()
    state = LedgerState.initial(rules)
    first = execute_rebalance(
        state,
        _plan(
            {"A": 100.0, "B": -50.0},
            {"A": 0.01, "B": -0.01},
            {"A": 100.0, "B": 200.0},
        ),
        rules,
    )
    second = execute_rebalance(
        state,
        _plan(
            {"B": -50.0, "A": 100.0},
            {"B": -0.01, "A": 0.01},
            {"B": 200.0, "A": 100.0},
        ),
        rules,
    )
    assert first.state == second.state
    assert first.executions == second.executions


def test_backtest_and_live_orchestration_produce_identical_rebalances(
    monkeypatch,
):
    rules = _rules(
        minimum_positions=2,
        minimum_long_positions=1,
        minimum_short_positions=1,
    )
    state = execute_rebalance(
        LedgerState.initial(rules),
        _plan(
            {"OLD_SHORT": -4.0, "OLD_LONG": 3.0},
            {"OLD_SHORT": -0.00036, "OLD_LONG": 0.0003},
            {"OLD_SHORT": 90.0, "OLD_LONG": 100.0},
            date="2026-04-01",
        ),
        rules,
        apply_fees=False,
        apply_spread=False,
    )
    execution_date = pd.Timestamp("2026-04-01")
    prices = pd.Series({"OLD_LONG": 100.0, "OLD_SHORT": 100.0})
    liquidity = pd.Series({"OLD_LONG": 50_000_000.0, "OLD_SHORT": 2_000_000.0})
    target_weights = pd.Series({"OLD_LONG": -0.0002, "OLD_SHORT": 0.0002})
    starting_snapshot = account_snapshot(
        state.state,
        prices,
        date=execution_date,
    )

    backtest_state = backtest_adapter._BacktestRunState(
        ledger=state.state,
        current_capital=starting_snapshot.equity,
        target_weights=target_weights.copy(),
    )
    from _strategy_test_helpers import momentum_test_strategy
    from portfolio_core.strategies.research_parameters import StockSelection
    strategy = momentum_test_strategy(selection=StockSelection(1, 1), turnover_threshold=0.0)
    backtest_context = SimpleNamespace(
        backtest_data=SimpleNamespace(
            data_close=pd.DataFrame([prices], index=[execution_date])
        ),
        strategy=strategy,
        apply_turnover_threshold=True,
        accounting_config=rules,
        apply_fees=True,
        apply_spread=True,
        strategy_audit=None,
    )
    period = SimpleNamespace(
        t_0=execution_date,
        dollar_volume=liquidity,
        interval_eligible_asset_ids=frozenset(prices.index),
        execution_sector_rows=pd.DataFrame({"GICS_Sector_Code": "20"}, index=prices.index),
    )
    backtest_result = backtest_adapter._process_rebalance_and_costs(
        backtest_context,
        backtest_state,
        period,
    )

    decision = StrategyDecision(
        raw_target_weights=target_weights,
        signal_audit=pd.DataFrame(index=prices.index),
        ranked_candidate_asset_ids=tuple(prices.index),
        original_long_asset_ids=("OLD_SHORT",),
        original_short_asset_ids=("OLD_LONG",),
        is_complete=True,
    )
    execution_decision = ExecutionDecision(
        final_target_weights=target_weights,
        final_long_asset_ids=("OLD_SHORT",),
        final_short_asset_ids=("OLD_LONG",),
    )
    metadata = pd.DataFrame(
        {
            "Asset_ID": prices.index,
            "Source_Ticker": prices.index,
            "Yahoo_Ticker": prices.index,
        }
    ).set_index("Asset_ID", drop=False)
    live_context = SimpleNamespace(
        inputs=SimpleNamespace(
            market_daily=pd.DataFrame(
                {
                    "Date": execution_date,
                    "Asset_ID": prices.index,
                    "Close": prices.to_numpy(),
                }
            ),
            sector_assignments=pd.DataFrame(),
        ),
        accounting=rules,
        metadata=metadata,
        strategy=strategy,
    )
    live_plan = SimpleNamespace(
        row=SimpleNamespace(
            Membership_Effective_Date=execution_date,
            Sizing_Field="Close",
            Execution_Field="Close",
        ),
        rebalance_id="R1",
        signal_cutoff=execution_date,
        sizing_date=execution_date,
        execution_date=execution_date,
        end_date=pd.Timestamp("2026-05-01"),
        dvol=liquidity,
        decision=decision,
        execution_decision=execution_decision,
        execution_eligible=set(prices.index),
        execution_sector_rows=pd.DataFrame({"GICS_Sector_Code": "20"}, index=prices.index),
    )
    monkeypatch.setattr(
        live_adapter,
        "trade_sector_audit_fields",
        lambda **_kwargs: {
            "GICS_Sector_Code": "20",
            "Sector": "Industrials",
            "Sector_As_Of_Date": execution_date,
            "Sector_Source_Type": "synthetic",
            "Sector_Source_Reference": "parity-test",
        },
    )
    monkeypatch.setattr(
        live_adapter,
        "strategy_signal_fields",
        lambda *_args, **_kwargs: {
            "Strategy_Score": 0.0,
            "Strategy_Rank": 1,
            "Strategy_Eligible": True,
            "Strategy_Exclusion_Reason": "",
            "Momentum": 0.0,
        },
    )
    live_result = live_adapter._execute_strategy_trades(
        live_context,
        live_adapter._StrategyRecords(),
        live_plan,
        state.state,
        state.state,
    )

    assert backtest_result.ledger == live_result.ledger_result
    assert backtest_result.ledger.order_count == 4
    assert backtest_result.ledger.fixed_fees == 8.0


def test_impossible_scaled_counts_raise_explicit_error():
    rules = _rules(
        initial_capital=100.0,
        minimum_positions=2,
        minimum_long_positions=1,
        minimum_short_positions=1,
    )
    with pytest.raises(
        InfeasibleRebalanceError,
        match=(
            "^No common scale factor produces a compliant rounded basket$"
        ),
    ):
        execute_rebalance(
            LedgerState.initial(rules),
            _plan(
                {"L": 1.0, "S": -1.0},
                {"L": 1.0, "S": -1.0},
                {"L": 100.0, "S": 100.0},
            ),
            rules,
        )
