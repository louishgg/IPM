"""Bounded reproductions of research execution and evaluation audit defects."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from backtest import evaluation
from backtest.engine import _BacktestAuditCollector, _run_backtest_impl, run_backtest
from live import analysis as live
from live.config import DEFAULT_CONFIG as LIVE_CONFIG
from portfolio_core.accounting_config import DEFAULT_ACCOUNTING_CONFIG
from portfolio_core.accounting_ledger import InfeasibleRebalanceError
from portfolio_core.strategies import build_registered_strategy
from portfolio_core.strategies.momentum import MomentumStrategy
from portfolio_core.strategies.low_volatility import LowVolatilityStrategy
from portfolio_core.strategies.research_execution import require_positive_equity, sector_execution_residuals
from portfolio_core.strategies.research_hold import cap_carried_targets
from portfolio_core.strategies.research_parameters import (
    BufferParameters, ExposureParameters, SignalParameters, StockSelection, VolatilityProfile,
)
from _strategy_test_helpers import momentum_test_parameters
from test_engine_offline import make_synthetic_backtest_data
from test_backtest_corporate_actions import _install_event, _leg
from test_live_research_history import _historical_inputs


def _shorts_before_shock(data, dates, strategy):
    _, holdings = run_backtest(data, dates, strategy, return_holdings=True)
    return holdings.loc[
        holdings.Date.eq(dates[-2]) & holdings.Weight.lt(0), "Asset_ID"
    ].tolist()


@pytest.mark.parametrize("extra_interval", [False, True])
def test_research_insolvency_fails_on_valuation_date_including_terminal(extra_interval):
    data, dates = make_synthetic_backtest_data()
    strategy = MomentumStrategy(momentum_test_parameters(exposure=ExposureParameters(2, .5)))
    window = dates[12:15 if extra_interval else 14]
    shorts = _shorts_before_shock(data, window[:2], strategy)
    data.data_close.loc[window[1]:, shorts] *= 3
    with pytest.raises(InfeasibleRebalanceError, match=f"{window[1].date()}.*nonpositive"):
        run_backtest(data, window, strategy)


@pytest.mark.parametrize("equity", [0., -1.])
def test_sizing_insolvency_is_distinct_from_invalid_valuation(equity):
    with pytest.raises(InfeasibleRebalanceError, match="2020-01-31.*nonpositive"):
        require_positive_equity(equity, "2020-01-31")
    with pytest.raises(ValueError, match="finite"):
        require_positive_equity(np.nan, "2020-01-31")


@pytest.mark.parametrize("counts", [(1, 1), (8, 12), (12, 8)])
def test_default_account_minima_survive_small_research_packets_in_both_domains(counts):
    strategy = build_registered_strategy("momentum", momentum_test_parameters(
        selection=StockSelection(*counts),
    ).payload())
    data, dates = make_synthetic_backtest_data()
    with pytest.raises(InfeasibleRebalanceError, match="position counts"):
        run_backtest(data, dates[12:14], strategy)
    config = replace(LIVE_CONFIG, strategy=strategy, paths=LIVE_CONFIG.paths.for_strategy("momentum"))
    with pytest.raises(InfeasibleRebalanceError, match="position counts"):
        live.run_strategy_analysis(_historical_inputs(), config)


def test_small_execution_fixture_requires_explicit_account_override():
    strategy = MomentumStrategy(momentum_test_parameters(selection=StockSelection(1, 1)))
    accounting = replace(DEFAULT_ACCOUNTING_CONFIG,
        minimum_positions=2, minimum_long_positions=1, minimum_short_positions=1)
    data, dates = make_synthetic_backtest_data()
    _, diag = run_backtest(data, dates[12:14], strategy,
        accounting_config=accounting, return_diagnostics=True)
    assert diag.Position_Count.tolist() == [2]
    assert diag.Long_Count.tolist() == diag.Short_Count.tolist() == [1]


def test_whole_sector_baskets_cannot_weaken_a_larger_account_minimum():
    from test_sector_momentum_research import dataset, packet
    from portfolio_core.strategies.sector_momentum import SectorMomentumStrategy
    data = dataset()  # Two complete 30-stock baskets.
    strategy = SectorMomentumStrategy(packet())
    accounting = replace(DEFAULT_ACCOUNTING_CONFIG, minimum_long_positions=31)
    with pytest.raises(InfeasibleRebalanceError, match="position counts"):
        run_backtest(data, pd.date_range("2015-02-28", periods=2, freq="ME"),
            strategy, accounting_config=accounting)


@pytest.mark.parametrize("gross", [1., 2.])
def test_live_neutral_suppression_reconciles_sizing_and_overnight_effects(gross):
    strategy = MomentumStrategy(momentum_test_parameters(
        sector_neutral=True, turnover_threshold=.05,
        buffer=BufferParameters(True, 1.5), exposure=ExposureParameters(gross, .5),
    ))
    config = replace(LIVE_CONFIG, strategy=strategy, paths=LIVE_CONFIG.paths.for_strategy("momentum"))
    result = live.run_strategy_analysis(_historical_inputs(), config)
    residuals = result.sector_residuals
    assert residuals.Constraint_Override.eq("neutrality_suppression_override").any()
    np.testing.assert_allclose(residuals.Target_Net_Dollars, 0, atol=1e-7)
    np.testing.assert_allclose(residuals.Suppression_Net_Dollars, 0, atol=1e-7)
    np.testing.assert_allclose(residuals.Applied_Net_At_Sizing_Dollars,
                               residuals.Mechanical_Net_Dollars, atol=1e-7)
    np.testing.assert_allclose(residuals.Applied_Net_Dollars,
        residuals.Target_Net_Dollars + residuals.Sizing_Rounding_Effect_Dollars
        + residuals.Turnover_Suppression_Effect_Dollars + residuals.Feasibility_Effect_Dollars
        + residuals.Overnight_Price_Effect_Dollars, atol=1e-7)
    assert residuals.Overnight_Price_Effect_Dollars.abs().gt(1e-7).any()
    if gross == 1:
        assert residuals.Feasibility_Scale.eq(1).all()
    else:
        assert residuals.Feasibility_Scale.lt(1).any()


def test_residuals_include_fully_suppressed_sector_and_execution_event_fractions():
    # Both positions would be absent from a trade-only audit when suppressed.
    current = pd.Series({"a": 20., "b": -10.})
    desired = pd.Series({"a": 10., "b": -10.})
    prices = pd.Series({"a": 100., "b": 100.})
    sectors = pd.Series({"a": "10", "b": "10"})
    values = sector_execution_residuals(desired, current, current, 1., prices, sectors,
        whole_share_orders=True)
    assert values.loc["10", "Suppression_Net_Dollars"] == 1000

    desired = pd.Series({"a": 10.25, "b": -10.25})
    execution_current = pd.Series({"a": .25, "b": -.25})
    applied = pd.Series({"a": 5.25, "b": -5.25})
    values = sector_execution_residuals(desired, execution_current, applied, .5, prices,
        pd.Series({"a": "10", "b": "15"}), whole_share_orders=True)
    assert values.Mechanical_Net_Dollars.tolist() == [525., -525.]
    assert values.Suppression_Net_Dollars.eq(0).all()


def test_live_terminal_insolvency_is_not_reported_as_performance():
    inputs = _historical_inputs()
    strategy = MomentumStrategy(momentum_test_parameters(exposure=ExposureParameters(2, .5)))
    config = replace(LIVE_CONFIG, strategy=strategy, paths=LIVE_CONFIG.paths.for_strategy("momentum"))
    baseline = live.run_strategy_analysis(inputs, config)
    shorts = baseline.holdings.loc[
        baseline.holdings.Rebalance_ID.eq("R4") & baseline.holdings.Weight.lt(0), "Asset_ID"
    ]
    terminal = pd.Timestamp(inputs.schedule.Valuation_End.max())
    market = inputs.market_daily.copy()
    mask = market.Date.eq(terminal) & market.Asset_ID.isin(shorts)
    assert mask.sum() == 10
    market.loc[mask, "Close"] *= 3
    with pytest.raises(InfeasibleRebalanceError, match=f"{terminal.date()}.*nonpositive"):
        live.run_strategy_analysis(replace(inputs, market_daily=market), config)


@pytest.mark.parametrize("strategy_gross,account_cap,held_gross,expected", [
    (1., 2., 1.5, 1.), (2., 1.5, 2.2, 1.5), (1.5, 2., 1., 1.),
])
def test_carried_gross_only_scales_down_and_preserves_relative_weights(
    strategy_gross, account_cap, held_gross, expected,
):
    strategy = MomentumStrategy(momentum_test_parameters(exposure=ExposureParameters(strategy_gross, .5)))
    weights = pd.Series({"a": held_gross * .3, "b": held_gross * .2, "c": -held_gross * .5})
    capped, audit = cap_carried_targets(weights, strategy,
        replace(DEFAULT_ACCOUNTING_CONFIG, maximum_gross_exposure=account_cap))
    assert capped.abs().sum() == pytest.approx(expected)
    np.testing.assert_allclose(capped / weights, expected / held_gross)
    assert audit["Carried_Gross_Before"] == pytest.approx(held_gross)
    assert audit["Carried_Gross_After"] == pytest.approx(expected)
    assert audit["Carried_Gross_Scale"] <= 1


def test_corporate_action_drift_can_use_incomplete_hold_after_gross_cap():
    data, dates = make_synthetic_backtest_data()
    window = dates[12:16]

    class MissingAfterEvent(MomentumStrategy):
        def _decide(self, context):
            decision = super()._decide(context)
            if context.signal_cutoff >= window[1]:
                return replace(decision, raw_target_weights=pd.Series(dtype=float),
                    original_long_asset_ids=(), original_short_asset_ids=(), is_complete=False)
            return decision

    strategy = MissingAfterEvent(momentum_test_parameters(exposure=ExposureParameters(2, .5)))
    shorts = _shorts_before_shock(data, window[:2], strategy)
    data.data_close.loc[window[1]:, shorts] *= 1.1
    asset = shorts[0]
    _install_event(data, event_id="same-key-drift", date=window[1] - pd.Timedelta(days=2),
        event_type="stock_exchange", continuity="predecessor_extinguished",
        legs=[_leg("same-key-drift", asset, "stock", to_asset=asset, quantity="1")])
    collector = _BacktestAuditCollector(strategy)
    nav, diag = _run_backtest_impl(data, window, strategy,
        strategy_audit=collector, return_diagnostics=True)
    holds = pd.DataFrame(collector.research_records).query("Status == 'eligible_prior_target_hold'")
    assert holds.Carried_Gross_Before.iloc[0] > 2
    assert holds.Carried_Gross_After.eq(2).all()
    assert holds.Carried_Gross_Override.iloc[0] == "configured_gross_cap"
    assert holds.Carried_Gross_Scale.iloc[1] == 1
    executions = pd.DataFrame(collector.research_records).query("Status == 'executed'")
    assert executions.Post_Trade_Gross.le(2 + 1e-10).all()
    assert diag.Position_Count.eq(20).all()
    assert len(nav) == len(window)


def test_independent_window_rejects_delayed_activation_but_development_allows_it():
    data, _ = make_synthetic_backtest_data()
    strategy = LowVolatilityStrategy(momentum_test_parameters(
        signal=SignalParameters("low_volatility", None, None, VolatilityProfile(60, 24), None),
    ))
    dates = pd.date_range("2015-12-31", "2016-03-31", freq="ME")
    with pytest.raises(InfeasibleRebalanceError, match="2015-12-31.*required NAV endpoint"):
        evaluation.evaluate_strategy(data, dates, strategy)
    nav = evaluation.evaluate_strategy(data, dates, strategy, require_complete_window=False)
    assert nav.index.equals(dates[1:])
    assert len(nav.pct_change(fill_method=None).dropna()) == 2


@pytest.mark.parametrize("defect", ["interior_gap", "nonfinite"])
def test_independent_window_data_errors_are_fatal(monkeypatch, defect):
    dates = pd.date_range("2015-01-31", periods=4, freq="ME")
    nav = pd.Series(1e6, index=dates)
    nav = nav.drop(dates[1]) if defect == "interior_gap" else nav.mask(nav.index == dates[1], np.inf)
    monkeypatch.setattr(evaluation, "_run_backtest_impl", lambda *a, **k: nav)
    with pytest.raises(ValueError, match="complete requested window|finite"):
        evaluation.evaluate_strategy(None, dates, MomentumStrategy(momentum_test_parameters()))


def test_complete_window_policy_reaches_net_gross_and_spread_sensitivity(monkeypatch):
    from test_reporting_configuration_matrix import EVALUATION
    data, _ = make_synthetic_backtest_data()
    strategy = MomentumStrategy(momentum_test_parameters())
    original = evaluation.evaluate_strategy
    policies = []
    def evaluate(*args, **kwargs):
        policies.append(kwargs["require_complete_window"])
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluation, "evaluate_strategy", evaluate)
    result = evaluation.run_strategy_evaluation(data, strategy, evaluation_config=EVALUATION)
    assert policies == [False, True, False, False, True, False]
    sensitivity = evaluation.calculate_spread_sensitivity(data, result)
    assert policies[6:] == [False, True, False, False, True, False]
    assert len(sensitivity) == 3
