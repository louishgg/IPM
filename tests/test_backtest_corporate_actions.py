"""Synthetic backtest coverage for explicit security-event economics."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import json

import numpy as np
import pandas as pd
import pytest

from backtest.config import DEFAULT_CONFIG
from backtest.data_loading import (
    BacktestDataset,
    derive_event_delivery_executions,
)
from backtest.price_sources import PRICE_OBSERVATION_COLUMNS
from backtest.engine import (
    _BacktestAuditCollector,
    _run_backtest_impl,
    run_backtest,
)
from portfolio_core.corporate_actions import (
    EVENT_COLUMNS,
    LEG_COLUMNS,
    SOURCE_COLUMNS,
    apply_corporate_actions,
)
from portfolio_core.portfolio_lifecycle import (
    asset_is_extinguished_from_events,
    attribute_event_interest_effects,
    eligible_assets_for_interval,
    event_valuation_result,
)
from portfolio_core.rebalance_planner import round_half_away_from_zero
from portfolio_core.sector_assignments import validate_sector_assignments
from _sector_test_helpers import (
    _synthetic_sector_assignments as _sector_assignments,
)
from _strategy_test_helpers import momentum_test_strategy
from portfolio_core.strategies.research_parameters import ExposureParameters, StockSelection


_STRATEGY = momentum_test_strategy(
    selection=StockSelection(20, 20), exposure=ExposureParameters(1.5, 0.7),
    turnover_threshold=0.015,
)


def _add_sector_asset(
    market: BacktestDataset,
    asset_id: str,
) -> None:
    dates = pd.DatetimeIndex(
        market.sector_assignments["As_Of_Date"].drop_duplicates()
    )
    additions = _sector_assignments(dates, [asset_id])
    market.sector_assignments = validate_sector_assignments(
        pd.concat([market.sector_assignments, additions], ignore_index=True)
    )


def _market() -> tuple[BacktestDataset, pd.DatetimeIndex]:
    dates = pd.date_range("2014-01-31", periods=30, freq="ME")
    assets = [f"T{i:02d}" for i in range(60)]
    time = np.arange(len(dates), dtype=float)[:, None]
    asset = np.arange(len(assets), dtype=float)[None, :]
    returns = (
        0.004
        + (asset - 29.5) * 0.00015
        + 0.008 * np.sin((time + 1.0) * (asset + 3.0) / 37.0)
    )
    close = pd.DataFrame(
        100.0 * np.cumprod(1.0 + returns, axis=0),
        index=dates,
        columns=assets,
    )
    volume = pd.DataFrame(
        np.broadcast_to(
            1_000_000.0 + asset * 25_000.0 + time * 1_000.0,
            close.shape,
        ),
        index=dates,
        columns=assets,
    )
    pit = pd.DataFrame(True, index=dates, columns=assets)
    market = BacktestDataset(
        data_close=close,
        data_volume=volume,
        pit_matrix=pit,
        sector_assignments=_sector_assignments(dates, assets),
        valid_trading_days=dates,
        rolling_dollar_vol=close * volume,
        asset_to_ticker={asset_id: asset_id for asset_id in assets},
    )
    return market, dates


def _install_event(
    market: BacktestDataset,
    *,
    event_id: str,
    date: pd.Timestamp,
    event_type: str,
    continuity: str,
    legs: list[dict[str, object]],
) -> None:
    market.security_events = pd.DataFrame(
        [
            {
                "Event_ID": event_id,
                "Effective_Date": pd.Timestamp(date),
                "Event_Type": event_type,
                "Continuity_Class": continuity,
                "Review_Status": "approved",
            }
        ],
        columns=EVENT_COLUMNS,
    )
    market.security_event_legs = pd.DataFrame(legs, columns=LEG_COLUMNS)
    market.security_event_sources = pd.DataFrame(
        [
            {
                "Event_ID": event_id,
                "Source_URL": f"https://example.test/{event_id}",
                "Review_Status": "approved",
            }
        ],
        columns=SOURCE_COLUMNS,
    )


def _install_event_execution_observation(
    market: BacktestDataset,
    *,
    event_id: str,
    asset_id: str,
    execution_date: pd.Timestamp,
    close: float,
    volume: float = 1_000_000.0,
) -> None:
    """Derive one compact execution from selected and distracting raw rows."""

    mapping_id = f"MAP-TEST-{event_id}-{asset_id}"
    mappings = pd.DataFrame([{
        "Scope": "backtest",
        "Event_ID": event_id,
        "Asset_ID": asset_id,
        "Provider": "yahoo",
        "Provider_Symbol": asset_id,
        "Mapping_ID": mapping_id,
        "Resolution_Method": "reviewed_effective_symbol",
        "Review_Status": "approved",
        "Effective_Start": execution_date,
        "Effective_End": "",
    }])
    observations = pd.DataFrame([
        {
            "Observation_Date": execution_date - pd.Timedelta(days=1),
            "Asset_ID": asset_id,
            "Provider": "yahoo",
            "Provider_Symbol": asset_id,
            "Mapping_ID": "MAP-UNSELECTED-EARLIER-IDENTITY",
            "Price_Close": 1.0,
            "Volume": volume,
        },
        {
            "Observation_Date": execution_date,
            "Asset_ID": asset_id,
            "Provider": "yahoo",
            "Provider_Symbol": asset_id,
            "Mapping_ID": mapping_id,
            "Price_Close": close,
            "Volume": volume,
        },
    ], columns=PRICE_OBSERVATION_COLUMNS)
    market.event_delivery_executions = derive_event_delivery_executions(
        events=market.security_events,
        legs=market.security_event_legs,
        mappings=mappings,
        observations=observations,
        pit_matrix=market.pit_matrix,
    )


def _leg(
    event_id: str,
    from_asset: str,
    leg_type: str,
    *,
    order: int = 1,
    to_asset: str = "",
    quantity: str = "",
    cash: str = "",
    currency: str = "",
    consumes: bool = True,
) -> dict[str, object]:
    return {
        "Event_ID": event_id,
        "Leg_Order": order,
        "From_Asset_ID": from_asset,
        "To_Asset_ID": to_asset,
        "Leg_Type": leg_type,
        "Quantity_Per_From_Share": quantity,
        "Cash_Per_From_Share": cash,
        "Currency": currency,
        "CVR_Units_Per_From_Share": "",
        "CVR_Base_Value_Per_Unit": "",
        "CVR_Max_Value_Per_Unit": "",
        "Consumes_From_Position": consumes,
        "Review_Status": "approved",
    }


@pytest.fixture(scope="module")
def first_period() -> tuple[pd.Timestamp, pd.Timestamp, str, float]:
    market, dates = _market()
    _, _, holdings = run_backtest(
        market,
        dates,
        _STRATEGY,
        return_diagnostics=True,
        return_holdings=True,
    )
    first = holdings.iloc[0]
    return (
        pd.Timestamp(first["Date"]),
        pd.Timestamp(first["Next_Date"]),
        str(first["Asset_ID"]),
        float(first["Weight"]),
    )


def _install_distribution_execution(
    market: BacktestDataset,
    t0: pd.Timestamp,
    parent: str,
    child: str,
    event_id: str,
    *,
    pit_eligible: bool,
    close: float = 50.0,
) -> pd.Timestamp:
    execution_date = t0 + pd.Timedelta(days=15)
    _install_event(
        market,
        event_id=event_id,
        date=t0 + pd.Timedelta(days=10),
        event_type="distribution",
        continuity="predecessor_survives",
        legs=[_leg(
            event_id, parent, "distribution", to_asset=child,
            quantity="0.25", consumes=False,
        )],
    )
    market.asset_to_ticker[child] = child
    market.pit_matrix[child] = pit_eligible
    if pit_eligible:
        market.data_close[child] = np.nan
        market.data_volume[child] = np.nan
        market.rolling_dollar_vol[child] = np.nan
    _install_event_execution_observation(
        market,
        event_id=event_id,
        asset_id=child,
        execution_date=execution_date,
        close=close,
    )
    return execution_date


def test_cash_event_supplies_missing_end_price_and_complete_predecessor_return(first_period):
    market, dates = _market()
    t0, t1, asset_id, _ = first_period
    event_id = "EVT-20150215-T06-CASH"
    settlement = Decimal("150")
    _install_event(
        market,
        event_id=event_id,
        date=t0 + pd.Timedelta(days=15),
        event_type="cash_settlement",
        continuity="predecessor_extinguished",
        legs=[
            _leg(
                event_id,
                asset_id,
                "cash",
                cash=str(settlement),
                currency="USD",
            )
        ],
    )
    market.data_close.loc[t1, asset_id] = np.nan
    market.pit_matrix.loc[market.pit_matrix.index >= t1, asset_id] = False
    audit: list[dict[str, object]] = []

    nav, _, holdings = _run_backtest_impl(
        market,
        dates,
        _STRATEGY,
        return_diagnostics=True,
        return_holdings=True,
        corporate_action_audit_records=audit,
    )

    assert np.isfinite(nav.to_numpy()).all()
    predecessor = holdings.loc[
        holdings["Date"].eq(t0) & holdings["Asset_ID"].eq(asset_id)
    ].iloc[0]
    start_price = Decimal(str(market.data_close.loc[t0, asset_id]))
    assert predecessor["Stock_Return"] == pytest.approx(
        float(settlement / start_price - Decimal("1"))
    )
    action = next(row for row in audit if row["Record_Type"] == "corporate_action")
    assert action["Event_ID"] == event_id
    assert action["Cash_Per_From_Share"] == settlement
    assert action["Fixed_Fee"] == Decimal("0")
    assert action["Spread_Cost"] == Decimal("0")
    financing = next(row for row in audit if row["Record_Type"] == "financing_summary")
    assert financing["Interest_Effect"] != 0.0
    assert action["Interest_Effect"] == pytest.approx(
        financing["Interest_Effect"]
    )


def test_stock_exchange_carries_exact_successor_units_outside_membership(first_period):
    market, dates = _market()
    t0, t1, asset_id, initial_weight = first_period
    successor = "SUCCESSOR"
    market.data_close[successor] = 200.0
    market.data_volume[successor] = 1_000_000.0
    market.rolling_dollar_vol[successor] = 200_000_000.0
    market.pit_matrix[successor] = False
    _add_sector_asset(market, successor)
    market.pit_matrix.loc[market.pit_matrix.index >= t1, asset_id] = False
    event_id = "EVT-20150215-T06-SUCCESSOR-STOCK"
    ratio = Decimal("0.5")
    _install_event(
        market,
        event_id=event_id,
        date=t0 + pd.Timedelta(days=15),
        event_type="stock_exchange",
        continuity="predecessor_extinguished",
        legs=[
            _leg(
                event_id,
                asset_id,
                "stock",
                to_asset=successor,
                quantity=str(ratio),
            )
        ],
    )
    audit: list[dict[str, object]] = []

    _, _, holdings = _run_backtest_impl(
        market,
        dates,
        _STRATEGY,
        return_diagnostics=True,
        return_holdings=True,
        corporate_action_audit_records=audit,
    )

    predecessor = holdings.loc[
        holdings["Date"].eq(t0) & holdings["Asset_ID"].eq(asset_id)
    ].iloc[0]
    start_price = Decimal(str(market.data_close.loc[t0, asset_id]))
    end_successor_price = Decimal(str(market.data_close.loc[t1, successor]))
    assert predecessor["Stock_Return"] == pytest.approx(
        float(ratio * end_successor_price / start_price - Decimal("1"))
    )
    action = next(row for row in audit if row["Record_Type"] == "corporate_action")
    successor_units = Fraction(
        json.loads(str(action["Successor_Units_JSON"]))[successor]
    )
    initial_shares = int(
        round_half_away_from_zero(
            pd.Series({asset_id: initial_weight * 1_000_000.0 / float(start_price)})
        ).loc[asset_id]
    )
    expected_units = Fraction(initial_shares) * Fraction(ratio)
    assert successor_units == expected_units
    assert successor not in set(market.pit_matrix.loc[t0][market.pit_matrix.loc[t0]])


def test_event_delivered_off_universe_child_is_liquidated_once_at_first_price(first_period):
    market, dates = _market()
    t0, _, asset_id, _ = first_period
    child = "OFF_UNIVERSE_CHILD"
    event_id = "EVT-20150210-PARENT-CHILD-DISTRIBUTION"
    reference_close = 50.0
    execution_date = _install_distribution_execution(
        market, t0, asset_id, child, event_id,
        pit_eligible=False,
        close=reference_close,
    )
    audit: list[dict[str, object]] = []
    strategy_audit = _BacktestAuditCollector(_STRATEGY)

    _run_backtest_impl(
        market,
        dates,
        _STRATEGY,
        corporate_action_audit_records=audit,
        strategy_audit=strategy_audit,
    )

    forced_trades = [
        row
        for row in strategy_audit.trade_records
        if row["Execution_Reason"] == "mandatory_off_universe_liquidation"
    ]
    assert len(forced_trades) == 1
    forced = forced_trades[0]
    assert forced["Execution_Date"] == execution_date
    assert forced["Execution_Price"] == reference_close
    assert forced["Applied_Target_Shares"] == 0.0
    assert forced["Applied_Rule"] == "mandatory_off_universe_liquidation"
    assert forced["Trigger_Event_ID"] == event_id


def test_pit_eligible_child_cannot_use_execution_quote_for_eligibility(first_period):
    market, dates = _market()
    t0, t1, asset_id, _ = first_period
    child = "ELIGIBLE_CHILD"
    event_id = "EVT-20150210-PARENT-ELIGIBLE-DISTRIBUTION"
    _install_distribution_execution(
        market, t0, asset_id, child, event_id,
        pit_eligible=True,
    )
    assert market.event_delivery_executions.empty
    assert asset_id not in eligible_assets_for_interval(
        market, [asset_id], t0, t1
    )


def test_explicit_distribution_cannot_use_lingering_predecessor_price(first_period):
    market, dates = _market()
    t0, t1, asset_id, _ = first_period
    event_id = "EVT-20150215-T06-CHILD-DISTRIBUTION"
    child = "CHILD"
    _install_event(
        market,
        event_id=event_id,
        date=t0 + pd.Timedelta(days=15),
        event_type="distribution",
        continuity="predecessor_survives",
        legs=[
            _leg(
                event_id,
                asset_id,
                "distribution",
                to_asset=child,
                quantity="0.25",
                consumes=False,
            )
        ],
    )

    assert market.data_close.loc[t1, asset_id] > 0.0
    assert event_valuation_result(market, asset_id, t0, t1) is None

    market.data_close[child] = 50.0
    assert event_valuation_result(market, asset_id, t0, t1) is not None


def test_short_cash_event_releases_restricted_proceeds_without_trade_costs():
    market, dates = _market()
    _, _, baseline_holdings = run_backtest(
        market,
        dates,
        _STRATEGY,
        return_diagnostics=True,
        return_holdings=True,
    )
    first_date = pd.Timestamp(baseline_holdings["Date"].min())
    short = baseline_holdings.loc[
        baseline_holdings["Date"].eq(first_date)
        & baseline_holdings["Weight"].lt(0.0)
    ].iloc[0]
    t0 = pd.Timestamp(short["Date"])
    t1 = pd.Timestamp(short["Next_Date"])
    asset_id = str(short["Asset_ID"])
    event_id = "EVT-20150215-SHORT-CASH"
    _install_event(
        market,
        event_id=event_id,
        date=t0 + pd.Timedelta(days=15),
        event_type="cash_settlement",
        continuity="predecessor_extinguished",
        legs=[
            _leg(
                event_id,
                asset_id,
                "cash",
                cash="70",
                currency="USD",
            )
        ],
    )
    market.pit_matrix.loc[market.pit_matrix.index >= t1, asset_id] = False
    audit: list[dict[str, object]] = []

    _run_backtest_impl(
        market,
        dates,
        _STRATEGY,
        corporate_action_audit_records=audit,
    )

    action = next(row for row in audit if row["Record_Type"] == "corporate_action")
    assert action["Shares_Before"] < 0
    assert action["Cash_Effect"] < 0
    assert action["Restricted_Proceeds_Released"] > 0
    assert action["Restricted_Proceeds_Transferred"] == Decimal("0")
    assert action["Fixed_Fee"] == Decimal("0")
    assert action["Spread_Cost"] == Decimal("0")
    financing = next(row for row in audit if row["Record_Type"] == "financing_summary")
    assert action["Interest_Effect"] == pytest.approx(
        financing["Interest_Effect"]
    )


def test_chained_event_interest_effects_reconcile_cash_and_short_release_timing():
    first = "EVT-20260111-SHORT-CASH"
    second = "EVT-20260121-LONG-CASH"
    events = pd.DataFrame(
        [
            {
                "Event_ID": first,
                "Effective_Date": "2026-01-11",
                "Event_Type": "cash_settlement",
                "Continuity_Class": "predecessor_extinguished",
                "Review_Status": "approved",
            },
            {
                "Event_ID": second,
                "Effective_Date": "2026-01-21",
                "Event_Type": "cash_settlement",
                "Continuity_Class": "predecessor_extinguished",
                "Review_Status": "approved",
            },
        ],
        columns=EVENT_COLUMNS,
    )
    legs = pd.DataFrame(
        [
            _leg(first, "SHORT", "cash", cash="70", currency="USD"),
            _leg(second, "LONG", "cash", cash="50", currency="USD"),
        ],
        columns=LEG_COLUMNS,
    )
    sources = pd.DataFrame(
        [
            {
                "Event_ID": event_id,
                "Source_URL": f"https://example.test/{event_id}",
                "Review_Status": "approved",
            }
            for event_id in (first, second)
        ],
        columns=SOURCE_COLUMNS,
    )
    result = apply_corporate_actions(
        {"SHORT": -1, "LONG": 1},
        {"SHORT": 100},
        events,
        legs,
        sources,
        start_exclusive="2026-01-01",
        end_inclusive="2026-01-31",
    )

    effects, actual, counterfactual = attribute_event_interest_effects(
        1_000.0,
        pd.Series({"SHORT": 100.0}),
        result,
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-31"),
        DEFAULT_CONFIG.accounting,
    )

    assert tuple(effects) == (first, second)
    assert effects[first] > 0.0  # $70 payment plus $100 restriction release.
    assert effects[second] > 0.0  # $50 receipt earns interest from January 21.
    assert sum(effects.values()) == pytest.approx(actual - counterfactual)


def test_effective_event_excludes_stale_predecessor_before_new_ranking(first_period):
    market, dates = _market()
    t0, _, asset_id, _ = first_period
    event_id = "EVT-20220228-INFO-SPGI-STOCK-EXCHANGE"
    successor = "SUCCESSOR"
    market.data_close[successor] = 200.0
    market.data_volume[successor] = 1_000_000.0
    market.rolling_dollar_vol[successor] = 200_000_000.0
    market.pit_matrix[successor] = False
    _add_sector_asset(market, successor)
    _install_event(
        market,
        event_id=event_id,
        date=t0,
        event_type="stock_exchange",
        continuity="predecessor_extinguished",
        legs=[
            _leg(
                event_id,
                asset_id,
                "stock",
                to_asset=successor,
                quantity="0.2838",
            )
        ],
    )

    assert market.pit_matrix.loc[t0, asset_id]
    assert market.data_close.loc[t0, asset_id] > 0.0
    assert asset_is_extinguished_from_events(
        market.security_events,
        market.security_event_legs,
        asset_id,
        t0,
    )

    _, _, holdings = run_backtest(
        market,
        dates,
        _STRATEGY,
        return_diagnostics=True,
        return_holdings=True,
    )

    assert asset_id not in set(holdings.loc[holdings["Date"].ge(t0), "Asset_ID"])


@pytest.mark.parametrize("asset,event_type,continuity,legs,extinguished", [
    ("CCEP.O", "cash_and_stock", "predecessor_extinguished", [
        _leg("event", "CCEP.O", "stock", to_asset="CCEP.O", quantity="1"),
        _leg("event", "CCEP.O", "cash", order=2, cash="14.50", currency="USD"),
    ], False),
    ("HWM", "distribution", "predecessor_survives", [
        _leg("event", "HWM", "distribution", to_asset="UNPRICED::ARNC", quantity="1/4"),
        _leg("event", "HWM", "relabel", order=2, to_asset="HWM", quantity="1"),
    ], False),
    ("PARENT", "distribution", "predecessor_survives", [
        _leg("event", "PARENT", "relabel", to_asset="PARENT", quantity="250/299"),
        _leg("event", "PARENT", "distribution", order=2, to_asset="CHILD", quantity="125/598"),
    ], True),
], ids=["same-key-stock", "unit-relabel", "nonunit-relabel"])
def test_same_key_event_preserves_only_unchanged_candidate_units(
    asset, event_type, continuity, legs, extinguished,
):
    date = pd.Timestamp("2024-04-01")
    events = pd.DataFrame([{
        "Event_ID": "event", "Effective_Date": date, "Event_Type": event_type,
        "Continuity_Class": continuity, "Review_Status": "approved",
    }], columns=EVENT_COLUMNS)
    event_legs = pd.DataFrame(legs, columns=LEG_COLUMNS)
    assert not asset_is_extinguished_from_events(events, event_legs, asset, date - pd.Timedelta(days=1))
    assert asset_is_extinguished_from_events(events, event_legs, asset, date) is extinguished


def test_event_free_window_has_exact_numerical_parity_with_identical_state():
    """A future approved event must not perturb an earlier event-free strategy."""
    baseline_market, dates = _market()
    event_aware_market, event_dates = _market()
    assert dates.equals(event_dates)
    comparison_dates = dates[:-1]
    event_id = "EVT-20160531-FUTURE-CASH"
    _install_event(
        event_aware_market,
        event_id=event_id,
        date=dates[-1],
        event_type="cash_settlement",
        continuity="predecessor_extinguished",
        legs=[
            _leg(
                event_id,
                "T00",
                "cash",
                cash="125",
                currency="USD",
            )
        ],
    )

    baseline = run_backtest(
        baseline_market,
        comparison_dates,
        _STRATEGY,
        return_diagnostics=True,
        return_holdings=True,
    )
    event_aware = run_backtest(
        event_aware_market,
        comparison_dates,
        _STRATEGY,
        return_diagnostics=True,
        return_holdings=True,
    )

    pd.testing.assert_series_equal(event_aware[0], baseline[0], check_exact=True)
    pd.testing.assert_frame_equal(event_aware[1], baseline[1], check_exact=True)
    pd.testing.assert_frame_equal(event_aware[2], baseline[2], check_exact=True)


def test_unheld_structured_event_is_audited_without_changing_results(first_period):
    baseline_market, dates = _market()
    baseline_nav = _run_backtest_impl(baseline_market, dates, _STRATEGY)
    event_market, event_dates = _market()
    assert dates.equals(event_dates)
    t0, _, _, _ = first_period
    event_id = "EVT-20150215-UNHELD-CASH"
    _install_event(
        event_market,
        event_id=event_id,
        date=t0 + pd.Timedelta(days=15),
        event_type="cash_settlement",
        continuity="predecessor_extinguished",
        legs=[
            _leg(
                event_id,
                "UNHELD",
                "cash",
                cash="100",
                currency="USD",
            )
        ],
    )
    audit: list[dict[str, object]] = []

    event_nav = _run_backtest_impl(
        event_market,
        dates,
        _STRATEGY,
        corporate_action_audit_records=audit,
    )

    pd.testing.assert_series_equal(event_nav, baseline_nav, check_exact=True)
    action = next(row for row in audit if row["Record_Type"] == "corporate_action")
    assert action["Event_ID"] == event_id
    assert not action["Applied_To_Position"]
    assert action["Shares_Before"] == 0
    assert action["Cash_Effect"] == 0
    assert action["Interest_Effect"] == 0.0


def test_first_fixed_parameter_divergence_is_explained_by_approved_event_id():
    """The first difference from continuous-price behavior has provenance."""
    baseline_market, dates = _market()
    baseline_nav, _, baseline_holdings = run_backtest(
        baseline_market,
        dates,
        _STRATEGY,
        return_diagnostics=True,
        return_holdings=True,
    )
    first_holding = baseline_holdings.iloc[0]
    t0 = pd.Timestamp(first_holding["Date"])
    t1 = pd.Timestamp(first_holding["Next_Date"])
    asset_id = str(first_holding["Asset_ID"])

    event_market, event_dates = _market()
    assert dates.equals(event_dates)
    event_id = "EVT-20150215-FIRST-APPROVED-DIVERGENCE"
    source_url = f"https://example.test/{event_id}"
    _install_event(
        event_market,
        event_id=event_id,
        date=t0 + pd.Timedelta(days=15),
        event_type="cash_settlement",
        continuity="predecessor_extinguished",
        legs=[
            _leg(
                event_id,
                asset_id,
                "cash",
                cash="150",
                currency="USD",
            )
        ],
    )
    audit: list[dict[str, object]] = []

    event_nav = _run_backtest_impl(
        event_market,
        dates,
        _STRATEGY,
        corporate_action_audit_records=audit,
    )

    common_dates = baseline_nav.index.intersection(event_nav.index)
    baseline_common = baseline_nav.reindex(common_dates)
    event_common = event_nav.reindex(common_dates)
    differs = ~np.isclose(
        baseline_common.to_numpy(),
        event_common.to_numpy(),
        rtol=0.0,
        atol=1e-9,
    )
    assert differs.any()
    first_divergence = pd.Timestamp(common_dates[np.flatnonzero(differs)[0]])
    assert first_divergence == t1
    pd.testing.assert_series_equal(
        event_common.loc[event_common.index < first_divergence],
        baseline_common.loc[baseline_common.index < first_divergence],
        check_exact=True,
    )

    approved_ids = set(
        event_market.security_events.loc[
            event_market.security_events["Review_Status"].eq("approved"),
            "Event_ID",
        ].astype(str)
    )
    action = next(
        row
        for row in audit
        if row["Record_Type"] == "corporate_action"
        and pd.Timestamp(row["Period_End"]) == first_divergence
    )
    assert action["Event_ID"] == event_id
    assert action["Event_ID"] in approved_ids
    assert pd.Timestamp(action["Period_Start"]) == t0
    assert source_url in json.loads(str(action["Source_URLs_JSON"]))
