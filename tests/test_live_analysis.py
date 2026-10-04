"""Offline causal strategy and attribution regression tests."""
from dataclasses import replace
from fractions import Fraction
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import live.analysis as live_analysis
import live.brinson_attribution as live_brinson
from live.analysis import (
    live_decision_audit_columns,
    live_trade_audit_columns,
    run_strategy_analysis,
)
from live.brinson_attribution import (
    build_live_benchmark_sector_series,
)
from live.config import DEFAULT_CONFIG
from live.corporate_action_policy import LiveCorporateActionBundle
from live.analysis_data import validate_analysis_inputs
from live.analysis_data import LiveAnalysisResult
from live.preparation_artifacts import PREPARE_COMMAND
from live.strategy_universe import (
    DOWNLOAD_BENCHMARK_COMMAND,
    DOWNLOAD_PRICES_COMMAND,
    sector_assignment_requirements,
)
from portfolio_core.strategies.research_parameters import (
    BufferParameters, ExposureParameters, StockSelection,
)
from portfolio_core.accounting_ledger import project_financing_amounts
from portfolio_core.price_basis import price_basis_spec
from portfolio_core.simulation_assumptions import build_simulation_assumptions
from _strategy_test_helpers import momentum_test_strategy


_STRATEGY = momentum_test_strategy(
    selection=StockSelection(20, 20), exposure=ExposureParameters(1.0, 0.7),
    buffer=BufferParameters(True, 1.6), turnover_threshold=0.01,
)

_GICS_SECTORS = (
    ("10", "Energy"),
    ("15", "Materials"),
    ("20", "Industrials"),
    ("25", "Consumer Discretionary"),
    ("30", "Consumer Staples"),
    ("35", "Health Care"),
    ("40", "Financials"),
    ("45", "Information Technology"),
    ("50", "Communication Services"),
    ("55", "Utilities"),
)


def _sector_assignments(
    schedule: pd.DataFrame,
    membership: pd.DataFrame,
    evaluation_start=None,
) -> pd.DataFrame:
    requirements = sector_assignment_requirements(
        schedule, membership, evaluation_start=evaluation_start,
    )
    assignments = requirements[["As_Of_Date", "Asset_ID"]].copy()
    asset_numbers = assignments["Asset_ID"].str.extract(r"(\d+)")[0].astype(int)
    assignments["GICS_Sector_Code"] = asset_numbers.map(
        lambda value: _GICS_SECTORS[value % len(_GICS_SECTORS)][0]
    )
    assignments["Sector"] = asset_numbers.map(
        lambda value: _GICS_SECTORS[value % len(_GICS_SECTORS)][1]
    )
    assignments["Source_Type"] = "Wikipedia"
    assignments["Source_Reference"] = assignments["As_Of_Date"].map(
        lambda date: f"revision-{pd.Timestamp(date):%Y%m%d}"
    )
    assignments["Source_Symbol"] = assignments["Asset_ID"]
    assignments["Resolution_Method"] = "exact_wikipedia_symbol"
    return assignments


def _validated_inputs(
    market: pd.DataFrame,
    membership: pd.DataFrame,
    metadata: pd.DataFrame,
    schedule: pd.DataFrame,
    benchmark: pd.DataFrame,
    actions: LiveCorporateActionBundle | None = None,
    *,
    evaluation_start=None,
):
    if evaluation_start is None:
        evaluation_start = schedule["Sizing_Date"].iloc[0]
    return validate_analysis_inputs(
        market,
        membership,
        metadata,
        _sector_assignments(schedule, membership, evaluation_start),
        schedule,
        benchmark,
        actions,
        evaluation_start=evaluation_start,
        evaluation_end=schedule["Valuation_End"].iloc[-1],
    )


def _synthetic_inputs():
    """Build a custom four-rebalance schedule, separate from the canonical replay."""
    asset_ids = [f"A{i:03d}" for i in range(260)]
    historical = list(pd.date_range("2024-01-31", "2025-12-31", freq="ME"))
    event_dates = [
        "2026-01-30",
        "2026-02-12",
        "2026-02-13",
        "2026-02-27",
        "2026-03-02",
        "2026-03-31",
        "2026-04-01",
        "2026-04-30",
        "2026-05-01",
        "2026-05-04",
        "2026-05-06",
    ]
    dates = sorted(set(historical + [pd.Timestamp(value) for value in event_dates]))
    rows = []
    for date_number, date in enumerate(dates):
        for asset_number, asset_id in enumerate(asset_ids):
            log_price = (
                np.log(75.0 + asset_number * 0.12)
                + 0.003 * date_number
                + 0.018
                * np.sin((date_number + 1.0) * (asset_number + 3.0) / 43.0)
            )
            close = float(np.exp(log_price))
            rows.append({
                "Price_Source": "yahoo",
                "Date": date,
                "Asset_ID": asset_id,
                "Source_Ticker": asset_id,
                "Yahoo_Ticker": asset_id,
                "Open": close * (0.998 + (asset_number % 3) * 0.0005),
                "Close": close,
                "Volume": 800_000.0 + asset_number * 4_000.0,
            })
    market = pd.DataFrame(rows)

    effective_dates = ["2026-01-14", "2026-02-09", "2026-03-23", "2026-04-09"]
    membership = pd.DataFrame(
        [
            {"Effective_Date": effective_date, "Asset_ID": asset_id}
            for effective_date in effective_dates
            for asset_id in asset_ids
        ]
    )
    metadata = pd.DataFrame([
        {
            "Asset_ID": asset_id,
            "Source_Ticker": asset_id,
            "Yahoo_Ticker": asset_id,
        }
        for asset_id in asset_ids
    ])
    schedule = pd.DataFrame(
        [
            ("R1", "2026-01-14", "2026-01-30", "2026-02-13", "Open", "2026-02-13", "Open", "2026-03-02", "Open"),
            ("R2", "2026-02-09", "2026-02-27", "2026-02-27", "Close", "2026-03-02", "Open", "2026-04-01", "Open"),
            ("R3", "2026-03-23", "2026-03-31", "2026-03-31", "Close", "2026-04-01", "Open", "2026-05-04", "Open"),
            ("R4", "2026-04-09", "2026-04-30", "2026-05-01", "Close", "2026-05-04", "Open", "2026-05-06", "Close"),
        ],
        columns=[
            "Rebalance_ID",
            "Membership_Effective_Date",
            "Signal_Cutoff",
            "Sizing_Date",
            "Sizing_Field",
            "Execution_Date",
            "Execution_Field",
            "Valuation_End",
            "Valuation_Field",
        ],
    )
    benchmark = pd.DataFrame({
        "Date": dates,
        "Open": np.linspace(100.0, 112.0, len(dates)),
        "Close": np.linspace(100.1, 112.1, len(dates)),
    })
    # Complete synthetic daily quotes, preserving all original decision prices.
    # This fixture's US session window excludes Presidents' Day and Good Friday.
    sessions = pd.bdate_range("2026-02-13", "2026-05-06").difference(
        pd.to_datetime(["2026-02-16", "2026-04-03"])
    )
    expanded_dates = pd.DatetimeIndex(dates).union(sessions).sort_values()
    expanded = []
    for asset, frame in market.groupby("Asset_ID", sort=False):
        frame = frame.set_index("Date").reindex(expanded_dates)
        frame[["Open", "Close", "Volume"]] = frame[["Open", "Close", "Volume"]].interpolate(method="time")
        frame[["Asset_ID", "Source_Ticker", "Yahoo_Ticker"]] = asset
        frame["Price_Source"] = "yahoo"
        expanded.append(frame.rename_axis("Date").reset_index())
    market = pd.concat(expanded, ignore_index=True)
    benchmark = benchmark.set_index("Date").reindex(expanded_dates).interpolate(method="time").rename_axis("Date").reset_index()
    return _validated_inputs(
        market, membership, metadata, schedule, benchmark
    )


def test_causal_adv_filters_before_grouping_and_preserves_input():
    market = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-01-15", "2026-02-15", "2026-03-15", "2026-04-15",
            "2026-05-01", "2026-04-16", "2026-04-17",
        ]),
        "Asset_ID": ["A"] * 7,
        "Close": [10., 10., 10., 10., 10., np.nan, 10.],
        "Volume": [1., 3., 5., 7., 999., 10., 0.],
    })
    original = market.copy(deep=True)
    actual = live_analysis._causal_trailing_adv(market, pd.Timestamp("2026-04-30"))
    assert actual.to_dict() == {"A": 50.0}
    pd.testing.assert_frame_equal(market, original, check_exact=True)
    with pytest.raises(RuntimeError, match="^No prepared prices through 2025-12-31$"):
        live_analysis._causal_trailing_adv(market, pd.Timestamp("2025-12-31"))
    unusable = market.assign(Volume=0.0)
    pd.testing.assert_series_equal(
        live_analysis._causal_trailing_adv(unusable, pd.Timestamp("2026-04-30")),
        pd.Series(dtype=float),
    )


def _config():
    return SimpleNamespace(
        strategy=_STRATEGY,
        accounting=DEFAULT_CONFIG.accounting,
    )


def _cash_cvr_bundle(asset_id: str, effective_date: str) -> LiveCorporateActionBundle:
    event_id = "EVT-20260220-SYNTHETIC-CASH-CVR"
    common = {
        "Event_ID": event_id,
        "From_Asset_ID": asset_id,
        "Consumes_From_Position": True,
        "Review_Status": "approved",
    }
    return LiveCorporateActionBundle(
        events=pd.DataFrame(
            [
                {
                    "Event_ID": event_id,
                    "Effective_Date": effective_date,
                    "Event_Type": "cash_settlement",
                    "Continuity_Class": "predecessor_extinguished",
                    "Review_Status": "approved",
                }
            ]
        ),
        legs=pd.DataFrame(
            [
                {
                    **common,
                    "Leg_Order": 1,
                    "To_Asset_ID": "",
                    "Leg_Type": "cash",
                    "Quantity_Per_From_Share": 0,
                    "Cash_Per_From_Share": 76,
                    "Currency": "USD",
                    "CVR_Units_Per_From_Share": 0,
                    "CVR_Base_Value_Per_Unit": 0,
                    "CVR_Max_Value_Per_Unit": 0,
                },
                {
                    **common,
                    "Leg_Order": 2,
                    "To_Asset_ID": f"{asset_id}.CVR",
                    "Leg_Type": "cvr",
                    "Quantity_Per_From_Share": 0,
                    "Cash_Per_From_Share": 0,
                    "Currency": "USD",
                    "CVR_Units_Per_From_Share": 1,
                    "CVR_Base_Value_Per_Unit": 0,
                    "CVR_Max_Value_Per_Unit": 3,
                },
            ]
        ),
        sources=pd.DataFrame(
            [
                {
                    "Event_ID": event_id,
                    "Source_URL": url,
                    "Review_Status": "approved",
                }
                for url in (
                    "https://example.test/closing",
                    "https://example.test/cvr",
                )
            ]
        ),
        policy=pd.DataFrame(
            [
                {
                    "Event_ID": event_id,
                    "Scope": "live",
                    "Apply_Accounting": True,
                    "CVR_Base_Value_Per_Unit": 0,
                    "Valuation_As_Of_Date": "",
                    "Valuation_Available_Date": "",
                    "Valuation_Source_ID": "",
                    "Review_Status": "approved",
                    "Notes": "synthetic settlement",
                }
            ]
        ),
    )


def _successor_bundle(
    transitions: list[dict[str, object]],
) -> LiveCorporateActionBundle:
    events = []
    legs = []
    sources = []
    policy = []
    for transition in transitions:
        event_id = str(transition["event_id"])
        events.append({
            "Event_ID": event_id,
            "Effective_Date": str(transition["effective_date"]),
            "Event_Type": str(transition["event_type"]),
            "Continuity_Class": str(transition["continuity_class"]),
            "Review_Status": "approved",
        })
        legs.append({
            "Event_ID": event_id,
            "Leg_Order": 1,
            "From_Asset_ID": str(transition["from_asset_id"]),
            "To_Asset_ID": str(transition["to_asset_id"]),
            "Leg_Type": str(transition["leg_type"]),
            "Quantity_Per_From_Share": transition["quantity"],
            "Cash_Per_From_Share": 0,
            "Currency": "USD",
            "CVR_Units_Per_From_Share": 0,
            "CVR_Base_Value_Per_Unit": 0,
            "CVR_Max_Value_Per_Unit": 0,
            "Consumes_From_Position": True,
            "Review_Status": "approved",
        })
        sources.extend(
            {
                "Event_ID": event_id,
                "Source_URL": f"https://example.test/{event_id}/{number}",
                "Review_Status": "approved",
            }
            for number in (1, 2)
        )
        policy.append({
            "Event_ID": event_id,
            "Scope": "live",
            "Apply_Accounting": True,
            "CVR_Base_Value_Per_Unit": 0,
            "Valuation_As_Of_Date": "",
            "Valuation_Available_Date": "",
            "Valuation_Source_ID": "",
            "Review_Status": "approved",
            "Notes": "synthetic successor transition",
        })
    return LiveCorporateActionBundle(
        events=pd.DataFrame(events),
        legs=pd.DataFrame(legs),
        sources=pd.DataFrame(sources),
        policy=pd.DataFrame(policy),
    )


def _netted_successor_case(one_period, baseline):
    long_holding = baseline.holdings.loc[baseline.holdings["Shares"].gt(0)].iloc[0]
    short_holding = baseline.holdings.loc[baseline.holdings["Shares"].lt(0)].iloc[0]
    held_assets = set(baseline.holdings["Asset_ID"].astype(str))
    successor = next(
        asset_id
        for asset_id in one_period.metadata["Asset_ID"].astype(str)
        if asset_id not in held_assets
    )
    long_shares = int(long_holding["Shares"])
    short_shares = int(short_holding["Shares"])
    actions = _successor_bundle(
        [
            {
                "event_id": "EVT-20260220-LONG-TICKER-CHANGE",
                "effective_date": "2026-02-20",
                "event_type": "identity_continuity",
                "continuity_class": "same_security",
                "from_asset_id": str(long_holding["Asset_ID"]),
                "to_asset_id": successor,
                "leg_type": "relabel",
                "quantity": 1,
            },
            {
                "event_id": "EVT-20260220-SHORT-STOCK-EXCHANGE",
                "effective_date": "2026-02-20",
                "event_type": "stock_exchange",
                "continuity_class": "predecessor_extinguished",
                "from_asset_id": str(short_holding["Asset_ID"]),
                "to_asset_id": successor,
                "leg_type": "stock",
                "quantity": f"{long_shares}/{abs(short_shares)}",
            },
        ]
    )
    return long_holding, short_holding, successor, actions


def test_causal_strategy_respects_custom_schedule_and_whole_shares():
    inputs = _synthetic_inputs()
    result = run_strategy_analysis(inputs, _config())

    assert result.performance.loc[
        result.performance["Series"].eq("Live_Strategy"), "Label"
    ].item() == (
        "canonical monthly replay; "
        "not actual competition-account performance"
    )
    assert result.nav["Rebalance_ID"].tolist() == ["R1", "R2", "R3", "R4"]
    assert result.nav["Period_Start"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-02-13",
        "2026-03-02",
        "2026-04-01",
        "2026-05-04",
    ]
    side_counts = result.holdings.assign(
        Side=np.where(result.holdings["Shares"] > 0, "Long", "Short")
    ).groupby(["Rebalance_ID", "Side"]).size()
    assert (side_counts.xs("Long", level="Side") == 20).all()
    assert (side_counts.xs("Short", level="Side") == 20).all()
    assert np.allclose(result.holdings["Shares"], np.round(result.holdings["Shares"]))
    provenance_columns = [
        "GICS_Sector_Code",
        "Sector",
        "Sector_As_Of_Date",
        "Sector_Source_Type",
        "Sector_Source_Reference",
    ]
    for frame in (result.decisions, result.trades, result.holdings):
        assert set(provenance_columns).issubset(frame.columns)
        assert not frame[provenance_columns].isna().any().any()
    assert pd.to_datetime(result.decisions["Sector_As_Of_Date"]).equals(
        pd.to_datetime(result.decisions["Signal_Cutoff"])
    )
    assert pd.to_datetime(result.trades["Sector_As_Of_Date"]).equals(
        pd.to_datetime(result.trades["Execution_Date"])
    )
    assert pd.to_datetime(result.holdings["Sector_As_Of_Date"]).equals(
        pd.to_datetime(result.holdings["Date"])
    )
    assert (result.trades["Fixed_Fee"] == result.trades["Order_Count"] * 2.0).all()
    assert np.allclose(
        result.nav["Start_NAV"] - result.nav["Post_Trade_NAV"],
        result.nav["Fixed_Fees"] + result.nav["Spread_Cost"],
        atol=1e-8,
    )
    gross = result.holdings.groupby("Rebalance_ID")["Weight"].apply(
        lambda values: values.abs().sum()
    )
    assert (gross <= 2.0 + 1e-12).all()
    assert np.allclose(
        result.nav["Free_Cash"],
        result.nav["Post_Trade_Cash"] - result.nav["Restricted_Short_Proceeds"],
    )
    assert np.allclose(
        result.nav["Post_Trade_NAV"],
        result.nav["Post_Trade_Cash"]
        + result.nav["Post_Trade_Signed_Market_Value"],
    )
    assert np.allclose(
        result.nav["End_NAV"],
        result.nav["End_Cash"] + result.nav["End_Signed_Market_Value"],
    )
    assert np.allclose(
        result.nav["Loan"],
        (-result.nav["Free_Cash"]).clip(lower=0.0),
    )
    assert np.allclose(
        result.nav["Interest"],
        result.nav["Cash_Interest_Credit"]
        - result.nav["Loan_Interest_Charge"],
    )
    assert np.array_equal(
        result.nav["End_NAV"].iloc[:-1].to_numpy(),
        result.nav["Start_NAV"].iloc[1:].to_numpy(),
    )
    assert not {
        "Interest_Before_Trade",
        "Cash_Interest_Before_Trade",
        "Loan_Interest_Before_Trade",
        "Holding_Period_Interest",
        "Holding_Cash_Interest",
        "Holding_Loan_Interest",
    }.intersection(result.nav.columns)
    for row in result.nav.itertuples(index=False):
        (
            expected_cash,
            expected_interest,
            expected_credit,
            expected_charge,
        ) = project_financing_amounts(
            row.Post_Trade_Cash,
            row.Restricted_Short_Proceeds,
            row.Period_Start,
            row.Period_End,
            DEFAULT_CONFIG.accounting,
        )
        assert row.Interest == pytest.approx(expected_interest)
        assert row.Cash_Interest_Credit == pytest.approx(expected_credit)
        assert row.Loan_Interest_Charge == pytest.approx(expected_charge)
        assert row.End_Cash == pytest.approx(expected_cash)
    expected_execution_price = result.trades["Execution_Price"] * np.where(
        result.trades["Trade_Shares"] > 0.0,
        1.0 + result.trades["Spread_Rate"],
        1.0 - result.trades["Spread_Rate"],
    )
    assert np.allclose(
        result.trades["Effective_Execution_Price"],
        expected_execution_price,
    )
    assert np.allclose(
        result.trades["Cash_Effect"],
        -result.trades["Trade_Shares"]
        * result.trades["Effective_Execution_Price"]
        - result.trades["Fixed_Fee"],
    )
    restricted_before = 0.0
    for rebalance_id, nav_row in result.nav.set_index("Rebalance_ID").iterrows():
        restricted_change = result.trades.loc[
            result.trades["Rebalance_ID"].eq(rebalance_id),
            "Restricted_Proceeds_Change",
        ].sum()
        assert nav_row["Restricted_Short_Proceeds"] == pytest.approx(
            restricted_before + restricted_change
        )
        restricted_before = float(nav_row["End_Restricted_Short_Proceeds"])
    assert (result.nav["Restricted_Short_Proceeds"] >= 0.0).all()
    assert (result.nav[["Fixed_Fees", "Spread_Cost"]] > 0.0).all().all()


def test_schedule_requires_exact_shared_boundary_fields_and_sizing_window():
    inputs = _synthetic_inputs()
    mismatched_field = inputs.schedule.copy()
    mismatched_field.loc[0, "Valuation_Field"] = "Close"
    with pytest.raises(ValueError, match="same valuation and execution"):
        _validated_inputs(
            inputs.market_daily,
            inputs.membership,
            inputs.metadata,
            mismatched_field,
            inputs.benchmark_daily,
        )

    early_sizing = inputs.schedule.copy()
    early_sizing.loc[1, "Signal_Cutoff"] = pd.Timestamp("2026-02-10")
    early_sizing.loc[1, "Sizing_Date"] = pd.Timestamp("2026-02-12")
    with pytest.raises(ValueError, match="sizing checkpoint"):
        _validated_inputs(
            inputs.market_daily,
            inputs.membership,
            inputs.metadata,
            early_sizing,
            inputs.benchmark_daily,
        )


def test_benchmark_price_recovery_message_is_exact():
    benchmark = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-03-30")],
            "Open": [100.0],
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        live_analysis._benchmark_price(
            benchmark,
            pd.Timestamp("2026-03-31"),
            "Open",
        )

    assert str(exc_info.value) == (
        "Prepared benchmark is missing Open on 2026-03-31. "
        f"Run `{DOWNLOAD_BENCHMARK_COMMAND}` then `{PREPARE_COMMAND}`."
    )


def test_incomplete_prepared_signal_recovery_message_is_exact():
    inputs = _synthetic_inputs()
    asset_ids = inputs.metadata["Asset_ID"].iloc[:39]
    limited = _validated_inputs(
        inputs.market_daily.loc[inputs.market_daily["Asset_ID"].isin(asset_ids)],
        inputs.membership.loc[inputs.membership["Asset_ID"].isin(asset_ids)],
        inputs.metadata.loc[inputs.metadata["Asset_ID"].isin(asset_ids)],
        inputs.schedule.iloc[:1].copy(),
        inputs.benchmark_daily,
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_strategy_analysis(limited, _config())

    assert "momentum signal/sizing and eligible holdings unready for R1" in str(exc_info.value)
    assert DOWNLOAD_PRICES_COMMAND in str(exc_info.value)
    assert PREPARE_COMMAND in str(exc_info.value)


def test_custom_spread_sensitivity_reuses_matching_nine_bps_base(monkeypatch):
    strategy = _STRATEGY
    custom_accounting = replace(
        DEFAULT_CONFIG.accounting,
        transaction_costs=replace(
            DEFAULT_CONFIG.accounting.transaction_costs,
            liquidity_bps=9.0,
        ),
    )
    config = replace(
        DEFAULT_CONFIG,
        strategy=strategy,
        paths=DEFAULT_CONFIG.paths.for_strategy(strategy.strategy_id),
        accounting=custom_accounting,
    )
    inputs = SimpleNamespace(price_basis=price_basis_spec("live"))
    called_coefficients: list[float] = []
    history = object()
    history_calls = []

    def prepare_history(received_inputs, received_strategy):
        assert received_inputs is inputs and received_strategy is strategy
        history_calls.append(True)
        return history

    monkeypatch.setattr(live_analysis, "prepare_live_research_history", prepare_history)

    monkeypatch.setattr(
        live_analysis.analysis_data,
        "load_analysis_inputs",
        lambda _paths, **kwargs: inputs,
    )

    def fake_analysis(_inputs, case_config, *, research_history=None):
        assert research_history is history
        accounting = case_config.accounting
        coefficient = accounting.transaction_costs.liquidity_bps
        called_coefficients.append(coefficient)
        assumptions = build_simulation_assumptions(
            accounting,
            _inputs.price_basis,
        )
        performance = pd.DataFrame([
            {
                "Series": "Live_Strategy",
                "Cumulative_Return": coefficient / 1_000.0,
                "Sharpe_Ratio": 1.0,
                "Price_Basis": _inputs.price_basis.price_basis_id,
                "Simulation_Fingerprint": assumptions[
                    "simulation_fingerprint"
                ],
            }
        ])
        return LiveAnalysisResult(
            nav=pd.DataFrame(),
            holdings=pd.DataFrame(),
            trades=pd.DataFrame(),
            decisions=pd.DataFrame(),
            performance=performance,
            corporate_actions=pd.DataFrame(),
        )

    monkeypatch.setattr(live_analysis, "run_strategy_analysis", fake_analysis)
    result = live_analysis.run_strategy(config)

    sensitivity = result.spread_sensitivity.set_index("Sensitivity_Case")
    assert sensitivity["Liquidity_Coefficient_Bps"].to_dict() == {
        "low": 4.5,
        "base": 9.0,
        "high": 13.5,
    }
    base_fingerprint = result.performance.loc[
        result.performance["Series"].eq("Live_Strategy"),
        "Simulation_Fingerprint",
    ].item()
    assert sensitivity.at["base", "Simulation_Fingerprint"] == base_fingerprint
    assert called_coefficients == [9.0, 4.5, 13.5]
    assert history_calls == [True]


def test_explicit_momentum_runs_in_live_analysis_with_dynamic_audits():
    inputs = _synthetic_inputs()
    strategy = momentum_test_strategy()
    result = run_strategy_analysis(
        inputs,
        SimpleNamespace(
            strategy=strategy,
            accounting=DEFAULT_CONFIG.accounting,
        ),
    )

    assert list(result.decisions.columns) == live_decision_audit_columns(strategy)
    assert list(result.trades.columns) == live_trade_audit_columns(strategy)
    assert result.decisions["Strategy_ID"].eq("momentum").all()
    assert result.decisions["Strategy_Version"].eq(strategy.strategy_version).all()
    side_counts = result.holdings.groupby("Rebalance_ID")["Shares"].agg(
        Long=lambda values: int((values > 0).sum()),
        Short=lambda values: int((values < 0).sum()),
    )
    assert side_counts.eq(10).all().all()
    selected = result.decisions.loc[
        result.decisions["Final_Selected_Side"].ne("")
    ]
    assert selected.groupby(
        ["Rebalance_ID", "Final_Selected_Side"]
    ).size().eq(10).all()
    assert selected.loc[
        selected["Final_Selected_Side"].eq("Long"), "Final_Target_Weight"
    ].eq(0.05).all()
    assert selected.loc[
        selected["Final_Selected_Side"].eq("Short"), "Final_Target_Weight"
    ].eq(-0.05).all()


def test_post_cutoff_and_sizing_prices_cannot_change_first_signal_decision():
    baseline_inputs = _synthetic_inputs()
    baseline = run_strategy_analysis(baseline_inputs, _config())

    changed_market = baseline_inputs.market_daily.copy()
    changed_market.loc[
        changed_market["Date"].eq(pd.Timestamp("2026-02-12")), "Close"
    ] *= 1.25
    changed_market.loc[
        changed_market["Date"].eq(pd.Timestamp("2026-02-27")), "Close"
    ] *= 0.75
    changed_inputs = _validated_inputs(
        changed_market,
        baseline_inputs.membership,
        baseline_inputs.metadata,
        baseline_inputs.schedule,
        baseline_inputs.benchmark_daily,
    )
    changed = run_strategy_analysis(changed_inputs, _config())

    columns = [
        "Asset_ID",
        *_STRATEGY.signal_columns,
        "Strategy_Score",
        "Strategy_Rank",
        "Signal_Selected_Side",
        "Signal_Raw_Target_Weight",
        "Final_Selected_Side",
        "Final_Target_Weight",
    ]
    expected = baseline.decisions.loc[
        baseline.decisions["Rebalance_ID"].eq("R1"), columns
    ].reset_index(drop=True)
    actual = changed.decisions.loc[
        changed.decisions["Rebalance_ID"].eq("R1"), columns
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_actual_history_controls_signal_eligibility_without_row_count_filter():
    inputs = _synthetic_inputs()
    shortened = inputs.market_daily.loc[
        ~(
            inputs.market_daily["Asset_ID"].eq("A000")
            & inputs.market_daily["Date"].lt(pd.Timestamp("2025-11-01"))
        )
    ]
    changed_inputs = _validated_inputs(
        shortened,
        inputs.membership,
        inputs.metadata,
        inputs.schedule,
        inputs.benchmark_daily,
    )
    result = run_strategy_analysis(changed_inputs, _config())
    audit = result.decisions.loc[
        result.decisions["Rebalance_ID"].eq("R1")
        & result.decisions["Asset_ID"].eq("A000")
    ].iloc[0]

    assert not audit["Strategy_Eligible"]
    assert audit["Strategy_Exclusion_Reason"] == "insufficient_signal_history"
    assert np.isnan(audit["Momentum"])


@pytest.mark.parametrize("position_sign", [1, -1], ids=["long", "short"])
def test_strategy_uses_cash_settlement_instead_of_post_event_price(position_sign):
    inputs = _synthetic_inputs()
    one_period = _validated_inputs(
        inputs.market_daily,
        inputs.membership,
        inputs.metadata,
        inputs.schedule.iloc[:1].copy(),
        inputs.benchmark_daily,
    )
    baseline = run_strategy_analysis(one_period, _config())
    holding = baseline.holdings.loc[
        np.sign(baseline.holdings["Shares"]).eq(position_sign)
    ].iloc[0]
    asset_id = str(holding["Asset_ID"])
    signed_shares = int(holding["Shares"])
    market = one_period.market_daily.loc[
        ~(
            one_period.market_daily["Asset_ID"].eq(asset_id)
            & one_period.market_daily["Date"].eq(pd.Timestamp("2026-03-02"))
        )
    ].copy()
    actions = _cash_cvr_bundle(asset_id, "2026-02-20")
    settled_inputs = _validated_inputs(
        market,
        one_period.membership,
        one_period.metadata,
        one_period.schedule,
        one_period.benchmark_daily,
        actions,
    )

    settled = run_strategy_analysis(settled_inputs, _config())

    audit = settled.corporate_actions.iloc[0]
    assert audit["Event_ID"] == "EVT-20260220-SYNTHETIC-CASH-CVR"
    assert audit["Asset_ID"] == asset_id
    assert audit["Shares_Before"] == signed_shares
    assert audit["Cash_Effect"] == pytest.approx(signed_shares * 76.0)
    assert np.isfinite(audit["Interest_Effect"])
    assert audit["Interest_Effect"] != 0.0
    assert audit["CVR_Units"] == pytest.approx(signed_shares)
    assert audit["CVR_Base_Value"] == 0.0
    assert audit["CVR_Max_Value"] == pytest.approx(signed_shares * 3.0)
    assert audit["Fixed_Fee"] == 0.0
    assert audit["Spread_Cost"] == 0.0
    assert not settled.corporate_actions.duplicated(
        ["Event_ID", "Asset_ID"]
    ).any()
    assert set(json.loads(audit["Source_URLs_JSON"])) == {
        "https://example.test/closing",
        "https://example.test/cvr",
    }
    settled_holding = settled.holdings.loc[
        settled.holdings["Asset_ID"].eq(asset_id)
    ].iloc[0]
    assert settled_holding["End_Value_Source"] == "corporate_action_settlement"
    assert settled_holding["End_Price"] == 76.0

    # The daily event mark removes the predecessor and includes settlement cash
    # immediately; it must not keep valuing the extinguished shares at a quote.
    event_date = pd.Timestamp("2026-02-20")
    period = settled.nav.iloc[0]
    days = (event_date - period.Period_Start).days
    free_cash = period.Post_Trade_Cash - period.Restricted_Short_Proceeds
    rate = .02 if free_cash > 0 else .08
    expected_cash = (
        period.Post_Trade_Cash + free_cash * ((1 + rate / 365) ** days - 1)
        + signed_shares * 76.
    )
    remaining = settled.holdings.loc[settled.holdings.Asset_ID.ne(asset_id)].set_index("Asset_ID")
    closes = market.loc[market.Date.eq(event_date)].set_index("Asset_ID").Close
    mark = settled.daily_nav.loc[
        settled.daily_nav.Date.eq(event_date) & settled.daily_nav.Observation.eq("session_close")
    ].iloc[0]
    assert mark.Cash == pytest.approx(expected_cash, abs=1e-7)
    assert mark.NAV == pytest.approx(expected_cash + (remaining.Shares * closes.reindex(remaining.index)).sum(), abs=1e-7)


def test_strategy_values_ticker_change_and_stock_exchange_when_successor_nets_zero():
    inputs = _synthetic_inputs()
    one_period = _validated_inputs(
        inputs.market_daily,
        inputs.membership,
        inputs.metadata,
        inputs.schedule.iloc[:1].copy(),
        inputs.benchmark_daily,
    )
    baseline = run_strategy_analysis(one_period, _config())
    long_holding, short_holding, successor, actions = _netted_successor_case(
        one_period, baseline
    )
    settled_inputs = _validated_inputs(
        one_period.market_daily,
        one_period.membership,
        one_period.metadata,
        one_period.schedule,
        one_period.benchmark_daily,
        actions,
    )

    settled = run_strategy_analysis(settled_inputs, _config())

    end_date = pd.Timestamp(one_period.schedule["Valuation_End"].iloc[0])
    successor_price = float(
        one_period.market_daily.loc[
            one_period.market_daily["Date"].eq(end_date)
            & one_period.market_daily["Asset_ID"].eq(successor),
            "Open",
        ].iloc[0]
    )
    long_asset = str(long_holding["Asset_ID"])
    short_asset = str(short_holding["Asset_ID"])
    by_asset = settled.holdings.set_index("Asset_ID")
    assert by_asset.at[long_asset, "End_Value_Source"] == (
        "corporate_action_settlement"
    )
    assert by_asset.at[short_asset, "End_Value_Source"] == (
        "corporate_action_settlement"
    )
    assert by_asset.at[long_asset, "End_Price"] == pytest.approx(successor_price)
    short_ratio = Fraction(
        int(long_holding["Shares"]), abs(int(short_holding["Shares"]))
    )
    assert by_asset.at[short_asset, "End_Price"] == pytest.approx(
        successor_price * float(short_ratio)
    )

    successor_units = sum(
        (
            Fraction(json.loads(value)[successor])
            for value in settled.corporate_actions["Successor_Units_JSON"]
            if successor in json.loads(value)
        ),
        Fraction(0),
    )
    assert successor_units == 0
    assert settled.nav.iloc[0]["End_Position_Count"] == (
        baseline.nav.iloc[0]["End_Position_Count"] - 2
    )
    assert set(settled.corporate_actions["Event_Type"]) == {
        "identity_continuity",
        "stock_exchange",
    }


def test_strategy_values_chained_events_from_final_successor_without_audit_rows(
    monkeypatch,
):
    inputs = _synthetic_inputs()
    one_period = _validated_inputs(
        inputs.market_daily,
        inputs.membership,
        inputs.metadata,
        inputs.schedule.iloc[:1].copy(),
        inputs.benchmark_daily,
    )
    baseline = run_strategy_analysis(one_period, _config())
    predecessor = str(baseline.holdings.iloc[0]["Asset_ID"])
    held_assets = set(baseline.holdings["Asset_ID"].astype(str))
    intermediate, final_successor = [
        asset_id
        for asset_id in one_period.metadata["Asset_ID"].astype(str)
        if asset_id not in held_assets
    ][:2]
    actions = _successor_bundle(
        [
            {
                "event_id": "EVT-20260220-A-TO-B",
                "effective_date": "2026-02-20",
                "event_type": "stock_exchange",
                "continuity_class": "predecessor_extinguished",
                "from_asset_id": predecessor,
                "to_asset_id": intermediate,
                "leg_type": "stock",
                "quantity": "3/2",
            },
            {
                "event_id": "EVT-20260221-B-TO-C",
                "effective_date": "2026-02-21",
                "event_type": "stock_exchange",
                "continuity_class": "predecessor_extinguished",
                "from_asset_id": intermediate,
                "to_asset_id": final_successor,
                "leg_type": "stock",
                "quantity": "2/3",
            },
        ]
    )
    end_date = pd.Timestamp(one_period.schedule["Valuation_End"].iloc[0])
    market = one_period.market_daily.loc[
        ~(
            one_period.market_daily["Date"].eq(end_date)
            & one_period.market_daily["Asset_ID"].eq(intermediate)
        )
    ].copy()
    settled_inputs = _validated_inputs(
        market,
        one_period.membership,
        one_period.metadata,
        one_period.schedule,
        one_period.benchmark_daily,
        actions,
    )
    original_audit_rows = live_analysis._corporate_action_audit_rows

    def reporting_rows(*args, **kwargs):
        rows = original_audit_rows(*args, **kwargs)
        for row in rows:
            row["Successor_Units_JSON"] = "report-only-sentinel"
        return rows

    monkeypatch.setattr(
        live_analysis, "_corporate_action_audit_rows", reporting_rows
    )

    settled = run_strategy_analysis(settled_inputs, _config())

    final_price = float(
        market.loc[
            market["Date"].eq(end_date)
            & market["Asset_ID"].eq(final_successor),
            "Open",
        ].iloc[0]
    )
    settled_holding = settled.holdings.loc[
        settled.holdings["Asset_ID"].eq(predecessor)
    ].iloc[0]
    assert settled_holding["End_Value_Source"] == "corporate_action_settlement"
    assert settled_holding["End_Price"] == pytest.approx(final_price)
    assert not (
        market["Date"].eq(end_date)
        & market["Asset_ID"].eq(intermediate)
    ).any()
    assert set(settled.corporate_actions["Successor_Units_JSON"]) == {
        "report-only-sentinel"
    }


@pytest.mark.parametrize("price_state", ["missing", "invalid"])
def test_strategy_rejects_unavailable_netted_successor_price(
    price_state,
    monkeypatch,
):
    inputs = _synthetic_inputs()
    one_period = _validated_inputs(
        inputs.market_daily,
        inputs.membership,
        inputs.metadata,
        inputs.schedule.iloc[:1].copy(),
        inputs.benchmark_daily,
    )
    baseline = run_strategy_analysis(one_period, _config())
    _, _, successor, actions = _netted_successor_case(one_period, baseline)
    end_date = pd.Timestamp(one_period.schedule["Valuation_End"].iloc[0])
    market = one_period.market_daily.copy()
    successor_end = market["Date"].eq(end_date) & market["Asset_ID"].eq(successor)
    if price_state == "missing":
        market = market.loc[~successor_end].copy()
    else:
        market.loc[successor_end, "Open"] = np.nan
    settled_inputs = _validated_inputs(
        market,
        one_period.membership,
        one_period.metadata,
        one_period.schedule,
        one_period.benchmark_daily,
        actions,
    )
    monkeypatch.setattr(
        live_analysis,
        "live_interval_eligible_assets",
        lambda _market, member_asset_ids, **_kwargs: set(member_asset_ids),
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_strategy_analysis(settled_inputs, _config())

    message = str(exc_info.value)
    assert successor in message
    assert "corporate-action settlement for predecessor" in message
    assert "Missing prepared open prices" in message
    assert "python -m data_acquisition.acquire live prices" in message
    assert "python -m live.prepare all" in message


def test_brinson_uses_cash_settlement_instead_of_post_event_price():
    inputs = _synthetic_inputs()
    one_period = inputs.schedule.iloc[:1].copy()
    asset_id = str(inputs.metadata["Asset_ID"].iloc[0])
    end_date = pd.Timestamp(one_period["Valuation_End"].iloc[0])
    market = inputs.market_daily.loc[
        ~(
            inputs.market_daily["Asset_ID"].eq(asset_id)
            & inputs.market_daily["Date"].eq(end_date)
        )
    ].copy()
    actions = _cash_cvr_bundle(asset_id, "2026-02-20")
    settled_inputs = _validated_inputs(
        market,
        inputs.membership,
        inputs.metadata,
        one_period,
        inputs.benchmark_daily,
        actions,
    )
    asset_ids = settled_inputs.metadata["Asset_ID"].tolist()
    shares = pd.DataFrame(
        {
            "Date": pd.Timestamp("2026-01-01"),
            "Asset_ID": asset_ids,
            "Shares_Outstanding": 100_000_000.0,
        }
    )

    benchmark_sector, benchmark_audit = build_live_benchmark_sector_series(
        settled_inputs, shares
    )

    start_date = pd.Timestamp(one_period["Execution_Date"].iloc[0])
    start_prices = settled_inputs.market_daily.loc[
        settled_inputs.market_daily["Date"].eq(start_date)
    ].set_index("Asset_ID")["Open"].reindex(asset_ids)
    end_values = settled_inputs.market_daily.loc[
        settled_inputs.market_daily["Date"].eq(end_date)
    ].set_index("Asset_ID")["Open"].reindex(asset_ids)
    end_values.loc[asset_id] = 76.0
    weights = start_prices / start_prices.sum()
    expected_return = float((weights * (end_values / start_prices - 1.0)).sum())
    assert benchmark_audit.iloc[0]["Reconstructed_Benchmark_Return"] == pytest.approx(
        expected_return
    )

    sector_rows = settled_inputs.sector_assignments.loc[
        settled_inputs.sector_assignments["As_Of_Date"].eq(start_date)
    ].set_index("Asset_ID")
    sector_code = sector_rows.at[asset_id, "GICS_Sector_Code"]
    sector_assets = sector_rows.index[
        sector_rows["GICS_Sector_Code"].eq(sector_code)
    ]
    sector_weights = start_prices.reindex(sector_assets)
    expected_sector_return = float(
        np.average(
            (end_values / start_prices - 1.0).reindex(sector_assets),
            weights=sector_weights,
        )
    )
    actual_sector_return = benchmark_sector.loc[
        benchmark_sector["GICS_Sector_Code"].eq(sector_code),
        "Benchmark_Return",
    ].iloc[0]
    assert actual_sector_return == pytest.approx(expected_sector_return)


def test_brinson_still_rejects_unexplained_missing_end_price():
    inputs = _synthetic_inputs()
    one_period = inputs.schedule.iloc[:1].copy()
    asset_id = str(inputs.metadata["Asset_ID"].iloc[0])
    end_date = pd.Timestamp(one_period["Valuation_End"].iloc[0])
    market = inputs.market_daily.loc[
        ~(
            inputs.market_daily["Asset_ID"].eq(asset_id)
            & inputs.market_daily["Date"].eq(end_date)
        )
    ].copy()
    missing_inputs = _validated_inputs(
        market,
        inputs.membership,
        inputs.metadata,
        one_period,
        inputs.benchmark_daily,
    )
    asset_ids = missing_inputs.metadata["Asset_ID"].tolist()
    shares = pd.DataFrame(
        {
            "Date": pd.Timestamp("2026-01-01"),
            "Asset_ID": asset_ids,
            "Shares_Outstanding": 100_000_000.0,
        }
    )

    with pytest.raises(RuntimeError, match=asset_id):
        build_live_benchmark_sector_series(missing_inputs, shares)


def test_brinson_event_adjusted_price_recovery_message_is_exact(monkeypatch):
    inputs = SimpleNamespace(
        corporate_actions=SimpleNamespace(
            events=pd.DataFrame(),
            legs=pd.DataFrame(),
            sources=pd.DataFrame(),
        ),
        market_daily=pd.DataFrame(),
    )
    monkeypatch.setattr(
        live_brinson,
        "has_interval_event",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        live_brinson,
        "value_one_share_through_events",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError) as exc_info:
        live_brinson._benchmark_event_values(
            inputs,
            ["ZZZ", "AAA"],
            pd.Timestamp("2026-02-13"),
            pd.Timestamp("2026-03-31"),
            "Open",
            context="R9 benchmark",
        )

    assert str(exc_info.value) == (
        "Missing prepared event-adjusted open value for R9 benchmark on "
        f"2026-03-31: ['ZZZ']. Run `{DOWNLOAD_PRICES_COMMAND}` then "
        f"`{PREPARE_COMMAND}`."
    )


def test_execution_membership_closes_departed_name_and_audits_refill():
    baseline_inputs = _synthetic_inputs()
    baseline = run_strategy_analysis(baseline_inputs, _config())
    first_selected = baseline.decisions.loc[
        baseline.decisions["Rebalance_ID"].eq("R1")
        & baseline.decisions["Final_Selected_Side"].ne(""),
        "Asset_ID",
    ].iloc[0]
    changed_membership = baseline_inputs.membership.loc[
        ~(
            baseline_inputs.membership["Effective_Date"].ge(pd.Timestamp("2026-02-09"))
            & baseline_inputs.membership["Asset_ID"].eq(first_selected)
        )
    ]
    changed_inputs = _validated_inputs(
        baseline_inputs.market_daily,
        changed_membership,
        baseline_inputs.metadata,
        baseline_inputs.schedule,
        baseline_inputs.benchmark_daily,
    )
    changed = run_strategy_analysis(changed_inputs, _config())

    departed = changed.decisions.loc[
        changed.decisions["Rebalance_ID"].eq("R1")
        & changed.decisions["Asset_ID"].eq(first_selected)
    ].iloc[0]
    assert not departed["Execution_Eligible"]
    assert departed["Final_Selected_Side"] == ""
    assert departed["Execution_Adjustment"] == "departed_before_execution"
    first_holdings = changed.holdings.loc[changed.holdings["Rebalance_ID"].eq("R1")]
    assert first_selected not in set(first_holdings["Asset_ID"])
    assert (first_holdings["Shares"] > 0).sum() == 20
    assert (first_holdings["Shares"] < 0).sum() == 20
    assert changed.decisions.loc[
        changed.decisions["Rebalance_ID"].eq("R1"), "Execution_Adjustment"
    ].str.contains("refill").any()


def test_carried_departed_name_exits_with_its_latest_prior_sector():
    baseline_inputs = _synthetic_inputs()
    baseline = run_strategy_analysis(baseline_inputs, _config())
    carried_asset = baseline.holdings.loc[
        baseline.holdings["Rebalance_ID"].eq("R2"), "Asset_ID"
    ].iloc[0]
    changed_membership = baseline_inputs.membership.loc[
        ~(
            baseline_inputs.membership["Effective_Date"].ge(
                pd.Timestamp("2026-03-23")
            )
            & baseline_inputs.membership["Asset_ID"].eq(carried_asset)
        )
    ]
    changed_inputs = _validated_inputs(
        baseline_inputs.market_daily,
        changed_membership,
        baseline_inputs.metadata,
        baseline_inputs.schedule,
        baseline_inputs.benchmark_daily,
    )

    changed = run_strategy_analysis(changed_inputs, _config())

    forced = changed.trades.loc[
        changed.trades["Rebalance_ID"].eq("R3")
        & changed.trades["Asset_ID"].eq(carried_asset)
    ]
    assert len(forced) == 1
    row = forced.iloc[0]
    assert not row["Execution_Eligible"]
    assert row["Current_Shares"] != 0.0
    assert row["Applied_Target_Shares"] == 0.0
    assert row["Sector_As_Of_Date"] == pd.Timestamp("2026-03-02")
    assert row["Sector_As_Of_Date"] < row["Execution_Date"]
    assert pd.isna(row["Strategy_Score"])
    assert row[list(_STRATEGY.signal_columns)].isna().all()
    exact_active = changed.trades.loc[changed.trades["Execution_Eligible"]]
    assert pd.to_datetime(exact_active["Sector_As_Of_Date"]).equals(
        pd.to_datetime(exact_active["Execution_Date"])
    )
