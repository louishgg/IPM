"""Offline canonical calendar, continuous account and attribution contracts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from data_acquisition.contracts import write_readiness
from live import acquisition_execution, preparation_builders
from live.analysis import run_strategy_analysis
from live.analysis_data import validate_account_nav
from live.brinson_attribution import run_live_brinson
from live.config import DEFAULT_CONFIG
from live.price_coverage import PRICE_REQUIREMENT_SET, build_price_requirements
from live.preparation_artifacts import PREPARE_COMMAND
from live.strategy_universe import DOWNLOAD_PRICES_COMMAND, DOWNLOAD_SHARES_COMMAND, decision_schedule
from portfolio_core.accounting_ledger import LedgerState
from portfolio_core.portfolio_lifecycle import advance_ledger
from tests.test_live_analysis import _synthetic_inputs, _validated_inputs, _config, _cash_cvr_bundle


@pytest.fixture(scope="module")
def canonical():
    source = _synthetic_inputs()
    inputs = _validated_inputs(
        source.market_daily, source.membership, source.metadata,
        decision_schedule(), source.benchmark_daily,
        evaluation_start=DEFAULT_CONFIG.market.competition_start,
    )
    return inputs, run_strategy_analysis(inputs, _config())


def test_canonical_cash_sizing_execution_and_full_window(canonical):
    inputs, result = canonical
    nav = validate_account_nav(result.nav, periods=inputs.evaluation_periods)
    assert result.trades["Execution_Date"].drop_duplicates().tolist() == list(
        pd.to_datetime(["2026-03-02", "2026-04-01", "2026-05-01"])
    )
    assert inputs.schedule["Sizing_Date"].tolist() == list(
        pd.to_datetime(["2026-02-27", "2026-03-31", "2026-04-30"])
    )
    assert (inputs.schedule["Sizing_Field"] == "Close").all()
    assert (inputs.schedule["Execution_Field"] == "Open").all()
    assert nav["End_Field"].tolist() == ["Open", "Open", "Open", "Close"]
    assert nav["Period_Type"].tolist() == ["cash", "invested", "invested", "invested"]
    assert (nav["Period_End"] - nav["Period_Start"]).dt.days.tolist() == [17, 30, 30, 5]
    cash = nav.iloc[0]
    assert cash.Rebalance_ID == ""
    assert cash.Start_NAV == 1_000_000
    assert cash.End_NAV == pytest.approx(1_000_931.9152929792, abs=1e-7)
    assert cash.Interest == pytest.approx(931.9152929792, abs=1e-7)
    assert cash.Period_Return == pytest.approx((1 + .02 / 365) ** 17 - 1)
    assert cash[["Position_Count", "Order_Count", "Fixed_Fees", "Spread_Cost"]].eq(0).all()
    assert result.holdings["Rebalance_ID"].unique().tolist() == ["R1", "R2", "R3"]
    assert nav.iloc[1].Sizing_NAV == pytest.approx(1_000_767.3965695585, abs=1e-7)
    assert nav.iloc[1].Start_NAV == pytest.approx(cash.End_NAV, abs=1e-7)
    np.testing.assert_array_equal(nav.End_NAV.iloc[:-1], nav.Start_NAV.iloc[1:])
    first_orders = result.trades.loc[result.trades["Rebalance_ID"].eq("R1")]
    np.testing.assert_allclose(
        first_orders["Target_Position_Value_At_Sizing"],
        first_orders["Target_Weight"] * nav.iloc[1].Sizing_NAV,
    )

    performance = result.performance.set_index("Series")
    p = performance.loc["Live_Strategy"]
    assert p.Evaluation_Start == pd.Timestamp("2026-02-13")
    assert p.Evaluation_End == pd.Timestamp("2026-05-06")
    assert p.Cumulative_Return == pytest.approx(nav.End_NAV.iloc[-1] / 1_000_000 - 1)
    assert p.Annualized_Return == pytest.approx((1 + p.Cumulative_Return) ** (365 / 82) - 1)
    daily_returns = result.daily_nav.loc[result.daily_nav.Include_In_Risk_Metrics, "Portfolio_Return"]
    assert p.Annualized_Volatility == pytest.approx(daily_returns.std(ddof=1) * np.sqrt(252))
    assert p.Risk_Observation_Count == 56
    benchmark = inputs.benchmark_daily.set_index("Date")
    expected_benchmark = benchmark.at[pd.Timestamp("2026-05-06"), "Close"] / benchmark.at[pd.Timestamp("2026-02-13"), "Open"] - 1
    assert performance.loc["S&P 500 Total Return", "Cumulative_Return"] == pytest.approx(expected_benchmark)


def test_split_financing_and_same_date_actions_are_not_repeated():
    accounting = DEFAULT_CONFIG.accounting
    initial = LedgerState.initial(accounting, state_date=pd.Timestamp("2026-02-13"))
    actions = _cash_cvr_bundle("A000", "2026-02-27")

    def advance(state, end):
        return advance_ledger(
            state, pd.Timestamp(end), actions.events, actions.legs, actions.sources,
            accounting_config=accounting,
        )

    sizing = advance(initial, "2026-02-27")
    repeat = advance(sizing.state, "2026-02-27")
    execution = advance(repeat.state, "2026-03-02")
    uninterrupted = advance(initial, "2026-03-02")
    assert repeat.state.cash == sizing.state.cash
    assert repeat.interest_amount == 0
    assert repeat.corporate_actions.audit.empty
    assert execution.state.cash == pytest.approx(uninterrupted.state.cash, abs=1e-7)
    assert sizing.interest_amount + execution.interest_amount == pytest.approx(
        uninterrupted.interest_amount, abs=1e-7,
    )
    with pytest.raises(ValueError, match="precede"):
        advance(execution.state, "2026-02-27")


def test_cash_and_invested_attribution_reconcile_interval_and_compounded_returns(canonical):
    inputs, strategy = canonical
    shares = pd.DataFrame({
        "Date": pd.Timestamp("2026-02-01"),
        "Asset_ID": inputs.metadata.Asset_ID,
        "Shares_Outstanding": np.linspace(50_000_000., 300_000_000., len(inputs.metadata)),
    })
    result = run_live_brinson(inputs, shares, strategy.holdings, strategy.nav)
    assert result.benchmark_audit.Dropped_Constituent_Count.eq(0).all()
    assert result.period_attribution.loc[
        result.period_attribution.Side.eq("Combined"), "Scaled_Residual"
    ].abs().max() < 1e-10
    missing_asset = inputs.metadata.Asset_ID.iloc[0]
    future_only = shares.copy()
    future_only.loc[future_only.Asset_ID.eq(missing_asset), "Date"] = pd.Timestamp("2026-05-07")
    for invalid in (shares.loc[shares.Asset_ID.ne(missing_asset)], future_only):
        with pytest.raises(RuntimeError) as error:
            run_live_brinson(inputs, invalid, strategy.holdings, strategy.nav)
        assert str(error.value) == (
            "Prepared shares have no observation on or before 2026-02-13 for "
            f"active constituents [{missing_asset!r}]. Run "
            f"`{DOWNLOAD_SHARES_COMMAND}` then `{PREPARE_COMMAND}`."
        )
    account = result.account_reconciliation
    assert len(result.benchmark_audit) == len(account) == 4
    assert result.period_attribution.loc[result.period_attribution.Side.eq("Combined")].shape[0] == 3
    cash = account.iloc[0]
    assert cash.Allocation_Effect == cash.Selection_Effect == cash.Interaction_Effect == 0
    assert cash.Linked_Interaction_Effect == 0
    assert account.Attribution_Method.eq("Brinson-Fachler").all()
    assert cash.Net_Exposure_Effect == pytest.approx(-cash.Reconstructed_Benchmark_Return)
    assert cash.Cash_Interest_Effect == pytest.approx(strategy.nav.iloc[0].Period_Return)
    np.testing.assert_allclose(account.Total_Effect, account.Account_Excess_Return, atol=1e-12)
    excess = np.prod(1 + strategy.nav.Period_Return) - np.prod(1 + strategy.nav.Benchmark_Return)
    assert account.Linked_Total_Effect.sum() == pytest.approx(excess, abs=1e-12)
    combined = result.period_attribution.loc[result.period_attribution.Side.eq("Combined")].set_index("Date")
    for index, row in account.iterrows():
        expected_link = (np.prod(1 + strategy.nav.Period_Return.iloc[:index])
                         * np.prod(1 + strategy.nav.Benchmark_Return.iloc[index + 1:]))
        assert row.Link_Factor == pytest.approx(expected_link, abs=1e-14)
        for effect in ("Allocation", "Selection", "Interaction"):
            contribution = row[f"{effect}_Effect"]
            assert row[f"Linked_{effect}_Effect"] == pytest.approx(contribution * expected_link, abs=1e-14)
            if row.Period_Type == "invested":
                expected = (combined.at[row.Period_Start, f"Scaled_{effect}_Effect"]
                            * row.Post_Trade_NAV / row.Start_NAV)
                assert contribution == pytest.approx(expected, abs=1e-14)
    linked_effects = [column for column in account if column.startswith("Linked_")
                     and column.endswith("_Effect") and column != "Linked_Total_Effect"]
    assert len(linked_effects) == 9
    assert account[linked_effects].sum().sum() == pytest.approx(excess, abs=1e-12)
    assert account.Fixed_Fee_Effect.iloc[1:].lt(0).all()
    assert account.Spread_Effect.iloc[1:].lt(0).all()
    with pytest.raises(ValueError, match="periods do not match"):
        run_live_brinson(inputs, shares, strategy.holdings.loc[strategy.holdings.Rebalance_ID.ne("R3")], strategy.nav)
    broken = strategy.nav.copy()
    broken.loc[1, "Start_NAV"] = 1_000_000
    with pytest.raises(ValueError, match="adjacent NAV"):
        run_live_brinson(inputs, shares, strategy.holdings, broken)


@pytest.mark.parametrize("table", ["market_daily", "benchmark_daily", "sector_assignments"])
def test_missing_may_1_inputs_remain_errors(canonical, table):
    inputs, _ = canonical
    frame = getattr(inputs, table)
    date_column = "As_Of_Date" if table == "sector_assignments" else "Date"
    missing = replace(inputs, **{table: frame.loc[frame[date_column].ne(pd.Timestamp("2026-05-01"))]})
    with pytest.raises((ValueError, RuntimeError, KeyError), match="[Mm]issing|[Nn]o |[Ss]ector|too few causal candidates"):
        run_strategy_analysis(missing, _config())


def test_noncausal_price_fields_are_rejected():
    period = DEFAULT_CONFIG.schedule[0]
    with pytest.raises(ValueError, match="Open/Close"):
        replace(period, sizing_date=period.execution_date, sizing_field="Close")
    with pytest.raises(ValueError, match="Open/Close"):
        replace(period, sizing_field="Open")


@pytest.mark.parametrize("missing_date", ["2026-02-13", "2026-05-01"])
def test_preparation_readiness_rejects_missing_required_open(canonical, tmp_path, missing_date):
    inputs, _ = canonical
    asset = "A000"
    market = inputs.market_daily.loc[inputs.market_daily.Asset_ID.eq(asset)].set_index("Date")
    panels = {
        field.lower(): market[[field]].rename(columns={field: asset})
        for field in ("Open", "Close", "Volume")
    }
    requirements = tuple(
        item for item in build_price_requirements(
            inputs.schedule, inputs.membership, evaluation=inputs.evaluation_periods,
        )
        if item.asset_id == asset
    )
    readiness_inputs = dict(
        requirements=requirements, symbol_by_asset={asset: asset},
        supplemental_dates={}, corporate_actions=pd.DataFrame(columns=["Asset_ID"]),
        checked_at_utc="2026-05-07T00:00:00Z",
    )
    _, ready = acquisition_execution._build_price_readiness(panels, **readiness_inputs)
    assert ready[0].status.value == "complete"

    panels["open"].loc[pd.Timestamp(missing_date), asset] = np.nan
    _, ready = acquisition_execution._build_price_readiness(panels, **readiness_inputs)
    assert ready[0].status.value == "partial"
    assert ready[0].missing_dates == (missing_date,)
    path = tmp_path / "readiness.csv"
    write_readiness(path, ready)
    with pytest.raises(RuntimeError, match="Incomplete downstream readiness"):
        preparation_builders._load_readiness(
            path, {asset}, DOWNLOAD_PRICES_COMMAND, requirement_set=PRICE_REQUIREMENT_SET,
        )
