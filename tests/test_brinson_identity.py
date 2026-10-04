"""Regression tests for Asset_ID boundaries in Brinson attribution."""

from types import SimpleNamespace

import pandas as pd
import pytest

import backtest.brinson_attribution as brinson_attribution
import backtest.acquisition_planning as acquisition_planning
from backtest.data_loading import BacktestDataset
from backtest.paths import BacktestPaths
from portfolio_core.corporate_actions import EVENT_COLUMNS, LEG_COLUMNS, SOURCE_COLUMNS
from portfolio_core.brinson_attribution import add_benchmark_consistency_columns, run_brinson_pipeline
from tests.test_portfolio_brinson_attribution import _two_sector_inputs


def test_backtest_csvs_save_three_effect_values_without_replacing_gross_holdings(tmp_path):
    benchmark, audit, holdings = _two_sector_inputs()
    result = run_brinson_pipeline(benchmark, add_benchmark_consistency_columns(audit), holdings)
    paths = BacktestPaths(tmp_path / "backtest").for_strategy("momentum")
    paths.brinson_holdings_gross_csv.parent.mkdir(parents=True)
    paths.brinson_holdings_gross_csv.write_text("preserved gross holdings\n")

    brinson_attribution.save_brinson_result(result, paths)

    assert paths.brinson_holdings_gross_csv.read_text() == "preserved gross holdings\n"
    for path in (paths.attribution.sector_attribution_csv, paths.attribution.monthly_attribution_csv,
                 paths.attribution.period_attribution_csv):
        saved = pd.read_csv(path)
        assert saved.columns[0] == "Attribution_Method"
        assert saved.Attribution_Method.eq("Brinson-Fachler").all()
    sector = pd.read_csv(paths.attribution.sector_attribution_csv)
    row = sector.loc[sector.Side.eq("Long") & sector.GICS_Sector_Code.eq(10)].iloc[0]
    assert row.Allocation_Effect == pytest.approx(-.0048, abs=1e-14)
    assert row.Selection_Effect == pytest.approx(.008, abs=1e-14)
    assert row.Interaction_Effect == pytest.approx(.004, abs=1e-14)
    assert row.Scaled_Interaction_Effect == pytest.approx(.0048, abs=1e-14)
    assert row.Active_Contribution - row.Benchmark_Centering_Adjustment == pytest.approx(row.Total_Effect, abs=1e-14)
    interval = pd.read_csv(paths.attribution.monthly_attribution_csv).set_index("Side")
    total = pd.read_csv(paths.attribution.period_attribution_csv).set_index("Side")
    assert interval.at["Combined", "Scaled_Active_Return"] == pytest.approx(.0096, abs=1e-14)
    assert total.at["Combined", "Total_Selection_Effect"] == pytest.approx(.016, abs=1e-14)
    assert total.at["Combined", "Total_Interaction_Effect"] == pytest.approx(0, abs=1e-14)
    assert len(paths.attribution.result_files) == 8
    assert all(path.is_file() for path in paths.attribution.result_files)


def _sector_assignments(
    dates: pd.DatetimeIndex,
    sectors: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "As_Of_Date": date,
                "Asset_ID": asset_id,
                "GICS_Sector_Code": code,
                "Sector": label,
                "Source_Type": "Wikipedia",
                "Source_Reference": f"revision-{date:%Y%m%d}",
                "Source_Symbol": asset_id,
                "Resolution_Method": "exact_wikipedia_symbol",
            }
            for date in dates
            for asset_id, (code, label) in sectors.items()
        ]
    )


def make_identity_backtest_data(
    asset_to_ticker: dict[str, str] | None = None,
) -> tuple[BacktestDataset, pd.DatetimeIndex]:
    dates = pd.DatetimeIndex([
        pd.Timestamp("2022-01-31"),
        pd.Timestamp("2022-02-28"),
    ])
    asset_ids = ["AAA.O^Z30", "BBB.K"]
    close = pd.DataFrame(
        [[10.0, 20.0], [11.0, 18.0]],
        index=dates,
        columns=asset_ids,
    )
    volume = pd.DataFrame(
        1_000_000.0,
        index=dates,
        columns=asset_ids,
    )
    pit_matrix = pd.DataFrame(True, index=dates, columns=asset_ids)
    backtest_data = BacktestDataset(
        data_close=close,
        data_volume=volume,
        pit_matrix=pit_matrix,
        sector_assignments=_sector_assignments(
            dates,
            {
                "AAA.O^Z30": ("45", "Information Technology"),
                "BBB.K": ("40", "Financials"),
            },
        ),
        valid_trading_days=dates,
        rolling_dollar_vol=close * volume,
        asset_to_ticker=asset_to_ticker or {
            "AAA.O^Z30": "AAA",
            "BBB.K": "BBB",
        },
    )
    return backtest_data, dates


def test_brinson_benchmark_requires_asset_id_keyed_shares():
    backtest_data, dates = make_identity_backtest_data()
    _, asset_to_ticker, membership = acquisition_planning._build_share_identity_context(
        backtest_data, backtest_data.data_close.index,
    )
    shares = pd.DataFrame(
        [[100.0, 200.0], [100.0, 200.0]],
        index=dates,
        columns=["AAA.O^Z30", "BBB.K"],
    )
    periods = pd.DataFrame({
        "Date": [dates[0]],
        "Next_Date": [dates[1]],
    })

    benchmark, audit = brinson_attribution.build_brinson_benchmark_sector_series(
        backtest_data,
        _reviewed_table(shares),
        periods,
    )

    assert asset_to_ticker == {"AAA.O^Z30": "AAA", "BBB.K": "BBB"}
    assert membership.loc[dates[0]].to_dict() == {
        "AAA.O^Z30": True,
        "BBB.K": True,
    }

    by_sector = benchmark.set_index("GICS_Sector_Code")
    assert by_sector.loc["45", "Benchmark_Weight"] == pytest.approx(0.2)
    assert by_sector.loc["40", "Benchmark_Weight"] == pytest.approx(0.8)
    assert by_sector.loc["45", "Benchmark_Return"] == pytest.approx(0.1)
    assert by_sector.loc["40", "Benchmark_Return"] == pytest.approx(-0.1)

    assert audit.loc[0, "PiT_Active_Count"] == 2
    assert audit.loc[0, "Priced_Constituent_Count"] == 2
    assert audit.loc[0, "Valid_Constituent_Count"] == 2
    assert audit.loc[0, "Missing_Price_Count"] == 0
    assert audit.loc[0, "Missing_Shares_Count"] == 0


def test_brinson_missing_asset_shares_has_exact_recovery_message():
    backtest_data, dates = make_identity_backtest_data()
    shares = pd.DataFrame(
        [[200.0]],
        index=[dates[0]],
        columns=["BBB.K"],
    )
    periods = pd.DataFrame({
        "Date": [dates[0]],
        "Next_Date": [dates[1]],
    })

    with pytest.raises(RuntimeError) as error:
        brinson_attribution.build_brinson_benchmark_sector_series(
            backtest_data,
            _reviewed_table(shares),
            periods,
        )

    assert str(error.value) == (
        "Missing prepared shares for priced membership constituents on "
        "2022-01-31: ['AAA.O^Z30']. "
        "Run `python -m backtest.prepare brinson`; inspect any unresolved "
        "reviewed-shares diagnostics and correct the required inputs."
    )


def test_brinson_keeps_display_ticker_collisions_asset_keyed():
    backtest_data, _ = make_identity_backtest_data({
        "AAA.O^Z30": "SAME",
        "BBB.K": "SAME",
    })

    _, asset_to_ticker, membership = acquisition_planning._build_share_identity_context(
        backtest_data, backtest_data.data_close.index,
    )

    assert asset_to_ticker == {"AAA.O^Z30": "SAME", "BBB.K": "SAME"}
    assert membership.columns.tolist() == ["AAA.O^Z30", "BBB.K"]


def test_holdings_loader_requires_asset_id(tmp_path):
    paths = BacktestPaths(tmp_path / "backtest").for_strategy(
        "momentum"
    )
    holdings_path = paths.brinson_holdings_gross_csv
    holdings_path.parent.mkdir(parents=True)
    pd.DataFrame({
        "Date": ["2022-01-31"],
        "Next_Date": ["2022-02-28"],
        "Ticker": ["AAA"],
        "GICS_Sector_Code": ["45"],
        "Sector": ["Information Technology"],
        "Sector_As_Of_Date": ["2022-01-31"],
        "Sector_Source_Type": ["Wikipedia"],
        "Sector_Source_Reference": ["revision-20220131"],
        "Weight": [0.25],
        "Stock_Return": [0.1],
    }).to_csv(holdings_path, index=False)
    with pytest.raises(ValueError, match="Asset_ID"):
        brinson_attribution.load_gross_brinson_holdings(paths)


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("fault", [None, "sector_date", "sector_label", "provenance"])
def test_saved_and_supplied_holdings_share_one_provenance_check(tmp_path, monkeypatch, saved, fault):
    benchmark, audit, holdings = _two_sector_inputs()
    assets = ["A", "B", "C", "D"]
    holdings = holdings.assign(
        Asset_ID=assets, Ticker=assets, Sector_As_Of_Date=holdings.Date,
        Sector_Source_Type="Wikipedia", Sector_Source_Reference="revision-20260302",
    )
    assignments = _sector_assignments(pd.DatetimeIndex([holdings.Date.iloc[0]]), {
        asset: (str(row.GICS_Sector_Code), str(row.Sector))
        for asset, row in zip(assets, holdings.itertuples(index=False))
    })
    data = SimpleNamespace(sector_assignments=assignments)
    if fault == "sector_date":
        holdings.loc[0, "Sector_As_Of_Date"] = holdings.Next_Date.iloc[0]
    elif fault == "sector_label":
        holdings.loc[0, "Sector"] = "Incorrect sector"
    elif fault == "provenance":
        holdings.loc[0, "Sector_Source_Reference"] = "Unknown"
    before = holdings.copy(deep=True)
    paths = BacktestPaths(tmp_path / "backtest").for_strategy("momentum")
    if saved:
        paths.brinson_holdings_gross_csv.parent.mkdir(parents=True)
        holdings.to_csv(paths.brinson_holdings_gross_csv, index=False)

    calls = []
    validate = brinson_attribution.validate_holding_sector_provenance

    def checked(frame, sector_assignments, *, context):
        calls.append(context)
        return validate(frame, sector_assignments, context=context)

    monkeypatch.setattr(brinson_attribution, "validate_holding_sector_provenance", checked)
    monkeypatch.setattr(brinson_attribution, "build_brinson_benchmark_sector_series", lambda *args: (benchmark, audit))
    monkeypatch.setattr(brinson_attribution, "add_benchmark_consistency_columns",
                        lambda frame, paths: add_benchmark_consistency_columns(frame))

    def calculate():
        return brinson_attribution.run_brinson_attribution(
            data, None if saved else holdings, shares_monthly=pd.DataFrame(), paths=paths,
        )

    if fault:
        with pytest.raises(ValueError, match="sector|provenance"):
            calculate()
    else:
        result = calculate()
        assert result.total_attribution.set_index("Side").at["Combined", "Total_Active_Return"] == pytest.approx(.0096)
    assert calls == ["Gross Brinson"]
    pd.testing.assert_frame_equal(holdings, before)


def _attach_stock_exchange(
    backtest_data: BacktestDataset,
    *,
    event_id: str,
    effective_date: str,
    predecessor: str,
    successor: str,
    ratio: str,
) -> None:
    backtest_data.security_events = pd.DataFrame(
        [[
            event_id,
            effective_date,
            "stock_exchange",
            "predecessor_extinguished",
            "approved",
        ]],
        columns=EVENT_COLUMNS,
    )
    backtest_data.security_event_legs = pd.DataFrame(
        [[
            event_id,
            1,
            predecessor,
            successor,
            "stock",
            ratio,
            "",
            "",
            "",
            "",
            "",
            True,
            "approved",
        ]],
        columns=LEG_COLUMNS,
    )
    backtest_data.security_event_sources = pd.DataFrame(
        [[event_id, "https://example.test/official", "approved"]],
        columns=SOURCE_COLUMNS,
    )


def test_brinson_uses_complete_event_economics_for_period_return():
    dates = pd.DatetimeIndex(["2022-01-31", "2022-02-28"])
    assets = ["PRE.K", "SUC.K", "OTHER.K"]
    close = pd.DataFrame(
        [[10.0, 25.0, 20.0], [12.0, 30.0, 20.0]],
        index=dates,
        columns=assets,
    )
    membership = pd.DataFrame(
        [[True, False, True], [False, True, True]],
        index=dates,
        columns=assets,
    )
    backtest_data = BacktestDataset(
        data_close=close,
        data_volume=pd.DataFrame(1_000.0, index=dates, columns=assets),
        pit_matrix=membership,
        sector_assignments=_sector_assignments(
            dates,
            {
                "PRE.K": ("45", "Information Technology"),
                "SUC.K": ("45", "Information Technology"),
                "OTHER.K": ("40", "Financials"),
            },
        ),
        valid_trading_days=dates,
        rolling_dollar_vol=close * 1_000.0,
        asset_to_ticker={asset_id: asset_id for asset_id in assets},
    )
    _attach_stock_exchange(
        backtest_data,
        event_id="EVT-20220215-PRE-SUC-STOCK-EXCHANGE",
        effective_date="2022-02-15",
        predecessor="PRE.K",
        successor="SUC.K",
        ratio="0.5",
    )
    shares = pd.DataFrame(
        [[100.0, float("nan"), 100.0]],
        index=[dates[0]],
        columns=assets,
    )
    periods = pd.DataFrame({"Date": [dates[0]], "Next_Date": [dates[1]]})

    benchmark, audit = brinson_attribution.build_brinson_benchmark_sector_series(
        backtest_data,
        _reviewed_table(shares),
        periods,
    )

    by_sector = benchmark.set_index("GICS_Sector_Code")
    # One PRE share becomes 0.5 SUC share, worth 15 at t1 versus 10 at t0;
    # the lingering predecessor quote of 12 must not replace event economics.
    assert by_sector.loc[
        "45", "Benchmark_Return"
    ] == pytest.approx(0.5)
    assert audit.loc[0, "Missing_Price_Count"] == 0


def test_brinson_excludes_info_extinguished_before_open_on_membership_date():
    dates = pd.DatetimeIndex(["2022-02-28", "2022-03-31"])
    price_assets = ["SPGI.K", "OTHER.K"]
    close = pd.DataFrame(
        [[400.0, 20.0], [410.0, 21.0]],
        index=dates,
        columns=price_assets,
    )
    membership = pd.DataFrame(
        [[False, True, True], [True, False, True]],
        index=dates,
        columns=["SPGI.K", "INFO.K^B22", "OTHER.K"],
    )
    backtest_data = BacktestDataset(
        data_close=close,
        data_volume=pd.DataFrame(1_000.0, index=dates, columns=price_assets),
        pit_matrix=membership,
        sector_assignments=_sector_assignments(
            dates,
            {
                "SPGI.K": ("40", "Financials"),
                "INFO.K^B22": ("40", "Financials"),
                "OTHER.K": ("20", "Industrials"),
            },
        ),
        valid_trading_days=dates,
        rolling_dollar_vol=close * 1_000.0,
        asset_to_ticker={
            asset_id: asset_id
            for asset_id in ["INFO.K^B22", *price_assets]
        },
    )
    _attach_stock_exchange(
        backtest_data,
        event_id="EVT-20220228-INFO-SPGI-STOCK-EXCHANGE",
        effective_date="2022-02-28",
        predecessor="INFO.K^B22",
        successor="SPGI.K",
        ratio="0.2838",
    )
    # No fictitious INFO price or shares are supplied at/after its before-open
    # extinction date; membership lag alone must not create a requirement.
    shares = pd.DataFrame(
        [[float("nan"), 100.0]],
        index=[dates[0]],
        columns=price_assets,
    )
    periods = pd.DataFrame({"Date": [dates[0]], "Next_Date": [dates[1]]})

    benchmark, audit = brinson_attribution.build_brinson_benchmark_sector_series(
        backtest_data,
        _reviewed_table(shares),
        periods,
    )

    assert set(benchmark["GICS_Sector_Code"]) == {"20"}
    assert audit.loc[0, "PiT_Active_Count"] == 2
    assert audit.loc[0, "Priced_Constituent_Count"] == 1
    assert audit.loc[0, "Extinguished_Predecessor_Count"] == 1
    assert audit.loc[0, "Extinguished_Predecessors"] == "INFO.K^B22"
    assert audit.loc[0, "Unavailable_Member_Count"] == 0


def test_brinson_keeps_approved_same_key_stock_collision_eligible():
    backtest_data, dates = make_identity_backtest_data()
    _attach_stock_exchange(
        backtest_data,
        event_id="EVT-20220131-AAA-AAA-STOCK-EXCHANGE",
        effective_date="2022-01-31",
        predecessor="AAA.O^Z30",
        successor="AAA.O^Z30",
        ratio="1",
    )
    shares = pd.DataFrame(
        [[100.0, 200.0]], index=[dates[0]], columns=backtest_data.data_close.columns
    )
    periods = pd.DataFrame({"Date": [dates[0]], "Next_Date": [dates[1]]})

    benchmark, audit = brinson_attribution.build_brinson_benchmark_sector_series(
        backtest_data,
        _reviewed_table(shares),
        periods,
    )

    assert set(benchmark["GICS_Sector_Code"]) == {"45", "40"}
    assert audit.loc[0, "Extinguished_Predecessor_Count"] == 0
    assert audit.loc[0, "Valid_Constituent_Count"] == 2


def test_brinson_fails_closed_when_event_successor_cannot_be_valued():
    backtest_data, dates = make_identity_backtest_data()
    backtest_data.data_close.loc[dates[1], "BBB.K"] = float("nan")
    _attach_stock_exchange(
        backtest_data,
        event_id="EVT-20220215-AAA-BBB-STOCK-EXCHANGE",
        effective_date="2022-02-15",
        predecessor="AAA.O^Z30",
        successor="BBB.K",
        ratio="0.5",
    )
    shares = pd.DataFrame(
        [[100.0, 200.0]], index=[dates[0]], columns=backtest_data.data_close.columns
    )
    periods = pd.DataFrame({"Date": [dates[0]], "Next_Date": [dates[1]]})

    with pytest.raises(RuntimeError, match="Missing price.*AAA.O\\^Z30"):
        brinson_attribution.build_brinson_benchmark_sector_series(
            backtest_data,
            _reviewed_table(shares),
            periods,
        )


def _reviewed_table(wide):
    from backtest.share_resolution import RESULT_COLUMNS
    rows = []
    for date, values in wide.iterrows():
        for asset, value in values.dropna().items():
            row = dict.fromkeys(RESULT_COLUMNS, "")
            row.update(Date=date, Asset_ID=asset, Shares_Outstanding=value,
                       Source="sec", Observation_ID="test-"+asset,
                       Observation_Date=date, Filed=date, Age_Days=0,
                       Share_Factor=1.0, Capitalization_Factor=1.0)
            rows.append(row)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(["Date", "Asset_ID"], ignore_index=True)
