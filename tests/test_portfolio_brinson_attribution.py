"""Regression tests for the shared Brinson calculation layer."""

from math import fsum

import numpy as np
import pandas as pd
import pytest

from portfolio_core import brinson_attribution as shared_brinson


def test_benchmark_consistency_columns_compound_period_returns():
    original = pd.DataFrame(
        {
            "Next_Date": pd.to_datetime(["2026-03-02", "2026-04-01"]),
            "Reconstructed_Benchmark_Return": [0.10, -0.05],
            "SP500TR_Return": [0.08, -0.02],
        }
    )

    result = shared_brinson.add_benchmark_consistency_columns(original)

    assert "Return_Diff" not in original.columns
    assert result["Return_Diff"].tolist() == pytest.approx([0.02, -0.03])
    assert result["Abs_Return_Diff"].tolist() == pytest.approx([0.02, 0.03])
    assert result["Cumulative_Reconstructed_Benchmark"].tolist() == pytest.approx(
        [0.10, 0.045]
    )
    assert result["Cumulative_SP500TR"].tolist() == pytest.approx(
        [0.08, 0.0584]
    )


def test_benchmark_consistency_requires_both_return_series():
    with pytest.raises(ValueError, match="SP500TR_Return"):
        shared_brinson.add_benchmark_consistency_columns(
            pd.DataFrame({"Reconstructed_Benchmark_Return": [0.01]})
        )


def test_benchmark_audit_sums_the_generated_sector_weights():
    constituents = pd.DataFrame({
        "Period_ID": ["period-1"] * 3,
        "Start_Date": pd.to_datetime(["2026-01-31"] * 3),
        "End_Date": pd.to_datetime(["2026-02-28"] * 3),
        "Asset_ID": ["AAA", "BBB", "CCC"],
        "GICS_Sector_Code": ["10", "20", "40"],
        "Sector": ["Energy", "Industrials", "Financials"],
        "Start_Price": [
            179.00570442533976,
            0.07629306754416253,
            14.851564224361978,
        ],
        "End_Value_Per_Start_Share": [
            179.00570442533976,
            0.07629306754416253,
            14.851564224361978,
        ],
        "Shares_Outstanding": [1.0, 1.0, 1.0],
    })

    sectors, audit = shared_brinson.build_brinson_benchmark(constituents)

    expected = float(fsum(sectors["Benchmark_Weight"].tolist()))
    assert expected == 1.0
    assert audit.at[0, "Sector_Weight_Sum"] == expected


def _two_sector_inputs():
    """Two normalized books with different gross exposures and hand-known effects."""
    benchmark = pd.DataFrame({
        "Date": pd.to_datetime(["2026-03-02"] * 2),
        "Next_Date": pd.to_datetime(["2026-04-01"] * 2),
        "GICS_Sector_Code": ["10", "20"],
        "Sector": ["Energy", "Industrials"],
        "Benchmark_Weight": [.4, .6],
        "Benchmark_Return": [.02, .06],
    })
    audit = benchmark[["Date", "Next_Date"]].drop_duplicates().assign(
        Sector_Weight_Sum=1.0, Reconstructed_Benchmark_Return=.044, SP500TR_Return=.04,
    )
    holdings = pd.concat([benchmark.iloc[:, :4]] * 2, ignore_index=True).assign(
        Weight=[.72, .48, -.24, -.16], Stock_Return=[.04, .08, .04, .08],
    )
    return benchmark, audit, holdings


def test_three_effect_bf_hand_calculation_and_long_short_scaling():
    benchmark, audit, holdings = _two_sector_inputs()
    result = shared_brinson.run_brinson_pipeline(benchmark, audit, holdings)
    sectors = result.sector_attribution
    expected = {
        "Allocation_Effect": [-.0048, -.0032],
        "Selection_Effect": [.008, .012],
        "Interaction_Effect": [.004, -.004],
        "Total_Effect": [.0072, .0048],
        "Active_Contribution": [.016, -.004],
        "Benchmark_Centering_Adjustment": [.0088, -.0088],
        "Effective_Benchmark_Total_Return": [.044, .044],
    }
    for side, sign, gross in (("Long", 1, 1.2), ("Short", -1, .4)):
        rows = sectors.loc[sectors.Side.eq(side)]
        for column, values in expected.items():
            np.testing.assert_allclose(rows[column], sign * np.array(values), atol=1e-14, rtol=0)
            if f"Scaled_{column}" in rows:
                np.testing.assert_allclose(rows[f"Scaled_{column}"], gross * sign * np.array(values),
                                           atol=1e-14, rtol=0)
        # A positive sector return below RB has negative BF allocation when overweight.
        assert rows.iloc[0].Allocation_Effect * sign < 0
        old_allocation = (rows.Portfolio_Weight - rows.Benchmark_Weight) * rows.Effective_Benchmark_Return
        old_selection = rows.Portfolio_Weight * (rows.Effective_Portfolio_Return - rows.Effective_Benchmark_Return)
        np.testing.assert_allclose(rows.Selection_Effect + rows.Interaction_Effect,
                                   old_selection, atol=1e-14, rtol=0)
        assert rows.Allocation_Effect.sum() == pytest.approx(old_allocation.sum(), abs=1e-14)
        np.testing.assert_allclose(rows.Residual, 0, atol=1e-14, rtol=0)

    combined = result.period_attribution.set_index("Side").loc["Combined"]
    assert combined.Scaled_Active_Return == pytest.approx(.0096, abs=1e-14)
    assert combined.Scaled_Allocation_Effect == pytest.approx(-.0064, abs=1e-14)
    assert combined.Scaled_Selection_Effect == pytest.approx(.016, abs=1e-14)
    assert combined.Scaled_Interaction_Effect == pytest.approx(0, abs=1e-14)
    assert np.isnan(combined.Interaction_Effect)
    totals = result.total_attribution.set_index("Side")
    assert totals.at["Long", "Total_Selection_Effect"] == pytest.approx(.02, abs=1e-14)
    assert totals.at["Long", "Total_Scaled_Selection_Effect"] == pytest.approx(.024, abs=1e-14)
    assert totals.at["Combined", "Total_Selection_Effect"] == pytest.approx(.016, abs=1e-14)
    np.testing.assert_allclose(totals[["Allocation_Share_Of_Active", "Selection_Share_Of_Active",
                                      "Interaction_Share_Of_Active"]].sum(axis=1), 1, atol=1e-14, rtol=0)
    for frame in (sectors, result.period_attribution, result.total_attribution):
        assert frame.columns[0] == "Attribution_Method"
        assert frame.Attribution_Method.eq("Brinson-Fachler").all()


@pytest.mark.parametrize("side, sign", [("Long", 1.), ("Short", -1.)])
def test_unheld_sector_has_no_fabricated_return_or_offsetting_selection(side, sign):
    benchmark, audit, holdings = _two_sector_inputs()
    holdings = holdings.iloc[[0]].assign(Weight=sign * 1.2)
    result = shared_brinson.run_brinson_pipeline(benchmark, audit, holdings)
    absent = result.sector_attribution.set_index("GICS_Sector_Code").loc["20"]
    assert absent.Missing_Portfolio_Sector
    assert not absent.Missing_Benchmark_Sector
    assert np.isnan(absent.Portfolio_Return) and np.isnan(absent.Effective_Portfolio_Return)
    assert absent.Portfolio_Weight == absent.Selection_Effect == absent.Interaction_Effect == 0
    assert absent.Allocation_Effect == pytest.approx(sign * -.0096, abs=1e-14)
    assert absent.Active_Contribution == pytest.approx(sign * -.036, abs=1e-14)
    assert absent.Benchmark_Centering_Adjustment == pytest.approx(sign * -.0264, abs=1e-14)
    sides = result.period_attribution.set_index("Side")
    assert sides.index.tolist() == ["Combined", side]
    assert sides.at[side, "Active_Return"] == pytest.approx(sign * -.004, abs=1e-14)
    assert sides.at["Combined", "Scaled_Active_Return"] == pytest.approx(sign * -.0048, abs=1e-14)


def test_zero_benchmark_weight_uses_supplied_return_and_retains_interaction():
    benchmark, audit, holdings = _two_sector_inputs()
    benchmark["Benchmark_Weight"] = [0., 1.]
    audit["Reconstructed_Benchmark_Return"] = .06
    result = shared_brinson.run_brinson_pipeline(benchmark, audit, holdings)
    row = result.sector_attribution.loc[
        result.sector_attribution.Side.eq("Long") & result.sector_attribution.GICS_Sector_Code.eq("10")
    ].iloc[0]
    assert row.Selection_Effect == 0
    assert row.Interaction_Effect == pytest.approx(.012, abs=1e-14)
    assert row.Allocation_Effect == pytest.approx(-.024, abs=1e-14)
    benchmark.loc[0, "Benchmark_Return"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        shared_brinson.run_brinson_pipeline(benchmark, audit, holdings)


@pytest.mark.parametrize("column, value", [("Weight", np.nan), ("Weight", np.inf),
                                           ("Stock_Return", np.nan), ("Stock_Return", -np.inf)])
def test_invalid_held_data_is_not_masked_by_sector_averaging(column, value):
    benchmark, audit, holdings = _two_sector_inputs()
    holdings.loc[0, column] = value
    with pytest.raises(ValueError, match="finite"):
        shared_brinson.run_brinson_pipeline(benchmark, audit, holdings)


@pytest.mark.parametrize("column", ["Weight", "Stock_Return"])
def test_nullable_missing_held_data_is_rejected(column):
    benchmark, audit, holdings = _two_sector_inputs()
    holdings[column] = holdings[column].astype("Float64")
    holdings.loc[0, column] = pd.NA
    with pytest.raises(ValueError, match="finite"):
        shared_brinson.run_brinson_pipeline(benchmark, audit, holdings)


def test_zero_positions_are_ignored_and_all_cash_does_not_invent_a_book():
    benchmark, audit, holdings = _two_sector_inputs()
    zero = holdings.iloc[[0]].assign(Weight=0., Stock_Return=np.nan, Sector=None)
    expected = shared_brinson.run_brinson_pipeline(benchmark, audit, holdings)
    actual = shared_brinson.run_brinson_pipeline(benchmark, audit, pd.concat([holdings, zero]))
    pd.testing.assert_frame_equal(actual.sector_attribution, expected.sector_attribution)
    with pytest.raises(RuntimeError, match="No long or short"):
        shared_brinson.build_brinson_portfolio_sector_series(zero)


@pytest.mark.parametrize("column, value", [("Benchmark_Weight", np.nan), ("Benchmark_Weight", -.1),
                                           ("Benchmark_Return", np.inf), ("Benchmark_Return", np.nan)])
def test_invalid_benchmark_data_fails_even_in_unheld_sectors(column, value):
    benchmark, audit, holdings = _two_sector_inputs()
    benchmark.loc[1, column] = value
    with pytest.raises(ValueError, match="finite|nonnegative"):
        shared_brinson.run_brinson_pipeline(benchmark, audit, holdings.iloc[[0]])


def test_held_sector_without_benchmark_is_an_error():
    benchmark, audit, holdings = _two_sector_inputs()
    with pytest.raises(ValueError, match="absent from the benchmark"):
        shared_brinson.run_brinson_pipeline(benchmark.iloc[[0]], audit, holdings)


def test_zero_active_return_leaves_effect_shares_undefined():
    benchmark, audit, holdings = _two_sector_inputs()
    holdings["Weight"] = [.4, .6, -.4, -.6]
    holdings["Stock_Return"] = [.02, .06, .02, .06]
    result = shared_brinson.run_brinson_pipeline(benchmark, audit, holdings)
    shares = ["Allocation_Share_Of_Active", "Selection_Share_Of_Active", "Interaction_Share_Of_Active"]
    assert result.total_attribution[shares].isna().all().all()


@pytest.mark.parametrize("table, column, value", [
    ("sector", "Interaction_Effect", np.nan),
    ("sector", "Benchmark_Centering_Adjustment", 0.),
    ("sector", "Scaled_Interaction_Effect", .2),
    ("sector", "Effective_Benchmark_Total_Return", -.044),
    ("interval", "Scaled_Interaction_Effect", np.nan),
    ("interval", "Interaction_Effect", .1),
])
def test_validator_rejects_nonfinite_or_broken_bf_contracts(table, column, value):
    benchmark, audit, holdings = _two_sector_inputs()
    result = shared_brinson.run_brinson_pipeline(benchmark, audit, holdings)
    sector = result.sector_attribution.copy()
    interval = result.period_attribution.copy()
    frame = sector if table == "sector" else interval
    frame.loc[0, column] = value
    with pytest.raises(RuntimeError):
        shared_brinson.validate_brinson_outputs(
            audit, result.portfolio_sector, interval, sector_attribution_df=sector,
        )


def _calculated_tables():
    benchmark, audit, holdings = _two_sector_inputs()
    result = shared_brinson.run_brinson_pipeline(benchmark, audit, holdings)
    return {
        "audit": audit, "portfolio": result.portfolio_sector,
        "sector": result.sector_attribution, "interval": result.period_attribution,
    }


def _validate_tables(tables):
    shared_brinson.validate_brinson_outputs(
        tables["audit"], tables["portfolio"], tables["interval"],
        sector_attribution_df=tables["sector"],
    )


@pytest.mark.parametrize("value", [.5, np.nan, np.inf])
def test_validator_matches_finite_audit_returns_to_attribution(value):
    tables = _calculated_tables()
    tables["audit"]["Reconstructed_Benchmark_Return"] = value
    with pytest.raises(RuntimeError, match="Reconstructed_Benchmark_Return.*2026-03-02"):
        _validate_tables(tables)


@pytest.mark.parametrize("side,column", [
    ("Long", "Scaled_Interaction_Effect"), ("Short", "Scaled_Allocation_Effect"),
    ("Long", "Scaled_Portfolio_Return"), ("Combined", "Scaled_Interaction_Effect"),
    ("Combined", "Scaled_Total_Effect"), ("Combined", "Side_Gross_Exposure"),
    ("Combined", "Interaction_Effect"),
])
def test_validator_rejects_broken_side_scaling_and_combined_sums(side, column):
    tables = _calculated_tables()
    interval = tables["interval"]
    interval.loc[interval.Side.eq(side), column] = 123.
    with pytest.raises(RuntimeError) as error:
        _validate_tables(tables)
    assert column in str(error.value) and side in str(error.value) and "2026-03-02" in str(error.value)


def test_validator_rejects_offsetting_summary_errors_even_when_totals_match():
    tables = _calculated_tables()
    interval = tables["interval"]
    interval.loc[interval.Side.eq("Long"), "Allocation_Effect"] += .01
    interval.loc[interval.Side.eq("Long"), "Selection_Effect"] -= .01
    with pytest.raises(RuntimeError, match="Allocation_Effect sector-to-side"):
        _validate_tables(tables)


@pytest.mark.parametrize("table", ["audit", "portfolio", "sector", "interval"])
@pytest.mark.parametrize("fault", ["duplicate", "missing_date"])
def test_validator_rejects_invalid_table_identities(table, fault):
    tables = _calculated_tables()
    frame = tables[table]
    if fault == "duplicate":
        tables[table] = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    else:
        frame.loc[0, "Date"] = pd.NaT
    with pytest.raises(RuntimeError, match="missing or duplicate identity"):
        _validate_tables(tables)


@pytest.mark.parametrize("fault", ["audit", "portfolio", "sector", "side", "combined", "extra_summary"])
def test_validator_rejects_missing_corresponding_rows_and_extra_summaries(fault):
    tables = _calculated_tables()
    if fault == "audit":
        tables["audit"]["Date"] = pd.Timestamp("2026-02-13")
    elif fault in {"portfolio", "sector"}:
        tables[fault] = tables[fault].iloc[1:]
    elif fault == "extra_summary":
        tables["interval"] = pd.concat([
            tables["interval"], tables["interval"].iloc[[0]].assign(Side="Unexpected"),
        ])
    else:
        side = "Combined" if fault == "combined" else "Long"
        tables["interval"] = tables["interval"].loc[lambda frame: frame.Side.ne(side)]
    with pytest.raises(RuntimeError, match="identities disagree|missing intervals"):
        _validate_tables(tables)


def test_validator_allows_cash_audit_intervals_and_uses_keys_without_mutating():
    tables = _calculated_tables()
    cash = tables["audit"].assign(Date=pd.Timestamp("2026-02-13"), Next_Date=pd.Timestamp("2026-03-02"))
    tables["audit"] = pd.concat([cash, tables["audit"]], ignore_index=True)
    tables = {name: frame.sample(frac=1, random_state=4) for name, frame in tables.items()}
    original = {name: frame.copy(deep=True) for name, frame in tables.items()}
    _validate_tables(tables)
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], original[name])
