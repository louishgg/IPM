"""Persistence contracts for validated live analysis data."""

from types import SimpleNamespace

import pandas as pd
import pytest

from live import analysis_data as live_analysis_data
from live.analysis_data import (
    LiveAnalysisResult,
    load_analysis_inputs,
    price_series,
    read_prepared_csv,
    save_analysis_result,
)
from live.paths import LivePaths
from live.preparation_artifacts import PREPARE_COMMAND
from live.strategy_universe import (
    DOWNLOAD_BENCHMARK_COMMAND,
    DOWNLOAD_PRICES_COMMAND,
    decision_schedule,
)


def test_read_prepared_csv_recovery_message_is_exact(tmp_path):
    path = tmp_path / "missing-signal-matrix.csv"

    with pytest.raises(FileNotFoundError) as exc_info:
        read_prepared_csv(path, "signal matrix")

    assert str(exc_info.value) == (
        f"Missing prepared signal matrix: {path}. "
        f"Run `{PREPARE_COMMAND}` first."
    )


def test_benchmark_loading_recovery_message_is_exact(tmp_path, monkeypatch):
    existing = tmp_path / "existing.csv"
    pd.DataFrame({"Value": [1]}).to_csv(existing, index=False)
    schedule_path = tmp_path / "schedule.csv"
    decision_schedule().to_csv(schedule_path, index=False)
    missing_benchmark = tmp_path / "missing-benchmark.csv"
    paths = SimpleNamespace(
        market_daily_csv=existing,
        pit_membership_csv=existing,
        asset_metadata_csv=existing,
        sector_assignments_csv=existing,
        decision_schedule_csv=schedule_path,
        prepared_benchmark_csv=missing_benchmark,
        prepared_corporate_action_events_csv=existing,
        prepared_corporate_action_legs_csv=existing,
        prepared_corporate_action_sources_csv=existing,
        prepared_corporate_action_policy_csv=existing,
        price_basis_csv=existing,
    )
    monkeypatch.setattr(
        live_analysis_data,
        "validate_analysis_manifest",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        live_analysis_data,
        "validate_price_basis",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        live_analysis_data,
        "load_sector_assignments",
        lambda _path: pd.DataFrame(),
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        load_analysis_inputs(paths)

    assert str(exc_info.value) == (
        f"Missing prepared benchmark: {missing_benchmark}. "
        f"Run `{DOWNLOAD_BENCHMARK_COMMAND}` first."
    )


def test_price_series_recovery_message_is_exact():
    market = pd.DataFrame(columns=["Date", "Asset_ID", "Close"])

    with pytest.raises(RuntimeError) as exc_info:
        price_series(
            market,
            pd.Timestamp("2026-02-13"),
            "Close",
            ["BBB", "AAA", "BBB"],
            context="R7 sizing",
        )

    assert str(exc_info.value) == (
        "Missing prepared close prices for R7 sizing on 2026-02-13: "
        f"['AAA', 'BBB']. Run `{DOWNLOAD_PRICES_COMMAND}` then "
        f"`{PREPARE_COMMAND}`."
    )


def test_analysis_result_persistence_writes_documented_tables(tmp_path):
    frames = [pd.DataFrame({"Value": [number]}) for number in range(6)]
    frames[0] = pd.DataFrame([{
        "Rebalance_ID": "R1",
        "Period_Type": "invested",
        "Period_Start": "2026-03-02",
        "Start_Field": "Open",
        "Period_End": "2026-04-01",
        "End_Field": "Open",
        "Start_NAV": 1_000.0,
        "Period_Return": 0.009,
        "Benchmark_Return": 0.01,
        "Fixed_Fees": 2.0,
        "Spread_Cost": 0.0,
        "Position_Count": 20,
        "End_Position_Count": 20,
        "Order_Count": 1,
        "Post_Trade_Gross_Market_Value": 400.0,
        "Post_Trade_NAV": 998.0,
        "End_NAV": 1_009.0,
        "Interest": -1.0,
        "Cash_Interest_Credit": 0.0,
        "Loan_Interest_Charge": 1.0,
        "Post_Trade_Cash": 798.0,
        "Post_Trade_Signed_Market_Value": 200.0,
        "Restricted_Short_Proceeds": 100.0,
        "Free_Cash": 698.0,
        "Loan": 0.0,
        "End_Cash": 809.0,
        "End_Signed_Market_Value": 200.0,
        "End_Restricted_Short_Proceeds": 1_000.0,
        "End_Free_Cash": -191.0,
        "End_Loan": 191.0,
    }])
    result = LiveAnalysisResult(*frames, daily_nav=pd.DataFrame({"Value": [6]}))

    paths = LivePaths(tmp_path / "live").for_strategy("momentum")
    save_analysis_result(result, paths)

    expected = dict(
        zip(
            (
                "nav.csv",
                "holdings.csv",
                "trades.csv",
                "decisions.csv",
                "performance.csv",
                "corporate_actions.csv",
            ),
            frames,
            strict=True,
        )
    )
    expected["daily_nav.csv"] = result.daily_nav
    assert {path.name for path in paths.strategy_tables_dir.glob("*.csv")} == set(expected)
    for filename, frame in expected.items():
        pd.testing.assert_frame_equal(
            pd.read_csv(paths.strategy_tables_dir / filename),
            frame,
        )
