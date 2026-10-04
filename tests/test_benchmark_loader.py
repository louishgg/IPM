"""Offline contracts for the prepared benchmark analysis boundary."""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest import acquisition_planning, benchmark
from backtest.config import BacktestBenchmarkConfig
from backtest.preparation_builders import prepare_benchmark_monthly
from data_acquisition.contracts import ReadinessStatus


class PreparedBenchmarkPaths:
    def __init__(self, benchmark_csv):
        self.benchmark_csv = benchmark_csv


def test_benchmark_loader_reads_only_valid_prepared_csv(tmp_path):
    path = tmp_path / "prepared" / "benchmark_monthly.csv"
    path.parent.mkdir()
    pd.DataFrame({
        "Date": ["2024-01-31", "2024-02-29"],
        "SP500TR_Close": [10.0, 11.0],
    }).to_csv(path, index=False)

    loaded = benchmark.load_benchmark(
        BacktestBenchmarkConfig(
            coverage_start_date=date(2024, 1, 1),
            coverage_end_date=date(2024, 2, 28),
        ),
        PreparedBenchmarkPaths(path),
    )

    assert loaded.name == "SP500TR_Close"
    assert loaded.index.equals(pd.DatetimeIndex(["2024-01-31", "2024-02-29"]))
    assert loaded.tolist() == [10.0, 11.0]


def test_benchmark_loader_never_falls_back_when_prepared_csv_is_missing(tmp_path):
    path = tmp_path / "prepared" / "benchmark_monthly.csv"

    with pytest.raises(FileNotFoundError, match="backtest.prepare"):
        benchmark.load_benchmark(BacktestBenchmarkConfig(), PreparedBenchmarkPaths(path))


@pytest.mark.parametrize(
    "frame, message",
    [
        (
            pd.DataFrame({"Date": ["2024-01-30"], "SP500TR_Close": [10.0]}),
            "month-end",
        ),
        (
            pd.DataFrame({"Date": ["2024-01-31"], "Wrong": [10.0]}),
            "must have columns",
        ),
    ],
)
def test_benchmark_loader_rejects_invalid_prepared_contract(
    tmp_path,
    frame,
    message,
):
    path = tmp_path / "benchmark_monthly.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        benchmark.load_benchmark(BacktestBenchmarkConfig(), PreparedBenchmarkPaths(path))


@pytest.mark.parametrize(
    "frame, message",
    [
        pytest.param(
            pd.DataFrame({
                "Date": ["2024-01-31"],
                "SP500TR_Close": [np.nan],
            }),
            "finite and nonmissing",
            id="nan",
        ),
        pytest.param(
            pd.DataFrame({
                "Date": ["2024-01-31"],
                "SP500TR_Close": [np.inf],
            }),
            "finite and nonmissing",
            id="positive-infinity",
        ),
        pytest.param(
            pd.DataFrame({
                "Date": ["2024-01-31"],
                "SP500TR_Close": [-np.inf],
            }),
            "finite and nonmissing",
            id="negative-infinity",
        ),
        pytest.param(
            pd.DataFrame({
                "Date": ["2024-01-31"],
                "SP500TR_Close": [0.0],
            }),
            "must be positive",
            id="zero",
        ),
        pytest.param(
            pd.DataFrame({
                "Date": ["2024-01-31"],
                "SP500TR_Close": [-1.0],
            }),
            "must be positive",
            id="negative",
        ),
        pytest.param(
            pd.DataFrame({
                "Date": ["not-a-date"],
                "SP500TR_Close": [10.0],
            }),
            "invalid Date",
            id="malformed-date",
        ),
        pytest.param(
            pd.DataFrame({
                "Date": ["2024-01-31", "2024-01-31"],
                "SP500TR_Close": [10.0, 11.0],
            }),
            "duplicate.*dates",
            id="duplicate-date",
        ),
    ],
)
def test_invalid_benchmark_is_rejected_across_all_stage_boundaries(
    tmp_path,
    monkeypatch,
    frame,
    message,
):
    raw_path = tmp_path / "sp500tr_raw.csv"
    prepared_path = tmp_path / "benchmark_monthly.csv"
    frame.to_csv(raw_path, index=False)
    frame.to_csv(prepared_path, index=False)
    paths = SimpleNamespace(
        benchmark_raw_csv=raw_path,
        benchmark_csv=prepared_path,
    )
    config = BacktestBenchmarkConfig(
        coverage_start_date=date(2024, 1, 1),
        coverage_end_date=date(2024, 1, 31),
    )
    monkeypatch.setattr(
        acquisition_planning,
        "DEFAULT_CONFIG",
        SimpleNamespace(paths=paths, benchmark=config),
    )

    with pytest.raises(ValueError, match=message):
        acquisition_planning.benchmark_readiness_record(
            contributing_source="supplied"
        )
    with pytest.raises(ValueError, match=message):
        prepare_benchmark_monthly(config, paths)
    with pytest.raises(ValueError, match=message):
        benchmark.load_benchmark(config, paths)


def test_missing_february_2024_is_uncovered_across_all_stage_boundaries(
    tmp_path,
    monkeypatch,
):
    raw_path = tmp_path / "sp500tr_raw.csv"
    prepared_path = tmp_path / "benchmark_monthly.csv"
    frame = pd.DataFrame({
        "Date": ["2024-01-31"],
        "SP500TR_Close": [10.0],
    })
    frame.to_csv(raw_path, index=False)
    frame.to_csv(prepared_path, index=False)
    paths = SimpleNamespace(
        benchmark_raw_csv=raw_path,
        benchmark_csv=prepared_path,
    )
    config = BacktestBenchmarkConfig(
        coverage_start_date=date(2024, 1, 1),
        coverage_end_date=date(2024, 2, 28),
    )
    monkeypatch.setattr(
        acquisition_planning,
        "DEFAULT_CONFIG",
        SimpleNamespace(paths=paths, benchmark=config),
    )
    monkeypatch.setattr(
        acquisition_planning,
        "utc_timestamp",
        lambda: "2026-08-26T00:00:00Z",
    )

    readiness = acquisition_planning.benchmark_readiness_record(
        contributing_source="supplied"
    )
    assert readiness.status is ReadinessStatus.PARTIAL
    assert readiness.required_count == 2
    assert readiness.covered_count == 1
    assert readiness.missing_dates == ("2024-02-29",)
    assert readiness.checked_at_utc == "2026-08-26T00:00:00Z"
    with pytest.raises(ValueError, match="missing.*2024-02-29"):
        prepare_benchmark_monthly(config, paths)
    with pytest.raises(ValueError, match="missing.*2024-02-29"):
        benchmark.load_benchmark(config, paths)
