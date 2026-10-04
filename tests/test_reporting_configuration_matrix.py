"""Synthetic reporting for every family and representative construction variants."""

from dataclasses import replace
from datetime import date
import json

import pandas as pd
import pytest

from backtest.benchmark import calculate_performance_summary
from backtest.analysis import BacktestStrategyArtifacts, save_strategy_artifacts
from backtest.config import DEFAULT_CONFIG as BACKTEST_CONFIG
from backtest.config import BacktestEvaluationConfig
from backtest.evaluation import run_strategy_evaluation
from backtest.paths import BacktestPaths
from live.analysis import run_strategy_analysis, save_strategy_result
from live.config import DEFAULT_CONFIG as LIVE_CONFIG
from live.paths import LivePaths
from live.research_history import prepare_live_research_history
from portfolio_core.strategies import build_registered_strategy
from portfolio_core.strategies.research_parameters import (
    BufferParameters, ExposureParameters, ResearchParameters, SignalParameters,
    SizingParameters, StockSelection, VolatilityProfile, WholeSectorSelection,
)
from test_engine_offline import make_synthetic_backtest_data
from test_live_research_history import _historical_inputs


SIGNALS = {
    "momentum": SignalParameters("momentum", 3, 1, None, None),
    "reversal": SignalParameters("reversal", 1, 0, None, None),
    "monthly_trend": SignalParameters("monthly_trend", None, None, None, 3),
    "low_volatility": SignalParameters(
        "low_volatility", None, None, VolatilityProfile(12, 6), None,
    ),
    "sector_momentum": SignalParameters("sector_momentum", 1, 0, None, None),
}
EVALUATION = BacktestEvaluationConfig(
    signal_warmup_start_date=date(2014, 1, 31),
    signal_warmup_end_date=date(2014, 12, 31),
    initial_research_start_date=date(2015, 1, 31),
    initial_research_end_date=date(2015, 6, 30),
    validation_start_date=date(2015, 7, 31),
    validation_end_date=date(2015, 12, 31),
    test_start_date=date(2016, 1, 31),
)


def _selected(family, variant):
    sector = family == "sector_momentum"
    parameters = ResearchParameters(
        signal=SIGNALS[family],
        selection=WholeSectorSelection(3, 3) if sector else StockSelection(10, 10),
        sizing=(
            SizingParameters("inverse_volatility", VolatilityProfile(12, 6))
            if variant == 1 else SizingParameters("equal", None)
        ),
        buffer=BufferParameters(variant != 0, 1.4 if variant != 0 else None),
        exposure=(
            ExposureParameters(1.5, 0.5)
            if variant == 2 else ExposureParameters(1.0, 0.5)
        ),
        sector_neutral=variant == 2 and not sector,
        turnover_threshold=0.01 if variant == 1 else (0.05 if variant == 2 and not sector else 0.0),
    )
    return build_registered_strategy(family, parameters.payload())


@pytest.fixture(scope="module")
def backtest_data():
    return make_synthetic_backtest_data()[0]


@pytest.fixture(scope="module")
def live_inputs():
    return _historical_inputs()


# Every family exercises both reporting adapters. Shared construction variants
# run through one stock family and the distinct whole-sector path.
CASES = [(family, 0) for family in SIGNALS] + [
    (family, variant) for family in ("momentum", "sector_momentum") for variant in (1, 2)
]


@pytest.mark.parametrize("family,variant", CASES)
def test_complete_packet_reports_in_both_domains(
    family, variant, backtest_data, live_inputs, tmp_path,
):
    strategy = _selected(family, variant)
    backtest = run_strategy_evaluation(
        backtest_data, strategy, evaluation_config=EVALUATION,
    )
    assert not backtest.test_results.empty
    assert not backtest.decisions.empty
    assert not backtest.trades.empty
    benchmark = pd.Series(
        100.0 * 1.01 ** pd.Series(range(len(backtest_data.valid_trading_days))).to_numpy(),
        index=backtest_data.valid_trading_days,
    )
    summary = calculate_performance_summary(backtest, benchmark)
    assert set(summary.Evaluation_Start) and set(summary.Evaluation_End)
    backtest_paths = BacktestPaths(tmp_path / "backtest").for_strategy(family)
    save_strategy_artifacts(
        BacktestStrategyArtifacts(backtest, benchmark, summary),
        replace(BACKTEST_CONFIG, strategy=strategy, paths=backtest_paths),
    )
    assert backtest_paths.strategy_decisions_csv.is_file()
    assert all(path.is_file() for path in (
        backtest_paths.cumulative_performance_png,
        backtest_paths.drawdown_png,
        backtest_paths.monthly_returns_png,
    ))
    assert backtest_paths.strategy_research_diagnostics_csv.is_file()
    assert not backtest.research_diagnostics.empty

    config = replace(
        LIVE_CONFIG, strategy=strategy,
        paths=LivePaths(tmp_path / "live").for_strategy(family),
    )
    history = prepare_live_research_history(live_inputs, strategy)
    live = run_strategy_analysis(live_inputs, config, research_history=history)
    assert live.decisions.Strategy_ID.eq(family).all()
    assert live.nav.End_NAV.notna().all()
    if family == "sector_momentum" and variant == 0:
        assert live.research_diagnostics.Selected_Long_Sectors_JSON.map(
            lambda value: len(json.loads(value))
        ).eq(3).all()
    save_strategy_result(live, config)
    assert config.paths.strategy_performance_csv.is_file()
    assert all(path.is_file() for path in (
        config.paths.cumulative_performance_png,
        config.paths.drawdown_png,
        config.paths.daily_returns_png,
    ))
    assert config.paths.strategy_research_diagnostics_csv.is_file()
    assert not live.research_diagnostics.empty
