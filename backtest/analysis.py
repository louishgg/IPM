"""Run explicit backtest and attribution stages from prepared CSV inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from portfolio_core.runtime_config import apply_runtime_settings
from portfolio_core.strategies import (
    Strategy, read_saved_strategy_identity, save_strategy_parameters,
)
from portfolio_core.simulation_assumptions import write_simulation_assumptions

from .benchmark import (
    calculate_performance_summary,
    load_benchmark,
    save_performance_plots,
    save_performance_summary,
)
from .brinson_attribution import (
    remove_saved_brinson_result,
    run_brinson_attribution,
    save_brinson_result,
    save_gross_brinson_holdings,
)
from .shares_preparation import load_prepared_shares, validate_share_coverage
from .config import BacktestConfig
from .data_loading import BacktestDataset, load_backtest_data
from .evaluation import (
    BacktestAnalysisResult,
    run_strategy_evaluation,
    calculate_spread_sensitivity,
    save_evaluation_summary_tables,
    save_strategy_audit_tables,
)


@dataclass(frozen=True)
class BacktestStrategyArtifacts:
    """Completed numerical strategy artifacts awaiting direct persistence."""

    result: BacktestAnalysisResult
    benchmark_monthly: pd.Series
    performance_summary: pd.DataFrame


def calculate_strategy_artifacts(
    backtest_data: BacktestDataset,
    benchmark_monthly: pd.Series,
    config: BacktestConfig,
) -> BacktestStrategyArtifacts:
    """Calculate every numerical strategy artifact before writing any output."""
    print("\nRunning backtest analysis from prepared CSV data...")
    result = run_strategy_evaluation(
        backtest_data,
        config.strategy,
        evaluation_config=config.evaluation,
        accounting_config=config.accounting,
    )
    result = replace(
        result,
        spread_sensitivity=calculate_spread_sensitivity(
            backtest_data,
            result,
        ),
    )
    performance_summary = calculate_performance_summary(
        result,
        benchmark_monthly,
    )
    if performance_summary.empty:
        raise RuntimeError("Backtest performance calculation produced no rows")
    return BacktestStrategyArtifacts(
        result=result,
        benchmark_monthly=benchmark_monthly,
        performance_summary=performance_summary,
    )


def save_strategy_artifacts(
    artifacts: BacktestStrategyArtifacts,
    config: BacktestConfig,
) -> None:
    """Write a completed strategy result directly to its final folder."""
    paths = config.paths
    save_evaluation_summary_tables(artifacts.result, paths)
    save_strategy_audit_tables(artifacts.result, paths)
    save_performance_summary(artifacts.performance_summary, paths)
    save_performance_plots(
        artifacts.result,
        artifacts.benchmark_monthly,
        paths,
    )
    save_gross_brinson_holdings(artifacts.result.test_holdings_gross, paths)
    save_strategy_parameters(
        artifacts.result.strategy,
        paths.strategy_parameters_json,
    )
    write_simulation_assumptions(
        paths.simulation_assumptions_json,
        artifacts.result.accounting_config,
        artifacts.result.price_basis,
    )


def run_analysis(stage: str, *, config: BacktestConfig):
    """Execute one prepared-only stage for an explicitly configured strategy."""
    if stage not in {"strategy", "brinson", "all"}:
        raise ValueError(f"Unsupported backtest analysis stage: {stage!r}")
    if stage != "brinson" and not isinstance(config.strategy, Strategy):
        raise ValueError(
            "Backtest analysis requires an explicitly resolved strategy"
        )

    apply_runtime_settings()

    if stage == "strategy":
        backtest_data = load_backtest_data(config.market, config.paths)
        benchmark_monthly = load_benchmark(config.benchmark, config.paths)
        artifacts = calculate_strategy_artifacts(
            backtest_data,
            benchmark_monthly,
            config,
        )
        remove_saved_brinson_result(config.paths)
        save_strategy_artifacts(artifacts, config)
        return artifacts

    if stage == "brinson":
        read_saved_strategy_identity(
            config.paths.strategy_id, config.paths.strategy_parameters_json,
        )

    backtest_data = load_backtest_data(
        config.market,
        config.paths,
        include_brinson=True,
    )
    shares_data = load_prepared_shares(config.paths.shares, config.brinson)
    validate_share_coverage(backtest_data, shares_data, config.brinson)

    if stage == "brinson":
        attribution_result = run_brinson_attribution(
            backtest_data,
            shares_monthly=shares_data,
            paths=config.paths,
        )
        save_brinson_result(attribution_result, config.paths)
        return attribution_result

    benchmark_monthly = load_benchmark(config.benchmark, config.paths)
    artifacts = calculate_strategy_artifacts(
        backtest_data,
        benchmark_monthly,
        config,
    )
    attribution_result = run_brinson_attribution(
        backtest_data,
        holdings_df=artifacts.result.test_holdings_gross,
        shares_monthly=shares_data,
        paths=config.paths,
    )
    save_strategy_artifacts(artifacts, config)
    save_brinson_result(attribution_result, config.paths)
    return artifacts, attribution_result


__all__ = [
    "BacktestStrategyArtifacts",
    "calculate_strategy_artifacts",
    "run_analysis",
    "save_strategy_artifacts",
]
