"""Benchmark integration, performance summary, and benchmark-relative plots."""
import pandas as pd
from portfolio_core.performance_metrics import calculate_risk_metrics
from portfolio_core.io import atomic_write_dataframe
from portfolio_core.performance_plotting import render_performance_plots

from .config import BacktestBenchmarkConfig, DEFAULT_CONFIG
from .data_loading import (
    PREPARE_CORE_COMMAND,
    VALIDATE_BENCHMARK_COMMAND,
    load_prepared_benchmark,
)
from .evaluation import BacktestAnalysisResult, evaluation_period_label
from .paths import BacktestPaths


def load_benchmark(
    benchmark_config: BacktestBenchmarkConfig = DEFAULT_CONFIG.benchmark,
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> pd.Series:
    """Load and validate the prepared monthly S&P 500 Total Return CSV.

    Analysis is intentionally a strict consumer of prepared data.  It never
    falls back to a pickle, downloads data, or prepares a missing artifact.
    ``benchmark_config`` defines the required monthly coverage.
    """
    print("\n--- S&P 500 Total Return Benchmark ---")

    if not paths.benchmark_csv.exists():
        raise FileNotFoundError(
            f"Missing prepared benchmark CSV: {paths.benchmark_csv}. "
            f"Run `{PREPARE_CORE_COMMAND}` before analysis. If the "
            "raw benchmark is also missing, run "
            f"`{VALIDATE_BENCHMARK_COMMAND}` first."
        )

    print(f"Loading prepared benchmark from CSV: {paths.benchmark_csv}")
    benchmark = load_prepared_benchmark(paths.benchmark_csv)

    required_dates = benchmark_config.required_dates
    available_dates = pd.DatetimeIndex(benchmark.index)
    missing_dates = required_dates.difference(available_dates)
    if len(missing_dates):
        raise ValueError(
            "Prepared benchmark does not cover the configured coverage window; "
            f"missing {missing_dates.strftime('%Y-%m-%d').tolist()}. "
            "Reacquire or correct the canonical Yahoo benchmark with "
            f"`{VALIDATE_BENCHMARK_COMMAND}`, then run "
            f"`{PREPARE_CORE_COMMAND}`."
        )

    return benchmark


def calculate_advanced_metrics(
    portfolio_series,
    bench_series,
    name="Period",
    cash_interest_rate: float = DEFAULT_CONFIG.accounting.cash_interest_rate,
):
    """Compute advanced risk/benchmark metrics from monthly NAV series.

    Conventions:
    - Annualized return is geometric.
    - Volatility is monthly return volatility annualized by sqrt(12).
    - Sharpe is monthly excess-return Sharpe annualized by sqrt(12).
    - Beta is estimated from monthly excess returns and is not annualized.
    - Treynor annualizes the average monthly excess return, then divides by beta.
    - Information ratio is monthly active-return IR annualized by sqrt(12).
    - CAPM alpha is the monthly regression intercept annualized by multiplying by 12.
    """

    # Align dates strictly
    common_dates = portfolio_series.index.intersection(bench_series.index)
    port_aligned = portfolio_series.loc[common_dates].dropna()
    bench_aligned = bench_series.loc[common_dates].dropna()
    common_dates = port_aligned.index.intersection(bench_aligned.index)
    port_aligned = port_aligned.loc[common_dates]
    bench_aligned = bench_aligned.loc[common_dates]

    months = len(port_aligned) - 1
    if months <= 0:
        return None

    # Portfolio and benchmark geometric annualized returns
    port_cum = (port_aligned.iloc[-1] / port_aligned.iloc[0]) - 1
    port_ann = (1 + port_cum) ** (12 / months) - 1
    bench_cum = (bench_aligned.iloc[-1] / bench_aligned.iloc[0]) - 1
    bench_ann = (1 + bench_cum) ** (12 / months) - 1
    excess_return_ann_geo = port_ann - bench_ann

    # Monthly return series
    port_pct = port_aligned.pct_change(fill_method=None).dropna()
    bench_pct = bench_aligned.pct_change(fill_method=None).dropna()
    common_ret_dates = port_pct.index.intersection(bench_pct.index)
    port_pct = port_pct.loc[common_ret_dates]
    bench_pct = bench_pct.loc[common_ret_dates]

    if len(port_pct) < 2:
        return None

    # Preserve the monthly adapter and historical undefined-ratio convention.
    rf_monthly = (1 + cash_interest_rate) ** (1 / 12) - 1
    risk = calculate_risk_metrics(
        port_pct, bench_pct, pd.Series(rf_monthly, index=port_pct.index),
        periods_per_year=12, zero_variance_value=0.0,
    )

    return {
        "Period": name,
        "Months": int(months),
        "Portfolio_Cum_Return": float(port_cum),
        "Benchmark_Cum_Return": float(bench_cum),
        "Portfolio_Ann_Return": float(port_ann),
        "Benchmark_Ann_Return": float(bench_ann),
        "Portfolio_Ann_Vol": risk["Portfolio_Ann_Vol"],
        "Benchmark_Ann_Vol": risk["Benchmark_Ann_Vol"],
        "Excess_Return_Geo_Ann": float(excess_return_ann_geo),
        **{key: value for key, value in risk.items() if key not in {"Portfolio_Ann_Vol", "Benchmark_Ann_Vol"}},
    }


def calculate_performance_summary(
    results: BacktestAnalysisResult,
    bench_monthly: pd.Series,
) -> pd.DataFrame:
    """Compute net/gross performance rows without writing outputs."""
    if bench_monthly.empty:
        raise RuntimeError("Benchmark data unavailable.")

    periods = (
        (
            "development",
            evaluation_period_label(
                "Development Period", results.development_dates
            ),
            results.development_results,
            results.development_results_gross,
        ),
        (
            "test_window",
            evaluation_period_label("Test Window", results.test_dates),
            results.test_results,
            results.test_results_gross,
        ),
        (
            "full",
            evaluation_period_label(
                "Full Backtest Window", results.full_results.index
            ),
            results.full_results,
            results.full_results_gross,
        ),
    )
    performance_summary_rows = []
    for cost_treatment in ("Net", "Gross"):
        for period_id, label, net_series, gross_series in periods:
            series = net_series if cost_treatment == "Net" else gross_series
            metrics = calculate_advanced_metrics(
                series,
                bench_monthly,
                name=f"{label} - {cost_treatment} of Costs",
                cash_interest_rate=results.accounting_config.cash_interest_rate,
            )
            if metrics is not None:
                metrics["Period_ID"] = period_id
                metrics["Cost_Treatment"] = cost_treatment
                metrics["Evaluation_Start"] = pd.Timestamp(series.index.min()).date().isoformat()
                metrics["Evaluation_End"] = pd.Timestamp(series.index.max()).date().isoformat()
                performance_summary_rows.append(metrics)

    performance_summary = pd.DataFrame(performance_summary_rows)
    if performance_summary.empty:
        return performance_summary

    performance_summary["Price_Basis"] = results.price_basis.price_basis_id
    performance_summary["Simulation_Fingerprint"] = results.simulation_fingerprint
    risk_free_label = f"{results.accounting_config.cash_interest_rate:.0%}"
    metric_notes = {
        "Period": "Notes",
        "Period_ID": "Stable machine-readable period identifier.",
        "Months": "Number of months in the period.",
        "Portfolio_Cum_Return": "Cumulative portfolio return over the period.",
        "Benchmark_Cum_Return": "Cumulative S&P 500 TR return over the period.",
        "Portfolio_Ann_Return": "Geometric annualized portfolio return.",
        "Benchmark_Ann_Return": "Geometric annualized S&P 500 TR return.",
        "Portfolio_Ann_Vol": "Annualized volatility of monthly portfolio returns.",
        "Benchmark_Ann_Vol": "Annualized volatility of monthly benchmark returns.",
        "Excess_Return_Geo_Ann": "Portfolio annualized return minus benchmark annualized return.",
        "Active_Return_Ann_Arith": "Mean monthly active return annualized by 12.",
        "Tracking_Error_Ann": "Annualized volatility of monthly active returns.",
        "Sharpe": f"Annualized excess return over {risk_free_label} risk-free rate per unit of volatility.",
        "Beta": "OLS market beta from monthly CAPM regression.",
        "Treynor": f"Annualized excess return over {risk_free_label} risk-free rate divided by beta.",
        "Information_Ratio": "Annualized active return per unit of tracking error.",
        "CAPM_Alpha_OLS_Ann": "Annualized intercept from monthly CAPM regression.",
        "Alpha_P_Value": "P-value for OLS alpha significance.",
        "Beta_P_Value": "P-value for OLS beta significance.",
        "Cost_Treatment": "Net includes fixed fees and spread/slippage costs; gross disables both cost deductions.",
        "Price_Basis": "Explicit prepared return and dividend-treatment basis.",
        "Simulation_Fingerprint": "SHA-256 fingerprint of run-defining simulation assumptions.",
    }
    performance_summary = pd.concat(
        [performance_summary, pd.DataFrame([metric_notes])],
        ignore_index=True,
    )

    return performance_summary


def save_performance_summary(
    performance_summary: pd.DataFrame,
    paths: BacktestPaths,
) -> None:
    """Persist an already-computed performance summary."""
    atomic_write_dataframe(
        performance_summary,
        paths.performance_summary_csv,
        index=False,
        date_format=None,
        float_format=None,
    )
    print(f"\nSaved performance summary to {paths.performance_summary_csv}")


def save_performance_plots(
    results: BacktestAnalysisResult,
    bench_monthly: pd.Series,
    paths: BacktestPaths,
) -> None:
    """Render net test-window NAV and benchmark at the same monthly endpoints."""
    strategy = results.test_results
    if strategy.index.has_duplicates or bench_monthly.index.has_duplicates:
        raise ValueError("Monthly performance levels require unique dates")
    levels = pd.DataFrame({
        "Strategy": strategy,
        "S&P 500 Total Return": bench_monthly.reindex(strategy.index),
    })
    render_performance_plots(
        levels, levels.pct_change(fill_method=None).iloc[1:],
        cumulative_path=paths.cumulative_performance_png,
        drawdown_path=paths.drawdown_png,
        returns_path=paths.monthly_returns_png,
        window_label="Test Window", return_frequency="monthly",
        window_start=strategy.index.min(), window_end=strategy.index.max(),
    )
