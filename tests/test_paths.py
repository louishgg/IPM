"""Tests for pure, typed repository path derivation."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import backtest
from backtest.paths import DEFAULT_PATHS, BacktestPaths
from live.paths import LivePaths


def test_default_paths_use_the_normalized_repository_layout():
    package_root = Path(backtest.__file__).resolve().parent
    project_root = package_root.parent
    paths = DEFAULT_PATHS

    assert paths.root == package_root
    assert paths.data_dir == project_root / "data/backtest"
    assert paths.raw_data_dir == project_root / "data/backtest/raw"
    assert paths.prepared_data_dir == project_root / "data/backtest/prepared"
    assert paths.prices_csv == (
        project_root
        / "data/shared/supplied/reuters/SP500_Full_2014_2026_Cleaned.csv"
    )
    assert paths.shared_provenance_dir == project_root / "data/shared/provenance"
    assert paths.membership.directory == (
        paths.shared_provenance_dir / "sp500_membership"
    )
    assert paths.security_identity.directory == (
        paths.shared_provenance_dir / "security_identity"
    )
    assert paths.sector_history.directory == (
        paths.shared_provenance_dir / "sector_history"
    )
    assert paths.sector_history.snapshots_csv == (
        paths.shared_provenance_dir
        / "sector_history"
        / "wikipedia_sector_snapshots.csv"
    )
    assert paths.sector_history.notices_csv == (
        paths.shared_provenance_dir
        / "sector_history"
        / "sp500_constituent_notices.csv"
    )
    assert paths.sector_assignments_csv == (
        paths.prepared_data_dir / "sector_assignments.csv"
    )
    assert paths.output_base_dir == project_root / "outputs/backtest"


def test_share_and_attribution_paths_have_separate_owners():
    base = DEFAULT_PATHS
    paths = base.for_strategy("reversal")
    shares = base.shares
    attribution = paths.attribution

    assert shares.raw_dir == paths.raw_data_dir / "shares"
    assert shares.prepared_dir == paths.prepared_data_dir
    assert shares.raw_shares_csv == (
        shares.raw_dir / "yahoo_shares_outstanding_raw.csv"
    )
    assert shares.acquisition_status_csv == shares.raw_dir / "acquisition_status.csv"
    assert shares.artifact_manifest_csv == shares.raw_dir / "artifact_manifest.csv"
    assert shares.readiness_csv == shares.raw_dir / "readiness.csv"
    assert paths.raw_provenance_dir == paths.raw_data_dir / "provenance"
    assert shares.reviewed_dir == paths.raw_provenance_dir / "shares"
    assert shares.final_shares_csv == (
        shares.prepared_dir / "shares_outstanding_monthly.csv"
    )

    assert attribution.tables_dir == paths.tables_dir / "brinson"
    assert attribution.figures_dir == paths.figures_dir / "brinson"
    assert paths.brinson_holdings_gross_csv == (
        attribution.tables_dir / "portfolio_holdings_gross.csv"
    )
    assert attribution.benchmark_audit_csv == (
        attribution.tables_dir / "benchmark_audit.csv"
    )
    assert attribution.period_attribution_csv == (
        attribution.tables_dir / "period_attribution_summary.csv"
    )
    assert attribution.benchmark_consistency_png == (
        attribution.figures_dir / "benchmark_consistency.png"
    )
    assert paths.brinson_holdings_gross_csv not in attribution.result_files


def test_generated_backtest_output_paths_are_normalized():
    paths = DEFAULT_PATHS.for_strategy("reversal")
    assert paths.performance_summary_csv == paths.tables_dir / "performance_summary.csv"
    assert paths.transaction_cost_summary_csv == (
        paths.tables_dir / "transaction_cost_summary.csv"
    )
    assert paths.strategy_tables_dir == paths.tables_dir / "strategy"
    assert paths.strategy_decisions_csv == (
        paths.strategy_tables_dir / "decisions.csv"
    )
    assert paths.strategy_trades_csv == paths.strategy_tables_dir / "trades.csv"
    assert paths.strategy_nav_csv == paths.strategy_tables_dir / "nav.csv"
    assert paths.cumulative_performance_png == (
        paths.figures_dir / "test_window/cumulative_performance.png"
    )
    assert paths.drawdown_png == paths.figures_dir / "test_window/drawdown.png"
    assert paths.monthly_returns_png == (
        paths.figures_dir / "test_window/monthly_returns.png"
    )


def test_strategy_outputs_are_automatically_isolated(tmp_path):
    base = BacktestPaths(tmp_path / "backtest")
    reversal = base.for_strategy("reversal")
    trend = base.for_strategy("monthly_trend")
    plain = base.for_strategy("momentum")

    assert reversal.prepared_data_dir == trend.prepared_data_dir == plain.prepared_data_dir
    assert trend.performance_summary_csv == (
        tmp_path / "outputs/backtest/strategies/monthly_trend/tables/performance_summary.csv"
    )
    assert len({paths.strategy_output_dir for paths in (reversal, trend, plain)}) == 3
    assert len({paths.attribution.tables_dir for paths in (reversal, trend, plain)}) == 3
    assert reversal.performance_summary_csv == (
        tmp_path
        / "outputs/backtest/strategies/reversal/tables"
        / "performance_summary.csv"
    )
    assert plain.performance_summary_csv == (
        tmp_path
        / "outputs/backtest/strategies/momentum/tables"
        / "performance_summary.csv"
    )
    assert reversal.strategy_parameters_json.parent == reversal.strategy_output_dir


def test_live_strategy_outputs_are_automatically_isolated(tmp_path):
    base = LivePaths(tmp_path / "live")
    reversal = base.for_strategy("reversal")
    trend = base.for_strategy("monthly_trend")
    plain = base.for_strategy("momentum")

    assert reversal.prepared_data_dir == trend.prepared_data_dir == plain.prepared_data_dir
    assert trend.strategy_performance_csv == (
        tmp_path / "outputs/live/strategies/monthly_trend/tables/strategy/performance.csv"
    )
    assert len({paths.strategy_output_dir for paths in (reversal, trend, plain)}) == 3
    assert len({paths.attribution.tables_dir for paths in (reversal, trend, plain)}) == 3
    assert reversal.strategy_performance_csv == (
        tmp_path
        / "outputs/live/strategies/reversal/tables/strategy"
        / "performance.csv"
    )
    assert plain.attribution.tables_dir == (
        tmp_path
        / "outputs/live/strategies/momentum/tables/brinson"
    )


def test_live_paths_use_prices_shares_strategy_and_attribution_owners(tmp_path):
    base = LivePaths(tmp_path / "live")
    paths = base.for_strategy("momentum")

    assert base.data_dir == tmp_path / "data/live"
    assert base.raw_prices_dir == tmp_path / "data/live/raw/prices"
    assert base.shares.raw_dir == tmp_path / "data/live/raw/shares"
    assert base.shares.prepared_dir == tmp_path / "data/live/prepared"
    assert base.sector_history.directory == (
        tmp_path / "data/shared/provenance/sector_history"
    )
    assert base.sector_assignments_csv == (
        tmp_path / "data/live/prepared/sector_assignments.csv"
    )
    assert paths.strategy_tables_dir == (
        tmp_path / "outputs/live/strategies/momentum/tables/strategy"
    )
    assert paths.cumulative_performance_png == (
        paths.strategy_output_dir / "figures/live_window/cumulative_performance.png"
    )
    assert paths.drawdown_png == (
        paths.strategy_output_dir / "figures/live_window/drawdown.png"
    )
    assert paths.daily_returns_png == (
        paths.strategy_output_dir / "figures/live_window/daily_returns.png"
    )
    assert paths.attribution.tables_dir == (
        tmp_path / "outputs/live/strategies/momentum/tables/brinson"
    )
    assert paths.attribution.figures_dir == (
        tmp_path / "outputs/live/strategies/momentum/figures/brinson"
    )


def test_path_derivation_has_no_filesystem_side_effects(tmp_path):
    project_root = tmp_path / "missing"
    base = BacktestPaths(project_root / "backtest")
    paths = base.for_strategy("momentum")

    assert not project_root.exists()
    _ = base.membership.components_csv
    _ = base.shares.final_shares_csv
    _ = paths.attribution.period_attribution_csv
    assert not project_root.exists()


@pytest.mark.parametrize("paths", (DEFAULT_PATHS, LivePaths.from_package()))
def test_unscoped_paths_reject_strategy_outputs(paths):
    with pytest.raises(ValueError, match="explicit strategy ID"):
        _ = paths.strategy_output_dir
    with pytest.raises(ValueError, match="explicit strategy ID"):
        _ = paths.attribution


def test_paths_are_frozen(tmp_path):
    paths = BacktestPaths(tmp_path)
    with pytest.raises(FrozenInstanceError):
        paths.root = tmp_path / "other"
