"""Typed filesystem layout for the CSV-based backtest pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from portfolio_core.sp500_membership import MembershipSourcePaths
from portfolio_core.sector_evidence import SectorHistoryPaths
from portfolio_core.security_identity import SecurityIdentityPaths


GRIDS_DIRECTORY = Path(__file__).resolve().with_name("grids")


@dataclass(frozen=True, slots=True)
class BacktestSharesPaths:
    """Acquired, reviewed, and prepared paths owned by the shares dataset."""

    raw_dir: Path
    prepared_dir: Path
    reviewed_dir: Path

    @property
    def raw_shares_csv(self) -> Path:
        return self.raw_dir / "yahoo_shares_outstanding_raw.csv"

    @property
    def acquisition_status_csv(self) -> Path:
        return self.raw_dir / "acquisition_status.csv"

    @property
    def artifact_manifest_csv(self) -> Path:
        return self.raw_dir / "artifact_manifest.csv"

    @property
    def readiness_csv(self) -> Path:
        return self.raw_dir / "readiness.csv"

    @property
    def final_shares_csv(self) -> Path:
        return self.prepared_dir / "shares_outstanding_monthly.csv"


@dataclass(frozen=True, slots=True)
class BacktestPricePaths:
    """Backtest-owned fallback-price capture paths."""

    raw_dir: Path

    @property
    def yahoo_close_csv(self) -> Path:
        return self.raw_dir / "yahoo_close.csv"

    @property
    def wiki_extract_csv(self) -> Path:
        return self.raw_dir / "wiki_prices_ten_identity_extract.csv"

    @property
    def acquisition_status_csv(self) -> Path:
        return self.raw_dir / "acquisition_status.csv"

    @property
    def readiness_csv(self) -> Path:
        return self.raw_dir / "readiness.csv"

    @property
    def artifact_manifest_csv(self) -> Path:
        return self.raw_dir / "artifact_manifest.csv"


@dataclass(frozen=True, slots=True)
class BacktestAttributionPaths:
    """Tables and figures produced by Brinson attribution."""

    tables_dir: Path
    figures_dir: Path

    @property
    def benchmark_sector_csv(self) -> Path:
        return self.tables_dir / "benchmark_sector_series.csv"

    @property
    def benchmark_audit_csv(self) -> Path:
        return self.tables_dir / "benchmark_audit.csv"

    @property
    def sector_attribution_csv(self) -> Path:
        return self.tables_dir / "sector_attribution.csv"

    @property
    def monthly_attribution_csv(self) -> Path:
        return self.tables_dir / "monthly_attribution_summary.csv"

    @property
    def period_attribution_csv(self) -> Path:
        return self.tables_dir / "period_attribution_summary.csv"

    @property
    def benchmark_consistency_png(self) -> Path:
        return self.figures_dir / "benchmark_consistency.png"

    @property
    def active_decomposition_monthly_png(self) -> Path:
        return self.figures_dir / "active_decomposition_monthly.png"

    @property
    def active_decomposition_cumulative_png(self) -> Path:
        return self.figures_dir / "active_decomposition_cumulative.png"

    @property
    def result_files(self) -> tuple[Path, ...]:
        """Generated attribution outputs, excluding strategy-owned holdings."""
        return (
            self.benchmark_sector_csv,
            self.benchmark_audit_csv,
            self.sector_attribution_csv,
            self.monthly_attribution_csv,
            self.period_attribution_csv,
            self.benchmark_consistency_png,
            self.active_decomposition_monthly_png,
            self.active_decomposition_cumulative_png,
        )


@dataclass(frozen=True, slots=True)
class BacktestPaths:
    """All durable backtest paths derived from the repository root."""

    root: Path
    strategy_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if self.strategy_id is not None:
            normalized = str(self.strategy_id).strip()
            if not normalized:
                raise ValueError("strategy_id cannot be empty")
            object.__setattr__(self, "strategy_id", normalized)

    @classmethod
    def from_package(cls) -> "BacktestPaths":
        return cls(Path(__file__).resolve().parent)

    @property
    def project_root(self) -> Path:
        return self.root.parent

    def for_strategy(self, strategy_id: str) -> "BacktestPaths":
        """Mirror canonical data paths under one strategy-owned output root."""
        return BacktestPaths(self.root, strategy_id=strategy_id)

    @property
    def output_base_dir(self) -> Path:
        return self.project_root / "outputs" / "backtest"

    @property
    def strategy_output_dir(self) -> Path:
        if self.strategy_id is None:
            raise ValueError(
                "Backtest strategy output paths require an explicit strategy ID"
            )
        return self.output_base_dir / "strategies" / self.strategy_id

    @property
    def shared_provenance_dir(self) -> Path:
        return self.project_root / "data" / "shared" / "provenance"

    @property
    def membership(self) -> MembershipSourcePaths:
        return MembershipSourcePaths(
            self.shared_provenance_dir / "sp500_membership"
        )

    @property
    def security_identity(self) -> SecurityIdentityPaths:
        return SecurityIdentityPaths.from_root(self.project_root)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data" / "backtest"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def raw_provenance_dir(self) -> Path:
        return self.raw_data_dir / "provenance"

    @property
    def prepared_data_dir(self) -> Path:
        return self.data_dir / "prepared"

    @property
    def tables_dir(self) -> Path:
        return self.strategy_output_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.strategy_output_dir / "figures"

    @property
    def prices_csv(self) -> Path:
        return (
            self.project_root
            / "data"
            / "shared"
            / "supplied"
            / "reuters"
            / "SP500_Full_2014_2026_Cleaned.csv"
        )

    @property
    def benchmark_raw_csv(self) -> Path:
        return self.raw_data_dir / "sp500tr_raw.csv"

    @property
    def benchmark_provenance_dir(self) -> Path:
        return self.raw_data_dir / "benchmark"

    @property
    def benchmark_readiness_csv(self) -> Path:
        return self.benchmark_provenance_dir / "readiness.csv"

    @property
    def benchmark_acquisition_status_csv(self) -> Path:
        return self.benchmark_provenance_dir / "acquisition_status.csv"

    @property
    def benchmark_artifact_manifest_csv(self) -> Path:
        return self.benchmark_provenance_dir / "artifact_manifest.csv"

    @property
    def sector_history(self) -> SectorHistoryPaths:
        return SectorHistoryPaths.from_project_root(self.project_root)

    @property
    def prices_artifact_manifest_csv(self) -> Path:
        return self.prices_csv.parent / "artifact_manifest.csv"

    @property
    def prices_monthly_csv(self) -> Path:
        return self.prepared_data_dir / "prices_monthly.csv"

    @property
    def price_sources(self) -> BacktestPricePaths:
        return BacktestPricePaths(
            raw_dir=self.raw_data_dir / "prices",
        )

    @property
    def pit_membership_csv(self) -> Path:
        return self.prepared_data_dir / "pit_membership.csv"

    @property
    def asset_metadata_csv(self) -> Path:
        return self.prepared_data_dir / "asset_metadata.csv"

    @property
    def sector_assignments_csv(self) -> Path:
        return self.prepared_data_dir / "sector_assignments.csv"

    @property
    def ticker_ric_resolution_csv(self) -> Path:
        return self.prepared_data_dir / "ticker_ric_resolution.csv"

    @property
    def security_identity_validation_csv(self) -> Path:
        return self.prepared_data_dir / "security_identity_validation.csv"

    @property
    def security_events_prepared_csv(self) -> Path:
        return self.prepared_data_dir / "security_events.csv"

    @property
    def security_event_legs_prepared_csv(self) -> Path:
        return self.prepared_data_dir / "security_event_legs.csv"

    @property
    def security_event_sources_prepared_csv(self) -> Path:
        return self.prepared_data_dir / "security_event_sources.csv"

    @property
    def security_event_crossing_audit_csv(self) -> Path:
        return self.prepared_data_dir / "security_event_crossing_audit.csv"

    @property
    def membership_coverage_audit_csv(self) -> Path:
        return self.prepared_data_dir / "membership_coverage_audit.csv"

    @property
    def benchmark_csv(self) -> Path:
        return self.prepared_data_dir / "benchmark_monthly.csv"

    @property
    def preparation_manifest_csv(self) -> Path:
        return self.prepared_data_dir / "preparation_manifest.csv"

    @property
    def price_basis_csv(self) -> Path:
        return self.prepared_data_dir / "price_basis.csv"

    @property
    def transaction_cost_summary_csv(self) -> Path:
        return self.tables_dir / "transaction_cost_summary.csv"

    @property
    def spread_sensitivity_csv(self) -> Path:
        return self.tables_dir / "spread_sensitivity.csv"

    @property
    def strategy_tables_dir(self) -> Path:
        return self.tables_dir / "strategy"

    @property
    def strategy_decisions_csv(self) -> Path:
        return self.strategy_tables_dir / "decisions.csv"

    @property
    def strategy_trades_csv(self) -> Path:
        return self.strategy_tables_dir / "trades.csv"

    @property
    def strategy_research_diagnostics_csv(self) -> Path:
        return self.strategy_tables_dir / "research_diagnostics.csv"


    @property
    def strategy_sector_residuals_csv(self) -> Path:
        return self.strategy_tables_dir / "sector_residuals.csv"

    @property
    def strategy_nav_csv(self) -> Path:
        return self.strategy_tables_dir / "nav.csv"

    @property
    def brinson_holdings_gross_csv(self) -> Path:
        """Strategy-owned gross holdings consumed by Brinson attribution."""
        return self.tables_dir / "brinson" / "portfolio_holdings_gross.csv"

    @property
    def performance_summary_csv(self) -> Path:
        return self.tables_dir / "performance_summary.csv"

    @property
    def strategy_parameters_json(self) -> Path:
        return self.strategy_output_dir / "strategy_parameters.json"

    @property
    def simulation_assumptions_json(self) -> Path:
        return self.strategy_output_dir / "simulation_assumptions.json"

    @property
    def security_event_accounting_audit_csv(self) -> Path:
        return self.tables_dir / "security_event_accounting_audit.csv"

    @property
    def cumulative_performance_png(self) -> Path:
        return self.figures_dir / "test_window" / "cumulative_performance.png"

    @property
    def drawdown_png(self) -> Path:
        return self.figures_dir / "test_window" / "drawdown.png"

    @property
    def monthly_returns_png(self) -> Path:
        return self.figures_dir / "test_window" / "monthly_returns.png"

    @property
    def shares(self) -> BacktestSharesPaths:
        return BacktestSharesPaths(
            raw_dir=self.raw_data_dir / "shares",
            prepared_dir=self.prepared_data_dir,
            reviewed_dir=self.raw_provenance_dir / "shares",
        )

    @property
    def attribution(self) -> BacktestAttributionPaths:
        return BacktestAttributionPaths(
            tables_dir=self.tables_dir / "brinson",
            figures_dir=self.figures_dir / "brinson",
        )


DEFAULT_PATHS = BacktestPaths.from_package()


__all__ = [
    "BacktestAttributionPaths",
    "BacktestPaths",
    "BacktestPricePaths",
    "BacktestSharesPaths",
    "DEFAULT_PATHS",
    "GRIDS_DIRECTORY",
]
