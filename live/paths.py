"""Typed filesystem layout for the historical live strategy pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from portfolio_core.sp500_membership import MembershipSourcePaths
from portfolio_core.sector_evidence import SectorHistoryPaths
from portfolio_core.security_identity import SecurityIdentityPaths


@dataclass(frozen=True, slots=True)
class LiveSharesPaths:
    """Raw and prepared paths owned by the live shares dataset."""

    raw_dir: Path
    prepared_dir: Path

    @property
    def raw_shares_csv(self) -> Path:
        return self.raw_dir / "yahoo_shares_outstanding_raw.csv"

    @property
    def acquisition_status_csv(self) -> Path:
        return self.raw_dir / "acquisition_status.csv"

    @property
    def readiness_csv(self) -> Path:
        return self.raw_dir / "readiness.csv"

    @property
    def artifact_manifest_csv(self) -> Path:
        return self.raw_dir / "artifact_manifest.csv"

    @property
    def prepared_shares_csv(self) -> Path:
        return self.prepared_dir / "shares_outstanding_asof.csv"


@dataclass(frozen=True, slots=True)
class LiveAttributionPaths:
    """Tables and figures produced by live Brinson attribution."""

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
    def account_reconciliation_csv(self) -> Path:
        return self.tables_dir / "account_reconciliation.csv"

    @property
    def benchmark_consistency_png(self) -> Path:
        return self.figures_dir / "benchmark_consistency.png"

    @property
    def active_decomposition_period_png(self) -> Path:
        return self.figures_dir / "active_decomposition_period.png"

    @property
    def active_decomposition_cumulative_png(self) -> Path:
        return self.figures_dir / "active_decomposition_cumulative.png"

    @property
    def result_files(self) -> tuple[Path, ...]:
        """Every generated live attribution table and plot."""
        return (
            self.benchmark_sector_csv,
            self.benchmark_audit_csv,
            self.sector_attribution_csv,
            self.monthly_attribution_csv,
            self.period_attribution_csv,
            self.account_reconciliation_csv,
            self.benchmark_consistency_png,
            self.active_decomposition_period_png,
            self.active_decomposition_cumulative_png,
        )


@dataclass(frozen=True, slots=True)
class LivePaths:
    """All durable live paths derived from the repository root."""

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
    def from_package(cls) -> "LivePaths":
        return cls(Path(__file__).resolve().parent)

    @property
    def project_root(self) -> Path:
        return self.root.parent

    def for_strategy(self, strategy_id: str) -> "LivePaths":
        """Mirror canonical data paths under one strategy-owned output root."""
        return LivePaths(self.root, strategy_id=strategy_id)

    @property
    def output_base_dir(self) -> Path:
        return self.project_root / "outputs" / "live"

    @property
    def strategy_output_dir(self) -> Path:
        if self.strategy_id is None:
            raise ValueError(
                "Live strategy output paths require an explicit strategy ID"
            )
        return self.output_base_dir / "strategies" / self.strategy_id

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data" / "live"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def prepared_data_dir(self) -> Path:
        return self.data_dir / "prepared"

    @property
    def raw_provenance_dir(self) -> Path:
        return self.raw_data_dir / "provenance"

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
    def sector_history(self) -> SectorHistoryPaths:
        return SectorHistoryPaths.from_project_root(self.project_root)

    @property
    def raw_prices_dir(self) -> Path:
        return self.raw_data_dir / "prices"

    @property
    def raw_monthly_dir(self) -> Path:
        return self.raw_prices_dir / "monthly"

    @property
    def market_monthly_csv(self) -> Path:
        return self.prepared_data_dir / "market_monthly.csv"

    @property
    def monthly_dividends_csv(self) -> Path:
        return self.prepared_data_dir / "monthly_dividends.csv"

    @property
    def raw_price_close_csv(self) -> Path:
        return self.raw_prices_dir / "close.csv"

    @property
    def raw_price_open_csv(self) -> Path:
        return self.raw_prices_dir / "open.csv"

    @property
    def raw_price_volume_csv(self) -> Path:
        return self.raw_prices_dir / "volume.csv"

    @property
    def raw_price_status_csv(self) -> Path:
        return self.raw_prices_dir / "acquisition_status.csv"

    @property
    def raw_price_readiness_csv(self) -> Path:
        return self.raw_prices_dir / "readiness.csv"

    @property
    def raw_price_artifact_manifest_csv(self) -> Path:
        return self.raw_prices_dir / "artifact_manifest.csv"

    @property
    def raw_price_requirements_csv(self) -> Path:
        return self.raw_prices_dir / "requirements.csv"

    @property
    def raw_price_supplemental_dir(self) -> Path:
        return self.raw_prices_dir / "supplemental"

    @property
    def raw_price_supplemental_csv(self) -> Path:
        return self.raw_price_supplemental_dir / "yahoo_ohlcv.csv"

    @property
    def raw_price_supplemental_manifest_csv(self) -> Path:
        return self.raw_price_supplemental_dir / "artifact_manifest.csv"

    @property
    def raw_corporate_action_policy_csv(self) -> Path:
        return self.raw_provenance_dir / "corporate_action_policy.csv"

    @property
    def raw_corporate_action_policy_manifest_csv(self) -> Path:
        return self.raw_provenance_dir / "corporate_action_policy_manifest.csv"

    @property
    def raw_benchmark_dir(self) -> Path:
        return self.raw_data_dir / "benchmark"

    @property
    def raw_benchmark_csv(self) -> Path:
        return self.raw_benchmark_dir / "sp500tr_ohlc.csv"

    @property
    def raw_benchmark_status_csv(self) -> Path:
        return self.raw_benchmark_dir / "acquisition_status.csv"

    @property
    def raw_benchmark_readiness_csv(self) -> Path:
        return self.raw_benchmark_dir / "readiness.csv"

    @property
    def raw_benchmark_artifact_manifest_csv(self) -> Path:
        return self.raw_benchmark_dir / "artifact_manifest.csv"

    @property
    def market_daily_csv(self) -> Path:
        return self.prepared_data_dir / "market_daily.csv"

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
    def decision_schedule_csv(self) -> Path:
        return self.prepared_data_dir / "decision_schedule.csv"

    @property
    def prepared_benchmark_csv(self) -> Path:
        return self.prepared_data_dir / "benchmark_daily.csv"

    @property
    def prepared_corporate_action_events_csv(self) -> Path:
        return self.prepared_data_dir / "corporate_action_events.csv"

    @property
    def prepared_corporate_action_legs_csv(self) -> Path:
        return self.prepared_data_dir / "corporate_action_legs.csv"

    @property
    def prepared_corporate_action_sources_csv(self) -> Path:
        return self.prepared_data_dir / "corporate_action_sources.csv"

    @property
    def prepared_corporate_action_policy_csv(self) -> Path:
        return self.prepared_data_dir / "corporate_action_policy.csv"

    @property
    def preparation_manifest_csv(self) -> Path:
        return self.prepared_data_dir / "preparation_manifest.csv"

    @property
    def price_basis_csv(self) -> Path:
        return self.prepared_data_dir / "price_basis.csv"

    @property
    def strategy_tables_dir(self) -> Path:
        return self.strategy_output_dir / "tables" / "strategy"

    @property
    def strategy_parameters_json(self) -> Path:
        return self.strategy_output_dir / "strategy_parameters.json"

    @property
    def simulation_assumptions_json(self) -> Path:
        return self.strategy_output_dir / "simulation_assumptions.json"

    @property
    def strategy_nav_csv(self) -> Path:
        return self.strategy_tables_dir / "nav.csv"

    @property
    def strategy_daily_nav_csv(self) -> Path:
        return self.strategy_tables_dir / "daily_nav.csv"

    @property
    def cumulative_performance_png(self) -> Path:
        return self.strategy_output_dir / "figures/live_window/cumulative_performance.png"

    @property
    def drawdown_png(self) -> Path:
        return self.strategy_output_dir / "figures/live_window/drawdown.png"

    @property
    def daily_returns_png(self) -> Path:
        return self.strategy_output_dir / "figures/live_window/daily_returns.png"

    @property
    def strategy_holdings_csv(self) -> Path:
        return self.strategy_tables_dir / "holdings.csv"

    @property
    def strategy_trades_csv(self) -> Path:
        return self.strategy_tables_dir / "trades.csv"

    @property
    def strategy_decisions_csv(self) -> Path:
        return self.strategy_tables_dir / "decisions.csv"

    @property
    def strategy_research_diagnostics_csv(self) -> Path:
        return self.strategy_tables_dir / "research_diagnostics.csv"


    @property
    def strategy_sector_residuals_csv(self) -> Path:
        return self.strategy_tables_dir / "sector_residuals.csv"

    @property
    def strategy_performance_csv(self) -> Path:
        return self.strategy_tables_dir / "performance.csv"

    @property
    def spread_sensitivity_csv(self) -> Path:
        return self.strategy_tables_dir / "spread_sensitivity.csv"

    @property
    def strategy_corporate_actions_csv(self) -> Path:
        return self.strategy_tables_dir / "corporate_actions.csv"

    @property
    def shares(self) -> LiveSharesPaths:
        return LiveSharesPaths(
            raw_dir=self.raw_data_dir / "shares",
            prepared_dir=self.prepared_data_dir,
        )

    @property
    def attribution(self) -> LiveAttributionPaths:
        return LiveAttributionPaths(
            tables_dir=self.strategy_output_dir / "tables" / "brinson",
            figures_dir=self.strategy_output_dir / "figures" / "brinson",
        )


DEFAULT_PATHS = LivePaths.from_package()


__all__ = [
    "DEFAULT_PATHS",
    "LiveAttributionPaths",
    "LivePaths",
    "LiveSharesPaths",
]
