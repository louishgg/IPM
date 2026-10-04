"""Backtest artifact specifications for the shared preparation manifest."""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd

from portfolio_core.artifacts import (
    ArtifactSpec,
    validate_preparation_manifest as _validate_shared_manifest,
    write_preparation_manifest as _write_shared_manifest,
)
from portfolio_core.sector_evidence import sector_analytical_evidence_catalog

from .config import DEFAULT_CONFIG
from .paths import BacktestPaths


PREPARE_ALL_COMMAND = "python -m backtest.prepare all"


def _artifact_specs(
    paths: BacktestPaths,
    include_brinson: bool,
) -> list[ArtifactSpec]:
    specs = [
        ArtifactSpec("raw", "fja_components", paths.membership.components_csv),
        ArtifactSpec("raw", "fja_changes", paths.membership.changes_csv),
        ArtifactSpec("raw", "fja_intervals", paths.membership.intervals_csv),
        ArtifactSpec("raw", "fja_source_manifest", paths.membership.manifest_csv),
        ArtifactSpec(
            "raw",
            "canonical_security_events",
            paths.security_identity.events_csv,
        ),
        ArtifactSpec(
            "raw",
            "canonical_security_event_legs",
            paths.security_identity.legs_csv,
        ),
        ArtifactSpec(
            "raw",
            "canonical_security_event_sources",
            paths.security_identity.sources_csv,
        ),
        ArtifactSpec(
            "raw",
            "provider_symbol_mappings",
            paths.security_identity.provider_mappings_csv,
        ),
        ArtifactSpec(
            "raw",
            "provider_row_equivalence",
            paths.security_identity.provider_row_equivalence_csv,
        ),
        ArtifactSpec(
            "raw",
            "security_identity_manifest",
            paths.security_identity.manifest_csv,
        ),
        ArtifactSpec("raw", "prices", paths.prices_csv),
        ArtifactSpec("raw", "benchmark", paths.benchmark_raw_csv),
        ArtifactSpec(
            "raw",
            "benchmark_acquisition_status",
            paths.benchmark_acquisition_status_csv,
        ),
        ArtifactSpec(
            "raw",
            "benchmark_readiness",
            paths.benchmark_readiness_csv,
        ),
        ArtifactSpec(
            "raw",
            "benchmark_artifact_manifest",
            paths.benchmark_artifact_manifest_csv,
        ),
        *[
            ArtifactSpec("raw", name, path)
            for name, path, _ in sector_analytical_evidence_catalog(
                paths.sector_history
            )
        ],
        ArtifactSpec("prepared", "prices_monthly", paths.prices_monthly_csv),
        ArtifactSpec("prepared", "pit_membership", paths.pit_membership_csv),
        ArtifactSpec("prepared", "asset_metadata", paths.asset_metadata_csv),
        ArtifactSpec(
            "prepared", "sector_assignments", paths.sector_assignments_csv
        ),
        ArtifactSpec(
            "prepared",
            "ticker_ric_resolution",
            paths.ticker_ric_resolution_csv,
        ),
        ArtifactSpec(
            "prepared",
            "security_identity_validation",
            paths.security_identity_validation_csv,
        ),
        ArtifactSpec(
            "prepared",
            "security_events",
            paths.security_events_prepared_csv,
        ),
        ArtifactSpec(
            "prepared",
            "security_event_legs",
            paths.security_event_legs_prepared_csv,
        ),
        ArtifactSpec(
            "prepared",
            "security_event_sources",
            paths.security_event_sources_prepared_csv,
        ),
        ArtifactSpec(
            "prepared",
            "security_event_crossing_audit",
            paths.security_event_crossing_audit_csv,
        ),
        ArtifactSpec(
            "prepared",
            "membership_coverage_audit",
            paths.membership_coverage_audit_csv,
        ),
        ArtifactSpec("prepared", "benchmark_monthly", paths.benchmark_csv),
        ArtifactSpec("prepared", "price_basis", paths.price_basis_csv),
    ]
    price_sources = paths.price_sources
    if any(
        path.exists()
        for path in (
            price_sources.yahoo_close_csv,
            price_sources.wiki_extract_csv,
            price_sources.acquisition_status_csv,
            price_sources.readiness_csv,
            price_sources.artifact_manifest_csv,
        )
    ):
        specs.extend([
            ArtifactSpec(
                "raw", "fallback_yahoo_close", price_sources.yahoo_close_csv
            ),
            ArtifactSpec(
                "raw", "fallback_wiki_extract", price_sources.wiki_extract_csv
            ),
            ArtifactSpec(
                "raw", "fallback_price_acquisition_status",
                price_sources.acquisition_status_csv,
            ),
            ArtifactSpec(
                "raw", "fallback_price_readiness", price_sources.readiness_csv
            ),
            ArtifactSpec(
                "raw", "fallback_price_artifact_manifest",
                price_sources.artifact_manifest_csv,
            ),
        ])
    if include_brinson:
        shares = paths.shares
        specs.extend([
            ArtifactSpec("raw", "brinson_reviewed", shares.reviewed_dir, None),
            ArtifactSpec("prepared", "brinson_final_shares", shares.final_shares_csv),
        ])
        specs.append(ArtifactSpec("raw", "brinson_yahoo_shares", shares.raw_shares_csv))
    return specs


def write_preparation_manifest(
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
    include_brinson: bool = False,
) -> pd.DataFrame:
    """Record hashes for raw sources and prepared CSV artifacts."""
    return _write_shared_manifest(
        _artifact_specs(paths, include_brinson),
        manifest_path=paths.preparation_manifest_csv,
        repository_root=paths.project_root,
        skip_missing=True,
    )


def validate_preparation_manifest(
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
    include_brinson: bool = False,
    *,
    check_raw: bool = False,
    validate_raw_artifacts: Collection[str] = (),
) -> None:
    """Reject missing, stale, relocated, or incompatible pipeline artifacts.

    Analysis leaves ``check_raw`` disabled so it reads prepared CSVs only. The
    preparation and acquisition preflights enable it to prove that prepared
    core data still derives from the current raw provenance.
    """
    required_specs = _artifact_specs(paths, include_brinson)
    # Core preflights must permit rebuilding shares after reviewed inputs change.
    # The shared validator already supports selecting directly consumed raw data.
    selected_raw = set(validate_raw_artifacts)
    if check_raw and not include_brinson:
        selected_raw.update(spec.artifact for spec in required_specs if spec.stage == "raw")
    _validate_shared_manifest(
        _artifact_specs(paths, include_brinson=True),
        manifest_path=paths.preparation_manifest_csv,
        repository_root=paths.project_root,
        recovery_command=PREPARE_ALL_COMMAND,
        check_raw=check_raw and include_brinson,
        required_artifacts={spec.artifact for spec in required_specs
                            if spec.artifact != "brinson_yahoo_shares"},
        validate_raw_artifacts=selected_raw,
    )


__all__ = [
    "PREPARE_ALL_COMMAND",
    "write_preparation_manifest",
    "validate_preparation_manifest",
]
