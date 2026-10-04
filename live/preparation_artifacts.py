"""Live artifact catalog for the shared preparation-manifest contract."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import pandas as pd

from portfolio_core.artifacts import (
    ArtifactSpec,
    validate_preparation_manifest as validate_shared_manifest,
    write_preparation_manifest as write_shared_manifest,
)
from portfolio_core.sector_evidence import sector_analytical_evidence_catalog

from .config import DEFAULT_CONFIG
from .paths import LivePaths


PREPARE_COMMAND = "python -m live.prepare all"


def _base_specs(paths: LivePaths) -> list[ArtifactSpec]:
    updated_path, changes_path, intervals_path = paths.membership.data_files
    identity_paths = paths.security_identity
    items = [
        ("raw", "fja_updated", updated_path),
        ("raw", "fja_changes", changes_path),
        ("raw", "fja_intervals", intervals_path),
        ("raw", "fja_source_manifest", paths.membership.manifest_csv),
        *[
            ("raw", name, path)
            for name, path, _ in sector_analytical_evidence_catalog(
                paths.sector_history
            )
        ],
        ("raw", "price_status", paths.raw_price_status_csv),
        ("raw", "price_readiness", paths.raw_price_readiness_csv),
        ("raw", "price_requirements", paths.raw_price_requirements_csv),
        ("raw", "price_acquisition_manifest", paths.raw_price_artifact_manifest_csv),
        ("raw", "price_open", paths.raw_price_open_csv),
        ("raw", "price_close", paths.raw_price_close_csv),
        ("raw", "price_volume", paths.raw_price_volume_csv),
        ("raw", "monthly_observations", paths.raw_monthly_dir / "observations.csv"),
        ("raw", "monthly_dividends_raw", paths.raw_monthly_dir / "dividends.csv"),
        ("raw", "monthly_basis_checks", paths.raw_monthly_dir / "basis_checks.csv"),
        ("raw", "monthly_evidence_manifest", paths.raw_monthly_dir / "artifact_manifest.csv"),
        ("raw", "monthly_reuters_source", paths.project_root / "data/shared/supplied/reuters/SP500_Full_2014_2026_Cleaned.csv"),
        ("raw", "corporate_action_policy", paths.raw_corporate_action_policy_csv),
        ("raw", "corporate_action_policy_manifest", paths.raw_corporate_action_policy_manifest_csv),
        ("raw", "security_events", identity_paths.events_csv),
        ("raw", "security_event_legs", identity_paths.legs_csv),
        ("raw", "security_event_sources", identity_paths.sources_csv),
        ("raw", "provider_symbol_mappings", identity_paths.provider_mappings_csv),
        (
            "raw",
            "provider_row_equivalence",
            identity_paths.provider_row_equivalence_csv,
        ),
        ("raw", "security_identity_manifest", identity_paths.manifest_csv),
        ("prepared", "decision_schedule", paths.decision_schedule_csv),
        ("prepared", "pit_membership", paths.pit_membership_csv),
        ("prepared", "asset_metadata", paths.asset_metadata_csv),
        ("prepared", "sector_assignments", paths.sector_assignments_csv),
        ("prepared", "market_daily", paths.market_daily_csv),
        ("prepared", "market_monthly", paths.market_monthly_csv),
        ("prepared", "monthly_dividends_prepared", paths.monthly_dividends_csv),
        ("prepared", "corporate_action_events_prepared", paths.prepared_corporate_action_events_csv),
        ("prepared", "corporate_action_legs_prepared", paths.prepared_corporate_action_legs_csv),
        ("prepared", "corporate_action_sources_prepared", paths.prepared_corporate_action_sources_csv),
        ("prepared", "corporate_action_policy_prepared", paths.prepared_corporate_action_policy_csv),
        ("prepared", "price_basis", paths.price_basis_csv),
    ]
    return [ArtifactSpec(stage, artifact, path, {}) for stage, artifact, path in items]


def _supplement_specs(paths: LivePaths) -> list[ArtifactSpec]:
    return [
        ArtifactSpec(
            "raw",
            "price_yahoo_supplement",
            paths.raw_price_supplemental_csv,
            {},
        ),
        ArtifactSpec(
            "raw",
            "price_yahoo_supplement_manifest",
            paths.raw_price_supplemental_manifest_csv,
            {},
        ),
    ]


def _brinson_specs(paths: LivePaths) -> list[ArtifactSpec]:
    items = [
        ("raw", "shares_raw", Path(paths.shares.raw_shares_csv)),
        ("raw", "shares_status", Path(paths.shares.acquisition_status_csv)),
        ("raw", "shares_readiness", Path(paths.shares.readiness_csv)),
        ("raw", "shares_acquisition_manifest", Path(paths.shares.artifact_manifest_csv)),
        ("raw", "benchmark_raw", paths.raw_benchmark_csv),
        ("raw", "benchmark_status", paths.raw_benchmark_status_csv),
        ("raw", "benchmark_readiness", paths.raw_benchmark_readiness_csv),
        ("raw", "benchmark_acquisition_manifest", paths.raw_benchmark_artifact_manifest_csv),
        ("prepared", "shares_prepared", Path(paths.shares.prepared_shares_csv)),
        ("prepared", "benchmark_prepared", paths.prepared_benchmark_csv),
    ]
    return [ArtifactSpec(stage, artifact, path, {}) for stage, artifact, path in items]


def _supplement_artifact_names(paths: LivePaths) -> frozenset[str]:
    return frozenset(spec.artifact for spec in _supplement_specs(paths))


def artifact_catalog(
    paths: LivePaths = DEFAULT_CONFIG.paths,
) -> tuple[ArtifactSpec, ...]:
    """Return every artifact name recognized by the live manifest contract."""
    return tuple(
        _base_specs(paths)
        + _supplement_specs(paths)
        + _brinson_specs(paths)
    )


def required_artifact_names(
    paths: LivePaths = DEFAULT_CONFIG.paths,
    *,
    include_brinson: bool,
) -> frozenset[str]:
    """Return the artifacts required by one requested preparation scope."""
    required = {
        spec.artifact for spec in _base_specs(paths)
    }
    required.update(_supplement_artifact_names(paths))
    if include_brinson:
        required.update(spec.artifact for spec in _brinson_specs(paths))
    return frozenset(required)


def _require_supplement(paths: LivePaths) -> None:
    if any(not spec.path.is_file() for spec in _supplement_specs(paths)):
        raise RuntimeError(
            "The authoritative live Yahoo supplement and its acquisition "
            f"manifest are required. Run `{PREPARE_COMMAND}` first."
        )


def write_preparation_manifest(
    paths: LivePaths = DEFAULT_CONFIG.paths,
    include_brinson: bool = False,
) -> pd.DataFrame:
    """Write the live artifact inventory using the shared contract."""
    _require_supplement(paths)
    selected = _base_specs(paths) + _supplement_specs(paths)
    if include_brinson:
        selected.extend(_brinson_specs(paths))
    return write_shared_manifest(
        selected,
        manifest_path=paths.preparation_manifest_csv,
        repository_root=paths.project_root,
        skip_missing=False,
    )


def validate_preparation_manifest(
    paths: LivePaths = DEFAULT_CONFIG.paths,
    include_brinson: bool = False,
    check_raw: bool = False,
    *,
    required_artifacts: Collection[str] | None = None,
) -> None:
    """Validate one live scope against the complete recognized catalog."""
    manifest_path = paths.preparation_manifest_csv
    required = (
        required_artifact_names(
            paths,
            include_brinson=include_brinson,
        )
        if required_artifacts is None
        else frozenset(str(name) for name in required_artifacts)
    )
    required = required | _supplement_artifact_names(paths)
    validate_shared_manifest(
        artifact_catalog(paths),
        manifest_path=manifest_path,
        repository_root=paths.project_root,
        recovery_command=PREPARE_COMMAND,
        check_raw=check_raw,
        required_artifacts=required,
    )


__all__ = [
    "PREPARE_COMMAND",
    "artifact_catalog",
    "required_artifact_names",
    "validate_preparation_manifest",
    "write_preparation_manifest",
]
