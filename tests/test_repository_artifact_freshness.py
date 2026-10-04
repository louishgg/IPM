"""Guards for the repository's materialized backtest preparation state."""

import pandas as pd
import pytest

from backtest.config import DEFAULT_CONFIG
from backtest.preparation_artifacts import (
    _artifact_specs,
    validate_preparation_manifest,
)
from portfolio_core.artifacts import (
    MANIFEST_COLUMNS,
    PREPARATION_SCHEMA_VERSION,
    artifact_sha256,
    artifact_shape,
    relative_artifact_path,
)


CANONICAL_IDENTITY_ARTIFACTS = {
    "canonical_security_events",
    "canonical_security_event_legs",
    "canonical_security_event_sources",
    "provider_symbol_mappings",
    "provider_row_equivalence",
    "security_identity_manifest",
}
POINT_IN_TIME_SECTOR_ARTIFACTS = {
    "sector_history_snapshots",
    "sector_history_notices",
    "sector_assignments",
}


def _assert_manifest_artifacts(
    paths,
    by_artifact: pd.DataFrame,
    expected_stages: dict[str, str],
) -> None:
    contracts = {
        spec.artifact: spec
        for spec in _artifact_specs(
            paths,
            include_brinson=False,
        )
        if spec.artifact in expected_stages
    }
    assert set(contracts) == set(expected_stages)
    for artifact, spec in contracts.items():
        assert spec.path.is_file(), spec.path
        assert artifact in by_artifact.index
        entry = by_artifact.loc[artifact]
        shape = artifact_shape(spec)
        assert str(entry["Stage"]) == spec.stage == expected_stages[artifact]
        assert str(entry["Relative_Path"]) == relative_artifact_path(
            spec.path, paths.project_root
        )
        assert str(entry["SHA256"]) == artifact_sha256(spec)
        assert int(entry["Rows"]) == int(shape[0])
        assert int(entry["Columns"]) == int(shape[1])


def test_tracked_preparation_manifest_matches_canonical_identity_provenance():
    paths = DEFAULT_CONFIG.paths
    manifest = pd.read_csv(
        paths.preparation_manifest_csv,
        dtype={"Artifact": "string"},
    )
    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert manifest["Schema_Version"].eq(PREPARATION_SCHEMA_VERSION).all()
    by_artifact = manifest.set_index("Artifact")
    assert by_artifact.index.is_unique

    _assert_manifest_artifacts(
        paths,
        by_artifact,
        {artifact: "raw" for artifact in CANONICAL_IDENTITY_ARTIFACTS},
    )


def test_tracked_preparation_manifest_uses_point_in_time_sector_history():
    paths = DEFAULT_CONFIG.paths
    manifest = pd.read_csv(
        paths.preparation_manifest_csv,
        dtype={"Artifact": "string"},
    )
    by_artifact = manifest.set_index("Artifact")
    assert by_artifact.index.is_unique

    _assert_manifest_artifacts(
        paths,
        by_artifact,
        {
            artifact: (
                "prepared" if artifact == "sector_assignments" else "raw"
            )
            for artifact in POINT_IN_TIME_SECTOR_ARTIFACTS
        },
    )


def test_complete_local_backtest_bundle_is_current_when_materialized():
    paths = DEFAULT_CONFIG.paths
    required_local_artifacts = (
        paths.prices_csv,
        paths.benchmark_raw_csv,
        paths.shares.raw_shares_csv,
        paths.shares.reviewed_dir,
        paths.prices_monthly_csv,
        paths.shares.final_shares_csv,
    )
    if not all(path.exists() for path in required_local_artifacts):
        pytest.skip("complete ignored backtest data bundle is not materialized")

    validate_preparation_manifest(
        paths,
        include_brinson=True,
        check_raw=True,
    )
