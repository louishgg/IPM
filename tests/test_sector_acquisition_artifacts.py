"""Tests for sector acquisition checkpoints and publication."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import data_acquisition.sector_acquisition_artifacts as sector_artifacts_module
from data_acquisition.contracts import (
    AcquisitionStatus,
    ProviderStatus,
    read_acquisition_statuses,
    write_acquisition_statuses,
)
from data_acquisition.sector_acquisition_planning import (
    build_sector_acquisition_requests,
)
from data_acquisition.runtime import AcquisitionRuntimePaths
from portfolio_core.artifacts import ArtifactOrigin, read_manifests
from portfolio_core.sector_evidence import (
    SectorHistoryPaths,
    load_sector_notices,
    load_sector_snapshots,
    validate_sector_acquisition_manifest,
)
from _sector_test_helpers import (
    PROJECT_ROOT,
    _acquisition_requirements,
    _notice,
    _notices,
    _snapshots,
    _write_sector_manifest,
)


def test_repository_sector_history_manifest_matches_current_shared_state():
    paths = SectorHistoryPaths.from_project_root(PROJECT_ROOT)
    validate_sector_acquisition_manifest(paths, repository_root=PROJECT_ROOT)
    manifests = read_manifests(paths.artifact_manifest_csv)
    statuses = read_acquisition_statuses(paths.acquisition_status_csv)
    snapshots = load_sector_snapshots(paths.snapshots_csv)
    status_dates = {status.identity.effective_start for status in statuses}
    snapshot_dates = {
        timestamp.strftime("%Y-%m-%d")
        for timestamp in snapshots["Requirement_Date"].unique()
    }

    assert len(statuses) == snapshots["Requirement_Date"].nunique() == 141
    assert "2026-05-01" in snapshot_dates
    assert status_dates == snapshot_dates
    assert {status.status for status in statuses} == {ProviderStatus.OK}
    assert len(manifests) == 3
    assert {item.origin for item in manifests} == {
        ArtifactOrigin.DOWNLOADED,
        ArtifactOrigin.MANUAL,
    }
    assert {Path(item.artifact_path).name for item in manifests} == {
        "wikipedia_sector_snapshots.csv",
        "sp500_constituent_notices.csv",
        "acquisition_status.csv",
    }


def test_fresh_sector_runtime_loads_without_creating_directories(tmp_path: Path):
    paths = SectorHistoryPaths.from_project_root(tmp_path)
    runtime = AcquisitionRuntimePaths(tmp_path)
    requests = build_sector_acquisition_requests(_acquisition_requirements())

    snapshots, pins, exact, unrelated = (
        sector_artifacts_module.load_sector_acquisition_state(
            paths,
            runtime,
            requests,
            project_root=tmp_path,
        )
    )

    assert snapshots.empty
    assert pins == {}
    assert exact == []
    assert unrelated == []
    assert not runtime.root.exists()


def test_successful_checkpoint_writes_only_the_acquired_runtime_date(tmp_path: Path):
    checked = "2026-08-05T12:00:00Z"
    runtime = AcquisitionRuntimePaths(tmp_path)
    request = build_sector_acquisition_requests(_acquisition_requirements())[0]
    status = AcquisitionStatus(
        identity=request.identity,
        status=ProviderStatus.OK,
        requested_start=request.requested_start,
        requested_end=request.requested_end,
        observation_count=1,
        observation_start="2020-01-31",
        observation_end="2020-01-31",
        attempted_at_utc=checked,
        client="requests",
        client_version="1",
        http_status="200",
    )
    outcome = SimpleNamespace(
        request=request,
        result=SimpleNamespace(
            payload=_snapshots([("2020-01-31", "AAA", "Energy")])
        ),
    )
    sector_artifacts_module.write_sector_runtime_checkpoint(
        runtime,
        outcome,
        (status,),
    )

    assert runtime.sector_history_snapshots.is_dir()
    assert runtime.sector_history_statuses.is_dir()
    assert not runtime.yfinance_cache.exists()
    assert not runtime.locks.exists()
    assert not runtime.sector_history_publications.exists()

    untouched = runtime.sector_history_snapshots / "2019-12-31.csv"
    untouched.write_bytes(b"untouched\n")
    sector_artifacts_module.write_sector_runtime_checkpoint(
        runtime,
        outcome,
        (status,),
    )

    assert untouched.read_bytes() == b"untouched\n"
    assert sorted(path.name for path in runtime.sector_history_snapshots.iterdir()) == [
        "2019-12-31.csv",
        "2020-01-31.csv",
    ]
    assert sorted(path.name for path in runtime.sector_history_statuses.iterdir()) == [
        "2020-01-31.csv"
    ]


@pytest.mark.parametrize(
    "failure_step",
    [
        "marker_written",
        "promoted:wikipedia_sector_snapshots.csv",
        "promoted:acquisition_status.csv",
        "promoted:artifact_manifest.csv",
    ],
)
def test_interrupted_sector_publication_recovers_before_provenance_validation(
    tmp_path: Path,
    monkeypatch,
    failure_step: str,
):
    paths = SectorHistoryPaths.from_project_root(tmp_path)
    paths.directory.mkdir(parents=True)
    notices = _notices([_notice(Ticker="OTHER")])
    notices.to_csv(paths.notices_csv, index=False)
    runtime = AcquisitionRuntimePaths(tmp_path)
    runtime.sector_history_snapshots.mkdir(parents=True)
    runtime.sector_history_statuses.mkdir()
    runtime.sector_history_publications.mkdir()
    (runtime.sector_history_snapshots / "2020-01-31.csv").write_text(
        "checkpoint\n"
    )
    (runtime.sector_history_statuses / "2020-01-31.csv").write_text(
        "checkpoint\n"
    )
    orphan = runtime.sector_history_publications / ("0" * 32)
    orphan.mkdir()
    (orphan / "orphan.csv").write_text("orphan\n")
    requirements = _acquisition_requirements()
    request = build_sector_acquisition_requests(requirements)[0]
    statuses = [
        AcquisitionStatus(
            identity=request.identity,
            status=ProviderStatus.OK,
            requested_start=request.requested_start,
            requested_end=request.requested_end,
            observation_count=1,
            observation_start="2020-01-31",
            observation_end="2020-01-31",
            attempted_at_utc="2026-08-05T12:00:00Z",
            client="requests",
            client_version="1",
            http_status="200",
        )
    ]
    injected = []

    def fail_once(step: str) -> None:
        if step == failure_step and not injected:
            injected.append(step)
            raise RuntimeError(f"injected publication failure: {step}")

    monkeypatch.setattr(sector_artifacts_module, "_publication_checkpoint", fail_once)
    with pytest.raises(RuntimeError, match="injected publication failure"):
        sector_artifacts_module.publish_sector_state(
            paths=paths,
            runtime=runtime,
            project_root=tmp_path,
            snapshots=_snapshots([("2020-01-31", "AAA", "Energy")]),
            statuses=statuses,
            notices=load_sector_notices(paths.notices_csv),
            captured_at_utc="2026-08-05T12:00:00Z",
        )

    assert injected == [failure_step]
    assert runtime.sector_history_pending_publish.is_file()
    monkeypatch.setattr(
        sector_artifacts_module,
        "_publication_checkpoint",
        lambda _step: None,
    )
    snapshots, pins, exact, unrelated = sector_artifacts_module.load_sector_acquisition_state(
        paths, runtime, (request,), project_root=tmp_path,
    )
    assert len(snapshots) == 1
    assert pins == {"2020-01-31": int(snapshots.iloc[0].Revision_ID)}
    assert exact == statuses
    assert unrelated == []
    assert not runtime.sector_history_pending_publish.exists()
    assert not any(runtime.sector_history_snapshots.iterdir())
    assert not any(runtime.sector_history_statuses.iterdir())
    assert not any(runtime.sector_history_publications.iterdir())
    validate_sector_acquisition_manifest(paths, repository_root=tmp_path)
    manifests = read_manifests(paths.artifact_manifest_csv)
    assert {Path(item.artifact_path).name for item in manifests} == {
        path.name for path in paths.acquisition_artifacts
    }
    assert {(item.origin, item.captured_at_utc) for item in manifests} == {
        (ArtifactOrigin.DOWNLOADED, "2026-08-05T12:00:00Z"),
        (ArtifactOrigin.MANUAL, ""),
    }


def test_tracked_manifest_allows_clean_downloaded_state_bootstrap(tmp_path):
    paths = SectorHistoryPaths.from_project_root(tmp_path)
    paths.directory.mkdir(parents=True)
    _snapshots([("2020-01-31", "AAA", "Energy")]).to_csv(
        paths.snapshots_csv,
        index=False,
    )
    _notices([_notice(Ticker="OTHER")]).to_csv(paths.notices_csv, index=False)
    write_acquisition_statuses(paths.acquisition_status_csv, ())
    _write_sector_manifest(paths, tmp_path)

    downloaded = (
        paths.snapshots_csv,
        paths.acquisition_status_csv,
    )
    for path in downloaded:
        path.unlink()

    sector_artifacts_module._validate_existing_downloaded_provenance(
        paths,
        project_root=tmp_path,
    )

    _snapshots([("2020-01-31", "AAA", "Energy")]).to_csv(
        paths.snapshots_csv,
        index=False,
    )
    with pytest.raises(ValueError, match="downloaded sector-history state is incomplete"):
        sector_artifacts_module._validate_existing_downloaded_provenance(
            paths,
            project_root=tmp_path,
        )
