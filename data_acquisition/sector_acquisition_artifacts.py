"""Resumable sector acquisition with recoverable, per-file atomic publication."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Mapping
from uuid import uuid4

import pandas as pd

from data_acquisition.contracts import (
    ACQUISITION_STATUS_COLUMNS,
    AcquisitionRequest,
    AcquisitionStatus,
    ProviderStatus,
    read_acquisition_statuses,
    write_acquisition_statuses,
)
from data_acquisition.engine import AcquisitionOutcome
from data_acquisition.runtime import AcquisitionRuntimePaths
from portfolio_core.artifacts import (
    ArtifactManifest,
    ArtifactOrigin,
    file_sha256,
    write_manifests,
)
from portfolio_core.io import atomic_write_bytes, atomic_write_dataframe
from portfolio_core.sector_evidence import (
    NOTICE_COLUMNS,
    SECTOR_ACQUISITION_DATASET,
    SECTOR_ACQUISITION_SCOPE,
    SNAPSHOT_COLUMNS,
    SectorHistoryPaths,
    load_sector_snapshots,
    sector_acquisition_artifact_catalog,
    sector_acquisition_artifact_origins,
    validate_sector_acquisition_manifest,
    validate_sector_snapshots,
)


_PUBLICATION_MARKER_VERSION = 1


def _publication_checkpoint(_step: str) -> None:
    """Test hook for simulating interruption at publication boundaries."""

def _validate_existing_downloaded_provenance(
    paths: SectorHistoryPaths,
    *,
    project_root: Path,
) -> None:
    """Authenticate downloaded resume state before reading checkpoints or pins.

    The manual notice ledger is deliberately excluded from byte validation
    here: reviewed edits to that input are allowed and are signed
    into the next successful checkpoint.  Their manifest identities remain
    part of the exact bundle contract, while every downloaded artifact must
    still match its previously recorded bytes and CSV shape.
    """
    project_root = Path(project_root)
    origins = sector_acquisition_artifact_origins(paths)
    downloaded = tuple(
        path for path, origin in origins.items()
        if origin is ArtifactOrigin.DOWNLOADED
    )
    existing = [path for path in downloaded if path.is_file()]
    if not existing:
        # A tracked manifest can describe the last reviewed local acquisition
        # without making ignored provider payloads mandatory in a clean clone.
        return
    if len(existing) != len(downloaded):
        missing = ", ".join(
            path.name for path in downloaded if not path.is_file()
        )
        raise ValueError(
            "Existing downloaded sector-history state is incomplete; "
            f"missing: {missing}"
        )
    if not paths.artifact_manifest_csv.is_file():
        names = ", ".join(path.name for path in existing)
        raise ValueError(
            "Existing downloaded sector-history state has no artifact "
            f"manifest: {names}"
        )

    validate_sector_acquisition_manifest(
        paths,
        repository_root=project_root,
        validate_artifacts=frozenset(
            name
            for name, _, origin in sector_acquisition_artifact_catalog(paths)
            if origin is ArtifactOrigin.DOWNLOADED
        ),
    )

def _load_snapshots(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    raw = pd.read_csv(path, keep_default_na=False, dtype=str)
    if raw.empty and tuple(raw.columns) == SNAPSHOT_COLUMNS:
        return raw
    return load_sector_snapshots(path)

def _checkpoint_date(value: str) -> str:
    parsed = pd.to_datetime(value, format="%Y-%m-%d", errors="raise")
    result = pd.Timestamp(parsed).date().isoformat()
    if result != str(value):
        raise ValueError(f"Sector checkpoint date is not canonical: {value!r}")
    return result

def _merge_runtime_snapshots(
    snapshots: pd.DataFrame,
    runtime: AcquisitionRuntimePaths,
) -> pd.DataFrame:
    result = snapshots.copy()
    for path in sorted(runtime.sector_history_snapshots.glob("*.csv")):
        checkpoint = load_sector_snapshots(path)
        checkpoint_dates = checkpoint["Requirement_Date"].dt.strftime("%Y-%m-%d").unique()
        if checkpoint_dates.tolist() != [path.stem]:
            raise ValueError(
                f"Sector runtime snapshot {path} does not match its checkpoint date"
            )
        if not result.empty:
            result_dates = pd.to_datetime(
                result["Requirement_Date"], errors="raise"
            ).dt.strftime("%Y-%m-%d")
            result = result.loc[~result_dates.eq(path.stem)].copy()
        result = pd.concat([result, checkpoint], ignore_index=True)
    if result.empty:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    return validate_sector_snapshots(result.loc[:, SNAPSHOT_COLUMNS])

def _merge_runtime_statuses(
    statuses: list[AcquisitionStatus],
    runtime: AcquisitionRuntimePaths,
) -> list[AcquisitionStatus]:
    by_key = {status.identity.key: status for status in statuses}
    for path in sorted(runtime.sector_history_statuses.glob("*.csv")):
        checkpoint = read_acquisition_statuses(path)
        if len(checkpoint) != 1:
            raise ValueError(
                f"Sector runtime status {path} must contain exactly one record"
            )
        status = checkpoint[0]
        date = status.requested_end or status.requested_start
        if _checkpoint_date(date) != path.stem:
            raise ValueError(
                f"Sector runtime status {path} does not match its checkpoint date"
            )
        by_key[status.identity.key] = status
    return sorted(by_key.values(), key=lambda status: status.identity.key)

def write_sector_runtime_checkpoint(
    runtime: AcquisitionRuntimePaths,
    outcome: AcquisitionOutcome,
    statuses: tuple[AcquisitionStatus, ...],
) -> None:
    """Persist only the provider outcome's affected date under ignored runtime."""
    requirement_date = _checkpoint_date(
        outcome.request.requested_end or outcome.request.requested_start
    )
    if outcome.result is not None:
        snapshot = validate_sector_snapshots(outcome.result.payload)
        dates = snapshot["Requirement_Date"].dt.strftime("%Y-%m-%d").unique()
        if dates.tolist() != [requirement_date]:
            raise ValueError(
                "Wikipedia provider payload does not match its requested date"
            )
        output = snapshot.copy()
        output["Requirement_Date"] = output["Requirement_Date"].dt.strftime(
            "%Y-%m-%d"
        )
        atomic_write_dataframe(
            output.loc[:, SNAPSHOT_COLUMNS],
            runtime.sector_history_snapshots / f"{requirement_date}.csv",
            index=False,
            lineterminator="\n",
        )
    matching = [
        status
        for status in statuses
        if status.identity.key == outcome.request.identity.key
    ]
    if len(matching) != 1:
        raise ValueError(
            "Acquisition checkpoint does not contain exactly one outcome status"
        )
    write_acquisition_statuses(
        runtime.sector_history_statuses / f"{requirement_date}.csv",
        matching,
    )

def _pinned_revisions(snapshots: pd.DataFrame) -> dict[str, int]:
    if snapshots.empty:
        return {}
    pins: dict[str, int] = {}
    for date, rows in snapshots.groupby("Requirement_Date", sort=True):
        revisions = rows["Revision_ID"].astype(int).unique()
        if len(revisions) != 1:
            raise ValueError(f"Sector snapshot date {date} has multiple revision IDs")
        pins[pd.Timestamp(date).date().isoformat()] = int(revisions[0])
    return pins

def _reconcile_statuses(
    requests: tuple[AcquisitionRequest, ...],
    statuses: list[AcquisitionStatus],
    snapshots: pd.DataFrame,
) -> tuple[list[AcquisitionStatus], list[AcquisitionStatus]]:
    request_keys = {request.identity.key for request in requests}
    exact = [status for status in statuses if status.identity.key in request_keys]
    unrelated = [status for status in statuses if status.identity.key not in request_keys]
    dates_with_snapshot = {
        pd.Timestamp(value).date().isoformat()
        for value in snapshots["Requirement_Date"].unique()
    } if not snapshots.empty else set()
    request_by_key = {request.identity.key: request for request in requests}
    reconciled: list[AcquisitionStatus] = []
    for status in exact:
        date = status.requested_end or status.requested_start
        if status.status is ProviderStatus.OK and date not in dates_with_snapshot:
            reconciled.append(AcquisitionStatus.pending(request_by_key[status.identity.key]))
        else:
            reconciled.append(status)
    return reconciled, unrelated


def load_sector_acquisition_state(
    paths: SectorHistoryPaths,
    runtime: AcquisitionRuntimePaths,
    requests: tuple[AcquisitionRequest, ...],
    *,
    project_root: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
    list[AcquisitionStatus],
    list[AcquisitionStatus],
]:
    """Recover and authenticate the complete resumable sector state once."""

    _recover_pending_publication(
        paths,
        runtime,
        project_root=project_root,
    )
    _validate_existing_downloaded_provenance(
        paths,
        project_root=project_root,
    )
    snapshots = _merge_runtime_snapshots(
        _load_snapshots(paths.snapshots_csv),
        runtime,
    )
    statuses = (
        read_acquisition_statuses(paths.acquisition_status_csv)
        if paths.acquisition_status_csv.is_file()
        else []
    )
    statuses = _merge_runtime_statuses(statuses, runtime)
    exact, unrelated = _reconcile_statuses(requests, statuses, snapshots)
    return snapshots, _pinned_revisions(snapshots), exact, unrelated

def _build_manifest_records(
    paths: SectorHistoryPaths,
    *,
    project_root: Path,
    captured_at_utc: str,
    artifact_sources: Mapping[Path, Path],
) -> list[ArtifactManifest]:
    """Build manifest metadata from the actual or staged source CSVs."""
    root = Path(project_root).resolve()
    origins = sector_acquisition_artifact_origins(paths)
    if set(artifact_sources) != set(paths.acquisition_artifacts):
        raise ValueError("Sector manifest sources do not match the artifact catalog")
    records = []
    for path in paths.acquisition_artifacts:
        records.append(
            ArtifactManifest.from_artifact(
                artifact_sources[path],
                scope=SECTOR_ACQUISITION_SCOPE,
                dataset=SECTOR_ACQUISITION_DATASET,
                origin=origins[path],
                artifact_path=path.resolve().relative_to(root).as_posix(),
                captured_at_utc=(
                    ""
                    if origins[path] is ArtifactOrigin.MANUAL
                    else captured_at_utc
                ),
            )
        )
    return records

def _publication_targets(paths: SectorHistoryPaths) -> tuple[Path, ...]:
    return (
        paths.snapshots_csv,
        paths.acquisition_status_csv,
        paths.artifact_manifest_csv,
    )

def _cleanup_orphan_publications(runtime: AcquisitionRuntimePaths) -> None:
    """Remove abandoned staging directories when no marker references them."""
    if not runtime.sector_history_publications.exists():
        return
    for directory in sorted(runtime.sector_history_publications.iterdir()):
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or re.fullmatch(r"[0-9a-f]{32}", directory.name) is None
        ):
            continue
        for path in directory.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    f"Unexpected sector publication staging entry: {path}"
                )
            path.unlink()
        directory.rmdir()

def _cleanup_consumed_checkpoints(runtime: AcquisitionRuntimePaths) -> None:
    """Discard per-date checkpoints after aggregate publication succeeds."""
    for directory in (
        runtime.sector_history_snapshots,
        runtime.sector_history_statuses,
    ):
        for path in directory.glob("*.csv"):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Unexpected sector checkpoint entry: {path}")
            path.unlink()

def _recover_pending_publication(
    paths: SectorHistoryPaths,
    runtime: AcquisitionRuntimePaths,
    *,
    project_root: Path,
) -> bool:
    """Complete a marked publication after checking its paths and hashes.

    Skip files already matching their hash and promote the rest, manifest last.
    Return whether recovery occurred; invalid markers or missing/corrupt stages
    raise ValueError and leave recovery pending.
    """
    marker_path = runtime.sector_history_pending_publish
    if not marker_path.is_file():
        _cleanup_orphan_publications(runtime)
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid pending sector publication marker: {marker_path}"
        ) from exc
    if set(marker) != {"version", "files"} or marker["version"] != _PUBLICATION_MARKER_VERSION:
        raise ValueError(f"Unsupported pending sector publication marker: {marker_path}")
    if not isinstance(marker["files"], list):
        raise ValueError(f"Invalid pending sector publication file list: {marker_path}")

    root = Path(project_root).resolve()
    runtime_root = runtime.root.resolve()
    expected_targets = {
        path.resolve().relative_to(root).as_posix()
        for path in _publication_targets(paths)
    }
    entries: list[tuple[Path, Path, str]] = []
    for raw in marker["files"]:
        if not isinstance(raw, dict) or set(raw) != {"staged", "target", "sha256"}:
            raise ValueError(f"Invalid pending sector publication entry: {raw!r}")
        digest = str(raw["sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Pending sector publication contains an invalid SHA-256")
        staged = (runtime_root / str(raw["staged"])).resolve()
        target = (root / str(raw["target"])).resolve()
        if not staged.is_relative_to(runtime_root) or not target.is_relative_to(root):
            raise ValueError("Pending sector publication path escapes its root")
        entries.append((staged, target, digest))
    target_names = {
        target.relative_to(root).as_posix() for _, target, _ in entries
    }
    if target_names != expected_targets or len(entries) != len(expected_targets):
        raise ValueError("Pending sector publication does not match the artifact catalog")
    if entries[-1][1] != paths.artifact_manifest_csv.resolve():
        raise ValueError("Pending sector publication must promote its manifest last")

    for staged, target, digest in entries:
        if target.is_file() and file_sha256(target) == digest:
            continue
        if not staged.is_file() or file_sha256(staged) != digest:
            raise ValueError(
                f"Pending sector publication stage is missing or corrupt: {staged}"
            )
        atomic_write_bytes(staged.read_bytes(), target)
        _publication_checkpoint(f"promoted:{target.name}")

    marker_path.unlink()
    for staged, _, _ in entries:
        staged.unlink(missing_ok=True)
    stage_directories = sorted(
        {staged.parent for staged, _, _ in entries},
        key=lambda value: len(value.parts),
        reverse=True,
    )
    for directory in stage_directories:
        try:
            directory.rmdir()
        except OSError:
            pass
    _cleanup_consumed_checkpoints(runtime)
    _cleanup_orphan_publications(runtime)
    return True

def publish_sector_state(
    *,
    paths: SectorHistoryPaths,
    runtime: AcquisitionRuntimePaths,
    project_root: Path,
    snapshots: pd.DataFrame,
    statuses: list[AcquisitionStatus] | tuple[AcquisitionStatus, ...],
    notices: pd.DataFrame,
    captured_at_utc: str,
) -> None:
    """Stage a recoverable publication, replacing files atomically, manifest last.

    A marker records expected hashes so recovery can complete an interrupted
    publication; the bundle is not replaced in one atomic operation.
    """
    paths.directory.mkdir(parents=True, exist_ok=True)
    if runtime.sector_history_pending_publish.is_file():
        raise RuntimeError("A pending sector publication must be recovered first")

    if snapshots.empty:
        if tuple(snapshots.columns) != SNAPSHOT_COLUMNS:
            raise ValueError("Empty sector snapshots must preserve their schema")
        validated_snapshots = snapshots.copy()
    else:
        validated_snapshots = validate_sector_snapshots(
            snapshots.loc[:, SNAPSHOT_COLUMNS]
        )
    snapshot_output = validated_snapshots.copy()
    if not snapshot_output.empty:
        snapshot_output["Requirement_Date"] = snapshot_output[
            "Requirement_Date"
        ].dt.strftime("%Y-%m-%d")
    ordered_statuses = sorted(statuses, key=lambda item: item.identity.key)
    status_output = pd.DataFrame(
        [record.to_row() for record in ordered_statuses],
        columns=ACQUISITION_STATUS_COLUMNS,
    )
    artifact_frames = {
        paths.snapshots_csv: snapshot_output.loc[:, SNAPSHOT_COLUMNS],
        paths.notices_csv: notices.loc[:, NOTICE_COLUMNS],
        paths.acquisition_status_csv: status_output,
    }

    publication_id = uuid4().hex
    stage_directory = runtime.sector_history_publications / publication_id
    stage_directory.mkdir(parents=True, exist_ok=False)
    staged_sources: dict[Path, Path] = {
        paths.notices_csv: paths.notices_csv,
    }
    for target in _publication_targets(paths)[:-1]:
        staged = stage_directory / target.name
        atomic_write_dataframe(
            artifact_frames[target],
            staged,
            index=False,
            lineterminator="\n",
        )
        staged_sources[target] = staged
    records = _build_manifest_records(
        paths,
        project_root=project_root,
        captured_at_utc=captured_at_utc,
        artifact_sources=staged_sources,
    )
    staged_manifest = stage_directory / paths.artifact_manifest_csv.name
    write_manifests(staged_manifest, records)

    publication_files = [
        (stage_directory / target.name, target)
        for target in _publication_targets(paths)
    ]
    marker = {
        "version": _PUBLICATION_MARKER_VERSION,
        "files": [
            {
                "staged": staged.relative_to(runtime.root).as_posix(),
                "target": target.resolve().relative_to(
                    Path(project_root).resolve()
                ).as_posix(),
                "sha256": file_sha256(staged),
            }
            for staged, target in publication_files
        ],
    }
    atomic_write_bytes(
        (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        ),
        runtime.sector_history_pending_publish,
    )
    _publication_checkpoint("marker_written")
    _recover_pending_publication(
        paths,
        runtime,
        project_root=project_root,
    )

__all__ = [
    "load_sector_acquisition_state",
    "publish_sector_state",
    "write_sector_runtime_checkpoint",
]
