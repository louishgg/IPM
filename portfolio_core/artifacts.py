"""Shared artifact-manifest models, hashing, persistence, and validation.

Acquisition manifests retain schema version 2 and preparation manifests retain
schema version 7. Domain modules own their artifact inventories and recovery
commands; this module owns the common integrity mechanics.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Self

import pandas as pd
from portfolio_core.io import atomic_write_dataframe
from portfolio_core.io import atomic_write_csv_rows


SCHEMA_VERSION = "2"
ACQUISITION_MANIFEST_COLUMNS = (
    "Schema_Version",
    "Scope",
    "Dataset",
    "Origin",
    "Captured_At_UTC",
    "Artifact_Path",
    "SHA256",
    "Rows",
    "Columns",
)


class ArtifactOrigin(StrEnum):
    DOWNLOADED = "downloaded"
    MIGRATED = "migrated"
    MANUAL = "manual"
    SUPPLIED = "supplied"


def _text(value: object, *, field_name: str, required: bool = False) -> str:
    result = str(value).strip()
    if required and not result:
        raise ValueError(f"{field_name} must not be empty")
    if "\n" in result or "\r" in result:
        raise ValueError(f"{field_name} must be a single line")
    return result


def _utc(value: object, *, field_name: str) -> str:
    result = _text(value, field_name=field_name)
    if not result:
        return result
    if not result.endswith("Z"):
        raise ValueError(f"{field_name} must be a UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(result[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid UTC timestamp") from exc
    return result


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a nonnegative integer") from exc
    if result < 0 or str(value).strip() not in {str(result), f"{result}.0"}:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return result


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_shape(path: Path) -> tuple[int, tuple[str, ...]]:
    """Return the data-row count and ordered header of one CSV artifact."""
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = tuple(next(reader))
        except StopIteration:
            return 0, ()
        return sum(1 for _ in reader), header


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """One acquisition artifact and its immutable integrity metadata."""

    schema_version: str
    scope: str
    dataset: str
    origin: ArtifactOrigin
    captured_at_utc: str
    artifact_path: str
    sha256: str
    rows: int
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _text(self.schema_version, field_name="schema_version", required=True),
        )
        for name in ("scope", "dataset", "artifact_path"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )
        object.__setattr__(self, "origin", ArtifactOrigin(self.origin))
        object.__setattr__(
            self,
            "captured_at_utc",
            _utc(self.captured_at_utc, field_name="captured_at_utc"),
        )
        digest = _text(self.sha256, field_name="sha256", required=True).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(
            self,
            "rows",
            _nonnegative_integer(self.rows, field_name="rows"),
        )
        columns = tuple(
            _text(item, field_name="column", required=True) for item in self.columns
        )
        if len(set(columns)) != len(columns):
            raise ValueError("manifest columns must be unique")
        object.__setattr__(self, "columns", columns)

    @classmethod
    def from_artifact(
        cls,
        path: Path,
        *,
        scope: str,
        dataset: str,
        origin: ArtifactOrigin,
        artifact_path: str | None = None,
        captured_at_utc: str = "",
        schema_version: str = SCHEMA_VERSION,
    ) -> Self:
        rows, columns = _csv_shape(path)
        return cls(
            schema_version=schema_version,
            scope=scope,
            dataset=dataset,
            origin=origin,
            captured_at_utc=captured_at_utc,
            artifact_path=artifact_path or str(path),
            sha256=file_sha256(path),
            rows=rows,
            columns=tuple(columns),
        )

    def to_row(self) -> dict[str, str]:
        return {
            "Schema_Version": self.schema_version,
            "Scope": self.scope,
            "Dataset": self.dataset,
            "Origin": self.origin.value,
            "Captured_At_UTC": self.captured_at_utc,
            "Artifact_Path": self.artifact_path,
            "SHA256": self.sha256,
            "Rows": str(self.rows),
            "Columns": json.dumps(self.columns, separators=(",", ":")),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> Self:
        return cls(
            schema_version=row["Schema_Version"],
            scope=row["Scope"],
            dataset=row["Dataset"],
            origin=ArtifactOrigin(str(row["Origin"]).strip()),
            captured_at_utc=row["Captured_At_UTC"],
            artifact_path=row["Artifact_Path"],
            sha256=row["SHA256"],
            rows=row["Rows"],
            columns=tuple(json.loads(str(row["Columns"]))),
        )


def validate_manifest_artifact(
    manifest: ArtifactManifest,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Validate existence, hash, and declared CSV shape."""
    configured = Path(manifest.artifact_path)
    if configured.is_absolute():
        target = configured
    else:
        if base_dir is None:
            raise ValueError("base_dir is required for a relative artifact path")
        root = Path(base_dir).resolve()
        target = (root / configured).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"manifest artifact escapes its base directory: {configured}")
    if not target.is_file():
        raise FileNotFoundError(f"manifest artifact does not exist: {target}")
    actual_digest = file_sha256(target)
    if actual_digest != manifest.sha256:
        raise ValueError(
            f"manifest hash mismatch for {target}: expected {manifest.sha256}, "
            f"found {actual_digest}"
        )
    if target.suffix.lower() == ".csv":
        row_count, header = _csv_shape(target)
        if header != manifest.columns:
            raise ValueError(
                f"manifest column mismatch for {target}: expected "
                f"{manifest.columns}, found {header}"
            )
        if row_count != manifest.rows:
            raise ValueError(
                f"manifest row-count mismatch for {target}: expected "
                f"{manifest.rows}, found {row_count}"
            )
    return target


def _require_unique_manifests(records: Iterable[ArtifactManifest]) -> None:
    seen: set[tuple[str, str, str]] = set()
    duplicates: set[tuple[str, str, str]] = set()
    for item in records:
        key = (item.scope, item.dataset, item.artifact_path)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"duplicate manifest artifact records: {sorted(duplicates)!r}")


def read_manifests(path: Path) -> list[ArtifactManifest]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != ACQUISITION_MANIFEST_COLUMNS:
            raise ValueError(
                f"{path} must have exactly {list(ACQUISITION_MANIFEST_COLUMNS)}; "
                f"found {reader.fieldnames or []}"
            )
        records = [ArtifactManifest.from_row(row) for row in reader]
    unsupported = sorted(
        {item.schema_version for item in records if item.schema_version != SCHEMA_VERSION}
    )
    if unsupported:
        raise ValueError(
            f"{path} uses unsupported acquisition-manifest schema versions "
            f"{unsupported}; expected {SCHEMA_VERSION}"
        )
    _require_unique_manifests(records)
    return sorted(records, key=lambda item: (item.scope, item.dataset, item.artifact_path))


def validate_exact_manifest_catalog(
    manifest_path: Path,
    *,
    scope: str,
    dataset: str,
    expected_origins: Mapping[str, ArtifactOrigin],
    base_dir: Path,
    validate_artifacts: Collection[str] | None = None,
) -> tuple[ArtifactManifest, ...]:
    """Validate one exact domain-owned acquisition artifact catalog."""

    expected = {
        str(artifact_path): ArtifactOrigin(origin)
        for artifact_path, origin in expected_origins.items()
    }
    selected = set(expected) if validate_artifacts is None else set(validate_artifacts)
    unknown = sorted(selected - set(expected))
    if unknown:
        raise ValueError(f"Unknown manifest artifacts selected for validation: {unknown}")

    records = tuple(read_manifests(Path(manifest_path)))
    by_path: dict[str, ArtifactManifest] = {}
    duplicate_paths: set[str] = set()
    for record in records:
        if record.artifact_path in by_path:
            duplicate_paths.add(record.artifact_path)
        by_path[record.artifact_path] = record
    if duplicate_paths:
        raise ValueError(
            f"Manifest {manifest_path} contains duplicate artifact paths: "
            f"{sorted(duplicate_paths)}"
        )
    if len(records) != len(expected) or set(by_path) != set(expected):
        raise ValueError(
            f"Manifest {manifest_path} contains {sorted(by_path)}, expected "
            f"{sorted(expected)}"
        )

    for artifact_path, expected_origin in expected.items():
        record = by_path[artifact_path]
        if (
            record.scope != scope
            or record.dataset != dataset
            or record.origin is not expected_origin
        ):
            raise ValueError(
                f"Manifest identity is invalid for {artifact_path}: expected "
                f"scope={scope!r}, dataset={dataset!r}, "
                f"origin={expected_origin.value!r}"
            )
        if artifact_path in selected:
            validate_manifest_artifact(record, base_dir=base_dir)
    return records


def write_manifests(path: Path, records: Iterable[ArtifactManifest]) -> None:
    values = list(records)
    unsupported = sorted(
        {item.schema_version for item in values if item.schema_version != SCHEMA_VERSION}
    )
    if unsupported:
        raise ValueError(
            f"Cannot write acquisition-manifest schema versions {unsupported}; "
            f"expected {SCHEMA_VERSION}"
        )
    _require_unique_manifests(values)
    ordered = sorted(values, key=lambda item: (item.scope, item.dataset, item.artifact_path))
    atomic_write_csv_rows(
        (item.to_row() for item in ordered),
        ACQUISITION_MANIFEST_COLUMNS,
        Path(path),
    )


PREPARATION_SCHEMA_VERSION = 7
MANIFEST_COLUMNS = [
    "Schema_Version",
    "Stage",
    "Artifact",
    "Relative_Path",
    "SHA256",
    "Rows",
    "Columns",
]


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """One raw or prepared artifact covered by a preparation manifest.

    ``read_csv_kwargs=None`` identifies a directory artifact.  Its row count is
    the number of files and its column count is zero.
    """

    stage: str
    artifact: str
    path: Path
    read_csv_kwargs: Mapping[str, object] | None = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in {"raw", "prepared"}:
            raise ValueError(f"Unsupported preparation stage: {self.stage!r}")
        if not str(self.artifact).strip():
            raise ValueError("Preparation artifact name cannot be empty")
        object.__setattr__(self, "path", Path(self.path))


def sha256_directory(path: Path) -> str:
    """Hash a directory from stable relative names and file contents."""
    path = Path(path)
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def artifact_exists(spec: ArtifactSpec) -> bool:
    return spec.path.is_dir() if spec.read_csv_kwargs is None else spec.path.is_file()


def artifact_shape(spec: ArtifactSpec) -> tuple[int, int]:
    if spec.read_csv_kwargs is None:
        return sum(item.is_file() for item in spec.path.rglob("*")), 0
    options = dict(spec.read_csv_kwargs)
    if "chunksize" in options or options.get("iterator"):
        raise ValueError(
            "ArtifactSpec read_csv_kwargs cannot control CSV iteration"
        )

    # Manifest validation needs only dimensions.  Reading every column as a
    # string prevents type inference warnings for sparse mixed-value columns,
    # while bounded chunks avoid loading a growing raw artifact all at once.
    options.setdefault("dtype", "string")
    header_options = dict(options)
    header_options["nrows"] = 0
    column_count = len(pd.read_csv(spec.path, **header_options).columns)
    row_count = sum(
        len(chunk)
        for chunk in pd.read_csv(spec.path, chunksize=50_000, **options)
    )
    return row_count, column_count


def artifact_sha256(spec: ArtifactSpec) -> str:
    if spec.read_csv_kwargs is None:
        return sha256_directory(spec.path)
    return file_sha256(spec.path)


def relative_artifact_path(path: Path, repository_root: Path) -> str:
    """Return a stable repository-relative path when one is available."""
    path = Path(path)
    repository_root = Path(repository_root)
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return path.as_posix()


def build_preparation_manifest(
    artifacts: list[ArtifactSpec] | tuple[ArtifactSpec, ...],
    *,
    repository_root: Path,
    skip_missing: bool,
) -> pd.DataFrame:
    """Build a deterministic schema-v7 manifest without writing it."""
    rows: list[dict[str, object]] = []
    for spec in artifacts:
        if not artifact_exists(spec):
            if skip_missing:
                continue
            raise FileNotFoundError(f"Cannot manifest missing artifact: {spec.path}")
        shape = artifact_shape(spec)
        rows.append({
            "Schema_Version": PREPARATION_SCHEMA_VERSION,
            "Stage": spec.stage,
            "Artifact": spec.artifact,
            "Relative_Path": relative_artifact_path(spec.path, repository_root),
            "SHA256": artifact_sha256(spec),
            "Rows": int(shape[0]),
            "Columns": int(shape[1]),
        })
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    stage_order = pd.CategoricalDtype(["raw", "prepared"], ordered=True)
    if not manifest.empty:
        manifest["Stage"] = manifest["Stage"].astype(stage_order)
        manifest = manifest.sort_values(
            ["Stage", "Artifact"], kind="stable"
        ).reset_index(drop=True)
        manifest["Stage"] = manifest["Stage"].astype("string")
    return manifest


def write_preparation_manifest(
    artifacts: list[ArtifactSpec] | tuple[ArtifactSpec, ...],
    *,
    manifest_path: Path,
    repository_root: Path,
    skip_missing: bool = False,
) -> pd.DataFrame:
    """Build and atomically persist a schema-v7 preparation manifest."""
    manifest = build_preparation_manifest(
        artifacts,
        repository_root=repository_root,
        skip_missing=skip_missing,
    )
    atomic_write_dataframe(
        manifest,
        manifest_path,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    return manifest


def validate_preparation_manifest(
    artifacts: list[ArtifactSpec] | tuple[ArtifactSpec, ...],
    *,
    manifest_path: Path,
    repository_root: Path,
    recovery_command: str,
    check_raw: bool = False,
    required_artifacts: Collection[str] | None = None,
    validate_raw_artifacts: Collection[str] = (),
) -> None:
    """Reject unknown, missing, stale, relocated, or incompatible artifacts.

    ``artifacts`` is the complete domain catalog. ``required_artifacts`` is the
    subset required by the requested preparation stage; catalogued later-stage
    rows may remain in a manifest produced by a fuller preparation run.
    ``validate_raw_artifacts`` identifies raw inputs consumed directly by a
    prepared-data reader without broadening validation to unrelated raw state.
    """
    manifest_path = Path(manifest_path)
    recovery = f"Run `{recovery_command}` first."
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Missing preparation manifest: {manifest_path}. {recovery}"
        )
    manifest = pd.read_csv(manifest_path, dtype={"Artifact": "string"})
    if list(manifest.columns) != MANIFEST_COLUMNS:
        raise RuntimeError(
            f"Invalid preparation manifest schema at {manifest_path}. {recovery}"
        )
    if manifest.empty or not manifest["Schema_Version"].eq(
        PREPARATION_SCHEMA_VERSION
    ).all():
        raise RuntimeError(
            f"Unsupported preparation manifest version at {manifest_path}. "
            f"{recovery}"
        )

    artifact_names = manifest["Artifact"].astype("string")
    if artifact_names.isna().any() or artifact_names.str.strip().eq("").any():
        raise RuntimeError(
            f"Preparation manifest has an empty artifact at {manifest_path}. "
            f"{recovery}"
        )
    if artifact_names.duplicated().any():
        raise RuntimeError(
            f"Preparation manifest has duplicate artifacts at {manifest_path}. "
            f"{recovery}"
        )

    catalog: dict[str, ArtifactSpec] = {}
    for spec in artifacts:
        if spec.artifact in catalog:
            raise ValueError(f"Duplicate artifact specification: {spec.artifact}")
        catalog[spec.artifact] = spec

    required = (
        set(catalog)
        if required_artifacts is None
        else {str(name) for name in required_artifacts}
    )
    invalid_required = sorted(required - set(catalog))
    if invalid_required:
        raise ValueError(
            f"Required artifacts are absent from the catalog: {invalid_required}"
        )
    selected_raw = {str(name) for name in validate_raw_artifacts}
    invalid_selected_raw = sorted(selected_raw - set(catalog))
    if invalid_selected_raw:
        raise ValueError(
            "Selected raw artifacts are absent from the catalog: "
            f"{invalid_selected_raw}"
        )
    non_raw = sorted(
        name for name in selected_raw if catalog[name].stage != "raw"
    )
    if non_raw:
        raise ValueError(
            f"Selected raw artifacts are not raw inputs: {non_raw}"
        )
    required |= selected_raw

    manifest_names = set(artifact_names.astype(str))
    unknown = sorted(manifest_names - set(catalog))
    if unknown:
        raise RuntimeError(
            f"Preparation manifest has unknown artifacts {unknown} at "
            f"{manifest_path}. {recovery}"
        )
    missing = sorted(required - manifest_names)
    if missing:
        raise RuntimeError(
            f"Preparation manifest is missing required artifacts {missing} at "
            f"{manifest_path}. {recovery}"
        )

    by_artifact = manifest.set_index("Artifact")

    def manifest_count(value: object) -> int:
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(value)
        return int(numeric)

    for artifact in sorted(manifest_names):
        spec = catalog[artifact]
        entry = by_artifact.loc[artifact]
        relative_path = relative_artifact_path(spec.path, repository_root)
        if (
            str(entry["Stage"]) != spec.stage
            or str(entry["Relative_Path"]) != relative_path
        ):
            raise RuntimeError(
                f"Prepared pipeline artifact is stale or relocated: {spec.path}. "
                f"{recovery}"
            )

        validate_contents = (
            spec.stage == "prepared"
            or check_raw
            or artifact in selected_raw
        )
        if not validate_contents:
            continue
        if not artifact_exists(spec):
            raise RuntimeError(
                f"Prepared pipeline artifact is missing: {spec.path}. {recovery}"
            )
        shape = artifact_shape(spec)
        try:
            matches = (
                str(entry["SHA256"]) == artifact_sha256(spec)
                and manifest_count(entry["Rows"]) == int(shape[0])
                and manifest_count(entry["Columns"]) == int(shape[1])
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise RuntimeError(
                f"Prepared pipeline artifact is stale or modified: {spec.path}. "
                f"{recovery}"
            )


__all__ = [
    "ACQUISITION_MANIFEST_COLUMNS",
    "ArtifactManifest",
    "ArtifactOrigin",
    "ArtifactSpec",
    "MANIFEST_COLUMNS",
    "PREPARATION_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "artifact_exists",
    "artifact_sha256",
    "artifact_shape",
    "build_preparation_manifest",
    "file_sha256",
    "read_manifests",
    "relative_artifact_path",
    "sha256_directory",
    "validate_exact_manifest_catalog",
    "validate_preparation_manifest",
    "validate_manifest_artifact",
    "write_manifests",
    "write_preparation_manifest",
]
