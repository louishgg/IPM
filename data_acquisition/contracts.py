"""Canonical, provider-neutral acquisition records.

Provider execution state and downstream data readiness are deliberately kept
separate.  A provider can successfully return observations (``ok``) while the
consumer's required dates remain only partially covered.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Mapping, Self

from portfolio_core.io import atomic_write_csv_rows


class ProviderStatus(StrEnum):
    """Terminal and retryable provider outcomes."""

    PENDING = "pending"
    OK = "ok"
    NO_DATA = "no_data"
    FAILED = "failed"


class ReadinessStatus(StrEnum):
    """Coverage of the requirement set consumed by preparation."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class CommandRequest:
    """Validated acquisition command passed from the CLI to domain handlers."""

    scope: str
    dataset: str
    tickers_file: Path | None
    refresh: bool
    dry_run: bool
    project_root: Path


def readiness_status(covered_count: int, required_count: int) -> ReadinessStatus:
    """Return the canonical readiness state for a coverage count pair."""

    if covered_count == required_count:
        return ReadinessStatus.COMPLETE
    if covered_count == 0:
        return ReadinessStatus.MISSING
    return ReadinessStatus.PARTIAL


IDENTITY_COLUMNS = (
    "Scope",
    "Dataset",
    "Asset_ID",
    "Provider",
    "Provider_Symbol",
    "Effective_Start",
    "Effective_End",
)

ACQUISITION_STATUS_COLUMNS = IDENTITY_COLUMNS + (
    "Status",
    "Requested_Start",
    "Requested_End",
    "Observation_Count",
    "Observation_Start",
    "Observation_End",
    "Attempted_At_UTC",
    "Client",
    "Client_Version",
    "HTTP_Status",
    "Error_Class",
    "Error_Message",
    "Migration_Note",
)

READINESS_COLUMNS = (
    "Scope",
    "Dataset",
    "Asset_ID",
    "Requirement_Set",
    "Required_Count",
    "Covered_Count",
    "Status",
    "Missing_Dates",
    "Contributing_Sources",
    "Checked_At_UTC",
)

def utc_timestamp(value: datetime | None = None) -> str:
    """Return a stable UTC timestamp suitable for durable CSV records."""

    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def read_asset_selection(path: Path) -> tuple[str, ...]:
    """Read one non-empty, normalized Asset_ID or Ticker selection column."""
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"Ticker request CSV is empty: {path}") from exc
        if len(header) != 1 or header[0] not in {"Asset_ID", "Ticker"}:
            raise ValueError(
                "Ticker request CSV must contain exactly one Asset_ID or "
                f"Ticker column: {path}"
            )
        values = set()
        for row in reader:
            if not row:
                continue
            if len(row) != 1:
                raise ValueError(
                    "Ticker request CSV must contain exactly one Asset_ID or "
                    f"Ticker column: {path}"
                )
            value = str(row[0]).strip().upper()
            if value:
                values.add(value)
    if not values:
        raise ValueError(f"Ticker request CSV is empty: {path}")
    return tuple(sorted(values))


def _text(value: object, *, field: str, required: bool = False) -> str:
    result = str(value).strip()
    if required and not result:
        raise ValueError(f"{field} must not be empty")
    if "\n" in result or "\r" in result:
        raise ValueError(f"{field} must be a single line")
    return result


def _iso_date(value: object, *, field: str, required: bool = False) -> str:
    result = _text(value, field=field, required=required)
    if result:
        try:
            result = date.fromisoformat(result).isoformat()
        except ValueError as exc:
            raise ValueError(f"{field} must use YYYY-MM-DD: {result!r}") from exc
    return result


def _utc(value: object, *, field: str, required: bool = False) -> str:
    result = _text(value, field=field, required=required)
    if not result:
        return result
    if not result.endswith("Z"):
        raise ValueError(f"{field} must be a UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(result[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid UTC timestamp: {result!r}") from exc
    return result


def _nonnegative_integer(value: object, *, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a nonnegative integer") from exc
    if result < 0 or str(value).strip() not in {str(result), f"{result}.0"}:
        raise ValueError(f"{field} must be a nonnegative integer")
    return result


def _http_status(value: object) -> str:
    result = _text(value, field="HTTP_Status")
    if result:
        try:
            code = int(result)
        except ValueError as exc:
            raise ValueError("HTTP_Status must be an integer or empty") from exc
        if not 100 <= code <= 599:
            raise ValueError("HTTP_Status must be between 100 and 599")
        result = str(code)
    return result


@dataclass(frozen=True, slots=True, order=True)
class AcquisitionIdentity:
    """One effective-dated provider identity in one acquisition scope."""

    scope: str
    dataset: str
    asset_id: str
    provider: str
    provider_symbol: str
    effective_start: str = ""
    effective_end: str = ""

    def __post_init__(self) -> None:
        for name in ("scope", "dataset", "asset_id", "provider", "provider_symbol"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field=name, required=True),
            )
        object.__setattr__(
            self,
            "effective_start",
            _iso_date(self.effective_start, field="effective_start"),
        )
        object.__setattr__(
            self,
            "effective_end",
            _iso_date(self.effective_end, field="effective_end"),
        )
        if (
            self.effective_start
            and self.effective_end
            and self.effective_start > self.effective_end
        ):
            raise ValueError("effective_start must not be after effective_end")

    @property
    def key(self) -> tuple[str, ...]:
        return tuple(self.to_row()[column] for column in IDENTITY_COLUMNS)

    def to_row(self) -> dict[str, str]:
        return {
            "Scope": self.scope,
            "Dataset": self.dataset,
            "Asset_ID": self.asset_id,
            "Provider": self.provider,
            "Provider_Symbol": self.provider_symbol,
            "Effective_Start": self.effective_start,
            "Effective_End": self.effective_end,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> Self:
        return cls(
            scope=row["Scope"],
            dataset=row["Dataset"],
            asset_id=row["Asset_ID"],
            provider=row["Provider"],
            provider_symbol=row["Provider_Symbol"],
            effective_start=row["Effective_Start"],
            effective_end=row["Effective_End"],
        )


@dataclass(frozen=True, slots=True)
class AcquisitionRequest:
    identity: AcquisitionIdentity
    requested_start: str = ""
    requested_end: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_start",
            _iso_date(self.requested_start, field="requested_start"),
        )
        object.__setattr__(
            self,
            "requested_end",
            _iso_date(self.requested_end, field="requested_end"),
        )
        if (
            self.requested_start
            and self.requested_end
            and self.requested_start > self.requested_end
        ):
            raise ValueError("requested_start must not be after requested_end")


@dataclass(frozen=True, slots=True)
class AcquisitionStatus:
    """Durable provider result; this record does not claim readiness."""

    identity: AcquisitionIdentity
    status: ProviderStatus = ProviderStatus.PENDING
    requested_start: str = ""
    requested_end: str = ""
    observation_count: int = 0
    observation_start: str = ""
    observation_end: str = ""
    attempted_at_utc: str = ""
    client: str = ""
    client_version: str = ""
    http_status: str = ""
    error_class: str = ""
    error_message: str = ""
    migration_note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ProviderStatus(self.status))
        for field_name in (
            "requested_start",
            "requested_end",
            "observation_start",
            "observation_end",
        ):
            object.__setattr__(
                self,
                field_name,
                _iso_date(getattr(self, field_name), field=field_name),
            )
        object.__setattr__(
            self,
            "observation_count",
            _nonnegative_integer(self.observation_count, field="observation_count"),
        )
        object.__setattr__(
            self,
            "attempted_at_utc",
            _utc(self.attempted_at_utc, field="attempted_at_utc"),
        )
        for field_name in (
            "client",
            "client_version",
            "error_class",
            "error_message",
            "migration_note",
        ):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field=field_name),
            )
        object.__setattr__(self, "http_status", _http_status(self.http_status))

        if self.requested_start and self.requested_end and self.requested_start > self.requested_end:
            raise ValueError("requested_start must not be after requested_end")
        if self.observation_start and self.observation_end and self.observation_start > self.observation_end:
            raise ValueError("observation_start must not be after observation_end")
        if self.status is ProviderStatus.OK:
            if self.observation_count <= 0:
                raise ValueError("ok provider status requires observations")
            if not self.observation_start or not self.observation_end:
                raise ValueError("ok provider status requires observation bounds")
            if self.error_class or self.error_message:
                raise ValueError("ok provider status cannot contain an error")
        elif self.observation_count != 0 or self.observation_start or self.observation_end:
            raise ValueError("non-ok provider status cannot contain observations")
        if self.status is ProviderStatus.FAILED and not (
            self.error_class and self.error_message
        ):
            raise ValueError("failed provider status requires a structured error")
        if self.status is ProviderStatus.PENDING and (
            self.attempted_at_utc or self.http_status or self.error_class or self.error_message
        ):
            raise ValueError("pending provider status cannot contain attempt results")

    @classmethod
    def pending(cls, request: AcquisitionRequest, *, migration_note: str = "") -> Self:
        return cls(
            identity=request.identity,
            status=ProviderStatus.PENDING,
            requested_start=request.requested_start,
            requested_end=request.requested_end,
            migration_note=migration_note,
        )

    def to_row(self) -> dict[str, str]:
        return {
            **self.identity.to_row(),
            "Status": self.status.value,
            "Requested_Start": self.requested_start,
            "Requested_End": self.requested_end,
            "Observation_Count": str(self.observation_count),
            "Observation_Start": self.observation_start,
            "Observation_End": self.observation_end,
            "Attempted_At_UTC": self.attempted_at_utc,
            "Client": self.client,
            "Client_Version": self.client_version,
            "HTTP_Status": self.http_status,
            "Error_Class": self.error_class,
            "Error_Message": self.error_message,
            "Migration_Note": self.migration_note,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> Self:
        return cls(
            identity=AcquisitionIdentity.from_row(row),
            status=ProviderStatus(str(row["Status"]).strip()),
            requested_start=row["Requested_Start"],
            requested_end=row["Requested_End"],
            observation_count=row["Observation_Count"],
            observation_start=row["Observation_Start"],
            observation_end=row["Observation_End"],
            attempted_at_utc=row["Attempted_At_UTC"],
            client=row["Client"],
            client_version=row["Client_Version"],
            http_status=row["HTTP_Status"],
            error_class=row["Error_Class"],
            error_message=row["Error_Message"],
            migration_note=row["Migration_Note"],
        )


@dataclass(frozen=True, slots=True)
class ReadinessRecord:
    scope: str
    dataset: str
    asset_id: str
    requirement_set: str
    required_count: int
    covered_count: int
    status: ReadinessStatus
    missing_dates: tuple[str, ...] = ()
    contributing_sources: tuple[str, ...] = ()
    checked_at_utc: str = ""

    def __post_init__(self) -> None:
        for field_name in ("scope", "dataset", "asset_id", "requirement_set"):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field=field_name, required=True),
            )
        object.__setattr__(self, "required_count", _nonnegative_integer(self.required_count, field="required_count"))
        object.__setattr__(self, "covered_count", _nonnegative_integer(self.covered_count, field="covered_count"))
        object.__setattr__(self, "status", ReadinessStatus(self.status))
        object.__setattr__(
            self,
            "missing_dates",
            tuple(_iso_date(item, field="missing_date", required=True) for item in self.missing_dates),
        )
        object.__setattr__(
            self,
            "contributing_sources",
            tuple(_text(item, field="contributing_source", required=True) for item in self.contributing_sources),
        )
        object.__setattr__(self, "checked_at_utc", _utc(self.checked_at_utc, field="checked_at_utc", required=True))
        if self.covered_count > self.required_count:
            raise ValueError("covered_count must not exceed required_count")
        expected = readiness_status(self.covered_count, self.required_count)
        if self.status is not expected:
            raise ValueError(
                f"readiness status must be {expected.value!r} for "
                f"{self.covered_count}/{self.required_count} coverage"
            )
        if self.status is ReadinessStatus.COMPLETE and self.missing_dates:
            raise ValueError("complete readiness cannot contain missing dates")

    def to_row(self) -> dict[str, str]:
        return {
            "Scope": self.scope,
            "Dataset": self.dataset,
            "Asset_ID": self.asset_id,
            "Requirement_Set": self.requirement_set,
            "Required_Count": str(self.required_count),
            "Covered_Count": str(self.covered_count),
            "Status": self.status.value,
            "Missing_Dates": json.dumps(self.missing_dates, separators=(",", ":")),
            "Contributing_Sources": json.dumps(self.contributing_sources, separators=(",", ":")),
            "Checked_At_UTC": self.checked_at_utc,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> Self:
        return cls(
            scope=row["Scope"],
            dataset=row["Dataset"],
            asset_id=row["Asset_ID"],
            requirement_set=row["Requirement_Set"],
            required_count=row["Required_Count"],
            covered_count=row["Covered_Count"],
            status=ReadinessStatus(str(row["Status"]).strip()),
            missing_dates=tuple(json.loads(str(row["Missing_Dates"]))),
            contributing_sources=tuple(json.loads(str(row["Contributing_Sources"]))),
            checked_at_utc=row["Checked_At_UTC"],
        )


def _read_records(path: Path, columns: tuple[str, ...], factory):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(
                f"{path} must have exactly {list(columns)}; "
                f"found {reader.fieldnames or []}"
            )
        return [factory(row) for row in reader]


def read_acquisition_statuses(path: Path) -> list[AcquisitionStatus]:
    records = _read_records(path, ACQUISITION_STATUS_COLUMNS, AcquisitionStatus.from_row)
    _require_unique(records, lambda item: item.identity.key, "acquisition identity")
    return sorted(records, key=lambda item: item.identity.key)


def write_acquisition_statuses(path: Path, records: Iterable[AcquisitionStatus]) -> None:
    values = list(records)
    _require_unique(values, lambda item: item.identity.key, "acquisition identity")
    atomic_write_csv_rows(
        (item.to_row() for item in sorted(values, key=lambda item: item.identity.key)),
        ACQUISITION_STATUS_COLUMNS,
        Path(path),
    )


def read_readiness(path: Path) -> list[ReadinessRecord]:
    records = _read_records(path, READINESS_COLUMNS, ReadinessRecord.from_row)
    _require_unique(
        records,
        lambda item: (item.scope, item.dataset, item.asset_id, item.requirement_set),
        "readiness key",
    )
    return sorted(records, key=lambda item: (item.scope, item.dataset, item.asset_id, item.requirement_set))


def write_readiness(path: Path, records: Iterable[ReadinessRecord]) -> None:
    values = list(records)
    _require_unique(
        values,
        lambda item: (item.scope, item.dataset, item.asset_id, item.requirement_set),
        "readiness key",
    )
    atomic_write_csv_rows(
        (
            item.to_row()
            for item in sorted(
                values,
                key=lambda item: (
                    item.scope,
                    item.dataset,
                    item.asset_id,
                    item.requirement_set,
                ),
            )
        ),
        READINESS_COLUMNS,
        Path(path),
    )


def _require_unique(values, key, label: str) -> None:
    seen = set()
    duplicates = set()
    for value in values:
        identity = key(value)
        if identity in seen:
            duplicates.add(identity)
        seen.add(identity)
    if duplicates:
        raise ValueError(f"duplicate {label} records: {sorted(duplicates)!r}")


__all__ = [
    "ACQUISITION_STATUS_COLUMNS",
    "IDENTITY_COLUMNS",
    "READINESS_COLUMNS",
    "AcquisitionIdentity",
    "AcquisitionRequest",
    "AcquisitionStatus",
    "CommandRequest",
    "ProviderStatus",
    "ReadinessRecord",
    "ReadinessStatus",
    "readiness_status",
    "read_acquisition_statuses",
    "read_asset_selection",
    "read_readiness",
    "utc_timestamp",
    "write_acquisition_statuses",
    "write_readiness",
]
