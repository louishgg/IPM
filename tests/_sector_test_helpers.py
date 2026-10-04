"""Shared deterministic fixtures for sector module tests."""

from pathlib import Path

import pandas as pd

import data_acquisition.sector_acquisition_artifacts as sector_artifacts_module
from data_acquisition.contracts import (
    AcquisitionIdentity,
    AcquisitionStatus,
    ProviderStatus,
    write_acquisition_statuses,
)
from data_acquisition.sector_acquisition_planning import (
    SECTOR_ACQUISITION_REQUIREMENT_COLUMNS,
    validate_sector_acquisition_requirements,
)
from portfolio_core.artifacts import (
    ArtifactManifest,
    ArtifactOrigin,
    write_manifests,
)
from portfolio_core.sector_assignments import (
    ASSIGNMENT_COLUMNS,
    ASSIGNMENT_REQUIREMENT_COLUMNS,
    validate_sector_assignments,
)
from portfolio_core.sector_evidence import (
    NOTICE_COLUMNS,
    SNAPSHOT_COLUMNS,
    SectorHistoryPaths,
    sector_acquisition_artifact_catalog,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SYNTHETIC_SECTORS = (
    ("10", "Energy"),
    ("15", "Materials"),
    ("20", "Industrials"),
    ("25", "Consumer Discretionary"),
    ("30", "Consumer Staples"),
    ("35", "Health Care"),
)


def _synthetic_sector_assignments(
    dates: pd.DatetimeIndex,
    asset_ids: list[str],
) -> pd.DataFrame:
    rows = []
    for date in dates:
        for index, asset_id in enumerate(asset_ids):
            code, sector = _SYNTHETIC_SECTORS[
                index % len(_SYNTHETIC_SECTORS)
            ]
            rows.append(
                {
                    "As_Of_Date": date,
                    "Asset_ID": asset_id,
                    "GICS_Sector_Code": code,
                    "Sector": sector,
                    "Source_Type": "Wikipedia",
                    "Source_Reference": f"synthetic-{date:%Y%m%d}",
                    "Source_Symbol": asset_id,
                    "Resolution_Method": "exact_wikipedia_symbol",
                }
            )
    return validate_sector_assignments(
        pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)
    )


def _write_sector_manifest(paths: SectorHistoryPaths, project_root: Path) -> None:
    write_manifests(
        paths.artifact_manifest_csv,
        sector_artifacts_module._build_manifest_records(
            paths,
            project_root=project_root,
            captured_at_utc="2026-08-05T12:00:00Z",
            artifact_sources={
                path: path for path in paths.acquisition_artifacts
            },
        ),
    )


def _write_sector_history_manifest(paths) -> None:
    history = paths.sector_history
    manifests = []
    for _, artifact_path, origin in sector_acquisition_artifact_catalog(history):
        manifests.append(
            ArtifactManifest.from_artifact(
                artifact_path,
                scope="shared",
                dataset="sectors",
                origin=origin,
                captured_at_utc=(
                    ""
                    if origin is ArtifactOrigin.MANUAL
                    else "2026-01-01T00:00:00Z"
                ),
                artifact_path=artifact_path.relative_to(
                    paths.project_root
                ).as_posix(),
            )
        )
    write_manifests(history.artifact_manifest_csv, manifests)


def _write_synthetic_sector_history(
    paths,
    requirement_dates,
    source_sectors: dict[str, str],
) -> None:
    history = paths.sector_history
    history.directory.mkdir(parents=True, exist_ok=True)
    dates = pd.DatetimeIndex(requirement_dates).normalize()

    snapshot_rows = []
    statuses = []
    for offset, requirement_date in enumerate(dates, start=1):
        date_text = requirement_date.strftime("%Y-%m-%d")
        cutoff = f"{date_text}T00:00:00Z"
        revision_id = str(1_000 + offset)
        revision_timestamp = (
            requirement_date - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%dT12:00:00Z")
        source_url = (
            "https://en.wikipedia.org/w/index.php?title="
            f"List_of_S%26P_500_companies&oldid={revision_id}"
        )
        for ticker, sector in source_sectors.items():
            snapshot_rows.append(
                {
                    "Requirement_Date": date_text,
                    "Cutoff_UTC": cutoff,
                    "Revision_ID": revision_id,
                    "Revision_Timestamp_UTC": revision_timestamp,
                    "Wikipedia_Ticker": ticker,
                    "Company_Name": f"{ticker} Incorporated",
                    "Raw_Sector": sector,
                    "Source_URL": source_url,
                }
            )
        identity = AcquisitionIdentity(
            "shared",
            "sectors",
            "S&P 500",
            "wikipedia",
            "List of S&P 500 companies",
            date_text,
            date_text,
        )
        statuses.append(
            AcquisitionStatus(
                identity=identity,
                status=ProviderStatus.OK,
                requested_start=date_text,
                requested_end=date_text,
                observation_count=len(source_sectors),
                observation_start=date_text,
                observation_end=date_text,
                attempted_at_utc="2026-01-01T00:00:00Z",
                client="synthetic-test-client",
                client_version="1",
                http_status="200",
            )
        )

    pd.DataFrame(snapshot_rows, columns=SNAPSHOT_COLUMNS).to_csv(
        history.snapshots_csv,
        index=False,
    )
    pd.DataFrame(
        [
            {
                "Notice_ID": "SPDJI-2030-01-01-SYNTHETIC",
                "Published_Date": "2030-01-01",
                "Effective_Date": "2030-01-02",
                "Index_Name": "S&P 500",
                "Action": "Addition",
                "Ticker": "NEW",
                "Sector": "Industrials",
                "Source_URL": "https://press.spglobal.com/synthetic-test-notice",
                "Review_Status": "approved",
            }
        ],
        columns=NOTICE_COLUMNS,
    ).to_csv(history.notices_csv, index=False)
    write_acquisition_statuses(history.acquisition_status_csv, statuses)
    _write_sector_history_manifest(paths)


def _snapshots(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    values = []
    revision_by_date: dict[str, int] = {}
    for date, ticker, sector in rows:
        revision_id = revision_by_date.setdefault(
            date,
            100 + len(revision_by_date) + 1,
        )
        cutoff = f"{date}T00:00:00Z"
        previous = (pd.Timestamp(date) - pd.Timedelta(days=1)).strftime(
            "%Y-%m-%dT12:00:00Z"
        )
        values.append(
            {
                "Requirement_Date": date,
                "Cutoff_UTC": cutoff,
                "Revision_ID": str(revision_id),
                "Revision_Timestamp_UTC": previous,
                "Wikipedia_Ticker": ticker,
                "Company_Name": f"{ticker} Company",
                "Raw_Sector": sector,
                "Source_URL": (
                    "https://en.wikipedia.org/w/index.php?title="
                    "List_of_S%26P_500_companies&oldid="
                    f"{revision_id}"
                ),
            }
        )
    return pd.DataFrame(values, columns=SNAPSHOT_COLUMNS)

def _requirements(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=ASSIGNMENT_REQUIREMENT_COLUMNS)

def _notices(rows: list[dict[str, str]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows or [], columns=NOTICE_COLUMNS)

def _notice(**overrides: str) -> dict[str, str]:
    row = {
        "Notice_ID": "NOTICE-1",
        "Published_Date": "2020-01-10",
        "Effective_Date": "2020-01-20",
        "Index_Name": "S&P 500",
        "Action": "Addition",
        "Ticker": "NEW",
        "Sector": "Industrials",
        "Source_URL": "https://press.spglobal.com/example",
        "Review_Status": "approved",
    }
    row.update(overrides)
    return row

def _acquisition_requirements(scope: str = "backtest") -> pd.DataFrame:
    return validate_sector_acquisition_requirements(
        pd.DataFrame(
            [
                (
                    scope,
                    "2020-01-31",
                    "2020-01-31T00:00:00Z",
                    "AAA",
                    "AAA",
                )
            ],
            columns=SECTOR_ACQUISITION_REQUIREMENT_COLUMNS,
        )
    )
