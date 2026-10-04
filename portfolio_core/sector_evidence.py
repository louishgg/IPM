"""Validated source evidence and artifact contracts for sector history."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import parse_qs, urlparse

import pandas as pd

from portfolio_core.artifacts import (
    ArtifactOrigin,
    validate_exact_manifest_catalog,
)


SECTOR_HISTORY_RELATIVE_DIR = Path("data/shared/provenance/sector_history")
SECTOR_ACQUISITION_SCOPE = "shared"
SECTOR_ACQUISITION_DATASET = "sectors"
WIKIPEDIA_SNAPSHOT_FILENAME = "wikipedia_sector_snapshots.csv"
NOTICE_LEDGER_FILENAME = "sp500_constituent_notices.csv"
STATUS_FILENAME = "acquisition_status.csv"
MANIFEST_FILENAME = "artifact_manifest.csv"

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_PAGE_URL = "https://en.wikipedia.org/w/index.php"
SP500_PAGE_TITLE = "List of S&P 500 companies"

SNAPSHOT_COLUMNS = (
    "Requirement_Date",
    "Cutoff_UTC",
    "Revision_ID",
    "Revision_Timestamp_UTC",
    "Wikipedia_Ticker",
    "Company_Name",
    "Raw_Sector",
    "Source_URL",
)
NOTICE_COLUMNS = (
    "Notice_ID",
    "Published_Date",
    "Effective_Date",
    "Index_Name",
    "Action",
    "Ticker",
    "Sector",
    "Source_URL",
    "Review_Status",
)
NOTICE_ACTIONS = ("Addition", "Deletion")

GICS_SECTOR_LABELS: Mapping[str, str] = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}
_SECTOR_ALIASES: Mapping[str, tuple[str, str]] = {
    "energy": ("10", "Energy"),
    "materials": ("15", "Materials"),
    "industrials": ("20", "Industrials"),
    "consumer discretionary": ("25", "Consumer Discretionary"),
    "consumer staples": ("30", "Consumer Staples"),
    "health care": ("35", "Health Care"),
    "healthcare": ("35", "Health Care"),
    "financials": ("40", "Financials"),
    "information technology": ("45", "Information Technology"),
    "communication services": ("50", "Communication Services"),
    "telecommunication services": ("50", "Telecommunication Services"),
    "telecommunications services": ("50", "Telecommunication Services"),
    "utilities": ("55", "Utilities"),
    "real estate": ("60", "Real Estate"),
}
_FOOTNOTE_PATTERN = re.compile(r"\[[^\]]+\]")
_WIKIPEDIA_PAGE = urlparse(WIKIPEDIA_PAGE_URL)
_OFFICIAL_SP_GLOBAL_NOTICE_HOSTS = frozenset({"press.spglobal.com"})


@dataclass(frozen=True, slots=True)
class SectorHistoryPaths:
    """Filesystem contract for shared sector-history acquisition."""

    directory: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))

    @classmethod
    def from_project_root(cls, project_root: Path) -> "SectorHistoryPaths":
        return cls(Path(project_root) / SECTOR_HISTORY_RELATIVE_DIR)

    @property
    def snapshots_csv(self) -> Path:
        return self.directory / WIKIPEDIA_SNAPSHOT_FILENAME

    @property
    def notices_csv(self) -> Path:
        return self.directory / NOTICE_LEDGER_FILENAME

    @property
    def acquisition_status_csv(self) -> Path:
        return self.directory / STATUS_FILENAME

    @property
    def artifact_manifest_csv(self) -> Path:
        return self.directory / MANIFEST_FILENAME

    @property
    def acquisition_artifacts(self) -> tuple[Path, ...]:
        return tuple(
            path
            for _, path, _ in sector_acquisition_artifact_catalog(self)
        )


@dataclass(frozen=True, slots=True)
class SectorEvidenceBundle:
    snapshots: pd.DataFrame
    notices: pd.DataFrame


def sector_acquisition_artifact_catalog(
    paths: SectorHistoryPaths,
) -> tuple[tuple[str, Path, ArtifactOrigin], ...]:
    """Return the complete acquired-evidence and operational-state inventory."""
    return (
        *sector_analytical_evidence_catalog(paths),
        (
            "sector_history_status",
            paths.acquisition_status_csv,
            ArtifactOrigin.DOWNLOADED,
        ),
    )


def sector_analytical_evidence_catalog(
    paths: SectorHistoryPaths,
) -> tuple[tuple[str, Path, ArtifactOrigin], ...]:
    """Return only source files whose bytes affect prepared assignments."""
    return (
        ("sector_history_snapshots", paths.snapshots_csv, ArtifactOrigin.DOWNLOADED),
        ("sector_history_notices", paths.notices_csv, ArtifactOrigin.MANUAL),
    )


def sector_acquisition_artifact_origins(
    paths: SectorHistoryPaths,
) -> dict[Path, ArtifactOrigin]:
    return {
        path: origin
        for _, path, origin in sector_acquisition_artifact_catalog(paths)
    }


def clean_sector_text(value: object) -> str:
    """Normalize whitespace in sector evidence and prepared identifiers."""

    return " ".join(str(value).replace("\xa0", " ").strip().split())


def clean_wikipedia_ticker(value: object) -> str:
    """Preserve source punctuation while removing presentation footnotes."""
    return _FOOTNOTE_PATTERN.sub("", clean_sector_text(value)).strip().upper()


def normalize_gics_sector(value: object) -> tuple[str, str]:
    """Return the official two-digit key and source-era display label."""
    raw = clean_sector_text(value)
    try:
        return _SECTOR_ALIASES[raw.casefold()]
    except KeyError as exc:
        raise ValueError(f"Unsupported GICS sector label: {raw!r}") from exc


def _validate_wikipedia_revision_url(
    value: str,
    *,
    revision_id: str,
) -> None:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != _WIKIPEDIA_PAGE.hostname
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != _WIKIPEDIA_PAGE.path
        or parsed.fragment
    ):
        raise ValueError(
            "Wikipedia Source_URL values must be canonical HTTPS revision URLs"
        )
    query = parse_qs(parsed.query, keep_blank_values=True)
    if not set(query).issubset({"title", "oldid"}):
        raise ValueError(
            "Wikipedia Source_URL query must contain only title and oldid"
        )
    revision_title = SP500_PAGE_TITLE.replace(" ", "_")
    if query.get("title") != [revision_title]:
        raise ValueError(
            "Wikipedia Source_URL title must identify "
            f"{revision_title}"
        )
    if query.get("oldid") != [revision_id]:
        raise ValueError(
            "Wikipedia Source_URL oldid must exactly match Revision_ID"
        )


def _validate_official_sp_global_url(value: str) -> None:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _OFFICIAL_SP_GLOBAL_NOTICE_HOSTS
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "S&P notice Source_URL values must use an approved official "
            "S&P Global HTTPS host"
        )


def validate_sector_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize one in-memory Wikipedia snapshot bundle."""
    if tuple(frame.columns) != SNAPSHOT_COLUMNS:
        raise ValueError(
            f"Wikipedia sector snapshots must have columns {list(SNAPSHOT_COLUMNS)}; "
            f"found {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("Wikipedia sector snapshots are empty")
    frame = frame.copy()

    def requirement_date_text(value: object) -> object:
        if isinstance(value, pd.Timestamp):
            if value != value.normalize():
                raise ValueError(
                    "Wikipedia requirement dates must not contain a time"
                )
            return value.strftime("%Y-%m-%d")
        return value

    frame["Requirement_Date"] = frame["Requirement_Date"].map(
        requirement_date_text
    )
    for column in SNAPSHOT_COLUMNS:
        frame[column] = frame[column].map(clean_sector_text)
    frame["Wikipedia_Ticker"] = frame["Wikipedia_Ticker"].map(
        clean_wikipedia_ticker
    )
    if frame[list(SNAPSHOT_COLUMNS)].eq("").any().any():
        raise ValueError("Wikipedia sector snapshots contain empty required values")
    frame["Requirement_Date"] = pd.to_datetime(
        frame["Requirement_Date"], format="%Y-%m-%d", errors="raise"
    )
    cutoffs = pd.to_datetime(frame["Cutoff_UTC"], utc=True, errors="raise")
    revisions = pd.to_datetime(
        frame["Revision_Timestamp_UTC"], utc=True, errors="raise"
    )
    expected_cutoffs = frame["Requirement_Date"].dt.tz_localize("UTC")
    if (cutoffs != expected_cutoffs).any():
        raise ValueError(
            "Sector snapshot cutoffs must be midnight UTC on the requirement date"
        )
    if (revisions >= cutoffs).any():
        raise ValueError("Wikipedia sector snapshots contain a future revision")
    if not frame["Revision_ID"].str.fullmatch(r"[1-9][0-9]*").all():
        raise ValueError("Wikipedia Revision_ID values must be positive integers")
    for source_url, revision_id in frame[
        ["Source_URL", "Revision_ID"]
    ].drop_duplicates().itertuples(index=False, name=None):
        _validate_wikipedia_revision_url(source_url, revision_id=revision_id)
    duplicate = frame.duplicated(
        ["Requirement_Date", "Wikipedia_Ticker"], keep=False
    )
    if duplicate.any():
        values = frame.loc[
            duplicate, ["Requirement_Date", "Wikipedia_Ticker"]
        ].to_dict("records")
        raise ValueError(f"Duplicate Wikipedia ticker rows: {values[:20]}")
    per_date = frame.groupby("Requirement_Date", sort=True).agg(
        Revision_ID=("Revision_ID", "nunique"),
        Revision_Timestamp_UTC=("Revision_Timestamp_UTC", "nunique"),
        Cutoff_UTC=("Cutoff_UTC", "nunique"),
        Source_URL=("Source_URL", "nunique"),
    )
    if (per_date != 1).any().any():
        raise ValueError(
            "Each requirement date must resolve to exactly one revision, "
            "cutoff, and source URL"
        )
    for sector in frame["Raw_Sector"].unique():
        normalize_gics_sector(sector)
    return frame.sort_values(
        ["Requirement_Date", "Wikipedia_Ticker"], kind="stable"
    ).reset_index(drop=True)


def load_sector_snapshots(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Wikipedia sector snapshots: {path}. Run a sector "
            "acquisition command first."
        )
    return validate_sector_snapshots(
        pd.read_csv(path, keep_default_na=False, dtype=str)
    )


def validate_sector_notices(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize one in-memory reviewed S&P notice ledger."""
    if tuple(frame.columns) != NOTICE_COLUMNS:
        raise ValueError(
            f"S&P notice ledger must have columns {list(NOTICE_COLUMNS)}; "
            f"found {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("S&P constituent notice ledger is empty")
    frame = frame.copy()
    for column in NOTICE_COLUMNS:
        frame[column] = frame[column].map(clean_sector_text)
    frame["Ticker"] = frame["Ticker"].map(clean_wikipedia_ticker)
    if frame[list(NOTICE_COLUMNS)].eq("").any().any():
        raise ValueError("S&P constituent notice ledger contains empty values")
    frame["Published_Date"] = pd.to_datetime(
        frame["Published_Date"], format="%Y-%m-%d", errors="raise"
    )
    frame["Effective_Date"] = pd.to_datetime(
        frame["Effective_Date"], format="%Y-%m-%d", errors="raise"
    )
    if (frame["Effective_Date"] < frame["Published_Date"]).any():
        raise ValueError("S&P notice effective dates cannot precede publication")
    unknown_indices = sorted(
        set(frame["Index_Name"]) - {"S&P 500", "S&P SmallCap 600"}
    )
    if unknown_indices:
        raise ValueError(
            f"S&P notice ledger contains unsupported indices: {unknown_indices}"
        )
    unknown_actions = sorted(set(frame["Action"]) - set(NOTICE_ACTIONS))
    if unknown_actions:
        raise ValueError(
            f"S&P notice ledger contains unsupported actions: {unknown_actions}"
        )
    if frame.duplicated(["Notice_ID", "Ticker", "Action"]).any():
        raise ValueError(
            "S&P notice ledger contains duplicate notice/ticker/action rows"
        )
    for source_url in frame["Source_URL"].unique():
        _validate_official_sp_global_url(source_url)
    allowed_reviews = {"approved", "unreviewed"}
    unknown_reviews = sorted(set(frame["Review_Status"]) - allowed_reviews)
    if unknown_reviews:
        raise ValueError(f"Unknown S&P notice review statuses: {unknown_reviews}")
    if frame["Review_Status"].ne("approved").any():
        rows = frame.loc[
            frame["Review_Status"].ne("approved"),
            ["Notice_ID", "Ticker"],
        ].to_dict("records")
        raise ValueError(f"Unreviewed S&P notice rows are not usable: {rows}")
    for sector in frame["Sector"].unique():
        normalize_gics_sector(sector)
    return frame.sort_values(
        ["Published_Date", "Notice_ID", "Ticker", "Action"], kind="stable"
    ).reset_index(drop=True)


def load_sector_notices(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing reviewed S&P constituent notice ledger: {path}"
        )
    return validate_sector_notices(
        pd.read_csv(path, keep_default_na=False, dtype=str)
    )


def validate_sector_acquisition_manifest(
    paths: SectorHistoryPaths,
    *,
    repository_root: Path,
    validate_artifacts: frozenset[str] | None = None,
) -> None:
    """Validate the exact acquisition catalog and selected artifact bytes."""
    repository_root = Path(repository_root)
    catalog = sector_acquisition_artifact_catalog(paths)
    paths_by_name = {
        name: path.relative_to(repository_root).as_posix()
        for name, path, _ in catalog
    }
    if validate_artifacts is not None:
        unknown = sorted(set(validate_artifacts) - set(paths_by_name))
        if unknown:
            raise ValueError(
                f"Unknown sector-history manifest artifacts: {unknown}"
            )
    try:
        validate_exact_manifest_catalog(
            paths.artifact_manifest_csv,
            scope=SECTOR_ACQUISITION_SCOPE,
            dataset=SECTOR_ACQUISITION_DATASET,
            expected_origins={
                paths_by_name[name]: origin for name, _, origin in catalog
            },
            base_dir=repository_root,
            validate_artifacts=(
                None
                if validate_artifacts is None
                else frozenset(paths_by_name[name] for name in validate_artifacts)
            ),
        )
    except ValueError as exc:
        if str(exc).startswith(f"Manifest {paths.artifact_manifest_csv} contains"):
            raise ValueError(
                "Sector-history manifest must list exactly the canonical "
                "acquisition artifact catalog"
            ) from exc
        raise


def load_sector_evidence(
    paths: SectorHistoryPaths,
    *,
    repository_root: Path,
) -> SectorEvidenceBundle:
    """Load the two source files consumed by sector assignment preparation."""
    validate_sector_acquisition_manifest(
        paths,
        repository_root=repository_root,
        validate_artifacts=frozenset(
            name
            for name, _, _ in sector_analytical_evidence_catalog(paths)
        ),
    )
    return SectorEvidenceBundle(
        snapshots=load_sector_snapshots(paths.snapshots_csv),
        notices=load_sector_notices(paths.notices_csv),
    )


__all__ = [
    "GICS_SECTOR_LABELS",
    "MANIFEST_FILENAME",
    "NOTICE_ACTIONS",
    "NOTICE_COLUMNS",
    "NOTICE_LEDGER_FILENAME",
    "SECTOR_ACQUISITION_DATASET",
    "SECTOR_ACQUISITION_SCOPE",
    "SECTOR_HISTORY_RELATIVE_DIR",
    "SNAPSHOT_COLUMNS",
    "SP500_PAGE_TITLE",
    "STATUS_FILENAME",
    "SectorEvidenceBundle",
    "SectorHistoryPaths",
    "WIKIPEDIA_API_URL",
    "WIKIPEDIA_PAGE_URL",
    "WIKIPEDIA_SNAPSHOT_FILENAME",
    "clean_sector_text",
    "clean_wikipedia_ticker",
    "load_sector_evidence",
    "load_sector_notices",
    "load_sector_snapshots",
    "normalize_gics_sector",
    "sector_acquisition_artifact_catalog",
    "sector_acquisition_artifact_origins",
    "sector_analytical_evidence_catalog",
    "validate_sector_acquisition_manifest",
    "validate_sector_notices",
    "validate_sector_snapshots",
]
