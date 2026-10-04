"""Offline contract for the shared fja05680 membership provenance.

The historical-components file is the canonical effective-dated membership
source.  The changes and interval files are independent representations used
to reject incomplete, internally inconsistent, or future-leaking inputs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from portfolio_core.artifacts import file_sha256


MEMBERSHIP_SOURCE_URL = "https://github.com/fja05680/sp500"
MEMBERSHIP_PROVENANCE_LABEL = (
    "post-competition retrospective effective-date reconstruction"
)
MEMBERSHIP_COMPONENTS_FILENAME = "S&P 500 Historical Components & Changes (Updated).csv"
MEMBERSHIP_CHANGES_FILENAME = "sp500_changes_since_2019.csv"
MEMBERSHIP_INTERVALS_FILENAME = "sp500_ticker_start_end.csv"
MEMBERSHIP_MANIFEST_FILENAME = "source_manifest.csv"

MANIFEST_COLUMNS = [
    "File",
    "Source_URL",
    "Recorded_At",
    "SHA256",
    "Rows",
    "Maximum_Date",
    "Provenance_Label",
]


@dataclass(frozen=True, slots=True)
class MembershipSourcePaths:
    """The one project-level location of all fja05680 source evidence."""

    directory: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))

    @property
    def components_csv(self) -> Path:
        return self.directory / MEMBERSHIP_COMPONENTS_FILENAME

    @property
    def changes_csv(self) -> Path:
        return self.directory / MEMBERSHIP_CHANGES_FILENAME

    @property
    def intervals_csv(self) -> Path:
        return self.directory / MEMBERSHIP_INTERVALS_FILENAME

    @property
    def manifest_csv(self) -> Path:
        return self.directory / MEMBERSHIP_MANIFEST_FILENAME

    @property
    def data_files(self) -> tuple[Path, Path, Path]:
        return (self.components_csv, self.changes_csv, self.intervals_csv)


def parse_ticker_list(value: object) -> tuple[str, ...]:
    """Parse one canonical comma-separated basket without losing share classes."""
    if pd.isna(value):
        return ()
    tickers = tuple(
        item.strip().upper() for item in str(value).split(",") if item.strip()
    )
    if len(tickers) != len(set(tickers)):
        raise ValueError("Constituent row contains duplicate tickers")
    return tickers


def _load_updated(path: Path) -> pd.DataFrame:
    updated = pd.read_csv(path, keep_default_na=False)
    if list(updated.columns) != ["date", "tickers"]:
        raise ValueError("Updated constituent CSV must contain exactly date,tickers")
    updated["date"] = pd.to_datetime(updated["date"], errors="raise")
    if updated["date"].duplicated().any() or not updated["date"].is_monotonic_increasing:
        raise ValueError("Updated constituent dates must be unique and ordered")
    updated["members"] = updated["tickers"].map(parse_ticker_list)
    if updated["members"].map(len).eq(0).any():
        raise ValueError("Updated constituent history contains an empty basket")
    return updated


def _load_changes(path: Path) -> pd.DataFrame:
    changes = pd.read_csv(path, keep_default_na=False)
    if list(changes.columns) != ["date", "add", "remove"]:
        raise ValueError("Changes CSV must contain exactly date,add,remove")
    changes["date"] = pd.to_datetime(changes["date"], errors="raise")
    if changes["date"].duplicated().any() or not changes["date"].is_monotonic_increasing:
        raise ValueError("Changes dates must be unique and ordered")
    changes["added"] = changes["add"].map(parse_ticker_list)
    changes["removed"] = changes["remove"].map(parse_ticker_list)
    for row in changes.itertuples(index=False):
        if not row.added and not row.removed:
            raise ValueError(f"Empty change row at {row.date.date()}")
        overlap = set(row.added) & set(row.removed)
        if overlap:
            raise ValueError(
                f"Change row adds and removes the same tickers at "
                f"{row.date.date()}: {sorted(overlap)}"
            )
    return changes


def _load_intervals(path: Path) -> pd.DataFrame:
    intervals = pd.read_csv(path, keep_default_na=False)
    expected = ["ticker", "start_date", "end_date"]
    if list(intervals.columns) != expected:
        raise ValueError(f"Ticker interval CSV must contain exactly {expected}")
    intervals["ticker"] = intervals["ticker"].astype(str).str.strip().str.upper()
    if intervals["ticker"].eq("").any():
        raise ValueError("Ticker interval CSV contains an empty ticker")
    intervals["start_date"] = pd.to_datetime(
        intervals["start_date"], errors="raise"
    )
    intervals["end_date"] = pd.to_datetime(
        intervals["end_date"].replace("", pd.NA), errors="coerce"
    )
    if intervals[["ticker", "start_date"]].duplicated().any():
        raise ValueError("Ticker interval starts must be unique per ticker")
    invalid = intervals["end_date"].notna() & intervals["end_date"].le(
        intervals["start_date"]
    )
    if invalid.any():
        raise ValueError("Ticker interval end dates must follow start dates")
    ordered = intervals.sort_values(["ticker", "start_date"], kind="stable")
    prior_end = ordered.groupby("ticker")["end_date"].shift()
    has_prior = ordered.groupby("ticker").cumcount().gt(0)
    overlaps = has_prior & (
        prior_end.isna() | ordered["start_date"].lt(prior_end)
    )
    if overlaps.any():
        raise ValueError("Ticker interval rows overlap")
    return intervals


def _validate_changes(updated: pd.DataFrame, changes: pd.DataFrame) -> None:
    history_dates = set(updated["date"])
    missing_dates = sorted(set(changes["date"]) - history_dates)
    if missing_dates:
        raise ValueError(f"Changes dates absent from membership: {missing_dates}")

    change_by_date = {
        row.date: (set(row.added), set(row.removed))
        for row in changes.itertuples(index=False)
    }
    first_change = changes["date"].min()
    for position in range(1, len(updated)):
        current_row = updated.iloc[position]
        current_date = current_row["date"]
        if current_date < first_change:
            continue
        previous = set(updated.iloc[position - 1]["members"])
        current = set(current_row["members"])
        actual = (current - previous, previous - current)
        expected = change_by_date.get(current_date)
        if expected is None:
            if actual != (set(), set()):
                raise ValueError(
                    f"Membership changes without a changes row at "
                    f"{current_date.date()}"
                )
            continue
        if actual != expected:
            raise ValueError(
                f"Changes CSV does not reconcile at {current_date.date()}"
            )


def _validate_intervals(updated: pd.DataFrame, intervals: pd.DataFrame) -> None:
    history_dates = set(updated["date"])
    boundaries = set(intervals["start_date"])
    boundaries.update(intervals["end_date"].dropna())
    missing_boundaries = sorted(boundaries - history_dates)
    if missing_boundaries:
        raise ValueError(
            "Ticker interval boundaries are absent from membership history: "
            f"{missing_boundaries}"
        )

    for row in updated.itertuples(index=False):
        active = intervals.loc[
            intervals["start_date"].le(row.date)
            & (intervals["end_date"].isna() | intervals["end_date"].gt(row.date)),
            "ticker",
        ]
        if set(active) != set(row.members):
            raise ValueError(
                f"Ticker intervals do not reconcile at {row.date.date()}"
            )


def _maximum_date(path: Path, frame: pd.DataFrame) -> pd.Timestamp:
    if path.name == MEMBERSHIP_COMPONENTS_FILENAME or path.name == MEMBERSHIP_CHANGES_FILENAME:
        return pd.to_datetime(frame["date"], errors="raise").max()
    starts = pd.to_datetime(frame["start_date"], errors="raise")
    ends = pd.to_datetime(frame["end_date"], errors="coerce")
    return max(starts.max(), ends.max())


def validate_membership_source_manifest(
    paths: MembershipSourcePaths,
) -> pd.DataFrame:
    """Verify the immutable source manifest against the exact local bytes."""
    if not paths.manifest_csv.is_file():
        raise FileNotFoundError(f"Missing fja05680 source manifest: {paths.manifest_csv}")
    manifest = pd.read_csv(paths.manifest_csv, keep_default_na=False)
    if list(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError(f"fja05680 source manifest must contain exactly {MANIFEST_COLUMNS}")
    if manifest["File"].duplicated().any():
        raise ValueError("fja05680 source manifest contains duplicate file rows")
    expected_files = {path.name for path in paths.data_files}
    if set(manifest["File"]) != expected_files:
        raise ValueError("fja05680 source manifest does not list exactly the three inputs")
    pd.to_datetime(manifest["Recorded_At"], format="%Y-%m-%d", errors="raise")
    manifest_by_file = manifest.set_index("File")
    for path in paths.data_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing fja05680 raw input: {path}")
        row = manifest_by_file.loc[path.name]
        frame = pd.read_csv(path, keep_default_na=False)
        if row["Source_URL"] != MEMBERSHIP_SOURCE_URL:
            raise ValueError(f"Unexpected fja05680 source URL for {path.name}")
        if row["Provenance_Label"] != MEMBERSHIP_PROVENANCE_LABEL:
            raise ValueError(f"Unexpected provenance label for {path.name}")
        if row["SHA256"] != file_sha256(path):
            raise ValueError(f"fja05680 source hash mismatch for {path.name}")
        if int(row["Rows"]) != len(frame):
            raise ValueError(f"fja05680 source row-count mismatch for {path.name}")
        maximum_date = _maximum_date(path, frame)
        actual_max = "" if pd.isna(maximum_date) else maximum_date.strftime("%Y-%m-%d")
        if row["Maximum_Date"] != actual_max:
            raise ValueError(f"fja05680 maximum-date mismatch for {path.name}")
    return manifest


def validate_membership_sources(
    paths: MembershipSourcePaths,
    *,
    required_through: object | None = None,
    validate_manifest: bool = True,
) -> pd.DataFrame:
    """Return canonical history only after all three representations agree."""
    for path in paths.data_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing fja05680 raw input: {path}")
    updated = _load_updated(paths.components_csv)
    changes = _load_changes(paths.changes_csv)
    intervals = _load_intervals(paths.intervals_csv)
    _validate_changes(updated, changes)
    _validate_intervals(updated, intervals)
    if required_through is not None:
        cutoff = pd.Timestamp(required_through)
        if updated["date"].max() < cutoff:
            raise ValueError(
                "Updated constituent history does not cover the required date "
                f"{cutoff.date()}"
            )
    if validate_manifest:
        validate_membership_source_manifest(paths)
    return updated


def membership_asof(
    history: pd.DataFrame,
    cutoff: object,
) -> tuple[pd.Timestamp, tuple[str, ...]]:
    """Select the latest effective basket at or before a causal cutoff."""
    timestamp = pd.Timestamp(cutoff)
    rows = history.loc[history["date"].le(timestamp)]
    if rows.empty:
        raise ValueError(
            f"No constituent membership exists on or before {timestamp.date()}"
        )
    row = rows.iloc[-1]
    effective_date = pd.Timestamp(row["date"])
    if effective_date > timestamp:  # Defensive assertion of the causal contract.
        raise AssertionError("Membership as-of lookup selected a future row")
    return effective_date, tuple(row["members"])


def build_active_months_by_ticker(
    history: pd.DataFrame,
    decision_dates: Iterable,
) -> dict[str, tuple[pd.Period, ...]]:
    """Project validated causal membership into active decision months."""
    dates = pd.DatetimeIndex(pd.to_datetime(list(decision_dates), errors="coerce"))
    if dates.isna().any():
        raise ValueError("Decision dates contain an invalid value")

    active: dict[str, list[pd.Period]] = {}
    for decision_date in dates:
        _, members = membership_asof(history, decision_date)
        month = decision_date.to_period("M")
        for ticker in members:
            active.setdefault(ticker, []).append(month)
    return {
        ticker: tuple(months) for ticker, months in sorted(active.items())
    }


__all__ = [
    "MEMBERSHIP_CHANGES_FILENAME",
    "MEMBERSHIP_COMPONENTS_FILENAME",
    "MEMBERSHIP_INTERVALS_FILENAME",
    "MEMBERSHIP_MANIFEST_FILENAME",
    "MEMBERSHIP_PROVENANCE_LABEL",
    "MEMBERSHIP_SOURCE_URL",
    "MembershipSourcePaths",
    "MANIFEST_COLUMNS",
    "build_active_months_by_ticker",
    "membership_asof",
    "parse_ticker_list",
    "validate_membership_source_manifest",
    "validate_membership_sources",
]
