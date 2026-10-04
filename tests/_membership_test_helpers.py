"""Shared test-only writers for reviewed membership-source fixtures."""

from __future__ import annotations

import pandas as pd

from portfolio_core.artifacts import file_sha256
from portfolio_core.sp500_membership import (
    MANIFEST_COLUMNS,
    MEMBERSHIP_CHANGES_FILENAME,
    MEMBERSHIP_COMPONENTS_FILENAME,
    MEMBERSHIP_PROVENANCE_LABEL,
    MEMBERSHIP_SOURCE_URL,
    MembershipSourcePaths,
)


def _write_membership_source_manifest(
    paths: MembershipSourcePaths,
    *,
    recorded_at: str = "2026-07-22",
) -> None:
    rows = []
    for path in paths.data_files:
        frame = pd.read_csv(path, keep_default_na=False)
        if path.name in {
            MEMBERSHIP_COMPONENTS_FILENAME,
            MEMBERSHIP_CHANGES_FILENAME,
        }:
            maximum_date = pd.to_datetime(frame["date"], errors="raise").max()
        else:
            maximum_date = pd.concat(
                (
                    pd.to_datetime(frame["start_date"], errors="raise"),
                    pd.to_datetime(frame["end_date"], errors="coerce"),
                )
            ).max()
        rows.append({
            "File": path.name,
            "Source_URL": MEMBERSHIP_SOURCE_URL,
            "Recorded_At": recorded_at,
            "SHA256": file_sha256(path),
            "Rows": len(frame),
            "Maximum_Date": (
                ""
                if pd.isna(maximum_date)
                else maximum_date.strftime("%Y-%m-%d")
            ),
            "Provenance_Label": MEMBERSHIP_PROVENANCE_LABEL,
        })
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(
        paths.manifest_csv,
        index=False,
    )
