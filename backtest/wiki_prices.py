"""Strict capture of the sole permitted frozen WIKI Prices parent object."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from portfolio_core.artifacts import file_sha256
from portfolio_core.io import atomic_write_csv_rows
from portfolio_core.security_identity import load_security_identity_bundle

from .config import DEFAULT_CONFIG
from .price_sources import (
    WIKI_COLUMNS,
    WIKI_PARENT_BYTES,
    WIKI_PARENT_SHA256,
    validate_wiki_extract_rows,
    wiki_mapping_rows,
)


def verify_and_extract_pinned_wiki_parent(
    parent_path: Path,
    extract_path: Path,
    *,
    mappings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Verify the exact parent before parsing and atomically write its extract."""
    parent_path = Path(parent_path)
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    actual_size = parent_path.stat().st_size
    if actual_size != WIKI_PARENT_BYTES:
        raise ValueError(
            f"Pinned WIKI parent byte-size mismatch: expected {WIKI_PARENT_BYTES}, "
            f"found {actual_size}"
        )
    actual_hash = file_sha256(parent_path)
    if actual_hash != WIKI_PARENT_SHA256:
        raise ValueError(
            f"Pinned WIKI parent SHA-256 mismatch: expected {WIKI_PARENT_SHA256}, "
            f"found {actual_hash}"
        )

    if mappings is None:
        mappings = load_security_identity_bundle(
            DEFAULT_CONFIG.paths.project_root
        ).provider_mappings
    mapped = wiki_mapping_rows(mappings)
    intervals = {
        str(row["Provider_Symbol"]): (
            str(row["Local_First_Date"]),
            str(row["Local_Last_Date"]),
        )
        for row in mapped.to_dict("records")
    }

    selected: list[dict[str, str]] = []
    with parent_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != WIKI_COLUMNS:
            raise ValueError(
                f"Pinned WIKI parent schema mismatch: expected {list(WIKI_COLUMNS)}, "
                f"found {reader.fieldnames or []}"
            )
        for row in reader:
            ticker = str(row["ticker"]).strip()
            date = str(row["date"]).strip()
            interval = intervals.get(ticker)
            if (
                interval is not None
                and interval[0] <= date <= interval[1]
            ):
                selected.append({column: row[column] for column in WIKI_COLUMNS})

    selected.sort(key=lambda row: (row["ticker"], row["date"]))
    candidate = pd.DataFrame(selected, columns=WIKI_COLUMNS)
    validate_wiki_extract_rows(candidate, mappings=mappings)
    atomic_write_csv_rows(selected, WIKI_COLUMNS, Path(extract_path))
    persisted = pd.read_csv(extract_path, keep_default_na=False, dtype=str)
    validate_wiki_extract_rows(persisted, mappings=mappings)
    return persisted


__all__ = [
    "verify_and_extract_pinned_wiki_parent",
]
