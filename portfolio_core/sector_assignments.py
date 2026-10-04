"""Prepared point-in-time sector assignment contracts and exact-date lookups."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from portfolio_core.sector_evidence import (
    GICS_SECTOR_LABELS,
    clean_sector_text,
    normalize_gics_sector,
)


ASSIGNMENT_COLUMNS = (
    "As_Of_Date",
    "Asset_ID",
    "GICS_Sector_Code",
    "Sector",
    "Source_Type",
    "Source_Reference",
    "Source_Symbol",
    "Resolution_Method",
)
ASSIGNMENT_REQUIREMENT_COLUMNS = (
    "As_Of_Date",
    "Asset_ID",
    "Source_Ticker",
)
SECTOR_AUDIT_COLUMNS = (
    "GICS_Sector_Code",
    "Sector",
    "Sector_As_Of_Date",
    "Sector_Source_Type",
    "Sector_Source_Reference",
)


def validate_sector_assignment_requirements(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate exact asset/date requirements shared by every preparation."""
    if tuple(frame.columns) != ASSIGNMENT_REQUIREMENT_COLUMNS:
        raise ValueError(
            "Sector assignment requirements must have columns "
            f"{list(ASSIGNMENT_REQUIREMENT_COLUMNS)}"
        )
    requirements = frame.copy()
    if requirements.empty:
        raise ValueError("Sector assignment requirements are empty")
    requirements["As_Of_Date"] = pd.to_datetime(
        requirements["As_Of_Date"], errors="raise"
    ).dt.normalize()
    for column in ("Asset_ID", "Source_Ticker"):
        requirements[column] = requirements[column].map(clean_sector_text)
    if requirements[["Asset_ID", "Source_Ticker"]].eq("").any().any():
        raise ValueError(
            "Sector assignment requirements contain empty identifiers"
        )
    if requirements.duplicated(["As_Of_Date", "Asset_ID"]).any():
        raise ValueError(
            "Sector assignment requirements contain duplicate pairs"
        )
    return requirements.sort_values(
        ["As_Of_Date", "Asset_ID"], kind="stable"
    ).reset_index(drop=True)


def validate_sector_assignments(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the prepared one-row-per-active-asset/date contract."""
    if tuple(frame.columns) != ASSIGNMENT_COLUMNS:
        raise ValueError(
            f"Sector assignments must have columns {list(ASSIGNMENT_COLUMNS)}; "
            f"found {list(frame.columns)}"
        )
    result = frame.copy()
    if result.empty:
        raise ValueError("Sector assignments are empty")
    result["As_Of_Date"] = pd.to_datetime(
        result["As_Of_Date"], errors="raise"
    ).dt.normalize()
    for column in ASSIGNMENT_COLUMNS[1:]:
        result[column] = result[column].map(clean_sector_text)
    if result[list(ASSIGNMENT_COLUMNS[1:])].eq("").any().any():
        raise ValueError("Sector assignments contain empty required values")
    if result.duplicated(["As_Of_Date", "Asset_ID"]).any():
        raise ValueError("Sector assignments contain duplicate asset/date pairs")
    unsupported_codes = sorted(
        set(result["GICS_Sector_Code"]) - set(GICS_SECTOR_LABELS)
    )
    if unsupported_codes:
        raise ValueError(f"Unsupported GICS sector codes: {unsupported_codes}")
    for row in result.itertuples(index=False):
        code, normalized_label = normalize_gics_sector(row.Sector)
        if code != row.GICS_Sector_Code or normalized_label != row.Sector:
            raise ValueError(
                f"Sector label/code mismatch for {row.Asset_ID} on "
                f"{row.As_Of_Date.date()}"
            )
    return result.sort_values(
        ["As_Of_Date", "Asset_ID"], kind="stable"
    ).reset_index(drop=True)


def load_sector_assignments(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing prepared sector assignments: {path}. Rebuild prepared data."
        )
    return validate_sector_assignments(
        pd.read_csv(path, keep_default_na=False, dtype=str)
    )


def sector_rows_asof(
    assignments: pd.DataFrame,
    date: object,
) -> pd.DataFrame:
    """Return assignments for the exact normalized date, without carrying older rows.

    A missing date raises KeyError; duplicate asset rows raise ValueError.
    """
    value = pd.Timestamp(date).normalize()
    frame = assignments
    if not pd.api.types.is_datetime64_any_dtype(frame["As_Of_Date"]):
        frame = frame.copy()
        frame["As_Of_Date"] = pd.to_datetime(frame["As_Of_Date"], errors="raise")
    rows = frame.loc[frame["As_Of_Date"].eq(value)].copy()
    if rows.empty:
        raise KeyError(f"No sector assignments exist for exact date {value.date()}")
    if rows["Asset_ID"].duplicated().any():
        raise ValueError(f"Sector lookup is ambiguous on {value.date()}")
    return rows.set_index("Asset_ID", drop=False).sort_index()


def sector_audit_fields(
    rows: pd.DataFrame,
    asset_id: str,
    expected_date: object,
    *,
    context: str,
) -> dict[str, object]:
    """Validate and return one exact-date sector provenance record."""
    date = pd.Timestamp(expected_date).normalize()
    if asset_id not in rows.index:
        raise RuntimeError(
            f"Missing {context} sector assignment for {asset_id} on {date.date()}"
        )
    sector = rows.loc[asset_id]
    if isinstance(sector, pd.DataFrame):
        raise RuntimeError(
            f"Duplicate {context} sector assignments for {asset_id} on "
            f"{date.date()}"
        )
    as_of_date = pd.Timestamp(sector["As_Of_Date"]).normalize()
    if as_of_date != date:
        raise RuntimeError(
            f"{context.capitalize()} sector assignment for {asset_id} uses "
            f"{as_of_date.date()} instead of {date.date()}"
        )
    values = {
        "GICS_Sector_Code": str(sector["GICS_Sector_Code"]),
        "Sector": str(sector["Sector"]),
        "Sector_As_Of_Date": as_of_date,
        "Sector_Source_Type": str(sector["Source_Type"]),
        "Sector_Source_Reference": str(sector["Source_Reference"]),
    }
    if any(
        value in {"", "Unknown"}
        for value in values.values()
        if isinstance(value, str)
    ):
        raise RuntimeError(
            f"Incomplete {context} sector provenance for {asset_id} on "
            f"{date.date()}"
        )
    return values


def prior_sector_audit_fields(
    assignments: pd.DataFrame,
    asset_id: str,
    execution_date: object,
    *,
    context: str,
) -> dict[str, object]:
    """Return the same asset's latest strictly earlier causal assignment.

    This lookup is intentionally separate from exact-date selection and
    holding lookups.  Callers may use it only after proving that an existing
    position is being closed completely because it is no longer eligible.
    """

    date = pd.Timestamp(execution_date).normalize()
    frame = assignments.copy()
    frame["As_Of_Date"] = pd.to_datetime(
        frame["As_Of_Date"], errors="raise"
    ).dt.normalize()
    rows = frame.loc[
        frame["Asset_ID"].astype(str).eq(str(asset_id))
        & frame["As_Of_Date"].lt(date)
    ].sort_values("As_Of_Date", kind="stable")
    if rows.empty:
        raise RuntimeError(
            f"Missing prior causal sector assignment for forced {context} "
            f"{asset_id} before {date.date()}"
        )
    latest_date = pd.Timestamp(rows.iloc[-1]["As_Of_Date"]).normalize()
    latest = rows.loc[rows["As_Of_Date"].eq(latest_date)]
    if len(latest) != 1:
        raise RuntimeError(
            f"Ambiguous prior causal sector assignment for forced {context} "
            f"{asset_id} before {date.date()}"
        )
    sector = latest.iloc[0]
    source_symbol = str(sector["Source_Symbol"])
    resolution_method = str(sector["Resolution_Method"])
    values = {
        "GICS_Sector_Code": str(sector["GICS_Sector_Code"]),
        "Sector": str(sector["Sector"]),
        "Sector_As_Of_Date": latest_date,
        "Sector_Source_Type": str(sector["Source_Type"]),
        "Sector_Source_Reference": str(sector["Source_Reference"]),
    }
    if any(
        value in {"", "Unknown"}
        for value in (*values.values(), source_symbol, resolution_method)
        if isinstance(value, str)
    ):
        raise RuntimeError(
            f"Incomplete prior sector provenance for forced {context} "
            f"{asset_id} before {date.date()}"
        )
    return values


def trade_sector_audit_fields(
    *,
    assignments: pd.DataFrame,
    execution_rows: pd.DataFrame,
    execution_date: object,
    execution_eligible: bool,
    asset_id: str,
    current_value: float,
    target_value: float,
    domain: str,
) -> dict[str, object]:
    """Resolve exact active or strictly prior complete-exit provenance."""
    if domain not in {"backtest", "live"}:
        raise ValueError(f"Unsupported trade-sector domain: {domain!r}")
    if execution_eligible:
        return sector_audit_fields(
            execution_rows,
            asset_id,
            execution_date,
            context="execution-date",
        )
    if current_value == 0.0 or target_value != 0.0:
        raise RuntimeError(
            f"An ineligible {domain} asset may only be closed completely: "
            f"{asset_id} on {execution_date.date()} has current={current_value}, "
            f"target={target_value}"
        )
    return prior_sector_audit_fields(
        assignments,
        asset_id,
        execution_date,
        context=f"{domain} exit",
    )


def validate_holding_sector_provenance(
    holdings: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    """Match every holding's audit fields to its exact-date assignment."""
    required = {"Date", "Asset_ID", *SECTOR_AUDIT_COLUMNS}
    missing = sorted(required - set(holdings.columns))
    if missing:
        raise ValueError(f"{context} holdings is missing sector columns: {missing}")

    result = holdings.copy()
    result["Date"] = pd.to_datetime(
        result["Date"], errors="raise"
    ).dt.tz_localize(None)
    result["Sector_As_Of_Date"] = pd.to_datetime(
        result["Sector_As_Of_Date"], errors="raise"
    ).dt.tz_localize(None)
    audit_text_columns = (
        "Asset_ID",
        "GICS_Sector_Code",
        "Sector",
        "Sector_Source_Type",
        "Sector_Source_Reference",
    )
    for column in audit_text_columns:
        result[column] = result[column].astype(str).str.strip()
    provenance_columns = audit_text_columns[1:]
    if result[list(provenance_columns)].isin(["", "Unknown"]).any().any():
        raise ValueError(f"{context} holdings contains incomplete sector provenance")
    if not result["Sector_As_Of_Date"].equals(result["Date"]):
        raise ValueError(
            f"{context} holdings must use execution-date sector assignments"
        )

    for date, group in result.groupby("Date", sort=True):
        expected = sector_rows_asof(assignments, date)
        for row in group.itertuples(index=False):
            asset_id = str(row.Asset_ID)
            if asset_id not in expected.index:
                raise ValueError(
                    f"{context} holding {asset_id} lacks a sector assignment "
                    f"on {pd.Timestamp(date).date()}"
                )
            source = expected.loc[asset_id]
            actual = (
                str(row.GICS_Sector_Code),
                str(row.Sector),
                pd.Timestamp(row.Sector_As_Of_Date),
                str(row.Sector_Source_Type),
                str(row.Sector_Source_Reference),
            )
            wanted = (
                str(source["GICS_Sector_Code"]),
                str(source["Sector"]),
                pd.Timestamp(source["As_Of_Date"]),
                str(source["Source_Type"]),
                str(source["Source_Reference"]),
            )
            if actual != wanted:
                raise ValueError(
                    f"{context} holding sector provenance disagrees with the "
                    f"prepared assignment for {asset_id} on "
                    f"{pd.Timestamp(date).date()}"
                )
    return result


__all__ = [
    "ASSIGNMENT_COLUMNS",
    "ASSIGNMENT_REQUIREMENT_COLUMNS",
    "SECTOR_AUDIT_COLUMNS",
    "load_sector_assignments",
    "prior_sector_audit_fields",
    "sector_audit_fields",
    "sector_rows_asof",
    "trade_sector_audit_fields",
    "validate_holding_sector_provenance",
    "validate_sector_assignment_requirements",
    "validate_sector_assignments",
]
