"""Point-in-time live price requirements and approved alternate coverage.

This module is deliberately offline.  It builds the strategy requirement graph,
validates the authoritative Yahoo supplement, and reconciles
market observations with reviewed cash-acquisition settlements.  Provider
status remains separate from downstream readiness throughout.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from portfolio_core.artifacts import (
    ArtifactOrigin,
    validate_exact_manifest_catalog,
)
from data_acquisition.contracts import (
    AcquisitionIdentity,
    AcquisitionStatus,
    ProviderStatus,
    ReadinessRecord,
    readiness_status,
    read_acquisition_statuses,
    write_acquisition_statuses,
)
from .strategy_universe import SCHEDULE_DATE_COLUMNS, prepared_membership_asof


PRICE_REQUIREMENT_SET = "strategy_market_and_position_lifecycle"
YAHOO_SUPPLEMENT_SOURCE = "yahoo_supplement"
YAHOO_SUPPLEMENT_DATASET = "prices_supplemental"
YAHOO_SUPPLEMENT_REFERENCE = "supplemental/yahoo_ohlcv.csv"
YAHOO_SUPPLEMENT_EXPECTED_ROWS = {
    "CTRA": 547,
    "HOLX": 526,
}

SUPPLEMENT_COLUMNS = (
    "Date",
    "Asset_ID",
    "Provider_Symbol",
    "Open",
    "Close",
    "Volume",
)

REQUIREMENT_COLUMNS = (
    "Requirement_Set",
    "Rebalance_ID",
    "Asset_ID",
    "Requirement_Date",
    "Role",
    "Field",
    "Holding_Start_Date",
    "Coverage_Status",
    "Coverage_Source",
    "Coverage_Reference",
)

@dataclass(frozen=True, slots=True, order=True)
class PriceRequirement:
    """One role-specific market or position-lifecycle requirement."""

    rebalance_id: str
    asset_id: str
    requirement_date: str
    role: str
    field: str
    holding_start_date: str = ""


def build_price_requirements(
    schedule: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    evaluation: pd.DataFrame | None = None,
) -> tuple[PriceRequirement, ...]:
    """Build the conservative signal/candidate/possible-holding requirement graph."""

    periods = schedule.copy()
    pit = membership.copy()
    for column in SCHEDULE_DATE_COLUMNS:
        periods[column] = pd.to_datetime(periods[column], errors="raise").dt.tz_localize(None)
    pit["Effective_Date"] = pd.to_datetime(
        pit["Effective_Date"], errors="raise"
    ).dt.tz_localize(None)

    requirements: set[PriceRequirement] = set()
    preceding_candidates: set[str] = set()
    preceding_execution: pd.Timestamp | None = None
    for row in periods.sort_values("Execution_Date", kind="stable").itertuples(index=False):
        signal_members = prepared_membership_asof(
            pit, pd.Timestamp(row.Membership_Effective_Date)
        )
        execution_members = prepared_membership_asof(
            pit, pd.Timestamp(row.Execution_Date)
        )
        execution_candidates = signal_members & execution_members

        for asset_id in signal_members:
            requirements.add(
                PriceRequirement(
                    str(row.Rebalance_ID),
                    asset_id,
                    pd.Timestamp(row.Signal_Cutoff).date().isoformat(),
                    "signal",
                    "Close",
                )
            )

        sizing_and_execution = execution_candidates | preceding_candidates
        for asset_id in sizing_and_execution:
            holding_start = (
                preceding_execution
                if asset_id in preceding_candidates and preceding_execution is not None
                else pd.Timestamp(row.Execution_Date)
            )
            requirements.add(
                PriceRequirement(
                    str(row.Rebalance_ID),
                    asset_id,
                    pd.Timestamp(row.Sizing_Date).date().isoformat(),
                    "sizing",
                    str(row.Sizing_Field).title(),
                    holding_start.date().isoformat(),
                )
            )
            requirements.add(
                PriceRequirement(
                    str(row.Rebalance_ID),
                    asset_id,
                    pd.Timestamp(row.Execution_Date).date().isoformat(),
                    "execution",
                    str(row.Execution_Field).title(),
                    holding_start.date().isoformat(),
                )
            )

        for asset_id in execution_candidates:
            requirements.add(
                PriceRequirement(
                    str(row.Rebalance_ID),
                    asset_id,
                    pd.Timestamp(row.Valuation_End).date().isoformat(),
                    "valuation",
                    str(row.Valuation_Field).title(),
                    pd.Timestamp(row.Execution_Date).date().isoformat(),
                )
            )

        preceding_candidates = execution_candidates
        preceding_execution = pd.Timestamp(row.Execution_Date)

    # The benchmark remains invested while the strategy account holds cash.
    # Its constituent prices must therefore cover every evaluation interval.
    if evaluation is not None:
        for row in evaluation.itertuples(index=False):
            start = pd.Timestamp(row.Period_Start).date().isoformat()
            for asset_id in prepared_membership_asof(pit, row.Period_Start):
                for boundary, field, role in (
                    (row.Period_Start, row.Start_Field, "benchmark_start"),
                    (row.Period_End, row.End_Field, "benchmark_end"),
                ):
                    requirements.add(PriceRequirement(
                        str(row.Rebalance_ID), asset_id,
                        pd.Timestamp(boundary).date().isoformat(),
                        role, str(field), start,
                    ))
    return tuple(sorted(requirements))


def read_yahoo_supplement(paths) -> pd.DataFrame:
    artifact = Path(paths.raw_price_supplemental_csv)
    validate_exact_manifest_catalog(
        Path(paths.raw_price_supplemental_manifest_csv),
        scope="live",
        dataset=YAHOO_SUPPLEMENT_DATASET,
        expected_origins={artifact.name: ArtifactOrigin.MIGRATED},
        base_dir=artifact.parent,
    )
    frame = pd.read_csv(artifact, keep_default_na=False)
    if tuple(frame.columns) != SUPPLEMENT_COLUMNS:
        raise ValueError("Yahoo supplement schema is inconsistent")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.tz_localize(None)
    for column in ("Open", "Close", "Volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if frame.duplicated(["Date", "Asset_ID"]).any():
        raise ValueError("Yahoo supplement contains duplicate asset-date rows")
    counts = frame.groupby("Asset_ID").size().to_dict()
    if counts != YAHOO_SUPPLEMENT_EXPECTED_ROWS:
        raise ValueError(f"Yahoo supplement has unexpected row counts: {counts}")
    return frame.sort_values(["Asset_ID", "Date"], kind="stable").reset_index(drop=True)


def merge_supplemental_statuses(path: Path, frame: pd.DataFrame) -> None:
    """Register the supplement as a distinct successful provider source."""

    path = Path(path)
    existing = read_acquisition_statuses(path) if path.is_file() else []
    retained = [
        item
        for item in existing
        if not (
            item.identity.dataset == "prices"
            and item.identity.provider == YAHOO_SUPPLEMENT_SOURCE
        )
    ]
    supplemental = []
    for asset_id, rows in frame.groupby("Asset_ID", sort=True):
        dates = pd.to_datetime(rows["Date"], errors="raise")
        supplemental.append(
            AcquisitionStatus(
                identity=AcquisitionIdentity(
                    scope="live",
                    dataset="prices",
                    asset_id=str(asset_id),
                    provider=YAHOO_SUPPLEMENT_SOURCE,
                    provider_symbol=str(rows["Provider_Symbol"].iloc[0]),
                ),
                status=ProviderStatus.OK,
                requested_start=dates.min().date().isoformat(),
                requested_end=dates.max().date().isoformat(),
                observation_count=len(rows),
                observation_start=dates.min().date().isoformat(),
                observation_end=dates.max().date().isoformat(),
                client="validated_supplement",
                client_version="",
                migration_note=(
                    "Authoritative Yahoo supplement; current Yahoo provider "
                    "status remains independently truthful."
                ),
            )
        )
    write_acquisition_statuses(path, [*retained, *supplemental])


def supplemental_available_dates(frame: pd.DataFrame) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        result[str(row.Asset_ID)].add(pd.Timestamp(row.Date).date().isoformat())
    return dict(result)


def build_coverage_ledger(
    requirements: Iterable[PriceRequirement],
    *,
    yahoo_dates: Mapping[str, set[str]],
    supplemental_dates: Mapping[str, set[str]],
    corporate_actions: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve each role through market data or a chronological settlement."""

    actions_by_asset = {
        str(asset_id): rows.sort_values("Effective_Date", kind="stable")
        for asset_id, rows in corporate_actions.groupby("Asset_ID", sort=False)
    }
    rows = []
    for item in sorted(requirements):
        source = ""
        reference = ""
        yahoo_asset_dates = yahoo_dates.get(item.asset_id, set())
        supplemental_asset_dates = supplemental_dates.get(item.asset_id, set())
        if item.requirement_date in yahoo_asset_dates:
            source = "yahoo"
        elif item.requirement_date in supplemental_asset_dates:
            source = YAHOO_SUPPLEMENT_SOURCE
            reference = YAHOO_SUPPLEMENT_REFERENCE
        elif item.holding_start_date and item.asset_id in actions_by_asset:
            eligible = actions_by_asset[item.asset_id]
            eligible = eligible.loc[
                eligible["Effective_Date"].ge(pd.Timestamp(item.holding_start_date))
                & eligible["Effective_Date"].le(pd.Timestamp(item.requirement_date))
            ]
            if not eligible.empty:
                action = eligible.iloc[0]
                source = "corporate_action_settlement"
                reference = str(action["Event_ID"])
        rows.append(
            {
                "Requirement_Set": PRICE_REQUIREMENT_SET,
                "Rebalance_ID": item.rebalance_id,
                "Asset_ID": item.asset_id,
                "Requirement_Date": item.requirement_date,
                "Role": item.role,
                "Field": item.field,
                "Holding_Start_Date": item.holding_start_date,
                "Coverage_Status": "covered" if source else "missing",
                "Coverage_Source": source,
                "Coverage_Reference": reference,
            }
        )
    return pd.DataFrame(rows, columns=REQUIREMENT_COLUMNS).sort_values(
        ["Asset_ID", "Requirement_Date", "Rebalance_ID", "Role"], kind="stable"
    ).reset_index(drop=True)


def aggregate_price_readiness(
    ledger: pd.DataFrame,
    *,
    checked_at_utc: str,
) -> list[ReadinessRecord]:
    """Aggregate role rows into unique asset-date readiness cells."""

    if tuple(ledger.columns) != REQUIREMENT_COLUMNS:
        raise ValueError("Price requirements ledger schema is inconsistent")
    records: list[ReadinessRecord] = []
    for asset_id, asset_rows in ledger.groupby("Asset_ID", sort=True):
        required_dates: set[str] = set()
        missing_dates: set[str] = set()
        sources: set[str] = set()
        for date_value, cell in asset_rows.groupby("Requirement_Date", sort=True):
            required_dates.add(str(date_value))
            if not cell["Coverage_Status"].eq("covered").all():
                missing_dates.add(str(date_value))
            else:
                sources.update(
                    value for value in cell["Coverage_Source"].astype(str) if value
                )
        covered = len(required_dates) - len(missing_dates)
        records.append(
            ReadinessRecord(
                scope="live",
                dataset="prices",
                asset_id=str(asset_id),
                requirement_set=PRICE_REQUIREMENT_SET,
                required_count=len(required_dates),
                covered_count=covered,
                status=readiness_status(covered, len(required_dates)),
                missing_dates=tuple(sorted(missing_dates)),
                contributing_sources=tuple(sorted(sources)),
                checked_at_utc=checked_at_utc,
            )
        )
    return records


__all__ = [
    "YAHOO_SUPPLEMENT_DATASET",
    "YAHOO_SUPPLEMENT_EXPECTED_ROWS",
    "YAHOO_SUPPLEMENT_SOURCE",
    "YAHOO_SUPPLEMENT_REFERENCE",
    "PRICE_REQUIREMENT_SET",
    "PriceRequirement",
    "REQUIREMENT_COLUMNS",
    "SUPPLEMENT_COLUMNS",
    "aggregate_price_readiness",
    "build_coverage_ledger",
    "build_price_requirements",
    "merge_supplemental_statuses",
    "read_yahoo_supplement",
    "supplemental_available_dates",
]
