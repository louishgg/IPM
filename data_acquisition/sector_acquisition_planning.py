"""Provider-free planning and readiness for sector acquisition."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from data_acquisition.contracts import (
    AcquisitionIdentity,
    AcquisitionRequest,
    CommandRequest,
    ReadinessRecord,
    readiness_status,
    utc_timestamp,
)
from data_acquisition.engine import build_acquisition_plan
from data_acquisition.runtime import AcquisitionRuntimePaths, acquisition_lock
from data_acquisition.sector_acquisition_artifacts import (
    load_sector_acquisition_state,
)
from portfolio_core.sector_evidence import (
    SECTOR_ACQUISITION_DATASET,
    SECTOR_ACQUISITION_SCOPE,
    SP500_PAGE_TITLE,
    SectorHistoryPaths,
    load_sector_notices,
)
from portfolio_core.sector_resolution import (
    SectorResolutionContext,
    WikipediaIdentityResolver,
)
from portfolio_core.security_identity import load_security_identity_bundle


SECTOR_ACQUISITION_REQUIREMENT_COLUMNS = (
    "Scope",
    "Requirement_Date",
    "Cutoff_UTC",
    "Asset_ID",
    "Source_Ticker",
)
SECTOR_ACQUISITION_REQUIREMENT_SET = "point_in_time_gics_sector"
PROVIDER = "wikipedia"

SectorRequirementBuilder = Callable[[], pd.DataFrame]


def project_sector_acquisition_requirements(
    requirements: pd.DataFrame,
    *,
    scope: str,
) -> pd.DataFrame:
    """Project domain sector requirements into the shared acquisition schema."""
    result = requirements.rename(
        columns={"As_Of_Date": "Requirement_Date"}
    ).copy()
    result.insert(0, "Scope", scope)
    result.insert(
        2,
        "Cutoff_UTC",
        result["Requirement_Date"].map(
            lambda value: f"{pd.Timestamp(value).date().isoformat()}T00:00:00Z"
        ),
    )
    return result.loc[:, list(SECTOR_ACQUISITION_REQUIREMENT_COLUMNS)]


def validate_sector_acquisition_requirements(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != SECTOR_ACQUISITION_REQUIREMENT_COLUMNS:
        raise ValueError(
            f"Sector requirements must have columns {list(SECTOR_ACQUISITION_REQUIREMENT_COLUMNS)}; "
            f"found {list(frame.columns)}"
        )
    result = frame.copy()
    if result.empty:
        raise ValueError("Sector requirement set is empty")
    for column in ("Scope", "Asset_ID", "Source_Ticker"):
        result[column] = result[column].astype(str).str.strip()
    result["Requirement_Date"] = pd.to_datetime(
        result["Requirement_Date"], errors="raise"
    ).dt.normalize()
    result["Cutoff_UTC"] = result["Cutoff_UTC"].astype(str).str.strip()
    cutoffs = pd.to_datetime(result["Cutoff_UTC"], utc=True, errors="raise")
    expected = result["Requirement_Date"].dt.tz_localize("UTC")
    if (cutoffs != expected).any():
        raise ValueError("Sector requirement cutoffs must be midnight UTC")
    if result[["Scope", "Asset_ID", "Source_Ticker"]].eq("").any().any():
        raise ValueError("Sector requirements contain empty identifiers")
    if result.duplicated(["Scope", "Requirement_Date", "Asset_ID"]).any():
        raise ValueError("Sector requirements contain duplicate asset/date pairs")
    return result.sort_values(
        ["Scope", "Requirement_Date", "Asset_ID"], kind="stable"
    ).reset_index(drop=True)

def build_sector_acquisition_requests(requirements: pd.DataFrame) -> tuple[AcquisitionRequest, ...]:
    dates = sorted(pd.DatetimeIndex(requirements["Requirement_Date"].unique()))
    return tuple(
        AcquisitionRequest(
            identity=AcquisitionIdentity(
                scope=SECTOR_ACQUISITION_SCOPE,
                dataset=SECTOR_ACQUISITION_DATASET,
                asset_id="S&P 500",
                provider=PROVIDER,
                provider_symbol=SP500_PAGE_TITLE,
                effective_start=date.date().isoformat(),
                effective_end=date.date().isoformat(),
            ),
            requested_start=date.date().isoformat(),
            requested_end=date.date().isoformat(),
        )
        for date in dates
    )

def build_sector_acquisition_readiness(
    requirements: pd.DataFrame,
    snapshots: pd.DataFrame,
    notices: pd.DataFrame,
    *,
    identity_resolver: WikipediaIdentityResolver,
    checked_at_utc: str,
) -> tuple[ReadinessRecord, ...]:
    """Aggregate exact asset/date coverage without conflating provider status."""
    covered: dict[tuple[str, str], dict[str, set[str]]] = {}
    boundary_source = requirements.copy()
    boundary_source["Requirement_Date"] = pd.to_datetime(
        boundary_source["Requirement_Date"], errors="raise"
    ).dt.normalize()
    resolution_context = SectorResolutionContext(
        boundary_source,
        snapshots,
        notices,
        date_column="Requirement_Date",
        scope_column="Scope",
        identity_resolver=identity_resolver,
    )
    for date, required in requirements.groupby("Requirement_Date", sort=True):
        for item in required.itertuples(index=False):
            scope = str(item.Scope)
            asset_id = str(item.Asset_ID)
            source_ticker = str(item.Source_Ticker)
            key = (scope, asset_id)
            state = covered.setdefault(
                key,
                {"required": set(), "covered": set(), "sources": set()},
            )
            date_text = pd.Timestamp(date).date().isoformat()
            state["required"].add(date_text)
            resolution = resolution_context.resolve(
                scope=scope,
                requirement_date=pd.Timestamp(date),
                asset_id=asset_id,
                source_ticker=source_ticker,
            )
            if not resolution.resolved:
                continue
            state["covered"].add(date_text)
            state["sources"].add(resolution.contributing_source)

    records: list[ReadinessRecord] = []
    for (scope, asset_id), state in sorted(covered.items()):
        required_dates = set(state["required"])
        covered_dates = set(state["covered"])
        missing = tuple(sorted(required_dates - covered_dates))
        records.append(
            ReadinessRecord(
                scope=scope,
                dataset=SECTOR_ACQUISITION_DATASET,
                asset_id=asset_id,
                requirement_set=SECTOR_ACQUISITION_REQUIREMENT_SET,
                required_count=len(required_dates),
                covered_count=len(covered_dates),
                status=readiness_status(len(covered_dates), len(required_dates)),
                missing_dates=missing,
                contributing_sources=tuple(sorted(state["sources"])),
                checked_at_utc=checked_at_utc,
            )
        )
    return tuple(records)

def report_scope_readiness(
    readiness: tuple[ReadinessRecord, ...],
    *,
    scope: str,
    reporter: Callable[[str], None],
) -> None:
    """Report deterministic in-memory coverage without persisting a ledger."""
    selected = sorted(
        (record for record in readiness if record.scope == scope),
        key=lambda record: record.asset_id,
    )
    required_count = sum(record.required_count for record in selected)
    covered_count = sum(record.covered_count for record in selected)
    missing_pairs = sorted(
        (date, record.asset_id)
        for record in selected
        for date in record.missing_dates
    )
    reporter(
        f"{scope} sector evidence readiness: {covered_count}/{required_count} "
        f"asset/date pair(s) resolved; {len(missing_pairs)} missing."
    )
    for date, asset_id in missing_pairs[:20]:
        reporter(f"  {date}: {asset_id}")
    remaining = len(missing_pairs) - 20
    if remaining > 0:
        reporter(f"  ... {remaining} additional missing pair(s).")

def current_scope_requirements(
    request: CommandRequest,
    planner: SectorRequirementBuilder,
) -> pd.DataFrame:
    if request.tickers_file is not None:
        raise ValueError("sector acquisition does not accept ticker selection")
    current = validate_sector_acquisition_requirements(planner())
    if set(current["Scope"]) != {request.scope}:
        raise ValueError(
            f"Sector planner for {request.scope} emitted another scope"
        )
    return current

def dry_run_sector_acquisition(
    request: CommandRequest,
    planner: SectorRequirementBuilder,
) -> tuple[AcquisitionRequest, ...]:
    current = current_scope_requirements(request, planner)
    paths = SectorHistoryPaths.from_project_root(request.project_root)
    runtime = AcquisitionRuntimePaths(request.project_root)
    with acquisition_lock(
        runtime.lock_path(SECTOR_ACQUISITION_SCOPE, SECTOR_ACQUISITION_DATASET)
    ):
        requests = build_sector_acquisition_requests(current)
        snapshots, pins, exact, _ = load_sector_acquisition_state(
            paths,
            runtime,
            requests,
            project_root=request.project_root,
        )
        plan = build_acquisition_plan(
            requests,
            statuses=exact,
            refresh=request.refresh,
        )
        print(
            f"{request.scope} sectors: {len(plan.execution_order)} Wikipedia "
            f"revision request(s), {len(requests) - len(plan.execution_order)} "
            "terminal checkpoint(s)."
        )
        for item in plan.execution_order:
            date = item.requested_end
            pin = pins.get(date)
            suffix = (
                f" verify pinned revision {pin}"
                if pin is not None
                else " select revision"
            )
            print(f"  {date}: {suffix.strip()}")
        if not paths.notices_csv.is_file():
            raise FileNotFoundError(
                f"Missing reviewed S&P notice ledger: {paths.notices_csv}"
            )
        notices = load_sector_notices(paths.notices_csv)
        identity_bundle = load_security_identity_bundle(
            request.project_root,
            validate_manifest=True,
            require_all_approved=True,
        )
        readiness = build_sector_acquisition_readiness(
            current,
            snapshots,
            notices,
            identity_resolver=WikipediaIdentityResolver(identity_bundle),
            checked_at_utc=utc_timestamp(),
        )
        report_scope_readiness(
            readiness,
            scope=request.scope,
            reporter=print,
        )
        return plan.execution_order

__all__ = [
    "SECTOR_ACQUISITION_REQUIREMENT_COLUMNS",
    "SECTOR_ACQUISITION_REQUIREMENT_SET",
    "SectorRequirementBuilder",
    "build_sector_acquisition_readiness",
    "build_sector_acquisition_requests",
    "current_scope_requirements",
    "dry_run_sector_acquisition",
    "project_sector_acquisition_requirements",
    "report_scope_readiness",
    "validate_sector_acquisition_requirements",
]
