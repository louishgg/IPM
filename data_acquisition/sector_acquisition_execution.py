"""Wikipedia policy and provider execution for sector acquisition."""

from __future__ import annotations

from dataclasses import replace
from importlib import metadata as importlib_metadata

import pandas as pd

from data_acquisition.contracts import (
    AcquisitionStatus,
    CommandRequest,
    ProviderStatus,
    ReadinessStatus,
    utc_timestamp,
)
from data_acquisition.engine import (
    AcquisitionOutcome,
    PRODUCTION_ACQUISITION_POLICY,
    SerialAcquisitionEngine,
)
from data_acquisition.provider_clients import ClientInfo
from data_acquisition.providers.wikipedia import make_wikipedia_sectors_adapter
from data_acquisition.runtime import (
    AcquisitionReporter,
    AcquisitionRuntimePaths,
    acquisition_lock,
)
from data_acquisition.sector_acquisition_artifacts import (
    load_sector_acquisition_state,
    publish_sector_state,
    write_sector_runtime_checkpoint,
)
from data_acquisition.sector_acquisition_planning import (
    SectorRequirementBuilder,
    build_sector_acquisition_readiness,
    build_sector_acquisition_requests,
    current_scope_requirements,
    report_scope_readiness,
)
from portfolio_core.sector_evidence import (
    SECTOR_ACQUISITION_DATASET,
    SECTOR_ACQUISITION_SCOPE,
    SectorHistoryPaths,
    load_sector_notices,
)
from portfolio_core.sector_resolution import WikipediaIdentityResolver
from portfolio_core.security_identity import load_security_identity_bundle


WIKIPEDIA_ACQUISITION_POLICY = replace(
    PRODUCTION_ACQUISITION_POLICY,
    # Each item already performs two serialized HTTP calls. This remains
    # deliberately sub-2 requests/second while avoiding multi-minute idle time.
    pacing_min_seconds=0.5,
    pacing_max_seconds=1.0,
)


def execute_sector_acquisition(
    request: CommandRequest,
    planner: SectorRequirementBuilder,
) -> int:
    """Acquire one scope's dates while preserving shared immutable snapshots."""
    if request.dry_run:
        raise ValueError(
            "Dry-run sector acquisition must be dispatched by a planning module"
        )
    current = current_scope_requirements(request, planner)
    paths = SectorHistoryPaths.from_project_root(request.project_root)
    runtime = AcquisitionRuntimePaths(request.project_root)
    with acquisition_lock(
        runtime.lock_path(SECTOR_ACQUISITION_SCOPE, SECTOR_ACQUISITION_DATASET)
    ):
        return _execute_sector_acquisition_locked(request, current, paths, runtime)

def _execute_sector_acquisition_locked(
    request: CommandRequest,
    current: pd.DataFrame,
    paths: SectorHistoryPaths,
    runtime: AcquisitionRuntimePaths,
) -> int:
    """Read, merge, acquire, and finalize while holding the shared lock."""
    requests = build_sector_acquisition_requests(current)
    snapshots, pins, exact, unrelated = load_sector_acquisition_state(
        paths,
        runtime,
        requests,
        project_root=request.project_root,
    )
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
    resolver = WikipediaIdentityResolver(identity_bundle)
    captured = utc_timestamp()
    reporter = AcquisitionReporter()
    client = ClientInfo(
        name="requests",
        version=importlib_metadata.version("requests"),
    )
    adapter = make_wikipedia_sectors_adapter(
        client,
        pinned_revisions=pins,
    )
    def checkpoint(
        current_statuses: tuple[AcquisitionStatus, ...],
        outcome: AcquisitionOutcome[pd.DataFrame] | None,
    ) -> None:
        nonlocal snapshots
        if outcome is not None and outcome.result is not None:
            date = outcome.request.requested_end or outcome.request.requested_start
            existing_dates = pd.to_datetime(
                snapshots["Requirement_Date"], errors="raise"
            ).dt.strftime("%Y-%m-%d") if not snapshots.empty else pd.Series(dtype=str)
            snapshots = snapshots.loc[~existing_dates.eq(date)].copy()
            addition = outcome.result.payload.copy()
            addition["Requirement_Date"] = pd.to_datetime(
                addition["Requirement_Date"], errors="raise"
            )
            snapshots = pd.concat(
                [snapshots, addition], ignore_index=True
            ).sort_values(
                ["Requirement_Date", "Wikipedia_Ticker"], kind="stable"
            ).reset_index(drop=True)
        if outcome is not None:
            write_sector_runtime_checkpoint(runtime, outcome, current_statuses)

    run = SerialAcquisitionEngine(
        adapter,
        policy=WIKIPEDIA_ACQUISITION_POLICY,
        reporter=reporter,
    ).run(
        requests,
        statuses=exact,
        checkpoint=checkpoint,
        refresh=request.refresh,
    )
    final_statuses = sorted(
        [*unrelated, *run.statuses], key=lambda status: status.identity.key
    )
    readiness = build_sector_acquisition_readiness(
        current,
        snapshots,
        notices,
        identity_resolver=resolver,
        checked_at_utc=captured,
    )
    publish_sector_state(
        paths=paths,
        runtime=runtime,
        project_root=request.project_root,
        snapshots=snapshots,
        statuses=final_statuses,
        notices=notices,
        captured_at_utc=captured,
    )
    report_scope_readiness(
        readiness,
        scope=request.scope,
        reporter=reporter,
    )
    final_by_key = {
        status.identity.key: status
        for status in final_statuses
    }
    requested_ok = all(
        final_by_key.get(item.identity.key) is not None
        and final_by_key[item.identity.key].status is ProviderStatus.OK
        for item in requests
    )
    scope_readiness = [
        record for record in readiness if record.scope == request.scope
    ]
    if (
        not run.complete
        or not requested_ok
        or not scope_readiness
    ):
        reporter(
            f"{request.scope} sectors remain incomplete; provider state and "
            "aggregate evidence were published for a resumable rerun."
        )
        return 1
    if all(
        record.status is ReadinessStatus.COMPLETE
        for record in scope_readiness
    ):
        reporter(
            f"{request.scope} sectors are complete with immutable Wikipedia "
            "revision provenance."
        )
    else:
        reporter(
            f"{request.scope} sector revisions are complete with immutable "
            "Wikipedia provenance; conservative readiness remains incomplete "
            "and is diagnostic only. Exact consumer coverage is enforced "
            "during preparation."
        )
    return 0

__all__ = [
    "WIKIPEDIA_ACQUISITION_POLICY",
    "execute_sector_acquisition",
]
