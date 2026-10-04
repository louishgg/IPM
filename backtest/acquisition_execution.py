"""Backtest provider execution, checkpoint persistence, and manifests."""

from __future__ import annotations

from pathlib import Path
import platform
import shutil
import tempfile
from typing import Iterable
from urllib.request import Request, urlopen

import pandas as pd

from data_acquisition.provider_clients import prepare_yfinance
from data_acquisition.engine import (
    AcquisitionPlan,
    PRODUCTION_ACQUISITION_POLICY,
    SerialAcquisitionEngine,
    build_acquisition_plan,
)
from data_acquisition.providers.yahoo import (
    make_yahoo_benchmark_adapter,
    make_yahoo_shares_adapter,
    make_yahoo_unadjusted_close_adapter,
)
from data_acquisition.providers.base import AcquisitionResult, ProviderAdapter
from data_acquisition.runtime import (
    AcquisitionReporter,
    AcquisitionRuntimePaths,
    acquisition_lock,
)
from data_acquisition.share_checkpoints import (
    replace_share_payload,
    validate_share_status_reconciliation,
)
from portfolio_core.io import (
    atomic_write_bytes,
    atomic_write_dataframe,
)
from portfolio_core.artifacts import (
    SCHEMA_VERSION,
    ArtifactManifest,
    ArtifactOrigin,
    read_manifests,
    validate_manifest_artifact,
    write_manifests,
)
from data_acquisition.contracts import (
    AcquisitionStatus,
    CommandRequest,
    ProviderStatus,
    ReadinessRecord,
    ReadinessStatus,
    read_acquisition_statuses,
    utc_timestamp,
    write_acquisition_statuses,
    write_readiness,
)
from portfolio_core.shares import load_raw_shares, RAW_SHARES_COLUMNS
from portfolio_core.security_identity import load_security_identity_bundle

from .acquisition_planning import (
    SCOPE,
    benchmark_readiness_record,
    build_benchmark_requests,
    build_price_requests,
    build_shares_requests,
    load_fresh_core_data,
    price_requirement_ledger,
    plan_share_sources,
    shares_readiness_records,
    price_readiness_records,
    validate_supplied_reuters_prices,
)
from .config import DEFAULT_CONFIG
from .data_loading import validate_raw_benchmark
from .price_sources import (
    WIKI_PARENT_FIRST_DATE,
    WIKI_PARENT_LAST_DATE,
    WIKI_PARENT_ROWS,
    WIKI_PARENT_URL,
    YAHOO_RAW_COLUMNS,
    load_price_observation_catalog,
    validate_yahoo_close_rows,
)
from .wiki_prices import (
    verify_and_extract_pinned_wiki_parent,
)


def _empty_yahoo_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=YAHOO_RAW_COLUMNS)


def _load_price_state() -> tuple[
    pd.DataFrame | None,
    list[AcquisitionStatus],
]:
    """Load and integrity-check the sole resumable price-domain state."""
    paths = DEFAULT_CONFIG.paths
    validate_supplied_reuters_prices(paths)
    price_paths = paths.price_sources
    if price_paths.yahoo_close_csv.is_file():
        raw = pd.read_csv(
            price_paths.yahoo_close_csv,
            keep_default_na=False,
            float_precision="round_trip",
        )
    else:
        raw = None
    statuses = (
        read_acquisition_statuses(price_paths.acquisition_status_csv)
        if price_paths.acquisition_status_csv.is_file()
        else []
    )
    if price_paths.artifact_manifest_csv.is_file():
        records = read_manifests(price_paths.artifact_manifest_csv)
        expected = {
            path.relative_to(paths.project_root).as_posix()
            for path in (
                price_paths.yahoo_close_csv,
                price_paths.wiki_extract_csv,
                price_paths.acquisition_status_csv,
                price_paths.readiness_csv,
            )
            if path.is_file()
        }
        actual = {record.artifact_path for record in records}
        if actual != expected or any(
            record.scope != SCOPE or record.dataset != "prices"
            for record in records
        ):
            raise ValueError(
                "Backtest fallback-price manifest catalog does not match the "
                "retained raw state"
            )
        for record in records:
            validate_manifest_artifact(record, base_dir=paths.project_root)
    return raw, statuses


def _replace_yahoo_price_payload(
    raw: pd.DataFrame,
    request,
    payload: pd.DataFrame | None,
    *,
    mappings: pd.DataFrame,
) -> pd.DataFrame:
    """Replace one successful capture while retaining last-good failures."""
    if payload is None:
        return raw
    identity = request.identity
    matches = mappings.loc[
        mappings["Scope"].eq(SCOPE)
        & mappings["Provider"].eq("yahoo")
        & mappings["Asset_ID"].eq(identity.asset_id)
        & mappings["Provider_Symbol"].eq(identity.provider_symbol)
        & mappings["Effective_Start"].eq(identity.effective_start)
        & mappings["Effective_End"].eq(identity.effective_end)
        & mappings["Review_Status"].eq("approved")
    ]
    if len(matches) != 1:
        raise ValueError(
            "Yahoo price request must resolve exactly one canonical mapping: "
            f"{identity.key}"
        )
    mapping = matches.iloc[0]
    mapping_id = str(mapping["Mapping_ID"])
    frame = pd.DataFrame(payload).copy()
    expected = ("Close", "Volume", "Dividends", "Stock Splits", "Capital Gains")
    if tuple(frame.columns) != expected:
        raise ValueError(
            f"Yahoo unadjusted price payload must have exactly {list(expected)}"
        )
    replacement = pd.DataFrame({
        "Date": pd.to_datetime(frame.index, errors="raise"),
        "Asset_ID": str(mapping["Asset_ID"]),
        "Source_Ticker": str(mapping["Source_Ticker"]),
        "Provider_Symbol": str(mapping["Provider_Symbol"]),
        "Mapping_ID": mapping_id,
        "Close": frame["Close"].to_numpy(),
        "Volume": frame["Volume"].to_numpy(),
        "Dividends": frame["Dividends"].to_numpy(),
        "Stock_Splits": frame["Stock Splits"].to_numpy(),
        "Capital_Gains": frame["Capital Gains"].to_numpy(),
    })
    retained = raw.loc[
        ~(
            raw["Mapping_ID"].astype(str).eq(mapping_id)
        )
    ].copy()
    combined = pd.concat([retained, replacement], ignore_index=True)
    return validate_yahoo_close_rows(
        combined,
        mappings=mappings,
    )


def _write_price_manifest(statuses: Iterable[AcquisitionStatus]) -> None:
    paths = DEFAULT_CONFIG.paths
    price_paths = paths.price_sources
    statuses = tuple(statuses)
    captured = max(
        (status.attempted_at_utc for status in statuses if status.attempted_at_utc),
        default="",
    )
    manifests = [
        ArtifactManifest.from_artifact(
            path,
            scope=SCOPE,
            dataset="prices",
            origin=ArtifactOrigin.DOWNLOADED,
            artifact_path=path.relative_to(paths.project_root).as_posix(),
            schema_version=SCHEMA_VERSION,
            captured_at_utc=captured,
        )
        for path in (
            price_paths.yahoo_close_csv,
            price_paths.wiki_extract_csv,
            price_paths.acquisition_status_csv,
            price_paths.readiness_csv,
        )
        if path.is_file()
    ]
    write_manifests(price_paths.artifact_manifest_csv, manifests)


def _persist_price_state(
    raw: pd.DataFrame,
    statuses: Iterable[AcquisitionStatus],
    *,
    ledger: pd.DataFrame,
    observations: pd.DataFrame,
    mappings: pd.DataFrame,
) -> list[ReadinessRecord]:
    paths = DEFAULT_CONFIG.paths
    price_paths = paths.price_sources
    raw = raw if not raw.empty else _empty_yahoo_prices()
    atomic_write_dataframe(
        raw.loc[:, list(YAHOO_RAW_COLUMNS)],
        price_paths.yahoo_close_csv,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    statuses = tuple(statuses)
    write_acquisition_statuses(price_paths.acquisition_status_csv, statuses)
    readiness = price_readiness_records(
        ledger,
        observations,
        mappings=mappings,
        checked_at_utc=utc_timestamp(),
    )
    write_readiness(price_paths.readiness_csv, readiness)
    _write_price_manifest(statuses)
    return readiness


def _download_pinned_wiki_parent(target: Path) -> tuple[int, str]:
    """Stream only the roadmap-pinned URL to a temporary local file."""
    request = Request(
        WIKI_PARENT_URL,
        headers={"User-Agent": "IPM/1.0"},
    )
    with urlopen(request, timeout=120) as response, Path(target).open("wb") as stream:
        status = int(getattr(response, "status", 200))
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    return status, platform.python_version()


def _replace_status(
    statuses: Iterable[AcquisitionStatus],
    replacement: AcquisitionStatus,
) -> list[AcquisitionStatus]:
    values = {
        status.identity.key: status for status in statuses
    }
    values[replacement.identity.key] = replacement
    return [values[key] for key in sorted(values)]


def _capture_wiki_prices(
    request,
    statuses: Iterable[AcquisitionStatus],
) -> tuple[list[AcquisitionStatus], bool]:
    """Capture/verify/extract the pinned parent, retaining last-good bytes on error."""
    paths = DEFAULT_CONFIG.paths
    price_paths = paths.price_sources
    attempted = utc_timestamp()
    try:
        with tempfile.TemporaryDirectory(prefix="ipm-wiki-prices-") as directory:
            temporary_dir = Path(directory)
            parent = temporary_dir / "WIKI_PRICES.csv"
            extract = temporary_dir / "wiki_extract.csv"
            http_status, client_version = _download_pinned_wiki_parent(parent)
            verified = verify_and_extract_pinned_wiki_parent(parent, extract)
            atomic_write_bytes(extract.read_bytes(), price_paths.wiki_extract_csv)
        status = AcquisitionStatus(
            identity=request.identity,
            status=ProviderStatus.OK,
            requested_start=request.requested_start,
            requested_end=request.requested_end,
            observation_count=WIKI_PARENT_ROWS,
            observation_start=WIKI_PARENT_FIRST_DATE,
            observation_end=WIKI_PARENT_LAST_DATE,
            attempted_at_utc=attempted,
            client="python-urllib",
            client_version=client_version,
            http_status=str(http_status),
        )
        if verified.empty:  # pragma: no cover - validator already forbids this
            raise ValueError("Pinned WIKI extract is empty")
        return _replace_status(statuses, status), True
    except Exception as exc:
        message = " ".join((str(exc) or repr(exc)).split())
        status = AcquisitionStatus(
            identity=request.identity,
            status=ProviderStatus.FAILED,
            requested_start=request.requested_start,
            requested_end=request.requested_end,
            attempted_at_utc=attempted,
            client="python-urllib",
            client_version=platform.python_version(),
            error_class=type(exc).__name__,
            error_message=message,
        )
        return _replace_status(statuses, status), False


def acquire_backtest_prices(request: CommandRequest) -> int:
    """Acquire every Yahoo price-domain role through one resumable engine."""
    if request.tickers_file:
        raise ValueError("backtest prices do not accept ticker selection")
    paths = DEFAULT_CONFIG.paths
    bundle = load_security_identity_bundle(paths.project_root)
    mappings = bundle.provider_mappings
    loaded_raw, statuses = _load_price_state()
    observations = load_price_observation_catalog(
        paths,
        yahoo_raw=loaded_raw,
        bundle=bundle,
    )
    raw = loaded_raw if loaded_raw is not None else _empty_yahoo_prices()
    ledger = price_requirement_ledger(
        paths,
        bundle=bundle,
    )
    requests = build_price_requests(ledger, mappings)
    yahoo_requests = [
        item for item in requests if item.identity.provider == "yahoo"
    ]
    wiki_request = next(
        item for item in requests if item.identity.provider == "wiki"
    )
    yahoo_plan = build_acquisition_plan(
        yahoo_requests,
        statuses=statuses,
        refresh=request.refresh,
    )
    wiki_plan = build_acquisition_plan(
        [wiki_request],
        statuses=yahoo_plan.statuses,
        refresh=request.refresh,
    )
    readiness_state = price_readiness_records(
        ledger,
        observations,
        mappings=mappings,
    )
    if request.dry_run:
        covered = sum(item.covered_count for item in readiness_state)
        required = sum(item.required_count for item in readiness_state)
        print(
            "Backtest prices dry-run: immutable Reuters validated; "
            f"one Yahoo plan with {len(yahoo_requests)} required mappings; "
            f"{len(yahoo_requests) - len(yahoo_plan.execution_order)} terminal; "
            f"{len(yahoo_plan.execution_order)} pending/retryable; reviewed coverage="
            f"{covered}/{required}."
        )
        _print_execution_plan(yahoo_plan, yahoo_plan.statuses)
        print(
            "Pinned WIKI capture dry-run: "
            f"{1 - len(wiki_plan.execution_order)} terminal; "
            f"{len(wiki_plan.execution_order)} pending/retryable."
        )
        _print_execution_plan(wiki_plan, wiki_plan.statuses)
        return 0

    runtime = AcquisitionRuntimePaths(request.project_root)
    report = AcquisitionReporter()
    raw_state = raw.copy()
    observations_state = observations
    with acquisition_lock(runtime.lock_path(SCOPE, "prices")):
        client = prepare_yfinance(runtime.yfinance_cache)
        adapter = make_yahoo_unadjusted_close_adapter(client)

        def checkpoint(current, outcome) -> None:
            nonlocal raw_state, observations_state, readiness_state
            if outcome is not None:
                raw_state = _replace_yahoo_price_payload(
                    raw_state,
                    outcome.request,
                    outcome.result.payload if outcome.result is not None else None,
                    mappings=mappings,
                )
                observations_state = load_price_observation_catalog(
                    paths,
                    yahoo_raw=raw_state,
                    bundle=bundle,
                )
            readiness_state = _persist_price_state(
                raw_state,
                current,
                ledger=ledger,
                observations=observations_state,
                mappings=mappings,
            )

        yahoo_run = SerialAcquisitionEngine(
            adapter,
            policy=PRODUCTION_ACQUISITION_POLICY,
            reporter=report,
        ).run(
            yahoo_requests,
            statuses=wiki_plan.statuses,
            checkpoint=checkpoint,
            refresh=request.refresh,
        )
        active_statuses = list(yahoo_run.statuses)
        wiki_plan = build_acquisition_plan(
            [wiki_request],
            statuses=active_statuses,
            refresh=request.refresh,
        )
        wiki_complete = True
        wiki_requested = bool(wiki_plan.execution_order)
        if wiki_plan.execution_order:
            active_statuses, wiki_complete = _capture_wiki_prices(
                wiki_request,
                wiki_plan.statuses,
            )
        if wiki_requested and wiki_complete:
            observations_state = load_price_observation_catalog(
                paths,
                yahoo_raw=raw_state,
                bundle=bundle,
            )
        readiness_state = _persist_price_state(
            raw_state,
            active_statuses,
            ledger=ledger,
            observations=observations_state,
            mappings=mappings,
        )

    covered = sum(item.covered_count for item in readiness_state)
    required = sum(item.required_count for item in readiness_state)
    causally_ready = all(
        item.status is ReadinessStatus.COMPLETE for item in readiness_state
    )
    provider_complete = yahoo_run.complete and wiki_complete
    if not (provider_complete and causally_ready):
        report(
            "Backtest prices remain incomplete: provider run complete="
            f"{provider_complete}; reviewed coverage={covered}/{required}; "
            "last-good raw payloads were retained."
        )
        return 1
    report(
        f"Backtest prices are causally ready: {covered}/{required} "
        "consumer-derived requirements."
    )
    return 0


def _benchmark_manifest(
    path,
    origin: ArtifactOrigin,
    captured_at_utc: str,
) -> ArtifactManifest:
    paths = DEFAULT_CONFIG.paths
    return ArtifactManifest.from_artifact(
        path,
        scope=SCOPE,
        dataset="benchmark",
        origin=origin,
        artifact_path=path.relative_to(paths.project_root).as_posix(),
        schema_version=SCHEMA_VERSION,
        captured_at_utc=captured_at_utc,
    )


def normalize_yahoo_benchmark_monthly(
    payload: pd.DataFrame,
) -> pd.DataFrame:
    """Select each calendar month's last Yahoo close and label it month-end."""
    frame = pd.DataFrame(payload)
    if "Close" not in frame.columns:
        raise ValueError("Yahoo benchmark payload is missing Close")
    monthly = pd.DataFrame({
        "Date": pd.to_datetime(frame.index, errors="raise"),
        "SP500TR_Close": pd.to_numeric(frame["Close"], errors="coerce").to_numpy(),
    }).sort_values("Date", kind="stable")
    monthly["Date"] = monthly["Date"] + pd.offsets.MonthEnd(0)
    monthly = monthly.drop_duplicates("Date", keep="last").reset_index(drop=True)
    return validate_raw_benchmark(monthly)


def _make_canonical_benchmark_adapter(client):
    """Normalize and validate the complete Yahoo window before checkpointing."""
    yahoo = make_yahoo_benchmark_adapter(client)

    def fetch(request):
        result = yahoo.fetch(request)
        candidate = normalize_yahoo_benchmark_monthly(result.payload)
        if not pd.DatetimeIndex(candidate["Date"]).equals(
            DEFAULT_CONFIG.benchmark.required_dates
        ):
            raise ValueError(
                "Yahoo benchmark does not exactly cover the configured month-ends"
            )
        return AcquisitionResult(
            payload=candidate,
            observation_count=len(candidate),
            observation_start=candidate["Date"].min().strftime("%Y-%m-%d"),
            observation_end=candidate["Date"].max().strftime("%Y-%m-%d"),
            http_status=result.http_status,
        )

    return ProviderAdapter(
        fetch,
        provider="yahoo",
        dataset="benchmark",
        client_name=yahoo.client_name,
        client_version=yahoo.client_version,
    )


def _load_benchmark_state() -> tuple[pd.DataFrame, ArtifactManifest]:
    """Load the trusted raw benchmark and its existing provenance."""
    paths = DEFAULT_CONFIG.paths
    records = read_manifests(paths.benchmark_artifact_manifest_csv)
    raw_path = paths.benchmark_raw_csv.relative_to(paths.project_root).as_posix()
    expected = {
        raw_path,
        paths.benchmark_acquisition_status_csv.relative_to(paths.project_root).as_posix(),
        paths.benchmark_readiness_csv.relative_to(paths.project_root).as_posix(),
    }
    if {item.artifact_path for item in records} != expected or any(
        item.scope != SCOPE or item.dataset != "benchmark" for item in records
    ):
        raise ValueError("Backtest benchmark manifest catalog is invalid")
    for item in records:
        validate_manifest_artifact(item, base_dir=paths.project_root)
    raw_manifest = next(item for item in records if item.artifact_path == raw_path)
    if raw_manifest.origin is not ArtifactOrigin.DOWNLOADED:
        raise ValueError("Backtest benchmark raw provenance origin must be downloaded")
    raw = validate_raw_benchmark(pd.read_csv(paths.benchmark_raw_csv))
    return raw, raw_manifest


def acquire_backtest_benchmark(request: CommandRequest) -> int:
    """Acquire, validate, and manifest the resumable Yahoo benchmark."""
    if request.tickers_file:
        raise ValueError("backtest benchmark does not accept ticker selection")
    paths = DEFAULT_CONFIG.paths
    raw, raw_manifest = _load_benchmark_state()
    requests = build_benchmark_requests(DEFAULT_CONFIG)
    statuses = (
        read_acquisition_statuses(paths.benchmark_acquisition_status_csv)
        if paths.benchmark_acquisition_status_csv.is_file()
        else []
    )
    if len(statuses) > 1 or any(
        status.identity.key != requests[0].identity.key for status in statuses
    ):
        raise ValueError("Backtest benchmark status must contain only ^SP500TR")
    plan = build_acquisition_plan(
        requests,
        statuses=statuses,
        refresh=request.refresh,
    )
    if request.dry_run:
        print(
            "Backtest benchmark dry-run: 1 Yahoo identity; "
            f"{1 - len(plan.execution_order)} terminal; "
            f"{len(plan.execution_order)} pending/retryable."
        )
        _print_execution_plan(plan, statuses)
        return 0

    runtime = AcquisitionRuntimePaths(request.project_root)
    report = AcquisitionReporter()
    captured = utc_timestamp()
    raw_state = raw
    readiness_state = benchmark_readiness_record(
        contributing_source="yahoo",
        raw=raw,
    )
    with acquisition_lock(runtime.lock_path(SCOPE, "benchmark")):
        client = prepare_yfinance(runtime.yfinance_cache)
        adapter = _make_canonical_benchmark_adapter(client)

        def checkpoint(current, outcome) -> None:
            nonlocal raw_state, raw_manifest, readiness_state
            if outcome is not None and outcome.result is not None:
                candidate = outcome.result.payload
                atomic_write_dataframe(candidate, paths.benchmark_raw_csv)
                raw_state = candidate
                raw_manifest = _benchmark_manifest(
                    paths.benchmark_raw_csv,
                    ArtifactOrigin.DOWNLOADED,
                    captured,
                )
            write_acquisition_statuses(paths.benchmark_acquisition_status_csv, current)
            readiness_state = benchmark_readiness_record(
                contributing_source="yahoo",
                raw=raw_state,
                checked_at_utc=captured,
            )
            write_readiness(paths.benchmark_readiness_csv, (readiness_state,))
            write_manifests(
                paths.benchmark_artifact_manifest_csv,
                (
                    raw_manifest,
                    *(
                        _benchmark_manifest(path, ArtifactOrigin.DOWNLOADED, captured)
                        for path in (
                            paths.benchmark_acquisition_status_csv,
                            paths.benchmark_readiness_csv,
                        )
                    ),
                ),
            )

        run = SerialAcquisitionEngine(
            adapter,
            policy=PRODUCTION_ACQUISITION_POLICY,
            reporter=report,
        ).run(
            requests,
            statuses=statuses,
            checkpoint=checkpoint,
            refresh=request.refresh,
        )
    if not run.complete or readiness_state.status is not ReadinessStatus.COMPLETE:
        report(
            "Backtest benchmark remains incomplete: "
            f"coverage={readiness_state.covered_count}/"
            f"{readiness_state.required_count}; the last-good raw file was retained."
        )
        return 1
    report(
        "Backtest benchmark is ready: "
        f"{readiness_state.covered_count}/{readiness_state.required_count} "
        "Yahoo-derived month-ends."
    )
    return 0


def _read_shares_state() -> tuple[pd.DataFrame, list[AcquisitionStatus]]:
    """Load retained canonical evidence, or initialize an empty acquisition."""
    path = DEFAULT_CONFIG.paths.shares.acquisition_status_csv
    raw_path = DEFAULT_CONFIG.paths.shares.raw_shares_csv
    statuses = read_acquisition_statuses(path) if path.exists() else []
    raw = (load_raw_shares(raw_path) if raw_path.exists()
           else pd.DataFrame(columns=RAW_SHARES_COLUMNS))
    validate_share_status_reconciliation(
        [status for status in statuses if status.identity.provider == "yahoo"],
        raw,
    )
    return raw, statuses


def _latest_attempted_at(statuses: Iterable[AcquisitionStatus]) -> str:
    return max(
        (item.attempted_at_utc for item in statuses if item.attempted_at_utc),
        default="",
    )


def _aggregate_shares_provenance(
    statuses: Iterable[AcquisitionStatus],
) -> tuple[ArtifactOrigin, str]:
    """Describe a consolidated checkpoint without erasing migrated lineage.

    Per-identity provider attempts live in acquisition_status.csv.  The
    aggregate raw/status/readiness files remain ``migrated`` while any Yahoo
    identity still comes from the imported checkpoint; they become wholly
    ``downloaded`` only after every migrated identity has been reacquired.
    """

    statuses = tuple(statuses)
    yahoo = [item for item in statuses if item.identity.provider == "yahoo"]
    captured = _latest_attempted_at(
        item for item in yahoo if item.status is ProviderStatus.OK
    )
    if any(item.migration_note for item in yahoo):
        return ArtifactOrigin.MIGRATED, captured
    if captured:
        return ArtifactOrigin.DOWNLOADED, captured
    return ArtifactOrigin.MIGRATED, ""


def _write_shares_manifest(statuses: Iterable[AcquisitionStatus]) -> None:
    paths = DEFAULT_CONFIG.paths.shares
    statuses = tuple(statuses)
    raw_origin, raw_captured_at = _aggregate_shares_provenance(statuses)
    checkpoint_captured_at = _latest_attempted_at(statuses)
    checkpoint_origin = (
        ArtifactOrigin.DOWNLOADED
        if checkpoint_captured_at else raw_origin
    )
    artifacts = [
        (
            paths.raw_shares_csv,
            raw_origin,
            raw_captured_at,
        ),
        (
            paths.acquisition_status_csv,
            checkpoint_origin,
            checkpoint_captured_at,
        ),
        (
            paths.readiness_csv,
            checkpoint_origin,
            checkpoint_captured_at,
        ),
    ]
    manifests = [
        ArtifactManifest.from_artifact(
            path,
            scope=SCOPE,
            dataset="shares",
            origin=artifact_origin,
            artifact_path=path.relative_to(DEFAULT_CONFIG.paths.project_root).as_posix(),
            schema_version=SCHEMA_VERSION,
            captured_at_utc=artifact_captured_at,
        )
        for path, artifact_origin, artifact_captured_at in artifacts if path.exists()
    ]
    write_manifests(paths.artifact_manifest_csv, manifests)


def _print_execution_plan(
    plan: AcquisitionPlan,
    previous_statuses: Iterable[AcquisitionStatus],
) -> None:
    """Print provider requests in the exact order used by the shared engine."""
    previous_by_key = {
        status.identity.key: status.status.value for status in previous_statuses
    }
    total = len(plan.execution_order)
    for position, acquisition_request in enumerate(plan.execution_order, start=1):
        identity = acquisition_request.identity
        previous = previous_by_key.get(identity.key, "missing")
        effective_start = identity.effective_start or "*"
        effective_end = identity.effective_end or "*"
        print(
            f"  [{position}/{total}] previous={previous} "
            f"{identity.asset_id} -> {identity.provider}:{identity.provider_symbol} "
            f"effective=[{effective_start},{effective_end}] "
            f"request=[{acquisition_request.requested_start},"
            f"{acquisition_request.requested_end}]"
        )


def acquire_backtest_shares(request: CommandRequest) -> int:
    market = load_fresh_core_data(DEFAULT_CONFIG)
    source_plan = plan_share_sources(market, config=DEFAULT_CONFIG)
    requests = build_shares_requests(
        source_plan,
        tickers_file=request.tickers_file,
    )
    needs_yahoo = bool(source_plan.yahoo_identities)
    raw, statuses = (_read_shares_state() if needs_yahoo
                     else (pd.DataFrame(columns=RAW_SHARES_COLUMNS), []))
    yahoo_plan = build_acquisition_plan(
        requests,
        statuses=statuses,
        refresh=request.refresh,
    )
    if request.dry_run:
        print(
            "Backtest shares dry-run: "
            f"{len(requests)} effective-dated Yahoo identities; "
            f"{len(requests) - len(yahoo_plan.execution_order)} terminal; "
            f"{len(yahoo_plan.execution_order)} pending/retryable."
        )
        _print_execution_plan(yahoo_plan, statuses)
        return 0

    runtime = AcquisitionRuntimePaths(request.project_root)
    paths = DEFAULT_CONFIG.paths.shares
    raw_state = raw.copy()
    report = AcquisitionReporter()
    with acquisition_lock(runtime.lock_path(SCOPE, "shares")):
        def checkpoint(current, outcome) -> None:
            nonlocal raw_state
            if outcome is not None:
                raw_state = replace_share_payload(
                    raw_state,
                    outcome.request,
                    outcome.result.payload if outcome.result is not None else None,
                )
            atomic_write_dataframe(raw_state, paths.raw_shares_csv)
            write_acquisition_statuses(paths.acquisition_status_csv, current)

        active_statuses = list(statuses)
        provider_complete = True
        if yahoo_plan.execution_order:
            client = prepare_yfinance(runtime.yfinance_cache)
            yahoo_run = SerialAcquisitionEngine(
                make_yahoo_shares_adapter(client),
                policy=PRODUCTION_ACQUISITION_POLICY, reporter=report,
            ).run(requests, statuses=statuses, checkpoint=checkpoint, refresh=request.refresh)
            active_statuses = list(yahoo_run.statuses)
            provider_complete = yahoo_run.complete
        readiness, _ = shares_readiness_records(source_plan, raw_state)
        write_readiness(paths.readiness_csv, readiness)
        _write_shares_manifest(active_statuses)
        if (
            not provider_complete
            or readiness[0].status is not ReadinessStatus.COMPLETE
        ):
            report(
                "Backtest shares remain incomplete: provider run complete="
                f"{provider_complete}; reviewed coverage={readiness[0].covered_count}/"
                f"{readiness[0].required_count}."
            )
            return 1
        report(
            "Backtest shares are ready: "
            f"{readiness[0].covered_count}/{readiness[0].required_count}."
        )
    return 0


__all__ = [
    "acquire_backtest_benchmark",
    "acquire_backtest_prices",
    "acquire_backtest_shares",
    "normalize_yahoo_benchmark_monthly",
]
