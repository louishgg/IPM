"""Live provider execution, checkpoint persistence, and acquisition manifests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from data_acquisition.providers.base import AcquisitionResult, ProviderAdapter
from data_acquisition.errors import AcquisitionReviewRequired
from portfolio_core.shares import (
    RAW_SHARES_COLUMNS,
    load_raw_shares,
    select_canonical_share_observations_asof,
    validate_raw_shares,
)

from . import acquisition_planning as live_plan
from .config import DEFAULT_CONFIG
from .corporate_action_policy import (
    CORPORATE_ACTION_COLUMNS,
    load_corporate_actions,
)
from .price_coverage import (
    PRICE_REQUIREMENT_SET,
    REQUIREMENT_COLUMNS,
    PriceRequirement,
    aggregate_price_readiness,
    build_coverage_ledger,
    build_price_requirements,
    merge_supplemental_statuses,
    read_yahoo_supplement,
    supplemental_available_dates,
)
from .strategy_universe import evaluation_periods, load_validated_strategy_universe

from data_acquisition.provider_clients import prepare_yfinance
from data_acquisition.sector_acquisition_execution import execute_sector_acquisition
from data_acquisition.engine import (
    AcquisitionOutcome,
    PRODUCTION_ACQUISITION_POLICY,
    SerialAcquisitionEngine,
)
from data_acquisition.providers.yahoo import (
    make_yahoo_benchmark_adapter,
    make_yahoo_ohlcv_adapter,
    make_yahoo_shares_adapter,
)
from data_acquisition.runtime import (
    AcquisitionReporter,
    AcquisitionRuntimePaths,
    acquisition_lock,
)
from data_acquisition.share_checkpoints import (
    reconcile_share_checkpoints,
    replace_share_payload,
    validate_share_status_reconciliation,
)
from portfolio_core.io import atomic_write_dataframe
from portfolio_core.artifacts import ArtifactManifest, ArtifactOrigin, write_manifests
from data_acquisition.contracts import (
    AcquisitionRequest,
    AcquisitionStatus,
    CommandRequest,
    ProviderStatus,
    ReadinessRecord,
    ReadinessStatus,
    readiness_status,
    read_acquisition_statuses,
    utc_timestamp,
    write_acquisition_statuses,
    write_readiness,
)


RAW_BENCHMARK_COLUMNS = ("Date", "Open", "High", "Low", "Close")


def _read_statuses(dataset: str) -> list[AcquisitionStatus]:
    path = live_plan.status_path(dataset, DEFAULT_CONFIG)
    return read_acquisition_statuses(path) if path.is_file() else []


def _statuses_for_requests(
    dataset: str,
    requests: Iterable[AcquisitionRequest],
) -> tuple[list[AcquisitionStatus], list[AcquisitionStatus]]:
    """Return exact resumable statuses and unrelated records to preserve."""

    requests = tuple(requests)
    existing = _read_statuses(dataset)
    request_keys = {item.identity.key for item in requests}
    exact = [item for item in existing if item.identity.key in request_keys]
    unrelated = [
        item
        for item in existing
        if item.identity.key not in request_keys
    ]
    return exact, unrelated


def _write_statuses(
    dataset: str,
    current: Iterable[AcquisitionStatus],
    unrelated: Iterable[AcquisitionStatus],
) -> list[AcquisitionStatus]:
    combined = sorted(
        [*unrelated, *current], key=lambda item: item.identity.key
    )
    write_acquisition_statuses(
        live_plan.status_path(dataset, DEFAULT_CONFIG),
        combined,
    )
    return combined


def _read_panel(path: Path) -> pd.DataFrame:
    if not path.is_file():
        result = pd.DataFrame(dtype=float)
        result.index = pd.DatetimeIndex([], name="Date")
        return result
    frame = pd.read_csv(path, index_col=0, float_precision="round_trip")
    frame.index = pd.to_datetime(frame.index, errors="raise")
    frame.index.name = "Date"
    return frame.apply(pd.to_numeric, errors="raise").sort_index()


def _write_panel(frame: pd.DataFrame, path: Path) -> None:
    # ``reset_index`` inserts the Date column.  Consolidate first so that a
    # resumable wide checkpoint assembled over many ticker updates does not
    # trigger pandas' highly-fragmented-frame warning during that insertion.
    result = frame.sort_index().reindex(sorted(frame.columns), axis=1).copy()
    result.index.name = "Date"
    atomic_write_dataframe(result.reset_index(), path)


def _align_price_panels(
    panels: dict[str, pd.DataFrame],
    *,
    extra_index: pd.Index | None = None,
) -> None:
    """Keep Open, Close, and Volume on one union index and column set."""

    union_index = pd.DatetimeIndex([], name="Date")
    union_columns: set[str] = set()
    for frame in panels.values():
        union_index = union_index.union(pd.DatetimeIndex(frame.index))
        union_columns.update(map(str, frame.columns))
    if extra_index is not None:
        union_index = union_index.union(pd.DatetimeIndex(extra_index))
    union_index = union_index.sort_values()
    columns = sorted(union_columns)
    for name, frame in tuple(panels.items()):
        # A deep copy consolidates blocks left by earlier incremental updates.
        panels[name] = frame.reindex(index=union_index, columns=columns).copy()
        panels[name].index.name = "Date"


def _merge_price_payload(
    panels: dict[str, pd.DataFrame],
    provider_symbol: str,
    payload: pd.DataFrame,
    *,
    allow_reprice: bool = False,
) -> None:
    """Extend cached Yahoo observations, rejecting a changed adjusted-price basis."""

    _align_price_panels(panels, extra_index=payload.index)
    for name, field in (
        ("open", "Open"),
        ("close", "Close"),
        ("volume", "Volume"),
    ):
        panel = panels[name]
        incoming = pd.to_numeric(payload[field], errors="raise").reindex(panel.index)
        prior = (
            pd.to_numeric(panel[provider_symbol], errors="raise")
            if provider_symbol in panel else pd.Series(np.nan, index=panel.index)
        )
        overlap = prior.notna() & incoming.notna()
        if not allow_reprice and overlap.any() and not np.isclose(
            prior.loc[overlap].to_numpy(dtype=float),
            incoming.loc[overlap].to_numpy(dtype=float),
            rtol=1e-6, atol=1e-8,
        ).all():
            conflicting = prior.index[overlap & ~np.isclose(
                prior.fillna(0).to_numpy(dtype=float),
                incoming.fillna(0).to_numpy(dtype=float),
                rtol=1e-6, atol=1e-8,
            )]
            raise ValueError(
                f"Yahoo {field} price basis changed for {provider_symbol} on "
                f"{conflicting.strftime('%Y-%m-%d').tolist()[:10]}"
            )
        merged = (
            incoming.combine_first(prior) if allow_reprice
            else prior.combine_first(incoming)
        ).rename(provider_symbol)
        panels[name] = pd.concat(
            [panel.drop(columns=provider_symbol, errors="ignore"), merged], axis=1
        ).copy()
        panels[name].index.name = "Date"


def _extension_price_adapter(
    adapter: ProviderAdapter,
    panels: dict[str, pd.DataFrame],
    statuses: Iterable[AcquisitionStatus],
    *,
    review_threshold: float = 0.01,
    evidence_dir: Path | None = None,
    replacement_notes: dict | None = None,
    supplemental: pd.DataFrame | None = None,
) -> ProviderAdapter:
    """Fetch the full window and review replacement before changing the cache.

    Accept small provider revisions directly, without scaling. Missing cached
    observations or revisions above the threshold hold the entire security for
    review. A refresh does not bypass this guard. Short history for a security
    with no cached observations remains an eligibility issue, not a fetch error.
    """
    statuses = tuple(statuses)
    held = {
        item.identity.provider_symbol: item.error_message
        for item in statuses
        if item.error_class == "AcquisitionReviewRequired"
    }
    held_notes = {
        item.identity.provider_symbol: item.migration_note
        for item in statuses
        if item.error_class == "AcquisitionReviewRequired"
    }
    fields = (("open", "Open"), ("close", "Close"), ("volume", "Volume"))
    supplemental_by_asset = (
        {
            asset_id: frame.set_index("Date")
            for asset_id, frame in supplemental.groupby("Asset_ID")
        }
        if supplemental is not None and not supplemental.empty else {}
    )

    def fetch(request: AcquisitionRequest) -> AcquisitionResult:
        symbol = request.identity.provider_symbol
        if symbol in held:
            if replacement_notes is not None and held_notes.get(symbol):
                replacement_notes[request.identity.key] = held_notes[symbol]
            raise AcquisitionReviewRequired(held[symbol])
        result = adapter.fetch(request)
        fresh = result.payload
        changes = {}
        missing = {}
        reasons = []
        supplement = supplemental_by_asset.get(request.identity.asset_id)
        for name, field in fields:
            prior = (
                panels[name][symbol] if symbol in panels[name]
                else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
            )
            # The prepared cache also consumes approved supplemental Yahoo rows.
            # Replacing only their overlap could otherwise silently mix bases.
            if supplement is not None:
                prior = prior.combine_first(supplement[field])
            prior = prior.dropna().loc[request.requested_start:request.requested_end]
            if prior.empty:
                continue
            incoming = fresh[field].reindex(prior.index)
            absent = incoming.isna()
            if absent.any():
                missing[field] = prior.index[absent].strftime("%Y-%m-%d").tolist()
            common = ~absent
            before = prior.loc[common].to_numpy(dtype=float)
            after = incoming.loc[common].to_numpy(dtype=float)
            # A nonzero revision of a cached zero-volume observation also needs
            # review. Equal zeros have zero relative change.
            delta = np.abs(after - before)
            relative = np.divide(
                delta, np.abs(before), out=np.full_like(delta, np.inf),
                where=before != 0,
            )
            relative[delta == 0] = 0
            maximum = float(relative.max()) if len(relative) else 0.0
            changes[field] = {
                "overlap_rows": int(common.sum()),
                "maximum_relative_change": maximum if np.isfinite(maximum) else "infinite",
            }
            if maximum > review_threshold + 1e-12:
                reasons.append(f"{field} revision {maximum:.3%} exceeds {review_threshold:.1%}")
        if missing:
            missing_dates = sorted({date for dates in missing.values() for date in dates})
            reasons.insert(0, f"new download omits {len(missing_dates)} cached dates")

        # Retain the provider response unchanged for inspection, including when
        # replacement is blocked. The checksum links it to its status record.
        payload_bytes = fresh.to_csv(index=True, index_label="Date").encode("utf-8")
        digest = hashlib.sha256(payload_bytes).hexdigest()
        if evidence_dir is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            evidence = evidence_dir / f"{digest}.csv"
            if not evidence.exists():
                evidence.write_bytes(payload_bytes)
        note = {
            "price_replacement": "review_required" if reasons else "accepted_provider_values",
            "review_threshold": review_threshold,
            "comparison": changes,
            "missing_cached_dates": missing,
            "review_reasons": reasons,
            "provider_response_sha256": digest,
            "provider_response": f"replacement_responses/{digest}.csv",
        }
        if replacement_notes is not None:
            replacement_notes[request.identity.key] = json.dumps(note, sort_keys=True)
        if reasons:
            message = f"{symbol}: {'; '.join(reasons)}; cached series retained pending review"
            held[symbol] = message
            held_notes[symbol] = json.dumps(note, sort_keys=True)
            raise AcquisitionReviewRequired(message)

        candidate = {
            name: frame.loc[:, [symbol]].copy()
            if symbol in frame else pd.DataFrame(index=frame.index)
            for name, frame in panels.items()
        }
        _merge_price_payload(candidate, symbol, fresh, allow_reprice=True)
        merged = pd.DataFrame({
            field: candidate[name][symbol] for name, field in fields
        }).dropna()
        return AcquisitionResult(
            payload=merged, observation_count=len(merged),
            observation_start=merged.index.min().date().isoformat(),
            observation_end=merged.index.max().date().isoformat(),
            http_status=result.http_status,
        )

    return ProviderAdapter(
        fetch, provider=adapter.provider, dataset=adapter.dataset,
        client_name=adapter.client_name, client_version=adapter.client_version,
    )


def _price_status_matches(
    status: AcquisitionStatus,
    panels: dict[str, pd.DataFrame],
) -> bool:
    symbol = status.identity.provider_symbol
    if not all(symbol in panel.columns for panel in panels.values()):
        return False
    valid = (
        panels["open"][symbol].gt(0)
        & panels["close"][symbol].gt(0)
        & panels["volume"][symbol].ge(0)
        & panels["open"][symbol].notna()
        & panels["close"][symbol].notna()
        & panels["volume"][symbol].notna()
    )
    dates = panels["open"].index[valid]
    return (
        len(dates) == status.observation_count
        and len(dates) > 0
        and dates.min().strftime("%Y-%m-%d") == status.observation_start
        and dates.max().strftime("%Y-%m-%d") == status.observation_end
    )


def _reconcile_price_checkpoint(
    requests: Iterable[AcquisitionRequest],
    statuses: Iterable[AcquisitionStatus],
    panels: dict[str, pd.DataFrame],
) -> list[AcquisitionStatus]:
    """Validate successful checkpoints while retaining last-good payloads."""

    by_key = {item.identity.key: item for item in statuses}
    reconciled = []
    for acquisition_request in requests:
        identity = acquisition_request.identity
        status = by_key.get(identity.key)
        if status is not None and status.status is ProviderStatus.OK:
            if _price_status_matches(status, panels):
                reconciled.append(status)
                continue
            reconciled.append(
                AcquisitionStatus.pending(
                    acquisition_request,
                    migration_note=(
                        "Terminal provider status was reset because the live "
                        "price payload did not match its checkpoint. The cached "
                        "observations were retained for comparison with the next download."
                    ),
                )
            )
            continue
        if status is not None:
            # Pending, failed, and no-data attempts do not erase a previously
            # successful payload. Readiness independently describes usability.
            reconciled.append(status)
    return reconciled


def _write_manifest(
    manifest_path: Path,
    dataset: str,
    artifacts: Iterable[Path],
    *,
    captured_at_utc: str,
) -> None:
    directory = manifest_path.parent
    manifests = []
    for artifact in artifacts:
        manifests.append(
            ArtifactManifest.from_artifact(
                artifact,
                scope="live",
                dataset=dataset,
                origin=ArtifactOrigin.DOWNLOADED,
                artifact_path=artifact.relative_to(directory).as_posix(),
                captured_at_utc=captured_at_utc,
            )
        )
    write_manifests(manifest_path, manifests)


def _available_price_dates(
    panels: Mapping[str, pd.DataFrame],
) -> dict[str, set[str]]:
    available: dict[str, set[str]] = {}
    common_columns = set.intersection(*(set(frame.columns) for frame in panels.values()))
    common_index = panels["open"].index
    for frame in panels.values():
        common_index = common_index.intersection(frame.index)
    aligned = {name: frame.reindex(common_index) for name, frame in panels.items()}
    for symbol in common_columns:
        valid = pd.Series(True, index=common_index)
        valid &= aligned["open"][symbol].gt(0)
        valid &= aligned["close"][symbol].gt(0)
        valid &= aligned["volume"][symbol].ge(0)
        available[str(symbol)] = set(common_index[valid].strftime("%Y-%m-%d"))
    return available


def _build_price_readiness(
    panels: Mapping[str, pd.DataFrame],
    *,
    requirements: Iterable[PriceRequirement],
    symbol_by_asset: Mapping[str, str],
    supplemental_dates: Mapping[str, set[str]],
    corporate_actions: pd.DataFrame,
    checked_at_utc: str,
) -> tuple[pd.DataFrame, list[ReadinessRecord]]:
    available_by_symbol = _available_price_dates(panels)
    yahoo_dates = {
        asset_id: available_by_symbol.get(symbol, set())
        for asset_id, symbol in symbol_by_asset.items()
    }
    ledger = build_coverage_ledger(
        requirements,
        yahoo_dates=yahoo_dates,
        supplemental_dates=supplemental_dates,
        corporate_actions=corporate_actions,
    )
    records = aggregate_price_readiness(
        ledger,
        checked_at_utc=checked_at_utc,
    )
    return ledger, records


def _load_raw_shares() -> pd.DataFrame:
    path = DEFAULT_CONFIG.paths.shares.raw_shares_csv
    if not path.is_file():
        return pd.DataFrame(columns=RAW_SHARES_COLUMNS)
    return load_raw_shares(path)


def _load_raw_benchmark(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=RAW_BENCHMARK_COLUMNS)
    frame = pd.read_csv(path, keep_default_na=False)
    if tuple(frame.columns) != RAW_BENCHMARK_COLUMNS:
        raise ValueError(
            f"Live raw benchmark must contain exactly "
            f"{list(RAW_BENCHMARK_COLUMNS)}"
        )
    if frame.empty:
        return frame
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if frame["Date"].duplicated().any():
        raise ValueError("Live raw benchmark dates must be unique")
    for field in ("Open", "High", "Low", "Close"):
        frame[field] = pd.to_numeric(frame[field], errors="raise")
    values = frame[["Open", "High", "Low", "Close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Live raw benchmark must be finite")
    return frame.sort_values("Date", kind="stable").reset_index(drop=True)


def _benchmark_status_matches(
    status: AcquisitionStatus,
    frame: pd.DataFrame,
) -> bool:
    if frame.empty:
        return False
    positive = (frame[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
    dates = frame.loc[positive, "Date"]
    return (
        bool(positive.all())
        and len(frame) == status.observation_count
        and dates.min().strftime("%Y-%m-%d") == status.observation_start
        and dates.max().strftime("%Y-%m-%d") == status.observation_end
    )


def _reconcile_benchmark_checkpoint(
    acquisition_request: AcquisitionRequest,
    statuses: Iterable[AcquisitionStatus],
    raw: pd.DataFrame,
) -> list[AcquisitionStatus]:
    status = next(
        (
            item
            for item in statuses
            if item.identity.key == acquisition_request.identity.key
        ),
        None,
    )
    if status is not None and status.status is ProviderStatus.OK:
        if _benchmark_status_matches(status, raw):
            return [status]
        return [
            AcquisitionStatus.pending(
                acquisition_request,
                migration_note=(
                    "Terminal provider status was reset because the live "
                    "benchmark payload did not match its checkpoint."
                ),
            )
        ]
    return [status] if status is not None else []


def _write_raw_shares(frame: pd.DataFrame) -> None:
    frame = validate_raw_shares(frame)
    atomic_write_dataframe(
        frame.loc[:, RAW_SHARES_COLUMNS],
        DEFAULT_CONFIG.paths.shares.raw_shares_csv,
    )


def _write_shares_readiness(
    raw: pd.DataFrame,
    checked_at_utc: str,
) -> list[ReadinessRecord]:
    requirements = live_plan.shares_requirement_dates()
    selected = select_canonical_share_observations_asof(
        raw,
        (
            (date, asset_id)
            for asset_id, dates in sorted(requirements.items())
            for date in dates
        ),
    )
    covered = {
        (row.Asset_ID, row.Date)
        for row in selected.itertuples(index=False)
    }
    records = []
    for asset_id, dates in sorted(requirements.items()):
        covered_dates = {
            value
            for value in dates
            if (asset_id, pd.Timestamp(value)) in covered
        }
        missing = tuple(sorted(set(dates) - covered_dates))
        records.append(
            ReadinessRecord(
                scope="live",
                dataset="shares",
                asset_id=asset_id,
                requirement_set="active_execution_valuation_shares",
                required_count=len(dates),
                covered_count=len(covered_dates),
                status=readiness_status(len(covered_dates), len(dates)),
                missing_dates=missing,
                contributing_sources=(("yahoo",) if covered_dates else ()),
                checked_at_utc=checked_at_utc,
            )
        )
    write_readiness(DEFAULT_CONFIG.paths.shares.readiness_csv, records)
    return records


def _write_benchmark_readiness(
    frame: pd.DataFrame,
    checked_at_utc: str,
) -> list[ReadinessRecord]:
    dates = live_plan.benchmark_requirement_dates()
    indexed = frame.copy()
    if not indexed.empty:
        indexed["Date"] = pd.to_datetime(indexed["Date"], errors="raise")
        indexed = indexed.set_index("Date")
    covered = []
    for value in dates:
        timestamp = pd.Timestamp(value)
        if timestamp not in indexed.index:
            continue
        row = indexed.loc[timestamp]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        if all(pd.notna(row.get(field)) and float(row[field]) > 0 for field in ("Open", "Close")):
            covered.append(value)
    missing = tuple(sorted(set(dates) - set(covered)))
    record = ReadinessRecord(
        scope="live",
        dataset="benchmark",
        asset_id=DEFAULT_CONFIG.benchmark.ticker,
        requirement_set="execution_valuation_benchmark",
        required_count=len(dates),
        covered_count=len(covered),
        status=readiness_status(len(covered), len(dates)),
        missing_dates=missing,
        contributing_sources=(("yahoo",) if covered else ()),
        checked_at_utc=checked_at_utc,
    )
    write_readiness(DEFAULT_CONFIG.paths.raw_benchmark_readiness_csv, (record,))
    return [record]


def _all_ready(records: Iterable[ReadinessRecord]) -> bool:
    values = tuple(records)
    return bool(values) and all(item.status is ReadinessStatus.COMPLETE for item in values)


def _execute_prices(
    request: CommandRequest,
    reporter: AcquisitionReporter,
    client,
) -> int:
    paths = DEFAULT_CONFIG.paths
    all_requests = live_plan.build_command_requests(request)
    panels = {
        "open": _read_panel(paths.raw_price_open_csv),
        "close": _read_panel(paths.raw_price_close_csv),
        "volume": _read_panel(paths.raw_price_volume_csv),
    }
    _align_price_panels(panels)
    captured = utc_timestamp()

    schedule, membership, metadata = load_validated_strategy_universe()
    requirements = build_price_requirements(
        schedule, membership,
        evaluation=evaluation_periods(
            schedule,
            evaluation_start=DEFAULT_CONFIG.market.competition_start,
            evaluation_end=DEFAULT_CONFIG.market.competition_end,
        ),
    )
    symbol_by_asset = metadata.set_index("Asset_ID")["Yahoo_Ticker"].to_dict()

    supplemental = pd.DataFrame()
    artifact_exists = paths.raw_price_supplemental_csv.is_file()
    manifest_exists = paths.raw_price_supplemental_manifest_csv.is_file()
    if artifact_exists != manifest_exists:
        raise ValueError("Yahoo supplement artifact/manifest is incomplete")
    if artifact_exists:
        supplemental = read_yahoo_supplement(paths)
        merge_supplemental_statuses(paths.raw_price_status_csv, supplemental)
    supplemental_dates = (
        supplemental_available_dates(supplemental)
        if not supplemental.empty
        else {}
    )

    actions = pd.DataFrame(columns=CORPORATE_ACTION_COLUMNS)
    artifact_exists = paths.raw_corporate_action_policy_csv.is_file()
    manifest_exists = paths.raw_corporate_action_policy_manifest_csv.is_file()
    if artifact_exists != manifest_exists:
        raise ValueError("Corporate-action policy artifact/manifest is incomplete")
    if artifact_exists:
        actions = load_corporate_actions(paths)

    exact, unrelated = _statuses_for_requests("prices", all_requests)
    exact = _reconcile_price_checkpoint(
        all_requests,
        exact,
        panels,
    )
    ledger, readiness = _build_price_readiness(
        panels,
        requirements=requirements,
        symbol_by_asset=symbol_by_asset,
        supplemental_dates=supplemental_dates,
        corporate_actions=actions,
        checked_at_utc=captured,
    )
    source_covered_assets = {
        item.asset_id
        for item in readiness
        if item.requirement_set == PRICE_REQUIREMENT_SET
        and item.status is ReadinessStatus.COMPLETE
    }
    exact_keys = {item.identity.key for item in exact}
    status_by_key = {item.identity.key: item for item in exact}
    requests = tuple(
        acquisition_request
        for acquisition_request in all_requests
        if request.refresh
        or acquisition_request.identity.asset_id not in source_covered_assets
        or acquisition_request.identity.key not in exact_keys
        or not live_plan.price_request_window_matches(
            acquisition_request, status_by_key[acquisition_request.identity.key],
        )
    )
    request_keys = {item.identity.key for item in requests}
    preserved_exact = [
        item for item in exact if item.identity.key not in request_keys
    ]
    exact = [item for item in exact if item.identity.key in request_keys]
    unrelated = [*unrelated, *preserved_exact]
    artifacts = (
        paths.raw_price_open_csv,
        paths.raw_price_close_csv,
        paths.raw_price_volume_csv,
        paths.raw_price_status_csv,
        paths.raw_price_readiness_csv,
        paths.raw_price_requirements_csv,
    )
    replacement_notes = {
        item.identity.key: item.migration_note for item in exact if item.migration_note
    }
    adapter = _extension_price_adapter(
        make_yahoo_ohlcv_adapter(client), panels, [*exact, *unrelated],
        review_threshold=DEFAULT_CONFIG.market.price_replacement_review_threshold,
        evidence_dir=paths.raw_prices_dir / "replacement_responses",
        replacement_notes=replacement_notes,
        supplemental=supplemental,
    )

    def checkpoint(current, outcome: AcquisitionOutcome | None) -> None:
        nonlocal ledger, readiness
        if outcome is not None:
            symbol = outcome.request.identity.provider_symbol
            if outcome.result is not None:
                payload = outcome.result.payload
                _merge_price_payload(
                    panels, symbol, payload, allow_reprice=True,
                )
                ledger, readiness = _build_price_readiness(
                    panels,
                    requirements=requirements,
                    symbol_by_asset=symbol_by_asset,
                    supplemental_dates=supplemental_dates,
                    corporate_actions=actions,
                    checked_at_utc=captured,
                )
        price_paths = {
            "open": paths.raw_price_open_csv,
            "close": paths.raw_price_close_csv,
            "volume": paths.raw_price_volume_csv,
        }
        for name, panel in panels.items():
            _write_panel(panel, price_paths[name])
        _write_statuses("prices", [
            replace(item, migration_note=replacement_notes.get(
                item.identity.key, item.migration_note,
            ))
            for item in current
        ], unrelated)
        atomic_write_dataframe(
            ledger.loc[:, REQUIREMENT_COLUMNS],
            paths.raw_price_requirements_csv,
        )
        write_readiness(paths.raw_price_readiness_csv, readiness)
        if all(path.is_file() for path in artifacts):
            _write_manifest(
                paths.raw_price_artifact_manifest_csv,
                "prices",
                artifacts,
                captured_at_utc=captured,
            )

    run = SerialAcquisitionEngine(
        adapter,
        policy=PRODUCTION_ACQUISITION_POLICY,
        reporter=reporter,
    ).run(
        requests,
        statuses=exact,
        checkpoint=checkpoint,
        refresh=request.refresh,
    )
    if not run.complete or not _all_ready(readiness):
        reporter(
            "Live prices remain incomplete; provider status and readiness were "
            "checkpointed for a resumable rerun."
        )
        return 1
    return 0


def _execute_shares(
    request: CommandRequest,
    reporter: AcquisitionReporter,
    client,
) -> int:
    paths = DEFAULT_CONFIG.paths
    requests = live_plan.build_command_requests(request)
    exact, unrelated = _statuses_for_requests("shares", requests)
    raw = _load_raw_shares()
    exact, raw = reconcile_share_checkpoints(
        requests,
        exact,
        raw,
        reset_note=(
            "Terminal provider status was reset because the live shares "
            "payload did not match its full identity checkpoint."
        ),
    )
    captured = utc_timestamp()
    adapter = make_yahoo_shares_adapter(client)
    readiness: list[ReadinessRecord] = []

    def checkpoint(current, outcome: AcquisitionOutcome | None) -> None:
        nonlocal raw, readiness
        if outcome is not None:
            raw = replace_share_payload(
                raw,
                outcome.request,
                outcome.result.payload if outcome.result is not None else None,
            )
        _write_raw_shares(raw)
        _write_statuses("shares", current, unrelated)
        readiness = _write_shares_readiness(raw, captured)
        _write_manifest(
            paths.shares.artifact_manifest_csv,
            "shares",
            (
                paths.shares.raw_shares_csv,
                paths.shares.acquisition_status_csv,
                paths.shares.readiness_csv,
            ),
            captured_at_utc=captured,
        )

    run = SerialAcquisitionEngine(
        adapter,
        policy=PRODUCTION_ACQUISITION_POLICY,
        reporter=reporter,
    ).run(
        requests,
        statuses=exact,
        checkpoint=checkpoint,
        refresh=request.refresh,
    )
    active_statuses = tuple(sorted(
        (*run.statuses, *unrelated),
        key=lambda item: item.identity.key,
    ))
    validate_share_status_reconciliation(
        active_statuses,
        raw,
    )
    if not run.complete or not _all_ready(readiness):
        reporter(
            "Live shares remain incomplete; provider status and causal coverage "
            "were checkpointed for a resumable rerun."
        )
        return 1
    return 0


def _execute_benchmark(
    request: CommandRequest,
    reporter: AcquisitionReporter,
    client,
) -> int:
    paths = DEFAULT_CONFIG.paths
    requests = live_plan.build_command_requests(request)
    exact, unrelated = _statuses_for_requests("benchmark", requests)
    raw = _load_raw_benchmark(paths.raw_benchmark_csv)
    exact = _reconcile_benchmark_checkpoint(
        requests[0],
        exact,
        raw,
    )
    captured = utc_timestamp()
    adapter = make_yahoo_benchmark_adapter(client)
    readiness: list[ReadinessRecord] = []

    def checkpoint(current, outcome: AcquisitionOutcome | None) -> None:
        nonlocal raw, readiness
        if outcome is not None and outcome.result is not None:
            raw = outcome.result.payload.reset_index().loc[
                :, RAW_BENCHMARK_COLUMNS
            ]
            atomic_write_dataframe(raw, paths.raw_benchmark_csv)
        _write_statuses("benchmark", current, unrelated)
        readiness = _write_benchmark_readiness(raw, captured)
        artifacts = (
            paths.raw_benchmark_csv,
            paths.raw_benchmark_status_csv,
            paths.raw_benchmark_readiness_csv,
        )
        if all(path.is_file() for path in artifacts):
            _write_manifest(
                paths.raw_benchmark_artifact_manifest_csv,
                "benchmark",
                artifacts,
                captured_at_utc=captured,
            )

    run = SerialAcquisitionEngine(
        adapter,
        policy=PRODUCTION_ACQUISITION_POLICY,
        reporter=reporter,
    ).run(
        requests,
        statuses=exact,
        checkpoint=checkpoint,
        refresh=request.refresh,
    )
    if not run.complete or not _all_ready(readiness):
        reporter(
            "Live benchmark remains incomplete; positive Open/Close coverage is "
            "required at every execution and valuation date."
        )
        return 1
    return 0


def execute(command_request: CommandRequest) -> int:
    """Execute one non-composite live dataset through the shared engine."""
    if command_request.dataset == "all":
        raise ValueError(
            "Composite live acquisition is dispatched by live.acquisition_handlers"
        )

    if command_request.dry_run:
        live_plan.dry_run_requests(command_request)
        return 0
    if command_request.dataset == "sectors":
        return execute_sector_acquisition(
            command_request,
            live_plan.live_sector_requirements,
        )

    runtime = AcquisitionRuntimePaths(command_request.project_root)
    dataset = command_request.dataset
    reporter = AcquisitionReporter()
    client = prepare_yfinance(runtime.yfinance_cache)
    with acquisition_lock(runtime.lock_path("live", dataset)):
        if dataset == "prices":
            return _execute_prices(command_request, reporter, client)
        if dataset == "shares":
            return _execute_shares(command_request, reporter, client)
        if dataset == "benchmark":
            return _execute_benchmark(command_request, reporter, client)
    raise ValueError(f"Unsupported live dataset: {dataset}")


__all__ = [
    "execute",
]
