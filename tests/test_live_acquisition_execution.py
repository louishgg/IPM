"""Offline integration tests for flattened live acquisition persistence."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import warnings
from types import SimpleNamespace

import pandas as pd
import pytest

from live import acquisition_execution, acquisition_handlers
from live.price_coverage import (
    PRICE_REQUIREMENT_SET,
    PriceRequirement,
)
from live.paths import LivePaths
import data_acquisition.engine as acquisition_engine
import data_acquisition.sector_acquisition_execution as sector_acquisition_execution
import data_acquisition.sector_acquisition_planning as sector_acquisition_planning
from data_acquisition.sector_acquisition_planning import (
    SECTOR_ACQUISITION_REQUIREMENT_COLUMNS,
)
from data_acquisition.provider_clients import ClientInfo
from data_acquisition.acquire import main
from data_acquisition.contracts import CommandRequest
from data_acquisition.engine import AcquisitionPolicy
from data_acquisition.errors import (
    ConfirmedNoData,
    ProviderRateLimited,
    RetryableProviderError,
)
from data_acquisition.providers.base import (
    AcquisitionResult,
    ProviderAdapter,
)
from data_acquisition.runtime import AcquisitionRuntimePaths
from portfolio_core.artifacts import (
    SCHEMA_VERSION,
    read_manifests,
    validate_manifest_artifact,
)
from data_acquisition.contracts import (
    AcquisitionIdentity,
    AcquisitionRequest,
    AcquisitionStatus,
    ProviderStatus,
    ReadinessRecord,
    ReadinessStatus,
    read_acquisition_statuses,
    read_readiness,
    write_acquisition_statuses,
    write_readiness,
)
from portfolio_core.sector_evidence import (
    NOTICE_COLUMNS,
    SNAPSHOT_COLUMNS,
    SectorHistoryPaths,
    validate_sector_acquisition_manifest,
)
from portfolio_core.sector_resolution import (
    WikipediaIdentityResolver,
)
from portfolio_core.shares import RAW_SHARES_COLUMNS


def _command(
    tmp_path,
    dataset: str,
    *,
    refresh: bool = False,
    dry_run: bool = False,
) -> CommandRequest:
    return CommandRequest(
        scope="live",
        dataset=dataset,
        tickers_file=None,
        refresh=refresh,
        dry_run=dry_run,
        project_root=tmp_path,
    )


def _request(
    dataset: str,
    asset_id: str,
    provider_symbol: str,
) -> AcquisitionRequest:
    return AcquisitionRequest(
        identity=AcquisitionIdentity(
            scope="live",
            dataset=dataset,
            asset_id=asset_id,
            provider="yahoo",
            provider_symbol=provider_symbol,
        ),
        requested_start="2024-01-01",
        requested_end="2024-03-31",
    )


def _live_sector_requirements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "live",
                "2020-01-31",
                "2020-01-31T00:00:00Z",
                "AAA",
                "AAA",
            ),
            (
                "live",
                "2020-02-29",
                "2020-02-29T00:00:00Z",
                "AAA",
                "AAA",
            ),
        ],
        columns=SECTOR_ACQUISITION_REQUIREMENT_COLUMNS,
    )


def _wikipedia_result(
    acquisition_request: AcquisitionRequest,
    revision_id: int,
) -> AcquisitionResult:
    requirement_date = acquisition_request.requested_end
    cutoff = f"{requirement_date}T00:00:00Z"
    revision_timestamp = (
        pd.Timestamp(requirement_date) - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%dT12:00:00Z")
    source_url = (
        "https://en.wikipedia.org/w/index.php?title="
        f"List_of_S%26P_500_companies&oldid={revision_id}"
    )
    rows = pd.DataFrame(
        [
            (
                requirement_date,
                cutoff,
                str(revision_id),
                revision_timestamp,
                "AAA",
                "AAA Company",
                "Industrials",
                source_url,
            )
        ],
        columns=SNAPSHOT_COLUMNS,
    )
    return AcquisitionResult(
        payload=rows,
        observation_count=1,
        observation_start=requirement_date,
        observation_end=requirement_date,
        http_status=200,
    )


@pytest.fixture
def isolated_live_workflow(tmp_path, monkeypatch):
    paths = LivePaths(tmp_path / "live")
    config = SimpleNamespace(
        paths=paths,
        benchmark=SimpleNamespace(ticker="^SP500TR"),
        market=SimpleNamespace(
            competition_start="2026-02-13", competition_end="2026-05-06",
            price_replacement_review_threshold=0.01,
        ),
    )
    monkeypatch.setattr(acquisition_execution, "DEFAULT_CONFIG", config)
    monkeypatch.setattr(
        acquisition_execution,
        "utc_timestamp",
        lambda value=None: "2026-08-26T10:00:00Z",
    )
    monkeypatch.setattr(
        acquisition_engine,
        "utc_timestamp",
        lambda value=None: "2026-08-26T10:00:00Z",
    )
    monkeypatch.setattr(
        acquisition_execution,
        "PRODUCTION_ACQUISITION_POLICY",
        AcquisitionPolicy(
            max_attempts=1,
            item_budget_seconds=90,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            backoff_jitter_seconds=0,
            pacing_min_seconds=0,
            pacing_max_seconds=0,
        ),
    )
    return paths


def _price_payload(dates, base: float) -> pd.DataFrame:
    index = pd.DatetimeIndex(dates, name="Date")
    return pd.DataFrame(
        {
            "Open": [base + value for value in range(len(index))],
            "High": [base + value + 1 for value in range(len(index))],
            "Low": [base + value - 1 for value in range(len(index))],
            "Close": [base + value + 0.5 for value in range(len(index))],
            "Volume": [1_000.0 + value for value in range(len(index))],
        },
        index=index,
    )


def _csv_hashes(directory) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*.csv"))
    }


def _track_checkpoint_persistence(monkeypatch):
    counts = {"statuses": 0, "readiness": 0, "manifests": 0}
    readiness_states = []
    status_states = []
    engine_return_counts = []

    write_statuses = acquisition_execution._write_statuses
    write_readiness_records = acquisition_execution.write_readiness
    write_manifest = acquisition_execution._write_manifest

    def counted_statuses(*args, **kwargs):
        result = write_statuses(*args, **kwargs)
        counts["statuses"] += 1
        status_states.append(tuple(item.status for item in result))
        return result

    def counted_readiness(path, records):
        records = tuple(records)
        counts["readiness"] += 1
        readiness_states.append(tuple(item.status for item in records))
        return write_readiness_records(path, records)

    def counted_manifest(*args, **kwargs):
        counts["manifests"] += 1
        return write_manifest(*args, **kwargs)

    class TrackingEngine(acquisition_engine.SerialAcquisitionEngine):
        def run(self, *args, **kwargs):
            result = super().run(*args, **kwargs)
            engine_return_counts.append(counts.copy())
            return result

    monkeypatch.setattr(acquisition_execution, "_write_statuses", counted_statuses)
    monkeypatch.setattr(acquisition_execution, "write_readiness", counted_readiness)
    monkeypatch.setattr(acquisition_execution, "_write_manifest", counted_manifest)
    monkeypatch.setattr(acquisition_execution, "SerialAcquisitionEngine", TrackingEngine)
    return counts, readiness_states, status_states, engine_return_counts


def _install_price_readiness(
    monkeypatch,
    assets: dict[str, tuple[str, str]],
) -> None:
    """Install a small requirement graph for the in-memory price panels."""

    metadata = pd.DataFrame(
        [
            {"Asset_ID": asset_id, "Yahoo_Ticker": symbol}
            for asset_id, (symbol, _) in assets.items()
        ]
    )
    requirements = tuple(
        PriceRequirement(
            "R1",
            asset_id,
            required_date,
            "signal",
            "Close",
        )
        for asset_id, (_, required_date) in assets.items()
    )
    monkeypatch.setattr(
        acquisition_execution,
        "load_validated_strategy_universe",
        lambda: (pd.DataFrame(), pd.DataFrame(), metadata),
    )
    monkeypatch.setattr(
        acquisition_execution,
        "build_price_requirements",
        lambda schedule, membership, **kwargs: requirements,
    )
    monkeypatch.setattr(acquisition_execution, "evaluation_periods", lambda *args, **kwargs: pd.DataFrame())


def test_live_price_panel_checkpoints_do_not_fragment(tmp_path, monkeypatch):
    dates = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03", "2024-01-04"],
        name="Date",
    )
    panels = {
        name: pd.DataFrame(index=dates, dtype=float)
        for name in ("open", "close", "volume")
    }
    paths = {name: tmp_path / f"{name}.csv" for name in panels}
    checkpoints = {}
    writer = acquisition_execution.atomic_write_dataframe

    def capture(frame, path, **kwargs):
        checkpoints[path] = (frame.copy(), kwargs)

    monkeypatch.setattr(acquisition_execution, "atomic_write_dataframe", capture)

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        for index in range(150):
            symbol = f"T{index:03d}"
            acquisition_execution._merge_price_payload(
                panels,
                symbol,
                _price_payload(dates, 100.0 + index),
            )
            for name, panel in panels.items():
                acquisition_execution._write_panel(panel, paths[name])

    expected_columns = [f"T{index:03d}" for index in range(150)]
    assert all(frame.shape == (3, 150) for frame in panels.values())
    assert all(list(frame.columns) == expected_columns for frame in panels.values())
    # Exercise wide-frame serialization once per panel after all 150 merges.
    assert set(checkpoints) == set(paths.values())
    for path, (frame, kwargs) in checkpoints.items():
        writer(frame, path, **kwargs)
    assert all(
        list(pd.read_csv(path).columns[1:]) == expected_columns
        for path in paths.values()
    )


def test_live_prices_checkpoint_directly_to_one_canonical_state(
    isolated_live_workflow,
    tmp_path,
    monkeypatch,
):
    paths = isolated_live_workflow
    requests = (
        _request("prices", "AAA", "AAA"),
        _request("prices", "BBB", "BBB"),
    )
    monkeypatch.setattr(
        acquisition_execution.live_plan,
        "build_command_requests",
        lambda command: requests,
    )
    _install_price_readiness(
        monkeypatch,
        {"AAA": ("AAA", "2024-01-03"), "BBB": ("BBB", "2024-01-03")},
    )

    # An interrupted checkpoint may have field-specific dates. Resumption must
    # normalize their union before adding the requested identities.
    acquisition_execution._write_panel(
        pd.DataFrame(
            {"OLD": [10.0]},
            index=pd.DatetimeIndex(["2023-12-28"], name="Date"),
        ),
        paths.raw_price_open_csv,
    )
    acquisition_execution._write_panel(
        pd.DataFrame(
            {"OLD": [11.0]},
            index=pd.DatetimeIndex(["2023-12-29"], name="Date"),
        ),
        paths.raw_price_close_csv,
    )
    acquisition_execution._write_panel(
        pd.DataFrame(
            {"OLD": [12.0]},
            index=pd.DatetimeIndex(["2023-12-30"], name="Date"),
        ),
        paths.raw_price_volume_csv,
    )
    payloads = {
        "AAA": _price_payload(["2024-01-01", "2024-01-03"], 100),
        "BBB": _price_payload(["2024-01-02", "2024-01-03"], 200),
    }
    calls = []

    def fetch(acquisition_request):
        calls.append(acquisition_request.identity.asset_id)
        payload = payloads[acquisition_request.identity.asset_id]
        return AcquisitionResult(
            payload=payload,
            observation_count=len(payload),
            observation_start=payload.index.min().strftime("%Y-%m-%d"),
            observation_end=payload.index.max().strftime("%Y-%m-%d"),
            http_status=200,
        )

    monkeypatch.setattr(
        acquisition_execution,
        "make_yahoo_ohlcv_adapter",
        lambda client: ProviderAdapter(
            fetch,
            provider="yahoo",
            dataset="prices",
            client_name="mock",
            client_version="1",
        ),
    )
    counts, _, _, engine_return_counts = _track_checkpoint_persistence(
        monkeypatch
    )
    command = _command(tmp_path, "prices")
    before = counts.copy()
    assert acquisition_execution._execute_prices(command, lambda message: None, ClientInfo("mock", "1")) == 0
    assert calls == ["AAA", "BBB"]
    assert {
        name: counts[name] - before[name]
        for name in counts
    } == {"statuses": 3, "readiness": 3, "manifests": 3}
    assert engine_return_counts[-1] == counts

    panels = {
        name: acquisition_execution._read_panel(path)
        for name, path in (
            ("open", paths.raw_price_open_csv),
            ("close", paths.raw_price_close_csv),
            ("volume", paths.raw_price_volume_csv),
        )
    }
    expected_index = pd.DatetimeIndex(
        [
            "2023-12-28",
            "2023-12-29",
            "2023-12-30",
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
        ],
        name="Date",
    )
    assert all(frame.index.equals(expected_index) for frame in panels.values())
    assert all(list(frame.columns) == ["AAA", "BBB", "OLD"] for frame in panels.values())
    assert {
        item.status for item in read_acquisition_statuses(paths.raw_price_status_csv)
    } == {ProviderStatus.OK}
    assert {
        item.status for item in read_readiness(paths.raw_price_readiness_csv)
    } == {ReadinessStatus.COMPLETE}

    manifests = read_manifests(paths.raw_price_artifact_manifest_csv)
    assert {item.schema_version for item in manifests} == {SCHEMA_VERSION}
    assert {item.artifact_path for item in manifests} == {
        "open.csv",
        "close.csv",
        "volume.csv",
        "acquisition_status.csv",
        "readiness.csv",
        "requirements.csv",
    }
    for manifest in manifests:
        validate_manifest_artifact(manifest, base_dir=paths.raw_prices_dir)

    first_hashes = _csv_hashes(paths.raw_prices_dir)
    calls.clear()
    before = counts.copy()
    assert acquisition_execution._execute_prices(command, lambda message: None, ClientInfo("mock", "1")) == 0
    assert calls == []
    assert {
        name: counts[name] - before[name]
        for name in counts
    } == {"statuses": 1, "readiness": 1, "manifests": 1}
    assert engine_return_counts[-1] == counts
    assert _csv_hashes(paths.raw_prices_dir) == first_hashes
    assert len(list((paths.raw_prices_dir / "replacement_responses").glob("*.csv"))) == 2
    for name, field, old_date, old_value in (
        ("open", "Open", "2023-12-28", 10.0),
        ("close", "Close", "2023-12-29", 11.0),
        ("volume", "Volume", "2023-12-30", 12.0),
    ):
        expected = pd.DataFrame(float("nan"), index=expected_index, columns=["AAA", "BBB", "OLD"])
        expected.loc[pd.Timestamp(old_date), "OLD"] = old_value
        for symbol, payload in payloads.items():
            expected.loc[payload.index, symbol] = payload[field].to_numpy()
        pd.testing.assert_frame_equal(panels[name], expected)
    assert read_readiness(paths.raw_price_readiness_csv) == [ReadinessRecord(
        scope="live", dataset="prices", asset_id=asset_id, requirement_set=PRICE_REQUIREMENT_SET,
        required_count=1, covered_count=1, status=ReadinessStatus.COMPLETE, missing_dates=(),
        contributing_sources=("yahoo",), checked_at_utc="2026-08-26T10:00:00Z",
    ) for asset_id in ("AAA", "BBB")]
    expected_requirements = pd.DataFrame([{
        "Requirement_Set": PRICE_REQUIREMENT_SET, "Rebalance_ID": "R1", "Asset_ID": asset_id,
        "Requirement_Date": "2024-01-03", "Role": "signal", "Field": "Close",
        "Holding_Start_Date": "", "Coverage_Status": "covered", "Coverage_Source": "yahoo",
        "Coverage_Reference": "",
    } for asset_id in ("AAA", "BBB")])
    pd.testing.assert_frame_equal(
        pd.read_csv(paths.raw_price_requirements_csv, keep_default_na=False), expected_requirements,
    )


def test_price_refresh_replaces_success_and_retains_last_good_on_failure(
    isolated_live_workflow,
    tmp_path,
    monkeypatch,
):
    paths = isolated_live_workflow
    request = _request("prices", "AAA", "AAA")
    monkeypatch.setattr(
        acquisition_execution.live_plan,
        "build_command_requests",
        lambda command: (request,),
    )
    _install_price_readiness(
        monkeypatch,
        {"AAA": ("AAA", "2024-01-03")},
    )
    state = {"base": 100.0, "fail": False, "calls": 0}

    def fetch(acquisition_request):
        state["calls"] += 1
        if state["fail"]:
            raise RetryableProviderError("temporary Yahoo failure")
        payload = _price_payload(["2024-01-03"], state["base"])
        return AcquisitionResult(
            payload=payload,
            observation_count=1,
            observation_start="2024-01-03",
            observation_end="2024-01-03",
            http_status=200,
        )

    monkeypatch.setattr(
        acquisition_execution,
        "make_yahoo_ohlcv_adapter",
        lambda client: ProviderAdapter(
            fetch,
            provider="yahoo",
            dataset="prices",
            client_name="mock",
            client_version="1",
        ),
    )
    assert acquisition_execution._execute_prices(
        _command(tmp_path, "prices"), lambda message: None, ClientInfo("mock", "1")
    ) == 0
    assert acquisition_execution._read_panel(paths.raw_price_open_csv).at[
        pd.Timestamp("2024-01-03"), "AAA"
    ] == 100.0

    state["base"] = 100.05
    assert acquisition_execution._execute_prices(
        _command(tmp_path, "prices", refresh=True),
        lambda message: None,
        ClientInfo("mock", "1"),
    ) == 0
    refreshed = acquisition_execution._read_panel(paths.raw_price_open_csv)
    assert refreshed.at[pd.Timestamp("2024-01-03"), "AAA"] == 100.05
    assert list(refreshed.columns) == ["AAA"]

    state["fail"] = True
    assert acquisition_execution._execute_prices(
        _command(tmp_path, "prices", refresh=True),
        lambda message: None,
        ClientInfo("mock", "1"),
    ) == 1
    retained = acquisition_execution._read_panel(paths.raw_price_open_csv)
    assert retained.at[pd.Timestamp("2024-01-03"), "AAA"] == 100.05
    status = read_acquisition_statuses(paths.raw_price_status_csv)[0]
    assert status.status is ProviderStatus.FAILED
    assert read_readiness(paths.raw_price_readiness_csv)[0].status is (
        ReadinessStatus.COMPLETE
    )

    calls_before = state["calls"]
    assert acquisition_execution._execute_prices(
        _command(tmp_path, "prices"), lambda message: None, ClientInfo("mock", "1")
    ) == 1
    assert state["calls"] == calls_before + 1
    assert read_acquisition_statuses(paths.raw_price_status_csv)[0].status is (
        ProviderStatus.FAILED
    )


def test_live_shares_do_not_backfill_future_observations(
    isolated_live_workflow,
    tmp_path,
    monkeypatch,
):
    paths = isolated_live_workflow
    initial = acquisition_execution._load_raw_shares()
    assert initial.empty
    assert tuple(initial.columns) == RAW_SHARES_COLUMNS
    assert not paths.shares.raw_shares_csv.exists()

    request = _request("shares", "AAA", "AAA")
    monkeypatch.setattr(
        acquisition_execution.live_plan,
        "build_command_requests",
        lambda command: (request,),
    )
    monkeypatch.setattr(
        acquisition_execution.live_plan,
        "shares_requirement_dates",
        lambda: {"AAA": ("2024-01-31", "2024-02-29")},
    )
    payload = pd.Series(
        [100.0], index=pd.DatetimeIndex(["2024-02-15"], name="Date")
    )
    monkeypatch.setattr(
        acquisition_execution,
        "make_yahoo_shares_adapter",
        lambda client: ProviderAdapter(
            lambda acquisition_request: AcquisitionResult(
                payload=payload,
                observation_count=1,
                observation_start="2024-02-15",
                observation_end="2024-02-15",
                http_status=200,
            ),
            provider="yahoo",
            dataset="shares",
            client_name="mock",
            client_version="1",
        ),
    )

    assert acquisition_execution._execute_shares(
        _command(tmp_path, "shares"), lambda message: None, ClientInfo("mock", "1")
    ) == 1
    readiness = read_readiness(paths.shares.readiness_csv)[0]
    assert readiness.status is ReadinessStatus.PARTIAL
    assert readiness.covered_count == 1
    assert readiness.missing_dates == ("2024-01-31",)


def test_live_shares_readiness_respects_inclusive_identity_end(
    isolated_live_workflow,
    monkeypatch,
):
    paths = isolated_live_workflow
    paths.shares.raw_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        acquisition_execution.live_plan,
        "shares_requirement_dates",
        lambda: {"AAA": ("2026-01-31", "2026-02-01")},
    )
    raw = pd.DataFrame(
        [["AAA", "AAA", "", "2026-01-31", "2026-01-15", 0, 100.0]],
        columns=RAW_SHARES_COLUMNS,
    )

    records = acquisition_execution._write_shares_readiness(
        raw,
        "2026-07-30T00:00:00Z",
    )

    assert len(records) == 1
    assert records[0].status is ReadinessStatus.PARTIAL
    assert records[0].covered_count == 1
    assert records[0].missing_dates == ("2026-02-01",)
    assert read_readiness(paths.shares.readiness_csv) == records


@pytest.mark.parametrize("dataset,existing_empty", [
    ("prices", False), ("shares", False), ("benchmark", False), ("benchmark", True),
], ids=["prices", "shares", "benchmark-absent", "benchmark-empty"])
def test_live_execution_persists_exactly_once_per_engine_checkpoint(
    dataset,
    existing_empty,
    isolated_live_workflow,
    tmp_path,
    monkeypatch,
):
    paths = isolated_live_workflow
    asset_id = "^SP500TR" if dataset == "benchmark" else "AAA"
    request = _request(dataset, asset_id, asset_id)
    monkeypatch.setattr(
        acquisition_execution.live_plan,
        "build_command_requests",
        lambda command: (request,),
    )
    state = {"calls": 0, "error": None}

    if dataset == "prices":
        _install_price_readiness(
            monkeypatch,
            {"AAA": ("AAA", "2024-01-03")},
        )
        payload = _price_payload(["2024-01-03"], 100.0)
        result = AcquisitionResult(
            payload=payload,
            observation_count=1,
            observation_start="2024-01-03",
            observation_end="2024-01-03",
            http_status=200,
        )
        adapter_factory = "make_yahoo_ohlcv_adapter"
        execute_dataset = acquisition_execution._execute_prices
        payload_paths = (
            paths.raw_price_open_csv,
            paths.raw_price_close_csv,
            paths.raw_price_volume_csv,
        )
        status_path = paths.raw_price_status_csv
        readiness_path = paths.raw_price_readiness_csv
    elif dataset == "shares":
        monkeypatch.setattr(
            acquisition_execution.live_plan,
            "shares_requirement_dates",
            lambda: {"AAA": ("2024-01-31",)},
        )
        payload = pd.Series(
            [100.0], index=pd.DatetimeIndex(["2024-01-10"], name="Date")
        )
        result = AcquisitionResult(
            payload=payload,
            observation_count=1,
            observation_start="2024-01-10",
            observation_end="2024-01-10",
            http_status=200,
        )
        adapter_factory = "make_yahoo_shares_adapter"
        execute_dataset = acquisition_execution._execute_shares
        payload_paths = (paths.shares.raw_shares_csv,)
        status_path = paths.shares.acquisition_status_csv
        readiness_path = paths.shares.readiness_csv
    else:
        monkeypatch.setattr(
            acquisition_execution.live_plan,
            "benchmark_requirement_dates",
            lambda: ("2024-01-31", "2024-02-29"),
        )
        payload = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.5, 101.5],
            },
            index=pd.DatetimeIndex(["2024-01-31", "2024-02-29"], name="Date"),
        )
        result = AcquisitionResult(
            payload=payload,
            observation_count=2,
            observation_start="2024-01-31",
            observation_end="2024-02-29",
            http_status=200,
        )
        adapter_factory = "make_yahoo_benchmark_adapter"
        execute_dataset = acquisition_execution._execute_benchmark
        payload_paths = (paths.raw_benchmark_csv,)
        status_path = paths.raw_benchmark_status_csv
        readiness_path = paths.raw_benchmark_readiness_csv
        if existing_empty:
            paths.raw_benchmark_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=acquisition_execution.RAW_BENCHMARK_COLUMNS).to_csv(
                paths.raw_benchmark_csv, index=False,
            )

    def fetch(acquisition_request):
        state["calls"] += 1
        if state["error"] is not None:
            raise state["error"]
        return result

    monkeypatch.setattr(
        acquisition_execution,
        adapter_factory,
        lambda client: ProviderAdapter(
            fetch,
            provider="yahoo",
            dataset=dataset,
            client_name="mock",
            client_version="1",
        ),
    )
    (
        counts,
        readiness_states,
        status_states,
        engine_return_counts,
    ) = _track_checkpoint_persistence(monkeypatch)
    command = _command(tmp_path, dataset)

    before = counts.copy()
    readiness_start = len(readiness_states)
    status_start = len(status_states)
    assert execute_dataset(
        command,
        lambda message: None,
        ClientInfo("mock", "1"),
    ) == 0
    assert {
        name: counts[name] - before[name]
        for name in counts
    } == {"statuses": 2, "readiness": 2,
          "manifests": 1 if dataset == "benchmark" and not existing_empty else 2}
    assert engine_return_counts[-1] == counts
    assert readiness_states[readiness_start:] == [
        (ReadinessStatus.MISSING,),
        (ReadinessStatus.COMPLETE,),
    ]
    assert status_states[status_start:] == [
        (ProviderStatus.PENDING,),
        (ProviderStatus.OK,),
    ]
    assert state["calls"] == 1
    if dataset == "shares":
        expected_payloads = [pd.DataFrame([
            ["AAA", "AAA", "", "", "2024-01-10", 0, 100.0],
        ], columns=RAW_SHARES_COLUMNS)]
    elif dataset == "benchmark":
        expected_payloads = [payload.reset_index().assign(Date=payload.index.strftime("%Y-%m-%d"))]
    else:
        expected_payloads = [pd.DataFrame({"Date": payload.index.strftime("%Y-%m-%d"),
                                          "AAA": payload[field].to_numpy()})
                             for field in ("Open", "Close", "Volume")]
    for path, expected in zip(payload_paths, expected_payloads, strict=True):
        pd.testing.assert_frame_equal(pd.read_csv(path, keep_default_na=False), expected, check_dtype=False)
    retained_payload = tuple(path.read_bytes() for path in payload_paths)

    requirement_set = {"prices": PRICE_REQUIREMENT_SET, "shares": "active_execution_valuation_shares",
                       "benchmark": "execution_valuation_benchmark"}[dataset]
    expected_readiness = ReadinessRecord(
        scope="live", dataset=dataset, asset_id=asset_id, requirement_set=requirement_set,
        required_count=2 if dataset == "benchmark" else 1,
        covered_count=2 if dataset == "benchmark" else 1, status=ReadinessStatus.COMPLETE,
        missing_dates=(), contributing_sources=("yahoo",), checked_at_utc="2026-08-26T10:00:00Z",
    )

    price_comparison = {}

    def assert_durable_state(status, error=None):
        successful = status is ProviderStatus.OK
        statuses = read_acquisition_statuses(status_path)
        assert len(statuses) == 1
        actual = statuses[0]
        if dataset == "prices":
            note = json.loads(actual.migration_note)
            evidence = status_path.parent / note["provider_response"]
            digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
            pd.testing.assert_frame_equal(
                pd.read_csv(evidence, index_col="Date", parse_dates=True), payload,
                check_dtype=False,
            )
            assert note == {
                "price_replacement": "accepted_provider_values", "review_threshold": 0.01,
                "comparison": price_comparison, "missing_cached_dates": {}, "review_reasons": [],
                "provider_response_sha256": digest, "provider_response": f"replacement_responses/{digest}.csv",
            }
            actual = replace(actual, migration_note="")
        assert actual == AcquisitionStatus(
            identity=request.identity, status=status,
            requested_start=request.requested_start, requested_end=request.requested_end,
            observation_count=result.observation_count if successful else 0,
            observation_start=result.observation_start if successful else "",
            observation_end=result.observation_end if successful else "",
            attempted_at_utc="2026-08-26T10:00:00Z", client="mock", client_version="1",
            http_status="200" if successful else str(error.http_status or ""),
            error_class="" if successful else type(error).__name__,
            error_message="" if successful else str(error),
        )
        assert read_readiness(readiness_path) == [expected_readiness]
        manifest_path = status_path.parent / "artifact_manifest.csv"
        manifests = read_manifests(manifest_path)
        expected_files = {path.name for path in (*payload_paths, status_path, readiness_path)}
        if dataset == "prices":
            expected_files.add("requirements.csv")
        assert {item.artifact_path for item in manifests} == expected_files
        for manifest in manifests:
            assert (manifest.schema_version, manifest.scope, manifest.dataset,
                    manifest.origin.value, manifest.captured_at_utc) == (
                SCHEMA_VERSION, "live", dataset, "downloaded", "2026-08-26T10:00:00Z",
            )
            validate_manifest_artifact(manifest, base_dir=status_path.parent)

    assert_durable_state(ProviderStatus.OK)

    before = counts.copy()
    assert execute_dataset(
        command,
        lambda message: None,
        ClientInfo("mock", "1"),
    ) == 0
    assert {
        name: counts[name] - before[name]
        for name in counts
    } == {"statuses": 1, "readiness": 1, "manifests": 1}
    assert engine_return_counts[-1] == counts
    assert state["calls"] == 1
    assert read_acquisition_statuses(status_path)[0].status is ProviderStatus.OK
    assert read_readiness(readiness_path)[0].status is ReadinessStatus.COMPLETE

    refresh_cases = (
        # Preserve the direct OK -> FAILED edge before testing NO_DATA -> FAILED.
        (RetryableProviderError("temporary failure"), ProviderStatus.FAILED, 1),
        (None, ProviderStatus.OK, 0),
        (ConfirmedNoData("confirmed absent", http_status=404), ProviderStatus.NO_DATA, 0),
        (RetryableProviderError("temporary failure"), ProviderStatus.FAILED, 1),
        (ProviderRateLimited("rate limited"), ProviderStatus.FAILED, 1),
    )
    for error, expected_status, expected_exit in refresh_cases:
        state["error"] = error
        before = counts.copy()
        assert execute_dataset(
            _command(tmp_path, dataset, refresh=True),
            lambda message: None,
            ClientInfo("mock", "1"),
        ) == expected_exit
        assert {
            name: counts[name] - before[name]
            for name in counts
        } == {"statuses": 2, "readiness": 2, "manifests": 2}
        assert engine_return_counts[-1] == counts
        if error is None and dataset == "prices":
            price_comparison = {field: {"overlap_rows": 1, "maximum_relative_change": 0.0}
                                for field in ("Open", "Close", "Volume")}
        assert_durable_state(expected_status, error)
        assert tuple(path.read_bytes() for path in payload_paths) == retained_payload


def test_live_sectors_resume_failed_date_and_preserve_pinned_revisions(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        acquisition_execution.live_plan,
        "build_command_requests",
        lambda *args, **kwargs: pytest.fail(
            "sector command used Yahoo command-request planning"
        ),
    )
    paths = SectorHistoryPaths.from_project_root(tmp_path)
    paths.directory.mkdir(parents=True)
    pd.DataFrame(
        [
            (
                "NOTICE-UNRELATED",
                "2019-01-01",
                "2019-01-02",
                "S&P 500",
                "Addition",
                "OTHER",
                "Energy",
                "https://press.spglobal.com/example",
                "approved",
            )
        ],
        columns=NOTICE_COLUMNS,
    ).to_csv(paths.notices_csv, index=False)
    monkeypatch.setattr(
        acquisition_execution.live_plan,
        "live_sector_requirements",
        lambda config=None: _live_sector_requirements(),
    )
    monkeypatch.setattr(
        sector_acquisition_execution,
        "WIKIPEDIA_ACQUISITION_POLICY",
        AcquisitionPolicy(
            max_attempts=1,
            item_budget_seconds=90,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            backoff_jitter_seconds=0,
            pacing_min_seconds=0,
            pacing_max_seconds=0,
        ),
    )
    monkeypatch.setattr(
        sector_acquisition_execution,
        "load_security_identity_bundle",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        sector_acquisition_planning,
        "load_security_identity_bundle",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        sector_acquisition_execution,
        "WikipediaIdentityResolver",
        lambda bundle: WikipediaIdentityResolver(),
    )
    state = {"fail_february": True, "missing_pinned_dates": set()}
    attempts = []
    observed_pins = []

    def make_adapter(client, *, pinned_revisions):
        observed_pins.append(dict(pinned_revisions))

        def fetch(acquisition_request):
            date = acquisition_request.requested_end
            attempts.append(date)
            if date in state["missing_pinned_dates"]:
                raise ConfirmedNoData(
                    "pinned Wikipedia revision is missing",
                    http_status=404,
                )
            if date == "2020-02-29" and state["fail_february"]:
                raise RetryableProviderError("temporary Wikipedia failure")
            revision_id = 101 if date == "2020-01-31" else 202
            return _wikipedia_result(acquisition_request, revision_id)

        return ProviderAdapter(
            fetch,
            provider="wikipedia",
            dataset="sectors",
            client_name=client.name,
            client_version=client.version,
        )

    monkeypatch.setattr(
        sector_acquisition_execution,
        "make_wikipedia_sectors_adapter",
        make_adapter,
    )
    publication_calls = []
    publication_readiness = []
    publish_state = sector_acquisition_execution.publish_sector_state
    build_readiness = (
        sector_acquisition_execution.build_sector_acquisition_readiness
    )

    def counted_publish_state(**kwargs):
        publication_calls.append(tuple(kwargs["snapshots"]["Requirement_Date"]))
        return publish_state(**kwargs)

    def counted_build_readiness(*args, **kwargs):
        readiness = build_readiness(*args, **kwargs)
        publication_readiness.append(readiness)
        return readiness

    monkeypatch.setattr(
        sector_acquisition_execution,
        "publish_sector_state",
        counted_publish_state,
    )
    monkeypatch.setattr(
        sector_acquisition_execution,
        "build_sector_acquisition_readiness",
        counted_build_readiness,
    )
    command = _command(tmp_path, "sectors")

    assert acquisition_execution.execute(command) == 1
    assert len(publication_calls) == 1
    first_statuses = read_acquisition_statuses(paths.acquisition_status_csv)
    assert {
        status.requested_end: status.status for status in first_statuses
    } == {
        "2020-01-31": ProviderStatus.OK,
        "2020-02-29": ProviderStatus.FAILED,
    }
    first_readiness = publication_readiness[0]
    assert len(first_readiness) == 1
    assert first_readiness[0].status is ReadinessStatus.PARTIAL
    assert first_readiness[0].covered_count == 1
    assert first_readiness[0].missing_dates == ("2020-02-29",)
    assert attempts == ["2020-01-31", "2020-02-29"]
    assert observed_pins == [{}]
    validate_sector_acquisition_manifest(paths, repository_root=tmp_path)
    assert len(read_manifests(paths.artifact_manifest_csv)) == 3
    runtime = AcquisitionRuntimePaths(tmp_path)
    assert not runtime.yfinance_cache.exists()
    assert not any(runtime.sector_history_snapshots.iterdir())
    assert not any(runtime.sector_history_statuses.iterdir())
    assert not any(runtime.sector_history_publications.iterdir())

    state["fail_february"] = False
    assert acquisition_execution.execute(command) == 0
    assert len(publication_calls) == 2
    assert attempts == ["2020-01-31", "2020-02-29", "2020-02-29"]
    assert observed_pins[-1] == {"2020-01-31": 101}
    final_statuses = read_acquisition_statuses(paths.acquisition_status_csv)
    assert {status.status for status in final_statuses} == {ProviderStatus.OK}
    final_readiness = publication_readiness[1]
    assert final_readiness[0].status is ReadinessStatus.COMPLETE
    assert final_readiness[0].required_count == 2
    assert final_readiness[0].covered_count == 2
    snapshots = pd.read_csv(paths.snapshots_csv, keep_default_na=False)
    assert snapshots.set_index("Requirement_Date")["Revision_ID"].to_dict() == {
        "2020-01-31": 101,
        "2020-02-29": 202,
    }
    validate_sector_acquisition_manifest(paths, repository_root=tmp_path)

    assert acquisition_execution.execute(
        _command(tmp_path, "sectors", dry_run=True)
    ) == 0
    assert "0 Wikipedia revision request(s), 2 terminal checkpoint(s)" in (
        capsys.readouterr().out
    )
    assert acquisition_execution.execute(
        _command(tmp_path, "sectors", refresh=True, dry_run=True)
    ) == 0
    output = capsys.readouterr().out
    assert "verify pinned revision 101" in output
    assert "verify pinned revision 202" in output

    # Reviewed manual-ledger edits are intentionally allowed even though their
    # old manifest hashes no longer match. The next successful checkpoint
    # records the reviewed replacement bytes.
    notices = pd.read_csv(paths.notices_csv, keep_default_na=False, dtype=str)
    notices.loc[len(notices)] = (
        "NOTICE-REVIEWED-CHANGE",
        "2019-02-01",
        "2019-02-02",
        "S&P 500",
        "Addition",
        "EXTRA",
        "Industrials",
        "https://press.spglobal.com/reviewed-change",
        "approved",
    )
    notices.to_csv(paths.notices_csv, index=False)
    attempts_before_manual_checkpoint = list(attempts)
    assert acquisition_execution.execute(command) == 0
    assert attempts == attempts_before_manual_checkpoint
    validate_sector_acquisition_manifest(paths, repository_root=tmp_path)

    # A pinned revision returning HTTP 404 is not a successful verification.
    # The old snapshot still provides complete data readiness, but the command
    # must fail until every requested pin has an OK provider status.
    state["missing_pinned_dates"] = {"2020-02-29"}
    assert acquisition_execution.execute(
        _command(tmp_path, "sectors", refresh=True)
    ) == 1
    refreshed_statuses = {
        status.requested_end: status.status
        for status in read_acquisition_statuses(paths.acquisition_status_csv)
    }
    assert refreshed_statuses == {
        "2020-01-31": ProviderStatus.OK,
        "2020-02-29": ProviderStatus.NO_DATA,
    }
    refreshed_readiness = publication_readiness[-1]
    assert refreshed_readiness[0].status is ReadinessStatus.COMPLETE
    assert observed_pins[-1] == {
        "2020-01-31": 101,
        "2020-02-29": 202,
    }

    # Schema-valid downloaded tampering cannot be adopted by either a dry run
    # or a mutating run and silently re-signed into a new manifest.
    attempts_before_tamper = list(attempts)
    mutations = (
        (paths.snapshots_csv, "Company_Name", "Tampered Company"),
        (paths.acquisition_status_csv, "Client", "tampered-client"),
    )
    for artifact, column, value in mutations:
        original = artifact.read_bytes()
        tampered = pd.read_csv(artifact, keep_default_na=False, dtype=str)
        tampered.loc[0, column] = value
        tampered.to_csv(artifact, index=False)
        with pytest.raises(ValueError, match="manifest hash mismatch"):
            acquisition_execution.execute(
                _command(tmp_path, "sectors", dry_run=True)
            )
        with pytest.raises(ValueError, match="manifest hash mismatch"):
            acquisition_execution.execute(command)
        artifact.write_bytes(original)
    assert attempts == attempts_before_tamper


def test_root_handler_mapping_dispatches_live_to_central_workflow(monkeypatch):
    captured = []
    monkeypatch.setattr(
        acquisition_handlers,
        "execute",
        lambda request: captured.append(request) or 9,
    )
    assert main(["live", "prices", "--dry-run"]) == 9
    assert len(captured) == 1
    assert captured[0].scope == "live"
    assert captured[0].dataset == "prices"
    assert captured[0].dry_run is True


def test_live_dry_run_and_execution_share_exact_command_request_tuple(
    tmp_path,
    monkeypatch,
):
    paths = LivePaths(tmp_path / "live")
    config = SimpleNamespace(paths=paths)
    expected = (
        _request("shares", "AAA", "AAA"),
        _request("shares", "BBB", "BBB"),
    )
    builder_commands = []

    def build_command_requests(command_request, **_kwargs):
        builder_commands.append(command_request)
        return expected

    monkeypatch.setattr(
        acquisition_execution.live_plan,
        "build_command_requests",
        build_command_requests,
    )
    execution_inputs = []

    class RequestsObserved(Exception):
        pass

    def capture_execution_requests(dataset, requests):
        assert dataset == "shares"
        execution_inputs.append(tuple(requests))
        raise RequestsObserved

    monkeypatch.setattr(
        acquisition_execution,
        "_statuses_for_requests",
        capture_execution_requests,
    )
    execution_command = _command(tmp_path, "shares")
    dry_run_command = replace(execution_command, dry_run=True)

    dry_run_requests = acquisition_execution.live_plan.dry_run_requests(
        dry_run_command,
        config=config,
    )
    with pytest.raises(RequestsObserved):
        acquisition_execution._execute_shares(
            execution_command,
            lambda message: None,
            ClientInfo("mock", "1"),
        )

    assert dry_run_requests == expected
    assert execution_inputs == [dry_run_requests]
    assert builder_commands == [dry_run_command, execution_command]


def test_live_dry_run_lists_pending_before_previous_failures(
    tmp_path,
    monkeypatch,
    capsys,
):
    paths = LivePaths(tmp_path / "live")
    config = SimpleNamespace(paths=paths)
    failed_requests = (
        _request("prices", "AAA", "AAA"),
        _request("prices", "AAB", "AAB"),
    )
    untouched = _request("prices", "BBB", "BBB")
    failures = tuple(
        AcquisitionStatus(
            identity=request.identity,
            status=ProviderStatus.FAILED,
            requested_start=request.requested_start,
            requested_end=request.requested_end,
            attempted_at_utc="2026-01-01T00:00:00Z",
            client="mock",
            client_version="1",
            error_class="RetryableProviderError",
            error_message="historical symbol unavailable",
        )
        for request in failed_requests
    )
    write_acquisition_statuses(paths.raw_price_status_csv, failures)
    monkeypatch.setattr(
        acquisition_execution.live_plan,
        "build_command_requests",
        lambda *args, **kwargs: (*failed_requests, untouched),
    )

    planned = acquisition_execution.live_plan.dry_run_requests(
        _command(tmp_path, "prices", dry_run=True), config=config
    )
    assert [item.identity.asset_id for item in planned] == ["BBB", "AAA", "AAB"]
    output = capsys.readouterr().out
    assert output.index("  BBB ->") < output.index("  AAA ->")
    assert output.index("  AAA ->") < output.index("  AAB ->")


def test_dry_run_retries_failures_despite_competition_coverage_and_fetches_changed_identity(
    tmp_path,
    monkeypatch,
):
    paths = LivePaths(tmp_path / "live")
    config = SimpleNamespace(paths=paths)
    retired = _request("prices", "AAA", "OLD")
    successor = _request("prices", "AAA", "NEW")
    write_acquisition_statuses(
        paths.raw_price_status_csv,
        (
            AcquisitionStatus(
                identity=retired.identity,
                status=ProviderStatus.FAILED,
                requested_start=retired.requested_start,
                requested_end=retired.requested_end,
                attempted_at_utc="2026-01-01T00:00:00Z",
                client="mock",
                client_version="1",
                error_class="RetryableProviderError",
                error_message="retired symbol",
            ),
        ),
    )
    write_readiness(
        paths.raw_price_readiness_csv,
        (
            ReadinessRecord(
                scope="live",
                dataset="prices",
                asset_id="AAA",
                requirement_set=PRICE_REQUIREMENT_SET,
                required_count=1,
                covered_count=1,
                status=ReadinessStatus.COMPLETE,
                contributing_sources=("yahoo_supplement",),
                checked_at_utc="2026-07-30T00:00:00Z",
            ),
        ),
    )

    monkeypatch.setattr(
        acquisition_execution.live_plan,
        "build_command_requests",
        lambda *args, **kwargs: (retired,),
    )
    normal = acquisition_execution.live_plan.dry_run_requests(
        _command(tmp_path, "prices", dry_run=True), config=config
    )
    refreshed = acquisition_execution.live_plan.dry_run_requests(
        _command(tmp_path, "prices", dry_run=True, refresh=True), config=config
    )
    assert normal == (retired,)
    assert refreshed == (retired,)

    monkeypatch.setattr(
        acquisition_execution.live_plan,
        "build_command_requests",
        lambda *args, **kwargs: (successor,),
    )
    changed = acquisition_execution.live_plan.dry_run_requests(
        _command(tmp_path, "prices", dry_run=True), config=config
    )
    assert changed == (successor,)


def test_live_all_stops_after_first_failed_dataset(tmp_path, monkeypatch):
    calls = []

    def execute_one(request):
        calls.append(request.dataset)
        if request.dataset == "prices":
            return 1
        pytest.fail("all continued after prices failure")

    monkeypatch.setattr(acquisition_handlers, "execute", execute_one)
    assert acquisition_handlers.acquire_live_all(_command(tmp_path, "all")) == 1
    assert calls == ["prices"]


@pytest.mark.parametrize("dry_run", (False, True), ids=("execute", "dry-run"))
def test_live_all_scopes_ticker_selection_to_shares(
    tmp_path,
    monkeypatch,
    dry_run,
):
    ticker_file = tmp_path / "tickers.csv"
    pd.DataFrame({"Asset_ID": ["AAA"]}).to_csv(ticker_file, index=False)
    calls = []

    def execute_one(request):
        calls.append((request.dataset, request.tickers_file, request.dry_run))
        return 0

    monkeypatch.setattr(acquisition_handlers, "execute", execute_one)
    request = replace(
        _command(tmp_path, "all", dry_run=dry_run),
        tickers_file=ticker_file,
    )

    assert acquisition_handlers.acquire_live_all(request) == 0
    assert calls == [
        ("prices", None, dry_run),
        ("benchmark", None, dry_run),
        ("sectors", None, dry_run),
        ("shares", ticker_file, dry_run),
    ]
