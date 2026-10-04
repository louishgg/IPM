"""Focused offline tests for the centralized acquisition package."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import data_acquisition.acquire as acquisition_cli
import data_acquisition.providers.wikipedia as wikipedia_provider
from data_acquisition.provider_clients import (
    ClientInfo,
    REQUIRED_CURL_CFFI_VERSION,
    REQUIRED_YFINANCE_VERSION,
    prepare_yfinance,
)
from data_acquisition.acquire import (
    DATASETS,
    SCOPES,
    main,
)
from data_acquisition.engine import (
    AcquisitionPolicy,
    SerialAcquisitionEngine,
    build_acquisition_plan,
)
from data_acquisition.errors import (
    ConfirmedNoData,
    ProviderRateLimited,
    RetryableProviderError,
)
from data_acquisition.providers.base import (
    AcquisitionResult,
    ProviderAdapter,
)
from data_acquisition.providers.yahoo import (
    fetch_yahoo_benchmark_once,
    fetch_yahoo_ohlcv_once,
    fetch_yahoo_shares_once,
    make_yahoo_benchmark_adapter,
    make_yahoo_ohlcv_adapter,
    make_yahoo_shares_adapter,
)
from data_acquisition.providers.wikipedia import make_wikipedia_sectors_adapter
from data_acquisition.runtime import (
    AcquisitionLockedError,
    AcquisitionReporter,
    AcquisitionRuntimePaths,
    acquisition_lock,
)
from portfolio_core.io import atomic_write_dataframe
from portfolio_core.artifacts import (
    ArtifactManifest,
    ArtifactOrigin,
    read_manifests,
    validate_exact_manifest_catalog,
    validate_manifest_artifact,
    write_manifests,
)
from data_acquisition.contracts import (
    ACQUISITION_STATUS_COLUMNS,
    AcquisitionStatus,
    ProviderStatus,
    ReadinessRecord,
    ReadinessStatus,
    read_acquisition_statuses,
    read_asset_selection,
    read_readiness,
    utc_timestamp,
    write_acquisition_statuses,
    write_readiness,
)
from _acquisition_test_helpers import (
    backtest_acquisition_request as _request,
)


def test_asset_selection_reader_normalizes_both_supported_headers(tmp_path):
    asset_ids = tmp_path / "asset_ids.csv"
    asset_ids.write_text("Asset_ID\n aaa \nBBB\nAAA\n", encoding="utf-8")
    tickers = tmp_path / "tickers.csv"
    tickers.write_text("Ticker\nbrk-b\n", encoding="utf-8")

    assert read_asset_selection(asset_ids) == ("AAA", "BBB")
    assert read_asset_selection(tickers) == ("BRK-B",)


@pytest.mark.parametrize(
    "contents",
    ["Wrong\nAAA\n", "Asset_ID,Ticker\nAAA,AAA\n", "Asset_ID\n"],
)
def test_asset_selection_reader_rejects_invalid_contract(tmp_path, contents):
    path = tmp_path / "selection.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="Ticker request CSV"):
        read_asset_selection(path)


FIXED_TIME = datetime(2026, 7, 25, 20, 0, tzinfo=timezone.utc)


def _result(payload="payload") -> AcquisitionResult:
    return AcquisitionResult(
        payload=payload,
        observation_count=2,
        observation_start="2024-01-02",
        observation_end="2024-01-03",
        http_status=200,
    )


def test_provider_adapter_uses_explicit_provider_and_dataset():
    adapter = ProviderAdapter(
        lambda request: _result(request.identity.asset_id),
        provider="yahoo",
        dataset="shares",
        client_name="mock",
        client_version="1",
    )

    assert adapter.provider == "yahoo"
    assert adapter.dataset == "shares"
    assert adapter.client_name == "mock"
    assert adapter.client_version == "1"
    assert adapter.fetch(_request()).payload == "AAA"

    with pytest.raises(ValueError, match="cannot serve provider"):
        adapter.fetch(_request(provider="sec"))
    with pytest.raises(ValueError, match="cannot serve dataset"):
        adapter.fetch(_request(dataset="prices"))


def test_provider_adapter_rejects_invalid_fetch_result():
    adapter = ProviderAdapter(
        lambda request: object(),
        provider="yahoo",
        dataset="shares",
        client_name="mock",
        client_version="1",
    )

    with pytest.raises(TypeError, match="must return AcquisitionResult"):
        adapter.fetch(_request())


def test_provider_factories_configure_exact_provider_datasets_and_clients():
    client = ClientInfo("client", "1")
    adapters = (
        make_yahoo_ohlcv_adapter(client),
        make_yahoo_shares_adapter(client),
        make_yahoo_benchmark_adapter(client),
        make_wikipedia_sectors_adapter(client),
    )

    assert [adapter.provider for adapter in adapters] == [
        "yahoo",
        "yahoo",
        "yahoo",
        "wikipedia",
    ]
    assert [adapter.dataset for adapter in adapters] == [
        "prices",
        "shares",
        "benchmark",
        "sectors",
    ]
    assert {
        (adapter.client_name, adapter.client_version) for adapter in adapters
    } == {("client", "1")}
    assert adapters[0]._fetcher is fetch_yahoo_ohlcv_once
    assert adapters[1]._fetcher is fetch_yahoo_shares_once
    assert adapters[2]._fetcher is fetch_yahoo_benchmark_once


def test_provider_factories_forward_only_explicit_meaningful_options(monkeypatch):
    wikipedia_calls = []
    monkeypatch.setattr(
        wikipedia_provider,
        "fetch_wikipedia_sector_snapshot_once",
        lambda request, *, pinned_revision_id: (
            wikipedia_calls.append((request, pinned_revision_id)) or _result()
        ),
    )
    client = ClientInfo("client", "1")

    make_wikipedia_sectors_adapter(
        client,
        pinned_revisions={"2024-01-31": 123},
    ).fetch(
        _request(
            dataset="sectors",
            provider="wikipedia",
            provider_symbol="List of S&P 500 companies",
        )
    )

    assert wikipedia_calls[0][1] == 123


def test_provider_status_and_readiness_are_separate_contracts(tmp_path):
    request = _request()
    provider = AcquisitionStatus(
        identity=request.identity,
        status=ProviderStatus.OK,
        requested_start=request.requested_start,
        requested_end=request.requested_end,
        observation_count=2,
        observation_start="2024-01-02",
        observation_end="2024-01-03",
        attempted_at_utc=utc_timestamp(FIXED_TIME),
        client="mock",
        client_version="1",
    )
    readiness = ReadinessRecord(
        scope="backtest",
        dataset="shares",
        asset_id="AAA",
        requirement_set="month_end",
        required_count=3,
        covered_count=2,
        status=ReadinessStatus.PARTIAL,
        missing_dates=("2024-01-31",),
        contributing_sources=("yahoo",),
        checked_at_utc=utc_timestamp(FIXED_TIME),
    )

    status_path = tmp_path / "acquisition_status.csv"
    readiness_path = tmp_path / "readiness.csv"
    write_acquisition_statuses(status_path, (provider,))
    write_readiness(readiness_path, (readiness,))

    assert read_acquisition_statuses(status_path) == [provider]
    assert read_readiness(readiness_path) == [readiness]
    assert status_path.read_text().splitlines()[0].split(",") == list(
        ACQUISITION_STATUS_COLUMNS
    )
    assert "partial" not in {value.value for value in ProviderStatus}


def test_schema_rejects_failed_status_without_structured_error():
    with pytest.raises(ValueError, match="structured error"):
        AcquisitionStatus(
            identity=_request().identity,
            status=ProviderStatus.FAILED,
        )


@pytest.mark.parametrize(
    ("contents", "expected_rows", "expected_columns"),
    (
        ("", 0, ()),
        ("A,B\n", 0, ("A", "B")),
        ("A,B\n1,2\n3,4\n", 2, ("A", "B")),
        (
            'A,B\n1,"quoted,\nvalue"\n2,plain\n',
            2,
            ("A", "B"),
        ),
    ),
    ids=("empty", "header-only", "ordinary", "quoted-multiline"),
)
def test_manifest_infers_csv_shape(
    tmp_path,
    contents,
    expected_rows,
    expected_columns,
):
    artifact = tmp_path / "raw.csv"
    artifact.write_text(contents, encoding="utf-8")

    manifest = ArtifactManifest.from_artifact(
        artifact,
        scope="backtest",
        dataset="shares",
        origin=ArtifactOrigin.MIGRATED,
        artifact_path="raw.csv",
    )

    assert manifest.rows == expected_rows
    assert manifest.columns == expected_columns
    assert validate_manifest_artifact(manifest, base_dir=tmp_path) == artifact


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ({"columns": ("A", "C")}, "manifest column mismatch"),
        ({"rows": 2}, "manifest row-count mismatch"),
    ),
    ids=("header", "row-count"),
)
def test_manifest_validation_rejects_incorrect_inferred_shape(
    tmp_path,
    replacement,
    message,
):
    artifact = tmp_path / "raw.csv"
    artifact.write_text("A,B\n1,2\n", encoding="utf-8")
    manifest = ArtifactManifest.from_artifact(
        artifact,
        scope="backtest",
        dataset="shares",
        origin=ArtifactOrigin.MIGRATED,
        artifact_path="raw.csv",
    )

    with pytest.raises(ValueError, match=message):
        validate_manifest_artifact(
            replace(manifest, **replacement),
            base_dir=tmp_path,
        )


def test_manifest_round_trip_and_artifact_validation(tmp_path):
    artifact = tmp_path / "raw.csv"
    artifact.write_text("A,B\n1,2\n", encoding="utf-8")
    manifest = ArtifactManifest.from_artifact(
        artifact,
        scope="backtest",
        dataset="shares",
        origin=ArtifactOrigin.MIGRATED,
        artifact_path="raw.csv",
    )
    manifest_path = tmp_path / "artifact_manifest.csv"
    write_manifests(manifest_path, (manifest,))

    assert read_manifests(manifest_path) == [manifest]
    assert validate_manifest_artifact(manifest, base_dir=tmp_path) == artifact

    artifact.write_text("A,B\n1,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_manifest_artifact(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("case,message", [
    ("valid", None), ("scope", "scope='live'"), ("dataset", "dataset='prices'"),
    ("extra", "contains"), ("duplicate-path", "duplicate artifact paths"),
    ("basename", "expected"),
])
def test_exact_manifest_catalog_enforces_identity_paths_and_bytes(tmp_path, case, message):
    artifact = tmp_path / ("nested/raw.csv" if case == "basename" else "raw.csv")
    artifact.parent.mkdir(exist_ok=True)
    artifact.write_text("A\n1\n", encoding="utf-8")
    manifest = ArtifactManifest.from_artifact(
        artifact, scope="live", dataset="prices", origin=ArtifactOrigin.MIGRATED,
        artifact_path=artifact.name,
    )
    records = [manifest]
    if case == "scope":
        records = [replace(manifest, scope="backtest")]
    elif case == "dataset":
        records = [replace(manifest, dataset="benchmark")]
    elif case == "extra":
        extra = tmp_path / "extra.csv"
        extra.write_text("A\n2\n", encoding="utf-8")
        records.append(ArtifactManifest.from_artifact(
            extra, scope="live", dataset="prices", origin=ArtifactOrigin.MIGRATED,
            artifact_path=extra.name,
        ))
    elif case == "duplicate-path":
        records.append(replace(manifest, scope="backtest"))
    manifest_path = tmp_path / "artifact_manifest.csv"
    write_manifests(manifest_path, records)
    with pytest.raises(ValueError, match=message) if message else nullcontext():
        actual = validate_exact_manifest_catalog(
            manifest_path, scope="live", dataset="prices",
            expected_origins={artifact.relative_to(tmp_path).as_posix(): ArtifactOrigin.MIGRATED},
            base_dir=tmp_path,
        )
        assert actual == (manifest,)


def test_manifest_rejects_malformed_hash():
    with pytest.raises(ValueError, match="64 hexadecimal"):
        ArtifactManifest(
            schema_version="2",
            scope="live",
            dataset="prices",
            origin=ArtifactOrigin.DOWNLOADED,
            captured_at_utc=utc_timestamp(FIXED_TIME),
            artifact_path="raw.csv",
            sha256="not-a-hash",
            rows=0,
            columns=("Date",),
        )


def test_runtime_paths_are_lazy_and_lock_creates_only_its_parent(tmp_path):
    paths = AcquisitionRuntimePaths(tmp_path)
    assert paths.root == tmp_path.resolve() / "runtime/acquisition"
    lock_path = paths.lock_path("live", "prices")
    assert lock_path.name == "live-prices.lock"
    assert not paths.root.exists()

    with acquisition_lock(lock_path):
        assert lock_path.is_file()
        assert paths.locks.is_dir()
        assert not paths.yfinance_cache.exists()
        assert not paths.sector_history.exists()

    assert list(paths.root.iterdir()) == [paths.locks]


def test_reporter_flushes_utc_line_to_terminal(monkeypatch):
    class RecordingTerminal:
        def __init__(self):
            self.writes = []
            self.flush_count = 0

        def write(self, text):
            self.writes.append(text)

        def flush(self):
            self.flush_count += 1

    terminal = RecordingTerminal()
    report = AcquisitionReporter(terminal=terminal, clock=lambda: FIXED_TIME)
    report("one\nmessage")
    expected = "2026-07-25T20:00:00Z one message\n"
    assert terminal.writes == [expected]
    assert terminal.flush_count == 1

    default_terminal = RecordingTerminal()
    monkeypatch.setattr("data_acquisition.runtime.sys.stderr", default_terminal)
    assert AcquisitionReporter().terminal is default_terminal


def test_atomic_dataframe_write_preserves_target_and_removes_temp_on_failure(
    tmp_path,
):
    target = tmp_path / "state.csv"
    target.write_text("old\n", encoding="utf-8")

    class BrokenFrame:
        @staticmethod
        def to_csv(path, **_kwargs):
            Path(path).write_text("new\n", encoding="utf-8")
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        atomic_write_dataframe(BrokenFrame(), target)
    assert target.read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_acquisition_lock_rejects_concurrent_owner(tmp_path):
    path = tmp_path / "dataset.lock"
    with acquisition_lock(path):
        with pytest.raises(AcquisitionLockedError, match="owns"):
            with acquisition_lock(path):
                pass


class _Clock:
    def __init__(self):
        self.value = 0.0
        self.sleeps: list[float] = []

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


def _engine(adapter, clock=None, **policy_kwargs):
    clock = clock or _Clock()
    policy = AcquisitionPolicy(
        max_attempts=3,
        item_budget_seconds=90,
        initial_backoff_seconds=2,
        max_backoff_seconds=30,
        backoff_jitter_seconds=0,
        **policy_kwargs,
    )
    return SerialAcquisitionEngine(
        adapter,
        policy=policy,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        random_value=lambda: 0,
        clock=lambda: FIXED_TIME,
        enforce_hard_deadline=False,
    ), clock


def test_engine_retries_with_bounded_exponential_backoff_and_checkpoints():
    attempts = []

    def fetch(request):
        attempts.append(request.identity.asset_id)
        if len(attempts) < 3:
            raise RetryableProviderError("temporary")
        return _result()

    adapter = ProviderAdapter(
        fetch,
        provider="yahoo",
        dataset="shares",
        client_name="mock",
        client_version="1",
    )
    engine, clock = _engine(adapter)
    checkpoints = []
    run = engine.run(
        (_request(),),
        checkpoint=lambda statuses, outcome: checkpoints.append((statuses, outcome)),
    )

    assert run.complete
    assert run.statuses[0].status is ProviderStatus.OK
    assert attempts == ["AAA", "AAA", "AAA"]
    assert clock.sleeps == [2, 4]
    assert len(checkpoints) == 2  # initialized pending, then terminal item
    assert checkpoints[0][1] is None
    assert checkpoints[1][1].result.payload == "payload"


def test_engine_stops_immediately_on_explicit_rate_limit():
    calls = []

    def fetch(request):
        calls.append(request.identity.asset_id)
        raise ProviderRateLimited("slow down")

    adapter = ProviderAdapter(
        fetch,
        provider="yahoo",
        dataset="shares",
        client_name="mock",
        client_version="1",
    )
    engine, _ = _engine(adapter)
    run = engine.run((_request("AAA"), _request("BBB")))

    by_asset = {item.identity.asset_id: item for item in run.statuses}
    assert calls == ["AAA"]
    assert run.stopped_reason == "rate_limited"
    assert by_asset["AAA"].status is ProviderStatus.FAILED
    assert by_asset["BBB"].status is ProviderStatus.PENDING
    assert [item.identity.asset_id for item in run.remaining] == ["AAA", "BBB"]
    assert not run.complete


def test_engine_continues_after_consecutive_failed_items():
    calls = []

    def fetch(request):
        calls.append(request.identity.asset_id)
        raise RetryableProviderError("provider broken")

    adapter = ProviderAdapter(
        fetch,
        provider="yahoo",
        dataset="shares",
        client_name="mock",
        client_version="1",
    )
    engine, _ = _engine(adapter)
    run = engine.run((_request("AAA"), _request("BBB"), _request("CCC")))

    assert run.stopped_reason == ""
    assert calls == ["AAA"] * 3 + ["BBB"] * 3 + ["CCC"] * 3
    assert [item.identity.asset_id for item in run.remaining] == ["AAA", "BBB", "CCC"]
    assert not run.complete


def test_engine_attempts_untouched_pending_before_previous_failures():
    calls = []
    failed_requests = (_request("AAA"), _request("AAB"))
    untouched = _request("BBB")
    previous_failures = tuple(
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

    def fetch(request):
        calls.append(request.identity.asset_id)
        if request.identity.asset_id in {"AAA", "AAB"}:
            raise RetryableProviderError("still unavailable")
        return _result()

    adapter = ProviderAdapter(
        fetch,
        provider="yahoo",
        dataset="shares",
        client_name="mock",
        client_version="1",
    )
    engine, _ = _engine(adapter)
    run = engine.run(
        (*failed_requests, untouched),
        statuses=previous_failures,
    )

    assert calls == ["BBB"] + ["AAA"] * 3 + ["AAB"] * 3
    assert run.statuses[-1].status is ProviderStatus.OK


@pytest.mark.parametrize("start", ["2024-01-01", "2024-02-01"], ids=["end-only", "both-endpoints"])
def test_shared_plan_initializes_changed_and_missing_requests_before_failures(start):
    failed = _request("AAA")
    pending = _request("BBB")
    missing = _request("CCC")
    changed = _request("DDD", start=start, end="2024-02-29")
    terminal = _request("EEE")
    old_changed = _request("DDD")
    statuses = (
        AcquisitionStatus(
            identity=failed.identity,
            status=ProviderStatus.FAILED,
            requested_start=failed.requested_start,
            requested_end=failed.requested_end,
            attempted_at_utc="2026-01-01T00:00:00Z",
            client="mock",
            client_version="1",
            error_class="RetryableProviderError",
            error_message="historical symbol unavailable",
        ),
        AcquisitionStatus.pending(pending),
        AcquisitionStatus(
            identity=old_changed.identity,
            status=ProviderStatus.OK,
            requested_start=old_changed.requested_start,
            requested_end=old_changed.requested_end,
            observation_count=1,
            observation_start="2024-01-02",
            observation_end="2024-01-02",
        ),
        AcquisitionStatus(
            identity=terminal.identity,
            status=ProviderStatus.OK,
            requested_start=terminal.requested_start,
            requested_end=terminal.requested_end,
            observation_count=1,
            observation_start="2024-01-02",
            observation_end="2024-01-02",
        ),
    )

    plan = build_acquisition_plan(
        (failed, pending, missing, changed, terminal),
        statuses=statuses,
    )

    assert [
        request.identity.asset_id for request in plan.execution_order
    ] == ["BBB", "CCC", "DDD", "AAA"]
    status_by_asset = {
        status.identity.asset_id: status for status in plan.statuses
    }
    assert status_by_asset["DDD"].status is ProviderStatus.PENDING
    assert status_by_asset["DDD"].requested_start == start
    assert status_by_asset["DDD"].requested_end == "2024-02-29"
    assert status_by_asset["EEE"].status is ProviderStatus.OK

    refreshed = build_acquisition_plan(
        (failed, pending, missing, changed, terminal),
        statuses=statuses,
        refresh=True,
    )
    assert [
        request.identity.asset_id for request in refreshed.execution_order
    ] == ["AAA", "BBB", "CCC", "DDD", "EEE"]


def test_engine_records_confirmed_no_data_without_retry():
    calls = []

    def fetch(request):
        calls.append(request)
        raise ConfirmedNoData("exact target absent", http_status=404)

    adapter = ProviderAdapter(
        fetch,
        provider="sec",
        dataset="shares",
        client_name="requests",
        client_version="1",
    )
    engine, _ = _engine(adapter)
    request = _request(provider="sec", provider_symbol="123")
    run = engine.run((request,))

    assert len(calls) == 1
    assert run.statuses[0].status is ProviderStatus.NO_DATA
    assert run.statuses[0].http_status == "404"


@pytest.mark.parametrize(
    "terminal_status",
    (ProviderStatus.OK, ProviderStatus.NO_DATA),
    ids=("ok", "no-data"),
)
def test_engine_skips_terminal_status_until_refresh(terminal_status):
    calls = []
    request = _request()
    is_ok = terminal_status is ProviderStatus.OK
    terminal = AcquisitionStatus(
        identity=request.identity,
        status=terminal_status,
        requested_start=request.requested_start,
        requested_end=request.requested_end,
        observation_count=1 if is_ok else 0,
        observation_start="2024-01-02" if is_ok else "",
        observation_end="2024-01-02" if is_ok else "",
        http_status="" if is_ok else "404",
    )
    adapter = ProviderAdapter(
        lambda item: calls.append(item) or _result(),
        provider="yahoo",
        dataset="shares",
        client_name="mock",
        client_version="1",
    )
    engine, _ = _engine(adapter)

    skipped = engine.run((request,), statuses=(terminal,))
    refreshed = engine.run((request,), statuses=(terminal,), refresh=True)
    assert skipped.complete
    assert skipped.statuses[0].status is terminal_status
    assert len(calls) == 1
    assert refreshed.statuses[0].status is ProviderStatus.OK


def test_engine_cooperative_budget_marks_slow_result_failed():
    clock = _Clock()

    def fetch(request):
        clock.value += 91
        return _result()

    adapter = ProviderAdapter(
        fetch,
        provider="yahoo",
        dataset="shares",
        client_name="mock",
        client_version="1",
    )
    engine, _ = _engine(adapter, clock=clock)
    run = engine.run((_request(),))
    assert run.statuses[0].status is ProviderStatus.FAILED
    assert run.statuses[0].error_class == "AcquisitionTimeout"
    assert not run.complete
    assert [item.identity.asset_id for item in run.remaining] == ["AAA"]


@pytest.mark.parametrize(
    ("yfinance_version", "curl_cffi_version", "found_versions"),
    (
        (
            "0.2.57",
            REQUIRED_CURL_CFFI_VERSION,
            "yfinance=0.2.57, curl-cffi=0.15.0",
        ),
        (
            REQUIRED_YFINANCE_VERSION,
            "0.14.0",
            "yfinance=1.5.1, curl-cffi=0.14.0",
        ),
    ),
    ids=("yfinance", "curl-cffi"),
)
def test_yfinance_version_validation_precedes_cache_creation(
    tmp_path,
    yfinance_version,
    curl_cffi_version,
    found_versions,
):
    calls = []
    module = SimpleNamespace(
        __version__=yfinance_version,
        set_tz_cache_location=lambda path: calls.append(path),
    )
    with pytest.raises(RuntimeError) as exc_info:
        prepare_yfinance(
            tmp_path / "cache",
            yfinance_module=module,
            package_version=lambda name: curl_cffi_version,
        )
    assert str(exc_info.value) == (
        "Yahoo acquisition requires yfinance 1.5.1 and curl-cffi 0.15.0 in "
        f"the Conda environment `finance`; found {found_versions}. Update the "
        "environment with `conda env update -n finance -f environment.yml` "
        "before retrying."
    )
    assert calls == []
    assert not (tmp_path / "cache").exists()


def test_yfinance_validation_and_project_cache_configuration(tmp_path):
    calls = []
    cache_dir = tmp_path / "cache"
    module = SimpleNamespace(
        __version__=REQUIRED_YFINANCE_VERSION,
        set_tz_cache_location=lambda path: calls.append(path),
    )
    info = prepare_yfinance(
        cache_dir,
        yfinance_module=module,
        package_version=lambda name: REQUIRED_CURL_CFFI_VERSION,
    )
    assert info == ClientInfo("yfinance", REQUIRED_YFINANCE_VERSION)
    assert cache_dir.is_dir()
    assert calls == [str(cache_dir)]


@pytest.mark.parametrize(
    "module",
    [
        SimpleNamespace(__version__=REQUIRED_YFINANCE_VERSION),
        SimpleNamespace(
            __version__=REQUIRED_YFINANCE_VERSION,
            set_tz_cache_location=None,
        ),
    ],
)
def test_yfinance_preparation_requires_callable_public_cache_setter(
    tmp_path, module
):
    cache_dir = tmp_path / "cache"
    with pytest.raises(RuntimeError) as exc_info:
        prepare_yfinance(
            cache_dir,
            yfinance_module=module,
            package_version=lambda name: REQUIRED_CURL_CFFI_VERSION,
        )
    assert str(exc_info.value) == (
        "The installed yfinance has no public cache-location API"
    )
    assert not cache_dir.exists()


@pytest.mark.parametrize(
    "fetcher,dataset,fields",
    [
        (fetch_yahoo_ohlcv_once, "prices", ("Open", "High", "Low", "Close", "Volume")),
        (fetch_yahoo_benchmark_once, "benchmark", ("Open", "High", "Low", "Close")),
    ],
)
def test_yahoo_price_fetchers_translate_and_enforce_request_bounds(
    fetcher, dataset, fields
):
    calls = []
    frame = pd.DataFrame(
        {field: [99.0, 100.0, 101.0, 102.0] for field in fields},
        index=pd.DatetimeIndex([
            "2023-12-31",
            "2024-01-02",
            "2024-01-03",
            "2024-02-01",
        ]),
    )
    if "Volume" in frame:
        frame["Volume"] = [900.0, 1000.0, 2000.0, 2100.0]

    result = fetcher(
        _request(dataset=dataset),
        downloader=lambda **kwargs: calls.append(kwargs) or frame,
    )
    assert result.observation_count == 2
    assert list(result.payload.index.strftime("%Y-%m-%d")) == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert calls[0]["start"] == "2024-01-01"
    assert calls[0]["end"] == "2024-02-01"
    assert calls[0]["auto_adjust"] is True
    assert "actions" not in calls[0]
    assert calls[0]["threads"] is False


def test_yahoo_empty_price_response_is_retryable_not_no_data():
    with pytest.raises(RetryableProviderError, match="absence was not confirmed"):
        fetch_yahoo_ohlcv_once(
            _request(dataset="prices"),
            downloader=lambda **kwargs: pd.DataFrame(),
        )


def test_yahoo_malformed_price_response_is_retryable_without_http_status():
    malformed = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": ["invalid"],
        },
        index=pd.DatetimeIndex(["2024-01-02"]),
    )

    with pytest.raises(RetryableProviderError) as caught:
        fetch_yahoo_benchmark_once(
            _request(dataset="benchmark"),
            downloader=lambda **kwargs: malformed,
        )

    assert caught.value.http_status is None


def test_yahoo_shares_default_probe_confirms_no_data_only_with_valid_prices():
    history_calls = []

    class Ticker:
        def get_shares_full(self, **kwargs):
            assert kwargs["end"] == "2024-02-01"
            return None

        def history(self, **kwargs):
            history_calls.append(kwargs)
            return pd.DataFrame(
                {"Close": [10.0]},
                index=pd.DatetimeIndex(["2024-01-10"]),
            )

    with pytest.raises(ConfirmedNoData, match="no shares series"):
        fetch_yahoo_shares_once(_request(), client_factory=lambda symbol: Ticker())
    assert history_calls[0]["end"] == "2024-02-01"
    assert history_calls[0]["auto_adjust"] is False


def test_yahoo_shares_clips_provider_rows_to_requested_inclusive_window():
    class Ticker:
        def get_shares_full(self, **kwargs):
            assert kwargs == {
                "start": "2024-01-01",
                "end": "2024-02-01",
            }
            return pd.Series(
                [90.0, 100.0, 110.0],
                index=pd.DatetimeIndex([
                    "2023-12-31",
                    "2024-01-15",
                    "2024-02-01",
                ]),
            )

    result = fetch_yahoo_shares_once(
        _request(),
        client_factory=lambda symbol: Ticker(),
    )

    assert result.observation_count == 1
    assert result.observation_start == "2024-01-15"
    assert result.observation_end == "2024-01-15"
    assert result.payload.tolist() == [100.0]


@pytest.mark.parametrize(
    "probe,message",
    (
        pytest.param(pd.DataFrame(), "unconfirmed", id="empty"),
        pytest.param(pd.DataFrame({"Close": [0.0]}), "invalid prices", id="invalid"),
    ),
)
def test_yahoo_shares_empty_target_requires_a_valid_probe(probe, message):
    class Ticker:
        def get_shares_full(self, **kwargs):
            return None

        def history(self, **kwargs):
            return probe

    with pytest.raises(RetryableProviderError, match=message):
        fetch_yahoo_shares_once(_request(), client_factory=lambda symbol: Ticker())


def test_yahoo_rate_limit_is_structured_without_phrase_matching():
    class YFRateLimitError(Exception):
        pass

    def downloader(**kwargs):
        raise YFRateLimitError("throttled")

    with pytest.raises(ProviderRateLimited) as caught:
        fetch_yahoo_benchmark_once(
            _request(dataset="benchmark"), downloader=downloader
        )

    assert caught.value.http_status == 429


def test_yahoo_response_http_429_is_structured_as_rate_limit():
    error = RuntimeError("throttled by response")
    error.response = SimpleNamespace(status_code="429")

    def downloader(**kwargs):
        raise error

    with pytest.raises(ProviderRateLimited) as caught:
        fetch_yahoo_benchmark_once(
            _request(dataset="benchmark"), downloader=downloader
        )

    assert caught.value.http_status == 429


@pytest.mark.parametrize(
    ("status_code", "expected_status"),
    (("503", 503), ("not-an-http-status", None)),
    ids=("convertible", "malformed"),
)
def test_yahoo_direct_http_status_is_normalized_for_retryable_errors(
    status_code,
    expected_status,
):
    error = RuntimeError("provider unavailable")
    error.status_code = status_code

    def downloader(**kwargs):
        raise error

    with pytest.raises(RetryableProviderError) as caught:
        fetch_yahoo_benchmark_once(
            _request(dataset="benchmark"), downloader=downloader
        )

    assert caught.value.http_status == expected_status


def test_root_cli_dispatches_dry_run_without_owning_domain_logic(
    tmp_path,
    monkeypatch,
):
    captured = []
    monkeypatch.setattr(
        acquisition_cli,
        "_handler_mapping",
        lambda: {
            ("backtest", "shares"): lambda request: captured.append(request) or 7
        },
    )
    exit_code = main(
        ["backtest", "shares", "--dry-run", "--tickers-file", str(tmp_path / "x.csv")]
    )
    assert exit_code == 7
    assert captured[0].scope == "backtest"
    assert captured[0].dataset == "shares"
    assert captured[0].dry_run is True
    assert captured[0].tickers_file == tmp_path / "x.csv"


@pytest.mark.parametrize(
    ("scope", "dataset"),
    [(scope, dataset) for scope in SCOPES for dataset in DATASETS],
)
def test_root_cli_accepts_every_supported_scope_dataset_pair(
    scope,
    dataset,
    monkeypatch,
):
    captured = []
    monkeypatch.setattr(
        acquisition_cli,
        "_handler_mapping",
        lambda: {
            (scope, dataset): lambda request: captured.append(request) or 0
        },
    )

    assert main([scope, dataset, "--dry-run"]) == 0
    assert [(item.scope, item.dataset) for item in captured] == [(scope, dataset)]


def test_fixed_handler_mapping_has_every_supported_pair():
    assert set(acquisition_cli._handler_mapping()) == {
        (scope, dataset) for scope in SCOPES for dataset in DATASETS
    }


def test_root_cli_rejects_invalid_option_dataset_combination():
    with pytest.raises(SystemExit) as caught:
        main(["backtest", "benchmark", "--tickers-file", "x.csv"])
    assert caught.value.code == 2
