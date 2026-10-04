"""Offline contracts for root-CLI backtest acquisition handlers."""

from dataclasses import replace
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

from backtest import acquisition_handlers
from backtest import acquisition_execution, acquisition_planning
from backtest.config import DEFAULT_CONFIG
from backtest.paths import BacktestSharesPaths
from portfolio_core.artifacts import (
    ArtifactOrigin,
    read_manifests,
    validate_manifest_artifact,
)
from data_acquisition.contracts import (
    AcquisitionIdentity,
    AcquisitionStatus,
    CommandRequest,
    ProviderStatus,
    ReadinessRecord,
    ReadinessStatus,
    read_acquisition_statuses,
    read_readiness,
)
from portfolio_core.shares import RAW_SHARES_COLUMNS
from _acquisition_test_helpers import backtest_acquisition_request


@pytest.fixture(scope="module")
def fresh_core_data():
    return acquisition_planning.load_fresh_core_data(DEFAULT_CONFIG)


def test_yahoo_requests_cover_only_primary_residuals(fresh_core_data):
    plan = acquisition_planning.plan_share_sources(fresh_core_data)
    requests = acquisition_planning.build_shares_requests(plan)
    assert len(plan.primary) == 12520
    assert len(plan.unresolved) == 56
    assert len(requests) == 20
    assert {r.identity.asset_id for r in requests} == set(plan.unresolved.Asset_ID)
    assert all(r.identity.provider == "yahoo" for r in requests)
    assert [r.identity.key for r in requests] == sorted(r.identity.key for r in requests)


def test_share_identity_source_window_filter_removes_nonoverlapping_aliases():
    identities = acquisition_planning._build_yahoo_shares_identities(
        ("META.O", "PSKY.O"),
        {"META.O": "META", "PSKY.O": "PSKY"},
    )

    assert {
        (identity.asset_id, identity.provider_symbol)
        for identity in identities
    } == {("META.O", "META"), ("PSKY.O", "PARA")}


def test_pre_extension_audit_tickers_preserve_the_january_2024_baseline(
    fresh_core_data,
):
    assert fresh_core_data.asset_to_ticker["DOC"] == "DOC"
    assert fresh_core_data.audit_ticker("DOC", "2024-01-31") == "PEAK"
    assert fresh_core_data.audit_ticker("FISV.O", "2023-12-31") == "FI"
    assert fresh_core_data.audit_ticker("DOC", "2024-02-29") == "DOC"


def test_materialized_shares_status_and_readiness_are_independently_valid(
    fresh_core_data,
    monkeypatch,
):
    raw, statuses = acquisition_execution._read_shares_state()
    yahoo_statuses = [
        status for status in statuses if status.identity.provider == "yahoo"
    ]
    readiness = pd.read_csv(DEFAULT_CONFIG.paths.shares.readiness_csv)
    monkeypatch.setattr(
        acquisition_planning,
        "utc_timestamp",
        lambda: "2026-08-26T00:00:00Z",
    )
    plan = acquisition_planning.plan_share_sources(fresh_core_data)
    actual, detail = acquisition_planning.shares_readiness_records(
        plan,
        raw,
    )

    request_keys = {
        request.identity.key
        for request in acquisition_planning.build_shares_requests(plan)
    }
    assert request_keys <= {status.identity.key for status in yahoo_statuses}
    assert set(status.status for status in yahoo_statuses) <= set(ProviderStatus)
    assert int(readiness.loc[0, "Required_Count"]) == 12_576
    covered = int(readiness.loc[0, "Covered_Count"])
    assert 0 <= covered <= 12_576
    expected_status = (
        "complete" if covered == 12_576 else "missing" if covered == 0 else "partial"
    )
    assert readiness.loc[0, "Status"] == expected_status
    assert actual == [
        ReadinessRecord(
            scope="backtest",
            dataset="shares",
            asset_id="*",
            requirement_set="brinson_active_asset_month",
            required_count=12_576,
            covered_count=12_576,
            status=ReadinessStatus.COMPLETE,
            contributing_sources=("event_estimate", "sec", "yahoo"),
            checked_at_utc="2026-08-26T00:00:00Z",
        )
    ]
    assert len(detail) == 12_576
    assert detail["Covered"].all()


def test_backtest_acquisition_initializes_absent_checkpoint(
    tmp_path,
    monkeypatch,
):
    paths = BacktestSharesPaths(
        tmp_path / "raw/shares",
        tmp_path / "prepared",
        tmp_path / "raw/provenance/shares",
    )
    monkeypatch.setattr(
        acquisition_execution,
        "DEFAULT_CONFIG",
        SimpleNamespace(paths=SimpleNamespace(shares=paths)),
    )

    raw, statuses = acquisition_execution._read_shares_state()
    assert raw.empty and tuple(raw.columns) == RAW_SHARES_COLUMNS
    assert statuses == []


def test_aggregate_manifest_keeps_migrated_lineage_until_full_reacquisition():
    migrated = AcquisitionStatus(
        identity=AcquisitionIdentity(
            scope="backtest",
            dataset="shares",
            asset_id="AAA",
            provider="yahoo",
            provider_symbol="AAA",
        ),
        status=ProviderStatus.OK,
        requested_start="2024-01-01",
        requested_end="2024-01-31",
        observation_count=1,
        observation_start="2024-01-31",
        observation_end="2024-01-31",
        migration_note="synthetic imported checkpoint",
    )
    downloaded = replace(
        migrated,
        identity=replace(migrated.identity, asset_id="BBB", provider_symbol="BBB"),
        migration_note="",
        attempted_at_utc="2026-07-25T12:00:00Z",
        client="mock",
        client_version="1",
    )

    assert acquisition_execution._aggregate_shares_provenance(
        [migrated, downloaded]
    ) == (
        acquisition_execution.ArtifactOrigin.MIGRATED,
        "2026-07-25T12:00:00Z",
    )
    assert acquisition_execution._aggregate_shares_provenance([downloaded]) == (
        acquisition_execution.ArtifactOrigin.DOWNLOADED,
        "2026-07-25T12:00:00Z",
    )


def test_backtest_shares_manifest_bytes_are_unchanged_with_inferred_shape(
    tmp_path,
    monkeypatch,
):
    source = DEFAULT_CONFIG.paths.shares
    target = BacktestSharesPaths(
        tmp_path / "data/backtest/raw/shares",
        tmp_path / "data/backtest/prepared",
        tmp_path / "data/backtest/raw/provenance/shares",
    )
    target.raw_dir.mkdir(parents=True)
    for attribute in (
        "raw_shares_csv",
        "acquisition_status_csv",
        "readiness_csv",
    ):
        shutil.copyfile(getattr(source, attribute), getattr(target, attribute))
    monkeypatch.setattr(
        acquisition_execution,
        "DEFAULT_CONFIG",
        SimpleNamespace(
            paths=SimpleNamespace(
                shares=target,
                project_root=tmp_path,
            )
        ),
    )

    acquisition_execution._write_shares_manifest(
        read_acquisition_statuses(target.acquisition_status_csv)
    )

    assert target.artifact_manifest_csv.read_bytes() == (
        source.artifact_manifest_csv.read_bytes()
    )


def test_ticker_subset_reads_the_complete_checkpoint_without_dropping_histories(
    tmp_path,
    fresh_core_data,
):
    ticker_file = tmp_path / "tickers.csv"
    ticker_file.write_text("Ticker\nMETA\n")
    requests = acquisition_planning.build_shares_requests(
        acquisition_planning.plan_share_sources(fresh_core_data),
        tickers_file=ticker_file,
    )

    raw, statuses = acquisition_execution._read_shares_state()
    durable_raw = pd.read_csv(
        DEFAULT_CONFIG.paths.shares.raw_shares_csv,
        keep_default_na=False,
    )
    durable_statuses = read_acquisition_statuses(
        DEFAULT_CONFIG.paths.shares.acquisition_status_csv
    )

    assert requests == []  # Reviewed reported counts cover META; no Yahoo request.
    pd.testing.assert_frame_equal(
        raw.reset_index(drop=True),
        durable_raw.assign(
            Date=pd.to_datetime(durable_raw["Date"]),
            Shares_Outstanding=pd.to_numeric(
                durable_raw["Shares_Outstanding"]
            ).astype("float64"),
        ),
    )
    assert {status.identity.key for status in statuses} == {
        status.identity.key for status in durable_statuses
    }


@pytest.mark.parametrize("retryable", [True, False], ids=["mixed", "all-terminal"])
def test_shares_dry_run_filters_orders_and_forwards_one_plan(capsys, monkeypatch, retryable):
    failed, pending, terminal, retired = [
        backtest_acquisition_request(asset) for asset in ("AAA", "BBB", "CCC", "RETIRED")
    ]
    requests = [failed, pending, terminal] if retryable else [terminal]
    statuses = [
        replace(AcquisitionStatus.pending(request), status=ProviderStatus.FAILED,
                error_class="RetryableProviderError", error_message="synthetic failure")
        for request in (failed, retired)
    ] + [
        AcquisitionStatus.pending(pending),
        replace(AcquisitionStatus.pending(terminal), status=ProviderStatus.OK,
                observation_count=1, observation_start=terminal.requested_start,
                observation_end=terminal.requested_start),
    ]
    source_plan = SimpleNamespace(yahoo_identities=(object(),))
    received_plans = []
    monkeypatch.setattr(acquisition_execution, "load_fresh_core_data", lambda config: object())
    monkeypatch.setattr(acquisition_execution, "plan_share_sources", lambda *a, **k: source_plan)
    monkeypatch.setattr(acquisition_execution, "build_shares_requests",
                        lambda plan, *, tickers_file: received_plans.append(plan) or requests)
    monkeypatch.setattr(acquisition_execution, "_read_shares_state",
                        lambda: (pd.DataFrame(columns=RAW_SHARES_COLUMNS), statuses))
    monkeypatch.setattr(acquisition_execution, "prepare_yfinance",
                        lambda *a: pytest.fail("Dry runs must not initialize a provider"))
    command = SimpleNamespace(tickers_file=None, refresh=False, dry_run=True,
                              project_root=DEFAULT_CONFIG.paths.project_root)
    assert acquisition_execution.acquire_backtest_shares(command) == 0
    assert received_plans == [source_plan]
    output = capsys.readouterr().out
    assert "RETIRED" not in output
    assert f"{len(requests)} effective-dated Yahoo identities; 1 terminal" in output
    assert f"{2 if retryable else 0} pending/retryable" in output
    lines = [line.strip() for line in output.splitlines() if "previous=" in line]
    assert len(lines) == (2 if retryable else 0)
    if retryable:
        assert lines[0].startswith("[1/2] previous=pending BBB ->")
        assert lines[1].startswith("[2/2] previous=failed AAA ->")


def test_stale_core_blocks_shares_before_checkpoint_or_provider_setup(monkeypatch):
    def stale_core(*args, **kwargs):
        raise RuntimeError("canonical identity hash changed")

    def forbidden(*args, **kwargs):
        pytest.fail("shares preflight continued after stale core detection")

    monkeypatch.setattr(acquisition_planning, "load_backtest_data", stale_core)
    monkeypatch.setattr(acquisition_execution, "_read_shares_state", forbidden)
    monkeypatch.setattr(acquisition_execution, "prepare_yfinance", forbidden)

    with pytest.raises(
        RuntimeError,
        match=r"backtest\.prepare core.*retry the shares command",
    ):
        acquisition_planning.load_fresh_core_data(DEFAULT_CONFIG)

    request = SimpleNamespace(
        tickers_file=None,
        refresh=False,
        dry_run=True,
        project_root=DEFAULT_CONFIG.paths.project_root,
    )
    with pytest.raises(
        RuntimeError,
        match=r"backtest\.prepare core.*retry the shares command",
    ):
        acquisition_execution.acquire_backtest_shares(request)


def test_backtest_all_propagates_the_fresh_core_recovery(monkeypatch):
    monkeypatch.setattr(
        acquisition_handlers, "acquire_backtest_prices", lambda request: 0
    )
    monkeypatch.setattr(
        acquisition_handlers, "acquire_backtest_benchmark", lambda request: 0
    )
    monkeypatch.setattr(
        acquisition_handlers, "acquire_backtest_sectors", lambda request: 0
    )
    monkeypatch.setattr(
        acquisition_planning,
        "load_backtest_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stale core")),
    )
    request = CommandRequest(
        scope="backtest",
        dataset="all",
        tickers_file=None,
        refresh=False,
        dry_run=True,
        project_root=DEFAULT_CONFIG.paths.project_root,
    )

    with pytest.raises(
        RuntimeError,
        match=r"backtest\.prepare core.*retry the shares command",
    ):
        acquisition_handlers.acquire_backtest_all(request)



def test_materialized_benchmark_provenance_is_complete_and_self_consistent():
    paths = DEFAULT_CONFIG.paths
    statuses = read_acquisition_statuses(paths.benchmark_acquisition_status_csv)
    readiness = read_readiness(paths.benchmark_readiness_csv)
    manifests = read_manifests(paths.benchmark_artifact_manifest_csv)

    assert len(statuses) == 1
    assert statuses[0].identity.provider_symbol == "^SP500TR"
    assert statuses[0].requested_start == "2014-01-01"
    assert statuses[0].requested_end == "2026-01-31"
    assert len(readiness) == 1
    assert readiness[0].required_count == 145
    assert {item.artifact_path for item in manifests} == {
        "data/backtest/raw/sp500tr_raw.csv",
        "data/backtest/raw/benchmark/acquisition_status.csv",
        "data/backtest/raw/benchmark/readiness.csv",
    }
    for manifest in manifests:
        validate_manifest_artifact(manifest, base_dir=paths.project_root)
    raw_manifest = next(
        item for item in manifests
        if item.artifact_path.endswith("sp500tr_raw.csv")
    )
    assert statuses[0].status is ProviderStatus.OK
    assert statuses[0].observation_count == 145
    assert readiness[0].status is ReadinessStatus.COMPLETE
    assert readiness[0].covered_count == 145
    assert readiness[0].contributing_sources == ("yahoo",)
    assert raw_manifest.origin.value == "downloaded"
    assert raw_manifest.captured_at_utc


def test_benchmark_state_rejects_migrated_raw_provenance(monkeypatch):
    paths = DEFAULT_CONFIG.paths
    raw_path = paths.benchmark_raw_csv.relative_to(paths.project_root).as_posix()
    manifests = [
        replace(item, origin=ArtifactOrigin.MIGRATED)
        if item.artifact_path == raw_path
        else item
        for item in read_manifests(paths.benchmark_artifact_manifest_csv)
    ]
    monkeypatch.setattr(acquisition_execution, "read_manifests", lambda path: manifests)

    with pytest.raises(ValueError, match="origin must be downloaded"):
        acquisition_execution._load_benchmark_state()


def test_supplied_reuters_prices_have_one_valid_canonical_manifest():
    paths = DEFAULT_CONFIG.paths
    manifests = read_manifests(paths.prices_artifact_manifest_csv)

    assert len(manifests) == 1
    assert manifests[0].origin.value == "supplied"
    validate_manifest_artifact(
        manifests[0],
        base_dir=paths.prices_csv.parent,
    )


def test_benchmark_request_and_dry_run_are_network_free(capsys, monkeypatch):
    requests = acquisition_planning.build_benchmark_requests()
    assert len(requests) == 1
    assert requests[0].identity.asset_id == "SP500TR"
    assert requests[0].identity.provider == "yahoo"
    assert requests[0].identity.provider_symbol == "^SP500TR"
    assert requests[0].requested_start == "2014-01-01"
    assert requests[0].requested_end == "2026-01-31"
    monkeypatch.setattr(
        acquisition_execution,
        "prepare_yfinance",
        lambda *args, **kwargs: pytest.fail("dry-run prepared a Yahoo client"),
    )
    request = SimpleNamespace(
        tickers_file=None,
        refresh=False,
        dry_run=True,
        project_root=DEFAULT_CONFIG.paths.project_root,
    )

    assert acquisition_execution.acquire_backtest_benchmark(request) == 0
    output = capsys.readouterr().out
    assert "1 Yahoo identity" in output
    status = read_acquisition_statuses(
        DEFAULT_CONFIG.paths.benchmark_acquisition_status_csv
    )[0]
    expected_pending = 0 if status.status is ProviderStatus.OK else 1
    assert f"{expected_pending} pending/retryable" in output
    if expected_pending:
        assert "SP500TR -> yahoo:^SP500TR" in output


def test_yahoo_benchmark_normalization_uses_last_trading_close_and_month_end():
    payload = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0, 103.0]},
        index=pd.DatetimeIndex([
            "2024-01-30",
            "2024-01-31",
            "2024-02-28",
            "2024-02-29",
        ], name="Date"),
    )

    actual = acquisition_execution.normalize_yahoo_benchmark_monthly(payload)

    pd.testing.assert_frame_equal(
        actual,
        pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "SP500TR_Close": [101.0, 103.0],
        }),
    )


def test_benchmark_adapter_accepts_current_complete_yahoo_history(monkeypatch):
    dates = DEFAULT_CONFIG.benchmark.required_dates
    payload = pd.DataFrame(
        {"Close": range(100, 100 + len(dates))},
        index=dates,
    )
    result = acquisition_execution.AcquisitionResult(
        payload=payload,
        observation_count=len(payload),
        observation_start=dates.min().strftime("%Y-%m-%d"),
        observation_end=dates.max().strftime("%Y-%m-%d"),
        http_status=200,
    )
    yahoo = acquisition_execution.ProviderAdapter(
        lambda request: result,
        provider="yahoo",
        dataset="benchmark",
        client_name="mock-yahoo",
        client_version="1",
    )
    monkeypatch.setattr(
        acquisition_execution,
        "make_yahoo_benchmark_adapter",
        lambda client: yahoo,
    )

    adapter = acquisition_execution._make_canonical_benchmark_adapter(object())
    actual = adapter.fetch(acquisition_planning.build_benchmark_requests()[0])

    assert actual.observation_count == 145
    february_2024 = actual.payload.loc[
        actual.payload["Date"] == "2024-02-29",
        "SP500TR_Close",
    ].item()
    assert february_2024 == 221


def test_sector_handler_routes_dry_run_and_execution(tmp_path, monkeypatch):
    calls = []
    request = CommandRequest(
        scope="backtest",
        dataset="sectors",
        tickers_file=None,
        refresh=False,
        dry_run=True,
        project_root=tmp_path,
    )
    execution_request = replace(request, dry_run=False)
    monkeypatch.setattr(
        acquisition_handlers,
        "dry_run_sector_acquisition",
        lambda received, planner: calls.append(("dry_run", received, planner)),
    )
    monkeypatch.setattr(
        acquisition_handlers,
        "execute_sector_acquisition",
        lambda received, planner: calls.append(("execute", received, planner)) or 7,
    )

    assert acquisition_handlers.acquire_backtest_sectors(request) == 0
    assert acquisition_handlers.acquire_backtest_sectors(execution_request) == 7
    assert calls == [
        ("dry_run", request, acquisition_planning.backtest_sector_requirements),
        (
            "execute",
            execution_request,
            acquisition_planning.backtest_sector_requirements,
        ),
    ]


def test_backtest_all_stops_after_first_failed_dataset(monkeypatch):
    calls = []

    def outcome(name, code):
        def handler(request):
            calls.append(name)
            return code

        return handler

    monkeypatch.setattr(
        acquisition_handlers, "acquire_backtest_prices", outcome("prices", 0)
    )
    monkeypatch.setattr(
        acquisition_handlers, "acquire_backtest_benchmark", outcome("benchmark", 1)
    )
    monkeypatch.setattr(
        acquisition_handlers,
        "acquire_backtest_sectors",
        lambda request: pytest.fail("all continued after benchmark failure"),
    )
    monkeypatch.setattr(
        acquisition_handlers,
        "acquire_backtest_shares",
        lambda request: pytest.fail("all continued after benchmark failure"),
    )
    request = CommandRequest(
        scope="backtest",
        dataset="all",
        tickers_file=None,
        refresh=False,
        dry_run=False,
        project_root=DEFAULT_CONFIG.paths.project_root,
    )

    assert acquisition_handlers.acquire_backtest_all(request) == 1
    assert calls == ["prices", "benchmark"]


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("selected_ticker", [None, "META", "ADM"])
def test_acquisition_reuses_one_plan_with_projected_identities(
    tmp_path, monkeypatch, fresh_core_data, dry_run, selected_ticker,
):
    from contextlib import nullcontext
    from backtest.paths import BacktestSharesPaths

    project = acquisition_planning.project_share_identity_interval
    projections = []

    def project_once(**kwargs):
        result = project(**kwargs)
        projections.append(result)
        return result

    monkeypatch.setattr(acquisition_planning, "project_share_identity_interval", project_once)
    plan = acquisition_planning.plan_share_sources(fresh_core_data)
    assert len(projections) == len(plan.yahoo_identities) == 20
    assert plan.canonical_identities == tuple(item[0] for item in projections)
    for item, (_, start, end) in zip(plan.yahoo_identities, projections):
        assert (item.requested_start, item.requested_end) == (start, end)
    monkeypatch.setattr(acquisition_planning, "project_share_identity_interval",
                        lambda **kwargs: pytest.fail("Identity was projected again"))

    raw, statuses = acquisition_execution._read_shares_state()
    ticker_file = None
    if selected_ticker:
        ticker_file = tmp_path / "tickers.csv"
        ticker_file.write_text(f"Ticker\n{selected_ticker}\n")
    requests = acquisition_planning.build_shares_requests(plan, tickers_file=ticker_file)
    assert len(requests) == {None: 20, "META": 0, "ADM": 1}[selected_ticker]
    for request in requests:
        item = next(item for item in plan.yahoo_identities
                    if item.identity.asset_id == request.identity.asset_id)
        assert (request.identity.asset_id, request.identity.provider_symbol,
                request.identity.effective_start, request.identity.effective_end) == item.identity.raw_key
        assert (request.requested_start, request.requested_end) == (item.requested_start, item.requested_end)

    calls = []
    def source_plan(market, *, config):
        assert market is fresh_core_data
        calls.append("plan")
        return plan

    readiness = acquisition_planning.shares_readiness_records
    def checked_readiness(actual_plan, actual_raw):
        assert actual_plan is plan
        pd.testing.assert_frame_equal(actual_raw, raw)
        calls.append("readiness")
        records, detail = readiness(actual_plan, actual_raw)
        assert records[0].covered_count == 12576 and detail.Covered.all()
        return records, detail

    paths = BacktestSharesPaths(
        tmp_path / "raw", tmp_path / "prepared", tmp_path / "raw/provenance/shares",
    )
    monkeypatch.setattr(acquisition_execution, "DEFAULT_CONFIG", SimpleNamespace(paths=SimpleNamespace(shares=paths)))
    monkeypatch.setattr(acquisition_execution, "load_fresh_core_data", lambda config: fresh_core_data)
    monkeypatch.setattr(acquisition_execution, "plan_share_sources", source_plan)
    monkeypatch.setattr(acquisition_execution, "shares_readiness_records", checked_readiness)
    monkeypatch.setattr(acquisition_execution, "_read_shares_state", lambda: (raw, statuses))
    monkeypatch.setattr(acquisition_execution, "prepare_yfinance", lambda *a: pytest.fail("Terminal checkpoints need no provider"))
    monkeypatch.setattr(acquisition_execution, "acquisition_lock", lambda *a: nullcontext())
    monkeypatch.setattr(acquisition_execution, "_write_shares_manifest", lambda *a: None)
    command = SimpleNamespace(tickers_file=ticker_file, refresh=False, dry_run=dry_run, project_root=tmp_path)
    assert acquisition_execution.acquire_backtest_shares(command) == 0
    assert calls == (["plan"] if dry_run else ["plan", "readiness"])
