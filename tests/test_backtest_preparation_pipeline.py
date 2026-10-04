"""Tests for the offline CSV preparation/read-only analysis boundary."""

from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

import backtest.data_loading as data_loading
import backtest.preparation_builders as preparation_builders
import backtest.preparation_pipeline as preparation_pipeline
from backtest.config import (
    BACKTEST_WARMUP_MONTHS,
    BacktestBenchmarkConfig,
    BacktestMarketConfig,
)
from backtest.data_loading import (
    PREPARED_ASSET_METADATA_COLUMNS,
    PREPARED_PRICE_COLUMNS,
    load_backtest_data,
)
from backtest.paths import BacktestPaths
from backtest.preparation_builders import (
    prepare_core_data,
    validate_prepared_data,
)
from backtest.preparation_artifacts import (
    write_preparation_manifest,
    validate_preparation_manifest,
)
from data_acquisition.contracts import (
    AcquisitionIdentity,
    AcquisitionStatus,
    ProviderStatus,
    ReadinessRecord,
    ReadinessStatus,
    read_acquisition_statuses,
    write_acquisition_statuses,
    write_readiness,
)
from portfolio_core.artifacts import (
    ArtifactManifest,
    ArtifactOrigin,
    file_sha256,
    read_manifests,
    write_manifests,
)
from _membership_test_helpers import _write_membership_source_manifest
from portfolio_core.security_identity import (
    PROVIDER_ROW_EQUIVALENCE_COLUMNS,
    PROVIDER_SYMBOL_MAPPINGS_COLUMNS,
    SECURITY_EVENTS_COLUMNS,
    SECURITY_EVENT_LEGS_COLUMNS,
    SECURITY_EVENT_SOURCES_COLUMNS,
)
from portfolio_core.sector_assignments import sector_rows_asof
from _sector_test_helpers import (
    _write_sector_history_manifest,
    _write_synthetic_sector_history,
)

def _write_synthetic_membership_sources(
    paths: BacktestPaths,
    end_date: date,
) -> None:
    directory = paths.membership.directory
    directory.mkdir(parents=True)
    pd.DataFrame({
        "date": ["2014-01-01", end_date.isoformat(), "2019-01-01"],
        "tickers": ["AAA,BBB", "AAA,BBB", "BBB,CCC"],
    }).to_csv(paths.membership.components_csv, index=False)
    pd.DataFrame([
        {"date": "2019-01-01", "add": "CCC", "remove": "AAA"},
    ]).to_csv(paths.membership.changes_csv, index=False)
    pd.DataFrame([
        {
            "ticker": "AAA",
            "start_date": "2014-01-01",
            "end_date": "2019-01-01",
        },
        {"ticker": "BBB", "start_date": "2014-01-01", "end_date": ""},
        {"ticker": "CCC", "start_date": "2019-01-01", "end_date": ""},
    ]).to_csv(paths.membership.intervals_csv, index=False)

    _write_membership_source_manifest(paths.membership)


def _write_synthetic_security_identity(paths: BacktestPaths) -> None:
    """Write a valid canonical bundle with no synthetic Reuters overrides."""
    directory = paths.security_identity.directory
    directory.mkdir(parents=True)
    frames = {
        paths.security_identity.events_csv: pd.DataFrame([{
            "Event_ID": "EVT-20300101-TEST-DOCUMENTARY",
            "Effective_Date": "2030-01-01",
            "Event_Type": "identity_continuity",
            "Continuity_Class": "same_security",
            "Accounting_Status": "documented_not_executable",
            "Legacy_Event_Type": "ticker_or_name_change",
            "Review_Status": "approved",
            "Summary": "Synthetic documentary event outside the test window.",
        }], columns=SECURITY_EVENTS_COLUMNS),
        paths.security_identity.legs_csv: pd.DataFrame(
            columns=SECURITY_EVENT_LEGS_COLUMNS
        ),
        paths.security_identity.sources_csv: pd.DataFrame([{
            "Source_ID": "SRC-20300101-TEST-DOCUMENTARY",
            "Event_ID": "EVT-20300101-TEST-DOCUMENTARY",
            "Source_Role": "primary",
            "Publisher": "SEC",
            "Document_Date": "2030-01-01",
            "Source_URL": "https://www.sec.gov/Archives/example.htm",
            "Evidence_Claim": "Synthetic official evidence for an isolated test.",
            "Review_Status": "approved",
        }], columns=SECURITY_EVENT_SOURCES_COLUMNS),
        paths.security_identity.provider_mappings_csv: pd.DataFrame(
            columns=PROVIDER_SYMBOL_MAPPINGS_COLUMNS
        ),
        paths.security_identity.provider_row_equivalence_csv: pd.DataFrame(
            columns=PROVIDER_ROW_EQUIVALENCE_COLUMNS
        ),
    }
    manifests = []
    for artifact_path, frame in frames.items():
        frame.to_csv(
            artifact_path,
            index=False,
            lineterminator="\n",
        )
        manifests.append(ArtifactManifest.from_artifact(
            artifact_path,
            scope="shared",
            dataset="security_identity",
            origin=(
                ArtifactOrigin.MIGRATED
                if artifact_path == paths.security_identity.provider_mappings_csv
                else ArtifactOrigin.MANUAL
            ),
            artifact_path=artifact_path.name,
        ))
    write_manifests(paths.security_identity.manifest_csv, manifests)


def _make_raw_core_data(tmp_path) -> tuple[BacktestPaths, BacktestMarketConfig]:
    paths = BacktestPaths(tmp_path / "backtest")
    paths.raw_data_dir.mkdir(parents=True)
    paths.prices_csv.parent.mkdir(parents=True)
    config = BacktestMarketConfig(
        start_date=date(2014, 1, 31),
        end_date=date(2015, 2, 28),
    )

    _write_synthetic_membership_sources(paths, config.end_date)
    _write_synthetic_security_identity(paths)
    _write_synthetic_sector_history(
        paths,
        pd.date_range(config.start_date, config.end_date, freq="ME")[
            BACKTEST_WARMUP_MONTHS:-1
        ],
        {
            "AAA": "Information Technology",
            "BBB": "Financials",
        },
    )

    dates = pd.date_range(config.start_date, config.end_date, freq="ME")
    price_rows = []
    for offset, observation_date in enumerate(dates):
        for asset_id, base_price, volume in (
            ("AAA.O", 100.0, 1_000_000.0),
            ("BBB^A20", 50.0, 2_000_000.0),
        ):
            price_rows.append({
                "Date": observation_date - pd.Timedelta(days=1),
                "Price Close": base_price + offset,
                "Volume": volume + offset,
                "RIC": asset_id,
            })
    pd.DataFrame(price_rows).to_csv(paths.prices_csv, index=False)

    pd.DataFrame([
        {
            "Date": observation_date - pd.Timedelta(days=1),
            "SP500TR_Close": 100.0 + offset,
        }
        for offset, observation_date in enumerate(dates)
    ]).to_csv(paths.benchmark_raw_csv, index=False)
    paths.benchmark_provenance_dir.mkdir(parents=True, exist_ok=True)
    captured = "2026-08-27T00:00:00Z"
    benchmark_config = _benchmark_config(config)
    write_acquisition_statuses(
        paths.benchmark_acquisition_status_csv,
        (AcquisitionStatus(
            identity=AcquisitionIdentity(
                scope="backtest",
                dataset="benchmark",
                asset_id="SP500TR",
                provider="yahoo",
                provider_symbol="^SP500TR",
            ),
            status=ProviderStatus.OK,
            requested_start=benchmark_config.coverage_start_date.isoformat(),
            requested_end=benchmark_config.coverage_end_date.isoformat(),
            observation_count=len(dates),
            observation_start=dates.min().strftime("%Y-%m-%d"),
            observation_end=dates.max().strftime("%Y-%m-%d"),
            attempted_at_utc=captured,
            client="mock-yahoo",
            client_version="1",
            http_status="200",
        ),),
    )
    write_readiness(
        paths.benchmark_readiness_csv,
        (ReadinessRecord(
            scope="backtest",
            dataset="benchmark",
            asset_id="SP500TR",
            requirement_set="monthly_total_return_benchmark",
            required_count=len(dates),
            covered_count=len(dates),
            status=ReadinessStatus.COMPLETE,
            contributing_sources=("yahoo",),
            checked_at_utc=captured,
        ),),
    )
    write_manifests(
        paths.benchmark_artifact_manifest_csv,
        (
            ArtifactManifest.from_artifact(
                path,
                scope="backtest",
                dataset="benchmark",
                origin=ArtifactOrigin.DOWNLOADED,
                artifact_path=path.relative_to(paths.project_root).as_posix(),
                captured_at_utc=captured,
            )
            for path in (
                paths.benchmark_raw_csv,
                paths.benchmark_acquisition_status_csv,
                paths.benchmark_readiness_csv,
            )
        ),
    )
    return paths, config


def _benchmark_config(
    market_config: BacktestMarketConfig,
) -> BacktestBenchmarkConfig:
    return BacktestBenchmarkConfig(
        coverage_start_date=market_config.start_date,
        coverage_end_date=market_config.end_date,
    )


def _prepare_core_pipeline(paths: BacktestPaths, config: BacktestMarketConfig):
    """Run the same core finalization transaction as the CLI coordinator."""
    benchmark_config = _benchmark_config(config)
    prepare_core_data(config, paths, benchmark_config)
    write_preparation_manifest(paths, include_brinson=False)
    return validate_prepared_data(
        paths,
        config,
        benchmark_config,
        check_raw=True,
    )


def test_prepare_core_data_writes_only_deterministic_csvs(tmp_path):
    paths, config = _make_raw_core_data(tmp_path)
    result = prepare_core_data(config, paths, _benchmark_config(config))

    assert result is None
    assert not paths.preparation_manifest_csv.exists()

    write_preparation_manifest(paths, include_brinson=False)
    backtest_data = validate_prepared_data(
        paths,
        config,
        _benchmark_config(config),
        check_raw=True,
    )

    assert backtest_data.data_close.shape == (14, 2)
    assert backtest_data.pit_matrix.shape == (14, 2)
    assert backtest_data.asset_to_ticker == {"AAA.O": "AAA", "BBB^A20": "BBB"}
    assert sector_rows_asof(
        backtest_data.sector_assignments, "2015-01-31"
    )["GICS_Sector_Code"].to_dict() == {
        "AAA.O": "45",
        "BBB^A20": "40",
    }
    assignment_rows = backtest_data.sector_assignments.set_index("Asset_ID")
    assert assignment_rows.loc["AAA.O", "Sector"] == "Information Technology"
    assert assignment_rows.loc["BBB^A20", "Sector"] == "Financials"
    assert assignment_rows["Source_Type"].eq("Wikipedia").all()
    assert backtest_data.data_close.loc[pd.Timestamp("2014-01-31"), "AAA.O"] == 100.0
    assert backtest_data.data_close.loc[
        pd.Timestamp("2014-04-30"), "BBB^A20"
    ] == 53.0

    prepared_paths = [
        paths.prices_monthly_csv,
        paths.pit_membership_csv,
        paths.asset_metadata_csv,
        paths.sector_assignments_csv,
        paths.ticker_ric_resolution_csv,
        paths.security_identity_validation_csv,
        paths.security_events_prepared_csv,
        paths.security_event_legs_prepared_csv,
        paths.security_event_sources_prepared_csv,
        paths.security_event_crossing_audit_csv,
        paths.membership_coverage_audit_csv,
        paths.benchmark_csv,
        paths.preparation_manifest_csv,
    ]
    assert all(path.is_file() for path in prepared_paths)
    assert not list(paths.project_root.rglob("*.pkl"))
    assert not list(paths.project_root.rglob("*.sqlite"))

    prices = pd.read_csv(paths.prices_monthly_csv)
    assert tuple(prices.columns) == PREPARED_PRICE_COLUMNS
    assert len(prices) == 14 * 2
    metadata = pd.read_csv(paths.asset_metadata_csv)
    assert tuple(metadata.columns) == PREPARED_ASSET_METADATA_COLUMNS

    first_hashes = {path.name: file_sha256(path) for path in prepared_paths}
    prepare_core_data(config, paths, _benchmark_config(config))
    second_hashes = {path.name: file_sha256(path) for path in prepared_paths}
    assert second_hashes == first_hashes

    manifest = write_preparation_manifest(paths, include_brinson=False)
    assert set(manifest["Artifact"]) == {
        "fja_components",
        "fja_changes",
        "fja_intervals",
        "fja_source_manifest",
        "canonical_security_events",
        "canonical_security_event_legs",
        "canonical_security_event_sources",
        "provider_symbol_mappings",
        "provider_row_equivalence",
        "security_identity_manifest",
        "prices",
        "benchmark",
        "benchmark_acquisition_status",
        "benchmark_readiness",
        "benchmark_artifact_manifest",
        "sector_history_snapshots",
        "sector_history_notices",
        "prices_monthly",
        "pit_membership",
        "asset_metadata",
        "sector_assignments",
        "ticker_ric_resolution",
        "security_identity_validation",
        "security_events",
        "security_event_legs",
        "security_event_sources",
        "security_event_crossing_audit",
        "membership_coverage_audit",
        "benchmark_monthly",
        "price_basis",
    }


def test_core_preparation_rejects_failed_benchmark_provider_status(tmp_path):
    paths, config = _make_raw_core_data(tmp_path)
    status = read_acquisition_statuses(
        paths.benchmark_acquisition_status_csv
    )[0]
    write_acquisition_statuses(
        paths.benchmark_acquisition_status_csv,
        (replace(
            status,
            status=ProviderStatus.FAILED,
            observation_count=0,
            observation_start="",
            observation_end="",
            error_class="ValueError",
            error_message="provider request failed",
        ),),
    )
    manifests = read_manifests(paths.benchmark_artifact_manifest_csv)
    write_manifests(
        paths.benchmark_artifact_manifest_csv,
        (
            ArtifactManifest.from_artifact(
                paths.benchmark_acquisition_status_csv,
                scope="backtest",
                dataset="benchmark",
                origin=ArtifactOrigin.DOWNLOADED,
                artifact_path=paths.benchmark_acquisition_status_csv.relative_to(
                    paths.project_root
                ).as_posix(),
                captured_at_utc="2026-08-27T00:00:00Z",
            )
            if item.artifact_path.endswith("acquisition_status.csv")
            else item
            for item in manifests
        ),
    )

    with pytest.raises(ValueError, match="provider status.*not a successful"):
        preparation_builders.prepare_benchmark_monthly(
            _benchmark_config(config),
            paths,
        )


def test_analysis_loader_is_read_only_and_manifest_validated(tmp_path):
    paths, config = _make_raw_core_data(tmp_path)
    _prepare_core_pipeline(paths, config)
    before = {
        path.relative_to(paths.prepared_data_dir): file_sha256(path)
        for path in paths.prepared_data_dir.rglob("*.csv")
    }

    # Analysis must consume only prepared CSVs; raw inputs need not be present.
    paths.raw_data_dir.rename(paths.data_dir / "raw-not-available-to-analysis")
    loaded = validate_prepared_data(paths, config, _benchmark_config(config))
    after = {
        path.relative_to(paths.prepared_data_dir): file_sha256(path)
        for path in paths.prepared_data_dir.rglob("*.csv")
    }
    assert loaded.data_close.shape == (14, 2)
    assert after == before

    with paths.asset_metadata_csv.open("a") as file:
        file.write("\n")
    with pytest.raises(RuntimeError, match="stale or modified"):
        load_backtest_data(config, paths)


def test_missing_prepared_data_has_actionable_error(tmp_path):
    paths, config = _make_raw_core_data(tmp_path)
    with pytest.raises(RuntimeError, match="python -m backtest.prepare"):
        load_backtest_data(config, paths)


def test_manifest_ignores_regenerable_acquisition_runtime_state(tmp_path):
    paths, config = _make_raw_core_data(tmp_path)
    _prepare_core_pipeline(paths, config)
    cache_path = paths.project_root / "runtime/acquisition/yfinance/cookies.db"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"first cache state")

    first = write_preparation_manifest(paths, include_brinson=True)
    cache_path.write_bytes(b"different cache state")
    second = write_preparation_manifest(paths, include_brinson=True)

    pd.testing.assert_frame_equal(first, second)
    assert cache_path.relative_to(paths.project_root).as_posix() not in set(
        second["Relative_Path"]
    )


def test_core_preparation_does_not_modify_raw_or_runtime_cache_bytes(tmp_path):
    paths, config = _make_raw_core_data(tmp_path)
    cache_path = paths.project_root / "runtime/acquisition/yfinance/cookies.db"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"persistent acquisition cache")
    raw_paths = [
        paths.prices_csv,
        *paths.sector_history.acquisition_artifacts,
        paths.sector_history.artifact_manifest_csv,
        paths.benchmark_raw_csv,
        paths.benchmark_acquisition_status_csv,
        paths.benchmark_readiness_csv,
        paths.benchmark_artifact_manifest_csv,
        paths.security_identity.events_csv,
        paths.security_identity.legs_csv,
        paths.security_identity.sources_csv,
        paths.security_identity.provider_mappings_csv,
        paths.security_identity.provider_row_equivalence_csv,
        paths.security_identity.manifest_csv,
        paths.membership.components_csv,
        paths.membership.changes_csv,
        paths.membership.intervals_csv,
        paths.membership.manifest_csv,
        cache_path,
    ]
    before = {path: file_sha256(path) for path in raw_paths}

    prepare_core_data(config, paths, _benchmark_config(config))

    assert {path: file_sha256(path) for path in raw_paths} == before


def test_raw_freshness_is_optional_for_analysis_but_available_to_check(tmp_path):
    paths, config = _make_raw_core_data(tmp_path)
    _prepare_core_pipeline(paths, config)
    with paths.sector_history.snapshots_csv.open("a") as file:
        file.write("\n")

    # Analysis remains a prepared-only consumer.
    assert load_backtest_data(config, paths).data_close.shape == (14, 2)
    with pytest.raises(RuntimeError, match="stale or modified"):
        load_backtest_data(config, paths, check_raw=True)
    with pytest.raises(RuntimeError, match="stale or modified"):
        validate_preparation_manifest(paths, check_raw=True)


def test_all_stage_finalizes_core_before_brinson_and_then_full_manifest(
    monkeypatch,
):
    calls = []
    backtest_data = object()
    paths = SimpleNamespace(shares="share_paths")
    config = SimpleNamespace(
        market="market",
        benchmark="benchmark",
        paths=paths,
        brinson="brinson",
    )
    monkeypatch.setattr(
        preparation_pipeline,
        "DEFAULT_CONFIG",
        config,
    )
    monkeypatch.setattr(
        preparation_pipeline,
        "prepare_core_data",
        lambda market, paths, benchmark: calls.append(
            ("build_core", market, paths, benchmark)
        ),
    )
    monkeypatch.setattr(
        preparation_pipeline,
        "write_preparation_manifest",
        lambda paths, *, include_brinson: calls.append(
            ("write_manifest", include_brinson)
        ),
    )
    monkeypatch.setattr(
        preparation_pipeline,
        "validate_prepared_data",
        lambda paths, market, benchmark, **kwargs: calls.append(
            (
                "load_validated_data",
                paths,
                market,
                benchmark,
                kwargs["include_brinson"],
                kwargs["check_raw"],
            )
        )
        or backtest_data,
    )
    monkeypatch.setattr(
        preparation_pipeline,
        "prepare_shares_dataset",
        lambda value, received_config: calls.append(
            ("build_brinson", value, received_config)
        ),
    )
    monkeypatch.setattr(
        preparation_pipeline,
        "load_prepared_shares",
        lambda received_paths, received_config: calls.append(
            ("load_brinson", received_paths, received_config)
        ),
    )

    preparation_pipeline.run_preparation("all")

    assert calls == [
        ("build_core", "market", paths, "benchmark"),
        ("write_manifest", False),
        (
            "load_validated_data",
            paths,
            "market",
            "benchmark",
            False,
            True,
        ),
        ("build_brinson", backtest_data, config),
        ("write_manifest", True),
        (
            "load_validated_data",
            paths,
            "market",
            "benchmark",
            True,
            True,
        ),
        ("load_brinson", "share_paths", "brinson"),
    ]


def test_prepared_data_loader_returns_dataset_and_requests_manifest_once(
    monkeypatch,
):
    calls = []
    backtest_data = object()
    paths = SimpleNamespace(benchmark_csv="benchmark.csv")

    monkeypatch.setattr(
        preparation_builders,
        "load_backtest_data",
        lambda market, received_paths, **kwargs: calls.append(
            (
                "manifest_and_dataset",
                market,
                received_paths,
                kwargs["include_brinson"],
                kwargs["check_raw"],
            )
        )
        or backtest_data,
    )
    monkeypatch.setattr(
        preparation_builders,
        "_validate_prepared_benchmark",
        lambda path, benchmark: calls.append(
            ("benchmark_coverage", path, benchmark)
        ),
    )

    actual = validate_prepared_data(
        paths,
        "market",
        "benchmark",
        include_brinson=True,
        check_raw=True,
    )

    assert actual is backtest_data
    assert calls == [
        ("manifest_and_dataset", "market", paths, True, True),
        ("benchmark_coverage", "benchmark.csv", "benchmark"),
    ]


@pytest.mark.parametrize("include_brinson", (False, True))
def test_prepared_data_loader_requests_exact_manifest_scope_once(
    tmp_path,
    monkeypatch,
    include_brinson,
):
    paths, market_config = _make_raw_core_data(tmp_path)
    benchmark_config = _benchmark_config(market_config)
    _prepare_core_pipeline(paths, market_config)
    calls = []
    monkeypatch.setattr(
        data_loading,
        "validate_preparation_manifest",
        lambda **kwargs: calls.append(
            (
                kwargs["paths"],
                kwargs["include_brinson"],
                kwargs["check_raw"],
                frozenset(kwargs["validate_raw_artifacts"]),
            )
        ),
    )

    loaded = validate_prepared_data(
        paths,
        market_config,
        benchmark_config,
        include_brinson=include_brinson,
        check_raw=True,
    )

    assert loaded.data_close.shape == (14, 2)
    assert calls == [(paths, include_brinson, True, frozenset())]


@pytest.mark.parametrize(
    ("stage", "expected_include_brinson"),
    (("core", False), ("brinson", True), ("all", True)),
)
def test_preparation_checks_load_the_requested_manifest_once(
    monkeypatch,
    stage,
    expected_include_brinson,
):
    calls = []
    paths = SimpleNamespace(shares="share_paths")
    config = SimpleNamespace(
        market="market",
        benchmark="benchmark",
        paths=paths,
        brinson="brinson",
    )
    monkeypatch.setattr(preparation_pipeline, "DEFAULT_CONFIG", config)
    monkeypatch.setattr(
        preparation_pipeline,
        "validate_prepared_data",
        lambda received_paths, market, benchmark, **kwargs: calls.append(
            (
                received_paths,
                market,
                benchmark,
                kwargs["include_brinson"],
                kwargs["check_raw"],
            )
        ),
    )
    monkeypatch.setattr(
        preparation_pipeline,
        "load_prepared_shares",
        lambda *args: None,
    )

    preparation_pipeline.run_preparation(stage, check=True)

    assert calls == [
        (paths, "market", "benchmark", expected_include_brinson, True)
    ]


def test_brinson_only_preparation_cannot_rebless_stale_core_raw(
    tmp_path,
    monkeypatch,
):
    paths, config = _make_raw_core_data(tmp_path)
    _prepare_core_pipeline(paths, config)
    with paths.sector_history.snapshots_csv.open("a") as file:
        file.write("\n")

    reached = []
    monkeypatch.setattr(
        preparation_pipeline,
        "DEFAULT_CONFIG",
        SimpleNamespace(
            market=config,
            benchmark=_benchmark_config(config),
            paths=paths,
        ),
    )
    monkeypatch.setattr(
        preparation_pipeline,
        "prepare_shares_dataset",
        lambda backtest_data: reached.append("brinson"),
    )
    monkeypatch.setattr(
        preparation_pipeline,
        "write_preparation_manifest",
        lambda *args, **kwargs: reached.append("manifest"),
    )

    with pytest.raises(RuntimeError, match="stale or modified"):
        preparation_pipeline.run_preparation("brinson")

    assert reached == []


def test_core_preparation_rejects_missing_point_in_time_sector_assignment(tmp_path):
    paths, config = _make_raw_core_data(tmp_path)
    snapshots = pd.read_csv(paths.sector_history.snapshots_csv)
    snapshots = snapshots.loc[snapshots["Wikipedia_Ticker"].ne("BBB")]
    snapshots.to_csv(paths.sector_history.snapshots_csv, index=False)
    _write_sector_history_manifest(paths)

    with pytest.raises(ValueError, match="No causal sector source for BBB"):
        prepare_core_data(config, paths, _benchmark_config(config))
