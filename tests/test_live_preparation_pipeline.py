"""Offline data-contract tests for the historical live strategy."""

from pathlib import Path
from shutil import copy2
from types import SimpleNamespace

import pandas as pd
import pytest
from portfolio_core.artifacts import (
    file_sha256,
    write_manifests,
)
from data_acquisition.contracts import (
    AcquisitionIdentity,
    AcquisitionRequest,
    AcquisitionStatus,
    ProviderStatus,
    ReadinessRecord,
    ReadinessStatus,
    read_readiness,
    write_readiness as write_readiness_records,
    write_acquisition_statuses,
    write_readiness,
)
from live import acquisition_execution as live_acquisition
from live import preparation_builders as live_preparation
from live.config import DEFAULT_CONFIG, LiveConfig
from live.corporate_action_policy import load_live_corporate_action_bundle
import live.preparation_pipeline as live_preparation_pipeline
from live.price_coverage import (
    PRICE_REQUIREMENT_SET,
    build_price_requirements,
    build_coverage_ledger,
)
from live.paths import LivePaths
from live.preparation_builders import (
    _load_raw_shares,
    _load_status,
    prepare_core_data,
)
from live.preparation_artifacts import (
    validate_preparation_manifest,
    write_preparation_manifest,
)
from live.strategy_universe import (
    DOWNLOAD_PRICES_COMMAND,
    DOWNLOAD_SHARES_COMMAND,
    METADATA_COLUMNS,
    decision_schedule,
    execution_valuation_dates,
    execution_valuation_membership_requirements,
    evaluation_periods,
    load_validated_strategy_universe,
    prepared_membership_asof,
    sector_assignment_requirements,
)
from portfolio_core.sp500_membership import (
    validate_membership_sources,
)
from portfolio_core.provider_identity import YahooIdentityResolver
from portfolio_core.io import atomic_write_dataframe
from portfolio_core.security_identity import load_security_identity_bundle
from portfolio_core.shares import (
    select_canonical_share_observations_asof,
    validate_raw_shares,
)
from _sector_test_helpers import (
    _write_synthetic_sector_history,
)
from _membership_test_helpers import _write_membership_source_manifest


def test_execution_valuation_dates_normalizes_sorts_and_deduplicates():
    schedule = pd.DataFrame({
        "Execution_Date": [
            "2026-03-02 15:45",
            "2026-02-13 09:30",
        ],
        "Valuation_End": [
            "2026-04-01 23:59",
            "2026-03-02 08:00",
        ],
    })

    assert execution_valuation_dates(schedule) == (
        pd.Timestamp("2026-02-13"),
        pd.Timestamp("2026-03-02"),
        pd.Timestamp("2026-04-01"),
    )


def test_execution_valuation_membership_requirements_are_causal_and_ordered():
    schedule = pd.DataFrame({
        "Execution_Date": ["2026-03-15", "2026-01-15"],
        "Valuation_End": ["2026-04-15", "2026-02-15"],
    })
    membership = pd.DataFrame([
        {"Effective_Date": "2026-01-01", "Asset_ID": "BBB"},
        {"Effective_Date": "2026-01-01", "Asset_ID": "AAA"},
        {"Effective_Date": "2026-02-01", "Asset_ID": "CCC"},
        {"Effective_Date": "2026-02-01", "Asset_ID": "BBB"},
        {"Effective_Date": "2026-04-01", "Asset_ID": "EEE"},
        {"Effective_Date": "2026-04-01", "Asset_ID": "DDD"},
    ])

    requirements = execution_valuation_membership_requirements(
        schedule,
        membership,
    )

    assert requirements == (
        (pd.Timestamp("2026-01-15"), "AAA"),
        (pd.Timestamp("2026-01-15"), "BBB"),
        (pd.Timestamp("2026-02-15"), "BBB"),
        (pd.Timestamp("2026-02-15"), "CCC"),
        (pd.Timestamp("2026-03-15"), "BBB"),
        (pd.Timestamp("2026-03-15"), "CCC"),
        (pd.Timestamp("2026-04-15"), "DDD"),
        (pd.Timestamp("2026-04-15"), "EEE"),
    )
    assert requirements == tuple(sorted(set(requirements)))


def test_strategy_universe_matches_prepared_schedule_membership_and_metadata():
    schedule, membership, metadata = load_validated_strategy_universe(
        DEFAULT_CONFIG
    )

    assert schedule["Rebalance_ID"].tolist() == ["R1", "R2", "R3"]
    assert membership.equals(
        membership.sort_values(
            ["Effective_Date", "Asset_ID"], kind="stable"
        ).reset_index(drop=True)
    )
    assert tuple(metadata.columns) == METADATA_COLUMNS
    assert metadata["Asset_ID"].tolist() == sorted(metadata["Asset_ID"])
    assert metadata["Source_Ticker"].equals(metadata["Asset_ID"])
    assert metadata.set_index("Asset_ID").loc[["ABC", "COR"], "Yahoo_Ticker"].tolist() == ["COR", "COR"]
    dated = membership.merge(metadata, on="Asset_ID", validate="many_to_one")
    assert not dated.duplicated(["Effective_Date", "Yahoo_Ticker"]).any()
    assert metadata.set_index("Asset_ID").loc[
        ["BK", "SATS"], "Yahoo_Ticker"
    ].to_dict() == {"BK": "BNY", "SATS": "ECHO"}
    prepared_schedule = pd.read_csv(
        DEFAULT_CONFIG.paths.decision_schedule_csv,
        parse_dates=[
            "Membership_Effective_Date",
            "Signal_Cutoff",
            "Sizing_Date",
            "Execution_Date",
            "Valuation_End",
        ],
    )
    prepared_membership = pd.read_csv(
        DEFAULT_CONFIG.paths.pit_membership_csv,
        parse_dates=["Effective_Date"],
    )
    prepared_metadata = pd.read_csv(
        DEFAULT_CONFIG.paths.asset_metadata_csv,
        keep_default_na=False,
    )
    pd.testing.assert_frame_equal(schedule, prepared_schedule, check_dtype=False)
    pd.testing.assert_frame_equal(
        membership,
        prepared_membership,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(metadata, prepared_metadata)


def test_materialized_live_brinson_outputs_match_canonical_preparation(
    tmp_path,
    monkeypatch,
):
    paths = DEFAULT_CONFIG.paths
    required = (
        paths.shares.raw_shares_csv,
        paths.shares.readiness_csv,
        paths.shares.prepared_shares_csv,
        paths.raw_benchmark_csv,
        paths.raw_benchmark_status_csv,
        paths.raw_benchmark_readiness_csv,
        paths.prepared_benchmark_csv,
        paths.preparation_manifest_csv,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("complete ignored live share bundle is not materialized")

    manifest_before = paths.preparation_manifest_csv.read_bytes()
    raw = validate_raw_shares(
        pd.read_csv(paths.shares.raw_shares_csv, keep_default_na=False)
    )
    readiness = read_readiness(paths.shares.readiness_csv)
    checked_at = {record.checked_at_utc for record in readiness}
    assert len(checked_at) == 1

    captured_readiness = {}

    def capture_readiness(path, records):
        captured_readiness[Path(path)] = list(records)

    monkeypatch.setattr(live_acquisition, "write_readiness", capture_readiness)
    live_acquisition._write_shares_readiness(raw, checked_at.pop())
    generated_readiness = tmp_path / "readiness.csv"
    write_readiness_records(
        generated_readiness,
        captured_readiness[paths.shares.readiness_csv],
    )
    assert generated_readiness.read_bytes() == paths.shares.readiness_csv.read_bytes()

    captured_prepared = {}

    def capture_prepared(frame, path, **kwargs):
        captured_prepared[Path(path)] = (frame.copy(), kwargs)

    monkeypatch.setattr(live_preparation, "atomic_write_dataframe", capture_prepared)
    shares, benchmark = live_preparation.prepare_brinson_data(DEFAULT_CONFIG)
    schedule, membership, _ = load_validated_strategy_universe(DEFAULT_CONFIG)
    expected_requirements = execution_valuation_membership_requirements(
        schedule,
        membership,
        evaluation_start=DEFAULT_CONFIG.market.competition_start,
    )
    assert tuple(
        shares[["Date", "Asset_ID"]].itertuples(index=False, name=None)
    ) == expected_requirements
    pd.testing.assert_frame_equal(
        shares,
        select_canonical_share_observations_asof(raw, expected_requirements),
    )
    generated_shares = tmp_path / "shares_outstanding_asof.csv"
    frame, kwargs = captured_prepared[paths.shares.prepared_shares_csv]
    atomic_write_dataframe(frame, generated_shares, **kwargs)
    assert generated_shares.read_bytes() == paths.shares.prepared_shares_csv.read_bytes()
    assert file_sha256(generated_shares) == (
        "9a921e5202440c9c0d81a858f3febc8eb5618a141b03ab07ef887aac4e03fe70"
    )
    assert len(shares) == 2_515

    generated_benchmark = tmp_path / "benchmark_daily.csv"
    frame, kwargs = captured_prepared[paths.prepared_benchmark_csv]
    atomic_write_dataframe(frame, generated_benchmark, **kwargs)
    assert generated_benchmark.read_bytes() == paths.prepared_benchmark_csv.read_bytes()
    assert file_sha256(generated_benchmark) == (
        "ac65f39996ec44eb93a6121d91aa6346e7100a5de758dcd71d8a2b871d744c31"
    )
    assert len(benchmark) == 58

    validate_preparation_manifest(paths, include_brinson=True, check_raw=True)
    manifest = pd.read_csv(paths.preparation_manifest_csv).set_index("Artifact")
    assert manifest.at["shares_readiness", "SHA256"] == file_sha256(
        paths.shares.readiness_csv
    )
    assert manifest.at["shares_prepared", "SHA256"] == file_sha256(
        paths.shares.prepared_shares_csv
    )
    assert manifest.at["benchmark_prepared", "SHA256"] == file_sha256(
        paths.prepared_benchmark_csv
    )
    assert paths.preparation_manifest_csv.read_bytes() == manifest_before


def test_brinson_preparation_preserves_missing_benchmark_boundary_error(
    tmp_path,
    monkeypatch,
):
    paths = LivePaths(tmp_path / "live")
    config = LiveConfig(paths=paths)
    schedule = decision_schedule(config)
    membership = pd.DataFrame({
        "Effective_Date": pd.to_datetime(["2026-01-14"]),
        "Asset_ID": ["AAA"],
    })
    metadata = pd.DataFrame({"Asset_ID": ["AAA"]})
    raw_shares = validate_raw_shares(pd.DataFrame([{
        "Asset_ID": "AAA",
        "Provider_Symbol": "AAA",
        "Effective_Start": "",
        "Effective_End": "",
        "Date": "2026-01-30",
        "Observation_Sequence": 0,
        "Shares_Outstanding": 100.0,
    }]))

    # Acquisition and identity loading have separate tests; exercise the real
    # as-of share selection and benchmark validation with only temporary data.
    monkeypatch.setattr(
        live_preparation,
        "load_validated_strategy_universe",
        lambda _config: (schedule, membership, metadata),
    )
    monkeypatch.setattr(live_preparation, "_load_status", lambda *_args: [])
    monkeypatch.setattr(live_preparation, "_load_readiness", lambda *_args: None)
    monkeypatch.setattr(
        live_preparation,
        "_load_raw_shares",
        lambda *_args, **_kwargs: raw_shares,
    )
    paths.raw_benchmark_csv.parent.mkdir(parents=True)
    pd.DataFrame({
        "Date": ["2026-02-13", "2026-03-02", "2026-04-01", "2026-05-01"],
        "Open": [100.0, 101.0, 102.0, 103.0],
        "Close": [100.5, 101.5, 102.5, 103.5],
    }).to_csv(paths.raw_benchmark_csv, index=False)

    with pytest.raises(RuntimeError) as exc_info:
        live_preparation.prepare_brinson_data(config)

    assert str(exc_info.value) == (
        "Benchmark lacks prices for 2026-05-06. Run "
        "`python -m data_acquisition.acquire live benchmark`."
    )
    assert not paths.prepared_data_dir.exists()


def test_sector_requirements_do_not_preassign_corporate_action_successors():
    schedule = pd.DataFrame([
        {
            "Membership_Effective_Date": "2026-01-01",
            "Signal_Cutoff": "2026-01-20",
            "Execution_Date": "2026-01-31",
            "Execution_Field": "Open",
            "Valuation_End": "2026-02-28",
            "Valuation_Field": "Open",
        },
        {
            "Membership_Effective_Date": "2026-01-01",
            "Signal_Cutoff": "2026-02-20",
            "Execution_Date": "2026-02-28",
            "Execution_Field": "Open",
            "Valuation_End": "2026-03-31",
            "Valuation_Field": "Open",
        },
    ])
    membership = pd.DataFrame([
        {"Effective_Date": "2026-01-01", "Asset_ID": "AAA"},
        {"Effective_Date": "2026-01-01", "Asset_ID": "BBB"},
    ])
    events = pd.DataFrame([
        {
            "Event_ID": "EVT-AAA-NEW",
            "Effective_Date": "2026-02-15",
            "Review_Status": "approved",
        },
        {
            "Event_ID": "EVT-NEW-NEXT",
            "Effective_Date": "2026-02-16",
            "Review_Status": "approved",
        },
    ])
    legs = pd.DataFrame([
        {
            "Event_ID": "EVT-AAA-NEW",
            "Leg_Order": 1,
            "From_Asset_ID": "AAA",
            "To_Asset_ID": "NEW",
            "Leg_Type": "stock",
            "Consumes_From_Position": True,
            "Review_Status": "approved",
        },
        {
            "Event_ID": "EVT-NEW-NEXT",
            "Leg_Order": 1,
            "From_Asset_ID": "NEW",
            "To_Asset_ID": "NEXT",
            "Leg_Type": "relabel",
            "Consumes_From_Position": True,
            "Review_Status": "approved",
        },
    ])
    requirements = sector_assignment_requirements(
        schedule,
        membership,
        corporate_action_events=events,
        corporate_action_legs=legs,
    )
    pairs = set(
        requirements[["As_Of_Date", "Asset_ID"]].itertuples(
            index=False,
            name=None,
        )
    )

    assert (pd.Timestamp("2026-02-28"), "NEW") not in pairs
    assert (pd.Timestamp("2026-02-28"), "NEXT") not in pairs
    assert (pd.Timestamp("2026-01-31"), "NEW") not in pairs


@pytest.mark.parametrize("stage", ("core", "brinson", "all"))
def test_live_preparation_stages_preserve_validation_order_and_scope(
    monkeypatch,
    stage,
):
    calls = []
    config = SimpleNamespace(paths="paths")
    monkeypatch.setattr(live_preparation_pipeline, "DEFAULT_CONFIG", config)
    monkeypatch.setattr(
        live_preparation_pipeline,
        "prepare_core_data",
        lambda value: calls.append(("build_market", value)),
    )
    monkeypatch.setattr(
        live_preparation_pipeline,
        "prepare_brinson_data",
        lambda value: calls.append(("build_brinson", value)),
    )
    monkeypatch.setattr(
        live_preparation_pipeline,
        "write_preparation_manifest",
        lambda paths, *, include_brinson: calls.append(
            ("write_manifest", paths, include_brinson)
        ),
    )
    monkeypatch.setattr(
        live_preparation_pipeline,
        "validate_preparation_manifest",
        lambda paths, *, include_brinson, check_raw: calls.append(
            ("validate_manifest", paths, include_brinson, check_raw)
        ),
    )
    monkeypatch.setattr(
        live_preparation_pipeline,
        "validate_prepared_sector_contract",
        lambda paths: calls.append(("validate_sectors", paths)),
    )

    live_preparation_pipeline.run_preparation(stage)

    expected = []
    if stage in {"core", "all"}:
        expected.extend([
            ("build_market", config),
            ("write_manifest", "paths", False),
        ])
    expected.extend([
        ("validate_manifest", "paths", False, True),
        ("validate_sectors", "paths"),
    ])
    if stage in {"brinson", "all"}:
        expected.extend([
            ("build_brinson", config),
            ("write_manifest", "paths", True),
            ("validate_manifest", "paths", True, True),
        ])
    assert calls == expected


@pytest.mark.parametrize("tampering", ("manifest", "sectors"))
def test_brinson_prebuild_tampering_aborts_before_generation(
    monkeypatch,
    tampering,
):
    calls = []
    config = SimpleNamespace(paths="paths")
    monkeypatch.setattr(live_preparation_pipeline, "DEFAULT_CONFIG", config)

    def validate_manifest(paths, *, include_brinson, check_raw):
        calls.append(("manifest", paths, include_brinson, check_raw))
        if tampering == "manifest":
            raise RuntimeError("manifest tampering")

    def validate_sectors(paths):
        calls.append(("sectors", paths))
        if tampering == "sectors":
            raise RuntimeError("sector tampering")

    monkeypatch.setattr(
        live_preparation_pipeline,
        "validate_preparation_manifest",
        validate_manifest,
    )
    monkeypatch.setattr(
        live_preparation_pipeline,
        "validate_prepared_sector_contract",
        validate_sectors,
    )
    monkeypatch.setattr(
        live_preparation_pipeline,
        "prepare_brinson_data",
        lambda value: calls.append(("build_brinson", value)),
    )
    monkeypatch.setattr(
        live_preparation_pipeline,
        "write_preparation_manifest",
        lambda paths, *, include_brinson: calls.append(
            ("write_manifest", paths, include_brinson)
        ),
    )

    message = (
        "manifest tampering" if tampering == "manifest" else "sector tampering"
    )
    with pytest.raises(RuntimeError, match=message):
        live_preparation_pipeline.run_preparation("brinson")

    expected = [("manifest", "paths", False, True)]
    if tampering == "sectors":
        expected.append(("sectors", "paths"))
    assert calls == expected


@pytest.mark.parametrize(
    ("stage", "include_brinson"),
    (("core", False), ("brinson", True), ("all", True)),
)
def test_live_preparation_checks_validate_requested_scope_once(
    monkeypatch,
    stage,
    include_brinson,
):
    calls = []
    config = SimpleNamespace(paths="paths")
    monkeypatch.setattr(live_preparation_pipeline, "DEFAULT_CONFIG", config)
    monkeypatch.setattr(
        live_preparation_pipeline,
        "validate_preparation_manifest",
        lambda paths, **kwargs: calls.append(("manifest", paths, kwargs)),
    )
    monkeypatch.setattr(
        live_preparation_pipeline,
        "validate_prepared_sector_contract",
        lambda paths: calls.append(("sectors", paths)),
    )

    live_preparation_pipeline.run_preparation(stage, check=True)

    assert calls == [
        (
            "manifest",
            "paths",
            {"include_brinson": include_brinson, "check_raw": True},
        ),
        ("sectors", "paths"),
    ]


def _copy_canonical_action_inputs(paths: LivePaths) -> None:
    from live.monthly_history import OBSERVATION_COLUMNS, DIVIDEND_COLUMNS
    from portfolio_core.artifacts import ArtifactManifest, ArtifactOrigin

    # Synthetic provider fixtures have no reviewed monthly gaps. Supply an
    # explicit empty evidence ledger rather than copying production repairs.
    paths.raw_monthly_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=OBSERVATION_COLUMNS).to_csv(paths.raw_monthly_dir / "observations.csv", index=False)
    pd.DataFrame(columns=DIVIDEND_COLUMNS).to_csv(paths.raw_monthly_dir / "dividends.csv", index=False)
    pd.DataFrame(columns=["Check"]).to_csv(paths.raw_monthly_dir / "basis_checks.csv", index=False)
    write_manifests(paths.raw_monthly_dir / "artifact_manifest.csv", [
        ArtifactManifest.from_artifact(paths.raw_monthly_dir / name, scope="live",
            dataset="monthly_history", origin=ArtifactOrigin.MANUAL, artifact_path=name)
        for name in ("observations.csv", "dividends.csv", "basis_checks.csv")
    ])
    # Empty monthly evidence needs only the source schema, never market history.
    destination = paths.project_root / "data/shared/supplied/reuters/SP500_Full_2014_2026_Cleaned.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["Date", "RIC", "Price Close"]).to_csv(destination, index=False)
    paths.security_identity.directory.mkdir(parents=True, exist_ok=True)
    for source in DEFAULT_CONFIG.paths.security_identity.directory.glob("*.csv"):
        copy2(source, paths.security_identity.directory / source.name)
    paths.raw_provenance_dir.mkdir(parents=True, exist_ok=True)
    for source, destination in (
        (
            DEFAULT_CONFIG.paths.raw_corporate_action_policy_csv,
            paths.raw_corporate_action_policy_csv,
        ),
        (
            DEFAULT_CONFIG.paths.raw_corporate_action_policy_manifest_csv,
            paths.raw_corporate_action_policy_manifest_csv,
        ),
    ):
        copy2(source, destination)


def _copy_authoritative_supplement(paths: LivePaths) -> None:
    paths.raw_price_supplemental_dir.mkdir(parents=True, exist_ok=True)
    copy2(
        DEFAULT_CONFIG.paths.raw_price_supplemental_csv,
        paths.raw_price_supplemental_csv,
    )
    copy2(
        DEFAULT_CONFIG.paths.raw_price_supplemental_manifest_csv,
        paths.raw_price_supplemental_manifest_csv,
    )


def test_reviewed_schedule_uses_causal_memberships():
    history = validate_membership_sources(
        DEFAULT_CONFIG.paths.membership,
        required_through=DEFAULT_CONFIG.market.competition_end,
    )
    schedule = decision_schedule(DEFAULT_CONFIG)

    expected = pd.to_datetime([
        "2026-02-09", "2026-03-23", "2026-04-09"
    ]).tolist()
    actual = []
    for cutoff in schedule["Signal_Cutoff"]:
        actual.append(history.loc[history["date"].le(cutoff), "date"].iloc[-1])
    assert actual == expected

    # The first post-competition change cannot leak into the final valuation.
    assert history.loc[history["date"].gt(pd.Timestamp("2026-04-09")), "date"].iloc[0] == pd.Timestamp("2026-05-07")

    expected_changes = {
        "2026-01-14": ({"MRSH"}, {"MMC"}),
        "2026-02-09": ({"CIEN"}, {"DAY"}),
        "2026-03-23": ({"COHR", "LITE", "SATS", "VRT"}, {"LW", "MOH", "MTCH", "PAYC"}),
        "2026-04-09": ({"CASY"}, {"HOLX"}),
    }
    for effective_date, (expected_add, expected_remove) in expected_changes.items():
        effective = pd.Timestamp(effective_date)
        current = set(history.loc[history["date"].eq(effective), "members"].iloc[0])
        previous = set(history.loc[history["date"].lt(effective), "members"].iloc[-1])
        assert len(current) == 503
        assert current - previous == expected_add
        assert previous - current == expected_remove


def test_prepared_membership_asof_selects_latest_complete_snapshot():
    membership = pd.DataFrame([
        {"Effective_Date": "2026-01-01", "Asset_ID": "AAA"},
        {"Effective_Date": "2026-01-01", "Asset_ID": "BBB"},
        {"Effective_Date": "2026-02-01", "Asset_ID": "CCC"},
        {"Effective_Date": "2026-02-01", "Asset_ID": "DDD"},
    ])

    assert prepared_membership_asof(membership, "2026-01-31 23:59") == {
        "AAA",
        "BBB",
    }
    assert prepared_membership_asof(membership, "2026-02-15") == {
        "CCC",
        "DDD",
    }


@pytest.mark.parametrize(
    ("membership", "cutoff", "message"),
    [
        (
            pd.DataFrame({"Effective_Date": ["2026-01-01"]}),
            "2026-01-31",
            "missing columns",
        ),
        (
            pd.DataFrame({
                "Effective_Date": ["2026-02-01"],
                "Asset_ID": ["AAA"],
            }),
            "2026-01-31",
            "No prepared constituent membership",
        ),
        (
            pd.DataFrame({
                "Effective_Date": ["2026-01-01"],
                "Asset_ID": [""],
            }),
            "2026-01-31",
            "Prepared membership is invalid",
        ),
        (
            pd.DataFrame({
                "Effective_Date": ["2026-01-01", "2026-01-01"],
                "Asset_ID": ["AAA", "AAA"],
            }),
            "2026-01-31",
            "Prepared membership is invalid",
        ),
    ],
)
def test_prepared_membership_asof_rejects_invalid_contracts(
    membership,
    cutoff,
    message,
):
    with pytest.raises(ValueError, match=message):
        prepared_membership_asof(membership, cutoff)
    schedule = pd.DataFrame({
        "Execution_Date": [cutoff],
        "Valuation_End": [cutoff],
    })
    with pytest.raises(ValueError, match=message):
        execution_valuation_membership_requirements(schedule, membership)


def test_share_class_aliases_are_explicit():
    resolver = YahooIdentityResolver(
        load_security_identity_bundle(DEFAULT_CONFIG.paths.project_root), scope="live",
    )

    def source_to_yahoo_ticker(ticker, *, as_of=None):
        return resolver.resolve(ticker, as_of=as_of, purpose="historical_prices").provider_symbol

    assert source_to_yahoo_ticker("BRK.B") == "BRK-B"
    assert source_to_yahoo_ticker("BF.B") == "BF-B"
    assert source_to_yahoo_ticker("AAPL") == "AAPL"
    assert source_to_yahoo_ticker("BK", as_of="2026-05-06") == "BNY"
    assert source_to_yahoo_ticker("BNY", as_of="2026-05-21") == "BNY"
    assert source_to_yahoo_ticker("SATS", as_of="2026-05-06") == "ECHO"
    assert source_to_yahoo_ticker("ECHO", as_of="2026-06-24") == "ECHO"
    assert source_to_yahoo_ticker("CTRA", as_of="2026-05-06") == "CTRA"
    with pytest.raises(ValueError, match="no reviewed Yahoo symbol effective"):
        source_to_yahoo_ticker("BK", as_of="2026-05-21")
    with pytest.raises(ValueError, match="no reviewed Yahoo symbol effective"):
        source_to_yahoo_ticker("BNY", as_of="2026-05-20")
    with pytest.raises(ValueError, match="no reviewed Yahoo symbol effective"):
        source_to_yahoo_ticker("SATS", as_of="2026-06-24")
    with pytest.raises(ValueError, match="no reviewed Yahoo symbol effective"):
        source_to_yahoo_ticker("ECHO", as_of="2026-06-23")


def test_point_in_time_sector_build_is_causal_and_notice_filled():
    schedule, membership, metadata = load_validated_strategy_universe(DEFAULT_CONFIG)
    assignments = live_preparation.prepare_sector_assignments(
        DEFAULT_CONFIG,
        schedule=schedule,
        membership=membership,
        actions=load_live_corporate_action_bundle(DEFAULT_CONFIG.paths),
    )
    assert not assignments["Resolution_Method"].str.contains(
        "carry_forward", case=False, regex=False
    ).any()
    deletion_exit_pairs = {
        (pd.Timestamp("2026-02-13"), "DAY"),
        (pd.Timestamp("2026-04-01"), "LW"),
        (pd.Timestamp("2026-04-01"), "MOH"),
        (pd.Timestamp("2026-04-01"), "MTCH"),
        (pd.Timestamp("2026-04-01"), "PAYC"),
        (pd.Timestamp("2026-05-04"), "HOLX"),
    }
    assignment_pair_series = pd.Series(
        list(
            assignments[["As_Of_Date", "Asset_ID"]].itertuples(
                index=False, name=None
            )
        ),
        index=assignments.index,
    )
    assert deletion_exit_pairs.isdisjoint(set(assignment_pair_series))
    assert not assignments["Resolution_Method"].str.contains(
        "deletion", case=False, regex=False
    ).any()
    assert not assignments.duplicated(["As_Of_Date", "Asset_ID"]).any()
    assert "Sector" not in metadata.columns
    latest_sectors = (
        assignments.sort_values("As_Of_Date", kind="stable")
        .drop_duplicates("Asset_ID", keep="last")
        .set_index("Asset_ID")["Sector"]
    )
    assert latest_sectors.loc["VRT"] == "Industrials"
    assert latest_sectors.loc["LITE"] == "Information Technology"
    assert latest_sectors.loc["COHR"] == "Information Technology"
    assert latest_sectors.loc["SATS"] == "Communication Services"
    assert latest_sectors.loc["CASY"] == "Consumer Staples"
    notices = pd.read_csv(DEFAULT_CONFIG.paths.sector_history.notices_csv)
    additions = notices.loc[
        notices["Index_Name"].eq("S&P 500")
        & notices["Action"].str.casefold().eq("addition")
        & notices["Review_Status"].eq("approved")
    ]
    assert len(notices) == 27
    assert len(additions) == 24
    assert {"CASY", "COHR", "CRH", "CVNA", "FIX", "LITE", "SATS", "VRT"}.issubset(
        set(additions["Ticker"])
    )
    assert additions["Source_URL"].str.startswith(
        "https://press.spglobal.com/"
    ).all()


def test_provider_status_requires_identity_but_not_latest_success(tmp_path: Path):
    path = tmp_path / "status.csv"
    request = AcquisitionRequest(
        AcquisitionIdentity("live", "prices", "AAA", "yahoo", "AAA")
    )
    write_acquisition_statuses(path, (AcquisitionStatus.pending(request),))

    assert _load_status(path, {"AAA"}, DOWNLOAD_PRICES_COMMAND) == [
        AcquisitionStatus.pending(request)
    ]
    with pytest.raises(RuntimeError, match="data_acquisition.acquire live prices"):
        _load_status(path, {"AAA", "BBB"}, DOWNLOAD_PRICES_COMMAND)


def test_raw_share_loading_reuses_validated_status_records(
    tmp_path: Path,
    monkeypatch,
):
    status_path = tmp_path / "status.csv"
    raw_path = tmp_path / "shares.csv"
    request = AcquisitionRequest(
        AcquisitionIdentity("live", "shares", "AAPL", "yahoo", "AAPL")
    )
    status = AcquisitionStatus(
        identity=request.identity,
        status=ProviderStatus.OK,
        observation_count=1,
        observation_start="2026-01-30",
        observation_end="2026-01-30",
    )
    write_acquisition_statuses(status_path, (status,))
    pd.DataFrame([{
        "Asset_ID": "AAPL",
        "Provider_Symbol": "AAPL",
        "Effective_Start": "",
        "Effective_End": "",
        "Date": "2026-01-30",
        "Observation_Sequence": 0,
        "Shares_Outstanding": 15_000_000_000.0,
    }]).to_csv(raw_path, index=False)
    records = _load_status(status_path, {"AAPL"}, DOWNLOAD_SHARES_COMMAND)

    monkeypatch.setattr(
        live_preparation,
        "read_acquisition_statuses",
        lambda path: pytest.fail(f"unexpected second status read: {path}"),
    )
    loaded = _load_raw_shares(
        raw_path,
        records,
        pd.DataFrame({"Asset_ID": ["AAPL"]}),
        identity_root=DEFAULT_CONFIG.paths.project_root,
    )

    assert loaded[["Asset_ID", "Provider_Symbol"]].to_dict("records") == [
        {"Asset_ID": "AAPL", "Provider_Symbol": "AAPL"}
    ]


def test_core_preparation_is_deterministic_and_preserves_source_precedence(tmp_path):
    paths = LivePaths(tmp_path / "live")
    config = LiveConfig(paths=paths)
    _copy_canonical_action_inputs(paths)
    paths.membership.directory.mkdir(parents=True)
    effective_dates = [
        "2019-01-01",
        "2026-01-14",
        "2026-02-09",
        "2026-03-23",
        "2026-04-09",
        "2026-05-07",
    ]
    pd.DataFrame({
        "date": effective_dates,
        "tickers": ["AAPL,CTRA"] * len(effective_dates),
    }).to_csv(paths.membership.components_csv, index=False)
    pd.DataFrame(columns=["date", "add", "remove"]).to_csv(
        paths.membership.changes_csv, index=False
    )
    pd.DataFrame([
        {"ticker": "AAPL", "start_date": "2019-01-01", "end_date": ""},
        {"ticker": "CTRA", "start_date": "2019-01-01", "end_date": ""},
    ]).to_csv(paths.membership.intervals_csv, index=False)
    _write_membership_source_manifest(paths.membership)
    schedule = decision_schedule(config)
    sector_dates = sorted(
        set(schedule["Signal_Cutoff"]) | set(schedule["Execution_Date"])
        | {pd.Timestamp(config.market.competition_start), pd.Timestamp("2026-01-30")}
        | set(pd.date_range("2024-02-29", "2025-12-31", freq="ME"))
    )
    _write_synthetic_sector_history(
        paths,
        sector_dates,
        {
            "AAPL": "Information Technology",
            "CTRA": "Energy",
        },
    )
    paths.raw_prices_dir.mkdir(parents=True, exist_ok=True)
    statuses = []
    readiness = []
    for ticker in ("AAPL", "CTRA"):
        identity = AcquisitionIdentity("live", "prices", ticker, "yahoo", ticker)
        statuses.append(
            AcquisitionStatus(
                identity=identity,
                status=ProviderStatus.OK,
                observation_count=2,
                observation_start="2026-01-29",
                observation_end="2026-01-30",
            )
        )
        readiness.append(
            ReadinessRecord(
                scope="live",
                dataset="prices",
                asset_id=ticker,
                requirement_set=PRICE_REQUIREMENT_SET,
                required_count=1,
                covered_count=1,
                status=ReadinessStatus.COMPLETE,
                checked_at_utc="2026-07-25T00:00:00Z",
            )
        )
    write_acquisition_statuses(paths.raw_price_status_csv, statuses)
    write_readiness(paths.raw_price_readiness_csv, readiness)
    write_manifests(paths.raw_price_artifact_manifest_csv, ())
    _copy_authoritative_supplement(paths)
    # This preparation fixture includes all revised evaluation boundaries.
    _, membership, _ = load_validated_strategy_universe(config)
    periods = evaluation_periods(
        schedule, evaluation_start=config.market.competition_start,
        evaluation_end=config.market.competition_end,
    )
    requirements = build_price_requirements(schedule, membership, evaluation=periods)
    dates = sorted({item.requirement_date for item in requirements})
    coverage = build_coverage_ledger(
        requirements, yahoo_dates={asset: set(dates) for asset in ("AAPL", "CTRA")},
        supplemental_dates={}, corporate_actions=pd.DataFrame(columns=["Asset_ID"]),
    )
    coverage.to_csv(paths.raw_price_requirements_csv, index=False)
    for path, base in (
        (paths.raw_price_close_csv, 100.0),
        (paths.raw_price_open_csv, 99.0),
        (paths.raw_price_volume_csv, 1_000.0),
    ):
        frame = pd.DataFrame({"Date": dates, "AAPL": base, "CTRA": base + 2})
        if path == paths.raw_price_close_csv:
            frame.loc[pd.to_datetime(frame.Date).eq(pd.Timestamp("2026-02-13")), "CTRA"] = 999.
        frame.to_csv(path, index=False)
    market = prepare_core_data(config)
    ctra = market.loc[market.Asset_ID.eq("CTRA")].set_index("Date")
    assert ctra.loc[pd.Timestamp("2026-02-13"), "Close"] == 999.
    assert ctra.loc[pd.Timestamp("2026-02-13"), "Price_Source"] == "yahoo"
    assert ctra.loc[pd.Timestamp("2024-03-01"), "Price_Source"] == "yahoo_supplement"
    write_preparation_manifest(paths, include_brinson=False)
    first = {
        path.relative_to(paths.prepared_data_dir): path.read_bytes()
        for path in paths.prepared_data_dir.rglob("*.csv")
    }
    prepare_core_data(config)
    write_preparation_manifest(paths, include_brinson=False)
    second = {
        path.relative_to(paths.prepared_data_dir): path.read_bytes()
        for path in paths.prepared_data_dir.rglob("*.csv")
    }

    assert second == first
    validate_preparation_manifest(paths, include_brinson=False, check_raw=True)
