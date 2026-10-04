"""Offline checks for the flattened live-acquisition state."""

from pathlib import Path

import pandas as pd
import pytest

from data_acquisition.contracts import CommandRequest, ReadinessStatus, read_readiness
import live.acquisition_planning as acquisition_planning
from live.acquisition_planning import (
    benchmark_requirement_dates,
    build_command_requests,
    build_requests,
    live_sector_requirements,
    shares_requirement_dates,
)
from live.config import DEFAULT_CONFIG
from live.strategy_universe import (
    execution_valuation_dates,
    execution_valuation_membership_requirements,
    load_validated_membership,
)
from portfolio_core.artifacts import (
    SCHEMA_VERSION,
    read_manifests,
    validate_manifest_artifact,
)


def test_live_config_encodes_causal_schedule():
    schedule = DEFAULT_CONFIG.schedule
    assert [period.rebalance_id for period in schedule] == ["R1", "R2", "R3"]
    assert schedule[-1].signal_cutoff.isoformat() == "2026-04-30"
    assert schedule[-1].sizing_date.isoformat() == "2026-04-30"
    assert schedule[-1].execution_date.isoformat() == "2026-05-01"
    assert schedule[-1].valuation_end.isoformat() == "2026-05-06"


def test_live_request_plan_is_provider_neutral_and_uses_exact_windows():
    prices = build_requests("prices")
    shares = build_requests("shares")
    benchmark = build_requests("benchmark")

    assert len(prices) == 569
    assert len(shares) == 508
    assert len(benchmark) == 1
    assert {item.identity.provider for item in prices + shares + benchmark} == {
        "yahoo"
    }
    assert prices[0].requested_start == "2023-01-01"
    assert prices[0].requested_end == "2026-05-06"
    assert shares[0].requested_start == "2025-02-13"
    assert shares[0].requested_end == "2026-05-06"
    assert [item.identity.key for item in shares] == sorted(
        item.identity.key for item in shares
    )
    price_symbols = {
        item.identity.asset_id: item.identity.provider_symbol for item in prices
    }
    share_symbols = {
        item.identity.asset_id: item.identity.provider_symbol for item in shares
    }
    assert (price_symbols["BK"], price_symbols["SATS"]) == ("BNY", "ECHO")
    assert (share_symbols["BK"], share_symbols["SATS"]) == ("BK", "SATS")
    assert (price_symbols["FRC"], price_symbols["EQR"]) == ("FRCB", "VMRK")
    assert share_symbols["EQR"] == "EQR"
    holx = next(item for item in shares if item.identity.asset_id == "HOLX")
    assert holx.identity.effective_end == "2026-04-06"
    assert holx.requested_end == "2026-04-06"
    assert benchmark[0].identity.asset_id == "^SP500TR"
    assert benchmark[0].requested_start == "2026-02-12"
    assert benchmark[0].requested_end == "2026-05-06"


def test_live_command_requests_canonicalize_share_selection(tmp_path, monkeypatch):
    expected = (object(),)
    calls = []

    def capture_requests(dataset, *, config, asset_ids=None):
        calls.append((dataset, config, asset_ids))
        return expected

    monkeypatch.setattr(acquisition_planning, "build_requests", capture_requests)
    first_selection = tmp_path / "first.csv"
    first_selection.write_text("Asset_ID\nHOLX\nBK\nHOLX\n", encoding="utf-8")
    second_selection = tmp_path / "second.csv"
    second_selection.write_text("Asset_ID\nBK\nHOLX\n", encoding="utf-8")
    first_command = CommandRequest(
        scope="live",
        dataset="shares",
        tickers_file=first_selection,
        refresh=False,
        dry_run=False,
        project_root=tmp_path,
    )
    second_command = CommandRequest(
        scope="live",
        dataset="shares",
        tickers_file=second_selection,
        refresh=False,
        dry_run=False,
        project_root=tmp_path,
    )
    assert build_command_requests(first_command) == expected
    assert build_command_requests(second_command) == expected
    assert calls == [
        ("shares", DEFAULT_CONFIG, ("BK", "HOLX")),
        ("shares", DEFAULT_CONFIG, ("BK", "HOLX")),
    ]


def test_live_command_requests_preserve_unknown_share_asset_error(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        acquisition_planning,
        "load_validated_strategy_universe",
        lambda _config: (
            object(),
            object(),
            pd.DataFrame({"Asset_ID": ["AAA"], "Yahoo_Ticker": ["AAA"]}),
        ),
    )
    monkeypatch.setattr(
        acquisition_planning,
        "execution_valuation_membership_requirements",
        lambda _schedule, _membership, **kwargs: ((pd.Timestamp("2026-02-13"), "AAA"),),
    )
    selection = tmp_path / "unknown.csv"
    selection.write_text("Asset_ID\nNOT_IN_LIVE_MEMBERSHIP\n", encoding="utf-8")
    command = CommandRequest(
        scope="live",
        dataset="shares",
        tickers_file=selection,
        refresh=False,
        dry_run=False,
        project_root=tmp_path,
    )

    with pytest.raises(ValueError) as exc_info:
        build_command_requests(command, config=DEFAULT_CONFIG)

    assert str(exc_info.value) == (
        "Live request contains assets outside membership: "
        "['NOT_IN_LIVE_MEMBERSHIP']"
    )


def test_live_command_requests_ignore_ticker_selection_outside_shares(
    tmp_path,
    monkeypatch,
):
    calls = []

    def capture_requests(dataset, *, config, asset_ids=None):
        calls.append((dataset, config, asset_ids))
        return (dataset,)

    monkeypatch.setattr(acquisition_planning, "build_requests", capture_requests)
    unused_selection = tmp_path / "must-not-be-read.csv"

    for dataset in ("prices", "benchmark"):
        command = CommandRequest(
            scope="live",
            dataset=dataset,
            tickers_file=unused_selection,
            refresh=False,
            dry_run=False,
            project_root=tmp_path,
        )
        assert build_command_requests(command) == (dataset,)

    assert calls == [
        ("prices", DEFAULT_CONFIG, None),
        ("benchmark", DEFAULT_CONFIG, None),
    ]
    assert not unused_selection.exists()


def test_live_boundary_requirements_match_requests_readiness_and_prepared_shares():
    schedule, membership = load_validated_membership(DEFAULT_CONFIG)
    boundaries = execution_valuation_dates(schedule, evaluation_start=DEFAULT_CONFIG.market.competition_start)
    canonical_pairs = execution_valuation_membership_requirements(
        schedule,
        membership,
        evaluation_start=DEFAULT_CONFIG.market.competition_start,
    )
    benchmark_boundaries = tuple(
        pd.Timestamp(value)
        for value in benchmark_requirement_dates(DEFAULT_CONFIG)
    )
    requirements_by_asset = shares_requirement_dates(DEFAULT_CONFIG)
    expanded_pairs = tuple(sorted(
        (pd.Timestamp(date), asset_id)
        for asset_id, dates in requirements_by_asset.items()
        for date in dates
    ))
    canonical_assets = tuple(sorted({asset_id for _, asset_id in canonical_pairs}))
    requested_assets = tuple(
        request.identity.asset_id
        for request in build_requests("shares", config=DEFAULT_CONFIG)
    )
    readiness = read_readiness(
        DEFAULT_CONFIG.paths.shares.readiness_csv
    )
    readiness_by_asset = {record.asset_id: record for record in readiness}
    prepared = pd.read_csv(
        DEFAULT_CONFIG.paths.shares.prepared_shares_csv,
        parse_dates=["Date"],
    )
    prepared_pairs = tuple(
        prepared[["Date", "Asset_ID"]].itertuples(index=False, name=None)
    )

    assert boundaries == benchmark_boundaries
    assert canonical_pairs == expanded_pairs == prepared_pairs
    assert canonical_assets == requested_assets == tuple(requirements_by_asset)
    assert len(boundaries) == 5
    assert len(canonical_pairs) == len(prepared_pairs) == 2_515
    assert len(canonical_assets) == len(readiness) == 508
    assert {
        boundary: sum(date == boundary for date, _ in canonical_pairs)
        for boundary in boundaries
    } == {boundary: 503 for boundary in boundaries}
    assert set(readiness_by_asset) == set(requirements_by_asset)
    for asset_id, dates in requirements_by_asset.items():
        record = readiness_by_asset[asset_id]
        assert record.required_count == len(dates)
        assert record.covered_count == len(dates)
        assert record.status is ReadinessStatus.COMPLETE
    assert sum(record.required_count for record in readiness) == 2_515
    assert sum(record.covered_count for record in readiness) == 2_515


def test_live_sector_requirements_cover_the_dated_consumer_lifecycle():
    requirements = live_sector_requirements()

    assert list(requirements.columns) == [
        "Scope",
        "Requirement_Date",
        "Cutoff_UTC",
        "Asset_ID",
        "Source_Ticker",
    ]
    assert requirements["Requirement_Date"].min() == pd.Timestamp("2023-01-31")
    assert requirements["Asset_ID"].nunique() >= 568
    history_dates = requirements.loc[
        requirements.Requirement_Date.lt("2026-01-01"), "Requirement_Date",
    ]
    assert history_dates.dt.is_month_end.all()
    assert pd.Timestamp("2023-12-31") in set(history_dates)
    assert pd.Timestamp("2026-01-30") in set(requirements.Requirement_Date)
    competition = requirements.loc[requirements.Requirement_Date.ge("2026-02-13")]
    assert competition.groupby("Requirement_Date").size().to_dict() == {
        pd.Timestamp("2026-02-13"): 503,
        pd.Timestamp("2026-02-27"): 503,
        pd.Timestamp("2026-03-02"): 503,
        pd.Timestamp("2026-03-31"): 503,
        pd.Timestamp("2026-04-01"): 503,
        pd.Timestamp("2026-04-30"): 503,
        pd.Timestamp("2026-05-01"): 503,
    }
    pairs = set(
        requirements[["Requirement_Date", "Asset_ID"]].itertuples(
            index=False,
            name=None,
        )
    )
    assert {
        (pd.Timestamp("2026-02-13"), "DAY"),
        (pd.Timestamp("2026-04-01"), "LW"),
        (pd.Timestamp("2026-04-01"), "MOH"),
        (pd.Timestamp("2026-04-01"), "MTCH"),
        (pd.Timestamp("2026-04-01"), "PAYC"),
        (pd.Timestamp("2026-05-04"), "HOLX"),
    }.isdisjoint(pairs)
    assert not requirements.duplicated(
        ["Scope", "Requirement_Date", "Asset_ID"]
    ).any()
    assert (
        pd.to_datetime(requirements["Cutoff_UTC"], utc=True)
        == requirements["Requirement_Date"].dt.tz_localize("UTC")
    ).all()
    march_2_assets = set(
        requirements.loc[
            requirements["Requirement_Date"].eq(pd.Timestamp("2026-03-02")),
            "Asset_ID",
        ]
    )
    assert "DAY" not in march_2_assets


def test_active_manifests_validate_current_durable_ledgers():
    paths = DEFAULT_CONFIG.paths
    expected = {
        "prices": (
            paths.raw_prices_dir,
            paths.raw_price_artifact_manifest_csv,
            {
                "open.csv",
                "close.csv",
                "volume.csv",
                "acquisition_status.csv",
                "readiness.csv",
                "requirements.csv",
            },
        ),
        "shares": (
            paths.shares.raw_dir,
            paths.shares.artifact_manifest_csv,
            {
                "yahoo_shares_outstanding_raw.csv",
                "acquisition_status.csv",
                "readiness.csv",
            },
        ),
        "benchmark": (
            paths.raw_benchmark_dir,
            paths.raw_benchmark_artifact_manifest_csv,
            {"sp500tr_ohlc.csv", "acquisition_status.csv", "readiness.csv"},
        ),
    }
    for dataset, (base_dir, path, artifacts) in expected.items():
        manifests = read_manifests(path)
        assert {item.schema_version for item in manifests} == {SCHEMA_VERSION}
        assert {item.artifact_path for item in manifests} == artifacts
        assert {item.dataset for item in manifests} == {dataset}
        assert all(not Path(item.artifact_path).is_absolute() for item in manifests)
        for item in manifests:
            validate_manifest_artifact(item, base_dir=base_dir)
