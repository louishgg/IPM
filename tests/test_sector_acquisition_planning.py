"""Tests for sector acquisition planning and readiness."""

from __future__ import annotations

from contextlib import contextmanager

import pandas as pd

import data_acquisition.sector_acquisition_planning as sector_planning_module
from data_acquisition.contracts import CommandRequest
from data_acquisition.contracts import (
    AcquisitionStatus,
    ProviderStatus,
    ReadinessRecord,
    ReadinessStatus,
    write_acquisition_statuses,
)
from data_acquisition.sector_acquisition_planning import (
    SECTOR_ACQUISITION_REQUIREMENT_COLUMNS,
    build_sector_acquisition_readiness,
    build_sector_acquisition_requests,
    dry_run_sector_acquisition,
    project_sector_acquisition_requirements,
    report_scope_readiness,
    validate_sector_acquisition_requirements,
)
from portfolio_core.sector_evidence import (
    SNAPSHOT_COLUMNS,
    SectorHistoryPaths,
)
from portfolio_core.sector_resolution import (
    WikipediaIdentityResolver,
)
from _sector_test_helpers import (
    _acquisition_requirements,
    _notice,
    _notices,
    _snapshots,
    _write_sector_manifest,
)


def test_project_sector_acquisition_requirements_preserves_source_frame():
    index = pd.Index([9, 2], dtype="int64", name="source_row")
    requirements = pd.DataFrame(
        {
            "As_Of_Date": pd.Series(
                pd.to_datetime(["2024-02-29", "2024-01-31"]),
                index=index,
                dtype="datetime64[ns]",
            ),
            "Asset_ID": pd.Series(
                pd.Categorical(
                    ["BBB.N", "AAA.N"],
                    categories=["AAA.N", "BBB.N"],
                    ordered=True,
                ),
                index=index,
            ),
            "Source_Ticker": pd.Series(
                ["BBB", "AAA"],
                index=index,
                dtype=object,
            ),
        },
        index=index,
    )
    original = requirements.copy(deep=True)
    expected = pd.DataFrame(
        {
            "Scope": pd.Series(["live", "live"], index=index, dtype="str"),
            "Requirement_Date": requirements["As_Of_Date"],
            "Cutoff_UTC": pd.Series(
                ["2024-02-29T00:00:00Z", "2024-01-31T00:00:00Z"],
                index=index,
                dtype="str",
            ),
            "Asset_ID": requirements["Asset_ID"],
            "Source_Ticker": requirements["Source_Ticker"],
        },
        index=index,
    )

    actual = project_sector_acquisition_requirements(
        requirements,
        scope="live",
    )

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(requirements, original, check_exact=True)
    assert tuple(actual.columns) == SECTOR_ACQUISITION_REQUIREMENT_COLUMNS
    assert actual.index.equals(index)
    assert actual["Cutoff_UTC"].tolist() == [
        "2024-02-29T00:00:00Z",
        "2024-01-31T00:00:00Z",
    ]
    assert actual["Requirement_Date"].dtype == requirements["As_Of_Date"].dtype
    assert actual["Asset_ID"].dtype == requirements["Asset_ID"].dtype
    assert actual["Source_Ticker"].dtype == requirements["Source_Ticker"].dtype


def test_readiness_is_asset_date_coverage_not_provider_status():
    requirements = validate_sector_acquisition_requirements(
        pd.DataFrame(
            [
                ("backtest", "2020-01-31", "2020-01-31T00:00:00Z", "AAA", "AAA"),
                ("backtest", "2020-02-29", "2020-02-29T00:00:00Z", "AAA", "AAA"),
            ],
            columns=SECTOR_ACQUISITION_REQUIREMENT_COLUMNS,
        )
    )
    readiness = build_sector_acquisition_readiness(
        requirements,
        _snapshots([("2020-01-31", "AAA", "Energy")]),
        _notices(),
        identity_resolver=WikipediaIdentityResolver(),
        checked_at_utc="2026-08-05T12:00:00Z",
    )
    assert len(readiness) == 1
    assert readiness[0].status.value == "partial"
    assert readiness[0].missing_dates == ("2020-02-29",)

def test_in_memory_readiness_report_limits_sorted_missing_pairs():
    readiness = tuple(
        ReadinessRecord(
            scope="backtest",
            dataset="sectors",
            asset_id=f"A{index:02d}",
            requirement_set="point_in_time_sector_assignment",
            required_count=1,
            covered_count=0,
            status=ReadinessStatus.MISSING,
            missing_dates=("2020-01-31",),
            checked_at_utc="2026-08-05T12:00:00Z",
        )
        for index in reversed(range(21))
    )
    messages = []

    report_scope_readiness(
        readiness,
        scope="backtest",
        reporter=messages.append,
    )

    assert messages[0] == (
        "backtest sector evidence readiness: 0/21 asset/date pair(s) "
        "resolved; 21 missing."
    )
    assert messages[1:21] == [
        f"  2020-01-31: A{index:02d}" for index in range(20)
    ]
    assert messages[21] == "  ... 1 additional missing pair(s)."


def test_readiness_indexes_each_wikipedia_revision_once(monkeypatch):
    date = "2020-01-31"
    requirements = validate_sector_acquisition_requirements(
        pd.DataFrame(
            [
                ("backtest", date, f"{date}T00:00:00Z", "AAA.N", "AAA"),
                ("backtest", date, f"{date}T00:00:00Z", "BBB.N", "BBB"),
            ],
            columns=SECTOR_ACQUISITION_REQUIREMENT_COLUMNS,
        )
    )
    resolver = WikipediaIdentityResolver()
    original = resolver._symbol_index
    calls = 0

    def counting_symbol_index(snapshot):
        nonlocal calls
        calls += 1
        return original(snapshot)

    monkeypatch.setattr(resolver, "_symbol_index", counting_symbol_index)
    readiness = build_sector_acquisition_readiness(
        requirements,
        _snapshots(
            [(date, "AAA", "Energy"), (date, "BBB", "Industrials")]
        ),
        _notices(),
        identity_resolver=resolver,
        checked_at_utc="2026-08-05T12:00:00Z",
    )

    assert calls == 1
    assert all(record.status is ReadinessStatus.COMPLETE for record in readiness)

def test_reviewed_addition_resolution_reuses_one_index_per_revision(monkeypatch):
    dates = ("2020-01-31", "2020-02-29", "2020-03-31")
    requirements = validate_sector_acquisition_requirements(
        pd.DataFrame(
            [
                ("backtest", date, f"{date}T00:00:00Z", asset, ticker)
                for date in dates
                for asset, ticker in (("NEW.N", "NEW"),)
            ],
            columns=SECTOR_ACQUISITION_REQUIREMENT_COLUMNS,
        )
    )
    snapshots = _snapshots(
        [
            *((date, "OTHER", "Materials") for date in dates),
            ("2020-03-31", "NEW", "Industrials"),
        ]
    )
    resolver = WikipediaIdentityResolver()
    original = resolver._symbol_index
    calls = 0

    def counting_symbol_index(snapshot):
        nonlocal calls
        calls += 1
        return original(snapshot)

    monkeypatch.setattr(resolver, "_symbol_index", counting_symbol_index)
    build_sector_acquisition_readiness(
        requirements,
        snapshots,
        _notices([_notice(Effective_Date="2020-02-01")]),
        identity_resolver=resolver,
        checked_at_utc="2026-08-05T12:00:00Z",
    )

    assert calls == len(dates)

def test_dry_run_holds_shared_lock_through_validation_reads_and_planning(
    tmp_path,
    monkeypatch,
):
    state = {"locked": False, "stages": []}

    @contextmanager
    def fake_lock(_path):
        assert not state["locked"]
        state["locked"] = True
        try:
            yield
        finally:
            state["locked"] = False

    def record(stage):
        assert state["locked"]
        state["stages"].append(stage)

    def load_state(*args, **kwargs):
        record("state")
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS), {}, [], []

    original_build_plan = sector_planning_module.build_acquisition_plan

    def build_plan(*args, **kwargs):
        record("plan")
        return original_build_plan(*args, **kwargs)

    monkeypatch.setattr(sector_planning_module, "acquisition_lock", fake_lock)
    monkeypatch.setattr(
        sector_planning_module,
        "load_sector_acquisition_state",
        load_state,
    )
    monkeypatch.setattr(
        sector_planning_module,
        "load_sector_notices",
        lambda path: _notices(),
    )
    monkeypatch.setattr(
        sector_planning_module,
        "load_security_identity_bundle",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        sector_planning_module,
        "build_acquisition_plan",
        build_plan,
    )
    paths = SectorHistoryPaths.from_project_root(tmp_path)
    paths.directory.mkdir(parents=True)
    paths.notices_csv.write_text("reviewed notice placeholder\n")
    request = CommandRequest(
        scope="backtest",
        dataset="sectors",
        tickers_file=None,
        refresh=False,
        dry_run=True,
        project_root=tmp_path,
    )

    assert len(dry_run_sector_acquisition(request, _acquisition_requirements)) == 1
    assert state == {
        "locked": False,
        "stages": ["state", "plan"],
    }

def test_dry_run_skips_checkpoint_and_refresh_verifies_pin(
    tmp_path,
    monkeypatch,
    capsys,
):
    requirements = _acquisition_requirements()
    request = build_sector_acquisition_requests(requirements)[0]
    paths = SectorHistoryPaths.from_project_root(tmp_path)
    paths.directory.mkdir(parents=True)
    _snapshots([("2020-01-31", "AAA", "Energy")]).to_csv(
        paths.snapshots_csv, index=False
    )
    write_acquisition_statuses(
        paths.acquisition_status_csv,
        (
            AcquisitionStatus(
                identity=request.identity,
                status=ProviderStatus.OK,
                requested_start=request.requested_start,
                requested_end=request.requested_end,
                observation_count=1,
                observation_start="2020-01-31",
                observation_end="2020-01-31",
                attempted_at_utc="2026-08-05T12:00:00Z",
                client="requests",
                client_version="1",
                http_status="200",
            ),
        ),
    )
    _notices([_notice(Ticker="OTHER")]).to_csv(paths.notices_csv, index=False)
    _write_sector_manifest(paths, tmp_path)
    monkeypatch.setattr(
        sector_planning_module,
        "load_security_identity_bundle",
        lambda *args, **kwargs: None,
    )

    command = CommandRequest(
        scope="backtest",
        dataset="sectors",
        tickers_file=None,
        refresh=False,
        dry_run=True,
        project_root=tmp_path,
    )
    assert dry_run_sector_acquisition(command, _acquisition_requirements) == ()

    refresh = CommandRequest(
        scope="backtest",
        dataset="sectors",
        tickers_file=None,
        refresh=True,
        dry_run=True,
        project_root=tmp_path,
    )
    assert dry_run_sector_acquisition(refresh, _acquisition_requirements) == (request,)
    assert "verify pinned revision 101" in capsys.readouterr().out
    runtime = sector_planning_module.AcquisitionRuntimePaths(tmp_path)
    assert runtime.locks.is_dir()
    assert not runtime.yfinance_cache.exists()
    assert not runtime.sector_history.exists()
