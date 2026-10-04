"""Tests for sector provider execution boundaries."""

from __future__ import annotations

from contextlib import contextmanager

import data_acquisition.sector_acquisition_execution as sector_execution_module
from data_acquisition.contracts import (
    AcquisitionStatus,
    CommandRequest,
    ProviderStatus,
    write_acquisition_statuses,
)
from data_acquisition.sector_acquisition_execution import execute_sector_acquisition
from _sector_test_helpers import (
    _acquisition_requirements,
    _notice,
    _notices,
    _snapshots,
    _write_sector_manifest,
)


def test_non_dry_run_holds_shared_lock_around_the_complete_mutation(
    tmp_path,
    monkeypatch,
):
    state = {"locked": False, "called": False}
    constructed = []
    runtime_factory = sector_execution_module.AcquisitionRuntimePaths

    def recording_runtime_paths(project_root):
        runtime = runtime_factory(project_root)
        constructed.append(runtime)
        return runtime

    @contextmanager
    def fake_lock(path):
        assert len(constructed) == 1
        assert path == constructed[0].lock_path("shared", "sectors")
        assert not state["locked"]
        state["locked"] = True
        try:
            yield
        finally:
            state["locked"] = False

    def fake_locked_acquisition(request, current, paths, runtime):
        assert state["locked"]
        assert request.scope == "backtest"
        assert len(current) == 1
        assert paths.directory.is_relative_to(tmp_path)
        assert runtime is constructed[0]
        state["called"] = True
        return 0

    monkeypatch.setattr(
        sector_execution_module,
        "AcquisitionRuntimePaths",
        recording_runtime_paths,
    )
    monkeypatch.setattr(sector_execution_module, "acquisition_lock", fake_lock)
    monkeypatch.setattr(
        sector_execution_module,
        "_execute_sector_acquisition_locked",
        fake_locked_acquisition,
    )
    request = CommandRequest(
        scope="backtest",
        dataset="sectors",
        tickers_file=None,
        refresh=False,
        dry_run=False,
        project_root=tmp_path,
    )

    assert execute_sector_acquisition(request, _acquisition_requirements) == 0
    assert len(constructed) == 1
    assert state == {"locked": False, "called": True}


def test_complete_provider_run_treats_partial_conservative_readiness_as_diagnostic(
    tmp_path,
    monkeypatch,
    capsys,
):
    current = _acquisition_requirements()
    provider_request = sector_execution_module.build_sector_acquisition_requests(
        current
    )[0]
    paths = sector_execution_module.SectorHistoryPaths.from_project_root(tmp_path)
    paths.directory.mkdir(parents=True)
    _snapshots([("2020-01-31", "BBB", "Energy")]).to_csv(
        paths.snapshots_csv,
        index=False,
    )
    _notices([_notice(Ticker="OTHER")]).to_csv(
        paths.notices_csv,
        index=False,
    )
    write_acquisition_statuses(
        paths.acquisition_status_csv,
        (
            AcquisitionStatus(
                identity=provider_request.identity,
                status=ProviderStatus.OK,
                requested_start=provider_request.requested_start,
                requested_end=provider_request.requested_end,
                observation_count=1,
                observation_start=provider_request.requested_start,
                observation_end=provider_request.requested_end,
                attempted_at_utc="2026-01-01T00:00:00Z",
                client="synthetic-test-client",
                client_version="1",
                http_status="200",
            ),
        ),
    )
    _write_sector_manifest(paths, tmp_path)
    monkeypatch.setattr(
        sector_execution_module,
        "load_security_identity_bundle",
        lambda *args, **kwargs: None,
    )
    request = CommandRequest(
        scope="backtest",
        dataset="sectors",
        tickers_file=None,
        refresh=False,
        dry_run=False,
        project_root=tmp_path,
    )

    assert execute_sector_acquisition(request, lambda: current) == 0
    stderr = capsys.readouterr().err
    assert (
        "backtest sector evidence readiness: 0/1 asset/date pair(s) resolved; "
        "1 missing."
    ) in stderr
    assert (
        "backtest sector revisions are complete with immutable Wikipedia "
        "provenance; conservative readiness remains incomplete and is "
        "diagnostic only. Exact consumer coverage is enforced during "
        "preparation."
    ) in stderr
