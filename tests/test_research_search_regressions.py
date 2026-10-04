"""Synthetic validation-boundary, checkpoint and spawned-worker regressions."""

from dataclasses import replace
import json
import os
from time import monotonic, sleep

import pandas as pd
import pytest

from backtest import research_search as search
from backtest import search as cli
from backtest.config import DEFAULT_CONFIG
from backtest.research_grid import expand_grid, frozen_identity, candidate_id, readiness_bounds
from portfolio_core.strategies import build_registered_strategy
from portfolio_core.strategies.research_parameters import ResearchParameters
from portfolio_core.strategies.research_parameters import SizingParameters, VolatilityProfile
from _strategy_test_helpers import momentum_test_parameters
from test_engine_offline import make_synthetic_backtest_data
from _research_test_helpers import tiny_grid
from test_research_execution_regressions import _shorts_before_shock
from test_reporting_configuration_matrix import EVALUATION


_REAL_EVALUATE = search._evaluate


@pytest.mark.parametrize(("sizing", "construction_index"), [
    (SizingParameters("equal", None), 21),
    (SizingParameters("inverse_volatility", VolatilityProfile(60, 24)), 25),
])
def test_readiness_preserves_missing_membership_for_signal_and_sizing(sizing, construction_index):
    data, dates = make_synthetic_backtest_data()
    membership = pd.DataFrame(True, index=dates[20:], columns=data.data_close.columns[:20], dtype=object)
    membership.iloc[0, 0] = None
    membership.loc[dates[24], membership.columns[0]] = None
    before = membership.copy(deep=True)
    bounds = readiness_bounds(
        momentum_test_parameters(sizing=sizing), replace(data, pit_matrix=membership),
    )
    assert bounds["First_Signal_Lower_Bound"] == str(dates[21].date())
    assert bounds["First_Construction_Lower_Bound"] == str(dates[construction_index].date())
    assert not bounds["History_Unready"]
    pd.testing.assert_frame_equal(membership, before, check_exact=True)


def _failing_spawned_candidate(row):
    """Importable spawn target: checkpoint one job, then fail behind a busy job."""
    directory = search._WORKER[2]
    number = row["Candidate_Number"]
    if number == 1:
        return _REAL_EVALUATE(row)
    if number == 2:
        (directory / "busy-worker").write_text(str(os.getpid()))
        sleep(30)
        raise AssertionError("The parent failed to terminate the busy worker")
    deadline = monotonic() + 10
    while not (directory / "busy-worker").exists():
        if monotonic() > deadline:
            raise AssertionError("The earlier worker did not start")
        sleep(.01)
    mode = (directory / "failure-mode").read_text()
    if mode == "crash":
        os._exit(17)
    if mode == "interrupt":
        raise KeyboardInterrupt("synthetic worker interruption")
    raise RuntimeError("synthetic later candidate failure")


@pytest.mark.parametrize("mode", ["error", "crash", "interrupt"])
def test_spawned_failure_interrupts_earlier_work_and_preserves_checkpoint(monkeypatch, tmp_path, mode):
    data, _ = make_synthetic_backtest_data()
    grid = tiny_grid()
    grid["gross"] = [1., 1.25, 1.5]
    (tmp_path / "failure-mode").write_text(mode)
    monkeypatch.setattr(search, "_evaluate", _failing_spawned_candidate)
    started = monotonic()
    with pytest.raises(KeyboardInterrupt if mode == "interrupt" else RuntimeError):
        search.run_research_search(data, grid, tmp_path, workers=2, evaluation_config=EVALUATION)
    assert monotonic() - started < 25  # Busy work lasts 30 seconds if allowed to drain.
    checkpoints = list((tmp_path / "candidates").glob("*/checkpoint.json"))
    assert len(checkpoints) == 1
    checkpoint_bytes = checkpoints[0].read_bytes()
    pid = int((tmp_path / "busy-worker").read_text())
    deadline = monotonic() + 5
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        if monotonic() > deadline:
            pytest.fail("The failed search left its worker alive")
        sleep(.01)
    monkeypatch.setattr(search, "_evaluate", _REAL_EVALUATE)
    result = search.run_research_search(data, grid, tmp_path, workers=2, evaluation_config=EVALUATION)
    assert result["completed_candidates"] == 3
    assert result["resumed_candidates"] == 1
    assert checkpoints[0].read_bytes() == checkpoint_bytes
    folds = pd.read_csv(tmp_path / "fold_metrics.csv.gz")
    assert folds.Candidate_Number.tolist() == [1, 2, 3]


def test_terminal_insolvency_is_unranked_without_folds_or_rank(tmp_path):
    data, _ = make_synthetic_backtest_data()
    grid = tiny_grid()
    grid["gross"] = [2.]
    rows, _, _ = expand_grid(grid)
    strategy = build_registered_strategy("momentum", json.loads(rows[0]["Requested_JSON"]))
    dates = pd.date_range(EVALUATION.initial_research_start_date, EVALUATION.validation_end_date, freq="ME")
    shorts = _shorts_before_shock(data, dates, strategy)
    data.data_close.loc[dates[-1], shorts] *= 3
    result = search.run_research_search(data, grid, tmp_path, evaluation_config=EVALUATION)
    assert result["ranked_candidates"] == 0
    candidates = pd.read_csv(tmp_path / "candidates.csv.gz")
    assert candidates.Status.tolist() == ["unranked_infeasible"]
    assert candidates.Reason.str.contains(f"{dates[-1].date()}.*nonpositive").all()
    assert candidates[["PctReturn_Rank", "Sharpe_Rank"]].isna().all().all()
    checkpoint = json.loads(next((tmp_path / "candidates").glob("*/checkpoint.json")).read_text())
    assert checkpoint["folds"] == []


def test_unexpected_engine_error_does_not_create_a_checkpoint(monkeypatch, tmp_path):
    data, _ = make_synthetic_backtest_data()
    def invalid_input(*args, **kwargs):
        raise ValueError("synthetic invalid valuation input")
    monkeypatch.setattr(search, "_run_backtest_impl", invalid_input)
    with pytest.raises(ValueError, match="invalid valuation"):
        search.run_research_search(data, tiny_grid(), tmp_path, evaluation_config=EVALUATION)
    assert not list((tmp_path / "candidates").glob("*/checkpoint.json"))


def test_small_stock_packet_is_unranked_under_default_account_minimums(tmp_path):
    data, _ = make_synthetic_backtest_data()
    grid = tiny_grid()
    grid["n_long"] = grid["n_short"] = [1]
    result = search.run_research_search(data, grid, tmp_path, evaluation_config=EVALUATION)
    assert result["ranked_candidates"] == 0
    row = pd.read_csv(tmp_path / "candidates.csv.gz").iloc[0]
    assert row.Status == "unranked_infeasible"
    assert "position counts" in row.Reason


@pytest.mark.parametrize("family", ["momentum", "reversal", "monthly_trend", "low_volatility", "sector_momentum"])
def test_every_search_cli_requests_development_only_before_loading(monkeypatch, family):
    class BoundaryChecked(Exception):
        pass
    def loader(*args, **kwargs):
        assert kwargs == {"development_only": True}
        raise BoundaryChecked
    monkeypatch.setattr(cli, "load_backtest_data", loader)
    with pytest.raises(BoundaryChecked):
        cli.main(["--strategy", family, "--workers", "1"])


def test_resolved_protocol_drives_search_readiness_folds_and_identity(tmp_path):
    data, _ = make_synthetic_backtest_data()
    grid = tiny_grid()
    result = search.run_research_search(data, grid, tmp_path, evaluation_config=EVALUATION)
    assert result["validation_returns"] == 6
    assert result["identity"]["evaluation"]["validation_end_date"] == "2015-12-31"
    candidate = pd.read_csv(tmp_path / "candidates.csv.gz").iloc[0]
    assert candidate.Validation_Return_Count == 6
    folds = pd.read_csv(tmp_path / "fold_metrics.csv.gz")
    assert folds.Validation_Year.tolist() == [2015]
    packet = ResearchParameters.from_payload(json.loads(candidate.Requested_JSON))
    late = replace(EVALUATION, initial_research_start_date=pd.Timestamp("2015-07-31").date(),
        initial_research_end_date=pd.Timestamp("2015-07-31").date(),
        validation_start_date=pd.Timestamp("2015-08-31").date())
    assert readiness_bounds(packet, data, evaluation_config=late)["First_Construction_Lower_Bound"] == "2015-07-31"


def test_sector_manifest_uses_the_same_complete_identity_as_search(tmp_path):
    from backtest.research_grid import read_grid
    data, _ = make_synthetic_backtest_data()
    grid = read_grid("backtest/grids/sector_momentum_grid.json")
    for key, values in grid.items():
        if isinstance(values, list):
            grid[key] = values[:1]
    result = search.run_research_search(data, grid, tmp_path,
        evaluation_config=EVALUATION, preflight_only=True)
    manifest = json.loads((tmp_path / "sector_return_manifest.json").read_text())
    assert manifest["identity"] == result["identity"]
    assert (manifest["start"], manifest["end"]) == ("2015-01-31", "2015-12-31")


@pytest.mark.parametrize("change", ("edit", "rename"))
def test_source_change_invalidates_checkpoint_identity_without_schema_migration(tmp_path, change):
    data, _ = make_synthetic_backtest_data()
    source = tmp_path / "backtest" / "engine.py"
    source.parent.mkdir()
    source.write_text("# old implementation\n")
    before = frozen_identity(data, DEFAULT_CONFIG.accounting, tmp_path)
    if change == "edit":
        source.write_text("# corrected implementation\n")
    else:
        source.rename(source.with_name("research_execution.py"))
    after = frozen_identity(data, DEFAULT_CONFIG.accounting, tmp_path)
    row = expand_grid(tiny_grid())[0][0]
    assert before["schema_version"] == after["schema_version"] == "2.0.0"
    assert candidate_id(row, before) != candidate_id(row, after)
    row["Candidate_ID"] = candidate_id(row, before)
    search._write_json({"identity": before, "row": row, "artifacts": {}, "folds": []},
                       tmp_path / "checkpoint.json")
    with pytest.raises(ValueError, match="Checkpoint identity mismatch"):
        search._load_checkpoint(tmp_path, row, after)
