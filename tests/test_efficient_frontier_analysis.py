"""Saved-run acceptance, matched sensitivity comparisons and separated outputs."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.analysis import analyze
from efficient_frontier.inputs import build_history, load_inputs
from efficient_frontier.optimization import FRONTIER_POINTS, estimate_moments, feasible_set
from efficient_frontier.outputs import save_outputs
from live.paths import LivePaths

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_FILES = {
    *(f"tables/{name}.csv" for name in ("summary", "frontier", "weights", "sensitivity", "formation_targets",
                                       "returns", "coverage", "missing_intervals")),
    "metadata/diagnostics.json", "metadata/artifact_manifest.json",
}
WEIGHT_COLUMNS = ["case", "comparison", "portfolio", "Asset_ID", "weight", "lower_bound", "upper_bound"]


@pytest.fixture(scope="module")
def saved_result():
    return analyze(ROOT / "outputs/live/strategies/momentum")


def test_saved_runs_reproduce_audited_universes_samples_and_certified_frontiers(saved_result):
    result = saved_result
    # Compare the diagnostic with its saved input, regardless of the selected candidate.
    decisions = pd.read_csv(result.snapshot.run_dir / "tables/strategy/decisions.csv")
    selected = decisions.loc[decisions.Signal_Raw_Target_Weight.fillna(0).ne(0)]
    first = selected.loc[selected.Signal_Cutoff.eq(selected.Signal_Cutoff.min())].sort_values("Asset_ID")
    assert result.snapshot.assets == first.Asset_ID.tolist()
    np.testing.assert_allclose(result.snapshot.weights, first.Signal_Raw_Target_Weight, rtol=0, atol=1e-15)
    assert result.snapshot.formation_date == pd.Timestamp(first.Signal_Cutoff.iloc[0])
    assert result.snapshot.execution_date == pd.Timestamp(first.Execution_Date.iloc[0])
    assert result.history.returns.columns.tolist() == first.Asset_ID.tolist()
    assert len(result.history.returns) >= 12 and np.isfinite(result.history.returns).all().all()
    last_month = result.snapshot.information_cutoff + pd.offsets.MonthEnd(0)
    assert result.history.returns.index.max() <= last_month
    assert result.history.returns.index.min() > last_month - pd.offsets.MonthEnd(result.settings.lookback)
    assert not result.blockers
    for comparison in result.comparisons.values():
        assert comparison.moments is result.estimates["primary"]
        assert comparison.moments.assets == tuple(result.snapshot.assets)
        assert 1 <= len(comparison.frontier) <= FRONTIER_POINTS
        for solution in [*comparison.solutions.values(), *comparison.frontier]:
            assert solution.accepted
            assert solution.diagnostics["constraint_residual"] <= 1e-8
            assert solution.diagnostics["first_order_gap"] <= 1e-6
        values = np.array([[comparison.moments.mean @ s.weights,
                            s.weights @ comparison.moments.covariance @ s.weights]
                           for s in comparison.frontier])
        assert (np.diff(values, axis=0) >= -1e-8).all()


def test_sensitivities_use_matched_reference_and_only_loo_can_have_eleven_rows(saved_result):
    result = saved_result
    observations = len(result.history.returns)
    records = result.sensitivity
    assert records.status.eq("complete").all()
    assert not {"solver_checks", "numerical_failures"} & set(records.columns)
    evidence = {row["case"]: row for row in result.sensitivity_evidence if row["status"] == "estimated"}
    for row in records.itertuples():
        case = evidence[row.case]
        checks = case["comparisons"][row.comparison]
        assert checks["status"] == row.status and checks["numerical_failures"] == 0
        assert set(checks["solver_checks"]) == {"utility", "reference_return"}
        assert all(check["accepted"] for check in checks["solver_checks"].values())
        moments = result.estimates[case["estimate_id"]]
        mean, covariance = moments.mean, moments.covariance
        w = result.snapshot.weights
        reference_utility = mean @ w - result.settings.risk_aversion * (w @ covariance @ w) / 2
        assert row.reference_utility == pytest.approx(reference_utility)
        assert row.utility_gain == pytest.approx(row.optimized_utility - reference_utility)
        assert row.utility_gain >= -1e-6
        if row.case.startswith("leave_out_"):
            assert row.observations == observations - 1
        else:
            assert row.observations >= 12
        if row.case.startswith("bounds_"):
            assert row.comparison == "bounded"
            assert case["estimate_id"] == "primary"
            assert row.case not in result.estimates
    assert len(records.loc[records.case.str.startswith("leave_out_")]) == 2 * observations


def test_outputs_are_separate_inspectable_reproducible_and_replace_previous_outputs(saved_result, tmp_path):
    result = saved_result
    count = len(result.snapshot.assets)
    saved_input = tmp_path / "saved_run/strategy_parameters.json"
    saved_input.parent.mkdir()
    saved_input.write_bytes(b"Saved input in a sibling directory")
    result = replace(result, snapshot=replace(result.snapshot, run_dir=saved_input.parent))
    old = tmp_path / result.snapshot.strategy_id / result.provenance["content_hash"][:16]
    old.mkdir(parents=True)
    (old / "report.html").write_text("Previous hash-scoped report")
    old_report = old.parent / "reports/report.html"
    old_report.parent.mkdir()
    old_report.write_text("Previous HTML report")
    for relative in ("tables/sensitivity_weights.csv", *(f"metadata/{name}.json" for name in (
            "provenance", "moments", "solver_diagnostics", "sensitivity_evidence", "strategy_parameters",
            "simulation_assumptions", "status"))):
        stale = old.parent / relative
        stale.parent.mkdir(exist_ok=True)
        stale.write_text("Obsolete export")
    destination = save_outputs(result, tmp_path)
    assert destination == tmp_path / result.snapshot.strategy_id
    assert {p.name for p in destination.iterdir()} == {"figures", "tables", "metadata"}
    assert {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()} == (
        EVIDENCE_FILES | {"figures/frontier.png"})
    assert not list(destination.rglob("*.html"))
    assert (destination / "figures/frontier.png").is_file() and not list(destination.rglob("*.svg"))
    assert len(pd.read_csv(destination / "tables/formation_targets.csv")) == count
    assert len(pd.read_csv(destination / "tables/frontier.csv")) == sum(
        len(comparison.frontier) for comparison in result.comparisons.values())
    diagnostics = json.loads((destination / "metadata/diagnostics.json").read_text())
    assert set(diagnostics) == {"schema_version", "provenance", "strategy_parameters", "simulation_assumptions",
                                "estimates", "comparison_estimates", "solver_checks", "sensitivity", "status"}
    assert diagnostics["schema_version"] == 1 and diagnostics["status"]["complete"]
    assert diagnostics["comparison_estimates"] == {"bounded": "primary", "relaxed": "primary"}
    assert all(check["accepted"] for check in diagnostics["solver_checks"])
    assert diagnostics["provenance"] == result.provenance
    assert diagnostics["strategy_parameters"] == result.snapshot.parameters
    assert diagnostics["simulation_assumptions"] == result.snapshot.assumptions
    for case in diagnostics["sensitivity"]:
        assert case["estimate_id"] is None or case["estimate_id"] in diagnostics["estimates"]
    for name, estimate in diagnostics["estimates"].items():
        assert set(estimate) == {"mean", "covariance", "diagnostics"}
        assert estimate["diagnostics"] == result.estimates[name].diagnostics
        assert result.estimates[name].assets == tuple(diagnostics["provenance"]["snapshot"]["assets"])
        np.testing.assert_array_equal(estimate["mean"], result.estimates[name].mean)
        np.testing.assert_array_equal(estimate["covariance"], result.estimates[name].covariance)

    allocations = pd.read_csv(destination / "tables/weights.csv")
    assert list(allocations) == WEIGHT_COLUMNS
    assert not allocations.duplicated(WEIGHT_COLUMNS[:4]).any()
    assert "formation_targets" not in set(allocations.portfolio)
    primary_count = 0
    for name, comparison in result.comparisons.items():
        solutions = {**comparison.solutions, **{f"frontier_{i}": s for i, s in enumerate(comparison.frontier)}}
        for label, solution in solutions.items():
            group = allocations.loc[allocations.case.eq("primary") & allocations.comparison.eq(name)
                                    & allocations.portfolio.eq(label)]
            assert group.Asset_ID.tolist() == result.snapshot.assets
            np.testing.assert_allclose(group.weight, solution.weights, rtol=0, atol=1e-15)
            np.testing.assert_allclose(group.lower_bound, comparison.lower, rtol=0, atol=1e-15)
            np.testing.assert_allclose(group.upper_bound, comparison.upper, rtol=0, atol=1e-15)
            primary_count += len(group)
    assert allocations.iloc[:primary_count].case.eq("primary").all()
    sensitivity_weights = allocations.iloc[primary_count:].reset_index(drop=True)
    pd.testing.assert_frame_equal(sensitivity_weights, result.sensitivity_weights, atol=1e-15, rtol=0)
    evidence = {case["case"]: case for case in result.sensitivity_evidence}
    for (case, name), group in sensitivity_weights.groupby(["case", "comparison"], sort=False):
        assert group.Asset_ID.tolist() == result.snapshot.assets and group.portfolio.eq("utility").all()
        bounds = evidence[case]["bounds"]
        floor, cap = bounds or ((result.settings.floor_multiplier, result.settings.cap_multiplier)
                               if name == "bounded" else (0, None))
        feasible = feasible_set(result.snapshot, floor, cap)
        np.testing.assert_allclose(group.lower_bound, feasible.lower, rtol=0, atol=1e-15)
        np.testing.assert_allclose(group.upper_bound, feasible.upper, rtol=0, atol=1e-15)
    sensitivity = pd.read_csv(destination / "tables/sensitivity.csv")
    assert not {"solver_checks", "numerical_failures"} & set(sensitivity)
    assert not any(value.startswith(("{", "[")) for value in sensitivity.to_numpy().astype(str).flat)

    manifest = (destination / "metadata/artifact_manifest.json").read_bytes()
    hashes = json.loads(manifest)["files"]
    assert set(hashes) == EVIDENCE_FILES - {"metadata/artifact_manifest.json"} | {"figures/frontier.png"}
    assert all(hashlib.sha256((destination / path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    duplicate = save_outputs(result, tmp_path / "another_root")
    assert manifest == (duplicate / "metadata/artifact_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="separate"):
        save_outputs(result, result.snapshot.run_dir)
    summary = (destination / "tables/summary.csv").read_bytes()
    (destination / "tables/summary.csv").write_text("changed")
    assert save_outputs(result, tmp_path) == destination
    assert (destination / "tables/summary.csv").read_bytes() == summary
    assert (destination / "metadata/artifact_manifest.json").read_bytes() == manifest
    assert saved_input.read_bytes() == b"Saved input in a sibling directory"


@pytest.mark.parametrize("protected_kind", ["saved_run", "data", "outputs/live", "outputs/backtest"])
@pytest.mark.parametrize("relationship", [
    "equal", "ancestor", "descendant", "root_symlink", "destination_symlink", "protected_symlink",
])
def test_output_path_overlaps_fail_before_writes_and_preserve_inputs(
        tmp_path, monkeypatch, capsys, protected_kind, relationship):
    from types import SimpleNamespace
    import efficient_frontier.analysis as analysis_module
    import efficient_frontier.outputs as outputs_module
    from efficient_frontier.analyze import main

    project = tmp_path / "project"
    run_dir = tmp_path / "saved/momentum"
    protected = run_dir if protected_kind == "saved_run" else project / protected_kind
    protected.mkdir(parents=True)
    (protected / "strategy_parameters.json").write_bytes(b"Original saved parameters")
    (protected / "tables").mkdir()
    (protected / "tables/decisions.csv").write_bytes(b"Asset_ID,weight\nA,1.0\n")
    monkeypatch.setattr(outputs_module, "__file__", str(project / "efficient_frontier/outputs.py"))

    if relationship == "equal":
        output_root, strategy_id = protected.parent, protected.name
    elif relationship == "ancestor":
        output_root, strategy_id = protected.parent.parent, protected.parent.name
    elif relationship == "descendant":
        output_root, strategy_id = protected / "new_outputs", "momentum"
    elif relationship == "root_symlink":
        output_root, strategy_id = tmp_path / "root_alias", protected.name
        output_root.symlink_to(protected.parent, target_is_directory=True)
    elif relationship == "destination_symlink":
        output_root, strategy_id = tmp_path / "reports", "momentum"
        output_root.mkdir()
        (output_root / strategy_id).symlink_to(protected, target_is_directory=True)
    else:
        physical = tmp_path / "physical_inputs"
        protected.rename(physical)
        protected.symlink_to(physical, target_is_directory=True)
        output_root, strategy_id = physical.parent, physical.name

    result = SimpleNamespace(snapshot=SimpleNamespace(run_dir=run_dir, strategy_id=strategy_id))
    monkeypatch.setattr(analysis_module, "analyze", lambda *args: result)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    paths_before = set(tmp_path.rglob("*"))

    def forbidden_write(*args, **kwargs):
        pytest.fail("Output-path validation must run before any filesystem write")

    monkeypatch.setattr(Path, "mkdir", forbidden_write)
    monkeypatch.setattr(outputs_module.tempfile, "mkdtemp", forbidden_write)
    monkeypatch.setattr(Path, "rename", forbidden_write)
    with pytest.raises(ValueError, match="must be separate") as error:
        save_outputs(result, output_root)
    conflicts = [protected]
    if relationship == "ancestor" and protected_kind.startswith("outputs/"):
        # Their common parent overlaps both protected output trees.
        conflicts = [project / "outputs/live", project / "outputs/backtest"]
    assert any(f"protected directory {path.resolve()}" in str(error.value) for path in conflicts)
    assert str(output_root.resolve()) in str(error.value)
    with pytest.raises(SystemExit) as exit_error:
        main(["--run-dir", str(run_dir), "--output-root", str(output_root)])
    assert exit_error.value.code == 2
    assert str(error.value) in capsys.readouterr().err
    assert set(tmp_path.rglob("*")) == paths_before
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_failed_output_publication_restores_the_previous_diagnostic(saved_result, tmp_path, monkeypatch):
    result = saved_result
    destination = save_outputs(result, tmp_path)
    before = {p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
    rename = Path.rename

    def fail_publication(path, target):
        if path.name == "current":
            raise OSError("publication failed")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_publication)
    with pytest.raises(OSError, match="publication failed"):
        save_outputs(result, tmp_path)
    after = {p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()}
    assert after == before
    assert not list(tmp_path.glob(".frontier-*"))


def test_future_observations_do_not_change_estimates_on_saved_inputs():
    snapshot, inputs, _ = load_inputs(ROOT / "outputs/live/strategies/momentum")
    baseline = build_history(inputs, snapshot, 60)
    monthly = inputs.market_monthly.copy()
    future = monthly.Observation_Date.gt(snapshot.information_cutoff)
    monthly.loc[future, "Close"] *= 100
    after = build_history(replace(inputs, market_monthly=monthly), snapshot, 60)
    pd.testing.assert_frame_equal(after.returns, baseline.returns)
    np.testing.assert_array_equal(estimate_moments(after.returns).covariance, estimate_moments(baseline.returns).covariance)


def test_insufficient_history_retains_every_holding_and_reports_blockers(synthetic_run, monkeypatch, tmp_path):
    import efficient_frontier.analysis as module
    snapshot, inputs = synthetic_run
    first_month = snapshot.information_cutoff + pd.offsets.MonthEnd(0) - pd.offsets.MonthEnd(12)
    monthly = inputs.market_monthly.loc[inputs.market_monthly.Month.ge(first_month)].copy()
    limiting_asset = snapshot.assets[0]
    monthly.loc[monthly.Asset_ID.eq(limiting_asset) & monthly.Month.eq(first_month), "Close"] = np.nan
    inputs.market_monthly = monthly
    result = analyze(snapshot.run_dir)
    assert result.snapshot.assets == snapshot.assets and len(result.history.returns) == 11
    assert result.blockers and limiting_asset in result.blockers[0]
    old_figure = tmp_path / result.snapshot.strategy_id / "figures/frontier.png"
    old_figure.parent.mkdir(parents=True)
    old_figure.write_bytes(b"Previous figure")
    destination = save_outputs(result, tmp_path)
    assert not (destination / "figures/frontier.png").exists()
    assert {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()} == EVIDENCE_FILES
    diagnostics = json.loads((destination / "metadata/diagnostics.json").read_text())
    assert not diagnostics["status"]["complete"] and diagnostics["status"]["blockers"] == result.blockers
    assert diagnostics["estimates"] == diagnostics["comparison_estimates"] == {}
    assert diagnostics["solver_checks"] == diagnostics["sensitivity"] == []
    for name in ("summary", "frontier", "weights", "sensitivity"):
        table = pd.read_csv(destination / f"tables/{name}.csv")
        assert table.empty and len(table.columns) > 0
    assert pd.read_csv(destination / "tables/weights.csv").columns.tolist() == WEIGHT_COLUMNS
    assert pd.read_csv(destination / "tables/returns.csv").columns.tolist() == ["Month", *result.snapshot.assets]
    assert pd.read_csv(destination / "tables/coverage.csv").columns.tolist() == result.history.coverage.columns.tolist()
    assert pd.read_csv(destination / "tables/missing_intervals.csv").columns.tolist() == result.history.missing.columns.tolist()
    formation = pd.read_csv(destination / "tables/formation_targets.csv")
    assert formation.columns.tolist() == result.snapshot.positions.columns.tolist()
    assert formation.Asset_ID.tolist() == snapshot.assets
    monkeypatch.setattr(module, "analyze", lambda *args: result)
    assert module.run_analysis(snapshot.run_dir, output_root=tmp_path) == 2


@pytest.fixture
def synthetic_run(monkeypatch):
    """Exercise failure paths cheaply while retaining the real estimator and solvers."""
    import efficient_frontier.analysis as module
    from portfolio_core.price_basis import price_basis_spec
    from test_efficient_frontier_identity import history_inputs, snapshot
    portfolio = snapshot()
    inputs = history_inputs(portfolio.assets)
    inputs.price_basis = price_basis_spec("live")
    rng = np.random.default_rng(31)
    for asset in portfolio.assets:
        selected = inputs.market_monthly.Asset_ID.eq(asset)
        inputs.market_monthly.loc[selected, "Close"] = 100 * np.cumprod(1 + rng.normal(0.01, 0.04, selected.sum()))
    monkeypatch.setattr(module, "load_inputs", lambda *args: (portfolio, inputs, {}))
    return portfolio, inputs


@pytest.mark.parametrize("utility_ok,reference_ok,status,failures", [
    (True, True, "complete", 0), (True, False, "partial_failure", 1),
    (False, True, "partial_failure", 1), (False, False, "failed", 2),
])
def test_sensitivity_statuses_preserve_checks_nulls_and_cli_failures(
        synthetic_run, monkeypatch, tmp_path, capsys, utility_ok, reference_ok, status, failures):
    import efficient_frontier.analysis as module
    from efficient_frontier.analyze import main
    original_compare = module.compare

    def compare_with_failures(*args, **kwargs):
        comparison = original_compare(*args, **kwargs)
        if not kwargs.get("frontier", True):
            for kind, accepted in (("utility", utility_ok), ("reference_return", reference_ok)):
                solution = comparison.solutions[kind]
                assert solution.accepted
                if not accepted:
                    # Early solver failures may omit an acceptance flag in their raw diagnostics.
                    comparison.solutions[kind] = replace(solution, accepted=False, weights=None,
                                                         diagnostics={"solver_status": 9, "message": "forced rejection"})
        return comparison

    monkeypatch.setattr(module, "compare", compare_with_failures)
    result = analyze(synthetic_run[0].run_dir)
    assert result.sensitivity.status.eq(status).all()
    assert bool(result.blockers) == bool(failures)
    if failures:
        assert result.blockers == [f"{failures * len(result.sensitivity)} sensitivity numerical solutions rejected."]
    optimized = result.sensitivity.filter(regex="^(optimized_|utility_gain$|weight_l1_)")
    assert (optimized.notna().all().all() if utility_ok else optimized.isna().all().all())
    reductions = result.sensitivity.volatility_reduction_at_reference_return
    assert (reductions.notna().all() if utility_ok and reference_ok else reductions.isna().all())
    assert len(result.sensitivity_weights) == (len(result.sensitivity) * len(result.snapshot.assets) if utility_ok else 0)
    monkeypatch.setattr(module, "analyze", lambda *args: result)
    assert main(["--run-dir", str(result.snapshot.run_dir), "--output-root", str(tmp_path)]) == (2 if failures else 0)
    assert ("Diagnostic incomplete:" in capsys.readouterr().out) == bool(failures)
    destination = tmp_path / result.snapshot.strategy_id
    diagnostics = json.loads((destination / "metadata/diagnostics.json").read_text())
    cases = {case["case"]: case for case in diagnostics["sensitivity"]}
    for row in result.sensitivity.itertuples():
        comparison = cases[row.case]["comparisons"][row.comparison]
        assert comparison["status"] == status and comparison["numerical_failures"] == failures
        for kind, accepted in (("utility", utility_ok), ("reference_return", reference_ok)):
            check = comparison["solver_checks"][kind]
            assert check["accepted"] is accepted
            if not accepted:
                assert check == {"accepted": False, "solver_status": 9, "message": "forced rejection"}
    exported = pd.read_csv(destination / "tables/sensitivity.csv")
    pd.testing.assert_frame_equal(exported.isna(), result.sensitivity.isna())
    for column in exported:
        pd.testing.assert_series_equal(exported[column].dropna(), result.sensitivity[column].dropna(), check_dtype=False)


def test_shared_estimates_resolve_deduplicated_windows_and_skip_insufficient_windows(synthetic_run, monkeypatch):
    import efficient_frontier.analysis as module
    portfolio, inputs = synthetic_run
    history = build_history(inputs, portfolio, 60)
    # No observations in the third year: window_36 aliases window_24, not primary.
    dates = pd.date_range("2024-03-31", "2026-02-28", freq="ME").difference([pd.Timestamp("2025-08-31")])
    dates = dates.insert(0, pd.Timestamp("2022-01-31"))
    returns = pd.DataFrame(np.random.default_rng(45).normal(0.01, 0.05, (len(dates), len(portfolio.assets))),
                           index=dates, columns=portfolio.assets)
    monkeypatch.setattr(module, "build_history", lambda *args: replace(history, returns=returns))
    original_compare = module.compare
    calls = []

    def capture_estimates(snapshot, settings, moments, name, **kwargs):
        calls.append((moments, kwargs))
        return original_compare(snapshot, settings, moments, name, **kwargs)

    monkeypatch.setattr(module, "compare", capture_estimates)
    result = analyze(portfolio.run_dir)
    assert not result.blockers
    cases = {case["case"]: case for case in result.sensitivity_evidence}
    assert cases["window_12"]["status"] == "insufficient_history" and cases["window_12"]["estimate_id"] is None
    insufficient = result.sensitivity.loc[result.sensitivity.case.eq("window_12")]
    assert insufficient.status.tolist() == ["insufficient_history"] * 2 and insufficient.observations.eq(11).all()
    assert insufficient.drop(columns=["case", "comparison", "status", "observations"]).isna().all().all()
    assert cases["window_36"]["same_sample_as"] == cases["window_36"]["estimate_id"] == "window_24"
    assert cases["window_60"]["same_sample_as"] == cases["window_60"]["estimate_id"] == "primary"
    assert not result.sensitivity.case.isin(["window_36", "window_60"]).any()
    for label in ("window_12", "window_36", "window_60"):
        for comparison in cases[label]["comparisons"].values():
            assert comparison["solver_checks"] == {} and comparison["numerical_failures"] == 0
    for case in cases.values():
        if case["estimate_id"] is not None:
            assert case["estimate_id"] in result.estimates
    bound_estimates = [moments for moments, kwargs in calls if "floor" in kwargs]
    assert len(bound_estimates) == 2 and all(moments is result.estimates["primary"] for moments in bound_estimates)
    assert set(result.estimates) == {"primary", *(case["case"] for case in cases.values()
        if case["status"] == "estimated" and case["bounds"] is None)}


def test_primary_rejection_without_raw_acceptance_flag_retains_evidence_and_blocks(synthetic_run, monkeypatch, tmp_path):
    import efficient_frontier.analysis as module
    original_compare = module.compare

    def reject_primary(*args, **kwargs):
        comparison = original_compare(*args, **kwargs)
        if kwargs.get("frontier", True):
            solution = comparison.solutions["utility"]
            comparison.solutions["utility"] = replace(solution, accepted=False, weights=None,
                                                       diagnostics={"solver_status": 9, "message": "forced primary rejection"})
        return comparison

    monkeypatch.setattr(module, "compare", reject_primary)
    result = analyze(synthetic_run[0].run_dir)
    assert result.blockers == [f"{name}: 1 numerical solutions rejected; see solver diagnostics."
                               for name in ("bounded", "relaxed")]
    assert result.sensitivity.weight_l1_from_primary_optimum.isna().all()
    monkeypatch.setattr(module, "analyze", lambda *args: result)
    assert module.run_analysis(result.snapshot.run_dir, output_root=tmp_path) == 2
    destination = tmp_path / result.snapshot.strategy_id
    diagnostics = json.loads((destination / "metadata/diagnostics.json").read_text())
    rejected = [check for check in diagnostics["solver_checks"] if not check["accepted"]]
    assert len(rejected) == 2 and all(check["portfolio"] == "utility" for check in rejected)
    assert all(check["solver_status"] == 9 and check["message"] == "forced primary rejection" for check in rejected)
    weights = pd.read_csv(destination / "tables/weights.csv")
    assert not (weights.case.eq("primary") & weights.portfolio.eq("utility")).any()


def test_provenance_records_defaults_sources_implementation_and_versions(saved_result):
    from efficient_frontier.optimization import FEASIBILITY_TOLERANCE, OPTIMALITY_TOLERANCE
    result = saved_result
    assert result.provenance["settings"] == {"lookback": 60, "floor_multiplier": 0.5, "cap_multiplier": 2.0, "brti": 5.2}
    assert result.provenance["numerical_tolerances"] == {
        "feasibility": FEASIBILITY_TOLERANCE, "annual_objective_gap": OPTIMALITY_TOLERANCE,
    }
    assert result.provenance["sources"] and result.provenance["implementation"]
    assert result.provenance["versions"]["scikit-learn"]
    for relative, digest in result.provenance["sources"].items():
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_provenance_is_unchanged_after_checkout_relocation(saved_result, tmp_path):
    checkout = tmp_path / "IPM"
    for relative in saved_result.provenance["sources"]:
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    run = saved_result.snapshot.run_dir.relative_to(ROOT)
    relocated = analyze(checkout / run, saved_result.settings, paths=LivePaths(checkout / "live"))
    assert relocated.snapshot.run_dir != saved_result.snapshot.run_dir
    assert relocated.provenance == saved_result.provenance


def test_external_saved_run_keeps_portable_source_paths(saved_result, tmp_path):
    run = saved_result.snapshot.run_dir.relative_to(ROOT)
    exported = tmp_path / "exported_run"
    expected = {}
    for relative, digest in saved_result.provenance["sources"].items():
        path = Path(relative)
        if path.is_relative_to(run):
            local = path.relative_to(run)
            destination = exported / local
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, destination)
            relative = (Path("saved_run") / local).as_posix()
        expected[relative] = digest
    _, _, before = load_inputs(exported)
    renamed = exported.rename(tmp_path / "renamed_run")
    _, _, after = load_inputs(renamed)
    assert before == after == expected


def test_stale_price_basis_requires_replay_not_metadata_relabelling(tmp_path):
    import hashlib
    run = ROOT / "outputs/live/strategies/momentum"
    (tmp_path / "strategy_parameters.json").write_bytes((run / "strategy_parameters.json").read_bytes())
    assumptions = json.loads((run / "simulation_assumptions.json").read_text())
    assumptions["price_basis"] = {"domain": "live", "price_basis_id": "yahoo_adjusted_total_return",
                                  "ordinary_dividend_treatment": "embedded in adjusted prices",
                                  "price_source": "Yahoo Finance adjusted OHLC",
                                  "price_treatment": "adjusted prices used consistently"}
    del assumptions["simulation_fingerprint"]
    assumptions["simulation_fingerprint"] = hashlib.sha256(json.dumps(
        assumptions, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    (tmp_path / "simulation_assumptions.json").write_text(json.dumps(assumptions))
    with pytest.raises(ValueError, match="Rerun the live strategy"):
        load_inputs(tmp_path)


def test_price_basis_mismatch_rejected_before_history(monkeypatch):
    import efficient_frontier.inputs as module
    from types import SimpleNamespace
    from portfolio_core.price_basis import price_basis_spec
    monkeypatch.setattr(module, "load_analysis_inputs", lambda paths: SimpleNamespace(price_basis=price_basis_spec("backtest")))
    with pytest.raises(ValueError, match="different price bases"):
        load_inputs(ROOT / "outputs/live/strategies/momentum")


def test_malformed_parameter_wrapper_has_actionable_error(tmp_path):
    (tmp_path / "strategy_parameters.json").write_text("[]")
    with pytest.raises(ValueError, match="must declare a strategy_id"):
        load_inputs(tmp_path)
