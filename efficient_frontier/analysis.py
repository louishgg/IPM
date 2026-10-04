"""Orchestrate the saved-portfolio diagnostic and one-at-a-time sensitivities."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from portfolio_core.artifacts import file_sha256

from .contracts import (
    DiagnosticResult, DiagnosticSettings, HistorySample, SENSITIVITY_COLUMNS,
    WEIGHT_COLUMNS, allocation_rows,
)
from .inputs import build_history, load_inputs
from .optimization import (
    FEASIBILITY_TOLERANCE, OPTIMALITY_TOLERANCE, compare, estimate_moments, metrics,
)


def implementation_receipt(project_root: Path) -> dict:
    """Include shared valuation/validation code, not just this package's entrypoint."""
    paths = [p for package in ("efficient_frontier", "live", "portfolio_core")
             for p in (project_root / package).rglob("*.py")]
    paths.append(project_root / "environment.yml")
    return {str(p.relative_to(project_root)): file_sha256(p) for p in sorted(paths)}


def _sensitivity_cases(history: HistorySample, settings: DiagnosticSettings):
    """Return ordered estimation cases and metadata for skipped windows."""
    base = history.returns
    cases, skipped = [], []
    # Windows vary calendar endpoints, never merely take the last n nonmissing rows.
    seen = {tuple(base.index): "primary"}
    for window in (12, 24, 36, 60):
        if window > settings.lookback:
            continue
        subset = base.loc[base.index > history.calendar_end - pd.offsets.MonthEnd(window)]
        key = tuple(subset.index)
        label = f"window_{window}"
        if key in seen or len(subset) < 12:
            status = "deduplicated" if key in seen else "insufficient_history"
            case = {"case": label, "status": status, "observations": len(subset),
                    "estimate_id": seen.get(key)}
            if key in seen:
                case["same_sample_as"] = seen[key]
            skipped.append(case)
        else:
            seen[key] = label
            cases.append((label, subset, {}, None))
    cases.extend([("means_half", base, {"mean_shrinkage": 0.5}, None),
                  ("covariance_diagonal_20", base, {"diagonal_shrinkage": 0.2}, None),
                  ("covariance_diagonal_50", base, {"diagonal_shrinkage": 0.5}, None),
                  ("bounds_0.75_1.5", base, {}, (0.75, 1.5)),
                  ("bounds_0.25_3", base, {}, (0.25, 3.0))])
    cases.extend((f"leave_out_{date.date()}", base.drop(index=date), {}, None) for date in base.index)
    return cases, skipped


def _sensitivity(snapshot, settings, history, primary):
    records, weights = [], []
    estimates = {"primary": primary["bounded"].moments}
    cases, evidence = _sensitivity_cases(history, settings)
    for case in evidence:
        case["comparisons"] = {
            name: {"status": case["status"], "numerical_failures": 0, "solver_checks": {}}
            for name in primary
        }
        if case["status"] == "insufficient_history":
            records.extend(
                {"case": case["case"], "comparison": name, "status": case["status"],
                 "observations": case["observations"]}
                for name in primary
            )
    for label, sample, estimator_options, bounds in cases:
        estimate_id = "primary" if bounds else label
        if bounds is None:
            estimates[estimate_id] = estimate_moments(sample, **estimator_options)
        moments = estimates[estimate_id]
        # The 11-observation exception is only a perturbation of an admitted sample.
        case = {"case": label, "status": "estimated", "estimate_id": estimate_id,
                "bounds": bounds, "leave_one_out_perturbation": label.startswith("leave_out_"), "comparisons": {}}
        evidence.append(case)
        for name in (["bounded"] if bounds else primary):
            kwargs = {} if bounds is None else {"floor": bounds[0], "cap": bounds[1]}
            result = compare(snapshot, settings, moments, name, frontier=False, **kwargs)
            ref = metrics(snapshot.weights, moments, settings.risk_aversion)
            optimum = result.solutions["utility"]
            matched = result.solutions["reference_return"]
            failures = sum(not solution.accepted for solution in result.solutions.values())
            status = "complete" if failures == 0 else "failed" if failures == 2 else "partial_failure"
            case["comparisons"][name] = {
                "status": status, "numerical_failures": failures,
                "solver_checks": {kind: {**solution.diagnostics, "accepted": solution.accepted}
                                  for kind, solution in result.solutions.items()},
            }
            record = {"case": label, "comparison": name, "status": status,
                      "observations": len(sample), "reference_return": ref["return"],
                      "reference_volatility": ref["volatility"], "reference_utility": ref["utility"]}
            if optimum.accepted:
                values = metrics(optimum.weights, moments, settings.risk_aversion, result.lower, result.upper)
                record.update({f"optimized_{k}": v for k, v in values.items()})
                record["utility_gain"] = values["utility"] - ref["utility"]
                record["weight_l1_from_reference"] = float(np.abs(optimum.weights - snapshot.weights).sum())
                base_solution = primary[name].solutions["utility"]
                record["weight_l1_from_primary_optimum"] = (float(np.abs(optimum.weights - base_solution.weights).sum())
                                                            if base_solution.accepted else None)
                record["volatility_reduction_at_reference_return"] = (
                    ref["volatility"] - metrics(matched.weights, moments, settings.risk_aversion)["volatility"]
                    if matched.accepted else None)
                weights.extend(allocation_rows(
                    snapshot.assets, optimum.weights, result.lower, result.upper,
                    case=label, comparison=name, portfolio="utility",
                ))
            records.append(record)
    return (pd.DataFrame(records, columns=SENSITIVITY_COLUMNS), pd.DataFrame(weights, columns=WEIGHT_COLUMNS),
            evidence, estimates)


def analyze(run_dir: Path, settings: DiagnosticSettings | None = None, *, paths=None) -> DiagnosticResult:
    """Build all-holdings diagnostics, retaining evidence when results are blocked.

    Insufficient history and rejected solutions populate result.blockers;
    invalid inputs and unexpected calculation failures raise instead.
    """
    settings = settings or DiagnosticSettings()
    snapshot, inputs, sources = load_inputs(run_dir, paths)
    history = build_history(inputs, snapshot, settings.lookback)
    project_root = Path(__file__).resolve().parents[1]
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")}
    versions["python"] = platform.python_version()
    provenance = {"schema_version": 1, "sources": sources, "implementation": implementation_receipt(project_root),
                  "versions": versions, "settings": asdict(settings),
                  "price_basis": inputs.price_basis.as_assumptions(),
                  "data_vintage_note": "Saved input hashes do not certify contemporaneously available provider data.",
                  "reference": "strategy formation targets: Signal_Raw_Target_Weight per original Sizing_NAV",
                  "return_definition": "stock-book return per sizing NAV, before financing and execution costs",
                  "snapshot": {"strategy_id": snapshot.strategy_id, "strategy_version": snapshot.strategy_version,
                               "rebalance_id": snapshot.rebalance_id, "formation_date": str(snapshot.formation_date.date()),
                               "execution_date": str(snapshot.execution_date.date()), "sizing_date": str(snapshot.sizing_date.date()),
                               "information_cutoff": str(snapshot.information_cutoff.date()), "sizing_nav": snapshot.sizing_nav,
                               "assets": snapshot.assets, "weights": snapshot.weights.tolist(),
                               "sector_neutral": snapshot.sector_neutral},
                  "calibration": {"BRTI": settings.brti, "q": settings.q, "lambda": settings.risk_aversion},
                  "numerical_tolerances": {"feasibility": FEASIBILITY_TOLERANCE,
                                           "annual_objective_gap": OPTIMALITY_TOLERANCE}}
    provenance["content_hash"] = hashlib.sha256(json.dumps(provenance, sort_keys=True, allow_nan=False).encode()).hexdigest()
    result = DiagnosticResult(snapshot=snapshot, settings=settings, history=history, comparisons={}, estimates={},
                              sensitivity=pd.DataFrame(columns=SENSITIVITY_COLUMNS),
                              sensitivity_weights=pd.DataFrame(columns=WEIGHT_COLUMNS),
                              sensitivity_evidence=[], provenance=provenance)
    if len(history.returns) < 12:
        coverage = history.coverage
        limiting = coverage.loc[coverage.Valid_Returns.eq(coverage.Valid_Returns.min()), "Asset_ID"].tolist()
        result.blockers.append(f"First formation has {len(history.returns)} complete monthly returns; 12 required. "
                               f"Limiting holdings: {', '.join(limiting)}. See missing_intervals.csv for all gaps.")
        return result
    moments = estimate_moments(history.returns)
    result.comparisons = {name: compare(snapshot, settings, moments, name) for name in ("bounded", "relaxed")}
    result.sensitivity, result.sensitivity_weights, result.sensitivity_evidence, result.estimates = _sensitivity(
        snapshot, settings, history, result.comparisons)
    for name, comparison in result.comparisons.items():
        failures = sum(not s.accepted for s in [*comparison.solutions.values(), *comparison.frontier])
        if failures:
            result.blockers.append(f"{name}: {failures} numerical solutions rejected; see solver diagnostics.")
    failed_sensitivity = sum(comparison["numerical_failures"] for case in result.sensitivity_evidence
                             for comparison in case["comparisons"].values())
    if failed_sensitivity:
        result.blockers.append(f"{failed_sensitivity} sensitivity numerical solutions rejected.")
    return result


def run_analysis(run_dir: Path, settings: DiagnosticSettings | None = None, output_root: Path | None = None) -> int:
    """Save evidence even with blockers; return 0 if complete or 2 if incomplete.

    Input, calculation and publication exceptions propagate.
    """
    from .outputs import save_outputs
    result = analyze(run_dir, settings)
    destination = save_outputs(result, output_root)
    print(f"Efficient-frontier diagnostic: {destination}")
    print(f"{len(result.snapshot.assets)} formation holdings; {len(result.history.returns)} common monthly returns")
    for blocker in result.blockers:
        print(f"Diagnostic incomplete: {blocker}")
    return 2 if result.blockers else 0
