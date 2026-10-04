"""Diagnostic tables, figures and numerical evidence."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from portfolio_core.artifacts import file_sha256
from portfolio_core.io import save_figure
from portfolio_core.plot_style import LINE_FIGSIZE, SERIES_COLORS, style_return_axes

from .contracts import METRIC_COLUMNS, WEIGHT_COLUMNS, allocation_rows
from .optimization import metrics

COLORS = {"bounded": SERIES_COLORS["Bounded frontier"], "relaxed": SERIES_COLORS["Relaxed frontier"]}


def _json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _tables(result):
    rows, weights, frontier, solver = [], [], [], []
    lam = result.settings.risk_aversion
    for name, comparison in result.comparisons.items():
        moments = comparison.moments
        ref = metrics(result.snapshot.weights, moments, lam, comparison.lower, comparison.upper)
        rows.append({"comparison": name, "portfolio": "formation_targets", "status": "accepted", **ref,
                     "utility_gain": 0.0, "volatility_reduction_at_reference_return": 0.0})
        for label, solution in comparison.solutions.items():
            solver.append({"comparison": name, "portfolio": label, "target_return": solution.target_return,
                           **solution.diagnostics, "accepted": solution.accepted})
            row = {"comparison": name, "portfolio": label, "status": "accepted" if solution.accepted else "failed"}
            if solution.accepted:
                values = metrics(solution.weights, moments, lam, comparison.lower, comparison.upper)
                row.update(values)
                row["utility_gain"] = values["utility"] - ref["utility"]
                if label == "reference_return":
                    row["volatility_reduction_at_reference_return"] = ref["volatility"] - values["volatility"]
                weights.extend(allocation_rows(
                    result.snapshot.assets, solution.weights, comparison.lower, comparison.upper,
                    case="primary", comparison=name, portfolio=label,
                ))
            rows.append(row)
        for i, solution in enumerate(comparison.frontier):
            row = {"comparison": name, "point": i, "target_return": solution.target_return,
                   "status": "accepted" if solution.accepted else "failed"}
            if solution.accepted:
                row.update(metrics(solution.weights, moments, lam, comparison.lower, comparison.upper))
                weights.extend(allocation_rows(
                    result.snapshot.assets, solution.weights, comparison.lower, comparison.upper,
                    case="primary", comparison=name, portfolio=f"frontier_{i}",
                ))
            frontier.append(row)
            solver.append({"comparison": name, "portfolio": f"frontier_{i}", "target_return": solution.target_return,
                           **solution.diagnostics, "accepted": solution.accepted})
    summary_columns = ("comparison", "portfolio", "status", *METRIC_COLUMNS,
                       "utility_gain", "volatility_reduction_at_reference_return")
    frontier_columns = ("comparison", "point", "target_return", "status", *METRIC_COLUMNS)
    return (pd.DataFrame(rows, columns=summary_columns), pd.DataFrame(weights, columns=WEIGHT_COLUMNS),
            pd.DataFrame(frontier, columns=frontier_columns), solver)


def _plot(result, destination):
    if not result.comparisons:
        return
    lam = result.settings.risk_aversion
    fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
    all_metrics = []
    reference = metrics(result.snapshot.weights, result.comparisons["bounded"].moments, lam)
    all_metrics.append(reference)
    for name, comparison in result.comparisons.items():
        points = [metrics(s.weights, comparison.moments, lam) if s.accepted else None for s in comparison.frontier]
        # NaNs break lines at rejected points; never visually interpolate a failure.
        ax.plot([100 * p["volatility"] if p else np.nan for p in points],
                [100 * p["return"] if p else np.nan for p in points], color=COLORS[name],
                linestyle="-" if name == "bounded" else "--", linewidth=2.2,
                marker="." if len(points) == 1 else None, label=f"{name.capitalize()} frontier")
        all_metrics.extend(p for p in points if p)
        for kind, marker, title in (("gmv", "s", "GMV"), ("utility", "*", "utility optimum")):
            solution = comparison.solutions[kind]
            if solution.accepted:
                value = metrics(solution.weights, comparison.moments, lam)
                all_metrics.append(value)
                ax.scatter(100 * value["volatility"], 100 * value["return"], color=COLORS[name], marker=marker,
                           s=125 if kind == "utility" else 48, edgecolors="white", linewidths=0.7,
                           zorder=4, label=f"{name.capitalize()} {title}")
    ax.scatter(100 * reference["volatility"], 100 * reference["return"], marker="D", color=SERIES_COLORS["Strategy"], s=65,
               label="Strategy formation targets", zorder=5)
    xmax = max(v["volatility"] for v in all_metrics) * 1.13 or 0.01
    ymin, ymax = min(v["return"] for v in all_metrics), max(v["return"] for v in all_metrics)
    margin = max((ymax - ymin) * 0.15, 0.03)
    grid = np.linspace(0, xmax, 300)
    curves = [(reference["utility"], SERIES_COLORS["Strategy"])]
    for name, comparison in result.comparisons.items():
        optimum = comparison.solutions["utility"]
        if optimum.accepted:
            curves.append((metrics(optimum.weights, comparison.moments, lam)["utility"], COLORS[name]))
    label_ceiling = ymax + margin - 0.06 * (ymax - ymin + 2 * margin)
    for i, (utility, color) in enumerate(curves):
        ax.plot(100 * grid, 100 * (utility + lam * grid ** 2 / 2), color=color, alpha=0.55, linestyle=":", linewidth=1.1,
                label="Indifference curves" if i == 0 else None)
        # Label the visible right end, leaving room below the top of the axes.
        label_x = min(0.985 * xmax, np.sqrt(max(0, 2 * (label_ceiling - utility) / lam)))
        label_y = utility + lam * label_x ** 2 / 2
        # Follow the curve's tangent so the text does not extend over a higher curve.
        ax.text(100 * label_x, 100 * label_y, f"U = {utility:.4f}",
                rotation=np.degrees(np.arctan(lam * label_x)), transform_rotates_text=True,
                rotation_mode="anchor", ha="right", va="bottom", fontsize=9, color=color, zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
    ax.set(xlim=(0, 100 * xmax), ylim=(100 * (ymin - margin), 100 * (ymax + margin)))
    title = (f"{result.snapshot.strategy_id.replace('_', ' ').title()} Efficient Frontier at First Formation\n"
             f"{result.snapshot.formation_date.date()} · {len(result.snapshot.assets)} holdings · "
             f"{len(result.history.returns)} monthly returns")
    style_return_axes(ax, title=title, xlabel="Estimated annual volatility (%)",
                      ylabel="Estimated annual arithmetic return (%)")
    ax.legend()
    fig.text(0.5, 0.015, "Fitted estimates, not realized performance.\n"
             "Stock-book returns before financing and execution costs.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, destination / "frontier.png")


def save_outputs(result, output_root: Path | None = None) -> Path:
    """Replace a strategy's current diagnostic only after every artifact is built."""
    project = Path(__file__).resolve().parents[1]
    root = (Path(output_root) if output_root is not None else project / "outputs/efficient_frontier").resolve()
    destination = root / result.snapshot.strategy_id
    resolved_destination = destination.resolve()
    protected = [path.resolve() for path in (
        result.snapshot.run_dir, project / "data", project / "outputs/live", project / "outputs/backtest",
    )]
    for path in protected:
        if root == path or root.is_relative_to(path):
            raise ValueError(f"Diagnostic output root {root} must be separate from protected directory {path}")
        # Publication replaces the whole destination, including any saved run beneath it.
        if (resolved_destination == path or resolved_destination.is_relative_to(path)
                or path.is_relative_to(resolved_destination)):
            raise ValueError(f"Diagnostic destination {destination} must be separate from protected directory {path}")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".frontier-", dir=root))
    staged, previous = temporary / "current", temporary / "previous"
    try:
        for folder in ("figures", "tables", "metadata"):
            (staged / folder).mkdir(parents=True)
        metadata = staged / "metadata"
        summary, weights, frontier, solver = _tables(result)
        weights = pd.concat([weights, result.sensitivity_weights], ignore_index=True)
        tables = {"summary": summary, "weights": weights, "frontier": frontier,
                  "formation_targets": result.snapshot.positions, "coverage": result.history.coverage,
                  "missing_intervals": result.history.missing, "sensitivity": result.sensitivity}
        for name, table in tables.items():
            table.to_csv(staged / "tables" / f"{name}.csv", index=False)
        result.history.returns.to_csv(staged / "tables/returns.csv", index_label="Month")
        _json(metadata / "diagnostics.json", {
            "schema_version": 1,
            "provenance": result.provenance,
            "strategy_parameters": result.snapshot.parameters,
            "simulation_assumptions": result.snapshot.assumptions,
            "estimates": {name: {"mean": moments.mean.tolist(), "covariance": moments.covariance.tolist(),
                                 "diagnostics": moments.diagnostics} for name, moments in result.estimates.items()},
            "comparison_estimates": {name: "primary" for name in result.comparisons},
            "solver_checks": solver,
            "sensitivity": result.sensitivity_evidence,
            "status": {"complete": not result.blockers, "blockers": result.blockers},
        })
        _plot(result, staged / "figures")
        _json(metadata / "artifact_manifest.json", {"content_hash": result.provenance["content_hash"],
              "files": {p.relative_to(staged).as_posix(): file_sha256(p) for p in sorted(staged.rglob("*")) if p.is_file()}})
        if destination.exists():
            destination.rename(previous)
        staged.rename(destination)
    finally:
        # Restore the preceding diagnostic if publication fails after moving it aside.
        if previous.exists() and not destination.exists():
            previous.rename(destination)
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination
