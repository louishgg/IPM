"""Convex mean-variance diagnostics with independently certified solutions."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from sklearn.covariance import LedoitWolf

from .contracts import Comparison, DiagnosticSettings, Moments, PortfolioSnapshot, Solution

FEASIBILITY_TOLERANCE = 1e-8
OPTIMALITY_TOLERANCE = 1e-6
FRONTIER_POINTS = 100


def estimate_moments(returns: pd.DataFrame, *, mean_shrinkage=0.0, diagonal_shrinkage=None) -> Moments:
    """Annual arithmetic moments; LW uses n, diagonal cases use n-1 normalization."""
    x = returns.to_numpy(dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("Moment estimation requires at least two finite return observations")
    if returns.columns.duplicated().any():
        raise ValueError("Moment estimation requires unique canonical assets")
    mean = x.mean(axis=0)
    mean = (1 - mean_shrinkage) * mean + mean_shrinkage * mean.mean()
    sample = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    if diagonal_shrinkage is None:
        fit = LedoitWolf().fit(x)
        covariance, shrinkage = fit.covariance_, float(fit.shrinkage_)
        estimator = "Ledoit-Wolf, centered ML covariance (divide by n), spherical target trace/n_assets"
    else:
        shrinkage = float(diagonal_shrinkage)
        if not 0 <= shrinkage <= 1:
            raise ValueError("Diagonal shrinkage must lie in [0, 1]")
        covariance = (1 - shrinkage) * sample + shrinkage * np.diag(np.diag(sample))
        estimator = "Diagonal shrinkage of unbiased sample covariance (divide by n-1)"
    covariance = np.asarray(covariance) * 12
    validate_covariance(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    condition = float(np.linalg.cond(covariance))
    diagnostics = {"observations": len(x), "assets": x.shape[1], "mean_shrinkage": mean_shrinkage,
                   "mean_estimator": "arithmetic monthly mean, annualized x12",
                   "covariance_estimator": estimator, "shrinkage": shrinkage,
                   "sample_covariance_rank": int(np.linalg.matrix_rank(sample)),
                   "covariance_rank": int(np.linalg.matrix_rank(covariance)),
                   "annual_eigenvalues": eigenvalues.tolist(),
                   "condition_number": condition if np.isfinite(condition) else None,
                   "sample_dates": [str(v.date()) for v in returns.index]}
    return Moments(tuple(returns.columns), mean * 12, covariance, diagnostics)


def validate_covariance(covariance):
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12)):
        raise ValueError("Covariance must be finite, square and symmetric")
    if np.linalg.eigvalsh(covariance).min() < -1e-12:
        raise ValueError("Covariance is not positive semidefinite")


@dataclass(frozen=True)
class FeasibleSet:
    lower: np.ndarray
    upper: np.ndarray
    a: np.ndarray
    b: np.ndarray
    all_a: np.ndarray
    all_b: np.ndarray
    redundant: tuple[str, ...]

    def residual(self, weights, mean=None, target=None) -> float:
        residuals = [0.0, float(np.max(self.lower - weights)), float(np.max(weights - self.upper)),
                     float(np.max(np.abs(self.all_a @ weights - self.all_b)))]
        if target is not None:
            residuals.append(float(target - mean @ weights))
        return max(residuals)

    def linear(self, objective, mean=None, target=None):
        return linprog(objective, A_eq=self.a, b_eq=self.b,
                       A_ub=None if target is None else -mean[None, :],
                       b_ub=None if target is None else [-target],
                       bounds=list(zip(self.lower, self.upper)), method="highs",
                       options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})


def _independent_rows(matrix):
    keep = []
    for i in range(len(matrix)):
        if np.linalg.matrix_rank(matrix[keep + [i]]) > len(keep):
            keep.append(i)
    return keep


def feasible_set(snapshot: PortfolioSnapshot, floor: float, cap: float | None) -> FeasibleSet:
    """Build signed NAV-weight bounds from nonzero reference formation targets.

    Floor/cap multiply each reference magnitude; cap=None allows its side's
    full gross exposure. Keep side totals fixed and, when sector_neutral, each
    sector's net at zero; invalid or infeasible references raise ValueError.
    """
    reference = snapshot.weights
    if not np.isfinite(reference).all() or np.any(reference == 0) or not len(reference):
        raise ValueError("Reference must contain every nonzero finite formation weight")
    long, short = reference > 0, reference < 0
    magnitude_lower = floor * np.abs(reference)
    magnitude_upper = (np.where(long, reference[long].sum(), -reference[short].sum())
                       if cap is None else cap * np.abs(reference))
    lower = np.where(long, magnitude_lower, -magnitude_upper)
    upper = np.where(long, magnitude_upper, -magnitude_lower)
    rows, totals, labels = [], [], []
    for mask, name in ((long, "long_sleeve"), (short, "short_sleeve")):
        if mask.any():
            rows.append(mask.astype(float))
            totals.append(float(reference[mask].sum()))
            labels.append(name)
    if snapshot.sector_neutral:
        sectors = snapshot.positions.GICS_Sector_Code.astype(str).to_numpy()
        for sector in sorted(set(sectors)):
            rows.append((sectors == sector).astype(float))
            totals.append(0.0)
            labels.append(f"sector_net_{sector}")
    all_a, all_b = np.array(rows), np.array(totals)
    keep = _independent_rows(all_a)
    redundant = tuple(name for i, name in enumerate(labels) if i not in keep)
    feasible = FeasibleSet(lower, upper, all_a[keep], all_b[keep], all_a, all_b, redundant)
    if feasible.residual(reference) > FEASIBILITY_TOLERANCE:
        raise ValueError("Reference formation targets are infeasible under the requested bounds/sleeves/sector neutrality")
    return feasible


def certify(weights, gradient, feasible, *, mean=None, target=None) -> dict:
    """Check feasibility and bound convex suboptimality by grad f(w) dot (w-z).

    The linear solve chooses z minimizing grad f(w) dot z over the feasible
    set; both the constraint residual and this gap must pass their tolerances.
    """
    if not np.isfinite(weights).all() or not np.isfinite(gradient).all():
        return {"accepted": False, "reason": "nonfinite solution or gradient"}
    residual = feasible.residual(weights, mean, target)
    certificate = feasible.linear(gradient, mean, target)
    gap = float(gradient @ weights - certificate.fun) if certificate.success else None
    accepted = (residual <= FEASIBILITY_TOLERANCE and gap is not None
                and -OPTIMALITY_TOLERANCE <= gap <= OPTIMALITY_TOLERANCE)
    return {"accepted": bool(accepted), "constraint_residual": residual,
            "first_order_gap": None if gap is None else max(0.0, gap),
            "certificate_status": int(certificate.status), "certificate_message": str(certificate.message)}


def solve(moments: Moments, feasible: FeasibleSet, reference, risk_aversion, kind, *, target=None, start=None) -> Solution:
    """Optimize signed weights using annualized means and covariance.

    Utility maximizes mean minus risk_aversion * variance / 2; maximum_return
    maximizes mean alone; other kinds minimize variance. Quadratic modes use
    target as a return lower bound and start from reference unless start is set.
    Solver/certificate rejection returns accepted=False with no weights;
    invalid moments and solver exceptions propagate.
    """
    mu, cov = moments.mean, moments.covariance
    validate_covariance(cov)
    if len(mu) != len(reference) or not np.isfinite(mu).all():
        raise ValueError("Expected returns and reference must agree and be finite")
    if kind == "maximum_return":
        result = feasible.linear(-mu)
        if not result.success:
            return Solution(False, None, {"solver_status": int(result.status), "message": result.message})
        checks = certify(result.x, -mu, feasible)
    else:
        factor, linear = (risk_aversion, mu) if kind == "utility" else (1.0, np.zeros_like(mu))
        def objective(w):
            return float(factor * (w @ cov @ w) / 2 - linear @ w)
        def gradient(w):
            return factor * (cov @ w) - linear
        constraints = [{"type": "eq", "fun": lambda w: feasible.a @ w - feasible.b,
                        "jac": lambda w: feasible.a}]
        if target is not None:
            constraints.append({"type": "ineq", "fun": lambda w: mu @ w - target, "jac": lambda w: mu})
        result = minimize(objective, reference if start is None else start, jac=gradient,
                          bounds=list(zip(feasible.lower, feasible.upper)), constraints=constraints,
                          method="SLSQP", options={"ftol": 1e-12, "maxiter": 2000})
        checks = certify(result.x, gradient(result.x), feasible, mean=mu, target=target)
    accepted = bool(result.success and checks["accepted"])
    # SciPy returns no status code when it resolves an entirely fixed box.
    status = getattr(result, "status", None)
    diagnostics = {**checks, "accepted": accepted, "solver_status": None if status is None else int(status),
                   "message": str(result.message), "iterations": int(getattr(result, "nit", 0)),
                   "redundant_equalities": list(feasible.redundant)}
    return Solution(accepted, result.x.copy() if accepted else None, diagnostics, target)


def _highest_return_gmv(moments, feasible, gmv):
    """Choose the highest-return GMV by moving within the covariance null space.

    Recertify feasibility and optimality, and check that variance is unchanged.
    """
    if not gmv.accepted:
        return gmv
    eigenvalues, eigenvectors = np.linalg.eigh(moments.covariance)
    rank_tolerance = len(eigenvalues) * np.finfo(float).eps * np.max(np.abs(eigenvalues))
    directions = eigenvectors[:, np.abs(eigenvalues) > rank_tolerance].T
    if len(directions) == len(eigenvalues):
        return gmv
    # Preserve every risk-bearing exposure; maximize return only along zero-risk directions.
    all_a = np.vstack((feasible.all_a, directions))
    all_b = np.concatenate((feasible.all_b, directions @ gmv.weights))
    keep = _independent_rows(all_a)
    redundant = feasible.redundant + tuple(f"covariance_direction_{i}" for i in range(len(directions))
                                         if len(feasible.all_a) + i not in keep)
    face = replace(feasible, a=all_a[keep], b=all_b[keep], all_a=all_a, all_b=all_b, redundant=redundant)
    tie = solve(moments, face, gmv.weights, 0.0, "maximum_return")
    diagnostics = {**gmv.diagnostics, "return_tiebreak": {
        **tie.diagnostics, "covariance_rank": len(directions), "rank_tolerance": float(rank_tolerance)}}
    if not tie.accepted:
        return Solution(False, None, {**diagnostics, "accepted": False})
    covariance = moments.covariance
    checks = certify(tie.weights, covariance @ tie.weights, feasible)
    variance = float(gmv.weights @ covariance @ gmv.weights)
    variance_change = float(tie.weights @ covariance @ tie.weights - variance)
    variance_tolerance = 1e-12 * max(1.0, abs(variance))
    accepted = bool(checks["accepted"] and abs(variance_change) <= variance_tolerance)
    diagnostics.update(checks)
    diagnostics.update(accepted=accepted, variance_change=variance_change, variance_tolerance=variance_tolerance)
    return Solution(accepted, tie.weights if accepted else None, diagnostics)


def compare(snapshot, settings: DiagnosticSettings, moments: Moments, name, *, floor=None, cap=None, frontier=True):
    """Compare reference targets and optima with matching canonical asset order.

    Bounded comparisons default to configured multipliers; relaxed ones allow
    zero weights and each side's full gross. Frontier targets span GMV to maximum
    return, retaining rejected points but using only accepted weights as starts.
    """
    if tuple(snapshot.assets) != moments.assets:
        raise ValueError("Moment asset order disagrees with the canonical snapshot")
    bounded = name == "bounded"
    floor = (settings.floor_multiplier if bounded else 0.0) if floor is None else floor
    cap = (settings.cap_multiplier if bounded else None) if cap is None else cap
    feasible = feasible_set(snapshot, floor, cap)
    reference = snapshot.weights
    kinds = ("gmv", "utility", "maximum_return") if frontier else ("utility",)
    solutions = {kind: solve(moments, feasible, reference, settings.risk_aversion, kind) for kind in kinds}
    solutions["reference_return"] = solve(moments, feasible, reference, settings.risk_aversion,
                                           "reference_return", target=float(moments.mean @ reference))
    points = []
    if frontier:
        solutions["gmv"] = _highest_return_gmv(moments, feasible, solutions["gmv"])
    if frontier and solutions["gmv"].accepted and solutions["maximum_return"].accepted:
        gmv, maximum = solutions["gmv"], solutions["maximum_return"]
        low, high = float(moments.mean @ gmv.weights), float(moments.mean @ maximum.weights)
        if high < low - FEASIBILITY_TOLERANCE:
            raise ValueError("Certified maximum return falls below GMV return")
        # At most tolerance-size ranges are numerically the same efficient point.
        targets = [low] if abs(high - low) <= 1e-10 else np.linspace(low, high, FRONTIER_POINTS)
        previous = gmv.weights
        for target in targets:
            point = solve(moments, feasible, reference, settings.risk_aversion,
                          "frontier", target=float(target), start=previous)
            points.append(point)
            if point.accepted:
                previous = point.weights
    return Comparison(moments, feasible.lower, feasible.upper, solutions, points)


def metrics(weights, moments, risk_aversion, lower=None, upper=None):
    variance = float(weights @ moments.covariance @ weights)
    gross = float(np.abs(weights).sum())
    long_count, short_count = int((weights > 1e-8).sum()), int((weights < -1e-8).sum())
    mean = float(moments.mean @ weights)
    return {"return": mean, "volatility": float(np.sqrt(max(0, variance))),
            "utility": mean - risk_aversion * variance / 2,
            "long_count": long_count, "short_count": short_count, "gross": gross, "net": float(weights.sum()),
            "largest_position": float(np.abs(weights).max()),
            "concentration_hhi": float(np.sum((weights / gross) ** 2)),
            "binding_lower": None if lower is None else int(np.isclose(weights, lower, atol=1e-8, rtol=0).sum()),
            "binding_upper": None if upper is None else int(np.isclose(weights, upper, atol=1e-8, rtol=0).sum()),
            "below_historical_position_counts": long_count < 10 or short_count < 10 or long_count + short_count < 20}
