"""Known convex solutions and independent certification of numerical results."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.contracts import DiagnosticSettings, Moments
from efficient_frontier.optimization import compare, estimate_moments, feasible_set, metrics, solve
from test_efficient_frontier_identity import snapshot


@pytest.mark.parametrize("kwargs", [{"lookback": 11}, {"lookback": 12.5}, {"floor_multiplier": 0},
                                    {"floor_multiplier": 1.1}, {"cap_multiplier": 0.9},
                                    {"cap_multiplier": float("inf")}, {"brti": float("nan")}, {"brti": 8}])
def test_settings_reject_invalid_overrides(kwargs):
    with pytest.raises(ValueError):
        DiagnosticSettings(**kwargs)


def test_calibration_and_singular_sample_covariance():
    settings = DiagnosticSettings()
    assert settings.q == pytest.approx(0.7)
    assert settings.risk_aversion == pytest.approx(2.8740685648340483)
    sample = pd.DataFrame(np.random.default_rng(42).normal(0, 0.1, (12, 30)),
                          index=pd.date_range("2025-03-31", periods=12, freq="ME"))
    moments = estimate_moments(sample)
    assert moments.diagnostics["sample_covariance_rank"] <= 11
    assert moments.diagnostics["covariance_rank"] == 30
    np.testing.assert_allclose(moments.mean, sample.mean() * 12)
    half = estimate_moments(sample, mean_shrinkage=0.5)
    np.testing.assert_allclose(half.mean, (sample.mean() + sample.mean().mean()) * 6)
    diagonal = estimate_moments(sample, diagonal_shrinkage=0.2)
    covariance = sample.cov().to_numpy()
    np.testing.assert_allclose(diagonal.covariance, 12 * (0.8 * covariance + 0.2 * np.diag(np.diag(covariance))))


def test_known_two_asset_gmv_utility_and_efficient_branch():
    portfolio = snapshot([0.5, 0.5])
    moments = Moments(tuple(portfolio.assets), np.array([0.1, 0.2]), np.diag([0.04, 0.09]), {})
    settings = DiagnosticSettings()
    comparison = compare(portfolio, settings, moments, "relaxed")
    np.testing.assert_allclose(comparison.solutions["gmv"].weights, [9/13, 4/13], atol=1e-6)
    analytic_w0 = (0.09 - 0.1 / settings.risk_aversion) / 0.13
    np.testing.assert_allclose(comparison.solutions["utility"].weights, [analytic_w0, 1-analytic_w0], atol=1e-6)
    assert len(comparison.frontier) == 100
    assert all(s.accepted for s in comparison.frontier)
    values = [metrics(s.weights, moments, settings.risk_aversion) for s in comparison.frontier]
    assert np.diff([v["return"] for v in values]).min() >= -1e-8
    assert np.diff([v["volatility"] for v in values]).min() >= -1e-8
    assert comparison.frontier[-1].target_return == pytest.approx(0.2)
    assert values[-1]["return"] == pytest.approx(0.2)


def test_bounds_signs_sleeves_and_asset_order_invariance():
    portfolio = snapshot()
    moments = Moments(tuple(portfolio.assets), np.array([0.1, 0.2, -0.1, 0.05]), np.diag([0.04, 0.09, 0.02, 0.07]), {})
    original = compare(portfolio, DiagnosticSettings(), moments, "bounded", frontier=False)
    weights = original.solutions["utility"].weights
    assert np.all(weights >= original.lower - 1e-8) and np.all(weights <= original.upper + 1e-8)
    assert weights[weights > 0].sum() == pytest.approx(0.7)
    assert weights[weights < 0].sum() == pytest.approx(-0.3)
    order = np.array([2, 0, 3, 1])
    reordered = replace(portfolio, positions=portfolio.positions.iloc[order])
    changed = compare(reordered, DiagnosticSettings(),
                      Moments(tuple(reordered.assets), moments.mean[order], moments.covariance[np.ix_(order, order)], {}),
                      "bounded", frontier=False)
    np.testing.assert_allclose(changed.solutions["utility"].weights, weights[order], atol=1e-7)
    with pytest.raises(ValueError, match="asset order"):
        compare(reordered, DiagnosticSettings(), moments, "bounded")


def test_sector_neutrality_retains_zero_net_not_sector_gross_and_removes_redundancy():
    portfolio = snapshot([0.25, 0.25, -0.25, -0.25], sector_neutral=True)
    portfolio.positions["GICS_Sector_Code"] = ["10", "20", "10", "20"]
    feasible = feasible_set(portfolio, 0, None)
    assert len(feasible.redundant) == 1
    moments = Moments(tuple(portfolio.assets), np.array([0.5, 0.1, -0.1, 0.1]), np.eye(4) * 0.01, {})
    result = solve(moments, feasible, portfolio.weights, 3, "utility")
    assert result.accepted
    assert result.weights[0] + result.weights[2] == pytest.approx(0, abs=1e-8)
    assert result.weights[1] + result.weights[3] == pytest.approx(0, abs=1e-8)
    assert result.weights[0] > 0.25
    with pytest.raises(ValueError, match="infeasible"):
        feasible_set(snapshot(sector_neutral=True), 0.5, 2)


@pytest.mark.parametrize("fixed", [False, True])
def test_degenerate_frontier_is_one_point(fixed):
    portfolio = snapshot([0.5, 0.5])
    moments = Moments(tuple(portfolio.assets), np.zeros(2), np.zeros((2, 2)), {})
    settings = DiagnosticSettings(floor_multiplier=1, cap_multiplier=1) if fixed else DiagnosticSettings()
    result = compare(portfolio, settings, moments, "bounded")
    assert len(result.frontier) == 1 and result.frontier[0].accepted


@pytest.mark.parametrize("comparison", ["bounded", "relaxed"])
@pytest.mark.parametrize("variance", [0.0, 0.04])
def test_flat_risk_frontier_keeps_only_the_highest_return(comparison, variance):
    portfolio = snapshot([0.5, 0.5])
    moments = Moments(tuple(portfolio.assets), np.array([0.1, 0.2]), np.full((2, 2), variance), {})
    result = compare(portfolio, DiagnosticSettings(), moments, comparison)
    expected = [0.25, 0.75] if comparison == "bounded" else [0.0, 1.0]
    assert result.solutions["gmv"].accepted
    np.testing.assert_allclose(result.solutions["gmv"].weights, expected, atol=1e-8)
    assert len(result.frontier) == 1 and result.frontier[0].accepted
    np.testing.assert_allclose(result.frontier[0].weights, expected, atol=1e-8)
    assert result.frontier[0].weights @ moments.covariance @ result.frontier[0].weights == pytest.approx(variance)


def test_partially_degenerate_frontier_starts_at_highest_return_gmv():
    portfolio = snapshot([0.25, 0.25, 0.5])
    moments = Moments(tuple(portfolio.assets), np.array([0.1, 0.2, 0.3]), np.diag([0.0, 0.0, 0.04]), {})
    result = compare(portfolio, DiagnosticSettings(), moments, "relaxed")
    np.testing.assert_allclose(result.solutions["gmv"].weights, [0, 1, 0], atol=1e-8)
    assert len(result.frontier) == 100
    assert result.frontier[0].target_return == pytest.approx(0.2)
    for point in result.frontier:
        assert point.accepted
        risky_weight = (point.target_return - 0.2) / 0.1
        np.testing.assert_allclose(point.weights, [0, 1 - risky_weight, risky_weight], atol=1e-6)


def test_nearly_singular_covariance_retains_the_risk_return_tradeoff():
    portfolio = snapshot([0.5, 0.5])
    covariance = np.full((2, 2), 0.04) + np.eye(2) * 1e-10
    moments = Moments(tuple(portfolio.assets), np.array([0.1, 0.2]), covariance, {})
    result = compare(portfolio, DiagnosticSettings(), moments, "relaxed")
    np.testing.assert_allclose(result.solutions["gmv"].weights, [0.5, 0.5], atol=1e-8)
    assert len(result.frontier) == 100 and all(point.accepted for point in result.frontier)
    first, last = result.frontier[0].weights, result.frontier[-1].weights
    assert last @ covariance @ last > first @ covariance @ first


def test_constant_return_history_produces_one_efficient_point():
    portfolio = snapshot([0.5, 0.5])
    returns = pd.DataFrame(np.tile([0.015625, 0.03125], (12, 1)), columns=portfolio.assets,
                           index=pd.date_range("2025-03-31", periods=12, freq="ME"))
    moments = estimate_moments(returns)
    result = compare(portfolio, DiagnosticSettings(), moments, "relaxed")
    assert moments.diagnostics["covariance_rank"] == 0
    assert len(result.frontier) == 1 and result.frontier[0].accepted
    np.testing.assert_allclose(result.frontier[0].weights, [0, 1], atol=1e-8)


def test_gmv_return_tiebreak_cannot_accept_a_risk_changing_solution(monkeypatch):
    import efficient_frontier.optimization as module
    portfolio = snapshot([0.25, 0.25, 0.5])
    moments = Moments(tuple(portfolio.assets), np.array([0.1, 0.2, 0.3]), np.diag([0.0, 0.0, 0.04]), {})
    original_linear = module.FeasibleSet.linear

    def claimed_success(feasible, objective, mean=None, target=None):
        if len(feasible.a) > 1:
            weights = np.array([0.0, 0.0, 1.0])
            return SimpleNamespace(x=weights, fun=objective @ weights, success=True, status=0,
                                   message="claimed success", nit=1)
        return original_linear(feasible, objective, mean, target)

    monkeypatch.setattr(module.FeasibleSet, "linear", claimed_success)
    result = compare(portfolio, DiagnosticSettings(), moments, "relaxed")
    gmv = result.solutions["gmv"]
    assert not gmv.accepted and gmv.weights is None and not result.frontier
    assert gmv.diagnostics["return_tiebreak"]["constraint_residual"] > 1e-8


def test_sensitivity_comparison_does_not_request_unused_solutions(monkeypatch):
    import efficient_frontier.optimization as module
    portfolio = snapshot([0.5, 0.5])
    moments = Moments(tuple(portfolio.assets), np.array([0.1, 0.2]), np.diag([0.04, 0.09]), {})
    primary = compare(portfolio, DiagnosticSettings(), moments, "relaxed")
    original_solve = module.solve
    requested = []

    def only_requested_solutions(moments, feasible, reference, risk_aversion, kind, **kwargs):
        assert kind in {"utility", "reference_return"}, f"Unnecessary sensitivity solve: {kind}"
        requested.append(kind)
        return original_solve(moments, feasible, reference, risk_aversion, kind, **kwargs)

    monkeypatch.setattr(module, "solve", only_requested_solutions)
    result = compare(portfolio, DiagnosticSettings(), moments, "relaxed", frontier=False)
    assert requested == ["utility", "reference_return"]
    assert set(result.solutions) == set(requested) and not result.frontier
    for kind, solution in result.solutions.items():
        assert solution.accepted
        np.testing.assert_allclose(solution.weights, primary.solutions[kind].weights, atol=1e-10)


def test_solver_success_alone_cannot_pass_independent_optimality_check(monkeypatch):
    import efficient_frontier.optimization as module
    portfolio = snapshot([0.5, 0.5])
    moments = Moments(tuple(portfolio.assets), np.array([0.1, 0.5]), np.eye(2) * 0.01, {})
    monkeypatch.setattr(module, "minimize", lambda *args, **kwargs: SimpleNamespace(
        x=portfolio.weights, success=True, status=0, message="claimed success", nit=1))
    result = solve(moments, feasible_set(portfolio, 0, None), portfolio.weights, 3, "utility")
    assert not result.accepted and result.weights is None
    assert result.diagnostics["first_order_gap"] > 1e-6


@pytest.mark.parametrize("covariance", [np.array([[1, 2], [0, 1]]), np.diag([1, -0.1]), np.diag([1, np.nan])])
def test_invalid_covariance_cannot_reach_optimizer(covariance):
    portfolio = snapshot([0.5, 0.5])
    with pytest.raises(ValueError, match="Covariance"):
        solve(Moments(tuple(portfolio.assets), np.zeros(2), covariance, {}),
              feasible_set(portfolio, 0, None), portfolio.weights, 3, "gmv")
