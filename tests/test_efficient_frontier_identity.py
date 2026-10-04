"""Snapshot and causal monthly-history regressions for canonical securities."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from efficient_frontier.contracts import PortfolioSnapshot
from efficient_frontier.inputs import build_history, snapshot_from_tables


def snapshot(weights=(0.4, 0.3, -0.2, -0.1), assets=None, sector_neutral=False):
    assets = assets or [f"asset_{i}" for i in range(len(weights))]
    return PortfolioSnapshot("example", "1.0.0", "R1", pd.Timestamp("2026-02-27"),
                             pd.Timestamp("2026-03-02"), pd.Timestamp("2026-02-27"),
                             pd.Timestamp("2026-02-27"), 1_000_123.0,
                             pd.DataFrame({"Asset_ID": assets, "Signal_Raw_Target_Weight": weights,
                                           "GICS_Sector_Code": ["10"] * len(weights)}),
                             sector_neutral, {"strategy_id": "example", "strategy_version": "1.0.0", "parameters": {}},
                             {}, Path("/saved/example"))


def history_inputs(assets=("A", "B"), periods=16):
    rows = []
    for i, month in enumerate(pd.date_range(end="2026-03-31", periods=periods, freq="ME")):
        observation = month if month.weekday() < 5 else month - pd.offsets.BDay(1)
        for j, asset in enumerate(assets):
            rows.append({"Month": month, "Asset_ID": asset, "Observation_Date": observation,
                         "Available_Date": observation, "Close": 100 * (1.01 + j / 100) ** i})
    return SimpleNamespace(market_monthly=pd.DataFrame(rows),
                           corporate_actions=SimpleNamespace(events=pd.DataFrame(), legs=pd.DataFrame()))


def saved_tables():
    positions = snapshot().positions.copy()
    for key, value in {"Source_Ticker": "DISPLAY", "Yahoo_Ticker": "DISPLAY", "Sector": "Energy",
                       "Rebalance_ID": "R1", "Signal_Cutoff": "2026-02-27", "Sizing_Date": "2026-02-27",
                       "Signal_Source_Max_Date": "2026-02-27", "Execution_Date": "2026-03-02",
                       "Sector_As_Of_Date": "2026-02-27", "Sector_Source_Type": "reviewed", "Sector_Source_Reference": "1",
                       "Strategy_ID": "example", "Strategy_Version": "1.0.0", "Final_Target_Weight": 0.0}.items():
        positions[key] = value
    nav = pd.DataFrame({"Rebalance_ID": ["R1"], "Sizing_NAV": [1_000_123.0],
                        "Period_Start": ["2026-03-02"], "Period_Type": ["invested"]})
    return positions, nav


def test_snapshot_uses_formation_targets_and_sizing_nav_not_final_weights():
    decisions, nav = saved_tables()
    result = snapshot_from_tables(decisions.sample(frac=1, random_state=3), nav, snapshot().parameters, {}, "/saved")
    np.testing.assert_allclose(result.weights, snapshot().weights)
    assert result.sizing_nav == 1_000_123
    assert result.information_cutoff == pd.Timestamp("2026-02-27")


@pytest.mark.parametrize("fault", ["duplicate", "identity", "future_signal", "missing_nav", "duplicate_nav", "bad_nav", "bad_sector", "sizing"])
def test_malformed_first_formation_fails_instead_of_selecting_later(fault):
    decisions, nav = saved_tables()
    if fault == "duplicate":
        decisions = pd.concat([decisions, decisions.iloc[[0]]])
    elif fault == "identity":
        decisions["Strategy_ID"] = "other"
    elif fault == "future_signal":
        decisions.loc[0, "Signal_Source_Max_Date"] = "2026-03-01"
    elif fault == "missing_nav":
        nav = nav.iloc[0:0]
    elif fault == "duplicate_nav":
        nav = pd.concat([nav, nav])
    elif fault == "bad_nav":
        nav["Sizing_NAV"] = np.nan
    elif fault == "bad_sector":
        decisions.loc[0, "GICS_Sector_Code"] = ""
    else:
        decisions["Sizing_Date"] = "2026-03-02"
    later = decisions.copy()
    later["Signal_Cutoff"] = "2026-03-31"
    later["Rebalance_ID"] = "R2"
    with pytest.raises(ValueError):
        snapshot_from_tables(pd.concat([decisions, later]), nav, snapshot().parameters, {}, "/saved")


def test_reused_display_ticker_never_stitches_canonical_securities():
    data = history_inputs(("SNDK.O", "SNDK.O^E16"), 30)
    current = data.market_monthly.Asset_ID.eq("SNDK.O")
    data.market_monthly = data.market_monthly.loc[~(current & data.market_monthly.Month.lt("2025-02-28"))]
    result = build_history(data, snapshot([1.0], ["SNDK.O"]), 60)
    assert result.returns.columns.tolist() == ["SNDK.O"]
    assert len(result.returns) == 12
    assert result.returns.index.min() == pd.Timestamp("2025-03-31")
    assert result.returns.index.max() == pd.Timestamp("2026-02-28")


@pytest.mark.parametrize("fault,reason", [("absent", "missing_month"), ("available", "unavailable_evidence"),
                                         ("observation", "future_observation"), ("nan", "missing_positive_close")])
def test_missing_or_unavailable_month_never_creates_a_return_bridge(fault, reason):
    data = history_inputs()
    row = data.market_monthly.Asset_ID.eq("A") & data.market_monthly.Month.eq("2025-07-31")
    if fault == "absent":
        data.market_monthly = data.market_monthly.loc[~row]
    elif fault == "available":
        data.market_monthly.loc[row, "Available_Date"] = pd.Timestamp("2026-03-01")
    elif fault == "observation":
        data.market_monthly.loc[row, "Observation_Date"] = pd.Timestamp("2026-03-01")
    else:
        data.market_monthly.loc[row, "Close"] = np.nan
    result = build_history(data, snapshot([0.7, -0.3], ["A", "B"]), 12)
    assert pd.Timestamp("2025-07-31") not in result.returns.index
    assert pd.Timestamp("2025-08-31") not in result.returns.index
    assert len(result.returns) == 10
    assert reason in result.missing.Reason.values


def test_february_month_end_and_future_invariance():
    data = history_inputs()
    portfolio = snapshot([0.7, -0.3], ["A", "B"])
    before = build_history(data, portfolio, 12)
    assert len(before.returns) == 12
    assert pd.Timestamp("2026-02-28") in before.returns.index
    future = data.market_monthly.Month.gt("2026-02-28")
    data.market_monthly.loc[future, "Close"] *= 999
    after = build_history(data, portfolio, 12)
    pd.testing.assert_frame_equal(before.returns, after.returns)


def test_reviewed_event_return_is_used_and_unsupported_event_fails(monkeypatch):
    import efficient_frontier.inputs as module
    data = history_inputs()
    portfolio = snapshot([0.7, -0.3], ["A", "B"])
    monkeypatch.setattr(module, "has_interval_event", lambda events, legs, asset, start, end: asset == "A" and end.month == 7)
    monkeypatch.setattr(module, "interval_holding_returns", lambda *args, **kwargs: pd.Series({"A": 0.25}))
    result = build_history(data, portfolio, 12)
    assert result.returns.loc["2025-07-31", "A"] == 0.25
    def unsupported(*args, **kwargs):
        raise ValueError("unreviewed event")
    monkeypatch.setattr(module, "interval_holding_returns", unsupported)
    with pytest.raises(ValueError, match="unreviewed event"):
        build_history(data, portfolio, 12)
