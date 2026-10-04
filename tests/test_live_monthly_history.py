"""Offline input acceptance and adverse cases; no strategy search or performance."""
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from live.analysis_data import load_analysis_inputs, membership_as_of
from live.config import DEFAULT_CONFIG
from live.monthly_history import (
    aggregate_monthly_market, compose_monthly_history, load_monthly_evidence,
    monthly_matrices, validate_dividends, validate_monthly_history,
)
from live.research_history import (
    build_live_research_requirements, interval_holding_returns, prepare_live_research_history,
)
from portfolio_core.corporate_actions import apply_corporate_actions
from portfolio_core.interval_valuation import value_one_share_through_events
from portfolio_core.strategies.portfolio_construction import stock_volatility
from portfolio_core.strategies.research_parameters import (
    SignalParameters, SizingParameters, VolatilityProfile, WholeSectorSelection,
)
from portfolio_core.strategies.sector_momentum import SectorMomentumStrategy
from _strategy_test_helpers import momentum_test_parameters


@pytest.fixture(scope="module")
def inputs():
    return load_analysis_inputs(DEFAULT_CONFIG.paths)


def test_required_repairs_are_unique_traceable_and_leave_daily_volume_real(inputs):
    quotes, dividends = load_monthly_evidence(DEFAULT_CONFIG.paths, inputs.market_daily)
    assert len(quotes.loc[quotes.Role.eq("fallback")]) == 65
    assert len(quotes.loc[quotes.Role.eq("override")]) == 1
    repaired = inputs.market_monthly.loc[inputs.market_monthly.Price_Source.isin(
        ["reuters_reconstructed", "reviewed_monthly_correction"])]
    assert len(repaired) == 66
    assert not repaired.duplicated(["Month", "Asset_ID"]).any()
    assert set(repaired.Evidence_ID) <= set(quotes.Evidence_ID)
    assert repaired.loc[repaired.Price_Source.eq("reuters_reconstructed"), "Volume"].isna().all()
    observed = aggregate_monthly_market(inputs.market_daily, through=inputs.schedule.Signal_Cutoff.max())
    ordinary = inputs.market_monthly.loc[~inputs.market_monthly.Price_Source.isin(
        ["reuters_reconstructed", "reviewed_monthly_correction"])]
    keys = ["Month", "Asset_ID"]
    pd.testing.assert_series_equal(ordinary.set_index(keys).Volume,
        observed.set_index(keys).Volume.reindex(ordinary.set_index(keys).index))
    ctra = repaired.loc[repaired.Asset_ID.eq("CTRA")].set_index("Month").Close
    assert ctra.loc["2024-02-29"] == pytest.approx(23.93441029, abs=1e-8)
    assert ctra.loc["2024-03-31"] == pytest.approx(26.09075021, abs=1e-8)
    assert dividends.loc[dividends.Asset_ID.eq("CTRA"), "Previous_Close_Method"].tolist() == ["cache_implied"]
    assert not ((dividends.Asset_ID == "DFS") & (dividends.Ex_Date.dt.month == 5)).any()


def test_independent_nominal_comparisons_and_adjustment_precision(inputs):
    checks = pd.read_csv(DEFAULT_CONFIG.paths.raw_monthly_dir / "basis_checks.csv")
    quotes = pd.read_csv(DEFAULT_CONFIG.paths.raw_monthly_dir / "observations.csv")
    nominal = checks.loc[checks.Check.eq("external_nominal")]
    joined = nominal.merge(quotes, left_on=["Asset_ID", "Date"],
        right_on=["Asset_ID", "Observation_Date"], validate="one_to_one", suffixes=("_check", "_input"))
    assert len(joined) == 63
    np.testing.assert_allclose(joined.Nominal_Close_input, joined.Observed_Close, rtol=0, atol=.005)
    factors = checks.loc[checks.Check.eq("dividend_factor")]
    assert len(factors) == 15
    np.testing.assert_allclose(factors.Observed_Factor,
        1-factors.Dividend/factors.Previous_Close, rtol=0, atol=2e-7)
    ctra = checks.loc[checks.Check.eq("ctra_anchor")]
    assert len(ctra) == 8
    np.testing.assert_allclose(ctra.Observed_Close/ctra.Nominal_Close,
        ctra.Expected_Factor, rtol=0, atol=2e-7)
    current = inputs.market_daily.loc[inputs.market_daily.Asset_ID.eq("CTRA")].set_index("Date").Close
    np.testing.assert_allclose(current.reindex(pd.to_datetime(ctra.Date)), ctra.Observed_Close, rtol=0, atol=1e-8)
    assert checks.Review_Status.eq("approved").all()
    assert checks.Source_SHA256.str.fullmatch(r"[a-f0-9]{64}").all()


def test_reviewed_identity_conflict_is_rejected(inputs, monkeypatch):
    read = pd.read_csv
    def tampered(path, *args, **kwargs):
        frame = read(path, *args, **kwargs)
        if str(path).endswith("monthly/observations.csv"):
            frame.loc[0, "Provider_Symbol"] = "WRONG"
        return frame
    monkeypatch.setattr(pd, "read_csv", tampered)
    with pytest.raises(ValueError, match="identity conflict"):
        load_monthly_evidence(DEFAULT_CONFIG.paths, inputs.market_daily)


EVENTS = [
    ("DFS", "2025-04-30", "2025-05-31", 1.0192*189.15/(1-.6/188.55999755859375)/182.67-1),
    ("ANSS", "2025-06-30", "2025-07-31", (199.91+.3399*633.47)/351.22-1),
    ("HES", "2025-06-30", "2025-07-31", 1.025*151.64/138.54-1),
    ("JNPR", "2025-06-30", "2025-07-31", 40/39.93-1),
    ("WBA", "2025-07-31", "2025-08-31", (11.45+.53)/11.64-1),
    ("IPG", "2025-10-31", "2025-11-30", .344*71.62/25.66-1),
    ("K", "2025-11-30", "2025-12-31", 83.5/(83.64*(1-.58/83.64))-1),
    ("DAY", "2026-01-31", "2026-02-28", 70/69.27-1),
    ("HOLX", "2026-03-31", "2026-04-30", 76/75.58999633789062-1),
]


@pytest.mark.parametrize("asset,start,end,expected", EVENTS)
def test_nine_event_returns_and_no_repeated_settlement(inputs, asset, start, end, expected):
    actual = interval_holding_returns(inputs, [asset], start, end, as_of=pd.Timestamp("2026-04-30"))
    assert actual[asset] == pytest.approx(expected, abs=1e-12)
    actions = inputs.corporate_actions
    event_ids = set(actions.legs.loc[actions.legs.From_Asset_ID.eq(asset), "Event_ID"])
    event = actions.events.loc[actions.events.Event_ID.isin(event_ids)].iloc[0]
    effective = event.Effective_Date
    first = apply_corporate_actions({asset: Fraction(1)}, {}, actions.events, actions.legs,
        actions.sources, start_exclusive=pd.Timestamp(start), end_inclusive=effective)
    second = apply_corporate_actions(first.positions, {}, actions.events, actions.legs,
        actions.sources, start_exclusive=effective, end_inclusive=pd.Timestamp(end))
    assert second.cash_flows.empty and second.nontradable_rights.empty
    if asset == "WBA":
        assert sum(map(float, first.cash_flows.Amount)) == pytest.approx(11.45)
        assert sum(map(float, first.nontradable_rights.Base_Value)) == pytest.approx(.53)


def test_stale_positive_holx_quote_cannot_override_settlement(inputs):
    monthly = inputs.market_monthly.copy()
    row = monthly.Asset_ID.eq("HOLX") & monthly.Month.eq(pd.Timestamp("2026-04-30"))
    assert monthly.loc[row, "Observation_Date"].tolist() == [pd.Timestamp("2026-04-07")]
    monthly.loc[row, "Close"] = 999
    result = interval_holding_returns(replace(inputs, market_monthly=monthly), ["HOLX"],
        "2026-03-31", "2026-04-30", as_of=pd.Timestamp("2026-04-30"))
    assert result.HOLX == pytest.approx(EVENTS[-1][-1])


def test_start_member_coverage_and_actual_signal_bounds(inputs):
    strategy = SectorMomentumStrategy(momentum_test_parameters(
        signal=SignalParameters("sector_momentum", 12, 1, None, None),
        sizing=SizingParameters("inverse_volatility", VolatilityProfile(60, 24)),
        selection=WholeSectorSelection(3, 3)))
    req = build_live_research_requirements(strategy, inputs.schedule)
    assert req.earliest_price_label == pd.Timestamp("2024-02-29")
    assert req.earliest_sector_label == pd.Timestamp("2025-01-31")
    history = prepare_live_research_history(inputs, strategy)
    assert len(history.sector_returns) == 15
    calendar = inputs.market_monthly.groupby("Month").Observation_Date.max().to_dict()
    total = 0
    for start, end in zip(pd.date_range("2025-01-31", "2026-03-31", freq="ME"),
                          pd.date_range("2025-02-28", "2026-04-30", freq="ME")):
        members = membership_as_of(inputs.membership, calendar[start])
        valued = interval_holding_returns(inputs, members, start, end,
            as_of=max(pd.Timestamp(inputs.schedule.Signal_Cutoff.min()), calendar[end]))
        assert len(valued) == len(members) and np.isfinite(valued).all()
        total += len(valued)
    assert total == 7545
    first = pd.Timestamp(inputs.schedule.Signal_Cutoff.min())
    close, _, _ = monthly_matrices(inputs.market_monthly, cutoff=first)
    members = membership_as_of(inputs.membership, first)
    vol = stock_volatility(close, VolatilityProfile(60, 24)).iloc[-1].reindex(members)
    assert set(vol.index[vol.isna()]) == {"GEV", "Q", "SNDK", "SOLV"}
    counts = close.pct_change(fill_method=None).tail(60).count()
    assert counts.loc[["CTRA", "HOLX"]].tolist() == [24, 24]


def test_future_evidence_calendar_and_start_member_gaps_fail(inputs):
    monthly = inputs.market_monthly.copy()
    row = monthly.Asset_ID.eq("CTRA") & monthly.Month.eq(pd.Timestamp("2024-02-29"))
    monthly.loc[row, "Available_Date"] = pd.Timestamp("2027-01-01")
    with pytest.raises(ValueError, match="future evidence"):
        monthly_matrices(monthly, cutoff=pd.Timestamp("2026-02-27"))
    with pytest.raises(ValueError, match="Nonconsecutive"):
        interval_holding_returns(inputs, ["CTRA"], "2025-01-31", "2025-03-31", as_of=pd.Timestamp("2026-02-27"))
    with pytest.raises(ValueError, match="starting constituent ABSENT, 2025-01-31"):
        interval_holding_returns(inputs, ["ABSENT"], "2025-01-31", "2025-02-28", as_of=pd.Timestamp("2026-02-27"))
    policy = inputs.corporate_actions.policy.copy()
    policy.loc[policy.CVR_Base_Value_Per_Unit.gt(0), "Valuation_Available_Date"] = "2027-01-01"
    future = replace(inputs, corporate_actions=replace(inputs.corporate_actions, policy=policy))
    with pytest.raises(ValueError, match="future CVR"):
        interval_holding_returns(future, ["WBA"], "2025-07-31", "2025-08-31", as_of=pd.Timestamp("2026-02-27"))


def test_calendar_gaps_stale_ordinary_quotes_and_future_prefixes(inputs):
    strategy = SectorMomentumStrategy(momentum_test_parameters(
        signal=SignalParameters("sector_momentum", 12, 1, None, None),
        sizing=SizingParameters("inverse_volatility", VolatilityProfile(60, 24)),
        selection=WholeSectorSelection(3, 3)))
    gap = inputs.market_monthly.loc[inputs.market_monthly.Month.ne(pd.Timestamp("2025-06-30"))]
    with pytest.raises(ValueError, match="requires prepared monthly history at 2025-06-30"):
        prepare_live_research_history(replace(inputs, market_monthly=gap), strategy)
    stale = inputs.market_monthly.copy()
    row = stale.Asset_ID.eq("AAPL") & stale.Month.eq(pd.Timestamp("2025-01-31"))
    stale.loc[row, "Observation_Date"] = pd.Timestamp("2025-01-30")
    with pytest.raises(ValueError, match="stale observation"):
        interval_holding_returns(replace(inputs, market_monthly=stale), ["AAPL"],
            "2025-01-31", "2025-02-28", as_of=pd.Timestamp("2026-02-27"))
    future = inputs.market_monthly.copy()
    cutoff = pd.Timestamp("2026-02-27")
    future.loc[future.Month.gt(cutoff+pd.offsets.MonthEnd(0)), "Close"] *= 7
    a = monthly_matrices(inputs.market_monthly, cutoff=cutoff)
    b = monthly_matrices(future, cutoff=cutoff)
    for original, changed in zip(a[:2], b[:2]):
        pd.testing.assert_frame_equal(original, changed)
    assert a[2] == b[2]


def test_invalid_adjustments_duplicates_and_unreviewed_overlap_fail(inputs):
    monthly = inputs.market_monthly
    with pytest.raises(ValueError, match="identity"):
        validate_monthly_history(pd.concat([monthly, monthly.iloc[:1]]))
    dividends = inputs.monthly_dividends.copy()
    dividends.loc[0, "Factor"] = 1.1
    with pytest.raises(ValueError, match="adjustment"):
        validate_dividends(dividends)
    dividends.loc[0, "Factor"] = -.1
    with pytest.raises(ValueError, match="invalid Factor"):
        validate_dividends(dividends)
    quotes, dividends = load_monthly_evidence(DEFAULT_CONFIG.paths, inputs.market_daily)
    bad = quotes.loc[quotes.Role.eq("event_price")].iloc[:1].copy()
    bad["Role"] = "fallback"
    with pytest.raises(ValueError, match="overlaps Yahoo"):
        compose_monthly_history(inputs.market_daily, bad, dividends, through=inputs.schedule.Signal_Cutoff.max())


def test_real_daily_execution_inputs_remain_separate(inputs):
    for row in inputs.schedule.itertuples():
        members = membership_as_of(inputs.membership, row.Execution_Date)
        actual = inputs.market_daily.loc[inputs.market_daily.Date.eq(row.Execution_Date)].set_index("Asset_ID")
        assert actual.reindex(members).Open.gt(0).all()
        recent = inputs.market_daily.loc[inputs.market_daily.Date.le(row.Signal_Cutoff)
            & inputs.market_daily.Date.gt(row.Signal_Cutoff-pd.DateOffset(months=3))]
        assert set(members) <= set(recent.loc[recent.Volume.gt(0), "Asset_ID"])
