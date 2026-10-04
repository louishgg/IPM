"""Reconciliation tests for the sole dated portfolio lifecycle transition."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from portfolio_core.accounting_config import DEFAULT_ACCOUNTING_CONFIG
from portfolio_core.accounting_ledger import (
    LedgerRebalancePlan,
    LedgerState,
    execute_rebalance,
)
from portfolio_core.corporate_actions import EVENT_COLUMNS, LEG_COLUMNS, SOURCE_COLUMNS
from portfolio_core.portfolio_lifecycle import (
    advance_ledger,
    period_has_position_event,
)


def _accounting():
    return replace(
        DEFAULT_ACCOUNTING_CONFIG,
        initial_capital=100.0,
        minimum_positions=0,
        minimum_long_positions=0,
        minimum_short_positions=0,
    )


def _dated_parent_short():
    accounting = _accounting()
    plan = LedgerRebalancePlan.from_series(
        pd.Series({"PARENT": -2.0}),
        pd.Series({"PARENT": -0.8}),
        pd.Series({"PARENT": 40.0}),
        pd.Series({"PARENT": 10_000_000.0}),
        {"PARENT"},
        "2026-01-01",
    )
    result = execute_rebalance(
        LedgerState.initial(accounting),
        plan,
        accounting,
        apply_fees=False,
        apply_spread=False,
    )
    assert result.state.cash == 180.0
    assert result.state.restricted_short_proceeds["PARENT"] == 80.0
    return result.state, accounting


def _events(currency: str = "USD"):
    distribution = "EVT-20260111-PARENT-CHILD-DISTRIBUTION"
    settlement = "EVT-20260121-PARENT-CASH-SETTLEMENT"
    events = pd.DataFrame(
        [
            {
                "Event_ID": distribution,
                "Effective_Date": "2026-01-11",
                "Event_Type": "distribution",
                "Continuity_Class": "predecessor_survives",
                "Review_Status": "approved",
            },
            {
                "Event_ID": settlement,
                "Effective_Date": "2026-01-21",
                "Event_Type": "cash_settlement",
                "Continuity_Class": "predecessor_extinguished",
                "Review_Status": "approved",
            },
        ],
        columns=EVENT_COLUMNS,
    )
    legs = pd.DataFrame(
        [
            {
                "Event_ID": distribution,
                "Leg_Order": 1,
                "From_Asset_ID": "PARENT",
                "To_Asset_ID": "CHILD",
                "Leg_Type": "distribution",
                "Quantity_Per_From_Share": "0.5",
                "Cash_Per_From_Share": "",
                "Currency": "",
                "CVR_Units_Per_From_Share": "",
                "CVR_Base_Value_Per_Unit": "",
                "CVR_Max_Value_Per_Unit": "",
                "Consumes_From_Position": False,
                "Review_Status": "approved",
            },
            {
                "Event_ID": settlement,
                "Leg_Order": 1,
                "From_Asset_ID": "PARENT",
                "To_Asset_ID": "",
                "Leg_Type": "cash",
                "Quantity_Per_From_Share": "",
                "Cash_Per_From_Share": "200",
                "Currency": currency,
                "CVR_Units_Per_From_Share": "",
                "CVR_Base_Value_Per_Unit": "",
                "CVR_Max_Value_Per_Unit": "",
                "Consumes_From_Position": True,
                "Review_Status": "approved",
            },
        ],
        columns=LEG_COLUMNS,
    )
    sources = pd.DataFrame(
        [
            {
                "Event_ID": event_id,
                "Source_URL": f"https://example.test/{event_id}",
                "Review_Status": "approved",
            }
            for event_id in (distribution, settlement)
        ],
        columns=SOURCE_COLUMNS,
    )
    return events, legs, sources


def test_segmented_and_direct_advancement_reconcile_across_event_dates():
    state, accounting = _dated_parent_short()
    events, legs, sources = _events()

    direct = advance_ledger(
        state,
        "2026-01-31",
        events,
        legs,
        sources,
        accounting,
    )
    first = advance_ledger(
        state,
        "2026-01-15",
        events,
        legs,
        sources,
        accounting,
    )
    segmented = advance_ledger(
        first.state,
        "2026-01-31",
        events,
        legs,
        sources,
        accounting,
    )

    pd.testing.assert_series_equal(segmented.state.shares, direct.state.shares)
    pd.testing.assert_series_equal(
        segmented.state.restricted_short_proceeds,
        direct.state.restricted_short_proceeds,
    )
    assert segmented.state.cash == pytest.approx(direct.state.cash, abs=1e-12)
    assert first.interest_amount + segmented.interest_amount == pytest.approx(
        direct.interest_amount,
        abs=1e-12,
    )
    assert (
        first.cash_interest_credit + segmented.cash_interest_credit
        == pytest.approx(direct.cash_interest_credit, abs=1e-12)
    )
    assert (
        first.loan_interest_charge + segmented.loan_interest_charge
        == pytest.approx(direct.loan_interest_charge, abs=1e-12)
    )
    direct_keys = list(
        direct.corporate_actions.audit[["Event_ID", "From_Asset_ID"]]
        .itertuples(index=False, name=None)
    )
    segmented_keys = [
        *first.corporate_actions.audit[["Event_ID", "From_Asset_ID"]]
        .itertuples(index=False, name=None),
        *segmented.corporate_actions.audit[["Event_ID", "From_Asset_ID"]]
        .itertuples(index=False, name=None),
    ]
    assert segmented_keys == direct_keys


@pytest.mark.parametrize("quantity", [1.0, -1.0, 1e-12])
def test_position_event_interval_is_start_exclusive_and_end_inclusive(quantity):
    events, legs, sources = _events()
    portfolio_data = SimpleNamespace(
        security_events=events,
        security_event_legs=legs,
        security_event_sources=sources,
    )
    held = pd.Series({"PARENT": quantity})

    assert not period_has_position_event(
        portfolio_data,
        held,
        pd.Timestamp("2026-01-11"),
        pd.Timestamp("2026-01-15"),
    )
    assert period_has_position_event(
        portfolio_data,
        held,
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-11"),
    )
    assert not period_has_position_event(
        portfolio_data,
        pd.Series({"PARENT": 0.0, "OTHER": 1.0}),
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-31"),
    )
    assert not period_has_position_event(
        portfolio_data,
        pd.Series(dtype=float),
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-31"),
    )
    assert not period_has_position_event(
        SimpleNamespace(
            security_events=pd.DataFrame(),
            security_event_legs=pd.DataFrame(),
        ),
        held,
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-31"),
    )


def test_event_financing_restrictions_and_distribution_short_reconcile():
    state, accounting = _dated_parent_short()
    events, legs, sources = _events()

    distribution_checkpoint = advance_ledger(
        state,
        "2026-01-15",
        events,
        legs,
        sources,
        accounting,
    )
    assert distribution_checkpoint.state.shares.to_dict() == {
        "CHILD": -1.0,
        "PARENT": -2.0,
    }
    assert distribution_checkpoint.state.restricted_short_proceeds.to_dict() == {
        "PARENT": 80.0
    }
    assert "CHILD" not in distribution_checkpoint.state.restricted_short_proceeds

    result = advance_ledger(
        state,
        "2026-01-31",
        events,
        legs,
        sources,
        accounting,
    )
    assert result.state.shares.to_dict() == {"CHILD": -1.0}
    assert result.state.restricted_short_proceeds.empty
    assert result.state.cash < 0.0
    assert result.cash_interest_credit > 0.0
    assert result.loan_interest_charge > 0.0
    assert result.interest_amount == pytest.approx(
        result.cash_interest_credit - result.loan_interest_charge
    )
    released = result.corporate_actions.restricted_proceeds_effects.loc[
        lambda frame: frame["Effect_Type"].eq("release"), "Amount"
    ]
    assert [float(value) for value in released] == [80.0]
    assert sum(result.event_interest_effects.values()) == pytest.approx(
        result.interest_amount
        - result.counterfactual_interest_without_actions
    )
    assert np.isfinite(result.state.cash)


def test_lifecycle_is_monotonic_and_rejects_non_usd_cash_flows():
    state, accounting = _dated_parent_short()
    events, legs, sources = _events()
    with pytest.raises(ValueError, match="cannot precede"):
        advance_ledger(
            state,
            "2025-12-31",
            events,
            legs,
            sources,
            accounting,
        )

    _, non_usd_legs, _ = _events(currency="EUR")
    with pytest.raises((ValueError, RuntimeError), match="USD|currency"):
        advance_ledger(
            state,
            "2026-01-31",
            events,
            non_usd_legs,
            sources,
            accounting,
        )
