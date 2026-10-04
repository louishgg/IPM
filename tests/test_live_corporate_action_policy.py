"""Offline live corporate-action policy and settlement tests."""

from __future__ import annotations

from fractions import Fraction
import json
from dataclasses import replace

import pandas as pd
import pytest

from live.analysis import _corporate_action_audit_rows
from live.config import DEFAULT_CONFIG
from live.corporate_action_policy import (
    LiveCorporateActionBundle,
    load_corporate_action_policy,
    load_live_corporate_action_bundle,
)
from portfolio_core.accounting_ledger import (
    LedgerRebalancePlan,
    LedgerState,
    execute_rebalance,
    project_financing_amounts,
)
from portfolio_core.interval_valuation import value_one_share_through_events
from portfolio_core.portfolio_lifecycle import advance_ledger


def _event_subset(
    bundle: LiveCorporateActionBundle, event_id: str
) -> LiveCorporateActionBundle:
    return LiveCorporateActionBundle(
        events=bundle.events.loc[bundle.events["Event_ID"].eq(event_id)].copy(),
        legs=bundle.legs.loc[bundle.legs["Event_ID"].eq(event_id)].copy(),
        sources=bundle.sources.loc[bundle.sources["Event_ID"].eq(event_id)].copy(),
        policy=bundle.policy.loc[bundle.policy["Event_ID"].eq(event_id)].copy(),
    )


def _ledger_state(
    positions: dict[str, float],
    prices: dict[str, float],
    *,
    initial_capital: float,
    state_date: pd.Timestamp,
):
    accounting = replace(
        DEFAULT_CONFIG.accounting,
        initial_capital=initial_capital,
        minimum_positions=0,
        minimum_long_positions=0,
        minimum_short_positions=0,
    )
    state = LedgerState.initial(accounting)
    if not positions:
        return LedgerState.initial(accounting, state_date=state_date), accounting
    price_series = pd.Series(prices, dtype=float)
    target_values = pd.Series(positions, dtype=float) * price_series
    plan = LedgerRebalancePlan.from_series(
        pd.Series(positions, dtype=float),
        target_values / initial_capital,
        price_series,
        pd.Series(10_000_000.0, index=price_series.index),
        price_series.index,
        state_date,
    )
    result = execute_rebalance(
        state,
        plan,
        accounting,
        apply_fees=False,
        apply_spread=False,
    )
    return result.state, accounting


def test_live_policy_references_canonical_events_without_duplicating_terms():
    policy = load_corporate_action_policy(DEFAULT_CONFIG.paths)
    expected = {
        "EVT-20260204-DAY-CASH-ACQUISITION",
        "EVT-20260407-HOLX-CASH-CVR-ACQUISITION",
        "EVT-20260507-CTRA-DVN-STOCK-EXCHANGE",
        "EVT-20260521-BK-BNY-TICKER-CHANGE",
        "EVT-20250717-ANSS-SNPS-CASH-AND-STOCK",
        "EVT-20250518-DFS-COF-STOCK-EXCHANGE",
        "EVT-20251126-IPG-OMC-STOCK-EXCHANGE",
        "EVT-20250702-JNPR-CASH-SETTLEMENT",
        "EVT-20251211-K-CASH-SETTLEMENT",
        "EVT-20250718-HES-CVX-STOCK-EXCHANGE",
        "EVT-20250828-WBA-CASH-DAP-ACQUISITION",
    }
    assert set(policy["Event_ID"]) == expected
    assert set(policy.columns) == {
        "Event_ID",
        "Scope",
        "Apply_Accounting",
        "CVR_Base_Value_Per_Unit",
        "Valuation_As_Of_Date",
        "Valuation_Available_Date",
        "Valuation_Source_ID",
        "Review_Status",
        "Notes",
    }
    assert policy["Apply_Accounting"].all()
    bundle = load_live_corporate_action_bundle(DEFAULT_CONFIG.paths)
    assert set(bundle.events["Event_ID"]) == expected
    assert set(bundle.legs["Event_ID"]) == expected
    assert set(bundle.sources["Event_ID"]) == expected


def test_future_window_applies_bk_relabel_and_exact_ctra_exchange():
    bundle = load_live_corporate_action_bundle(DEFAULT_CONFIG.paths)
    start = pd.Timestamp("2026-05-06")
    end = pd.Timestamp("2026-05-22")
    state, accounting = _ledger_state(
        {"BK": 5.0, "CTRA": 3.0},
        {"BK": 10.0, "CTRA": 10.0},
        initial_capital=1_080.0,
        state_date=start,
    )
    advance = advance_ledger(
        state,
        end,
        bundle.events,
        bundle.legs,
        bundle.sources,
        accounting,
    )
    audit = _corporate_action_audit_rows(
        advance.corporate_actions,
        advance.event_interest_effects,
        rebalance_id="",
        phase="synthetic_future_window",
    )

    expected_cash, _, _, _ = project_financing_amounts(
        1_000.0,
        0.0,
        start,
        end,
        accounting,
    )
    assert advance.state.cash == pytest.approx(expected_cash)
    assert advance.state.shares.to_dict() == pytest.approx(
        {"BNY": 5.0, "DVN": 2.1}
    )
    assert advance.state.restricted_short_proceeds.empty
    assert {row["Event_ID"] for row in audit} == {
        "EVT-20260507-CTRA-DVN-STOCK-EXCHANGE",
        "EVT-20260521-BK-BNY-TICKER-CHANGE",
    }
    assert all(row["Interest_Effect"] == pytest.approx(0.0) for row in audit)
    audit_by_event = {row["Event_ID"]: row for row in audit}
    assert json.loads(
        audit_by_event["EVT-20260507-CTRA-DVN-STOCK-EXCHANGE"][
            "Successor_Units_JSON"
        ]
    ) == {"DVN": "2.1"}
    assert json.loads(
        audit_by_event["EVT-20260521-BK-BNY-TICKER-CHANGE"][
            "Successor_Units_JSON"
        ]
    ) == {"BNY": "5"}
    assert all(json.loads(row["Source_URLs_JSON"]) for row in audit)


@pytest.mark.parametrize(
    ("position", "initial_cash", "restricted", "expected_cash_effect", "expected_cvr_max"),
    [
        (10, 1_000.0, 0.0, 760.0, 30.0),
        (-10, 1_700.0, 700.0, -760.0, -30.0),
        (0, 1_000.0, 0.0, 0.0, 0.0),
    ],
)
def test_holx_settlement_handles_long_short_and_zero_positions(
    position,
    initial_cash,
    restricted,
    expected_cash_effect,
    expected_cvr_max,
):
    actions = _event_subset(
        load_live_corporate_action_bundle(DEFAULT_CONFIG.paths),
        "EVT-20260407-HOLX-CASH-CVR-ACQUISITION",
    )
    start = pd.Timestamp("2026-04-01")
    event = pd.Timestamp("2026-04-07")
    end = pd.Timestamp("2026-04-10")
    prices = {"HOLX": 76.0}
    if position > 0:
        state, accounting = _ledger_state(
            {"HOLX": float(position)},
            prices,
            initial_capital=initial_cash + position * prices["HOLX"],
            state_date=start,
        )
    elif position < 0:
        state, accounting = _ledger_state(
            {"HOLX": float(position)},
            {"HOLX": restricted / abs(position)},
            initial_capital=initial_cash - restricted,
            state_date=start,
        )
    else:
        state, accounting = _ledger_state(
            {},
            {},
            initial_capital=initial_cash,
            state_date=start,
        )

    cash_at_event, first_interest, _, _ = project_financing_amounts(
        initial_cash,
        restricted,
        start,
        event,
        accounting,
    )
    expected_after_event = cash_at_event + expected_cash_effect
    expected_end, second_interest, _, _ = project_financing_amounts(
        expected_after_event,
        0.0,
        event,
        end,
        accounting,
    )
    _, counterfactual_interest, _, _ = project_financing_amounts(
        initial_cash,
        restricted,
        start,
        end,
        accounting,
    )
    advance = advance_ledger(
        state,
        end,
        actions.events,
        actions.legs,
        actions.sources,
        accounting,
    )
    audit = _corporate_action_audit_rows(
        advance.corporate_actions,
        advance.event_interest_effects,
        rebalance_id="R3",
        phase="holding_period",
    )
    one_share_value = value_one_share_through_events(
        actions.events,
        actions.legs,
        actions.sources,
        "HOLX",
        start,
        end,
        lambda asset_id: pytest.fail(
            f"cash-plus-CVR settlement requested a price for {asset_id}"
        ),
    )

    assert advance.state.cash == pytest.approx(expected_end)
    assert advance.interest_amount == pytest.approx(
        first_interest + second_interest
    )
    assert "HOLX" not in advance.state.shares
    assert "HOLX" not in advance.state.restricted_short_proceeds
    assert one_share_value is not None
    assert one_share_value[1] == Fraction(76)
    assert len(audit) == 1
    row = audit[0]
    assert row["Event_ID"] == "EVT-20260407-HOLX-CASH-CVR-ACQUISITION"
    assert row["Shares_Before"] == position
    assert row["Cash_Effect"] == pytest.approx(expected_cash_effect)
    assert row["Interest_Effect"] == pytest.approx(
        first_interest + second_interest - counterfactual_interest
    )
    if position > 0:
        assert row["Interest_Effect"] > 0.0
    elif position < 0:
        assert row["Interest_Effect"] < 0.0
    else:
        assert row["Interest_Effect"] == pytest.approx(0.0)
    assert row["Restricted_Short_Proceeds_Released"] == pytest.approx(restricted)
    assert row["CVR_Units"] == pytest.approx(position)
    assert row["CVR_Base_Value"] == 0.0
    assert row["CVR_Max_Value"] == pytest.approx(expected_cvr_max)
    assert row["Fixed_Fee"] == 0.0
    assert row["Spread_Cost"] == 0.0
    assert row["Applied_To_Position"] is bool(position)
