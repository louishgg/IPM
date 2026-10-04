"""Parity tests for the shared backtest/live lifecycle-eligibility rule."""

from types import SimpleNamespace

import pandas as pd

from live.strategy_universe import live_interval_eligible_assets
from portfolio_core.corporate_actions import EVENT_COLUMNS, LEG_COLUMNS, SOURCE_COLUMNS
from portfolio_core.portfolio_lifecycle import (
    eligible_assets_for_interval,
    lifecycle_eligible_assets,
)


START = pd.Timestamp("2026-01-31")
END = pd.Timestamp("2026-02-28")


def _leg(
    event_id: str,
    from_asset: str,
    leg_type: str,
    *,
    to_asset: str = "",
    quantity: str = "",
    cash: str = "",
    currency: str = "",
) -> dict[str, object]:
    return {
        "Event_ID": event_id,
        "Leg_Order": 1,
        "From_Asset_ID": from_asset,
        "To_Asset_ID": to_asset,
        "Leg_Type": leg_type,
        "Quantity_Per_From_Share": quantity,
        "Cash_Per_From_Share": cash,
        "Currency": currency,
        "CVR_Units_Per_From_Share": "",
        "CVR_Base_Value_Per_Unit": "",
        "CVR_Max_Value_Per_Unit": "",
        "Consumes_From_Position": True,
        "Review_Status": "approved",
    }


def _event_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    definitions = (
        (
            "EVT-CASHED",
            "2026-02-15",
            "cash_settlement",
            "predecessor_extinguished",
            _leg(
                "EVT-CASHED",
                "CASHED",
                "cash",
                cash="17",
                currency="USD",
            ),
        ),
        (
            "EVT-STOCKED",
            "2026-02-16",
            "stock_exchange",
            "predecessor_extinguished",
            _leg(
                "EVT-STOCKED",
                "STOCKED",
                "stock",
                to_asset="STOCKED_NEW",
                quantity="1/2",
            ),
        ),
        (
            "EVT-UNPRICED",
            "2026-02-17",
            "stock_exchange",
            "predecessor_extinguished",
            _leg(
                "EVT-UNPRICED",
                "UNPRICED",
                "stock",
                to_asset="UNPRICED_NEW",
                quantity="1",
            ),
        ),
        (
            "EVT-EXTINCT",
            START,
            "stock_exchange",
            "predecessor_extinguished",
            _leg(
                "EVT-EXTINCT",
                "EXTINCT",
                "stock",
                to_asset="EXTINCT_NEW",
                quantity="1",
            ),
        ),
        (
            "EVT-SAME-KEY",
            START,
            "stock_exchange",
            "predecessor_extinguished",
            _leg(
                "EVT-SAME-KEY",
                "SAME_KEY",
                "stock",
                to_asset="SAME_KEY",
                quantity="1",
            ),
        ),
    )
    events = pd.DataFrame(
        [
            {
                "Event_ID": event_id,
                "Effective_Date": effective_date,
                "Event_Type": event_type,
                "Continuity_Class": continuity,
                "Review_Status": "approved",
            }
            for event_id, effective_date, event_type, continuity, _ in definitions
        ],
        columns=EVENT_COLUMNS,
    )
    legs = pd.DataFrame(
        [leg for _, _, _, _, leg in definitions],
        columns=LEG_COLUMNS,
    )
    sources = pd.DataFrame(
        [
            {
                "Event_ID": event_id,
                "Source_URL": f"https://example.test/{event_id}",
                "Review_Status": "approved",
            }
            for event_id, *_ in definitions
        ],
        columns=SOURCE_COLUMNS,
    )
    return events, legs, sources


def _price_inputs() -> tuple[dict[str, float], dict[str, float]]:
    start_prices = {
        "PLAIN": 10.0,
        "NO_START": float("nan"),
        "NO_END": 12.0,
        "CASHED": 20.0,
        "STOCKED": 30.0,
        "UNPRICED": 40.0,
        "EXTINCT": 50.0,
        "SAME_KEY": 60.0,
        "ZERO": 0.0,
    }
    end_prices = {
        "PLAIN": 11.0,
        "NO_START": 13.0,
        "CASHED": float("nan"),
        "STOCKED": float("nan"),
        "STOCKED_NEW": 70.0,
        "UNPRICED": float("nan"),
        "UNPRICED_NEW": float("inf"),
        "EXTINCT": 51.0,
        "EXTINCT_NEW": 52.0,
        "SAME_KEY": 61.0,
        "ZERO": 1.0,
    }
    return start_prices, end_prices


def test_backtest_and_live_adapters_preserve_exact_eligibility_semantics():
    events, legs, sources = _event_tables()
    start_prices, end_prices = _price_inputs()
    candidates = tuple(start_prices)
    expected = frozenset({"PLAIN", "CASHED", "STOCKED", "SAME_KEY"})

    canonical = lifecycle_eligible_assets(
        candidates,
        start=START,
        end=END,
        start_prices=start_prices,
        end_prices=end_prices,
        events=events,
        legs=legs,
        sources=sources,
    )

    wide_prices = pd.DataFrame(
        [start_prices, end_prices],
        index=[START, END],
    )
    prepared = SimpleNamespace(
        data_close=wide_prices,
        security_events=events,
        security_event_legs=legs,
        security_event_sources=sources,
    )
    backtest = eligible_assets_for_interval(prepared, candidates, START, END)

    market = pd.concat(
        [
            pd.DataFrame(
                {
                    "Date": START,
                    "Asset_ID": wide_prices.columns,
                    "Open": wide_prices.loc[START].to_numpy(),
                    "Close": float("nan"),
                }
            ),
            pd.DataFrame(
                {
                    "Date": END,
                    "Asset_ID": wide_prices.columns,
                    "Open": float("nan"),
                    "Close": wide_prices.loc[END].to_numpy(),
                }
            ),
        ],
        ignore_index=True,
    )
    live = live_interval_eligible_assets(
        market,
        candidates,
        execution_date=START,
        execution_field="Open",
        valuation_end=END,
        valuation_field="Close",
        corporate_action_events=events,
        corporate_action_legs=legs,
        corporate_action_sources=sources,
    )

    assert canonical == expected
    assert backtest == expected
    assert live == set(expected)
