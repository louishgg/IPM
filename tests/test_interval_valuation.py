"""Tests for shared one-share corporate-action interval valuation."""

from fractions import Fraction

import pandas as pd
import pytest

from portfolio_core.interval_valuation import (
    has_any_interval_event,
    has_interval_event,
    value_one_share_through_events,
)


@pytest.mark.parametrize(("date", "expected"), [
    ("2026-02-09", False), ("2026-02-10", False),
    ("2026-02-11", True), ("2026-02-20", True), ("2026-02-21", False),
])
def test_bulk_events_observe_interval_boundaries_and_only_supplied_assets(date, expected):
    events = pd.DataFrame({"Event_ID": ["E"], "Effective_Date": [date]})
    legs = pd.DataFrame({"Event_ID": ["E", "UNRELATED"], "From_Asset_ID": ["HELD", "OTHER"]})
    originals = events.copy(deep=True), legs.copy(deep=True)
    start, end = pd.Timestamp("2026-02-10"), pd.Timestamp("2026-02-20")
    assert has_any_interval_event(events, legs, iter(("FIRST", "HELD")), start, end) is expected
    assert not has_any_interval_event(events, legs, ("OTHER",), start, end)
    pd.testing.assert_frame_equal(events, originals[0], check_exact=True)
    pd.testing.assert_frame_equal(legs, originals[1], check_exact=True)


@pytest.mark.parametrize("empty", ["events", "legs", "assets"])
def test_bulk_events_accept_empty_inputs(empty):
    events = pd.DataFrame({"Event_ID": ["E"], "Effective_Date": ["2026-02-20"]})
    legs = pd.DataFrame({"Event_ID": ["E"], "From_Asset_ID": ["HELD"]})
    assert not has_any_interval_event(
        pd.DataFrame() if empty == "events" else events,
        pd.DataFrame() if empty == "legs" else legs,
        () if empty == "assets" else ("HELD",),
        pd.Timestamp("2026-02-10"), pd.Timestamp("2026-02-20"),
    )


@pytest.mark.parametrize("event_id", [1, "1"])
@pytest.mark.parametrize("leg_id", [1, "1"])
@pytest.mark.parametrize("from_asset", [7, "7"])
@pytest.mark.parametrize("requested_asset", [7, "7"])
def test_scalar_and_bulk_events_normalize_identifiers(event_id, leg_id, from_asset, requested_asset):
    events = pd.DataFrame({"Event_ID": [event_id], "Effective_Date": ["2026-02-20"]})
    legs = pd.DataFrame({"Event_ID": [leg_id], "From_Asset_ID": [from_asset]})
    start, end = pd.Timestamp("2026-02-10"), pd.Timestamp("2026-02-20")
    assert has_interval_event(events, legs, requested_asset, start, end)
    assert has_any_interval_event(events, legs, (requested_asset,), start, end)


def _event_tables(
    *,
    event_type: str,
    leg_type: str,
    to_asset_id: str = "",
    quantity: object = 0,
    cash: object = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_id = "EVT-20260220-TEST"
    events = pd.DataFrame(
        [
            {
                "Event_ID": event_id,
                "Effective_Date": "2026-02-20",
                "Event_Type": event_type,
                "Continuity_Class": "predecessor_extinguished",
                "Review_Status": "approved",
            }
        ]
    )
    legs = pd.DataFrame(
        [
            {
                "Event_ID": event_id,
                "Leg_Order": 1,
                "From_Asset_ID": "OLD",
                "To_Asset_ID": to_asset_id,
                "Leg_Type": leg_type,
                "Quantity_Per_From_Share": quantity,
                "Cash_Per_From_Share": cash,
                "Currency": "USD",
                "CVR_Units_Per_From_Share": 0,
                "CVR_Base_Value_Per_Unit": 0,
                "CVR_Max_Value_Per_Unit": 0,
                "Consumes_From_Position": True,
                "Review_Status": "approved",
            }
        ]
    )
    sources = pd.DataFrame(
        [
            {
                "Event_ID": event_id,
                "Source_URL": f"https://example.test/source-{number}",
                "Review_Status": "approved",
            }
            for number in (1, 2)
        ]
    )
    return events, legs, sources


def _chained_event_tables(
    *, same_date: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first_event = "EVT-Z-A-TO-B"
    second_event = "EVT-A-B-TO-C"
    events = pd.DataFrame(
        [
            {
                "Event_ID": first_event,
                "Effective_Date": "2026-02-20",
                "Event_Type": "stock_exchange",
                "Continuity_Class": "predecessor_extinguished",
                "Review_Status": "approved",
            },
            {
                "Event_ID": second_event,
                "Effective_Date": "2026-02-20" if same_date else "2026-02-21",
                "Event_Type": "stock_exchange",
                "Continuity_Class": "predecessor_extinguished",
                "Review_Status": "approved",
            },
        ]
    )
    legs = pd.DataFrame(
        [
            {
                "Event_ID": first_event,
                "Leg_Order": 1,
                "From_Asset_ID": "A",
                "To_Asset_ID": "B",
                "Leg_Type": "stock",
                "Quantity_Per_From_Share": "3/2",
                "Cash_Per_From_Share": 0,
                "Currency": "USD",
                "CVR_Units_Per_From_Share": 0,
                "CVR_Base_Value_Per_Unit": 0,
                "CVR_Max_Value_Per_Unit": 0,
                "Consumes_From_Position": True,
                "Review_Status": "approved",
            },
            {
                "Event_ID": second_event,
                "Leg_Order": 1,
                "From_Asset_ID": "B",
                "To_Asset_ID": "C",
                "Leg_Type": "stock",
                "Quantity_Per_From_Share": "2/3",
                "Cash_Per_From_Share": 0,
                "Currency": "USD",
                "CVR_Units_Per_From_Share": 0,
                "CVR_Base_Value_Per_Unit": 0,
                "CVR_Max_Value_Per_Unit": 0,
                "Consumes_From_Position": True,
                "Review_Status": "approved",
            },
        ]
    )
    sources = pd.DataFrame(
        [
            {
                "Event_ID": event_id,
                "Source_URL": f"https://example.test/{event_id}/{number}",
                "Review_Status": "approved",
            }
            for event_id in (first_event, second_event)
            for number in (1, 2)
        ]
    )
    return events, legs, sources


def test_no_event_leaves_ordinary_price_valuation_to_the_caller():
    events = pd.DataFrame()
    legs = pd.DataFrame()
    sources = pd.DataFrame()
    start = pd.Timestamp("2026-02-13")
    end = pd.Timestamp("2026-03-02")

    assert not has_interval_event(events, legs, "OLD", start, end)
    assert (
        value_one_share_through_events(
            events,
            legs,
            sources,
            "OLD",
            start,
            end,
            lambda _: 100.0,
        )
        is None
    )


def test_cash_settlement_does_not_require_an_extinguished_price():
    events, legs, sources = _event_tables(
        event_type="cash_settlement",
        leg_type="cash",
        cash=76,
    )

    result = value_one_share_through_events(
        events,
        legs,
        sources,
        "OLD",
        pd.Timestamp("2026-02-13"),
        pd.Timestamp("2026-03-02"),
        lambda asset_id: pytest.fail(
            f"cash-only settlement requested an end price for {asset_id}"
        ),
    )

    assert result is not None
    _, end_value = result
    assert end_value == Fraction(76)


def test_cash_and_cvr_base_value_are_both_included_exactly():
    events, legs, sources = _event_tables(
        event_type="cash_settlement",
        leg_type="cash",
        cash=76,
    )
    cvr_leg = legs.iloc[0].copy()
    cvr_leg["Leg_Order"] = 2
    cvr_leg["To_Asset_ID"] = "OLD.CVR"
    cvr_leg["Leg_Type"] = "cvr"
    cvr_leg["Cash_Per_From_Share"] = 0
    cvr_leg["CVR_Units_Per_From_Share"] = "3/2"
    cvr_leg["CVR_Base_Value_Per_Unit"] = "5/3"
    cvr_leg["CVR_Max_Value_Per_Unit"] = 3
    legs = pd.concat([legs, cvr_leg.to_frame().T], ignore_index=True)

    result = value_one_share_through_events(
        events,
        legs,
        sources,
        "OLD",
        pd.Timestamp("2026-02-13"),
        pd.Timestamp("2026-03-02"),
        lambda asset_id: pytest.fail(
            f"cash-plus-CVR settlement requested an end price for {asset_id}"
        ),
    )

    assert result is not None
    _, end_value = result
    assert end_value == Fraction(157, 2)


def test_successor_units_use_the_successor_end_price():
    events, legs, sources = _event_tables(
        event_type="stock_exchange",
        leg_type="stock",
        to_asset_id="NEW",
        quantity="0.5",
    )

    result = value_one_share_through_events(
        events,
        legs,
        sources,
        "OLD",
        pd.Timestamp("2026-02-13"),
        pd.Timestamp("2026-03-02"),
        lambda asset_id: 50 if asset_id == "NEW" else None,
    )

    assert result is not None
    action_result, end_value = result
    assert action_result.positions.to_dict() == {"NEW": Fraction(1, 2)}
    assert end_value == Fraction(25)


@pytest.mark.parametrize(
    "invalid_price",
    [None, 0, -1, float("nan"), "not-a-price"],
)
def test_missing_or_invalid_successor_price_makes_event_valuation_unavailable(
    invalid_price,
):
    events, legs, sources = _event_tables(
        event_type="stock_exchange",
        leg_type="stock",
        to_asset_id="NEW",
        quantity="0.5",
    )

    assert (
        value_one_share_through_events(
            events,
            legs,
            sources,
            "OLD",
            pd.Timestamp("2026-02-13"),
            pd.Timestamp("2026-03-02"),
            lambda _: invalid_price,
        )
        is None
    )


@pytest.mark.parametrize("same_date", [False, True])
def test_chained_events_value_only_the_final_successor(same_date):
    events, legs, sources = _chained_event_tables(same_date=same_date)
    requested_prices: list[str] = []

    def end_price(asset_id: str):
        requested_prices.append(asset_id)
        return 42 if asset_id == "C" else pytest.fail(
            f"chained valuation requested an intermediate price for {asset_id}"
        )

    result = value_one_share_through_events(
        events,
        legs,
        sources,
        "A",
        pd.Timestamp("2026-02-13"),
        pd.Timestamp("2026-03-02"),
        end_price,
    )

    assert result is not None
    action_result, end_value = result
    assert action_result.positions.to_dict() == {"C": Fraction(1)}
    assert end_value == Fraction(42)
    assert requested_prices == ["C"]
