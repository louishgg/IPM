"""Synthetic tests for the shared mandatory corporate-action kernel."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import json

import pandas as pd
import pytest

from portfolio_core.corporate_actions import (
    EVENT_COLUMNS,
    LEG_COLUMNS,
    SOURCE_COLUMNS,
    apply_corporate_actions,
)


D = Decimal
ZERO = Decimal("0")


def _event(
    event_id: str,
    event_type: str,
    continuity: str,
    *,
    date: str = "2026-05-07",
    status: str = "approved",
) -> dict[str, object]:
    return {
        "Event_ID": event_id,
        "Effective_Date": date,
        "Event_Type": event_type,
        "Continuity_Class": continuity,
        "Review_Status": status,
    }


def _leg(
    event_id: str,
    order: int,
    from_asset: str,
    leg_type: str,
    *,
    to_asset: str = "",
    quantity: object = "",
    cash: object = "",
    currency: str = "",
    cvr_units: object = "",
    cvr_base: object = "",
    cvr_max: object = "",
    consumes: bool = True,
    status: str = "approved",
) -> dict[str, object]:
    return {
        "Event_ID": event_id,
        "Leg_Order": order,
        "From_Asset_ID": from_asset,
        "To_Asset_ID": to_asset,
        "Leg_Type": leg_type,
        "Quantity_Per_From_Share": quantity,
        "Cash_Per_From_Share": cash,
        "Currency": currency,
        "CVR_Units_Per_From_Share": cvr_units,
        "CVR_Base_Value_Per_Unit": cvr_base,
        "CVR_Max_Value_Per_Unit": cvr_max,
        "Consumes_From_Position": consumes,
        "Review_Status": status,
    }


def _source(event_id: str, *, status: str = "approved") -> dict[str, str]:
    return {
        "Event_ID": event_id,
        "Source_URL": f"https://example.test/{event_id}",
        "Review_Status": status,
    }


def _apply(
    events: list[dict[str, object]],
    legs: list[dict[str, object]],
    *,
    positions: dict[str, object] | None = None,
    restricted: dict[str, object] | None = None,
    sources: list[dict[str, str]] | None = None,
    start: str = "2026-01-01",
    end: str = "2026-12-31",
):
    event_frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
    leg_frame = pd.DataFrame(legs, columns=LEG_COLUMNS)
    source_frame = pd.DataFrame(
        sources if sources is not None else [_source(row["Event_ID"]) for row in events],
        columns=SOURCE_COLUMNS,
    )
    return apply_corporate_actions(
        positions or {},
        restricted or {},
        event_frame,
        leg_frame,
        source_frame,
        start_exclusive=start,
        end_inclusive=end,
    )


def test_same_security_relabel_preserves_fractional_units_and_nets_restriction():
    event_id = "EVT-20260521-BK-BNY-RELABEL"
    result = _apply(
        [_event(event_id, "identity_continuity", "same_security", date="2026-05-21")],
        [_leg(event_id, 1, "BK", "relabel", to_asset="BNY", quantity="1")],
        positions={"BK": "-10.25", "BNY": "2.25"},
        restricted={"BK": "1025"},
    )

    assert result.positions.to_dict() == {"BNY": D("-8.00")}
    assert result.restricted_short_proceeds.to_dict() == {"BNY": D("800")}
    effects = result.restricted_proceeds_effects
    assert effects["Effect_Type"].tolist() == ["transfer", "release"]
    assert effects["Amount"].tolist() == [D("800"), D("225")]
    audit = result.audit.iloc[0]
    assert audit["Shares_Before"] == D("-10.25")
    assert audit["Shares_After"] == ZERO
    assert json.loads(audit["Successor_Units_JSON"]) == {"BNY": "-10.25"}
    assert audit["Fixed_Fee"] == ZERO
    assert audit["Spread_Cost"] == ZERO


def test_otc_transition_is_an_economic_no_op_beyond_asset_relabel():
    event_id = "EVT-20230320-OLD-OTC"
    result = _apply(
        [_event(event_id, "otc_transition", "same_security", date="2023-03-20")],
        [_leg(event_id, 1, "OLD", "relabel", to_asset="OLDQ", quantity="1")],
        positions={"OLD": "4.125"},
        start="2023-03-01",
        end="2023-03-31",
    )

    assert result.positions.to_dict() == {"OLDQ": D("4.125")}
    assert result.cash_flows.empty
    assert result.audit.iloc[0]["Cash_Effect"] == ZERO


@pytest.mark.parametrize(
    ("shares", "restricted", "cash", "right_units", "right_max"),
    [
        ("10.5", {}, "798.0", "10.5", "31.5"),
        ("-10.5", {"HOLX": "700"}, "-798.0", "-10.5", "-31.5"),
        ("0", {}, "0", "0", "0"),
    ],
)
def test_cash_plus_cvr_handles_long_short_and_zero_without_rounding(
    shares,
    restricted,
    cash,
    right_units,
    right_max,
):
    event_id = "EVT-20260407-HOLX-CASH-CVR"
    result = _apply(
        [_event(event_id, "cash_settlement", "predecessor_extinguished", date="2026-04-07")],
        [
            _leg(event_id, 1, "HOLX", "cash", cash="76", currency="USD"),
            _leg(
                event_id,
                2,
                "HOLX",
                "cvr",
                to_asset="HOLX-CVR",
                currency="USD",
                cvr_units="1",
                cvr_base="0",
                cvr_max="3",
            ),
        ],
        positions={"HOLX": shares},
        restricted=restricted,
    )

    assert "HOLX" not in result.positions
    assert result.restricted_short_proceeds.empty
    audit = result.audit.iloc[0]
    assert audit["Cash_Effect"] == D(cash)
    assert audit["CVR_Units"] == D(right_units)
    assert audit["CVR_Units_Per_From_Share"] == D("1")
    assert audit["CVR_Base_Value_Per_From_Share"] == ZERO
    assert audit["CVR_Max_Value_Per_From_Share"] == D("3")
    assert audit["CVR_Base_Value"] == ZERO
    assert audit["CVR_Max_Value"] == D(right_max)
    assert bool(audit["Applied_To_Position"]) is (D(shares) != ZERO)
    if D(shares) == ZERO:
        assert result.cash_flows.empty
        assert result.nontradable_rights.empty
    else:
        assert result.cash_flows.iloc[0]["Amount"] == D(cash)
        assert result.nontradable_rights.iloc[0]["Units"] == D(right_units)
    if restricted:
        released = result.restricted_proceeds_effects.iloc[0]
        assert released["Effect_Type"] == "release"
        assert released["Amount"] == D("700")


def test_stock_exchange_aggregates_successor_and_releases_offset_restriction():
    event_id = "EVT-20260507-CTRA-DVN-STOCK-EXCHANGE"
    result = _apply(
        [_event(event_id, "stock_exchange", "predecessor_extinguished")],
        [_leg(event_id, 1, "CTRA", "stock", to_asset="DVN", quantity="0.70")],
        positions={"CTRA": "-3", "DVN": "1"},
        restricted={"CTRA": "210"},
    )

    assert result.positions.to_dict() == {"DVN": D("-1.10")}
    assert result.restricted_short_proceeds.to_dict() == {"DVN": D("110")}
    effects = result.restricted_proceeds_effects
    assert effects.loc[effects["Effect_Type"].eq("transfer"), "Amount"].item() == D(
        "110"
    )
    assert effects.loc[effects["Effect_Type"].eq("release"), "Amount"].item() == D(
        "100"
    )


def test_multi_predecessor_event_aggregates_all_successor_entitlements_atomically():
    event_id = "EVT-20260507-MULTI-DVN-STOCK-EXCHANGE"
    result = _apply(
        [_event(event_id, "stock_exchange", "predecessor_extinguished")],
        [
            _leg(event_id, 1, "OLD-A", "stock", to_asset="DVN", quantity="0.5"),
            _leg(event_id, 2, "OLD-B", "stock", to_asset="DVN", quantity="0.25"),
        ],
        positions={"OLD-A": "1", "OLD-B": "2", "DVN": "1"},
    )

    assert result.positions.to_dict() == {"DVN": D("2.00")}
    assert result.audit["From_Asset_ID"].tolist() == ["OLD-A", "OLD-B"]


def test_cash_and_stock_uses_exact_fractional_shares_and_dated_cash_flow():
    event_id = "EVT-20260301-OLD-NEW-CASH-STOCK"
    result = _apply(
        [_event(event_id, "cash_and_stock", "predecessor_extinguished", date="2026-03-01")],
        [
            _leg(event_id, 1, "OLD", "cash", cash="0.75", currency="USD"),
            _leg(event_id, 2, "OLD", "stock", to_asset="NEW", quantity="0.25"),
        ],
        positions={"OLD": "1.5", "NEW": "0.5"},
    )

    assert result.positions.to_dict() == {"NEW": D("0.875")}
    flow = result.cash_flows.iloc[0]
    assert flow["Effective_Date"] == pd.Timestamp("2026-03-01")
    assert flow["Amount"] == D("1.125")
    assert result.audit.iloc[0]["Successor_Units_JSON"] == '{"NEW":"0.375"}'


def test_source_collision_cash_stock_and_relabel_aggregate_atomically():
    event_id = "EVT-20160115-ACE-CB-SOURCE-COLLISION"
    result = _apply(
        [_event(event_id, "cash_and_stock", "predecessor_extinguished", date="2016-01-15")],
        [
            _leg(
                event_id,
                1,
                "ACE",
                "relabel",
                to_asset="NEW-CB",
                quantity="1",
            ),
            _leg(
                event_id,
                2,
                "OLD-CB",
                "cash",
                cash="62.93",
                currency="USD",
            ),
            _leg(
                event_id,
                3,
                "OLD-CB",
                "stock",
                to_asset="NEW-CB",
                quantity="0.6019",
            ),
        ],
        positions={"ACE": "2", "OLD-CB": "1", "NEW-CB": "0.5"},
        start="2015-12-31",
        end="2016-01-31",
    )

    assert result.positions.to_dict() == {"NEW-CB": D("3.1019")}
    assert result.cash_flows.iloc[0]["From_Asset_ID"] == "OLD-CB"
    assert result.cash_flows.iloc[0]["Amount"] == D("62.93")
    by_from = result.audit.set_index("From_Asset_ID")
    assert by_from.loc["ACE", "Cash_Effect"] == ZERO
    assert by_from.loc["OLD-CB", "Cash_Effect"] == D("62.93")


@pytest.mark.parametrize(
    ("shares", "restricted", "expected_cash"),
    [
        ("10", {}, "145"),
        ("-4", {"CCEP.O": "400"}, "-58"),
    ],
)
def test_provider_asset_id_collision_consumes_and_recreates_same_security_key(
    shares,
    restricted,
    expected_cash,
):
    event_id = "EVT-20160528-CCE-CCEP-CASH-AND-STOCK"
    result = _apply(
        [
            _event(
                event_id,
                "cash_and_stock",
                "predecessor_extinguished",
                date="2016-05-28",
            )
        ],
        [
            _leg(
                event_id,
                1,
                "CCEP.O",
                "cash",
                cash="14.50",
                currency="USD",
            ),
            _leg(
                event_id,
                2,
                "CCEP.O",
                "stock",
                to_asset="CCEP.O",
                quantity="1",
            ),
        ],
        positions={"CCEP.O": shares},
        restricted=restricted,
        start="2016-04-30",
        end="2016-06-30",
    )

    assert result.positions.to_dict() == {"CCEP.O": D(shares)}
    assert result.cash_flows.iloc[0]["Amount"] == D(expected_cash)
    audit = result.audit.iloc[0]
    assert audit["Shares_Before"] == D(shares)
    assert audit["Shares_After"] == ZERO
    assert json.loads(audit["Successor_Units_JSON"]) == {"CCEP.O": shares}
    assert audit["Fixed_Fee"] == ZERO
    assert audit["Spread_Cost"] == ZERO
    if restricted:
        assert result.restricted_short_proceeds.to_dict() == {
            "CCEP.O": D("400")
        }
        effect = result.restricted_proceeds_effects.iloc[0]
        assert effect["Effect_Type"] == "transfer"
        assert effect["From_Asset_ID"] == "CCEP.O"
        assert effect["To_Asset_ID"] == "CCEP.O"
        assert effect["Amount"] == D("400")
    else:
        assert result.restricted_short_proceeds.empty
        assert result.restricted_proceeds_effects.empty


def test_provider_asset_id_collision_aggregates_multi_predecessor_recapitalization():
    event_id = "EVT-AMBIGUOUS-COLLISION"
    events = [
        _event(event_id, "cash_and_stock", "predecessor_extinguished")
    ]
    legs = [
        _leg(event_id, 1, "A", "cash", cash="1", currency="USD"),
        _leg(event_id, 2, "A", "stock", to_asset="A", quantity="1"),
        _leg(event_id, 3, "B", "cash", cash="1", currency="USD"),
        _leg(event_id, 4, "B", "stock", to_asset="A", quantity="0.5"),
    ]

    result = _apply(events, legs, positions={"A": 1, "B": 1})

    assert result.positions.to_dict() == {"A": D("1.5")}
    assert result.cash_flows["Amount"].tolist() == [D("1"), D("1")]
    assert set(result.audit["From_Asset_ID"]) == {"A", "B"}


def test_distribution_retains_predecessor_and_creates_exact_child_obligation():
    event_id = "EVT-20260201-PARENT-CHILD-DISTRIBUTION"
    result = _apply(
        [_event(event_id, "distribution", "predecessor_survives", date="2026-02-01")],
        [
            _leg(
                event_id,
                1,
                "PARENT",
                "distribution",
                to_asset="CHILD",
                quantity="0.5",
                consumes=False,
            )
        ],
        positions={"PARENT": "-2", "CHILD": "0.25"},
        restricted={"PARENT": "180"},
    )

    assert result.positions.to_dict() == {
        "CHILD": D("-0.75"),
        "PARENT": D("-2"),
    }
    assert result.restricted_short_proceeds.to_dict() == {"PARENT": D("180")}
    assert result.restricted_proceeds_effects.empty
    assert result.position_deliveries.to_dict("records") == [{
        "Event_ID": event_id,
        "Effective_Date": pd.Timestamp("2026-02-01"),
        "From_Asset_ID": "PARENT",
        "To_Asset_ID": "CHILD",
        "Units": D("-1"),
    }]
    assert result.audit.iloc[0]["Shares_After"] == D("-2")


def test_composite_distribution_relabels_survivor_and_distributes_child_units():
    event_id = "EVT-20200401-ARNC-HWM-ARNC-DISTRIBUTION"
    result = _apply(
        [_event(event_id, "distribution", "predecessor_survives", date="2020-04-01")],
        [
            _leg(
                event_id,
                1,
                "OLD-ARNC",
                "relabel",
                to_asset="HWM",
                quantity="1",
                consumes=True,
            ),
            _leg(
                event_id,
                2,
                "OLD-ARNC",
                "distribution",
                to_asset="NEW-ARNC",
                quantity="0.25",
                consumes=True,
            ),
        ],
        positions={"OLD-ARNC": "-4", "HWM": "1"},
        restricted={"OLD-ARNC": "400"},
        start="2020-03-31",
        end="2020-04-30",
    )

    assert result.positions.to_dict() == {
        "HWM": D("-3"),
        "NEW-ARNC": D("-1.00"),
    }
    assert result.restricted_short_proceeds.to_dict() == {"HWM": D("300")}
    effects = result.restricted_proceeds_effects
    assert effects.loc[effects["Effect_Type"].eq("transfer"), "Amount"].item() == D(
        "300"
    )
    assert effects.loc[effects["Effect_Type"].eq("release"), "Amount"].item() == D(
        "100"
    )
    audit = result.audit.iloc[0]
    assert audit["Shares_After"] == ZERO
    assert json.loads(audit["Successor_Units_JSON"]) == {
        "HWM": "-4",
        "NEW-ARNC": "-1",
    }


@pytest.mark.parametrize(
    ("shares", "restricted", "existing_child", "expected_child"),
    [
        ("8", {}, "1", "3"),
        ("-4", {"HWM": "400"}, "0", "-1"),
    ],
)
def test_distribution_provider_collision_recreates_survivor_and_child_atomically(
    shares,
    restricted,
    existing_child,
    expected_child,
):
    event_id = "EVT-20200401-ARNC-HWM-DISTRIBUTION"
    positions = {"HWM": shares}
    if D(existing_child) != ZERO:
        positions["UNPRICED::ARNC"] = existing_child
    result = _apply(
        [
            _event(
                event_id,
                "distribution",
                "predecessor_survives",
                date="2020-04-01",
            )
        ],
        [
            _leg(
                event_id,
                1,
                "HWM",
                "distribution",
                to_asset="UNPRICED::ARNC",
                quantity="1/4",
                consumes=True,
            ),
            _leg(
                event_id,
                2,
                "HWM",
                "relabel",
                to_asset="HWM",
                quantity="1",
                consumes=True,
            ),
        ],
        positions=positions,
        restricted=restricted,
        start="2020-03-31",
        end="2020-04-30",
    )

    assert result.positions.to_dict() == {
        "HWM": D(shares),
        "UNPRICED::ARNC": D(expected_child),
    }
    audit = result.audit.iloc[0]
    assert audit["Shares_Before"] == D(shares)
    assert audit["Shares_After"] == ZERO
    assert json.loads(audit["Successor_Units_JSON"]) == {
        "HWM": shares,
        "UNPRICED::ARNC": str(D(shares) / D("4")),
    }
    if restricted:
        assert result.restricted_short_proceeds.to_dict() == {"HWM": D("400")}
        effect = result.restricted_proceeds_effects.iloc[0]
        assert effect["Effect_Type"] == "transfer"
        assert effect["From_Asset_ID"] == "HWM"
        assert effect["To_Asset_ID"] == "HWM"
        assert effect["Amount"] == D("400")
    else:
        assert result.restricted_short_proceeds.empty
        assert result.restricted_proceeds_effects.empty


def test_distribution_provider_collision_aggregates_all_holder_classes():
    event_id = "EVT-AMBIGUOUS-DISTRIBUTION-COLLISION"
    events = [_event(event_id, "distribution", "predecessor_survives")]
    legs = [
        _leg(
            event_id,
            1,
            "A",
            "relabel",
            to_asset="A",
            quantity="1",
            consumes=True,
        ),
        _leg(
            event_id,
            2,
            "A",
            "distribution",
            to_asset="CHILD",
            quantity="0.25",
            consumes=True,
        ),
        _leg(
            event_id,
            3,
            "B",
            "relabel",
            to_asset="A",
            quantity="1",
            consumes=True,
        ),
        _leg(
            event_id,
            4,
            "B",
            "distribution",
            to_asset="CHILD-B",
            quantity="0.5",
            consumes=True,
        ),
    ]

    result = _apply(events, legs, positions={"A": 1, "B": 1})

    assert result.positions.to_dict() == {
        "A": D("2"),
        "CHILD": D("0.25"),
        "CHILD-B": D("0.5"),
    }
    assert set(result.audit["From_Asset_ID"]) == {"A", "B"}


@pytest.mark.parametrize("shares", ["3.591", "-3.591"])
def test_normalized_distribution_is_exact_for_long_and_short_fractional_units(
    shares,
):
    event_id = "EVT-20240927-J-AMTM-DISTRIBUTION"
    kwargs = {}
    if shares.startswith("-"):
        kwargs["restricted"] = {"J": "359.10"}
    result = _apply(
        [
            _event(
                event_id,
                "distribution",
                "predecessor_survives",
                date="2024-09-27",
            )
        ],
        [
            _leg(
                event_id,
                1,
                "J",
                "relabel",
                to_asset="J",
                quantity="1000/1197",
            ),
            _leg(
                event_id,
                2,
                "J",
                "distribution",
                to_asset="AMTM",
                quantity="1000/1197",
            ),
        ],
        positions={"J": shares},
        start="2024-08-31",
        end="2024-09-30",
        **kwargs,
    )

    expected = Fraction(3591, 1000) * Fraction(1000, 1197)
    if shares.startswith("-"):
        expected = -expected
    assert result.positions.to_dict() == {"AMTM": expected, "J": expected}
    assert len(result.audit) == 1
    assert result.audit.iloc[0]["Shares_Before"] == Fraction(shares)
    assert json.loads(result.audit.iloc[0]["Successor_Units_JSON"]) == {
        "AMTM": str(expected),
        "J": str(expected),
    }


def test_normalized_event_boundary_is_start_exclusive_and_rerun_deterministic():
    event_id = "EVT-20240927-J-AMTM-DISTRIBUTION"
    events = [
        _event(
            event_id,
            "distribution",
            "predecessor_survives",
            date="2024-09-27",
        )
    ]
    legs = [
        _leg(
            event_id,
            1,
            "J",
            "relabel",
            to_asset="J",
            quantity="1000/1197",
        ),
        _leg(
            event_id,
            2,
            "J",
            "distribution",
            to_asset="AMTM",
            quantity="1000/1197",
        ),
    ]
    first = _apply(
        events,
        legs,
        positions={"J": "1"},
        start="2024-08-31",
        end="2024-09-27",
    )
    repeated = _apply(
        events,
        legs,
        positions={"J": "1"},
        start="2024-08-31",
        end="2024-09-27",
    )
    after = _apply(
        events,
        legs,
        positions=first.positions.to_dict(),
        start="2024-09-27",
        end="2024-09-30",
    )

    assert first.positions.to_dict() == repeated.positions.to_dict()
    assert first.audit.to_dict("records") == repeated.audit.to_dict("records")
    assert len(first.audit) == 1
    assert after.audit.empty
    assert after.positions.to_dict() == first.positions.to_dict()


def test_repeating_rational_distribution_ratio_remains_mathematically_exact():
    event_id = "EVT-20210803-LB-BBWI-VSCO-DISTRIBUTION"
    result = _apply(
        [_event(event_id, "distribution", "predecessor_survives", date="2021-08-03")],
        [
            _leg(
                event_id,
                1,
                "LB",
                "relabel",
                to_asset="BBWI",
                quantity="1",
                consumes=True,
            ),
            _leg(
                event_id,
                2,
                "LB",
                "distribution",
                to_asset="VSCO",
                quantity="1/3",
                consumes=True,
            ),
        ],
        positions={"LB": "1"},
        start="2021-07-31",
        end="2021-08-31",
    )

    assert result.positions["VSCO"].numerator == 1
    assert result.positions["VSCO"].denominator == 3
    assert json.loads(result.audit.iloc[0]["Successor_Units_JSON"])["VSCO"] == "1/3"


def test_cancellation_extinguishes_position_and_releases_short_restriction():
    event_id = "EVT-20260601-OLD-CANCELLED"
    result = _apply(
        [_event(event_id, "cancellation", "cancelled", date="2026-06-01")],
        [_leg(event_id, 1, "OLD", "cancellation")],
        positions={"OLD": "-4.5"},
        restricted={"OLD": "90"},
    )

    assert result.positions.empty
    assert result.restricted_short_proceeds.empty
    assert result.cash_flows.empty
    assert result.restricted_proceeds_effects.iloc[0]["Amount"] == D("90")


def test_same_date_events_are_dependency_ordered_for_chained_transitions():
    # Lexical order is deliberately the reverse of dependency order.
    first = "EVT-Z-A-B"
    second = "EVT-A-B-C"
    result = _apply(
        [
            _event(first, "identity_continuity", "same_security", date="2026-02-01"),
            _event(second, "identity_continuity", "same_security", date="2026-02-01"),
        ],
        [
            _leg(first, 1, "A", "relabel", to_asset="B", quantity="1"),
            _leg(second, 1, "B", "relabel", to_asset="C", quantity="1"),
        ],
        positions={"A": "2.125"},
    )

    assert result.positions.to_dict() == {"C": D("2.125")}
    assert result.audit["Event_ID"].tolist() == [first, second]


def test_interval_is_start_exclusive_and_end_inclusive():
    before = "EVT-BEFORE"
    start = "EVT-START"
    end = "EVT-END"
    after = "EVT-AFTER"
    events = [
        _event(before, "identity_continuity", "same_security", date="2026-01-31"),
        _event(start, "identity_continuity", "same_security", date="2026-02-01"),
        _event(end, "identity_continuity", "same_security", date="2026-03-01"),
        _event(after, "identity_continuity", "same_security", date="2026-03-02"),
    ]
    legs = [
        _leg(before, 1, "X", "relabel", to_asset="A", quantity="1"),
        _leg(start, 1, "A", "relabel", to_asset="B", quantity="1"),
        _leg(end, 1, "A", "relabel", to_asset="C", quantity="1"),
        _leg(after, 1, "C", "relabel", to_asset="D", quantity="1"),
    ]

    result = _apply(
        events,
        legs,
        positions={"A": "1"},
        start="2026-02-01",
        end="2026-03-01",
    )

    assert result.positions.to_dict() == {"C": D("1")}
    assert result.audit["Event_ID"].tolist() == [end]


@pytest.mark.parametrize("approval_target", ["event", "leg", "source"])
def test_unapproved_terms_or_sources_fail_closed(approval_target):
    event_id = "EVT-UNAPPROVED"
    events = [
        _event(
            event_id,
            "cash_settlement",
            "predecessor_extinguished",
            status="pending" if approval_target == "event" else "approved",
        )
    ]
    legs = [
        _leg(
            event_id,
            1,
            "OLD",
            "cash",
            cash="1",
            currency="USD",
            status="pending" if approval_target == "leg" else "approved",
        )
    ]
    sources = [
        _source(event_id, status="pending" if approval_target == "source" else "approved")
    ]

    with pytest.raises(ValueError, match="must be approved"):
        _apply(events, legs, positions={"OLD": 1}, sources=sources)


@pytest.mark.parametrize(
    ("events", "legs", "message"),
    [
        (
            [_event("EVT-BAD", "mystery_merger", "predecessor_extinguished")],
            [_leg("EVT-BAD", 1, "OLD", "cash", cash="1", currency="USD")],
            "Unsupported corporate-action event types",
        ),
        (
            [_event("EVT-BAD", "cash_and_stock", "predecessor_extinguished")],
            [_leg("EVT-BAD", 1, "OLD", "cash", cash="1", currency="USD")],
            "incomplete or ambiguous",
        ),
        (
            [_event("EVT-BAD", "identity_continuity", "same_security")],
            [_leg("EVT-BAD", 1, "OLD", "relabel", to_asset="NEW", quantity="0.9")],
            "relabel ratio must equal one",
        ),
    ],
)
def test_unsupported_or_ambiguous_structures_fail_closed(events, legs, message):
    with pytest.raises(ValueError, match=message):
        _apply(events, legs, positions={"OLD": 1})


def test_short_multi_successor_exchange_fails_without_restriction_allocation_terms():
    event_id = "EVT-MULTI-SUCCESSOR"
    events = [_event(event_id, "stock_exchange", "predecessor_extinguished")]
    legs = [
        _leg(event_id, 1, "OLD", "stock", to_asset="NEW-A", quantity="0.5"),
        _leg(event_id, 2, "OLD", "stock", to_asset="NEW-B", quantity="0.25"),
    ]

    with pytest.raises(ValueError, match="cannot allocate restricted"):
        _apply(events, legs, positions={"OLD": -2}, restricted={"OLD": 100})


def test_zero_position_is_audited_without_inventing_effects():
    event_id = "EVT-ZERO"
    result = _apply(
        [_event(event_id, "cash_settlement", "predecessor_extinguished")],
        [_leg(event_id, 1, "OLD", "cash", cash="70", currency="USD")],
    )

    assert result.positions.empty
    assert result.cash_flows.empty
    assert result.restricted_proceeds_effects.empty
    assert result.nontradable_rights.empty
    audit = result.audit.iloc[0]
    assert bool(audit["Applied_To_Position"]) is False
    assert audit["Cash_Per_From_Share"] == D("70")
    assert audit["Cash_Effect"] == ZERO
    assert json.loads(audit["Source_URLs_JSON"]) == [
        f"https://example.test/{event_id}"
    ]
