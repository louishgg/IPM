"""Deterministic accounting for reviewed mandatory corporate actions.

The canonical provenance tables describe *what* happened.  This module is the
small accounting kernel that applies already-resolved asset-level event legs to
signed share positions. It deliberately does not fetch evidence, value traded
securities, or accrue interest. Public composition belongs in
``portfolio_core.portfolio_lifecycle``.

All arithmetic uses exact :class:`fractions.Fraction` values constructed from
canonical decimal or rational strings.  Ratios such as ``1/3`` therefore stay
mathematically exact, shares are never rounded to integers, and no cash in lieu
is inferred.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
from typing import Mapping

import pandas as pd


EVENT_COLUMNS = [
    "Event_ID",
    "Effective_Date",
    "Event_Type",
    "Continuity_Class",
    "Review_Status",
]

LEG_COLUMNS = [
    "Event_ID",
    "Leg_Order",
    "From_Asset_ID",
    "To_Asset_ID",
    "Leg_Type",
    "Quantity_Per_From_Share",
    "Cash_Per_From_Share",
    "Currency",
    "CVR_Units_Per_From_Share",
    "CVR_Base_Value_Per_Unit",
    "CVR_Max_Value_Per_Unit",
    "Consumes_From_Position",
    "Review_Status",
]

SOURCE_COLUMNS = ["Event_ID", "Source_URL", "Review_Status"]

CASH_FLOW_COLUMNS = [
    "Event_ID",
    "Effective_Date",
    "Leg_Order",
    "From_Asset_ID",
    "Currency",
    "From_Shares",
    "Cash_Per_From_Share",
    "Amount",
]

POSITION_DELIVERY_COLUMNS = [
    "Event_ID",
    "Effective_Date",
    "From_Asset_ID",
    "To_Asset_ID",
    "Units",
]

RESTRICTED_PROCEEDS_EFFECT_COLUMNS = [
    "Event_ID",
    "Effective_Date",
    "From_Asset_ID",
    "To_Asset_ID",
    "Effect_Type",
    "Amount",
]

NONTRADABLE_RIGHT_COLUMNS = [
    "Event_ID",
    "Effective_Date",
    "Leg_Order",
    "From_Asset_ID",
    "Right_Asset_ID",
    "Currency",
    "From_Shares",
    "Units_Per_From_Share",
    "Units",
    "Base_Value_Per_Unit",
    "Max_Value_Per_Unit",
    "Base_Value",
    "Max_Value",
]

AUDIT_COLUMNS = [
    "Event_ID",
    "Effective_Date",
    "Event_Type",
    "Continuity_Class",
    "From_Asset_ID",
    "Shares_Before",
    "Shares_After",
    "Successor_Units_JSON",
    "Cash_Per_From_Share",
    "Cash_Effect",
    "Currency",
    "Restricted_Proceeds_Transferred",
    "Restricted_Proceeds_Released",
    "CVR_Units_Per_From_Share",
    "CVR_Base_Value_Per_From_Share",
    "CVR_Max_Value_Per_From_Share",
    "CVR_Units",
    "CVR_Base_Value",
    "CVR_Max_Value",
    "Fixed_Fee",
    "Spread_Cost",
    "Applied_To_Position",
    "Source_URLs_JSON",
]


SUPPORTED_EVENT_TYPES = frozenset(
    {
        "identity_continuity",
        "otc_transition",
        "cash_settlement",
        "stock_exchange",
        "cash_and_stock",
        "distribution",
        "cancellation",
    }
)
SUPPORTED_CONTINUITY_CLASSES = frozenset(
    {
        "same_security",
        "predecessor_extinguished",
        "predecessor_survives",
        "cancelled",
    }
)
SUPPORTED_LEG_TYPES = frozenset(
    {
        "relabel",
        "cash",
        "stock",
        "distribution",
        "cvr",
        "cancellation",
    }
)

ZERO = Fraction(0)
ONE = Fraction(1)


@dataclass(frozen=True, slots=True)
class CorporateActionResult:
    """State and dated effects after applying an event interval.

    ``positions`` and ``restricted_short_proceeds`` contain ``Fraction`` values
    in sorted asset order.  The result data frames also retain exact fractions;
    callers should convert only at their explicit storage or valuation boundary.
    """

    positions: pd.Series
    restricted_short_proceeds: pd.Series
    position_deliveries: pd.DataFrame
    cash_flows: pd.DataFrame
    restricted_proceeds_effects: pd.DataFrame
    nontradable_rights: pd.DataFrame
    audit: pd.DataFrame


@dataclass(frozen=True, slots=True)
class _IncomingRestriction:
    from_asset_id: str
    short_units: Fraction
    amount: Fraction


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _text(value: object, label: str, *, allow_empty: bool = False) -> str:
    if pd.isna(value):
        result = ""
    else:
        result = str(value).strip()
    if not allow_empty and not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _exact_number(
    value: object,
    label: str,
    *,
    allow_empty: bool = True,
) -> Fraction:
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        if allow_empty:
            return ZERO
        raise ValueError(f"{label} must not be empty")
    text = str(value).strip()
    try:
        if "/" in text:
            if text.count("/") != 1:
                raise ValueError
            numerator, denominator = text.split("/", maxsplit=1)
            if not numerator or not denominator:
                raise ValueError
            result = Fraction(int(numerator), int(denominator))
        else:
            decimal = Decimal(text)
            if not decimal.is_finite():
                raise ValueError
            result = Fraction(decimal)
    except (InvalidOperation, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"{label} must be a finite decimal or integer ratio N/D"
        ) from error
    return result


def parse_exact_number(value: object, *, label: str = "value") -> Fraction:
    """Parse a finite decimal or integer ratio without binary/decimal rounding."""

    return _exact_number(value, label, allow_empty=False)


def parse_explicit_boolean(value: object, label: str) -> bool:
    """Parse only explicit true/false representations."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{label} must be an explicit boolean")


def _exact_series(
    values: Mapping[str, object] | pd.Series,
    label: str,
    *,
    nonnegative: bool = False,
) -> pd.Series:
    series = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
    if series.index.has_duplicates:
        raise ValueError(f"{label} contains duplicate asset identifiers")
    normalized: dict[str, Fraction] = {}
    for raw_asset, raw_value in series.items():
        asset_id = _text(raw_asset, f"{label} asset identifier")
        value = _exact_number(
            raw_value, f"{label}[{asset_id}]", allow_empty=False
        )
        if nonnegative and value < ZERO:
            raise ValueError(f"{label}[{asset_id}] must be nonnegative")
        if value != ZERO:
            normalized[asset_id] = value
    return pd.Series(normalized, dtype=object).sort_index()


def _frame(rows: list[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def _factor_count(value: int, factor: int) -> int:
    count = 0
    while value % factor == 0:
        value //= factor
        count += 1
    return count


def _exact_string(value: Fraction) -> str:
    """Use canonical decimal text when finite, otherwise an exact ratio."""

    denominator = value.denominator
    reduced = denominator
    for factor in (2, 5):
        while reduced % factor == 0:
            reduced //= factor
    if reduced != 1:
        return str(value)
    places = max(
        _factor_count(denominator, 2),
        _factor_count(denominator, 5),
    )
    scaled = value.numerator * (10**places) // denominator
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).zfill(places + 1)
    if places == 0:
        return f"{sign}{digits}"
    integer = digits[:-places]
    fractional = digits[-places:].rstrip("0")
    return f"{sign}{integer}.{fractional}" if fractional else f"{sign}{integer}"


def _json_decimal_mapping(values: Mapping[str, Fraction]) -> str:
    return json.dumps(
        {
            key: _exact_string(value)
            for key, value in sorted(values.items())
            if value != ZERO
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_tables(
    events: pd.DataFrame,
    legs: pd.DataFrame,
    sources: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(events, EVENT_COLUMNS, "Corporate-action events")
    _require_columns(legs, LEG_COLUMNS, "Corporate-action legs")
    _require_columns(sources, SOURCE_COLUMNS, "Corporate-action sources")

    normalized_events = events.copy()
    normalized_legs = legs.copy()
    normalized_sources = sources.copy()

    for frame, label in (
        (normalized_events, "event"),
        (normalized_legs, "leg"),
        (normalized_sources, "source"),
    ):
        frame["Event_ID"] = frame["Event_ID"].map(
            lambda value: _text(value, f"{label} Event_ID")
        )

    if normalized_events["Event_ID"].duplicated().any():
        raise ValueError("Corporate-action Event_ID values must be unique")
    event_ids = set(normalized_events["Event_ID"])
    if not event_ids:
        if not normalized_legs.empty or not normalized_sources.empty:
            raise ValueError("Corporate-action legs or sources exist without events")
        normalized_events["Effective_Date"] = pd.to_datetime(
            normalized_events["Effective_Date"]
        )
        return normalized_events, normalized_legs, normalized_sources

    orphan_legs = sorted(set(normalized_legs["Event_ID"]) - event_ids)
    orphan_sources = sorted(set(normalized_sources["Event_ID"]) - event_ids)
    if orphan_legs:
        raise ValueError(f"Corporate-action legs reference unknown events: {orphan_legs}")
    if orphan_sources:
        raise ValueError(
            f"Corporate-action sources reference unknown events: {orphan_sources}"
        )
    missing_legs = sorted(event_ids - set(normalized_legs["Event_ID"]))
    missing_sources = sorted(event_ids - set(normalized_sources["Event_ID"]))
    if missing_legs:
        raise ValueError(f"Corporate-action events have no legs: {missing_legs}")
    if missing_sources:
        raise ValueError(f"Corporate-action events have no sources: {missing_sources}")

    try:
        normalized_events["Effective_Date"] = pd.to_datetime(
            normalized_events["Effective_Date"], errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Corporate-action Effective_Date values are invalid") from error
    if normalized_events["Effective_Date"].isna().any():
        raise ValueError("Corporate-action Effective_Date values are invalid")
    if normalized_events["Effective_Date"].dt.tz is not None:
        raise ValueError("Corporate-action Effective_Date values must be timezone-naive")
    normalized_events["Effective_Date"] = normalized_events[
        "Effective_Date"
    ].dt.normalize()

    for column in ("Event_Type", "Continuity_Class", "Review_Status"):
        normalized_events[column] = normalized_events[column].map(
            lambda value, name=column: _text(value, f"event {name}").lower()
        )
    if not normalized_events["Review_Status"].eq("approved").all():
        raise ValueError("Every executable corporate-action event must be approved")
    unsupported_events = sorted(
        set(normalized_events["Event_Type"]) - SUPPORTED_EVENT_TYPES
    )
    if unsupported_events:
        raise ValueError(f"Unsupported corporate-action event types: {unsupported_events}")
    unsupported_continuity = sorted(
        set(normalized_events["Continuity_Class"])
        - SUPPORTED_CONTINUITY_CLASSES
    )
    if unsupported_continuity:
        raise ValueError(
            "Unsupported corporate-action continuity classes: "
            f"{unsupported_continuity}"
        )

    normalized_legs["Leg_Type"] = normalized_legs["Leg_Type"].map(
        lambda value: _text(value, "leg Leg_Type").lower()
    )
    normalized_legs["From_Asset_ID"] = normalized_legs["From_Asset_ID"].map(
        lambda value: _text(value, "leg From_Asset_ID")
    )
    normalized_legs["To_Asset_ID"] = normalized_legs["To_Asset_ID"].map(
        lambda value: _text(value, "leg To_Asset_ID", allow_empty=True)
    )
    normalized_legs["Currency"] = normalized_legs["Currency"].map(
        lambda value: _text(value, "leg Currency", allow_empty=True).upper()
    )
    normalized_legs["Review_Status"] = normalized_legs["Review_Status"].map(
        lambda value: _text(value, "leg Review_Status").lower()
    )
    if not normalized_legs["Review_Status"].eq("approved").all():
        raise ValueError("Every executable corporate-action leg must be approved")
    unsupported_legs = sorted(
        set(normalized_legs["Leg_Type"]) - SUPPORTED_LEG_TYPES
    )
    if unsupported_legs:
        raise ValueError(f"Unsupported corporate-action leg types: {unsupported_legs}")

    leg_orders: list[int] = []
    for row in normalized_legs.itertuples(index=False):
        raw_order = _exact_number(
            row.Leg_Order, f"{row.Event_ID} Leg_Order", allow_empty=False
        )
        if raw_order.denominator != 1 or raw_order < ONE:
            raise ValueError(f"{row.Event_ID} Leg_Order must be a positive integer")
        leg_orders.append(int(raw_order))
    normalized_legs["Leg_Order"] = leg_orders
    if normalized_legs[["Event_ID", "Leg_Order"]].duplicated().any():
        raise ValueError("Leg_Order values must be unique within each Event_ID")

    decimal_columns = [
        "Quantity_Per_From_Share",
        "Cash_Per_From_Share",
        "CVR_Units_Per_From_Share",
        "CVR_Base_Value_Per_Unit",
        "CVR_Max_Value_Per_Unit",
    ]
    for column in decimal_columns:
        normalized_legs[column] = [
            _exact_number(value, f"{event_id} {column}")
            for value, event_id in zip(
                normalized_legs[column], normalized_legs["Event_ID"], strict=True
            )
        ]
        if normalized_legs[column].map(lambda value: value < ZERO).any():
            raise ValueError(f"Corporate-action {column} values must be nonnegative")
    normalized_legs["Consumes_From_Position"] = [
        parse_explicit_boolean(value, f"{event_id} Consumes_From_Position")
        for value, event_id in zip(
            normalized_legs["Consumes_From_Position"],
            normalized_legs["Event_ID"],
            strict=True,
        )
    ]

    normalized_sources["Source_URL"] = normalized_sources["Source_URL"].map(
        lambda value: _text(value, "source Source_URL")
    )
    normalized_sources["Review_Status"] = normalized_sources["Review_Status"].map(
        lambda value: _text(value, "source Review_Status").lower()
    )
    if not normalized_sources["Review_Status"].eq("approved").all():
        raise ValueError("Every corporate-action source must be approved")
    _validate_event_structures(normalized_events, normalized_legs)
    return normalized_events, normalized_legs, normalized_sources


def _validate_event_structures(events: pd.DataFrame, legs: pd.DataFrame) -> None:
    events_by_id = events.set_index("Event_ID")
    for event_id, event_legs in legs.groupby("Event_ID", sort=False):
        event = events_by_id.loc[event_id]
        event_type = str(event["Event_Type"])
        continuity = str(event["Continuity_Class"])
        leg_types = set(event_legs["Leg_Type"])

        expected: tuple[str, set[str], set[str]]
        if event_type in {"identity_continuity", "otc_transition"}:
            expected = ("same_security", {"relabel"}, {"relabel"})
        elif event_type == "cash_settlement":
            expected = ("predecessor_extinguished", {"cash"}, {"cash", "cvr"})
        elif event_type == "stock_exchange":
            expected = ("predecessor_extinguished", {"stock"}, {"stock"})
        elif event_type == "cash_and_stock":
            expected = (
                "predecessor_extinguished",
                {"cash", "stock"},
                {"cash", "stock", "cvr", "relabel"},
            )
        elif event_type == "distribution":
            expected = (
                "predecessor_survives",
                {"distribution"},
                {"distribution", "relabel"},
            )
        else:
            expected = ("cancelled", {"cancellation"}, {"cancellation"})
        expected_continuity, required_types, allowed_types = expected
        if continuity != expected_continuity:
            raise ValueError(
                f"{event_id} has {event_type} with incompatible continuity "
                f"class {continuity}"
            )
        if not required_types.issubset(leg_types) or not leg_types.issubset(
            allowed_types
        ):
            raise ValueError(
                f"{event_id} has an incomplete or ambiguous {event_type} leg set"
            )

        for from_asset, from_legs in event_legs.groupby(
            "From_Asset_ID", sort=False
        ):
            consumes = set(from_legs["Consumes_From_Position"])
            if len(consumes) != 1:
                raise ValueError(
                    f"{event_id}/{from_asset} legs disagree about predecessor consumption"
                )
            consumes_from = consumes.pop()
            # ``predecessor_survives`` is an economic continuity statement.
            # A composite separation can still consume the old *identifier*,
            # create its renamed ratio-1 survivor, and distribute child units.
            # A pure distribution retains the exact predecessor identifier.
            if event_type == "distribution":
                should_consume = from_legs["Leg_Type"].eq("relabel").any()
            else:
                should_consume = continuity != "predecessor_survives"
            if bool(consumes_from) != should_consume:
                raise ValueError(
                    f"{event_id}/{from_asset} has an invalid consumption policy"
                )

            relabels = from_legs.loc[from_legs["Leg_Type"].eq("relabel")]
            if len(relabels) > 1:
                raise ValueError(
                    f"{event_id}/{from_asset} has multiple survivor relabel legs"
                )
            if event_type in {"identity_continuity", "otc_transition"} and len(
                relabels
            ) != 1:
                raise ValueError(
                    f"{event_id}/{from_asset} must have exactly one relabel leg"
                )

            for leg in from_legs.itertuples(index=False):
                quantity = leg.Quantity_Per_From_Share
                cash = leg.Cash_Per_From_Share
                cvr_units = leg.CVR_Units_Per_From_Share
                cvr_base = leg.CVR_Base_Value_Per_Unit
                cvr_max = leg.CVR_Max_Value_Per_Unit
                to_asset = str(leg.To_Asset_ID)
                currency = str(leg.Currency)
                label = f"{event_id} leg {leg.Leg_Order}"

                if leg.Leg_Type in {
                    "relabel",
                    "stock",
                    "distribution",
                }:
                    if not to_asset:
                        raise ValueError(f"{label} requires a To_Asset_ID")
                    same_asset_stock = (
                        leg.Leg_Type == "stock"
                        and event_type in {"stock_exchange", "cash_and_stock"}
                        and to_asset == from_asset
                    )
                    same_asset_distribution_survivor = (
                        leg.Leg_Type == "relabel"
                        and event_type == "distribution"
                        and to_asset == from_asset
                        and quantity > ZERO
                    )
                    if (
                        to_asset == from_asset
                        and not same_asset_stock
                        and not same_asset_distribution_survivor
                    ):
                        raise ValueError(f"{label} requires a distinct To_Asset_ID")
                    if quantity <= ZERO:
                        raise ValueError(f"{label} requires a positive share quantity")
                    if cash != ZERO or cvr_units != ZERO or cvr_base != ZERO or cvr_max != ZERO:
                        raise ValueError(f"{label} mixes incompatible leg economics")
                    if (
                        leg.Leg_Type == "relabel"
                        and not same_asset_distribution_survivor
                        and quantity != ONE
                    ):
                        raise ValueError(f"{label} same-security relabel ratio must equal one")
                elif leg.Leg_Type == "cash":
                    if to_asset or not currency or cash <= ZERO:
                        raise ValueError(f"{label} has incomplete cash terms")
                    if quantity != ZERO or cvr_units != ZERO or cvr_base != ZERO or cvr_max != ZERO:
                        raise ValueError(f"{label} mixes incompatible leg economics")
                elif leg.Leg_Type == "cvr":
                    units = cvr_units if cvr_units != ZERO else quantity
                    if not to_asset or not currency or units <= ZERO:
                        raise ValueError(f"{label} has incomplete CVR terms")
                    if cash != ZERO or cvr_max < cvr_base:
                        raise ValueError(f"{label} has invalid CVR economics")
                    if cvr_units != ZERO and quantity != ZERO:
                        raise ValueError(f"{label} supplies two CVR unit quantities")
                else:
                    if to_asset or currency or any(
                        value != ZERO
                        for value in (quantity, cash, cvr_units, cvr_base, cvr_max)
                    ):
                        raise ValueError(f"{label} cancellation terms must be empty")

            monetary_currencies = set(
                from_legs.loc[
                    from_legs["Leg_Type"].isin(["cash", "cvr"]), "Currency"
                ]
            )
            if len(monetary_currencies) > 1:
                raise ValueError(
                    f"{event_id}/{from_asset} contains multiple monetary currencies"
                )

    # A provider can reuse one local Asset_ID for a predecessor and the event's
    # successor.  This also occurs in multi-predecessor combinations where one
    # holder class is recapitalized in place while another converts into it
    # (Tyco/Johnson Controls is the real example).  The executor intentionally
    # snapshots every predecessor, removes all consumed units, and only then
    # aggregates successor deltas, so these are not execution-order cycles.
    # Leg-level validation above still gates the only valid same-key forms.


def _ordered_events(events: pd.DataFrame, legs: pd.DataFrame) -> list[str]:
    """Order events chronologically and dependency-order same-date chains."""

    ordered: list[str] = []
    for _, date_events in events.groupby("Effective_Date", sort=True):
        ids = set(date_events["Event_ID"])
        date_legs = legs.loc[legs["Event_ID"].isin(ids)]
        from_by_event = {
            event_id: set(rows["From_Asset_ID"])
            for event_id, rows in date_legs.groupby("Event_ID", sort=False)
        }
        to_by_event = {
            event_id: set(rows["To_Asset_ID"]) - {""}
            for event_id, rows in date_legs.groupby("Event_ID", sort=False)
        }
        dependencies: dict[str, set[str]] = {event_id: set() for event_id in ids}
        for predecessor in ids:
            for successor in ids - {predecessor}:
                if to_by_event.get(predecessor, set()) & from_by_event.get(
                    successor, set()
                ):
                    dependencies[successor].add(predecessor)

        remaining = set(ids)
        while remaining:
            ready = sorted(
                event_id
                for event_id in remaining
                if not (dependencies[event_id] & remaining)
            )
            if not ready:
                raise ValueError(
                    "Same-date corporate-action dependencies contain a cycle: "
                    f"{sorted(remaining)}"
                )
            ordered.extend(ready)
            remaining.difference_update(ready)
    return ordered


def apply_corporate_actions(
    positions: Mapping[str, object] | pd.Series,
    restricted_short_proceeds: Mapping[str, object] | pd.Series,
    events: pd.DataFrame,
    legs: pd.DataFrame,
    sources: pd.DataFrame,
    *,
    start_exclusive: object,
    end_inclusive: object,
) -> CorporateActionResult:
    """Apply all approved corporate actions in ``(start, end]``.

    Positions are signed shares.  A positive restricted-proceeds balance may
    exist only for an asset with a negative position.  The kernel returns dated
    cash flows and restricted-proceeds movements but never changes a cash
    balance or calculates interest.
    """

    current_positions = _exact_series(positions, "positions")
    current_restricted = _exact_series(
        restricted_short_proceeds,
        "restricted_short_proceeds",
        nonnegative=True,
    )
    for asset_id, amount in current_restricted.items():
        if current_positions.get(asset_id, ZERO) >= ZERO and amount > ZERO:
            raise ValueError(
                "Restricted short proceeds require a negative position for "
                f"{asset_id}"
            )

    normalized_events, normalized_legs, normalized_sources = _normalize_tables(
        events, legs, sources
    )
    try:
        start = pd.Timestamp(start_exclusive).normalize()
        end = pd.Timestamp(end_inclusive).normalize()
    except (TypeError, ValueError) as error:
        raise ValueError("Corporate-action interval dates are invalid") from error
    if pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None:
        raise ValueError("Corporate-action interval dates must be finite and timezone-naive")
    if end < start:
        raise ValueError("Corporate-action interval end precedes its start")

    selected_events = normalized_events.loc[
        normalized_events["Effective_Date"].gt(start)
        & normalized_events["Effective_Date"].le(end)
    ].copy()
    selected_ids = set(selected_events["Event_ID"])
    selected_legs = normalized_legs.loc[
        normalized_legs["Event_ID"].isin(selected_ids)
    ].copy()
    source_urls = {
        event_id: sorted(set(rows["Source_URL"]))
        for event_id, rows in normalized_sources.groupby("Event_ID", sort=False)
    }

    cash_rows: list[dict[str, object]] = []
    delivery_rows: list[dict[str, object]] = []
    restricted_rows: list[dict[str, object]] = []
    right_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    events_by_id = selected_events.set_index("Event_ID")
    for event_id in _ordered_events(selected_events, selected_legs):
        event = events_by_id.loc[event_id]
        event_date = pd.Timestamp(event["Effective_Date"])
        event_legs = selected_legs.loc[selected_legs["Event_ID"].eq(event_id)].sort_values(
            "Leg_Order", kind="stable"
        )
        pre_positions = current_positions.to_dict()
        pre_restricted = current_restricted.to_dict()
        position_deltas: dict[str, Fraction] = defaultdict(lambda: ZERO)
        incoming_by_successor: dict[str, list[_IncomingRestriction]] = defaultdict(list)
        audit_details: dict[str, dict[str, object]] = {}
        consumed_assets: set[str] = set()

        for from_asset, from_legs in event_legs.groupby(
            "From_Asset_ID", sort=False
        ):
            before = current_positions.get(from_asset, ZERO)
            restriction_before = current_restricted.get(from_asset, ZERO)
            consumes = bool(from_legs["Consumes_From_Position"].iloc[0])
            share_effects: dict[str, Fraction] = defaultdict(lambda: ZERO)
            cash_per_share = ZERO
            cash_effect = ZERO
            cvr_units_total = ZERO
            cvr_units_per_share_total = ZERO
            cvr_base_per_share_total = ZERO
            cvr_max_per_share_total = ZERO
            cvr_base_total = ZERO
            cvr_max_total = ZERO
            currency = ""

            for leg in from_legs.itertuples(index=False):
                if leg.Leg_Type in {
                    "relabel",
                    "stock",
                    "distribution",
                }:
                    quantity = before * leg.Quantity_Per_From_Share
                    position_deltas[leg.To_Asset_ID] += quantity
                    share_effects[leg.To_Asset_ID] += quantity
                elif leg.Leg_Type == "cash":
                    amount = before * leg.Cash_Per_From_Share
                    cash_per_share += leg.Cash_Per_From_Share
                    cash_effect += amount
                    currency = leg.Currency
                    if amount != ZERO:
                        cash_rows.append(
                            {
                                "Event_ID": event_id,
                                "Effective_Date": event_date,
                                "Leg_Order": int(leg.Leg_Order),
                                "From_Asset_ID": from_asset,
                                "Currency": leg.Currency,
                                "From_Shares": before,
                                "Cash_Per_From_Share": leg.Cash_Per_From_Share,
                                "Amount": amount,
                            }
                        )
                elif leg.Leg_Type == "cvr":
                    units_per_share = (
                        leg.CVR_Units_Per_From_Share
                        if leg.CVR_Units_Per_From_Share != ZERO
                        else leg.Quantity_Per_From_Share
                    )
                    units = before * units_per_share
                    base_value = units * leg.CVR_Base_Value_Per_Unit
                    max_value = units * leg.CVR_Max_Value_Per_Unit
                    cvr_units_total += units
                    cvr_units_per_share_total += units_per_share
                    cvr_base_per_share_total += (
                        units_per_share * leg.CVR_Base_Value_Per_Unit
                    )
                    cvr_max_per_share_total += (
                        units_per_share * leg.CVR_Max_Value_Per_Unit
                    )
                    cvr_base_total += base_value
                    cvr_max_total += max_value
                    currency = leg.Currency
                    if units != ZERO:
                        right_rows.append(
                            {
                                "Event_ID": event_id,
                                "Effective_Date": event_date,
                                "Leg_Order": int(leg.Leg_Order),
                                "From_Asset_ID": from_asset,
                                "Right_Asset_ID": leg.To_Asset_ID,
                                "Currency": leg.Currency,
                                "From_Shares": before,
                                "Units_Per_From_Share": units_per_share,
                                "Units": units,
                                "Base_Value_Per_Unit": leg.CVR_Base_Value_Per_Unit,
                                "Max_Value_Per_Unit": leg.CVR_Max_Value_Per_Unit,
                                "Base_Value": base_value,
                                "Max_Value": max_value,
                            }
                        )

            if consumes:
                consumed_assets.add(from_asset)
                current_positions = current_positions.drop(
                    index=from_asset, errors="ignore"
                )
                current_restricted = current_restricted.drop(
                    index=from_asset, errors="ignore"
                )
                if restriction_before > ZERO:
                    share_successors = sorted(
                        {
                            str(row.To_Asset_ID)
                            for row in from_legs.itertuples(index=False)
                            if row.Leg_Type
                            in {"relabel", "stock"}
                        }
                    )
                    if len(share_successors) > 1:
                        raise ValueError(
                            f"{event_id}/{from_asset} cannot allocate restricted "
                            "short proceeds across multiple successors"
                        )
                    if share_successors:
                        successor = share_successors[0]
                        successor_units = -min(share_effects[successor], ZERO)
                        if successor_units <= ZERO:
                            raise ValueError(
                                f"{event_id}/{from_asset} has restricted proceeds "
                                "without a successor short obligation"
                            )
                        incoming_by_successor[successor].append(
                            _IncomingRestriction(
                                from_asset_id=from_asset,
                                short_units=successor_units,
                                amount=restriction_before,
                            )
                        )
                    else:
                        restricted_rows.append(
                            {
                                "Event_ID": event_id,
                                "Effective_Date": event_date,
                                "From_Asset_ID": from_asset,
                                "To_Asset_ID": "",
                                "Effect_Type": "release",
                                "Amount": restriction_before,
                            }
                        )

            audit_details[from_asset] = {
                "before": before,
                "after": ZERO if consumes else before,
                "share_effects": share_effects,
                "cash_per_share": cash_per_share,
                "cash_effect": cash_effect,
                "currency": currency,
                "cvr_units": cvr_units_total,
                "cvr_units_per_share": cvr_units_per_share_total,
                "cvr_base_per_share": cvr_base_per_share_total,
                "cvr_max_per_share": cvr_max_per_share_total,
                "cvr_base": cvr_base_total,
                "cvr_max": cvr_max_total,
            }
            for to_asset, units in sorted(share_effects.items()):
                if units != ZERO:
                    delivery_rows.append({
                        "Event_ID": event_id,
                        "Effective_Date": event_date,
                        "From_Asset_ID": from_asset,
                        "To_Asset_ID": to_asset,
                        "Units": units,
                    })

        # All predecessor removals happen before successor additions, making a
        # multi-predecessor event independent of leg row order.
        for to_asset, delta in sorted(position_deltas.items()):
            current_positions.loc[to_asset] = (
                current_positions.get(to_asset, ZERO) + delta
            )

        # Reconcile restrictions after netting successor positions.  Existing
        # and incoming restrictions are released proportionally when mandatory
        # share receipts offset an existing short obligation.
        for to_asset, delta in sorted(position_deltas.items()):
            # A provider may resolve both predecessor and successor to the
            # same local Asset_ID.  In that narrow case the old position and
            # restriction were consumed above; only the recreated successor
            # units and their incoming restriction belong in reconciliation.
            if to_asset in consumed_assets:
                old_position = ZERO
                old_restriction = ZERO
            else:
                old_position = pre_positions.get(to_asset, ZERO)
                old_restriction = pre_restricted.get(to_asset, ZERO)
            incoming = incoming_by_successor.get(to_asset, [])
            total_short_units = -min(old_position, ZERO) + sum(
                (item.short_units for item in incoming), ZERO
            )
            final_short_units = -min(current_positions.get(to_asset, ZERO), ZERO)
            retained_ratio = (
                final_short_units / total_short_units
                if total_short_units > ZERO
                else ZERO
            )
            if retained_ratio > ONE:
                # This can only be new short units without associated proceeds
                # (for example, a distribution owed by an existing short).
                retained_ratio = ONE

            old_retained = old_restriction * retained_ratio
            old_released = old_restriction - old_retained
            final_restriction = old_retained
            if old_released > ZERO:
                restricted_rows.append(
                    {
                        "Event_ID": event_id,
                        "Effective_Date": event_date,
                        "From_Asset_ID": to_asset,
                        "To_Asset_ID": "",
                        "Effect_Type": "release",
                        "Amount": old_released,
                    }
                )

            for item in incoming:
                transferred = item.amount * retained_ratio
                released = item.amount - transferred
                final_restriction += transferred
                if transferred > ZERO:
                    restricted_rows.append(
                        {
                            "Event_ID": event_id,
                            "Effective_Date": event_date,
                            "From_Asset_ID": item.from_asset_id,
                            "To_Asset_ID": to_asset,
                            "Effect_Type": "transfer",
                            "Amount": transferred,
                        }
                    )
                if released > ZERO:
                    restricted_rows.append(
                        {
                            "Event_ID": event_id,
                            "Effective_Date": event_date,
                            "From_Asset_ID": item.from_asset_id,
                            "To_Asset_ID": "",
                            "Effect_Type": "release",
                            "Amount": released,
                        }
                    )
            if final_restriction > ZERO:
                current_restricted.loc[to_asset] = final_restriction
            else:
                current_restricted = current_restricted.drop(
                    index=to_asset, errors="ignore"
                )

        transferred_by_from: dict[str, Fraction] = defaultdict(lambda: ZERO)
        released_by_from: dict[str, Fraction] = defaultdict(lambda: ZERO)
        for row in restricted_rows:
            if row["Event_ID"] != event_id:
                continue
            if row["Effect_Type"] == "transfer":
                transferred_by_from[str(row["From_Asset_ID"])] += row["Amount"]
            else:
                released_by_from[str(row["From_Asset_ID"])] += row["Amount"]

        urls_json = json.dumps(
            source_urls[event_id], separators=(",", ":")
        )
        for from_asset, details in audit_details.items():
            audit_rows.append(
                {
                    "Event_ID": event_id,
                    "Effective_Date": event_date,
                    "Event_Type": str(event["Event_Type"]),
                    "Continuity_Class": str(event["Continuity_Class"]),
                    "From_Asset_ID": from_asset,
                    "Shares_Before": details["before"],
                    "Shares_After": details["after"],
                    "Successor_Units_JSON": _json_decimal_mapping(
                        details["share_effects"]
                    ),
                    "Cash_Per_From_Share": details["cash_per_share"],
                    "Cash_Effect": details["cash_effect"],
                    "Currency": details["currency"],
                    "Restricted_Proceeds_Transferred": transferred_by_from[from_asset],
                    "Restricted_Proceeds_Released": released_by_from[from_asset],
                    "CVR_Units_Per_From_Share": details["cvr_units_per_share"],
                    "CVR_Base_Value_Per_From_Share": details[
                        "cvr_base_per_share"
                    ],
                    "CVR_Max_Value_Per_From_Share": details[
                        "cvr_max_per_share"
                    ],
                    "CVR_Units": details["cvr_units"],
                    "CVR_Base_Value": details["cvr_base"],
                    "CVR_Max_Value": details["cvr_max"],
                    "Fixed_Fee": ZERO,
                    "Spread_Cost": ZERO,
                    "Applied_To_Position": details["before"] != ZERO,
                    "Source_URLs_JSON": urls_json,
                }
            )

        current_positions = current_positions.loc[
            current_positions.map(lambda value: value != ZERO)
        ].sort_index()
        current_restricted = current_restricted.loc[
            current_restricted.map(lambda value: value != ZERO)
        ].sort_index()
        for asset_id, amount in current_restricted.items():
            if amount < ZERO or current_positions.get(asset_id, ZERO) >= ZERO:
                raise RuntimeError(
                    f"Corporate-action restriction invariant failed for {asset_id}"
                )

    return CorporateActionResult(
        positions=current_positions,
        restricted_short_proceeds=current_restricted,
        position_deliveries=_frame(
            delivery_rows, POSITION_DELIVERY_COLUMNS
        ),
        cash_flows=_frame(cash_rows, CASH_FLOW_COLUMNS),
        restricted_proceeds_effects=_frame(
            restricted_rows, RESTRICTED_PROCEEDS_EFFECT_COLUMNS
        ),
        nontradable_rights=_frame(right_rows, NONTRADABLE_RIGHT_COLUMNS),
        audit=_frame(audit_rows, AUDIT_COLUMNS),
    )


__all__ = [
    "AUDIT_COLUMNS",
    "CASH_FLOW_COLUMNS",
    "CorporateActionResult",
    "EVENT_COLUMNS",
    "LEG_COLUMNS",
    "NONTRADABLE_RIGHT_COLUMNS",
    "POSITION_DELIVERY_COLUMNS",
    "RESTRICTED_PROCEEDS_EFFECT_COLUMNS",
    "SOURCE_COLUMNS",
    "SUPPORTED_CONTINUITY_CLASSES",
    "SUPPORTED_EVENT_TYPES",
    "SUPPORTED_LEG_TYPES",
    "apply_corporate_actions",
    "parse_explicit_boolean",
    "parse_exact_number",
]
