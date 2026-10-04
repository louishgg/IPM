"""One-share valuation across prepared corporate-action intervals."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from fractions import Fraction

import pandas as pd

from .corporate_actions import (
    CorporateActionResult,
    apply_corporate_actions,
    parse_exact_number,
)


EndPriceLookup = Callable[[str], object | None]


def has_any_interval_event(
    events: pd.DataFrame,
    legs: pd.DataFrame,
    asset_ids: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> bool:
    """Return whether any supplied asset has a prepared event in ``(start, end]``."""
    if events.empty or legs.empty:
        return False
    asset_ids = {str(asset_id) for asset_id in asset_ids}
    if not asset_ids:
        return False
    dates = pd.to_datetime(events["Effective_Date"])
    event_ids = set(
        events.loc[
            dates.gt(pd.Timestamp(start)) & dates.le(pd.Timestamp(end)),
            "Event_ID",
        ].astype(str)
    )
    if not event_ids:
        return False
    touched = legs.loc[
        legs["Event_ID"].astype(str).isin(event_ids), "From_Asset_ID"
    ]
    return not asset_ids.isdisjoint(map(str, touched))


def has_interval_event(
    events: pd.DataFrame,
    legs: pd.DataFrame,
    asset_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> bool:
    """Return whether an asset has a prepared event in ``(start, end]``."""
    return has_any_interval_event(events, legs, (asset_id,), start, end)


def value_one_share_through_events(
    events: pd.DataFrame,
    legs: pd.DataFrame,
    sources: pd.DataFrame,
    asset_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    end_price: EndPriceLookup,
) -> tuple[CorporateActionResult, Fraction] | None:
    """Return complete end value for one share affected by interval events.

    Cash, nontradable rights, and successor positions are valued together. A
    missing or invalid price for any surviving/successor position makes the
    interval unavailable and returns ``None``.
    """
    if not has_interval_event(events, legs, asset_id, start, end):
        return None

    result = apply_corporate_actions(
        {str(asset_id): Fraction(1)},
        {},
        events,
        legs,
        sources,
        start_exclusive=start,
        end_inclusive=end,
    )
    end_value = sum(
        (parse_exact_number(row.Amount) for row in result.cash_flows.itertuples()),
        Fraction(0),
    )
    end_value += sum(
        (
            parse_exact_number(row.Base_Value)
            for row in result.nontradable_rights.itertuples()
        ),
        Fraction(0),
    )
    for successor, units in result.positions.items():
        price = end_price(str(successor))
        if price is None:
            return None
        try:
            exact_price = parse_exact_number(price)
        except (TypeError, ValueError):
            return None
        if exact_price <= 0:
            return None
        end_value += parse_exact_number(units) * exact_price
    return result, end_value


__all__ = [
    "EndPriceLookup",
    "has_any_interval_event",
    "has_interval_event",
    "value_one_share_through_events",
]
