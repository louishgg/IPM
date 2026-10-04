"""Public lifecycle valuation and event-financing services."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd

from .corporate_actions import (
    CorporateActionResult,
    EVENT_COLUMNS,
    LEG_COLUMNS,
    SOURCE_COLUMNS,
    apply_corporate_actions,
    parse_exact_number,
)
from .interval_valuation import (
    has_any_interval_event, has_interval_event, value_one_share_through_events,
)
from .accounting_config import (
    DEFAULT_ACCOUNTING_CONFIG,
    PortfolioAccountingConfig,
)
from .accounting_ledger import (
    LedgerState,
    project_financing_amounts,
)


EVENT_DELIVERY_EXECUTION_COLUMNS = (
    "Event_ID",
    "From_Asset_IDs",
    "Asset_ID",
    "Execution_Date",
    "Reference_Close",
    "Volume",
)


@dataclass(frozen=True, slots=True)
class LedgerAdvanceResult:
    """One reconciled account transition through dated mandatory events."""

    state: LedgerState
    corporate_actions: CorporateActionResult
    interest_amount: float
    cash_interest_credit: float
    loan_interest_charge: float
    event_interest_items: tuple[tuple[str, float], ...]
    counterfactual_interest_without_actions: float

    @property
    def event_interest_effects(self) -> dict[str, float]:
        return dict(self.event_interest_items)


@dataclass(frozen=True, slots=True)
class _FinancingProjection:
    cash: float
    restricted_total: float
    interest_amount: float
    cash_interest_credit: float
    loan_interest_charge: float


def prepared_event_tables(
    portfolio_data: object,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = getattr(portfolio_data, "security_events", pd.DataFrame())
    legs = getattr(portfolio_data, "security_event_legs", pd.DataFrame())
    sources = getattr(portfolio_data, "security_event_sources", pd.DataFrame())
    return events, legs, sources


def finite_positive_price(
    data_close: pd.DataFrame,
    date: pd.Timestamp,
    asset_id: str,
) -> float | None:
    if asset_id not in data_close.columns or date not in data_close.index:
        return None
    value = data_close.at[date, asset_id]
    if pd.isna(value):
        return None
    price = float(value)
    return price if np.isfinite(price) and price > 0.0 else None


def event_delivery_execution_candidates(
    portfolio_data: object,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Return pre-derived off-universe deliveries executable in an interval."""

    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if pd.isna(start_date) or pd.isna(end_date) or end_date < start_date:
        raise RuntimeError("Event-delivery execution bounds are invalid")
    result = getattr(
        portfolio_data, "event_delivery_executions", pd.DataFrame()
    ).copy()
    if result.empty:
        return pd.DataFrame(columns=EVENT_DELIVERY_EXECUTION_COLUMNS)
    result = result.loc[
        result["Execution_Date"].gt(start_date)
        & result["Execution_Date"].le(end_date)
    ].copy()
    return result.sort_values(
        ["Execution_Date", "Event_ID", "Asset_ID"], kind="stable"
    ).reset_index(drop=True)


def _finite_positive_price_map(
    prices: Mapping[str, object],
) -> dict[str, float]:
    """Return only finite positive prices under normalized Asset_ID keys."""

    result: dict[str, float] = {}
    for asset_id, value in prices.items():
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(price) and price > 0.0:
            result[str(asset_id)] = price
    return result


def event_valuation_result(
    portfolio_data: object,
    asset_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    successor_price_overrides: Mapping[str, object] | None = None,
):
    """Return a one-share event result when it supplies end-date value."""
    events, legs, sources = prepared_event_tables(portfolio_data)
    overrides = _finite_positive_price_map(successor_price_overrides or {})
    return value_one_share_through_events(
        events,
        legs,
        sources,
        str(asset_id),
        pd.Timestamp(start),
        pd.Timestamp(end),
        lambda successor: overrides.get(str(successor))
        or finite_positive_price(
            portfolio_data.data_close, pd.Timestamp(end), successor
        ),
    )


def lifecycle_eligible_assets(
    asset_ids: Iterable[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    start_prices: Mapping[str, object],
    end_prices: Mapping[str, object],
    events: pd.DataFrame,
    legs: pd.DataFrame,
    sources: pd.DataFrame,
    successor_price_overrides: Mapping[str, object] | None = None,
) -> frozenset[str]:
    """Apply the sole lifecycle-eligibility rule to normalized interval inputs.

    Most constituents never appear as a corporate-action predecessor. Their
    lifecycle contract reduces to positive start and end prices. Only the small
    reviewed predecessor set needs the full extinction and event-valuation
    checks.
    """

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    valid_start_prices = _finite_positive_price_map(start_prices)
    valid_end_prices = _finite_positive_price_map(end_prices)
    valid_successor_prices = {
        **valid_end_prices,
        **_finite_positive_price_map(successor_price_overrides or {}),
    }
    event_assets = (
        set(legs["From_Asset_ID"].astype(str)) if not legs.empty else set()
    )
    eligible: set[str] = set()
    for value in asset_ids:
        asset_id = str(value)
        if asset_id not in valid_start_prices:
            continue
        if asset_id not in event_assets:
            if asset_id in valid_end_prices:
                eligible.add(asset_id)
            continue
        if asset_is_extinguished_from_events(
            events,
            legs,
            asset_id,
            start,
        ):
            continue
        if has_interval_event(events, legs, asset_id, start, end):
            valuation = value_one_share_through_events(
                events,
                legs,
                sources,
                asset_id,
                start,
                end,
                lambda successor: valid_successor_prices.get(str(successor)),
            )
            if valuation is None:
                continue
        elif asset_id not in valid_end_prices:
            continue
        eligible.add(asset_id)
    return frozenset(eligible)


def eligible_assets_for_interval(
    portfolio_data: object,
    asset_ids: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> frozenset[str]:
    """Adapt prepared wide prices to the canonical lifecycle eligibility rule."""

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    data_close = portfolio_data.data_close
    events, legs, sources = prepared_event_tables(portfolio_data)
    execution_candidates = event_delivery_execution_candidates(
        portfolio_data, start, end
    )
    if execution_candidates["Asset_ID"].astype(str).duplicated().any():
        raise RuntimeError("Event-delivery successor prices are ambiguous")
    successor_price_overrides = execution_candidates.set_index("Asset_ID")[
        "Reference_Close"
    ].to_dict()
    return lifecycle_eligible_assets(
        asset_ids,
        start=start,
        end=end,
        start_prices=(
            data_close.loc[start].to_dict()
            if start in data_close.index
            else {}
        ),
        end_prices=(
            data_close.loc[end].to_dict()
            if end in data_close.index
            else {}
        ),
        events=events,
        legs=legs,
        sources=sources,
        successor_price_overrides=successor_price_overrides,
    )


def asset_has_interval_event(
    portfolio_data: object,
    asset_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> bool:
    events, legs, _ = prepared_event_tables(portfolio_data)
    return has_interval_event(
        events,
        legs,
        str(asset_id),
        pd.Timestamp(start),
        pd.Timestamp(end),
    )


def prepared_action_bool(value: object, *, label: str) -> bool:
    """Parse one already-prepared action boolean without truthy strings."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise RuntimeError(f"Invalid prepared corporate-action {label}: {value!r}")


def asset_is_extinguished_from_events(
    events: pd.DataFrame,
    legs: pd.DataFrame,
    asset_id: str,
    as_of: pd.Timestamp,
) -> bool:
    """Return whether event tables extinguish a predecessor by ``as_of``.

    Point-in-time membership and a stale provider quote can both linger on the
    effective date of a before-open merger.  They must not make the predecessor
    eligible for a new position.  A consuming event therefore extinguishes its
    prepared ``From_Asset_ID`` unless an approved stock or relabel leg recreates
    the same local key.  The latter is the reviewed provider-key collision used
    by events such as CCE/CCEP and old-Arconic/Howmet.
    """

    if events.empty or legs.empty:
        return False

    required_event_columns = {"Event_ID", "Effective_Date", "Review_Status"}
    required_leg_columns = {
        "Event_ID",
        "From_Asset_ID",
        "To_Asset_ID",
        "Leg_Type",
        "Consumes_From_Position",
        "Review_Status",
    }
    if not required_event_columns.issubset(events.columns):
        missing = sorted(required_event_columns - set(events.columns))
        raise RuntimeError(
            f"Prepared corporate-action events lack columns: {missing}"
        )
    if not required_leg_columns.issubset(legs.columns):
        missing = sorted(required_leg_columns - set(legs.columns))
        raise RuntimeError(
            f"Prepared corporate-action legs lack columns: {missing}"
        )

    try:
        cutoff = pd.Timestamp(as_of)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Corporate-action eligibility date is invalid") from error
    if pd.isna(cutoff) or cutoff.tz is not None:
        raise RuntimeError(
            "Corporate-action eligibility date must be finite and timezone-naive"
        )

    event_dates = pd.to_datetime(events["Effective_Date"], errors="coerce")
    if event_dates.isna().any():
        raise RuntimeError("Prepared corporate-action events contain invalid dates")
    selected_events = events.loc[event_dates.le(cutoff)].copy()
    if selected_events.empty:
        return False
    if not selected_events["Review_Status"].astype(str).str.lower().eq(
        "approved"
    ).all():
        raise RuntimeError(
            "Unapproved corporate-action evidence cannot control candidate "
            "eligibility"
        )

    selected_ids = set(selected_events["Event_ID"].astype(str))
    selected_legs = legs.loc[
        legs["Event_ID"].astype(str).isin(selected_ids)
        & legs["From_Asset_ID"].astype(str).eq(str(asset_id))
    ].copy()
    if selected_legs.empty:
        return False
    if not selected_legs["Review_Status"].astype(str).str.lower().eq(
        "approved"
    ).all():
        raise RuntimeError(
            "Unapproved corporate-action terms cannot control candidate "
            "eligibility"
        )

    event_dates_by_id = dict(
        zip(
            selected_events["Event_ID"].astype(str),
            event_dates.loc[selected_events.index],
            strict=True,
        )
    )
    extinguished = False
    for event_id, event_legs in sorted(
        selected_legs.groupby("Event_ID", sort=False),
        key=lambda item: (event_dates_by_id[str(item[0])], str(item[0])),
    ):
        consumes = {
            prepared_action_bool(
                value,
                label=f"{event_id}/{asset_id} Consumes_From_Position",
            )
            for value in event_legs["Consumes_From_Position"]
        }
        if len(consumes) != 1:
            raise RuntimeError(
                f"Prepared event {event_id}/{asset_id} has inconsistent "
                "predecessor-consumption terms"
            )
        if not consumes.pop():
            continue
        same_key = event_legs["To_Asset_ID"].astype(str).eq(str(asset_id))
        leg_types = event_legs["Leg_Type"].astype(str).str.lower()
        unit_relabels = pd.Series(False, index=event_legs.index, dtype=bool)
        relabel_rows = leg_types.eq("relabel")
        if relabel_rows.any():
            unit_relabels.loc[relabel_rows] = event_legs.loc[
                relabel_rows, "Quantity_Per_From_Share"
            ].map(lambda value: parse_exact_number(value) == Fraction(1))
        # Preserve the checkpoint's candidate-identity contract: an ordinary
        # same-key relabel survives selection, while a non-unit executable
        # relabel consumes the pre-event provider-normalized candidate units.
        # The action ledger still recreates and values the resulting position.
        same_key_survives = (
            same_key
            & (
                leg_types.eq("stock")
                | (leg_types.eq("relabel") & unit_relabels)
            )
        ).any()
        extinguished = not bool(same_key_survives)
    return extinguished


def period_has_position_event(
    portfolio_data: object,
    positions: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> bool:
    """Return whether an approved interval event can touch a live position."""

    events, legs, _ = prepared_event_tables(portfolio_data)
    return has_any_interval_event(
        events, legs, positions.loc[positions.ne(0.0)].index, start, end,
    )


def _project_event_financing(
    cash: float,
    restricted_total: float,
    cash_flows: pd.DataFrame,
    restricted_proceeds_effects: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    accounting_config: PortfolioAccountingConfig,
    *,
    validate_restricted_balance: bool,
) -> _FinancingProjection:
    """Project financing once across dated cash and restriction effects."""

    current_cash = float(cash)
    current_restricted = float(restricted_total)
    total_interest = 0.0
    cash_interest_credit = 0.0
    loan_interest_charge = 0.0
    cursor = pd.Timestamp(start)
    effect_dates = sorted(
        set(pd.to_datetime(cash_flows.get("Effective_Date", [])))
        | set(
            pd.to_datetime(
                restricted_proceeds_effects.get("Effective_Date", [])
            )
        )
    )
    for raw_event_date in effect_dates:
        event_date = pd.Timestamp(raw_event_date)
        (
            current_cash,
            interest,
            cash_credit,
            loan_charge,
        ) = project_financing_amounts(
            current_cash,
            current_restricted,
            cursor,
            event_date,
            accounting_config,
        )
        total_interest += interest
        cash_interest_credit += cash_credit
        loan_interest_charge += loan_charge
        if not cash_flows.empty:
            dated_cash = cash_flows.loc[
                pd.to_datetime(cash_flows["Effective_Date"]).eq(event_date),
                "Amount",
            ]
            current_cash += sum(float(value) for value in dated_cash)
        if not restricted_proceeds_effects.empty:
            released = restricted_proceeds_effects.loc[
                pd.to_datetime(
                    restricted_proceeds_effects["Effective_Date"]
                ).eq(event_date)
                & restricted_proceeds_effects["Effect_Type"].eq("release"),
                "Amount",
            ]
            current_restricted -= sum(float(value) for value in released)
            if validate_restricted_balance:
                if current_restricted < -1e-8:
                    raise RuntimeError(
                        "Corporate actions released more restricted proceeds "
                        "than exist"
                    )
                current_restricted = max(0.0, current_restricted)
        cursor = event_date
    (
        current_cash,
        interest,
        cash_credit,
        loan_charge,
    ) = project_financing_amounts(
        current_cash,
        current_restricted,
        cursor,
        pd.Timestamp(end),
        accounting_config,
    )
    return _FinancingProjection(
        cash=float(current_cash),
        restricted_total=float(current_restricted),
        interest_amount=float(total_interest + interest),
        cash_interest_credit=float(cash_interest_credit + cash_credit),
        loan_interest_charge=float(loan_interest_charge + loan_charge),
    )


def attribute_event_interest_effects(
    cash: float,
    starting_restricted: pd.Series,
    result,
    start: pd.Timestamp,
    end: pd.Timestamp,
    accounting_config: PortfolioAccountingConfig,
) -> tuple[dict[str, float], float, float]:
    """Attribute financing timing to events by chronological marginal effect.

    The no-action scenario is the counterfactual baseline. Events are then
    introduced in the exact order emitted by the shared action kernel; each
    event receives the change in total interval interest caused by adding its
    dated cash flows and restricted-proceeds releases. The contributions
    therefore reconcile exactly (within floating precision) to actual interest
    less the no-action counterfactual, including chained events.
    """

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    initial_restricted = float(
        sum(
            (parse_exact_number(value) for value in starting_restricted),
            Fraction(0),
        )
    )
    event_order = tuple(dict.fromkeys(result.audit["Event_ID"].astype(str)))

    def _scenario_interest(included: set[str]) -> float:
        cash_flows = result.cash_flows.loc[
            result.cash_flows["Event_ID"].astype(str).isin(included)
        ]
        restriction_effects = result.restricted_proceeds_effects.loc[
            result.restricted_proceeds_effects["Event_ID"]
            .astype(str)
            .isin(included)
        ]
        projection = _project_event_financing(
            cash,
            initial_restricted,
            cash_flows,
            restriction_effects,
            start,
            end,
            accounting_config,
            validate_restricted_balance=False,
        )
        return projection.interest_amount

    counterfactual_interest = _scenario_interest(set())
    previous_interest = counterfactual_interest
    included: set[str] = set()
    effects: dict[str, float] = {}
    for event_id in event_order:
        included.add(event_id)
        scenario_interest = _scenario_interest(included)
        effects[event_id] = float(scenario_interest - previous_interest)
        previous_interest = scenario_interest
    return effects, float(previous_interest), float(counterfactual_interest)


def advance_ledger(
    state: LedgerState,
    end: object,
    events: pd.DataFrame,
    legs: pd.DataFrame,
    sources: pd.DataFrame,
    accounting_config: PortfolioAccountingConfig = DEFAULT_ACCOUNTING_CONFIG,
) -> LedgerAdvanceResult:
    """Advance a dated ledger through events in (state_date, end] and return its audit.

    Financing accrues between event dates using the cash and restricted proceeds
    then in effect. A same-date advance accrues no interest and replays no events.
    """

    if state.state_date is None:
        raise ValueError("Ledger state must be dated before lifecycle advancement")
    start = pd.Timestamp(state.state_date)
    end_date = pd.Timestamp(end)
    if pd.isna(end_date) or end_date.tz is not None:
        raise ValueError("Lifecycle end must be finite and timezone-naive")
    if end_date < start:
        raise ValueError("Lifecycle end cannot precede the ledger state date")

    normalized_events = (
        pd.DataFrame(columns=EVENT_COLUMNS) if events.empty else events
    )
    normalized_legs = pd.DataFrame(columns=LEG_COLUMNS) if legs.empty else legs
    normalized_sources = (
        pd.DataFrame(columns=SOURCE_COLUMNS) if sources.empty else sources
    )

    action_result = apply_corporate_actions(
        state.shares,
        state.restricted_short_proceeds,
        normalized_events,
        normalized_legs,
        normalized_sources,
        start_exclusive=start,
        end_inclusive=end_date,
    )
    (
        event_interest_effects,
        attributed_interest,
        counterfactual_interest,
    ) = attribute_event_interest_effects(
        state.cash,
        state.restricted_short_proceeds,
        action_result,
        start,
        end_date,
        accounting_config,
    )

    if not action_result.cash_flows.empty:
        currencies = set(action_result.cash_flows["Currency"].astype(str))
        if currencies - {"USD"}:
            raise RuntimeError(
                "Corporate-action cash flows must be denominated in USD"
            )
    projection = _project_event_financing(
        state.cash,
        state.restricted_total,
        action_result.cash_flows,
        action_result.restricted_proceeds_effects,
        start,
        end_date,
        accounting_config,
        validate_restricted_balance=True,
    )
    expected_restricted = sum(
        (float(value) for value in action_result.restricted_short_proceeds),
        0.0,
    )
    if not np.isclose(
        projection.restricted_total,
        expected_restricted,
        atol=1e-8,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Corporate-action restricted-proceeds effects do not reconcile "
            "to the transformed position state"
        )
    if not np.isclose(
        projection.interest_amount,
        projection.cash_interest_credit - projection.loan_interest_charge,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError("Financing split does not reconcile to net interest")
    if not np.isclose(
        attributed_interest,
        projection.interest_amount,
        atol=1e-8,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Corporate-action interest attribution does not reconcile"
        )
    next_state = LedgerState._from_series(
        action_result.positions.map(float),
        projection.cash,
        action_result.restricted_short_proceeds.map(float),
        end_date,
    )
    return LedgerAdvanceResult(
        state=next_state,
        corporate_actions=action_result,
        interest_amount=projection.interest_amount,
        cash_interest_credit=projection.cash_interest_credit,
        loan_interest_charge=projection.loan_interest_charge,
        event_interest_items=tuple(event_interest_effects.items()),
        counterfactual_interest_without_actions=float(counterfactual_interest),
    )


__all__ = [
    "EVENT_DELIVERY_EXECUTION_COLUMNS",
    "asset_is_extinguished_from_events",
    "eligible_assets_for_interval",
    "lifecycle_eligible_assets",
    "asset_has_interval_event",
    "LedgerAdvanceResult",
    "advance_ledger",
    "attribute_event_interest_effects",
    "event_valuation_result",
    "event_delivery_execution_candidates",
    "finite_positive_price",
    "period_has_position_event",
    "prepared_event_tables",
]
