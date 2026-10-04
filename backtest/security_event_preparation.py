"""Resolve canonical security events into backtest Asset_ID accounting inputs.

The shared provenance bundle is ticker/provider based.  The historical engine
holds the immutable Reuters ``Asset_ID`` values from the supplied price file,
so preparation performs the one allowed scope-specific step: resolve each
approved event leg to the exact predecessor and successor identifiers present
in the prepared backtest namespace.  Free-text terms are never interpreted.
"""

from __future__ import annotations

import pandas as pd
from portfolio_core.io import atomic_write_dataframe
from portfolio_core.dates import month_end_index

from portfolio_core.corporate_actions import (
    EVENT_COLUMNS,
    LEG_COLUMNS,
    SOURCE_COLUMNS,
    parse_exact_number,
)
from portfolio_core.security_identity import (
    SecurityIdentityBundle,
    UNAVAILABLE_PREFIX,
    load_security_identity_bundle,
)
from portfolio_core.provider_identity import normalize_identity_key
from portfolio_core.portfolio_lifecycle import finite_positive_price

from .config import DEFAULT_CONFIG, BacktestMarketConfig
from .paths import BacktestPaths


AUDIT_COLUMNS = [
    "Event_ID",
    "Effective_Date",
    "From_Ticker",
    "From_Asset_ID",
    "Previous_Decision_Date",
    "Next_Valuation_Date",
    "Potential_Holding_Crossing",
    "Accounting_Treatment",
    "Executing_Event_ID",
    "End_Valuation_Status",
    "Validation_Status",
    "Source_URLs",
    "Notes",
]

def _verified_continuous_distribution_policy(
    bundle: SecurityIdentityBundle,
    event: dict[str, object],
    from_ticker: str,
    from_asset_id: str,
    resolved_legs: list[dict[str, object]],
) -> str | None:
    """Recognize one adjusted Reuters survivor series across a separation.

    The policy is deliberately structural rather than ticker-specific.  It
    requires a reviewed canonical composite distribution, a ratio-1 survivor
    relabel back to the same prepared Asset_ID, only unpriced child legs, and
    one hashed Reuters mapping whose local series spans the effective date.
    A separately priced distribution such as DWDP/DOW does not match.
    """

    event_id = str(event["Event_ID"])
    if (
        str(event["Event_Type"]) != "distribution"
        or str(event["Continuity_Class"]) != "predecessor_survives"
        or str(event["Accounting_Status"]) != "executable"
        or str(event["Review_Status"]) != "approved"
    ):
        return None

    canonical = bundle.legs.loc[
        bundle.legs["Event_ID"].eq(event_id)
        & bundle.legs["From_Ticker"].eq(from_ticker)
    ].copy()
    relabels = canonical.loc[canonical["Leg_Type"].eq("relabel")]
    distributions = canonical.loc[canonical["Leg_Type"].eq("distribution")]
    if (
        len(relabels) != 1
        or distributions.empty
        or len(canonical) != len(relabels) + len(distributions)
        or not canonical["Review_Status"].eq("approved").all()
        or not canonical["Retain_Predecessor"].astype(str).eq("False").all()
        or parse_exact_number(relabels.iloc[0]["Share_Ratio"]) != 1
    ):
        return None

    resolved = pd.DataFrame(resolved_legs)
    if resolved.empty:
        return None
    resolved_relabels = resolved.loc[resolved["Leg_Type"].eq("relabel")]
    resolved_distributions = resolved.loc[
        resolved["Leg_Type"].eq("distribution")
    ]
    if (
        len(resolved_relabels) != 1
        or resolved_distributions.empty
        or len(resolved) != len(resolved_relabels) + len(resolved_distributions)
        or str(resolved_relabels.iloc[0]["To_Asset_ID"]) != str(from_asset_id)
        or not resolved_distributions["To_Asset_ID"]
        .astype(str)
        .str.startswith("UNPRICED::")
        .all()
    ):
        return None

    mappings = bundle.provider_mappings
    evidence_columns = {
        "Scope",
        "Provider",
        "Event_ID",
        "Source_Ticker",
        "Asset_ID",
        "Legacy_From_Ticker",
        "Legacy_To_Ticker",
        "Local_First_Date",
        "Local_Last_Date",
        "Simultaneous_Symbol_Conflict",
        "Provider_Evidence_Status",
        "Review_Status",
    }
    if not evidence_columns.issubset(mappings.columns):
        return None
    survivor_ticker = str(relabels.iloc[0]["To_Ticker"])
    rows = mappings.loc[
        mappings["Scope"].eq("backtest")
        & mappings["Provider"].eq("reuters")
        & mappings["Event_ID"].eq(event_id)
        & mappings["Source_Ticker"].eq(from_ticker)
        & mappings["Asset_ID"].eq(from_asset_id)
        & mappings["Legacy_From_Ticker"].eq(from_ticker)
        & mappings["Legacy_To_Ticker"].eq(survivor_ticker)
    ]
    if len(rows) != 1:
        return None
    mapping = rows.iloc[0]
    effective = pd.Timestamp(event["Effective_Date"])
    try:
        first = pd.Timestamp(mapping["Local_First_Date"])
        last = pd.Timestamp(mapping["Local_Last_Date"])
    except (TypeError, ValueError):
        return None
    conflict = str(mapping["Simultaneous_Symbol_Conflict"]).strip().lower()
    if (
        pd.isna(first)
        or pd.isna(last)
        or first > effective
        or last < effective
        or conflict not in {"false", "0"}
        or str(mapping["Provider_Evidence_Status"])
        != "local_reuters_observation"
        or str(mapping["Review_Status"]) != "approved"
    ):
        return None
    children = ";".join(
        sorted(
            resolved_distributions["To_Asset_ID"]
            .astype(str)
            .str.removeprefix("UNPRICED::")
        )
    )
    return (
        f"The reviewed, hashed Reuters {from_asset_id} series spans the "
        f"separation into survivor {survivor_ticker}; distributed child "
        f"{children} is not a separate priced Asset_ID, so the provider's "
        "continuous adjusted price ratio is used without applying the child "
        "leg again."
    )


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    values = normalized.map({"true": True, "false": False, "1": True, "0": False})
    if values.isna().any():
        raise ValueError("Ticker/RIC resolution contains an invalid Is_Priced value")
    return values.astype(bool)


def _event_mapping_assets(
    bundle: SecurityIdentityBundle,
    event_id: str,
    ticker: str,
    *,
    relation: str,
) -> tuple[str, ...]:
    mappings = bundle.provider_mappings
    rows = mappings.loc[
        mappings["Scope"].eq("backtest")
        & mappings["Provider"].eq("reuters")
        & mappings["Event_ID"].eq(event_id)
    ].copy()
    legacy_column = (
        "Legacy_From_Ticker" if relation == "from" else "Legacy_To_Ticker"
    )
    exact = rows.loc[rows[legacy_column].eq(ticker)]
    # ``Source_Ticker`` can identify the predecessor when the migrated record
    # omitted a legacy-from label.  It is not safe successor evidence: after a
    # separation the old source ticker may name a newly distributed security
    # while the same Reuters row follows the renamed survivor (ARNC/HWM).
    if exact.empty and relation == "from":
        exact = rows.loc[rows["Source_Ticker"].eq(ticker)]
    return tuple(sorted(
        asset
        for asset in exact["Asset_ID"].astype(str).unique()
        if asset and not asset.startswith(UNAVAILABLE_PREFIX)
    ))


def _resolved_assets_at(
    resolution: pd.DataFrame,
    ticker: str,
    date: pd.Timestamp,
) -> tuple[str, ...]:
    rows = resolution.loc[
        resolution["Date"].eq(date)
        & resolution["Source_Ticker"].eq(ticker)
        & resolution["Is_Priced"]
    ]
    return tuple(sorted(
        asset for asset in rows["Asset_ID"].astype(str).unique() if asset
    ))


def _predecessor_assets(
    bundle: SecurityIdentityBundle,
    resolution: pd.DataFrame,
    prices: pd.DataFrame,
    event_id: str,
    ticker: str,
    previous_date: pd.Timestamp,
) -> tuple[str, ...]:
    active = _resolved_assets_at(resolution, ticker, previous_date)
    linked = _event_mapping_assets(
        bundle, event_id, ticker, relation="from"
    )
    intersection = tuple(sorted(set(active) & set(linked)))
    if intersection:
        return intersection
    if len(active) == 1:
        return active
    if linked:
        return linked
    if active:
        return active
    normalized = tuple(sorted(
        str(asset)
        for asset in prices.columns
        if normalize_identity_key(str(asset)) == normalize_identity_key(ticker)
    ))
    return normalized


def _successor_asset(
    bundle: SecurityIdentityBundle,
    resolution: pd.DataFrame,
    prices: pd.DataFrame,
    event_id: str,
    ticker: str,
    next_date: pd.Timestamp,
) -> str:
    linked = list(_event_mapping_assets(
        bundle, event_id, ticker, relation="to"
    ))
    active = list(_resolved_assets_at(resolution, ticker, next_date))
    normalized = [
        str(asset)
        for asset in prices.columns
        if normalize_identity_key(str(asset)) == normalize_identity_key(ticker)
    ]
    candidates = list(dict.fromkeys(linked + active + normalized))
    valued = [
        asset
        for asset in candidates
        if finite_positive_price(prices, next_date, asset) is not None
    ]
    if len(valued) == 1:
        return valued[0]
    if len(active) == 1 and active[0] in valued:
        return active[0]
    if len(linked) == 1 and linked[0] in valued:
        return linked[0]
    if not candidates:
        # Keeping the reviewed ticker visible makes a missing successor price
        # explicit; the engine will exclude the candidate rather than inventing
        # a value or a cash-in-lieu amount.
        return f"UNPRICED::{ticker}"
    raise ValueError(
        f"Cannot resolve one Reuters successor for {event_id}/{ticker}: "
        f"{candidates}"
    )


def _from_tickers_for_event(
    bundle: SecurityIdentityBundle,
    event_id: str,
) -> tuple[str, ...]:
    legs = bundle.legs.loc[bundle.legs["Event_ID"].eq(event_id)]
    values = set(legs["From_Ticker"].astype(str))
    mappings = bundle.provider_mappings.loc[
        bundle.provider_mappings["Scope"].eq("backtest")
        & bundle.provider_mappings["Provider"].eq("reuters")
        & bundle.provider_mappings["Event_ID"].eq(event_id)
    ]
    values.update(
        value for value in mappings["Legacy_From_Ticker"].astype(str) if value
    )
    return tuple(sorted(value for value in values if value))


def _executor_leg_rows(
    bundle: SecurityIdentityBundle,
    event_id: str,
    from_ticker: str,
    from_asset_id: str,
    resolution: pd.DataFrame,
    prices: pd.DataFrame,
    next_date: pd.Timestamp,
) -> list[dict[str, object]]:
    canonical = bundle.legs.loc[
        bundle.legs["Event_ID"].eq(event_id)
        & bundle.legs["From_Ticker"].eq(from_ticker)
    ].sort_values("Leg_Sequence", kind="stable")
    rows: list[dict[str, object]] = []
    for leg in canonical.to_dict("records"):
        to_ticker = str(leg["To_Ticker"])
        if leg["Leg_Type"] == "cvr":
            to_asset_id = to_ticker or f"RIGHT::{event_id}"
        elif to_ticker:
            to_asset_id = _successor_asset(
                bundle,
                resolution,
                prices,
                event_id,
                to_ticker,
                next_date,
            )
        else:
            to_asset_id = ""
        leg_type = str(leg["Leg_Type"])
        quantity = str(
            leg.get("Executable_Share_Ratio", "") or leg["Share_Ratio"]
        )
        cash_amount = str(leg["Cash_Amount"])
        rows.append({
            "Event_ID": event_id,
            "Leg_Order": leg["Leg_Sequence"],
            "From_Asset_ID": from_asset_id,
            "To_Asset_ID": to_asset_id,
            "Leg_Type": leg_type,
            "Quantity_Per_From_Share": quantity,
            "Cash_Per_From_Share": cash_amount,
            "Currency": leg["Currency"],
            "CVR_Units_Per_From_Share": leg["CVR_Units"],
            "CVR_Base_Value_Per_Unit": leg["CVR_Base_Value_Per_Unit"],
            "CVR_Max_Value_Per_Unit": leg["CVR_Max_Value_Per_Unit"],
            "Consumes_From_Position": not (
                str(leg["Retain_Predecessor"]).strip().lower() == "true"
            ),
            "Review_Status": leg["Review_Status"],
        })
    return rows


def _link_adjacent_documentary_events(
    audit: pd.DataFrame,
    bundle: SecurityIdentityBundle,
    source_urls: dict[str, str],
) -> pd.DataFrame:
    """Link a documentary event to its adjacent executable accounting event.

    Some transactions were recorded twice in the migrated evidence: an
    issuer-level legal reorganization followed one day later by the exact
    holder consideration.  The documentary row must not be treated as a
    second economic event or accepted through a generic price-ratio fallback.
    Instead, this routine identifies the one approved structured event for the
    same prepared predecessor within the same holding period and records that
    explicit execution link in the crossing audit.
    """

    if audit.empty:
        return audit
    event_status = bundle.events.set_index("Event_ID")["Accounting_Status"]
    documentary_ids = set(
        event_status.loc[event_status.eq("documented_not_executable")].index
    )
    explicit = audit.loc[
        audit["Potential_Holding_Crossing"].map(bool)
        & audit["Accounting_Treatment"].eq("explicit_structured_action")
        & audit["Executing_Event_ID"].ne("")
    ]
    candidates = audit.index[
        audit["Potential_Holding_Crossing"].map(bool)
        & audit["Event_ID"].isin(documentary_ids)
        & audit["Accounting_Treatment"].eq("verified_continuous_price_ratio")
    ]
    for index in candidates:
        row = audit.loc[index]
        event_date = pd.Timestamp(row["Effective_Date"])
        matches = explicit.loc[
            explicit["From_Asset_ID"].eq(row["From_Asset_ID"])
            & explicit["Previous_Decision_Date"].eq(row["Previous_Decision_Date"])
            & explicit["Next_Valuation_Date"].eq(row["Next_Valuation_Date"])
            & pd.to_datetime(explicit["Effective_Date"]).gt(event_date)
            & pd.to_datetime(explicit["Effective_Date"]).le(
                event_date + pd.Timedelta(days=7)
            )
        ]
        if matches.empty:
            continue
        executing_ids = tuple(matches["Executing_Event_ID"].drop_duplicates())
        if len(executing_ids) != 1:
            raise ValueError(
                f"Documentary event {row['Event_ID']}/{row['From_Asset_ID']} "
                f"has ambiguous adjacent accounting events: {executing_ids}"
            )
        executing_id = str(executing_ids[0])
        audit.at[index, "Accounting_Treatment"] = "explicit_structured_action"
        audit.at[index, "Executing_Event_ID"] = executing_id
        audit.at[index, "Source_URLs"] = ";".join(sorted({
            value
            for value in (
                str(row["Source_URLs"]),
                str(source_urls.get(executing_id, "")),
            )
            for value in value.split(";")
            if value
        }))
        audit.at[index, "Notes"] = (
            f"Documentary event {row['Event_ID']} is not executed separately; "
            f"approved structured event {executing_id} supplies the holder "
            "consideration for this same crossing."
        )
    return audit


def prepare_backtest_security_events(
    market_config: BacktestMarketConfig = DEFAULT_CONFIG.market,
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic, Asset_ID-resolved action inputs and crossing audit."""

    bundle = load_security_identity_bundle(paths.project_root)
    resolution = pd.read_csv(paths.ticker_ric_resolution_csv, keep_default_na=False)
    resolution["Date"] = pd.to_datetime(resolution["Date"], errors="raise")
    resolution["Is_Priced"] = _as_bool(resolution["Is_Priced"])
    monthly = pd.read_csv(paths.prices_monthly_csv)
    monthly["Date"] = pd.to_datetime(monthly["Date"], errors="raise")
    prices = monthly.pivot(index="Date", columns="Asset_ID", values="Price_Close")
    dates = month_end_index(market_config.start_date, market_config.end_date)

    source_urls = {
        event_id: ";".join(sorted(set(group["Source_URL"].astype(str))))
        for event_id, group in bundle.sources.groupby("Event_ID", sort=True)
    }
    prepared_event_ids: set[str] = set()
    prepared_leg_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for event in bundle.events.sort_values(
        ["Effective_Date", "Event_ID"], kind="stable"
    ).to_dict("records"):
        event_id = event["Event_ID"]
        effective = pd.Timestamp(event["Effective_Date"])
        previous = dates[dates < effective]
        following = dates[dates >= effective]
        if not len(previous) or not len(following):
            continue
        previous_date = pd.Timestamp(previous[-1])
        next_date = pd.Timestamp(following[0])
        for from_ticker in _from_tickers_for_event(bundle, event_id):
            predecessor_assets = _predecessor_assets(
                bundle,
                resolution,
                prices,
                event_id,
                from_ticker,
                previous_date,
            )
            if not predecessor_assets:
                audit_rows.append({
                    "Event_ID": event_id,
                    "Effective_Date": effective,
                    "From_Ticker": from_ticker,
                    "From_Asset_ID": "",
                    "Previous_Decision_Date": previous_date,
                    "Next_Valuation_Date": next_date,
                    "Potential_Holding_Crossing": False,
                    "Accounting_Treatment": "not_holding_relevant",
                    "Executing_Event_ID": "",
                    "End_Valuation_Status": "not_required",
                    "Validation_Status": "approved",
                    "Source_URLs": source_urls.get(event_id, ""),
                    "Notes": "Predecessor was not a priced member at the prior decision date.",
                })
                continue

            for from_asset_id in predecessor_assets:
                active_assets = set(
                    resolution.loc[
                        resolution["Date"].eq(previous_date)
                        & resolution["Is_Priced"],
                        "Asset_ID",
                    ].astype(str)
                )
                is_active = from_asset_id in active_assets
                has_start = (
                    finite_positive_price(
                        prices,
                        previous_date,
                        from_asset_id,
                    )
                    is not None
                )
                potential = bool(is_active and has_start)
                canonical_legs = bundle.legs.loc[
                    bundle.legs["Event_ID"].eq(event_id)
                    & bundle.legs["From_Ticker"].eq(from_ticker)
                ]
                executable = (
                    event["Accounting_Status"] == "executable"
                    and not canonical_legs.empty
                )
                leg_rows = (
                    _executor_leg_rows(
                        bundle,
                        event_id,
                        from_ticker,
                        from_asset_id,
                        resolution,
                        prices,
                        next_date,
                    )
                    if executable
                    else []
                )
                continuous_policy_notes = (
                    _verified_continuous_distribution_policy(
                        bundle,
                        event,
                        from_ticker,
                        from_asset_id,
                        leg_rows,
                    )
                )
                creates_distinct_economics = any(
                    row["Leg_Type"] not in {"relabel"}
                    or row["To_Asset_ID"] != from_asset_id
                    for row in leg_rows
                )
                if potential and continuous_policy_notes is not None:
                    if (
                        finite_positive_price(
                            prices,
                            next_date,
                            from_asset_id,
                        )
                        is None
                    ):
                        raise ValueError(
                            f"Reviewed continuous-price event "
                            f"{event_id}/{from_asset_id} lacks its required "
                            "end valuation"
                        )
                    treatment = "verified_continuous_price_ratio"
                    end_status = "complete"
                    notes = continuous_policy_notes
                    executing_event_id = ""
                elif potential and executable and creates_distinct_economics:
                    treatment = "explicit_structured_action"
                    prepared_event_ids.add(event_id)
                    prepared_leg_rows.extend(leg_rows)
                    missing_successors = sorted({
                        str(row["To_Asset_ID"])
                        for row in leg_rows
                        if row["Leg_Type"]
                        in {
                            "relabel",
                            "stock",
                            "distribution",
                        }
                        and finite_positive_price(
                            prices,
                            next_date,
                            str(row["To_Asset_ID"]),
                        )
                        is None
                    })
                    end_status = (
                        "complete"
                        if not missing_successors
                        else "candidate_excluded_missing_successor_price:"
                        + ";".join(missing_successors)
                    )
                    notes = "Approved structured terms replace vendor price-ratio treatment."
                    executing_event_id = event_id
                elif potential:
                    if (
                        finite_positive_price(
                            prices,
                            next_date,
                            from_asset_id,
                        )
                        is None
                    ):
                        raise ValueError(
                            f"Holding-relevant event {event_id}/{from_asset_id} has "
                            "neither executable structured terms nor a continuous "
                            "end valuation"
                        )
                    treatment = "verified_continuous_price_ratio"
                    end_status = "complete"
                    notes = (
                        "The approved Reuters mapping retains one locally observed "
                        "Asset_ID through the event-period valuation boundary."
                    )
                    executing_event_id = ""
                else:
                    treatment = "not_holding_relevant"
                    end_status = "not_required"
                    notes = "Predecessor was not eligible to be held across the event."
                    executing_event_id = ""
                audit_rows.append({
                    "Event_ID": event_id,
                    "Effective_Date": effective,
                    "From_Ticker": from_ticker,
                    "From_Asset_ID": from_asset_id,
                    "Previous_Decision_Date": previous_date,
                    "Next_Valuation_Date": next_date,
                    "Potential_Holding_Crossing": potential,
                    "Accounting_Treatment": treatment,
                    "Executing_Event_ID": executing_event_id,
                    "End_Valuation_Status": end_status,
                    "Validation_Status": "approved",
                    "Source_URLs": source_urls.get(event_id, ""),
                    "Notes": notes,
                })

    # One canonical event may have several predecessor assets; its header and
    # source rows are stored once while every resolved predecessor keeps its
    # own ordered leg set.
    prepared_events = bundle.events.loc[
        bundle.events["Event_ID"].isin(prepared_event_ids),
        EVENT_COLUMNS,
    ].copy()
    prepared_events = prepared_events.sort_values(
        ["Effective_Date", "Event_ID"], kind="stable"
    ).reset_index(drop=True)
    prepared_legs = pd.DataFrame(prepared_leg_rows, columns=LEG_COLUMNS)
    if not prepared_legs.empty:
        prepared_legs = prepared_legs.sort_values(
            ["Event_ID", "From_Asset_ID", "Leg_Order"], kind="stable"
        ).reset_index(drop=True)
        # Leg order is event-global in the executor contract.  Canonical events
        # with multiple predecessor assets therefore receive deterministic
        # consecutive orders after scope resolution.
        prepared_legs["Leg_Order"] = prepared_legs.groupby(
            "Event_ID", sort=False
        ).cumcount() + 1
    prepared_sources = bundle.sources.loc[
        bundle.sources["Event_ID"].isin(prepared_event_ids),
        SOURCE_COLUMNS,
    ].drop_duplicates().sort_values(
        ["Event_ID", "Source_URL"], kind="stable"
    ).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS).sort_values(
        ["Effective_Date", "Event_ID", "From_Asset_ID"], kind="stable"
    ).reset_index(drop=True)
    audit = _link_adjacent_documentary_events(audit, bundle, source_urls)
    for frame, path in (
        (prepared_events, paths.security_events_prepared_csv),
        (prepared_legs, paths.security_event_legs_prepared_csv),
        (prepared_sources, paths.security_event_sources_prepared_csv),
        (audit, paths.security_event_crossing_audit_csv),
    ):
        atomic_write_dataframe(
            frame,
            path,
            index=False,
            date_format="%Y-%m-%d",
            float_format="%.17g",
            lineterminator="\n",
        )
    return prepared_events, prepared_legs, prepared_sources, audit


__all__ = [
    "AUDIT_COLUMNS",
    "prepare_backtest_security_events",
]
