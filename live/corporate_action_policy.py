"""Validation and loading of the reviewed live corporate-action policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_core.artifacts import (
    ArtifactOrigin,
    validate_exact_manifest_catalog,
)
from portfolio_core.corporate_actions import parse_explicit_boolean


CORPORATE_ACTION_COLUMNS = (
    "Event_ID",
    "Asset_ID",
    "Effective_Date",
    "Event_Type",
    "Continuity_Class",
    "Review_Status",
)

CORPORATE_ACTION_POLICY_COLUMNS = (
    "Event_ID",
    "Scope",
    "Apply_Accounting",
    "CVR_Base_Value_Per_Unit",
    "Valuation_As_Of_Date",
    "Valuation_Available_Date",
    "Valuation_Source_ID",
    "Review_Status",
    "Notes",
)


@dataclass(frozen=True, slots=True)
class LiveCorporateActionBundle:
    """Policy-selected canonical events resolved to live ``Asset_ID`` values."""

    events: pd.DataFrame
    legs: pd.DataFrame
    sources: pd.DataFrame
    policy: pd.DataFrame


def load_corporate_action_policy(paths) -> pd.DataFrame:
    """Validate the small live policy without duplicating event economics."""
    artifact = Path(paths.raw_corporate_action_policy_csv)
    validate_exact_manifest_catalog(
        Path(paths.raw_corporate_action_policy_manifest_csv),
        scope="live",
        dataset="corporate_action_policy",
        expected_origins={artifact.name: ArtifactOrigin.MANUAL},
        base_dir=artifact.parent,
    )
    frame = pd.read_csv(artifact, keep_default_na=False)
    if tuple(frame.columns) != CORPORATE_ACTION_POLICY_COLUMNS:
        raise ValueError("Corporate-action policy schema is inconsistent")
    for column in ("Event_ID", "Scope", "Review_Status"):
        frame[column] = frame[column].astype(str).str.strip()
    if frame["Event_ID"].eq("").any() or frame["Event_ID"].duplicated().any():
        raise ValueError("Corporate-action policy requires unique nonblank Event_IDs")
    if not frame["Scope"].eq("live").all():
        raise ValueError("Corporate-action policy contains a non-live scope")
    if not frame["Review_Status"].eq("approved").all():
        raise ValueError("Corporate-action policy contains an unapproved row")
    frame["Apply_Accounting"] = [
        parse_explicit_boolean(value, f"{event_id} Apply_Accounting")
        for value, event_id in zip(
            frame["Apply_Accounting"], frame["Event_ID"], strict=True
        )
    ]
    frame["CVR_Base_Value_Per_Unit"] = pd.to_numeric(
        frame["CVR_Base_Value_Per_Unit"], errors="raise"
    ).astype(float)
    base_values = frame["CVR_Base_Value_Per_Unit"].to_numpy(dtype=float)
    if not np.isfinite(base_values).all() or (base_values < 0.0).any():
        raise ValueError(
            "Corporate-action policy CVR base values must be finite and nonnegative"
        )
    _validate_valuation_dates(frame)
    return frame.sort_values("Event_ID", kind="stable").reset_index(drop=True)


def _validate_valuation_dates(frame: pd.DataFrame) -> None:
    """A positive non-traded mark needs its own date and publication evidence."""
    for column in ("Valuation_As_Of_Date", "Valuation_Available_Date"):
        parsed = pd.to_datetime(frame[column].replace("", pd.NaT), errors="raise")
        if parsed.dt.tz is not None or not parsed.dropna().eq(parsed.dropna().dt.normalize()).all():
            raise ValueError("Corporate-action valuations require date-only timestamps")
    positive = pd.to_numeric(frame.CVR_Base_Value_Per_Unit).gt(0)
    if frame.loc[positive, ["Valuation_As_Of_Date", "Valuation_Available_Date", "Valuation_Source_ID"]].eq("").any().any():
        raise ValueError("Nonzero CVR marks require dated valuation and publication provenance")
    if not pd.to_datetime(frame.loc[positive, "Valuation_Available_Date"]).ge(
        pd.to_datetime(frame.loc[positive, "Valuation_As_Of_Date"])
    ).all():
        raise ValueError("CVR valuation publication precedes its as-of date")


def load_live_corporate_action_bundle(paths) -> LiveCorporateActionBundle:
    """Join approved live policy to canonical event facts and resolve ticker IDs."""
    from portfolio_core.corporate_actions import EVENT_COLUMNS, LEG_COLUMNS, SOURCE_COLUMNS
    from portfolio_core.security_identity import load_security_identity_bundle

    policy = load_corporate_action_policy(paths)
    selected_policy = policy.loc[policy["Apply_Accounting"]].copy()
    canonical = load_security_identity_bundle(
        Path(paths.security_identity.directory),
        validate_manifest=True,
        require_all_approved=True,
    )
    event_ids = set(selected_policy["Event_ID"])
    available_ids = set(canonical.events["Event_ID"].astype(str))
    missing = sorted(event_ids - available_ids)
    if missing:
        raise ValueError(
            f"Corporate-action policy references unknown canonical events: {missing}"
        )

    events = canonical.events.loc[canonical.events["Event_ID"].isin(event_ids)].copy()
    if not events["Accounting_Status"].eq("executable").all():
        blocked = sorted(
            events.loc[
                ~events["Accounting_Status"].eq("executable"), "Event_ID"
            ].astype(str)
        )
        raise ValueError(
            "Live accounting policy selected non-executable canonical events: "
            f"{blocked}"
        )
    events = events.loc[:, EVENT_COLUMNS].copy()

    raw_legs = canonical.legs.loc[canonical.legs["Event_ID"].isin(event_ids)].copy()
    for row in selected_policy.loc[selected_policy.CVR_Base_Value_Per_Unit.gt(0)].itertuples():
        source = canonical.sources.loc[canonical.sources.Source_ID.eq(row.Valuation_Source_ID)]
        if len(source) != 1 or source.iloc[0].Event_ID != row.Event_ID:
            raise ValueError(f"Unknown CVR valuation source for {row.Event_ID}")
    policy_base = selected_policy.set_index("Event_ID")["CVR_Base_Value_Per_Unit"]
    legs = pd.DataFrame(
        {
            "Event_ID": raw_legs["Event_ID"].astype(str),
            "Leg_Order": raw_legs["Leg_Sequence"],
            "From_Asset_ID": raw_legs["From_Ticker"].astype(str).str.strip(),
            "To_Asset_ID": raw_legs["To_Ticker"].astype(str).str.strip(),
            "Leg_Type": raw_legs["Leg_Type"],
            "Quantity_Per_From_Share": raw_legs["Share_Ratio"],
            "Cash_Per_From_Share": raw_legs["Cash_Amount"],
            "Currency": raw_legs["Currency"],
            "CVR_Units_Per_From_Share": raw_legs["CVR_Units"],
            "CVR_Base_Value_Per_Unit": raw_legs["Event_ID"].map(policy_base).where(
                raw_legs["Leg_Type"].eq("cvr"), 0.0,
            ),
            "CVR_Max_Value_Per_Unit": raw_legs["CVR_Max_Value_Per_Unit"],
            "Consumes_From_Position": [
                not parse_explicit_boolean(
                    value, f"{event_id} Retain_Predecessor"
                )
                for value, event_id in zip(
                    raw_legs["Retain_Predecessor"],
                    raw_legs["Event_ID"],
                    strict=True,
                )
            ],
            "Review_Status": raw_legs["Review_Status"],
        }
    ).loc[:, LEG_COLUMNS]
    if legs["From_Asset_ID"].eq("").any():
        raise ValueError("Canonical live action legs contain a blank predecessor ticker")

    sources = canonical.sources.loc[
        canonical.sources["Event_ID"].isin(event_ids), SOURCE_COLUMNS
    ].copy()
    return LiveCorporateActionBundle(
        events=events.sort_values(
            ["Effective_Date", "Event_ID"], kind="stable"
        ).reset_index(drop=True),
        legs=legs.sort_values(["Event_ID", "Leg_Order"], kind="stable").reset_index(
            drop=True
        ),
        sources=sources.sort_values(
            ["Event_ID", "Source_URL"], kind="stable"
        ).reset_index(drop=True),
        policy=selected_policy.sort_values("Event_ID", kind="stable").reset_index(
            drop=True
        ),
    )


def load_corporate_actions(paths) -> pd.DataFrame:
    """Return the event-level view needed by composite price readiness."""
    bundle = load_live_corporate_action_bundle(paths)
    event_index = bundle.events.set_index("Event_ID")
    records: list[dict[str, object]] = []
    predecessors = bundle.legs[["Event_ID", "From_Asset_ID"]].drop_duplicates()
    for row in predecessors.itertuples(index=False):
        event = event_index.loc[str(row.Event_ID)]
        records.append(
            {
                "Event_ID": str(row.Event_ID),
                "Asset_ID": str(row.From_Asset_ID),
                "Effective_Date": pd.Timestamp(event["Effective_Date"]),
                "Event_Type": str(event["Event_Type"]),
                "Continuity_Class": str(event["Continuity_Class"]),
                "Review_Status": str(event["Review_Status"]),
            }
        )
    return pd.DataFrame(records, columns=CORPORATE_ACTION_COLUMNS).sort_values(
        ["Effective_Date", "Asset_ID", "Event_ID"], kind="stable"
    ).reset_index(drop=True)


__all__ = [
    "CORPORATE_ACTION_COLUMNS",
    "CORPORATE_ACTION_POLICY_COLUMNS",
    "LiveCorporateActionBundle",
    "load_corporate_action_policy",
    "load_corporate_actions",
    "load_live_corporate_action_bundle",
]
