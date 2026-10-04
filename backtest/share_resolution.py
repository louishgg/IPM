"""Reviewed retrospective shares for Brinson; never trading-signal inputs.

Reported counts precede event estimates. Yahoo is a residual source only when
one distinct response value agrees with a qualified prior reference. Publication
can follow observation: all age limits use factual observation dates.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_core.artifacts import file_sha256
from portfolio_core.shares import (
    CanonicalShareIdentity, RAW_IDENTITY_COLUMNS, validate_raw_shares,
)

MAX_AGE_DAYS = 120
REFERENCE_AGE_DAYS = 365
YAHOO_TOLERANCE = 0.05
PAIR_COLUMNS = ["Date", "Asset_ID"]
RESULT_COLUMNS = [
    *PAIR_COLUMNS, "Shares_Outstanding", "Source", "Observation_ID",
    "Observation_Date", "Filed", "Age_Days", "Share_Factor", "Applied_Events",
    "Capitalization_Factor", "Capitalization_Interval_ID", "Provider_Symbol",
    "Effective_Start", "Effective_End", "Observation_Sequences",
    "Reference_ID", "Reference_Date", "Reference_Age_Days", "Relative_Gap",
    "Source_URLs",
]


def required_pairs(membership: pd.DataFrame) -> pd.DataFrame:
    """Project the existing active-membership mask without another worklist."""
    rows = [(date, str(asset)) for date, row in membership.iterrows()
            for asset in row.index[row]]
    return pd.DataFrame(rows, columns=PAIR_COLUMNS).sort_values(
        PAIR_COLUMNS, ignore_index=True
    )


def pairs_digest(pairs: pd.DataFrame) -> str:
    frame = pairs[PAIR_COLUMNS].sort_values(PAIR_COLUMNS).copy()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def _age(start: str, end: str) -> int:
    return (pd.Timestamp(end) - pd.Timestamp(start)).days


def _positive(values: pd.Series, label: str) -> None:
    numbers = pd.to_numeric(values, errors="raise")
    if (~np.isfinite(numbers) | numbers.le(0)).any():
        raise ValueError(f"{label} must be finite and positive")


def _dates(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        values = frame.loc[frame[column].ne(""), column]
        parsed = pd.to_datetime(values, format="%Y-%m-%d", errors="raise")
        if not parsed.dt.strftime("%Y-%m-%d").equals(values):
            raise ValueError(f"Invalid ISO dates in {column}")


def _intervals(frame: pd.DataFrame, *, inclusive: bool) -> None:
    for _, group in frame.sort_values("Effective_Start").groupby("Asset_ID"):
        previous = None
        for row in group.itertuples():
            if row.Effective_Start > row.Effective_End or (
                not inclusive and row.Effective_Start == row.Effective_End
            ):
                raise ValueError("Invalid reviewed interval")
            if previous is not None and (
                row.Effective_Start <= previous if inclusive else row.Effective_Start < previous
            ):
                raise ValueError("Overlapping reviewed intervals")
            previous = row.Effective_End


@dataclass(frozen=True)
class ReviewedShares:
    observations: pd.DataFrame
    issuers: pd.DataFrame
    events: pd.DataFrame
    factors: pd.DataFrame
    review: dict

    @classmethod
    def load(cls, directory: Path) -> ReviewedShares:
        """Validate maintained decisions, including blocked canonical records."""
        def read(name: str, required: list[str]) -> pd.DataFrame:
            frame = pd.read_csv(directory / name, dtype=str, keep_default_na=False)
            if not set(required).issubset(frame.columns):
                raise ValueError(f"Invalid reviewed schema: {name}")
            return frame

        obs = read("observations.csv", [
            "Observation_ID", "Asset_ID", "CIK", "Observation_Date", "Shares",
            "Filed", "Kind", "Status", "Post_Event", "Base_Observation_Date",
            "Source_URL", "Class_Tier", "Review_Notes", "Base_Shares", "Issued_Shares",
        ])
        issuers = read("issuer_intervals.csv", [
            "Asset_ID", "Effective_Start", "Effective_End", "Expected_CIK", "Source_URLs",
        ])
        events = read("events.csv", [
            "Asset_ID", "Event_ID", "Effective_Date", "Estimator_Action",
            "Estimator_Share_Multiplier", "Share_Continuity_Effect", "CIK_Before",
            "CIK_After", "Earliest_Effective_Date", "Latest_Effective_Date", "Source_URLs", "Review_Claim",
        ])
        factors = read("capitalization_price_factors.csv", [
            "Interval_ID", "Asset_ID", "Effective_Start", "Effective_End", "Factor",
            "Source_URLs", "Review_Notes",
        ])
        for frame, key in [(obs, ["Asset_ID", "Observation_Date", "Kind"]),
                           (obs, ["Observation_ID"]), (events, ["Asset_ID", "Event_ID"]),
                           (factors, ["Interval_ID"])]:
            if frame.duplicated(key).any():
                raise ValueError(f"Duplicate reviewed key: {key}")
        if not obs.Kind.isin(["sec", "event_estimate"]).all():
            raise ValueError("Unknown observation kind")
        if not obs.Status.isin(["selected", "conflicting_values", "blocked"]).all():
            raise ValueError("Unknown observation status")
        if not obs.Post_Event.isin(["True", "False"]).all():
            raise ValueError("Post_Event must be True or False")
        if obs[["Observation_ID", "Asset_ID", "CIK", "Observation_Date", "Filed",
                "Source_URL", "Class_Tier", "Review_Notes"]].eq("").any().any():
            raise ValueError("Observations require dates, identity, provenance and review notes")
        _positive(obs.loc[obs.Status.eq("selected"), "Shares"], "Selected shares")
        _positive(factors.Factor, "Capitalization factors")
        for frame, columns in [
            (obs, ["Observation_Date", "Filed", "Base_Observation_Date"]),
            (issuers, ["Effective_Start", "Effective_End"]),
            (events, ["Effective_Date", "Earliest_Effective_Date", "Latest_Effective_Date"]),
            (factors, ["Effective_Start", "Effective_End"]),
        ]:
            _dates(frame, columns)
        if (obs.Filed < obs.Observation_Date).any():
            raise ValueError("Publication precedes observation")
        if (obs.Base_Observation_Date.ne("") & obs.Base_Observation_Date.gt(obs.Observation_Date)).any():
            raise ValueError("Event estimate base follows its observation")
        for frame, columns in [
            (issuers, ["Asset_ID", "Effective_Start", "Effective_End", "Expected_CIK", "Source_URLs"]),
            (events, ["Asset_ID", "Event_ID", "Source_URLs", "Review_Claim"]),
            (factors, ["Asset_ID", "Effective_Start", "Effective_End", "Source_URLs", "Review_Notes"]),
        ]:
            if frame[columns].eq("").any().any():
                raise ValueError("Reviewed intervals/events require identity, boundaries and evidence")
        if not obs.CIK.str.fullmatch(r"\d{10}").all() or not issuers.Expected_CIK.str.fullmatch(r"\d{10}").all():
            raise ValueError("Reviewed issuer CIKs must contain ten digits")
        uncertain = events.Estimator_Action.eq("uncertain")
        if events.loc[~uncertain, "Effective_Date"].eq("").any():
            raise ValueError("Events require an effective date")
        uncertainty = events.loc[uncertain]
        if (uncertainty[["Earliest_Effective_Date", "Latest_Effective_Date"]].eq("").any().any()
                or uncertainty.Earliest_Effective_Date.gt(uncertainty.Latest_Effective_Date).any()):
            raise ValueError("Invalid uncertain event boundaries")
        additions = obs.loc[obs.Kind.eq("event_estimate") & obs.Base_Observation_Date.ne("")]
        for column in ("Base_Shares", "Issued_Shares"):
            _positive(additions[column], column)
        if not np.allclose(pd.to_numeric(additions.Shares),
                           pd.to_numeric(additions.Base_Shares) + pd.to_numeric(additions.Issued_Shares),
                           rtol=1e-12, atol=0):
            raise ValueError("Event estimate does not reconcile to base plus issuance")
        _intervals(issuers, inclusive=False)
        _intervals(factors, inclusive=True)
        if not events.Estimator_Action.isin([
            "multiply", "new_observation_required", "continue_parent", "uncertain",
            "continue", "continue_existing_listed_class",
        ]).all():
            raise ValueError("Unknown event action")
        _positive(events.loc[events.Estimator_Action.eq("multiply"), "Estimator_Share_Multiplier"], "Split factors")
        split_values = events.loc[events.Estimator_Action.eq("multiply")].copy()
        split_values["Estimator_Share_Multiplier"] = pd.to_numeric(split_values.Estimator_Share_Multiplier)
        if split_values.groupby(["Asset_ID", "Effective_Date"]).Estimator_Share_Multiplier.nunique().gt(1).any():
            raise ValueError("Conflicting legal split multipliers")
        review = json.loads((directory / "review.json").read_text())
        expected = {"schema_version", "reviewed_start_date", "reviewed_end_date",
                    "required_pairs_sha256", "prepared_prices_sha256", "price_basis_sha256"}
        if set(review) != expected or review["schema_version"] != 1:
            raise ValueError("Invalid shares review metadata")
        return cls(obs, issuers, events, factors, review)

    def validate_inputs(self, pairs: pd.DataFrame, prices: Path, basis: Path) -> None:
        """Bind reviewed capitalization factors to the fixed price dataset."""
        dates = pd.to_datetime(pairs.Date).dt.strftime("%Y-%m-%d")
        if (dates.min() != self.review["reviewed_start_date"]
                or dates.max() != self.review["reviewed_end_date"]
                or pairs_digest(pairs) != self.review["required_pairs_sha256"]):
            raise ValueError("Shares requirements differ from the reviewed population")
        for path, key in [(prices, "prepared_prices_sha256"), (basis, "price_basis_sha256")]:
            if file_sha256(path) != self.review[key]:
                raise ValueError("Price inputs changed; review capitalization factors before preparation")


class ShareResolver:
    """One deterministic selector shared by readiness and preparation."""

    def __init__(self, reviewed: ReviewedShares):
        self.reviewed = reviewed
        self.observations = {a: g.to_dict("records") for a, g in reviewed.observations.groupby("Asset_ID")}
        self.issuers = {a: g.to_dict("records") for a, g in reviewed.issuers.groupby("Asset_ID")}
        self.events = {a: g.sort_values("Effective_Date").to_dict("records") for a, g in reviewed.events.groupby("Asset_ID")}
        self.factors = {a: g.to_dict("records") for a, g in reviewed.factors.groupby("Asset_ID")}

    def issuer(self, asset: str, day: str) -> str | None:
        matches = [r["Expected_CIK"] for r in self.issuers.get(asset, [])
                   if r["Effective_Start"] <= day < r["Effective_End"]]
        return matches[0] if len(matches) == 1 else None

    def issuer_carry(self, asset: str, cik: str, observed: str, target: str) -> bool:
        if self.issuer(asset, observed) != cik:
            return False
        reachable = {cik}
        for event in self.events.get(asset, []):
            if (observed < event["Effective_Date"] <= target
                    and event["Share_Continuity_Effect"] == "documented_same_share_unit"
                    and event["CIK_Before"] in reachable and event["CIK_After"]):
                reachable.add(event["CIK_After"])
        return self.issuer(asset, target) in reachable

    def carry(self, asset: str, observed: str, target: str, post_event: bool = False) -> tuple[float, str, str]:
        factor, applied, blocked, seen = 1.0, [], [], set()
        for event in self.events.get(asset, []):
            action, day = event["Estimator_Action"], event["Effective_Date"]
            if action == "uncertain":
                if observed < event["Latest_Effective_Date"] and target >= event["Earliest_Effective_Date"]:
                    blocked.append(event["Event_ID"])
                continue
            if not observed <= day <= target:
                continue
            if action == "new_observation_required":
                if day != observed or not post_event:
                    blocked.append(event["Event_ID"])
            elif day > observed:
                if action == "multiply":
                    key = (day, float(event["Estimator_Share_Multiplier"]))
                    if key in seen:
                        continue
                    seen.add(key)
                    factor *= key[1]
                applied.append(event["Event_ID"])
        return factor, ";".join(sorted(set(applied))), ";".join(sorted(set(blocked)))

    @lru_cache(maxsize=65536)
    def primary(self, asset: str, target: str, kind: str, max_age: int = MAX_AGE_DAYS) -> tuple[dict | None, str]:
        """Return the latest eligible SEC/event estimate, or None and a rejection reason.

        Age uses observation dates; event estimates also retain the base count's
        age limit. Rejecting the latest observation never revives an older one
        within the same source.
        """
        eligible = [r for r in self.observations.get(asset, [])
                    if r["Kind"] == kind and r["Observation_Date"] <= target
                    and self.issuer_carry(asset, r["CIK"], r["Observation_Date"], target)]
        if not eligible:
            return None, "no_observation"
        row = max(eligible, key=lambda r: r["Observation_Date"])
        if row["Status"] != "selected":
            return None, row["Status"]
        age = _age(row["Observation_Date"], target)
        if age > max_age:
            return None, "stale_observation"
        base = row["Base_Observation_Date"]
        if kind == "event_estimate" and base and _age(base, target) > MAX_AGE_DAYS:
            return None, "stale_event_base"
        factor, applied, blocked = self.carry(asset, row["Observation_Date"], target, row["Post_Event"] == "True")
        if blocked:
            return None, "event_block:" + blocked
        return {
            "Shares_Outstanding": float(row["Shares"]) * factor, "Source": kind,
            "Observation_ID": row["Observation_ID"], "Observation_Date": row["Observation_Date"],
            "Filed": row["Filed"], "Age_Days": age, "Share_Factor": factor,
            "Applied_Events": applied, "Source_URLs": ";".join(dict.fromkeys(
                (row["Source_URL"] + ";" + row.get("Class_Evidence_URLs", "")).strip(";").split(";")
            )),
            "Post_Event": row["Post_Event"] == "True", "CIK": row["CIK"],
        }, ""

    def yahoo(self, asset: str, target: str, rows: list[dict]) -> tuple[dict | None, str]:
        """Return a qualified latest Yahoo value, or None and a rejection reason.

        Require one identity and one distinct value within tolerance of a prior
        SEC/event reference, subject to age and event-carry checks. Rejected
        latest data never falls back to older Yahoo observations.
        """
        eligible = [r for r in rows if r["Date"] <= target
                    and (not r["Effective_Start"] or r["Effective_Start"] <= target)
                    and (not r["Effective_End"] or target <= r["Effective_End"])]
        if not eligible:
            return None, "no_yahoo_observation"
        day = max(r["Date"] for r in eligible)
        latest = [r for r in eligible if r["Date"] == day]
        if len({tuple(r[c] for c in RAW_IDENTITY_COLUMNS) for r in eligible}) != 1:
            return None, "overlapping_yahoo_identities"
        if _age(day, target) > MAX_AGE_DAYS:
            return None, "stale_yahoo"
        # Qualify the reference at the Yahoo observation date, before carrying it.
        anchor = None
        for kind in ("sec", "event_estimate"):
            anchor, _ = self.primary(asset, day, kind, REFERENCE_AGE_DAYS)
            if anchor is not None:
                break
        if anchor is None:
            return None, "no_qualified_reference"
        if not self.issuer_carry(asset, anchor["CIK"], anchor["Observation_Date"], target):
            return None, "issuer_change"
        factor, applied, blocked = self.carry(asset, day, target, anchor["Post_Event"] and anchor["Observation_Date"] == day)
        if blocked:
            return None, "event_block:" + blocked
        reference = anchor["Shares_Outstanding"]
        matches = [r for r in latest if abs(r["Shares_Outstanding"] - reference) <= YAHOO_TOLERANCE * reference]
        values = {r["Shares_Outstanding"] for r in matches}
        if len(values) != 1:
            return None, f"compatible_values:{len(values)}"
        row, value = matches[0], next(iter(values))
        return {
            "Shares_Outstanding": value * factor, "Source": "yahoo",
            "Observation_ID": "|".join([asset, row["Provider_Symbol"], day]),
            "Observation_Date": day, "Filed": "", "Age_Days": _age(day, target),
            "Share_Factor": factor, "Applied_Events": applied,
            **{c: row[c] for c in RAW_IDENTITY_COLUMNS if c != "Asset_ID"},
            "Observation_Sequences": ";".join(str(r["Observation_Sequence"]) for r in matches),
            "Reference_ID": anchor["Observation_ID"], "Reference_Date": anchor["Observation_Date"],
            "Reference_Age_Days": anchor["Age_Days"], "Relative_Gap": abs(value / reference - 1),
            "Source_URLs": anchor["Source_URLs"] + ";https://finance.yahoo.com/quote/" + row["Provider_Symbol"],
        }, ""

    def resolve(self, pairs: pd.DataFrame, raw: pd.DataFrame | None = None,
                identities: tuple[CanonicalShareIdentity, ...] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return selected/unresolved frames using SEC, event estimates, then Yahoo.

        Yahoo is restricted to the supplied identities. Missing coverage produces
        unresolved rows with reasons; invalid requirements or resolved values raise.
        """
        if pairs.empty or pairs.duplicated(PAIR_COLUMNS).any() or pairs[PAIR_COLUMNS].isna().any().any():
            raise ValueError("Requirements must contain unique, nonempty security/month pairs")
        yahoo = {}
        if raw is not None and identities:
            raw = validate_raw_shares(raw)
            keys = {identity.raw_key for identity in identities}
            raw = raw.loc[raw[list(RAW_IDENTITY_COLUMNS)].apply(tuple, axis=1).isin(keys)].copy()
            raw["Date"] = raw.Date.dt.strftime("%Y-%m-%d")
            yahoo = {a: g.to_dict("records") for a, g in raw.groupby("Asset_ID")}
        selected, unresolved = [], []
        for pair in pairs.itertuples(index=False):
            asset, target = str(pair.Asset_ID), pd.Timestamp(pair.Date).strftime("%Y-%m-%d")
            choice, reasons = None, []
            for kind in ("sec", "event_estimate"):
                choice, reason = self.primary(asset, target, kind)
                if choice is not None:
                    break
                reasons.append(kind + ":" + reason)
            if choice is None and raw is not None:
                choice, reason = self.yahoo(asset, target, yahoo.get(asset, []))
                reasons.append("yahoo:" + reason)
            if choice is None:
                unresolved.append({"Date": pd.Timestamp(target), "Asset_ID": asset, "Reason": ";".join(reasons)})
                continue
            intervals = [r for r in self.factors.get(asset, []) if r["Effective_Start"] <= target <= r["Effective_End"]]
            interval = intervals[0] if intervals else None
            if not np.isfinite(choice["Shares_Outstanding"]) or choice["Shares_Outstanding"] <= 0:
                raise ValueError("Resolved shares must be finite and positive")
            choice = choice.copy()
            if interval is not None:
                choice["Source_URLs"] += ";" + interval["Source_URLs"]
            selected.append({**dict.fromkeys(RESULT_COLUMNS, ""), **choice,
                             "Date": pd.Timestamp(target), "Asset_ID": asset,
                             "Capitalization_Factor": float(interval["Factor"]) if interval else 1.0,
                             "Capitalization_Interval_ID": interval["Interval_ID"] if interval else "nominal"})
        return (pd.DataFrame(selected, columns=RESULT_COLUMNS).sort_values(PAIR_COLUMNS, ignore_index=True),
                pd.DataFrame(unresolved, columns=[*PAIR_COLUMNS, "Reason"]))


def validate_prepared_shares(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != RESULT_COLUMNS:
        raise ValueError("Invalid shares schema; rebuild with `python -m backtest.prepare brinson`")
    result = frame.copy()
    result["Date"] = pd.to_datetime(result.Date, errors="raise")
    if result.empty or result.duplicated(PAIR_COLUMNS).any():
        raise ValueError("Prepared shares must have one row per security/month")
    if not result[PAIR_COLUMNS].equals(result.sort_values(PAIR_COLUMNS)[PAIR_COLUMNS]):
        raise ValueError("Prepared shares are not sorted")
    for column in ("Shares_Outstanding", "Share_Factor", "Capitalization_Factor"):
        _positive(result[column], column)
        result[column] = pd.to_numeric(result[column])
    if not result.Source.isin(["sec", "event_estimate", "yahoo"]).all():
        raise ValueError("Unknown prepared share source")
    dates = pd.to_datetime(result.Observation_Date, errors="raise")
    age = (result.Date - dates).dt.days
    if (age.lt(0) | age.gt(MAX_AGE_DAYS)).any() or not np.array_equal(age, pd.to_numeric(result.Age_Days)):
        raise ValueError("Invalid prepared observation ages")
    return result
