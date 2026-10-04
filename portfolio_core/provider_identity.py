"""Canonical backtest-provider and purpose-specific Yahoo identity resolution.

Operational callers consume a validated :class:`SecurityIdentityBundle`
directly; legacy-ledger reconstruction is not part of the runtime API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import pandas as pd

from .security_identity import SecurityIdentityBundle


YahooPurpose = Literal["historical_prices", "effective_security"]
IDENTITY_VALIDATION_COLUMNS = (
    "Mapping_ID",
    "Scope",
    "Provider",
    "Source_Ticker",
    "Provider_Symbol",
    "Asset_ID",
    "Effective_Start",
    "Effective_End",
    "Event_ID",
    "Primary_Source_ID",
    "Observed_First_Date",
    "Observed_Last_Date",
    "Validation_Status",
)
CONTRIBUTED_PRICE_RESOLUTION_METHODS = frozenset({
    "reviewed_yahoo_close_fallback",
    "reviewed_wiki_close_fallback",
})


class ProviderIdentityError(ValueError):
    """A provider identity cannot be resolved without an unsupported guess."""


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    """One canonical provider mapping with a start-inclusive/end-exclusive interval."""

    mapping_id: str
    source_ticker: str
    provider_symbol: str
    asset_id: str
    effective_start: pd.Timestamp | None
    effective_end: pd.Timestamp | None
    resolution_method: str
    event_id: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "ProviderIdentity":
        return cls(
            mapping_id=str(row["Mapping_ID"]),
            source_ticker=str(row["Source_Ticker"]),
            provider_symbol=str(row["Provider_Symbol"]),
            asset_id=str(row["Asset_ID"]),
            effective_start=_optional_date(row["Effective_Start"]),
            effective_end=_optional_date(row["Effective_End"]),
            resolution_method=str(row["Resolution_Method"]),
            event_id=str(row["Event_ID"]),
        )

    def is_effective(self, value: object) -> bool:
        date = _date(value)
        return (
            (self.effective_start is None or self.effective_start <= date)
            and (self.effective_end is None or date < self.effective_end)
        )


@dataclass(frozen=True, slots=True)
class ResolvedIdentity:
    """One deterministic source resolution, including collision expansions."""

    identities: tuple[ProviderIdentity, ...]
    resolution_method: str

    @property
    def asset_ids(self) -> tuple[str, ...]:
        return tuple(sorted(identity.asset_id for identity in self.identities))

    @property
    def provider_symbols(self) -> tuple[str, ...]:
        return tuple(sorted(
            identity.provider_symbol
            for identity in self.identities
            if identity.provider_symbol
        ))


@dataclass(frozen=True, slots=True)
class PrematureSuccessorRule:
    """Canonical event evidence excluding a successor before its official date."""

    event_date: pd.Timestamp
    primary_source_url: str


def _date(value: object) -> pd.Timestamp:
    date = pd.Timestamp(value)
    if date.tzinfo is not None:
        date = date.tz_localize(None)
    return date.normalize()


def _optional_date(value: object) -> pd.Timestamp | None:
    text = str(value).strip()
    return _date(text) if text else None


def normalize_identity_key(value: str) -> str:
    """Normalize syntax while retaining ambiguous RICs for fail-closed handling."""
    ticker = str(value).strip()
    if not ticker:
        raise ProviderIdentityError("Cannot normalize an empty ticker")
    ticker = ticker.split("^", 1)[0]
    for suffix in (".PK", ".O", ".K"):
        if ticker.upper().endswith(suffix):
            ticker = ticker[: -len(suffix)]
            break
    return "".join(
        character for character in ticker.upper() if character.isalnum()
    )


def _backtest_price_mappings(bundle: SecurityIdentityBundle) -> pd.DataFrame:
    mappings = bundle.provider_mappings
    provider = mappings["Provider"].astype(str).str.strip().str.lower()
    return mappings.loc[
        mappings["Scope"].str.lower().eq("backtest")
        & (
            provider.eq("reuters")
            | mappings["Resolution_Method"].isin(
                CONTRIBUTED_PRICE_RESOLUTION_METHODS
            )
        )
    ].sort_values("Mapping_ID", kind="stable").reset_index(drop=True)


class BacktestProviderIdentityResolver:
    """Resolve backtest identities from one provider observation catalog."""

    def __init__(
        self,
        bundle: SecurityIdentityBundle,
        provider_observations: pd.DataFrame,
    ) -> None:
        observations = provider_observations.copy()
        required = {"Provider", "Provider_Symbol"}
        if not required.issubset(observations.columns):
            raise ProviderIdentityError(
                f"Provider observations must contain {sorted(required)}"
            )
        observations["Provider"] = (
            observations["Provider"].astype("string").str.strip().str.lower()
        )
        observations["Provider_Symbol"] = (
            observations["Provider_Symbol"].astype("string").str.strip()
        )
        if (
            observations[["Provider", "Provider_Symbol"]].isna().any().any()
            or observations[["Provider", "Provider_Symbol"]].eq("").any().any()
        ):
            raise ProviderIdentityError(
                "Provider observations contain invalid identities"
            )

        candidates = _backtest_price_mappings(bundle)
        if candidates["Review_Status"].ne("approved").any():
            unresolved = sorted(
                candidates.loc[
                    candidates["Review_Status"].ne("approved"), "Mapping_ID"
                ].astype(str)
            )
            raise ProviderIdentityError(
                f"Unapproved backtest provider mappings remain: {unresolved}"
            )
        self._mappings = candidates
        identities_by_source: dict[
            str, list[tuple[str, ProviderIdentity]]
        ] = {}
        for row in self._mappings.to_dict("records"):
            source = str(row["Source_Ticker"]).upper()
            identities_by_source.setdefault(source, []).append((
                str(row["Provider"]).lower(),
                ProviderIdentity.from_mapping(row),
            ))
        self._identities_by_source = {
            source: tuple(identities)
            for source, identities in identities_by_source.items()
        }
        self._collision_mapping_ids = frozenset(
            self._mappings.loc[
                self._mappings["Simultaneous_Symbol_Conflict"]
                .astype(str)
                .str.lower()
                .eq("true"),
                "Mapping_ID",
            ].astype(str)
        )

        grouped: dict[str, list[str]] = {}
        reuters_symbols = set(
            observations.loc[
                observations["Provider"].eq("reuters"), "Provider_Symbol"
            ].astype(str)
        )
        for price_ric in sorted(map(str, reuters_symbols)):
            grouped.setdefault(normalize_identity_key(price_ric), []).append(price_ric)
        self._normalized_reuters_symbols = {
            key: tuple(values) for key, values in grouped.items()
        }

    def resolve(
        self,
        source_ticker: str,
        effective_date: object,
    ) -> ResolvedIdentity:
        """Resolve one source ticker to its effective canonical price identity."""
        ticker = str(source_ticker).strip().upper()
        if not ticker:
            raise ProviderIdentityError("Source ticker cannot be empty")
        date = _date(effective_date)
        source_identities = self._identities_by_source.get(ticker, ())
        contributed = tuple(
            identity
            for provider, identity in source_identities
            if provider != "reuters" and identity.is_effective(date)
        )
        if len(contributed) > 1:
            raise ProviderIdentityError(
                f"Contributed identity {ticker} has multiple mappings active on "
                f"{date.date()}"
            )
        if contributed:
            identity = contributed[0]
            return ResolvedIdentity(
                identities=(identity,),
                resolution_method=identity.resolution_method,
            )

        reuters_identities = tuple(
            identity
            for provider, identity in source_identities
            if provider == "reuters"
        )
        if reuters_identities:
            identities = reuters_identities
            active = tuple(
                identity for identity in identities if identity.is_effective(date)
            )
            if active:
                methods = {identity.resolution_method for identity in active}
                if methods == {"reviewed_missing_price"}:
                    return ResolvedIdentity(
                        identities=active,
                        resolution_method="reviewed_missing_price",
                    )
                if "reviewed_missing_price" in methods:
                    raise ProviderIdentityError(
                        f"{ticker} mixes mapped and missing-price identities on "
                        f"{date.date()}"
                    )
                if len(active) > 1 and not all(
                    identity.mapping_id in self._collision_mapping_ids
                    for identity in active
                ):
                    raise ProviderIdentityError(
                        f"{ticker} has multiple mappings without a collision split"
                    )
                return ResolvedIdentity(
                    identities=tuple(sorted(active, key=lambda item: item.asset_id)),
                    resolution_method="+".join(sorted(methods)),
                )

        candidates = self._normalized_reuters_symbols.get(
            normalize_identity_key(ticker), ()
        )
        if reuters_identities and (
            len(candidates) != 1
            or date >= min(
                identity.effective_start
                for identity in reuters_identities
                if identity.effective_start is not None
            )
            or candidates[0]
            in {identity.provider_symbol for identity in reuters_identities}
        ):
            raise ProviderIdentityError(
                f"Reviewed source {ticker} has no mapping active on {date.date()}"
            )
        if len(candidates) != 1:
            raise ProviderIdentityError(
                f"Automatic resolution for {ticker} requires exactly one "
                f"normalized local RIC; found {list(candidates)}"
            )
        provider_symbol = candidates[0]
        identity = ProviderIdentity(
            mapping_id="",
            source_ticker=ticker,
            provider_symbol=provider_symbol,
            asset_id=provider_symbol,
            effective_start=None,
            effective_end=None,
            resolution_method="automatic_unique_ric",
            event_id="",
        )
        return ResolvedIdentity(
            identities=(identity,),
            resolution_method="automatic_unique_ric",
        )


def provider_identity_validation_audit(
    bundle: SecurityIdentityBundle,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Validate mapped source bounds once and return the persisted audit."""

    required = {"Observation_Date", "Provider", "Provider_Symbol", "Mapping_ID"}
    if not required.issubset(observations.columns):
        raise ProviderIdentityError(
            f"Price observations must contain {sorted(required)}"
        )
    catalog = observations.loc[:, sorted(required)].copy()
    catalog["Observation_Date"] = pd.to_datetime(
        catalog["Observation_Date"], errors="coerce"
    )
    catalog["Provider"] = catalog["Provider"].astype(str).str.lower()
    catalog["Provider_Symbol"] = catalog["Provider_Symbol"].astype(str)
    catalog["Mapping_ID"] = catalog["Mapping_ID"].astype(str)
    if catalog["Observation_Date"].isna().any():
        raise ProviderIdentityError("Price observations contain an invalid date")

    candidates = _backtest_price_mappings(bundle)

    observed_bounds: dict[str, tuple[str, str]] = {}
    errors: list[str] = []
    for mapping in candidates.to_dict("records"):
        mapping_id = str(mapping["Mapping_ID"])
        source = catalog.loc[
            catalog["Provider"].eq(str(mapping["Provider"]).lower())
            & catalog["Provider_Symbol"].eq(str(mapping["Provider_Symbol"]))
        ]
        if str(mapping["Provider"]).lower() != "reuters":
            source = source.loc[source["Mapping_ID"].eq(mapping_id)]
        missing = str(mapping["Resolution_Method"]) == "reviewed_missing_price"
        bounds = None if missing or source.empty else (
            source["Observation_Date"].min().normalize(),
            source["Observation_Date"].max().normalize(),
        )
        if not missing and bounds is None:
            errors.append(f"{mapping_id}: mapped provider has no observations")
        expected_first = _optional_date(mapping["Local_First_Date"])
        expected_last = _optional_date(mapping["Local_Last_Date"])
        if bounds is not None and (
            (expected_first is not None and expected_first != bounds[0])
            or (expected_last is not None and expected_last != bounds[1])
        ):
            errors.append(f"{mapping_id}: local observation bounds do not match")
        observed_bounds[mapping_id] = (
            bounds[0].strftime("%Y-%m-%d") if bounds else "",
            bounds[1].strftime("%Y-%m-%d") if bounds else "",
        )
    if errors:
        raise ProviderIdentityError("; ".join(errors))
    audit = candidates.loc[:, list(IDENTITY_VALIDATION_COLUMNS[:10])].copy()
    audit["Observed_First_Date"] = audit["Mapping_ID"].map(
        lambda mapping_id: observed_bounds[str(mapping_id)][0]
    )
    audit["Observed_Last_Date"] = audit["Mapping_ID"].map(
        lambda mapping_id: observed_bounds[str(mapping_id)][1]
    )
    audit["Validation_Status"] = "approved_offline"
    return audit.loc[:, list(IDENTITY_VALIDATION_COLUMNS)]


class YahooIdentityResolver:
    """Resolve Yahoo symbols according to their analytical purpose."""

    def __init__(self, bundle: SecurityIdentityBundle, *, scope: str) -> None:
        scope = str(scope).strip().lower()
        if scope not in {"backtest", "live"}:
            raise ProviderIdentityError(f"Unsupported Yahoo scope: {scope!r}")
        self.scope = scope
        mappings = bundle.provider_mappings.loc[
            bundle.provider_mappings["Scope"].str.lower().eq(scope)
            & bundle.provider_mappings["Provider"].str.lower().eq("yahoo")
            & bundle.provider_mappings["Resolution_Method"].ne(
                "reviewed_yahoo_event_treatment"
            )
        ].copy()
        if mappings["Review_Status"].ne("approved").any():
            raise ProviderIdentityError("Unapproved Yahoo mappings remain")
        self._mappings = mappings.sort_values(
            ["Source_Ticker", "Effective_Start", "Effective_End", "Mapping_ID"],
            kind="stable",
        ).reset_index(drop=True)

    def _for_purpose(
        self,
        identity: ProviderIdentity,
        purpose: YahooPurpose,
    ) -> ProviderIdentity:
        if purpose not in {"historical_prices", "effective_security"}:
            raise ProviderIdentityError(f"Unsupported Yahoo purpose: {purpose!r}")
        if (
            purpose == "effective_security"
            and identity.resolution_method == "reviewed_yahoo_alias"
            and identity.event_id
            and identity.provider_symbol != identity.source_ticker
        ):
            return replace(identity, provider_symbol=identity.source_ticker)
        return identity

    def resolve(
        self,
        source_ticker: str,
        *,
        purpose: YahooPurpose,
        as_of: object | None = None,
    ) -> ProviderIdentity:
        value = str(source_ticker).strip().split("^", 1)[0].strip()
        if not value:
            raise ProviderIdentityError("Ticker cannot be empty")
        reviewed = self._mappings.loc[
            self._mappings["Source_Ticker"].str.upper().eq(value.upper())
        ]
        if (
            self.scope == "backtest"
            and as_of is None
            and not reviewed.empty
            and reviewed["Resolution_Method"].eq(
                "reviewed_effective_symbol"
            ).all()
        ):
            # A display-symbol normalization without an effective date is not
            # allowed to choose one side of a shares-specific symbol interval.
            reviewed = reviewed.iloc[0:0]
        candidates = tuple(
            ProviderIdentity.from_mapping(row)
            for row in reviewed.to_dict("records")
        )
        if as_of is not None and candidates:
            candidates = tuple(
                identity for identity in candidates if identity.is_effective(as_of)
            )
            if not candidates:
                raise ProviderIdentityError(
                    f"Ticker {value.upper()!r} has no reviewed Yahoo symbol "
                    f"effective on {_date(as_of).date()}"
                )
        candidates = tuple(
            self._for_purpose(identity, purpose) for identity in candidates
        )
        symbols = sorted({identity.provider_symbol for identity in candidates})
        if len(symbols) == 1:
            matching = [
                identity
                for identity in candidates
                if identity.provider_symbol == symbols[0]
            ]
            if len(matching) != 1:
                raise ProviderIdentityError(
                    f"Ticker {value.upper()!r} has duplicate effective Yahoo mappings"
                )
            return matching[0]
        if len(symbols) > 1:
            raise ProviderIdentityError(
                f"Ticker {value.upper()!r} has multiple effective Yahoo symbols; "
                "supply as_of"
            )
        if reviewed.empty:
            if self.scope == "live" and "." in value:
                raise ProviderIdentityError(
                    f"Missing reviewed Yahoo alias for source ticker {value.upper()!r}"
                )
            symbol = value.replace(".", "-")
            return ProviderIdentity(
                mapping_id="",
                source_ticker=value,
                provider_symbol=symbol,
                asset_id=value,
                effective_start=None,
                effective_end=None,
                resolution_method="ordinary_symbol",
                event_id="",
            )
        raise ProviderIdentityError(
            f"Ticker {value.upper()!r} has no effective Yahoo mapping"
        )

    def identities_for_asset(
        self,
        asset_id: str,
        *,
        purpose: YahooPurpose,
    ) -> tuple[ProviderIdentity, ...]:
        """Return every effective-dated Yahoo identity for one canonical asset."""
        rows = self._mappings.loc[self._mappings["Asset_ID"].eq(str(asset_id))]
        return tuple(
            self._for_purpose(ProviderIdentity.from_mapping(row), purpose)
            for row in rows.to_dict("records")
        )


def derive_premature_successor_rules(
    bundle: SecurityIdentityBundle,
) -> dict[str, PrematureSuccessorRule]:
    """Derive successor cutoffs from canonical events, legs, mappings, and sources."""
    eligible_events = bundle.events.loc[
        bundle.events["Legacy_Event_Type"].isin(
            {"ticker_or_name_change", "merger_or_successor"}
        )
    ].set_index("Event_ID")
    sources = bundle.sources.set_index("Source_ID")
    reuters = bundle.provider_mappings.loc[
        bundle.provider_mappings["Scope"].eq("backtest")
        & bundle.provider_mappings["Provider"].eq("reuters")
        & bundle.provider_mappings["Event_ID"].isin(eligible_events.index)
    ]
    candidates: dict[str, list[PrematureSuccessorRule]] = {}
    for mapping in reuters.to_dict("records"):
        event_id = str(mapping["Event_ID"])
        event = eligible_events.loc[event_id]
        event_date = _date(event["Effective_Date"])
        legs = bundle.legs.loc[
            bundle.legs["Event_ID"].eq(event_id)
            & bundle.legs["To_Ticker"].ne("")
            & bundle.legs["Leg_Type"].isin(["relabel", "stock"])
        ]
        source_id = str(mapping["Primary_Source_ID"])
        source_url = (
            str(sources.loc[source_id]["Source_URL"])
            if source_id in sources.index
            else ""
        )
        for successor in sorted(set(legs["To_Ticker"].astype(str))):
            candidates.setdefault(successor, []).append(
                PrematureSuccessorRule(
                    event_date=event_date,
                    primary_source_url=source_url,
                )
            )

    rules: dict[str, PrematureSuccessorRule] = {}
    for successor, values in sorted(candidates.items()):
        dates = {value.event_date for value in values}
        if len(dates) != 1:
            raise ProviderIdentityError(
                f"Conflicting official event dates for successor {successor}"
            )
        event_date = values[0].event_date
        has_pre_event_identity = (
            reuters["Source_Ticker"].eq(successor)
            & pd.to_datetime(reuters["Effective_Start"], errors="raise").lt(
                event_date
            )
        ).any()
        if not has_pre_event_identity:
            rules[successor] = values[0]
    return rules


__all__ = [
    "BacktestProviderIdentityResolver",
    "IDENTITY_VALIDATION_COLUMNS",
    "PrematureSuccessorRule",
    "ProviderIdentity",
    "ProviderIdentityError",
    "ResolvedIdentity",
    "YahooIdentityResolver",
    "YahooPurpose",
    "derive_premature_successor_rules",
    "normalize_identity_key",
    "provider_identity_validation_audit",
]
