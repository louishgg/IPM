"""Resolve point-in-time S&P 500 sectors from reviewed evidence.

Wikipedia revisions are raw, dated observations.  They are never treated as a
stable security identifier: effective-dated, reviewed mappings in the shared
security-identity bundle are the only non-syntactic bridge from an index-source
ticker to a different Wikipedia ticker.  S&P constituent notices are causal,
fill-only evidence and never override a populated Wikipedia classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping

import pandas as pd

from portfolio_core.provider_identity import ProviderIdentityError
from portfolio_core.sector_assignments import (
    ASSIGNMENT_COLUMNS,
    validate_sector_assignment_requirements,
    validate_sector_assignments,
)
from portfolio_core.sector_evidence import (
    SectorHistoryPaths,
    clean_sector_text,
    clean_wikipedia_ticker,
    load_sector_evidence,
    normalize_gics_sector,
)
from portfolio_core.security_identity import (
    SecurityIdentityBundle,
    load_security_identity_bundle,
)

_HISTORICAL_TICKER_ANNOTATION = re.compile(
    r"\s*\((?:previously|formerly)\s+[^)]+\)\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _WikipediaSnapshotContext:
    available: frozenset[str]
    by_key: Mapping[str, tuple[str, ...]]


class WikipediaIdentityMissingError(ValueError):
    """No Wikipedia candidate exists for an otherwise valid identity."""


class WikipediaIdentityAmbiguityError(ValueError):
    """Wikipedia identity evidence admits more than one candidate."""


def normalize_symbol_key(value: object) -> str:
    """Normalize punctuation/presentation only; ticker changes need review."""
    ticker = clean_wikipedia_ticker(value)
    if not ticker:
        raise ProviderIdentityError("Cannot normalize an empty sector ticker")
    ticker = _HISTORICAL_TICKER_ANNOTATION.sub("", ticker)
    return "".join(character for character in ticker if character.isalnum())


class WikipediaIdentityResolver:
    """Resolve FJA symbols to Wikipedia without company-name guessing."""

    def __init__(self, bundle: SecurityIdentityBundle | None = None) -> None:
        self._active_reuters_collision_cache: dict[
            tuple[str, pd.Timestamp], bool
        ] = {}
        self._notice_symbol_keys_cache: dict[
            tuple[str, pd.Timestamp, str], tuple[str, ...]
        ] = {}
        if bundle is None:
            self._mappings = pd.DataFrame()
            self._reuters_mappings = pd.DataFrame()
            self._mapping_sources = frozenset()
            self._reuters_collision_sources = frozenset()
            return
        mappings = bundle.provider_mappings.loc[
            bundle.provider_mappings["Scope"].str.casefold().eq("shared")
            & bundle.provider_mappings["Provider"].str.casefold().eq("wikipedia")
        ].copy()
        if mappings["Review_Status"].ne("approved").any():
            raise ProviderIdentityError("Unapproved Wikipedia identity mappings remain")
        self._mappings = mappings.sort_values(
            ["Source_Ticker", "Effective_Start", "Effective_End", "Mapping_ID"],
            kind="stable",
        ).reset_index(drop=True)
        self._mappings["_Effective_Start_TS"] = pd.to_datetime(
            self._mappings["Effective_Start"], errors="raise"
        )
        self._mappings["_Effective_End_TS"] = pd.to_datetime(
            self._mappings["Effective_End"].replace("", pd.NA), errors="coerce"
        )
        self._mapping_sources = frozenset(
            self._mappings["Source_Ticker"].str.upper()
        )
        self._reuters_mappings = bundle.provider_mappings.loc[
            bundle.provider_mappings["Scope"].str.casefold().eq("backtest")
            & bundle.provider_mappings["Provider"].str.casefold().eq("reuters")
            & bundle.provider_mappings["Review_Status"].eq("approved")
        ].copy()
        self._reuters_mappings["_Effective_Start_TS"] = pd.to_datetime(
            self._reuters_mappings["Effective_Start"], errors="raise"
        )
        self._reuters_mappings["_Effective_End_TS"] = pd.to_datetime(
            self._reuters_mappings["Effective_End"].replace("", pd.NA),
            errors="coerce",
        )
        reuters_asset_counts = self._reuters_mappings.groupby(
            self._reuters_mappings["Source_Ticker"].str.upper()
        )["Asset_ID"].nunique()
        self._reuters_collision_sources = frozenset(
            reuters_asset_counts.loc[reuters_asset_counts.gt(1)].index
        )

    @staticmethod
    def _symbol_index(snapshot: pd.DataFrame) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for symbol in snapshot["Wikipedia_Ticker"].astype(str):
            grouped.setdefault(normalize_symbol_key(symbol), []).append(symbol)
        return {key: tuple(sorted(set(values))) for key, values in grouped.items()}

    def snapshot_context(self, snapshot: pd.DataFrame) -> _WikipediaSnapshotContext:
        """Index one immutable revision once for all assets on its date."""
        return _WikipediaSnapshotContext(
            available=frozenset(snapshot["Wikipedia_Ticker"].astype(str)),
            by_key=self._symbol_index(snapshot),
        )

    def potential_notice_source_keys(
        self,
        notice_symbols: Iterable[object],
    ) -> frozenset[str]:
        """Return source keys that could match notices through reviewed maps."""
        notice_keys = {
            normalize_symbol_key(symbol) for symbol in notice_symbols
        }
        source_keys = set(notice_keys)
        if self._mappings.empty:
            return frozenset(source_keys)
        provider_keys = self._mappings["Provider_Symbol"].map(
            normalize_symbol_key
        )
        source_keys.update(
            self._mappings.loc[
                provider_keys.isin(notice_keys), "Source_Ticker"
            ].map(normalize_symbol_key)
        )
        return frozenset(source_keys)

    @staticmethod
    def _resolve_mapping_target(
        row: pd.Series,
        *,
        available: set[str] | frozenset[str],
        by_key: Mapping[str, tuple[str, ...]],
    ) -> tuple[str, str]:
        target = clean_wikipedia_ticker(row["Provider_Symbol"])
        if target in available:
            resolved = target
        else:
            candidates = by_key.get(normalize_symbol_key(target), ())
            if not candidates:
                raise WikipediaIdentityMissingError(
                    f"Reviewed Wikipedia mapping {row['Mapping_ID']} targets "
                    f"missing symbol {target!r}"
                )
            if len(candidates) > 1:
                raise WikipediaIdentityAmbiguityError(
                    f"Reviewed Wikipedia mapping {row['Mapping_ID']} targets "
                    f"ambiguous symbol {target!r}: {list(candidates)}"
                )
            resolved = candidates[0]
        return resolved, f"reviewed_wikipedia_mapping:{row['Mapping_ID']}"

    def _effective_mapping_rows(
        self,
        source_ticker: str,
        as_of_date: object,
    ) -> pd.DataFrame:
        source = clean_wikipedia_ticker(source_ticker)
        if source not in self._mapping_sources:
            return self._mappings.iloc[0:0]
        rows = (
            self._mappings.loc[
                self._mappings["Source_Ticker"].str.upper().eq(source)
            ].copy()
            if not self._mappings.empty
            else self._mappings.copy()
        )
        if rows.empty:
            return rows
        date = pd.Timestamp(as_of_date).normalize()
        starts = rows["_Effective_Start_TS"]
        ends = rows["_Effective_End_TS"]
        return rows.loc[starts.le(date) & (ends.isna() | ends.gt(date))]

    def _source_level_mapping_rows(
        self,
        rows: pd.DataFrame,
        *,
        source: str,
        asset: str,
        as_of_date: object,
    ) -> pd.DataFrame:
        """Select a source bridge without borrowing another collision leg.

        Source-level mappings use the membership ticker as ``Asset_ID``. They
        may bridge another Reuters RIC only when the reviewed identity contract
        has at most one active Reuters security for that source/date. With two
        simultaneous securities, only an exact asset-specific Wikipedia
        mapping is safe.
        """
        if rows.empty:
            return rows
        if not asset:
            return rows
        if self._has_active_reuters_collision(source, as_of_date):
            return rows.iloc[0:0]
        return rows.loc[rows["Asset_ID"].str.upper().eq(source)]

    def _has_active_reuters_collision(
        self,
        source_ticker: str,
        as_of_date: object,
    ) -> bool:
        source = clean_wikipedia_ticker(source_ticker)
        if source not in self._reuters_collision_sources:
            return False
        date = pd.Timestamp(as_of_date).normalize()
        cache_key = (source, date)
        cached = self._active_reuters_collision_cache.get(cache_key)
        if cached is not None:
            return cached
        reuters = self._reuters_mappings.loc[
            self._reuters_mappings["Source_Ticker"].str.upper().eq(source)
        ].copy()
        if reuters.empty:
            self._active_reuters_collision_cache[cache_key] = False
            return False
        starts = reuters["_Effective_Start_TS"]
        ends = reuters["_Effective_End_TS"]
        collision = (
            reuters.loc[
                starts.le(date) & (ends.isna() | ends.gt(date)), "Asset_ID"
            ].nunique()
            > 1
        )
        self._active_reuters_collision_cache[cache_key] = collision
        return collision

    def notice_symbol_keys(
        self,
        source_ticker: str,
        as_of_date: object,
        *,
        asset_id: str | None = None,
    ) -> tuple[str, ...]:
        """Return exact source/mapped keys that may identify an S&P notice.

        Only approved, effective-dated Wikipedia mappings are considered. The
        asset-specific precedence mirrors :meth:`resolve`, which prevents a
        collision leg's notice from being borrowed by another security.
        """
        source = clean_wikipedia_ticker(source_ticker)
        asset = str(asset_id).strip() if asset_id is not None else ""
        if asset_id is not None and not asset:
            raise ProviderIdentityError("Wikipedia Asset_ID cannot be empty")
        date = pd.Timestamp(as_of_date).normalize()
        cache_key = (source, date, asset)
        cached = self._notice_symbol_keys_cache.get(cache_key)
        if cached is not None:
            return cached
        has_source_collision = self._has_active_reuters_collision(
            source, date
        )
        keys = set() if has_source_collision else {normalize_symbol_key(source)}
        rows = self._effective_mapping_rows(source, date)
        asset_rows = (
            rows.loc[rows["Asset_ID"].eq(asset)]
            if asset and not rows.empty
            else rows.iloc[0:0]
        )
        if len(asset_rows) > 1:
            raise WikipediaIdentityAmbiguityError(
                f"{source} ({asset}) has multiple approved Wikipedia mappings "
                f"on {pd.Timestamp(as_of_date).normalize().date()}"
            )
        if len(asset_rows) == 1:
            rows = asset_rows
        else:
            rows = self._source_level_mapping_rows(
                rows,
                source=source,
                asset=asset,
                as_of_date=date,
            )
        if len(rows) > 1:
            raise WikipediaIdentityAmbiguityError(
                f"{source}{f' ({asset})' if asset else ''} has ambiguous notice "
                f"identity on {date.date()}"
            )
        if len(rows) == 1:
            keys.add(normalize_symbol_key(rows.iloc[0]["Provider_Symbol"]))
        result = tuple(sorted(keys))
        self._notice_symbol_keys_cache[cache_key] = result
        return result

    def resolve(
        self,
        source_ticker: str,
        as_of_date: object,
        snapshot: pd.DataFrame,
        *,
        asset_id: str | None = None,
        snapshot_context: _WikipediaSnapshotContext | None = None,
    ) -> tuple[str, str]:
        source = clean_wikipedia_ticker(source_ticker)
        asset = str(asset_id).strip() if asset_id is not None else ""
        if asset_id is not None and not asset:
            raise ProviderIdentityError("Wikipedia Asset_ID cannot be empty")
        date = pd.Timestamp(as_of_date).normalize()
        context = snapshot_context or self.snapshot_context(snapshot)
        available = context.available
        by_key = context.by_key

        rows = self._effective_mapping_rows(source, date)

        # A reviewed mapping for the exact economic asset must win even when
        # the retrospective membership ticker is also present in Wikipedia.
        # This is what distinguishes simultaneous source-symbol collisions
        # such as old JCI from the TYC security that later assumed JCI.
        asset_rows = (
            rows.loc[rows["Asset_ID"].eq(asset)]
            if asset and not rows.empty
            else rows.iloc[0:0]
        )
        if len(asset_rows) > 1:
            raise WikipediaIdentityAmbiguityError(
                f"{source} ({asset}) has multiple approved Wikipedia mappings "
                f"on {date.date()}"
            )
        if len(asset_rows) == 1:
            return self._resolve_mapping_target(
                asset_rows.iloc[0], available=available, by_key=by_key
            )

        if source in available:
            return source, "exact_wikipedia_symbol"
        normalized = by_key.get(normalize_symbol_key(source), ())
        if len(normalized) == 1:
            method = (
                "normalized_wikipedia_presentation"
                if _HISTORICAL_TICKER_ANNOTATION.search(normalized[0])
                else "normalized_wikipedia_punctuation"
            )
            return normalized[0], method
        if len(normalized) > 1:
            raise WikipediaIdentityAmbiguityError(
                f"Wikipedia punctuation normalization for {source} is ambiguous: "
                f"{list(normalized)}"
            )

        if self._mappings.empty:
            raise WikipediaIdentityMissingError(
                f"No Wikipedia symbol matches {source} on {date.date()}"
            )
        rows = self._source_level_mapping_rows(
            rows,
            source=source,
            asset=asset,
            as_of_date=date,
        )
        if rows.empty:
            raise WikipediaIdentityMissingError(
                f"{source}{f' ({asset})' if asset else ''} requires exactly one "
                "approved Wikipedia mapping on "
                f"{date.date()}; found 0"
            )
        if len(rows) > 1:
            raise WikipediaIdentityAmbiguityError(
                f"{source}{f' ({asset})' if asset else ''} has multiple approved "
                f"Wikipedia mappings on {date.date()}: found {len(rows)}"
            )
        return self._resolve_mapping_target(
            rows.iloc[0], available=available, by_key=by_key
        )


def _applicable_sector_notices(
    notices: pd.DataFrame,
    *,
    requirement_date: pd.Timestamp,
    notice_symbol_keys: Iterable[str],
    addition_handoff_dates: Mapping[tuple[str, str, str], pd.Timestamp],
    asset_id: str,
) -> pd.DataFrame:
    """Return causal index-notice sector evidence without creating eligibility.

    An approved addition may classify a security once the announcement is
    public, including the narrow interval before its stated effective date.
    This supports an event-delivered forced exit on its actual execution date
    while membership remains independently authoritative.  A same-index
    deletion closes that addition.  Deletion evidence can otherwise fill only
    a still-live membership boundary such as BRCM's final January 2016 close.
    """
    requirement_date = pd.Timestamp(requirement_date).normalize()
    candidate_keys = {
        normalize_symbol_key(value) for value in notice_symbol_keys
    }
    keys = notices["Ticker"].map(normalize_symbol_key)
    symbol_match = keys.isin(candidate_keys)
    if not symbol_match.any():
        return notices.iloc[0:0]
    candidates = notices.loc[symbol_match]
    candidates = candidates.loc[
        pd.to_datetime(candidates["Published_Date"], errors="raise")
        .dt.normalize()
        .lt(requirement_date)
        & candidates["Index_Name"].isin({"S&P 500", "S&P SmallCap 600"})
        & candidates["Review_Status"].eq("approved")
    ].copy()
    if candidates.empty:
        return candidates
    candidates["Effective_Date_Normalized"] = pd.to_datetime(
        candidates["Effective_Date"], errors="raise"
    ).dt.normalize()
    candidates["Action_Normalized"] = candidates["Action"].str.casefold()

    additions = candidates.loc[
        candidates["Action_Normalized"].eq("addition")
    ].copy()
    active_additions: list[bool] = []
    for addition in additions.itertuples(index=False):
        closed = candidates.loc[
            candidates["Action_Normalized"].eq("deletion")
            & candidates["Index_Name"].eq(addition.Index_Name)
            & candidates["Effective_Date_Normalized"].ge(
                addition.Effective_Date_Normalized
            )
            & candidates["Effective_Date_Normalized"].le(requirement_date)
        ]
        handoff = addition_handoff_dates.get(
            (*_notice_key(addition), asset_id)
        )
        active_additions.append(
            closed.empty
            and (handoff is None or requirement_date <= handoff)
        )
    additions = additions.loc[active_additions]
    if not additions.empty:
        return additions.drop(
            columns=["Effective_Date_Normalized", "Action_Normalized"]
        )

    deletion_boundary = candidates.loc[
        candidates["Action_Normalized"].eq("deletion")
        & candidates["Effective_Date_Normalized"].le(requirement_date)
    ].copy()
    return deletion_boundary.drop(
        columns=["Effective_Date_Normalized", "Action_Normalized"]
    )


def _notice_key(row: object) -> tuple[str, str]:
    return (
        str(getattr(row, "Notice_ID") if hasattr(row, "Notice_ID") else row["Notice_ID"]),
        normalize_symbol_key(
            getattr(row, "Ticker") if hasattr(row, "Ticker") else row["Ticker"]
        ),
    )


def _notice_handoff_dates(
    notices: pd.DataFrame,
    *,
    requirements: pd.DataFrame,
    identity_resolver: WikipediaIdentityResolver,
    snapshots_by_date: Mapping[pd.Timestamp, pd.DataFrame],
    snapshot_contexts: Mapping[pd.Timestamp, _WikipediaSnapshotContext],
) -> dict[tuple[str, str, str], pd.Timestamp]:
    """Return the first resolved Wikipedia handoff for each notice and asset."""
    if notices.empty or not snapshots_by_date:
        return {}
    result: dict[tuple[str, str, str], pd.Timestamp] = {}
    additions = notices.loc[notices["Action"].str.casefold().eq("addition")]
    required_columns = {
        "As_Of_Date",
        "Source_Ticker",
        "Asset_ID",
    }
    missing_columns = sorted(required_columns - set(requirements.columns))
    if missing_columns:
        raise ValueError(
            f"Notice handoff requirements are missing columns {missing_columns}"
        )
    required = requirements.copy()
    required["As_Of_Date"] = pd.to_datetime(
        required["As_Of_Date"], errors="raise"
    ).dt.normalize()
    addition_key_values = frozenset(
        additions["Ticker"].map(normalize_symbol_key)
    )
    potential_source_keys = identity_resolver.potential_notice_source_keys(
        addition_key_values
    )
    required = required.loc[
        required["Source_Ticker"]
        .map(normalize_symbol_key)
        .isin(potential_source_keys)
    ].copy()
    addition_keys = additions["Ticker"].map(normalize_symbol_key)
    addition_effective_dates = pd.to_datetime(
        additions["Effective_Date"], errors="raise"
    ).dt.normalize()
    identities = required[["Asset_ID", "Source_Ticker"]].drop_duplicates()
    for record in identities.sort_values(
        ["Asset_ID", "Source_Ticker"], kind="stable"
    ).to_dict("records"):
        source_ticker = str(record["Source_Ticker"])
        asset_id = str(record["Asset_ID"])
        for date in sorted(snapshots_by_date):
            identity_keys = set(
                identity_resolver.notice_symbol_keys(
                    source_ticker,
                    date,
                    asset_id=asset_id,
                )
            )
            candidates = additions.loc[
                addition_keys.isin(identity_keys)
                & addition_effective_dates.le(date)
            ]
            if candidates.empty:
                continue
            snapshot = snapshots_by_date[date]
            try:
                identity_resolver.resolve(
                    source_ticker,
                    date,
                    snapshot,
                    asset_id=asset_id,
                    snapshot_context=snapshot_contexts[date],
                )
            except WikipediaIdentityMissingError:
                continue
            for notice in candidates.itertuples(index=False):
                key = (*_notice_key(notice), asset_id)
                result.setdefault(key, date)
    return result


def _notice_resolution_method(
    notice: pd.Series,
    requirement_date: pd.Timestamp,
) -> str:
    """Describe the exact reviewed notice treatment without implying membership."""

    action = str(notice["Action"]).casefold()
    index_name = str(notice["Index_Name"])
    index_label = (
        "sp500" if index_name == "S&P 500" else "sp_smallcap"
    )
    if action == "deletion":
        return f"approved_{index_label}_deletion_boundary_fill"
    effective = pd.Timestamp(notice["Effective_Date"]).normalize()
    if pd.Timestamp(requirement_date).normalize() < effective:
        return f"approved_{index_label}_announced_sector_fill"
    return f"approved_{index_label}_addition_fill"


@dataclass(frozen=True, slots=True)
class SectorResolutionResult:
    """One complete sector assignment or one explicit evidence gap."""

    as_of_date: pd.Timestamp
    asset_id: str
    source_ticker: str
    gics_sector_code: str = ""
    sector: str = ""
    source_type: str = ""
    source_reference: str = ""
    source_symbol: str = ""
    resolution_method: str = ""
    missing_evidence: str = ""

    @property
    def resolved(self) -> bool:
        return not self.missing_evidence

    @property
    def contributing_source(self) -> str:
        if self.source_type == "Wikipedia":
            return "wikipedia"
        if self.source_type == "S&P Notice":
            return "sp_global_notice"
        return ""

    def assignment_row(self) -> dict[str, str | pd.Timestamp]:
        if not self.resolved:
            raise ValueError(self.missing_evidence)
        return {
            "As_Of_Date": self.as_of_date,
            "Asset_ID": self.asset_id,
            "GICS_Sector_Code": self.gics_sector_code,
            "Sector": self.sector,
            "Source_Type": self.source_type,
            "Source_Reference": self.source_reference,
            "Source_Symbol": self.source_symbol,
            "Resolution_Method": self.resolution_method,
        }


class SectorResolutionContext:
    """Precompute and cache one policy for readiness and prepared assignments."""

    _SCOPE_COLUMN = "_Resolution_Scope"

    def __init__(
        self,
        requirements: pd.DataFrame,
        snapshots: pd.DataFrame,
        notices: pd.DataFrame,
        *,
        identity_resolver: WikipediaIdentityResolver,
        date_column: str,
        scope_column: str | None = None,
    ) -> None:
        required_columns = {date_column, "Asset_ID", "Source_Ticker"}
        if scope_column is not None:
            required_columns.add(scope_column)
        missing_columns = sorted(required_columns - set(requirements.columns))
        if missing_columns:
            raise ValueError(
                f"Sector resolution requirements are missing columns {missing_columns}"
            )
        normalized = pd.DataFrame({
            self._SCOPE_COLUMN: (
                requirements[scope_column].astype(str)
                if scope_column is not None
                else "assignment"
            ),
            "As_Of_Date": pd.to_datetime(
                requirements[date_column], errors="raise"
            ).dt.normalize(),
            "Asset_ID": requirements["Asset_ID"].map(clean_sector_text),
            "Source_Ticker": requirements["Source_Ticker"].map(clean_sector_text),
        })
        if normalized.empty or normalized[
            [self._SCOPE_COLUMN, "Asset_ID", "Source_Ticker"]
        ].eq("").any().any():
            raise ValueError("Sector resolution requirements are empty or incomplete")
        key_columns = [self._SCOPE_COLUMN, "As_Of_Date", "Asset_ID"]
        if normalized.duplicated(key_columns).any():
            raise ValueError("Sector resolution requirements contain duplicate pairs")
        self.requirements = normalized.sort_values(
            ["As_Of_Date", self._SCOPE_COLUMN, "Asset_ID"], kind="stable"
        ).reset_index(drop=True)
        self.snapshots = snapshots.copy()
        self.snapshots["Requirement_Date"] = pd.to_datetime(
            self.snapshots["Requirement_Date"], errors="raise"
        ).dt.normalize()
        self.notices = notices.copy()
        for column in ("Published_Date", "Effective_Date"):
            self.notices[column] = pd.to_datetime(
                self.notices[column], errors="raise"
            ).dt.normalize()
        self.identity_resolver = identity_resolver
        self._snapshots_by_date = {
            pd.Timestamp(date).normalize(): frame.copy()
            for date, frame in self.snapshots.groupby(
                "Requirement_Date", sort=False
            )
        }
        self._snapshot_contexts = {
            date: identity_resolver.snapshot_context(frame)
            for date, frame in self._snapshots_by_date.items()
        }
        self._handoff_dates = _notice_handoff_dates(
            self.notices,
            requirements=self.requirements,
            identity_resolver=identity_resolver,
            snapshots_by_date=self._snapshots_by_date,
            snapshot_contexts=self._snapshot_contexts,
        )
        self._results: dict[
            tuple[str, pd.Timestamp, str, str], SectorResolutionResult
        ] = {}
        for record in self.requirements.to_dict("records"):
            key = self._key(
                record[self._SCOPE_COLUMN],
                record["As_Of_Date"],
                record["Asset_ID"],
                record["Source_Ticker"],
            )
            self._results[key] = self._resolve_requirement(
                requirement_date=key[1],
                asset_id=key[2],
                source_ticker=key[3],
            )

    @staticmethod
    def _key(
        scope: object,
        requirement_date: object,
        asset_id: object,
        source_ticker: object,
    ) -> tuple[str, pd.Timestamp, str, str]:
        return (
            clean_sector_text(scope),
            pd.Timestamp(requirement_date).normalize(),
            clean_sector_text(asset_id),
            clean_sector_text(source_ticker),
        )

    def _missing(
        self,
        *,
        requirement_date: pd.Timestamp,
        asset_id: str,
        source_ticker: str,
        reason: str,
    ) -> SectorResolutionResult:
        return SectorResolutionResult(
            as_of_date=requirement_date,
            asset_id=asset_id,
            source_ticker=source_ticker,
            missing_evidence=reason,
        )

    def _resolve_requirement(
        self,
        *,
        requirement_date: pd.Timestamp,
        asset_id: str,
        source_ticker: str,
    ) -> SectorResolutionResult:
        applicable = _applicable_sector_notices(
            self.notices,
            requirement_date=requirement_date,
            notice_symbol_keys=self.identity_resolver.notice_symbol_keys(
                source_ticker,
                requirement_date,
                asset_id=asset_id,
            ),
            addition_handoff_dates=self._handoff_dates,
            asset_id=asset_id,
        )
        if len(applicable) > 1:
            raise ValueError(
                f"Multiple applicable S&P notices for {source_ticker} "
                f"on {requirement_date.date()}"
            )
        snapshot = self._snapshots_by_date.get(requirement_date)
        if snapshot is None or snapshot.empty:
            if len(applicable) != 1:
                return self._missing(
                    requirement_date=requirement_date,
                    asset_id=asset_id,
                    source_ticker=source_ticker,
                    reason=(
                        "Missing Wikipedia sector snapshot and approved notice "
                        f"for {requirement_date.date()}"
                    ),
                )
            notice = applicable.iloc[0]
            code, label = normalize_gics_sector(notice["Sector"])
            return SectorResolutionResult(
                as_of_date=requirement_date,
                asset_id=asset_id,
                source_ticker=source_ticker,
                gics_sector_code=code,
                sector=label,
                source_type="S&P Notice",
                source_reference=str(notice["Notice_ID"]),
                source_symbol=str(notice["Ticker"]),
                resolution_method=_notice_resolution_method(
                    notice,
                    requirement_date,
                ),
            )
        if snapshot["Revision_ID"].nunique() != 1:
            raise ValueError(
                f"Multiple Wikipedia revisions exist for {requirement_date.date()}"
            )
        try:
            source_symbol, resolution_method = self.identity_resolver.resolve(
                source_ticker,
                requirement_date,
                snapshot,
                asset_id=asset_id,
                snapshot_context=self._snapshot_contexts[requirement_date],
            )
        except WikipediaIdentityMissingError as identity_error:
            if len(applicable) != 1:
                return self._missing(
                    requirement_date=requirement_date,
                    asset_id=asset_id,
                    source_ticker=source_ticker,
                    reason=(
                        f"No causal sector source for {source_ticker} "
                        f"({asset_id}) on {requirement_date.date()}: "
                        f"{identity_error}"
                    ),
                )
            notice = applicable.iloc[0]
            code, label = normalize_gics_sector(notice["Sector"])
            return SectorResolutionResult(
                as_of_date=requirement_date,
                asset_id=asset_id,
                source_ticker=source_ticker,
                gics_sector_code=code,
                sector=label,
                source_type="S&P Notice",
                source_reference=str(notice["Notice_ID"]),
                source_symbol=str(notice["Ticker"]),
                resolution_method=_notice_resolution_method(
                    notice,
                    requirement_date,
                ),
            )

        wikipedia = snapshot.loc[
            snapshot["Wikipedia_Ticker"].eq(source_symbol)
        ]
        if len(wikipedia) != 1:
            raise ValueError(
                f"Wikipedia symbol {source_symbol} is not unique on "
                f"{requirement_date.date()}"
            )
        wikipedia_row = wikipedia.iloc[0]
        code, label = normalize_gics_sector(wikipedia_row["Raw_Sector"])
        if len(applicable) == 1:
            notice_code, _ = normalize_gics_sector(applicable.iloc[0]["Sector"])
            if notice_code != code:
                raise ValueError(
                    f"Wikipedia/S&P notice sector conflict for {source_ticker} "
                    f"on {requirement_date.date()}: {code} != {notice_code}"
                )
        return SectorResolutionResult(
            as_of_date=requirement_date,
            asset_id=asset_id,
            source_ticker=source_ticker,
            gics_sector_code=code,
            sector=label,
            source_type="Wikipedia",
            source_reference=str(wikipedia_row["Revision_ID"]),
            source_symbol=source_symbol,
            resolution_method=resolution_method,
        )

    def resolve(
        self,
        *,
        scope: object,
        requirement_date: object,
        asset_id: object,
        source_ticker: object,
    ) -> SectorResolutionResult:
        key = self._key(scope, requirement_date, asset_id, source_ticker)
        try:
            return self._results[key]
        except KeyError as exc:
            raise KeyError(f"Unknown sector resolution requirement: {key}") from exc


def _assignment_pairs(frame: pd.DataFrame) -> set[tuple[pd.Timestamp, str]]:
    return set(
        frame[["As_Of_Date", "Asset_ID"]].itertuples(index=False, name=None)
    )


def build_sector_assignments(
    requirements: pd.DataFrame,
    snapshots: pd.DataFrame,
    notices: pd.DataFrame,
    *,
    identity_resolver: WikipediaIdentityResolver | None = None,
) -> pd.DataFrame:
    """Resolve complete dated assignments from primary and fill-only evidence.

    Wikipedia is primary on its dated snapshots. Reviewed S&P index notices
    bridge missing sector evidence without creating membership. Missing evidence
    outside that lifecycle fails closed.
    """
    requirements = validate_sector_assignment_requirements(requirements)

    resolver = identity_resolver or WikipediaIdentityResolver()
    context = SectorResolutionContext(
        requirements,
        snapshots,
        notices,
        date_column="As_Of_Date",
        identity_resolver=resolver,
    )
    rows = []
    for item in requirements.sort_values(
        ["As_Of_Date", "Asset_ID"], kind="stable"
    ).itertuples(index=False):
        resolution = context.resolve(
            scope="assignment",
            requirement_date=item.As_Of_Date,
            asset_id=item.Asset_ID,
            source_ticker=item.Source_Ticker,
        )
        if not resolution.resolved:
            raise ValueError(resolution.missing_evidence)
        rows.append(resolution.assignment_row())

    result = pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)
    return validate_sector_assignments(result)


def build_sector_assignments_from_evidence(
    requirements: pd.DataFrame,
    *,
    paths: SectorHistoryPaths,
    repository_root: Path,
) -> pd.DataFrame:
    """Resolve and reconcile requirements against repository evidence."""
    requirements = validate_sector_assignment_requirements(requirements)
    evidence = load_sector_evidence(
        paths,
        repository_root=repository_root,
    )
    identity_bundle = load_security_identity_bundle(
        repository_root,
        validate_manifest=True,
        require_all_approved=True,
    )
    assignments = build_sector_assignments(
        requirements,
        evidence.snapshots,
        evidence.notices,
        identity_resolver=WikipediaIdentityResolver(identity_bundle),
    )
    if _assignment_pairs(assignments) != _assignment_pairs(requirements):
        raise ValueError(
            "Prepared sector assignments do not match consumed requirements"
        )
    return assignments


__all__ = [
    "SectorResolutionContext",
    "WikipediaIdentityAmbiguityError",
    "WikipediaIdentityMissingError",
    "WikipediaIdentityResolver",
    "build_sector_assignments",
    "build_sector_assignments_from_evidence",
]
