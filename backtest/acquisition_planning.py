"""Backtest-owned plans and preflights for the centralized acquisition CLI.

Provider execution, retries, checkpoints, and manifests live in
``backtest.acquisition_execution``. This module owns only backtest planning,
readiness, freshness preflights, and supplied-artifact validation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from portfolio_core.artifacts import (
    ArtifactOrigin,
    validate_exact_manifest_catalog,
)
from data_acquisition.sector_acquisition_planning import (
    project_sector_acquisition_requirements,
)
from data_acquisition.contracts import (
    AcquisitionIdentity,
    AcquisitionRequest,
    ReadinessRecord,
    readiness_status,
    read_asset_selection,
    utc_timestamp,
)
from data_acquisition.share_checkpoints import (
    project_share_identity_interval,
)
from portfolio_core.dates import month_end_index
from portfolio_core.provider_identity import (
    CONTRIBUTED_PRICE_RESOLUTION_METHODS,
    ProviderIdentity,
    YahooIdentityResolver,
)
from portfolio_core.security_identity import (
    SecurityIdentityBundle,
    load_security_identity_bundle,
)
from portfolio_core.shares import (
    CanonicalShareIdentity,
)
from portfolio_core.sp500_membership import (
    build_active_months_by_ticker,
    validate_membership_sources,
)

from .config import (
    BacktestConfig,
    BacktestBrinsonConfig,
    BacktestMarketConfig,
    DEFAULT_CONFIG,
)
from .paths import BacktestPaths
from .data_loading import (
    BacktestDataset,
    PREPARE_CORE_COMMAND,
    active_pit_asset_ids,
    load_backtest_data,
    validate_raw_benchmark,
)
from .share_resolution import ReviewedShares, ShareResolver, required_pairs
from .price_sources import (
    PRICE_OBSERVATION_COLUMNS,
    REUTERS_REMAP_METHOD,
    WIKI_PARENT_COMMIT,
    WIKI_PROVIDER,
    yahoo_normalization_requirements,
)

SCOPE = "backtest"
_SHARES_CORE_RECOVERY = (
    "Backtest shares acquisition requires prepared core data derived from "
    f"current raw provenance; run `{PREPARE_CORE_COMMAND}`, then "
    "retry the shares command."
)


@dataclass(frozen=True)
class YahooSharesIdentity:
    """One effective-dated Yahoo symbol for an economic asset."""

    asset_id: str
    display_ticker: str
    provider_symbol: str
    effective_start: pd.Timestamp | None = None
    effective_end_exclusive: pd.Timestamp | None = None


@dataclass(frozen=True)
class ProjectedYahooSharesIdentity:
    """One residual identity and its already-projected acquisition window."""

    identity: CanonicalShareIdentity
    display_ticker: str
    requested_start: str
    requested_end: str


@dataclass(frozen=True)
class SharesSourcePlan:
    """Reviewed primary coverage and only residual Yahoo identities."""

    required_asset_ids: tuple[str, ...]
    asset_to_ticker: dict[str, str]
    resolver: ShareResolver
    pairs: pd.DataFrame
    primary: pd.DataFrame
    unresolved: pd.DataFrame
    yahoo_identities: tuple[ProjectedYahooSharesIdentity, ...]

    @property
    def canonical_identities(self) -> tuple[CanonicalShareIdentity, ...]:
        return tuple(item.identity for item in self.yahoo_identities)


def load_fresh_core_data(
    config: BacktestConfig = DEFAULT_CONFIG,
) -> BacktestDataset:
    """Load core data only when its manifest still matches raw provenance."""
    try:
        return load_backtest_data(
            config.market,
            config.paths,
            check_raw=True,
        )
    except RuntimeError as error:
        raise RuntimeError(f"{_SHARES_CORE_RECOVERY} Cause: {error}") from error


def _shares_window(
    config: BacktestBrinsonConfig = DEFAULT_CONFIG.brinson,
) -> tuple[str, str]:
    start = (
        pd.Timestamp(config.start_date)
        - pd.DateOffset(months=config.yahoo_lookback_months)
    ).strftime("%Y-%m-%d")
    end = (
        pd.Timestamp(config.end_date) + pd.DateOffset(months=2)
    ).strftime("%Y-%m-%d")
    return start, end


def _build_yahoo_shares_identities(
    required_asset_ids: tuple[str, ...],
    asset_to_ticker: dict[str, str],
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
    *,
    config: BacktestBrinsonConfig = DEFAULT_CONFIG.brinson,
) -> tuple[YahooSharesIdentity, ...]:
    """Build reviewed identities that overlap the requested source window."""
    requested_start_text, requested_end_text = _shares_window(config)
    requested_start = pd.Timestamp(requested_start_text)
    requested_end = pd.Timestamp(requested_end_text)
    canonical = load_security_identity_bundle(
        paths.project_root,
        validate_manifest=True,
        require_all_approved=True,
    )
    resolver = YahooIdentityResolver(canonical, scope=SCOPE)
    rows: list[YahooSharesIdentity] = []
    for asset_id in required_asset_ids:
        display_ticker = asset_to_ticker[str(asset_id)]
        identities = resolver.identities_for_asset(
            str(asset_id),
            purpose="effective_security",
        )
        if not identities:
            identities = (
                resolver.resolve(
                    display_ticker,
                    purpose="effective_security",
                ),
            )
        for identity in identities:
            effective_start = identity.effective_start
            effective_end_exclusive = identity.effective_end
            if (
                effective_end_exclusive is not None
                and pd.Timestamp(effective_end_exclusive) <= requested_start
            ) or (
                effective_start is not None
                and pd.Timestamp(effective_start) > requested_end
            ):
                continue
            rows.append(
                YahooSharesIdentity(
                    asset_id=str(asset_id),
                    display_ticker=display_ticker,
                    provider_symbol=identity.provider_symbol,
                    effective_start=effective_start,
                    effective_end_exclusive=effective_end_exclusive,
                )
            )

    missing_assets = sorted(
        set(required_asset_ids) - {row.asset_id for row in rows}
    )
    if missing_assets:
        raise RuntimeError(
            "Active share assets have no Yahoo identity overlapping the requested "
            f"source window: {missing_assets}"
        )
    provider_to_assets: dict[str, set[str]] = {}
    for row in rows:
        provider_to_assets.setdefault(row.provider_symbol, set()).add(row.asset_id)
    collisions = {
        symbol: sorted(asset_ids)
        for symbol, asset_ids in provider_to_assets.items()
        if len(asset_ids) > 1
    }
    if collisions:
        raise RuntimeError(
            "Yahoo shares provider symbols map to multiple share assets: "
            f"{collisions}"
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.asset_id,
                row.effective_start or pd.Timestamp.min,
                row.provider_symbol,
            ),
        )
    )


def get_share_month_end_dates(
    backtest_data: BacktestDataset,
    config: BacktestBrinsonConfig = DEFAULT_CONFIG.brinson,
) -> pd.DatetimeIndex:
    """Return unique local month-end dates for the configured share window."""
    data_close = backtest_data.data_close
    dates = data_close.index[
        (data_close.index >= pd.Timestamp(config.start_date))
        & (data_close.index <= pd.Timestamp(config.end_date))
    ]
    return pd.DatetimeIndex(sorted(pd.unique(dates)))


def _build_share_identity_context(
    backtest_data: BacktestDataset,
    share_dates: pd.DatetimeIndex,
) -> tuple[tuple[str, ...], dict[str, str], pd.DataFrame]:
    """Return the exact Asset-ID boundary used by shares preparation."""
    active_asset_ids: set[str] = set()
    for date in share_dates:
        active_asset_ids.update(active_pit_asset_ids(backtest_data, date))
    price_asset_ids = set(backtest_data.data_close.columns.astype(str))
    required_asset_ids = tuple(sorted(active_asset_ids & price_asset_ids))
    asset_to_ticker = {
        asset_id: str(backtest_data.asset_to_ticker[asset_id])
        for asset_id in required_asset_ids
    }
    membership = pd.DataFrame(
        False,
        index=share_dates,
        columns=required_asset_ids,
    )
    membership.index.name = "Date"
    pit_matrix = backtest_data.pit_matrix
    for date in share_dates:
        closest_date = pit_matrix.index[pit_matrix.index <= date][-1]
        active = pit_matrix.loc[closest_date]
        active_ids = sorted(
            set(active.index[active].astype(str)) & set(required_asset_ids)
        )
        membership.loc[date, active_ids] = True
    return required_asset_ids, asset_to_ticker, membership


def plan_share_sources(
    backtest_data: BacktestDataset,
    *,
    config: BacktestConfig = DEFAULT_CONFIG,
) -> SharesSourcePlan:
    """Evaluate reviewed primary coverage before resolving Yahoo identities."""
    share_dates = get_share_month_end_dates(backtest_data, config.brinson)
    if share_dates.empty:
        raise RuntimeError("No local price dates found for the configured share window")
    assets, tickers, membership = _build_share_identity_context(backtest_data, share_dates)
    pairs = required_pairs(membership)
    reviewed = ReviewedShares.load(config.paths.shares.reviewed_dir)
    reviewed.validate_inputs(pairs, config.paths.prices_monthly_csv, config.paths.price_basis_csv)
    resolver = ShareResolver(reviewed)
    primary, unresolved = resolver.resolve(pairs)
    residual = tuple(sorted(set(unresolved.Asset_ID)))
    identities = _build_yahoo_shares_identities(
        residual, tickers, config.paths, config=config.brinson,
    ) if residual else ()
    requested_start, requested_end = _shares_window(config.brinson)
    projected = []
    for identity in identities:
        canonical, start, end = project_share_identity_interval(
            asset_id=identity.asset_id,
            provider_symbol=identity.provider_symbol,
            requested_start=requested_start,
            requested_end=requested_end,
            effective_start=identity.effective_start,
            effective_end_exclusive=identity.effective_end_exclusive,
        )
        projected.append(ProjectedYahooSharesIdentity(
            canonical, identity.display_ticker, start, end,
        ))
    return SharesSourcePlan(
        required_asset_ids=assets,
        asset_to_ticker=tickers,
        resolver=resolver,
        pairs=pairs,
        primary=primary,
        unresolved=unresolved,
        yahoo_identities=tuple(projected),
    )


def shares_readiness_records(
    plan: SharesSourcePlan,
    raw: pd.DataFrame,
) -> tuple[list[ReadinessRecord], pd.DataFrame]:
    """Readiness measures usable evidence independently of provider attempts."""
    selected, unresolved = plan.resolver.resolve(plan.pairs, raw, plan.canonical_identities)
    detail = plan.pairs.merge(selected[["Date", "Asset_ID", "Source"]],
                             on=["Date", "Asset_ID"], how="left", validate="one_to_one")
    detail["Covered"] = detail.Source.notna()
    detail["Source"] = detail.Source.fillna("missing")
    detail["Ticker"] = detail.Asset_ID.map(plan.asset_to_ticker)
    detail = detail.merge(unresolved, on=["Date", "Asset_ID"], how="left", validate="one_to_one")
    record = ReadinessRecord(
        scope=SCOPE, dataset="shares", asset_id="*",
        requirement_set="brinson_active_asset_month",
        required_count=len(plan.pairs), covered_count=len(selected),
        status=readiness_status(len(selected), len(plan.pairs)),
        missing_dates=tuple(sorted(unresolved.Date.dt.strftime("%Y-%m-%d").unique())) if len(unresolved) else (),
        contributing_sources=tuple(sorted(selected.Source.unique())),
        checked_at_utc=utc_timestamp(),
    )
    return [record], detail


def build_shares_requests(
    plan: SharesSourcePlan,
    *,
    tickers_file: Path | None = None,
) -> list[AcquisitionRequest]:
    """Request Yahoo only for requirements lacking reviewed primary coverage."""
    selected = set(read_asset_selection(tickers_file)) if tickers_file else None
    identities = list(plan.yahoo_identities)
    if selected is not None:
        known = set(plan.required_asset_ids) | set(plan.asset_to_ticker.values()) | {
            identity.identity.provider_symbol for identity in identities
        } | {
            identity.display_ticker for identity in identities
        } | {
            identity.identity.asset_id for identity in identities
        }
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"Unknown Yahoo share identities: {unknown}")
        identities = [
            identity
            for identity in identities
            if identity.identity.provider_symbol in selected
            or identity.display_ticker in selected
            or identity.identity.asset_id in selected
        ]
    requests = []
    for identity in identities:
        canonical_identity = identity.identity
        requests.append(AcquisitionRequest(
            AcquisitionIdentity(
                scope=SCOPE,
                dataset="shares",
                asset_id=canonical_identity.asset_id,
                provider="yahoo",
                provider_symbol=canonical_identity.provider_symbol,
                effective_start=canonical_identity.effective_start,
                effective_end=canonical_identity.effective_end,
            ),
            requested_start=identity.requested_start,
            requested_end=identity.requested_end,
        ))
    return sorted(requests, key=lambda item: item.identity.key)


def benchmark_readiness_record(
    *,
    contributing_source: str,
    raw: pd.DataFrame | None = None,
    config: BacktestConfig | None = None,
    checked_at_utc: str | None = None,
) -> ReadinessRecord:
    """Validate the canonical benchmark's analytical month-end coverage."""
    active_config = config or DEFAULT_CONFIG
    paths = active_config.paths
    required = active_config.benchmark.required_dates
    covered = pd.DatetimeIndex([])
    if raw is None and paths.benchmark_raw_csv.exists():
        raw = pd.read_csv(paths.benchmark_raw_csv)
    if raw is not None:
        raw = validate_raw_benchmark(raw)
        covered = pd.DatetimeIndex(raw["Date"]).intersection(required)
    missing = required.difference(covered)
    covered_count = len(covered)
    return ReadinessRecord(
        scope=SCOPE,
        dataset="benchmark",
        asset_id="SP500TR",
        requirement_set="monthly_total_return_benchmark",
        required_count=len(required),
        covered_count=covered_count,
        status=readiness_status(covered_count, len(required)),
        missing_dates=tuple(missing.strftime("%Y-%m-%d")),
        contributing_sources=((contributing_source,) if covered_count else ()),
        checked_at_utc=checked_at_utc or utc_timestamp(),
    )


def build_benchmark_requests(
    config: BacktestConfig = DEFAULT_CONFIG,
) -> list[AcquisitionRequest]:
    """Return the single canonical Yahoo total-return benchmark request."""
    return [AcquisitionRequest(
        AcquisitionIdentity(
            scope=SCOPE,
            dataset="benchmark",
            asset_id="SP500TR",
            provider="yahoo",
            provider_symbol="^SP500TR",
        ),
        requested_start=config.benchmark.coverage_start_date.isoformat(),
        requested_end=config.benchmark.coverage_end_date.isoformat(),
    )]


def validate_supplied_reuters_prices(
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> None:
    """Validate the immutable first-priority Reuters artifact."""
    path = paths.prices_csv
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing supplied Reuters price baseline: {path}")
    validate_exact_manifest_catalog(
        paths.prices_artifact_manifest_csv,
        scope=SCOPE,
        dataset="prices",
        expected_origins={path.name: ArtifactOrigin.SUPPLIED},
        base_dir=path.parent,
    )


PRICE_REQUIREMENT_COLUMNS = (
    "Mapping_ID",
    "Required_Date",
    "Match_Kind",
    "Role",
)


def price_requirement_ledger(
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
    *,
    market_config: BacktestMarketConfig = DEFAULT_CONFIG.market,
    bundle: SecurityIdentityBundle | None = None,
) -> pd.DataFrame:
    """Project price needs without depending on prepared backtest artifacts."""
    dates = month_end_index(market_config.start_date, market_config.end_date)
    membership = validate_membership_sources(
        paths.membership,
        required_through=market_config.end_date,
    )
    active_months = build_active_months_by_ticker(membership, dates)
    if bundle is None:
        bundle = load_security_identity_bundle(
            paths.project_root,
            validate_manifest=True,
            require_all_approved=True,
        )
    mapping_methods = set(CONTRIBUTED_PRICE_RESOLUTION_METHODS) | {REUTERS_REMAP_METHOD}
    price_mappings = bundle.provider_mappings.loc[
        bundle.provider_mappings["Scope"].eq(SCOPE)
        & bundle.provider_mappings["Resolution_Method"].isin(mapping_methods)
        & bundle.provider_mappings["Review_Status"].eq("approved")
    ].copy()
    rows: list[tuple[str, str, str, str]] = []

    def require(
        mapping_id: object,
        date: object,
        match_kind: str,
        role: str,
    ) -> None:
        rows.append((
            str(mapping_id),
            pd.Timestamp(date).strftime("%Y-%m-%d"),
            match_kind,
            role,
        ))

    for mapping in price_mappings.sort_values(
        "Mapping_ID", kind="stable"
    ).to_dict("records"):
        mapping_id = str(mapping["Mapping_ID"])
        source_ticker = str(mapping["Source_Ticker"])
        mapping_identity = ProviderIdentity.from_mapping(mapping)
        required_dates = [
            pd.Timestamp(date)
            for date in dates
            if date.to_period("M") in active_months.get(source_ticker, set())
            and mapping_identity.is_effective(date)
        ]
        if not required_dates:
            continue
        for date in required_dates:
            require(mapping_id, date, "month", "monthly_valuation")

    events = bundle.events.set_index("Event_ID")
    valuation_legs = bundle.legs.loc[
        bundle.legs["Leg_Type"].isin(["relabel", "stock", "distribution"])
        & bundle.legs["Tradable"].eq("True")
        & bundle.legs["Review_Status"].eq("approved")
        & bundle.legs["To_Ticker"].ne(""),
        ["Event_ID", "From_Ticker", "To_Ticker"],
    ].drop_duplicates()
    for leg in valuation_legs.to_dict("records"):
        event_id = str(leg["Event_ID"])
        event = events.loc[event_id]
        if (
            str(event["Accounting_Status"]) != "executable"
            or str(event["Review_Status"]) != "approved"
        ):
            continue
        effective_date = pd.Timestamp(event["Effective_Date"])
        previous_dates = dates[dates < effective_date]
        following_dates = dates[dates >= effective_date]
        if not len(previous_dates) or not len(following_dates):
            continue
        previous_date = pd.Timestamp(previous_dates[-1])
        from_ticker = str(leg["From_Ticker"])
        if previous_date.to_period("M") not in active_months.get(
            from_ticker, set()
        ):
            continue
        valuation_date = pd.Timestamp(following_dates[0])
        matches = price_mappings.loc[
            price_mappings["Source_Ticker"].eq(str(leg["To_Ticker"]))
        ]
        if matches.empty:
            continue
        matches = matches.loc[
            [
                ProviderIdentity.from_mapping(mapping).is_effective(
                    valuation_date
                )
                for mapping in matches.to_dict("records")
            ]
        ]
        if matches.empty:
            continue
        if len(matches) != 1:
            raise ValueError(
                "Event valuation has ambiguous provider identity for "
                f"{event_id}/{leg['To_Ticker']} on {valuation_date.date()}"
            )
        mapping = matches.iloc[0]
        require(mapping["Mapping_ID"], valuation_date, "month", "event_valuation")

    for requirement in yahoo_normalization_requirements(bundle):
        require(
            requirement["Mapping_ID"],
            requirement["Treatment_Date"],
            "exact",
            "normalization",
        )

    deliveries = bundle.legs.loc[
        bundle.legs["Review_Status"].eq("approved")
        & bundle.legs["Leg_Type"].eq("distribution")
        & bundle.legs["Tradable"].eq("True")
        & bundle.legs["To_Ticker"].ne(""),
        ["Event_ID", "To_Ticker"],
    ].drop_duplicates()
    execution_mappings = bundle.provider_mappings.loc[
        bundle.provider_mappings["Scope"].eq(SCOPE)
        & bundle.provider_mappings["Resolution_Method"].eq(
            "reviewed_effective_symbol"
        )
        & bundle.provider_mappings["Review_Status"].eq("approved")
    ]
    for delivery in deliveries.to_dict("records"):
        event_id = str(delivery["Event_ID"])
        source_ticker = str(delivery["To_Ticker"])
        matches = execution_mappings.loc[
            execution_mappings["Event_ID"].eq(event_id)
            & execution_mappings["Source_Ticker"].eq(source_ticker)
        ]
        if matches.empty:
            continue
        if len(matches) != 1:
            raise ValueError(
                "Event execution must resolve exactly one provider mapping for "
                f"{event_id}/{source_ticker}"
            )
        event = events.loc[event_id]
        if str(event["Review_Status"]) != "approved":
            raise ValueError(f"Event execution lacks approved event {event_id}")
        mapping = matches.iloc[0]
        bound = max(
            pd.Timestamp(event["Effective_Date"]),
            pd.Timestamp(mapping["Effective_Start"]),
        )
        effective_end = str(mapping["Effective_End"])
        if effective_end and pd.Timestamp(effective_end) <= bound:
            raise ValueError(
                f"Event execution has an empty interval for {event_id}/{source_ticker}"
            )
        require(mapping["Mapping_ID"], bound, "exact", "event_execution")

    ledger = pd.DataFrame(rows, columns=PRICE_REQUIREMENT_COLUMNS)
    if ledger.empty:
        raise ValueError("Price requirement ledger must not be empty")
    if ledger.duplicated(list(PRICE_REQUIREMENT_COLUMNS)).any():
        raise ValueError("Price requirement ledger contains duplicate requirements")
    return ledger.sort_values(
        ["Mapping_ID", "Role", "Required_Date", "Match_Kind"],
        kind="stable",
    ).reset_index(drop=True)


def build_price_requests(
    ledger: pd.DataFrame,
    mappings: pd.DataFrame,
) -> list[AcquisitionRequest]:
    """Build one request per required Yahoo mapping plus the pinned WIKI object."""

    by_id = mappings.set_index("Mapping_ID", drop=False)
    if by_id.index.duplicated().any():
        raise ValueError("Provider mappings contain duplicate Mapping_ID values")
    unknown = sorted(set(ledger["Mapping_ID"]) - set(by_id.index))
    if unknown:
        raise ValueError(f"Price requirements reference unknown mappings: {unknown}")

    requests: list[AcquisitionRequest] = []
    for mapping_id, requirements in ledger.groupby("Mapping_ID", sort=True):
        mapping = by_id.loc[str(mapping_id)]
        if str(mapping["Provider"]) != "yahoo":
            continue
        effective_end = str(mapping["Effective_End"])
        requested_start = str(mapping["Effective_Start"])
        requested_end = (
            (pd.Timestamp(effective_end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if effective_end
            else max(requirements["Required_Date"])
        )
        requests.append(AcquisitionRequest(
            AcquisitionIdentity(
                scope=SCOPE,
                dataset="prices",
                asset_id=str(mapping["Asset_ID"]),
                provider="yahoo",
                provider_symbol=str(mapping["Provider_Symbol"]),
                effective_start=str(mapping["Effective_Start"]),
                effective_end=effective_end,
            ),
            requested_start=requested_start,
            requested_end=requested_end,
        ))

    required_mappings = by_id.loc[list(dict.fromkeys(ledger["Mapping_ID"]))]
    wiki = required_mappings.loc[required_mappings["Provider"].eq(WIKI_PROVIDER)]
    if not wiki.empty:
        requests.append(AcquisitionRequest(
            AcquisitionIdentity(
                scope=SCOPE,
                dataset="prices",
                asset_id="WIKI-MAPPED-EXTRACT",
                provider=WIKI_PROVIDER,
                provider_symbol=f"WIKI_PRICES.csv@{WIKI_PARENT_COMMIT[:8]}",
                effective_start=min(wiki["Effective_Start"]),
                effective_end=max(wiki["Effective_End"]),
            ),
            requested_start=min(wiki["Effective_Start"]),
            requested_end=(
                pd.Timestamp(max(wiki["Effective_End"])) - pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d"),
        ))
    return sorted(requests, key=lambda item: item.identity.key)


def price_readiness_records(
    ledger: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    mappings: pd.DataFrame,
    checked_at_utc: str | None = None,
) -> list[ReadinessRecord]:
    """Evaluate every consumer-derived price role from retained raw data."""
    observations = observations.loc[:, list(PRICE_OBSERVATION_COLUMNS)]
    observations = observations.loc[observations["Price_Close"].notna()].copy()
    observations["Observation_Date"] = pd.to_datetime(
        observations["Observation_Date"], errors="raise"
    )
    observations.loc[:, ["Provider", "Provider_Symbol", "Asset_ID", "Mapping_ID"]] = (
        observations.loc[
            :, ["Provider", "Provider_Symbol", "Asset_ID", "Mapping_ID"]
        ].astype(str)
    )
    by_id = mappings.set_index("Mapping_ID", drop=False)
    if by_id.index.duplicated().any():
        raise ValueError("Provider mappings contain duplicate Mapping_ID values")
    checked = checked_at_utc or utc_timestamp()
    records: list[ReadinessRecord] = []
    for (mapping_id, role), requirements in ledger.groupby(
        ["Mapping_ID", "Role"], sort=True
    ):
        mapping_id = str(mapping_id)
        if mapping_id not in by_id.index:
            raise ValueError(
                f"Price requirements reference unknown mapping {mapping_id}"
            )
        mapping = by_id.loc[mapping_id]
        effective_identity = ProviderIdentity.from_mapping(mapping)
        source = observations.loc[
            observations["Provider"].eq(mapping["Provider"])
            & observations["Provider_Symbol"].eq(mapping["Provider_Symbol"])
            & observations["Asset_ID"].eq(mapping["Asset_ID"])
        ]
        if str(mapping["Provider"]) != "reuters":
            source = source.loc[
                source["Mapping_ID"].eq(mapping["Mapping_ID"])
            ]
        source = source.loc[
            source["Observation_Date"].map(effective_identity.is_effective)
        ]
        exact = set(source["Observation_Date"].dt.strftime("%Y-%m-%d"))
        months = set(source["Month_End"].dt.strftime("%Y-%m-%d"))
        missing: list[str] = []
        for row in requirements.to_dict("records"):
            required_date = str(row["Required_Date"])
            if row["Match_Kind"] == "month":
                covered_row = required_date in months
            elif row["Match_Kind"] == "exact":
                covered_row = required_date in exact
            else:
                raise ValueError(f"Unknown price match kind {row['Match_Kind']!r}")
            if not covered_row:
                missing.append(required_date)
        required = len(requirements)
        covered = required - len(missing)
        records.append(ReadinessRecord(
            scope=SCOPE,
            dataset="prices",
            asset_id=str(mapping["Asset_ID"]),
            requirement_set=f"{role}::{mapping_id}",
            required_count=required,
            covered_count=covered,
            status=readiness_status(covered, required),
            missing_dates=tuple(missing),
            contributing_sources=(str(mapping["Provider"]),) if covered else (),
            checked_at_utc=checked,
        ))
    return sorted(records, key=lambda item: (item.asset_id, item.requirement_set))


def validate_price_readiness_records(
    ledger: pd.DataFrame,
    observations: pd.DataFrame,
    records: Sequence[ReadinessRecord],
    *,
    mappings: pd.DataFrame,
) -> None:
    """Require persisted readiness to equal identity-aware retained coverage."""

    records = tuple(records)
    expected = price_readiness_records(
        ledger,
        observations,
        mappings=mappings,
        checked_at_utc=(
            records[0].checked_at_utc
            if records
            else "1970-01-01T00:00:00Z"
        ),
    )

    def comparable(values: Sequence[ReadinessRecord]) -> list[tuple[object, ...]]:
        return sorted(
            tuple(
                value
                for key, value in item.to_row().items()
                if key != "Checked_At_UTC"
            )
            for item in values
        )

    if comparable(records) != comparable(expected):
        raise ValueError(
            "Fallback-price readiness does not exactly match the current "
            "identity-aware requirement ledger; rerun `python -m "
            "data_acquisition.acquire backtest prices`"
        )


def backtest_sector_requirements() -> pd.DataFrame:
    """Return exact full-RIC period-start sector requirements."""
    config = DEFAULT_CONFIG
    from .sector_requirements import build_active_sector_requirements

    requirements = build_active_sector_requirements(
        config.market,
        config.paths,
    )
    return project_sector_acquisition_requirements(requirements, scope=SCOPE)


__all__ = [
    "SCOPE",
    "SharesSourcePlan",
    "YahooSharesIdentity",
    "benchmark_readiness_record",
    "build_benchmark_requests",
    "build_price_requests",
    "build_shares_requests",
    "get_share_month_end_dates",
    "load_fresh_core_data",
    "plan_share_sources",
    "shares_readiness_records",
    "price_readiness_records",
    "validate_price_readiness_records",
    "validate_supplied_reuters_prices",
    "backtest_sector_requirements",
]
