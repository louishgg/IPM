"""Resolve effective-dated backtest membership from raw provenance."""

from __future__ import annotations

import pandas as pd
from portfolio_core.dates import month_end_index
from portfolio_core.provider_identity import (
    BacktestProviderIdentityResolver,
    derive_premature_successor_rules,
    provider_identity_validation_audit,
)
from portfolio_core.security_identity import (
    UNAVAILABLE_PREFIX,
    load_security_identity_bundle,
)
from portfolio_core.sp500_membership import (
    membership_asof,
    validate_membership_sources,
)

from .config import DEFAULT_CONFIG, BacktestMarketConfig
from .paths import BacktestPaths
from .price_sources import load_verified_price_observations


def build_membership_resolution_outputs(
    market_config: BacktestMarketConfig = DEFAULT_CONFIG.market,
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resolve effective-dated membership without reading prepared artifacts."""
    dates = month_end_index(market_config.start_date, market_config.end_date)
    history = validate_membership_sources(
        paths.membership,
        required_through=market_config.end_date,
    )
    identity_bundle = load_security_identity_bundle(
        paths.project_root,
        validate_manifest=True,
        require_all_approved=True,
    )
    provider_observations = load_verified_price_observations(
        paths,
        bundle=identity_bundle,
    )
    resolver = BacktestProviderIdentityResolver(
        identity_bundle,
        provider_observations,
    )
    premature_successor_rules = derive_premature_successor_rules(identity_bundle)

    resolved_by_date: dict[pd.Timestamp, set[str]] = {}
    resolution_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    all_asset_ids: set[str] = set()

    for date in dates:
        membership_effective_date, members = membership_asof(history, date)
        excluded_successors = {
            source_ticker: premature_successor_rules[source_ticker]
            for source_ticker in members
            if source_ticker in premature_successor_rules
            and date < premature_successor_rules[source_ticker].event_date
        }
        resolved_assets: set[str] = set()
        asset_sources: dict[str, str] = {}
        unavailable_sources: list[str] = []
        for source_ticker, rule in sorted(excluded_successors.items()):
            resolution_rows.append({
                "Date": date,
                "FJA_Effective_Date": membership_effective_date,
                "Source_Ticker": source_ticker,
                "Asset_ID": "",
                "Price_RIC": "",
                "Resolution_Method": (
                    "evidence_backed_premature_successor_exclusion"
                ),
                "Is_Priced": False,
                "Eligibility_Status": "excluded_before_official_event",
                "Corporate_Event_Date": rule.event_date,
                "Primary_Source_URL": rule.primary_source_url,
            })
        for source_ticker in members:
            if source_ticker in excluded_successors:
                continue
            identity = resolver.resolve(source_ticker, date)
            price_by_asset = dict(zip(identity.asset_ids, identity.provider_symbols))
            for asset_id in identity.asset_ids:
                prior_source = asset_sources.get(asset_id)
                if prior_source is not None and prior_source != source_ticker:
                    raise ValueError(
                        "Simultaneously active membership tickers resolve to the same "
                        f"Asset_ID on {date.date()}: {prior_source}, "
                        f"{source_ticker} -> {asset_id}"
                    )
                asset_sources[asset_id] = source_ticker
                resolved_assets.add(asset_id)
                is_priced = not asset_id.startswith(UNAVAILABLE_PREFIX)
                if not is_priced:
                    unavailable_sources.append(source_ticker)
                resolution_rows.append({
                    "Date": date,
                    "FJA_Effective_Date": membership_effective_date,
                    "Source_Ticker": source_ticker,
                    "Asset_ID": asset_id,
                    "Price_RIC": price_by_asset.get(asset_id, ""),
                    "Resolution_Method": identity.resolution_method,
                    "Is_Priced": is_priced,
                    "Eligibility_Status": (
                        "eligible_priced" if is_priced else "eligible_unavailable"
                    ),
                    "Corporate_Event_Date": "",
                    "Primary_Source_URL": "",
                })

        if not resolved_assets:
            raise ValueError(
                f"membership history resolved to no assets on {date.date()}"
            )
        all_asset_ids.update(resolved_assets)
        resolved_by_date[date] = resolved_assets
        unavailable_sources = sorted(set(unavailable_sources))
        priced_count = sum(
            not asset.startswith(UNAVAILABLE_PREFIX)
            for asset in resolved_assets
        )
        reconciled_source_count = len(members) - len(excluded_successors)
        expansion_count = len(resolved_assets) - reconciled_source_count
        if expansion_count < 0:
            raise ValueError(
                f"Identity resolution lost membership members on {date.date()}"
            )
        audit_rows.append({
            "Date": date,
            "FJA_Effective_Date": membership_effective_date,
            "Source_Member_Count": len(members),
            "Resolved_Asset_Count": len(resolved_assets),
            "Priced_Asset_Count": priced_count,
            "Unavailable_Member_Count": len(unavailable_sources),
            "Evidence_Backed_Premature_Successor_Exclusion_Count": len(
                excluded_successors
            ),
            "Intentional_Collision_Expansion_Count": expansion_count,
            "Unavailable_Source_Tickers": ";".join(unavailable_sources),
            "Excluded_Premature_Successor_Tickers": ";".join(
                sorted(excluded_successors)
            ),
            "Source_Row_Reconciled": True,
        })

    asset_ids = sorted(all_asset_ids)
    pit = pd.DataFrame(False, index=dates, columns=asset_ids, dtype=bool)
    for date, resolved_assets in resolved_by_date.items():
        pit.loc[date, sorted(resolved_assets)] = True
    pit.index.name = "Date"
    pit.columns.name = "Asset_ID"

    resolution = pd.DataFrame(resolution_rows).sort_values(
        ["Date", "Source_Ticker", "Asset_ID"], kind="stable"
    ).reset_index(drop=True)
    coverage_audit = pd.DataFrame(audit_rows)
    evidence_validation = provider_identity_validation_audit(
        identity_bundle,
        provider_observations,
    )
    return pit, asset_ids, resolution, coverage_audit, evidence_validation


__all__ = ["build_membership_resolution_outputs"]
