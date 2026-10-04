"""Focused tests for the canonical membership/provider identity boundary."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from portfolio_core.provider_identity import (
    BacktestProviderIdentityResolver,
    IDENTITY_VALIDATION_COLUMNS,
    ProviderIdentityError,
    YahooIdentityResolver,
    derive_premature_successor_rules,
    normalize_identity_key,
    provider_identity_validation_audit,
)
from backtest.config import DEFAULT_CONFIG
from backtest.price_sources import load_verified_price_observations
from portfolio_core.security_identity import (
    PROVIDER_SYMBOL_MAPPINGS_COLUMNS,
    load_security_identity_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
PRICES = (
    ROOT
    / "data"
    / "shared"
    / "supplied"
    / "reuters"
    / "SP500_Full_2014_2026_Cleaned.csv"
)


def _prices(rows):
    return pd.DataFrame(rows, columns=["Date", "RIC", "Price Close"])


def _mapping(
    source,
    ric,
    start,
    end="",
    *,
    method="official_same_security_local_ric",
    first="2020-01-31",
    last="2020-03-31",
    conflict=False,
    review="approved",
    proof="local_reuters_observation",
):
    values = {column: "" for column in PROVIDER_SYMBOL_MAPPINGS_COLUMNS}
    values.update({
        "Mapping_ID": (
            f"MAP-BACKTEST-REUTERS-{source}-{start}-{ric or 'MISSING'}"
            .replace(".", "-")
        ),
        "Scope": "backtest",
        "Provider": "reuters",
        "Source_Ticker": source,
        "Provider_Symbol": ric,
        "Asset_ID": (
            f"FJA_UNAVAILABLE::{source}"
            if method == "reviewed_missing_price"
            else ric
        ),
        "Effective_Start": start,
        "Effective_End": end,
        "Resolution_Method": method,
        "Local_First_Date": first,
        "Local_Last_Date": last,
        "Simultaneous_Symbol_Conflict": str(conflict),
        "Provider_Evidence_Status": proof,
        "Review_Status": review,
    })
    return values


def _bundle(*rows):
    canonical = load_security_identity_bundle(ROOT)
    mappings = pd.DataFrame(rows, columns=PROVIDER_SYMBOL_MAPPINGS_COLUMNS)
    return replace(canonical, provider_mappings=mappings)


def _reuters_resolver(bundle, prices):
    return BacktestProviderIdentityResolver(
        bundle,
        pd.DataFrame({
            "Provider": "reuters",
            "Provider_Symbol": prices["RIC"],
        }),
    )


def test_normalization_handles_venue_delisting_and_share_class_syntax():
    assert normalize_identity_key("ABC.O^A20") == "ABC"
    assert normalize_identity_key("BRK.B") == normalize_identity_key("BRKb")
    assert normalize_identity_key("BF.B") == normalize_identity_key("BFb")


def test_automatic_resolution_requires_one_candidate():
    prices = _prices([
        ("2020-01-31", "AAA.O", 10.0),
        ("2020-02-29", "AAA.O", 11.0),
        ("2020-03-31", "AAA.O", 12.0),
        ("2020-01-31", "OLD.O", 20.0),
        ("2020-02-29", "OLD.O", 21.0),
        ("2020-03-31", "OLD.O", 22.0),
    ])
    resolver = _reuters_resolver(
        _bundle(_mapping("OLD", "OLD.O", "2020-01-01")),
        prices,
    )

    result = resolver.resolve(
        "AAA",
        "2020-03-31",
    )
    assert result.asset_ids == ("AAA.O",)
    assert result.resolution_method == "automatic_unique_ric"

    ambiguous_prices = pd.concat([
        prices,
        _prices([
            ("2020-01-31", "AAA.K", 30.0),
            ("2020-02-29", "AAA.K", 31.0),
            ("2020-03-31", "AAA.K", 32.0),
        ]),
    ], ignore_index=True)
    ambiguous = _reuters_resolver(
        _bundle(_mapping("OLD", "OLD.O", "2020-01-01")),
        ambiguous_prices,
    )
    with pytest.raises(ProviderIdentityError, match="exactly one"):
        ambiguous.resolve(
            "AAA",
            "2020-03-31",
        )


def test_effective_dated_collision_expands_only_before_boundary():
    prices = _prices([
        ("2020-01-31", "NEW.O", 10.0),
        ("2020-02-29", "NEW.O", 11.0),
        ("2020-03-31", "NEW.O", 12.0),
        ("2020-01-31", "OLD.O^C20", 20.0),
        ("2020-02-29", "OLD.O^C20", 21.0),
    ])
    resolver = _reuters_resolver(
        _bundle(
            _mapping("COLLIDE", "NEW.O", "2020-01-01", conflict=True),
            _mapping(
                "COLLIDE",
                "OLD.O^C20",
                "2020-01-01",
                "2020-03-01",
                first="2020-01-31",
                last="2020-02-29",
                conflict=True,
            ),
        ),
        prices,
    )
    before = resolver.resolve("COLLIDE", "2020-02-29")
    after = resolver.resolve("COLLIDE", "2020-03-31")
    assert before.asset_ids == ("NEW.O", "OLD.O^C20")
    assert after.asset_ids == ("NEW.O",)


def test_unresolved_review_and_unreviewed_interval_gap_fail_closed():
    prices = _prices([
        ("2020-01-31", "NEW.O", 10.0),
        ("2020-02-29", "NEW.O", 11.0),
        ("2020-03-31", "NEW.O", 12.0),
    ])
    with pytest.raises(ProviderIdentityError, match="Unapproved"):
        _reuters_resolver(
            _bundle(
                _mapping(
                    "OLD",
                    "NEW.O",
                    "2020-01-01",
                    review="unresolved",
                )
            ),
            prices,
        )

    resolver = _reuters_resolver(
        _bundle(_mapping("OLD", "NEW.O", "2020-01-01", "2020-03-01")),
        prices,
    )
    with pytest.raises(ProviderIdentityError, match="no mapping active"):
        resolver.resolve(
            "OLD",
            "2020-03-31",
        )


def test_distinct_local_ric_resolves_before_a_later_successor_mapping():
    resolver = _reuters_resolver(
        _bundle(_mapping("OLD", "NEW.O", "2020-03-01")),
        _prices([("2020-01-31", "OLD.O", 10.0)]),
    )

    result = resolver.resolve("OLD", "2020-01-31")

    assert result.asset_ids == ("OLD.O",)
    assert result.resolution_method == "automatic_unique_ric"


def test_reviewed_missing_price_resolves_to_explicit_placeholder():
    prices = _prices([("2020-01-31", "OTHER.O", 10.0)])
    resolver = _reuters_resolver(
        _bundle(
            _mapping(
                "MISSING",
                "",
                "2020-01-01",
                method="reviewed_missing_price",
                first="",
                last="",
            )
        ),
        prices,
    )
    result = resolver.resolve(
        "MISSING",
        "2020-01-31",
    )
    assert result.asset_ids == ("FJA_UNAVAILABLE::MISSING",)
    assert result.provider_symbols == ()


def test_real_bundle_is_fully_approved_and_locally_valid():
    prices = pd.read_csv(PRICES, low_memory=False)
    bundle = load_security_identity_bundle(ROOT)
    observations = load_verified_price_observations(
        DEFAULT_CONFIG.paths,
        bundle=bundle,
    )
    audit = provider_identity_validation_audit(bundle, observations)

    assert tuple(audit.columns) == IDENTITY_VALIDATION_COLUMNS
    assert audit["Validation_Status"].eq("approved_offline").all()
    assert (
        audit[["Observed_First_Date", "Observed_Last_Date"]]
        .ne("")
        .all()
        .all()
    )
    fi = audit.loc[audit["Source_Ticker"].eq("FI")].iloc[0]
    assert fi["Provider_Symbol"] == "FISV.O"
    assert fi["Event_ID"] == "EVT-20230607-FISV-FI-IDENTITY-CONTINUITY"
    otc = bundle.provider_mappings["Resolution_Method"].eq(
        "official_otc_transition_local_ric"
    )
    local_rics = set(prices["RIC"].astype(str))
    assert bundle.provider_mappings.loc[otc, "Provider_Symbol"].isin(
        local_rics
    ).all()


def test_successor_rules_and_yahoo_purposes_come_from_canonical_records():
    bundle = load_security_identity_bundle(ROOT)
    rules = derive_premature_successor_rules(bundle)
    assert rules["CCEP"].event_date == pd.Timestamp("2016-05-28")
    assert rules["CPRI"].event_date == pd.Timestamp("2019-01-02")
    assert "DD" not in rules

    yahoo = YahooIdentityResolver(bundle, scope="live")
    historical = yahoo.resolve(
        "BK",
        as_of="2026-05-06",
        purpose="historical_prices",
    )
    effective = yahoo.resolve(
        "BK",
        as_of="2026-05-06",
        purpose="effective_security",
    )
    assert historical.mapping_id == effective.mapping_id
    assert historical.provider_symbol == "BNY"
    assert effective.provider_symbol == "BK"
    assert effective.effective_end == pd.Timestamp("2026-05-21")
