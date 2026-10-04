"""Focused offline contracts for the canonical mixed backtest price basis."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtest.acquisition_planning import (
    build_price_requests,
    price_requirement_ledger,
    price_readiness_records,
    validate_price_readiness_records,
)
from backtest.acquisition_execution import _replace_yahoo_price_payload
from backtest.config import DEFAULT_CONFIG
from backtest.price_sources import (
    WIKI_COLUMNS,
    WIKI_PARENT_BYTES,
    WIKI_PARENT_COMMIT,
    WIKI_PARENT_SHA256,
    WIKI_PARENT_URL,
    YAHOO_RAW_COLUMNS,
    PRICE_OBSERVATION_COLUMNS,
    build_price_observation_catalog,
    build_mixed_monthly_prices,
    load_price_observation_catalog,
    validate_wiki_extract_rows,
    validate_yahoo_close_rows,
    wiki_mapping_rows,
)
from backtest.wiki_prices import (
    verify_and_extract_pinned_wiki_parent,
)
from data_acquisition.contracts import (
    AcquisitionIdentity,
    AcquisitionRequest,
    ReadinessRecord,
    ReadinessStatus,
)
from data_acquisition.providers.yahoo import fetch_yahoo_unadjusted_close_once
from portfolio_core.artifacts import file_sha256
from portfolio_core.security_identity import load_security_identity_bundle


def _mapping(**overrides) -> dict[str, str]:
    row = {
        "Mapping_ID": "MAP-TEST-YAHOO-AAA",
        "Scope": "backtest",
        "Provider": "yahoo",
        "Provider_Symbol": "AAA",
        "Source_Ticker": "AAA",
        "Asset_ID": "AAA",
        "Effective_Start": "2014-01-01",
        "Effective_End": "2014-03-01",
        "Resolution_Method": "reviewed_yahoo_close_fallback",
        "Event_ID": "",
        "Review_Status": "approved",
        "Local_First_Date": "2014-01-31",
        "Local_Last_Date": "2014-01-31",
    }
    row.update(overrides)
    return row


def _yahoo_row(date: str = "2014-01-31", **overrides) -> dict[str, object]:
    row = {
        "Date": date,
        "Asset_ID": "AAA",
        "Source_Ticker": "AAA",
        "Provider_Symbol": "AAA",
        "Mapping_ID": "MAP-TEST-YAHOO-AAA",
        "Close": 10.0,
        "Volume": 100.0,
        "Dividends": 0.0,
        "Stock_Splits": 0.0,
        "Capital_Gains": 0.0,
    }
    row.update(overrides)
    return row


def _mixed_prices(
    reuters: pd.DataFrame,
    *,
    mappings: pd.DataFrame,
    yahoo_raw: pd.DataFrame | None = None,
    wiki_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    observations = build_price_observation_catalog(
        reuters,
        mappings=mappings,
        yahoo_raw=yahoo_raw,
        wiki_raw=wiki_raw,
    )
    return build_mixed_monthly_prices(observations, mappings=mappings)


def _readiness_requirement(**overrides) -> pd.DataFrame:
    row = {
        "Mapping_ID": "MAP-CURRENT",
        "Required_Date": "2014-01-31",
        "Match_Kind": "exact",
        "Role": "event_execution",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _price_observations(**overrides) -> pd.DataFrame:
    row = {
        "Observation_Date": "2014-01-31",
        "Month_End": pd.Timestamp("2014-01-31"),
        "Provider": "yahoo",
        "Provider_Symbol": "AAA",
        "Asset_ID": "AAA",
        "Mapping_ID": "MAP-CURRENT",
        "Price_Close": 10.0,
        "Volume": 100.0,
    }
    row.update(overrides)
    return pd.DataFrame([row], columns=PRICE_OBSERVATION_COLUMNS)


def test_yahoo_unadjusted_close_disables_adjustment_and_clips_bounds():
    calls = []
    payload = pd.DataFrame(
        {
            "Close": [9.0, 10.0, 11.0, 12.0],
            "Volume": [90.0, 100.0, 110.0, 120.0],
            "Adj Close": [1.0, 1.0, 1.0, 1.0],
            "Dividends": [0.0, 0.0, 0.5, 0.0],
            "Stock Splits": [0.0, 0.0, 1.196, 0.0],
        },
        index=pd.to_datetime([
            "2013-12-31", "2014-01-01", "2014-01-31", "2014-02-01"
        ]),
    )
    request = AcquisitionRequest(
        AcquisitionIdentity(
            scope="backtest",
            dataset="prices",
            asset_id="AAA",
            provider="yahoo",
            provider_symbol="AAA",
        ),
        requested_start="2014-01-01",
        requested_end="2014-01-31",
    )

    result = fetch_yahoo_unadjusted_close_once(
        request,
        downloader=lambda **kwargs: calls.append(kwargs) or payload,
    )

    assert calls == [{
        "tickers": "AAA",
        "auto_adjust": False,
        "actions": True,
        "progress": False,
        "threads": False,
        "timeout": 30.0,
        "start": "2014-01-01",
        "end": "2014-02-01",
    }]
    assert list(result.payload.columns) == [
        "Close", "Volume", "Dividends", "Stock Splits", "Capital Gains"
    ]
    assert list(result.payload.index.strftime("%Y-%m-%d")) == [
        "2014-01-01", "2014-01-31"
    ]
    assert "Adj Close" not in result.payload
    assert result.payload["Capital Gains"].eq(0.0).all()
    assert result.payload.loc[pd.Timestamp("2014-01-31"), "Close"] == 11.0


def test_reuters_wins_over_yahoo_for_the_same_asset_month():
    mappings = pd.DataFrame([_mapping()])
    reuters = pd.DataFrame([{
        "Date": "2014-01-31",
        "RIC": "AAA",
        "Price Close": 20.0,
        "Volume": 200.0,
    }])
    yahoo = pd.DataFrame([_yahoo_row()])

    prices = _mixed_prices(
        reuters,
        mappings=mappings,
        yahoo_raw=yahoo,
    )

    assert prices.to_dict("records") == [{
        "Date": pd.Timestamp("2014-01-31"),
        "Asset_ID": "AAA",
        "Price_Close": 20.0,
        "Volume": 200.0,
    }]


def test_valid_yahoo_fallback_wins_when_latest_reuters_close_is_missing():
    mappings = pd.DataFrame([_mapping()])
    reuters = pd.DataFrame([
        {
            "Date": "2014-01-15",
            "RIC": "AAA",
            "Price Close": 19.0,
            "Volume": 190.0,
        },
        {
            "Date": "2014-01-31",
            "RIC": "AAA",
            "Price Close": np.nan,
            "Volume": 200.0,
        },
    ])
    yahoo = pd.DataFrame([_yahoo_row()])

    prices = _mixed_prices(
        reuters,
        mappings=mappings,
        yahoo_raw=yahoo,
    )
    reversed_prices = _mixed_prices(
        reuters.iloc[::-1].reset_index(drop=True),
        mappings=mappings,
        yahoo_raw=yahoo,
    )

    assert prices.to_dict("records") == [{
        "Date": pd.Timestamp("2014-01-31"),
        "Asset_ID": "AAA",
        "Price_Close": 10.0,
        "Volume": 100.0,
    }]
    pd.testing.assert_frame_equal(prices, reversed_prices)


def test_invalid_reuters_close_fails_closed_instead_of_using_fallback():
    mappings = pd.DataFrame([_mapping()])
    reuters = pd.DataFrame([{
        "Date": "2014-01-31",
        "RIC": "AAA",
        "Price Close": "not-a-price",
        "Volume": 200.0,
    }])
    yahoo = pd.DataFrame([_yahoo_row()])

    with pytest.raises(ValueError, match="invalid price/volume values"):
        _mixed_prices(
            reuters,
            mappings=mappings,
            yahoo_raw=yahoo,
        )


def test_conflicting_reuters_duplicate_winner_fails_closed():
    reuters = pd.DataFrame([
        {"Date": "2014-01-31", "RIC": "AAA", "Price Close": 20.0, "Volume": 1.0},
        {"Date": "2014-01-31", "RIC": "AAA", "Price Close": 21.0, "Volume": 1.0},
    ])

    with pytest.raises(ValueError, match="conflicting duplicate"):
        _mixed_prices(
            reuters,
            mappings=pd.DataFrame([_mapping()]).iloc[0:0],
        )


def test_yahoo_effective_date_bound_rejects_ticker_reuse():
    mappings = pd.DataFrame([_mapping()])
    late = pd.DataFrame([_yahoo_row(date="2014-03-01")], columns=YAHOO_RAW_COLUMNS)

    with pytest.raises(ValueError, match="outside its canonical identity"):
        validate_yahoo_close_rows(late, mappings=mappings)


def test_wiki_uses_close_not_adjusted_close_in_tiny_fixture():
    mappings = pd.DataFrame([
        _mapping(
            Mapping_ID="MAP-TEST-WIKI-OLD",
            Scope="backtest",
            Provider="wiki",
            Provider_Symbol="OLD",
            Source_Ticker="OLD",
            Asset_ID="OLD",
            Effective_End="2014-02-01",
            Resolution_Method="reviewed_wiki_close_fallback",
            Local_First_Date="2014-01-31",
            Local_Last_Date="2014-01-31",
            Review_Status="approved",
        )
    ])
    row = {column: "0" for column in WIKI_COLUMNS}
    row.update({
        "ticker": "OLD",
        "date": "2014-01-31",
        "low": "9",
        "high": "11",
        "close": "10",
        "adj_close": "999",
    })
    prices = _mixed_prices(
        pd.DataFrame(columns=["Date", "RIC", "Price Close", "Volume"]),
        mappings=mappings,
        wiki_raw=pd.DataFrame([row], columns=WIKI_COLUMNS),
    )

    assert prices.to_dict("records") == [{
        "Date": pd.Timestamp("2014-01-31"),
        "Asset_ID": "OLD",
        "Price_Close": 10.0,
        "Volume": 0,
    }]


def test_only_pinned_wiki_parent_request_is_authorized(tmp_path):
    bundle = load_security_identity_bundle(DEFAULT_CONFIG.paths.project_root)
    ledger = price_requirement_ledger(bundle=bundle)
    requests = build_price_requests(ledger, bundle.provider_mappings)
    wiki_requests = [item for item in requests if item.identity.provider == "wiki"]

    assert len(wiki_requests) == 1
    request = wiki_requests[0]
    assert request.identity.asset_id == "WIKI-MAPPED-EXTRACT"
    assert request.identity.provider_symbol == "WIKI_PRICES.csv@aff4c4f3"
    assert WIKI_PARENT_COMMIT == "aff4c4f3b677b0434bfedbc12b4137facaf7a0bb"
    assert WIKI_PARENT_URL.endswith(
        f"/{WIKI_PARENT_COMMIT}/WIKI_PRICES.csv"
    )
    assert WIKI_PARENT_BYTES == 235_562_224
    assert WIKI_PARENT_SHA256 == (
        "dd5127aae478d270150904fcbad6e96a42e461e13c3d48a1587edb9b89cea43e"
    )
    fake_parent = tmp_path / "WIKI_PRICES.csv"
    fake_parent.write_text(",".join(WIKI_COLUMNS) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="byte-size mismatch"):
        verify_and_extract_pinned_wiki_parent(
            fake_parent,
            tmp_path / "extract.csv",
        )


def test_yahoo_checkpoint_replaces_the_requested_mapping():
    mappings = pd.DataFrame([_mapping(
        Effective_End="2014-02-01",
        Local_Last_Date="2014-01-31",
    )])
    raw = validate_yahoo_close_rows(
        pd.DataFrame([_yahoo_row()], columns=YAHOO_RAW_COLUMNS),
        mappings=mappings,
    )
    request = AcquisitionRequest(
        AcquisitionIdentity(
            scope="backtest",
            dataset="prices",
            asset_id="AAA",
            provider="yahoo",
            provider_symbol="AAA",
            effective_start="2014-01-01",
            effective_end="2014-02-01",
        ),
        requested_start="2014-01-01",
        requested_end="2014-01-31",
    )
    payload = pd.DataFrame(
        [[11.0, 101.0, 0.0, 0.0, 0.0]],
        columns=("Close", "Volume", "Dividends", "Stock Splits", "Capital Gains"),
        index=pd.DatetimeIndex(["2014-01-31"]),
    )
    updated = _replace_yahoo_price_payload(
        raw,
        request,
        payload,
        mappings=mappings,
    )

    assert updated["Date"].tolist() == [pd.Timestamp("2014-01-31")]
    assert updated.loc[0, "Close"] == 11.0


def test_pinned_wiki_extract_is_exact_bounded_and_deterministic():
    extract_path = DEFAULT_CONFIG.paths.price_sources.wiki_extract_csv
    extract = pd.read_csv(extract_path, keep_default_na=False, dtype=str)
    mappings = load_security_identity_bundle(
        DEFAULT_CONFIG.paths.project_root
    ).provider_mappings
    validate_wiki_extract_rows(extract, mappings=mappings)

    assert tuple(extract.columns) == WIKI_COLUMNS
    assert tuple(sorted(extract["ticker"].unique())) == tuple(
        sorted(wiki_mapping_rows(mappings)["Provider_Symbol"])
    )
    assert file_sha256(extract_path) == (
        "ac07e8cbfb05007765314ce0cf2ef9bb00c109cc2f7955b3f9b447cc71705bd7"
    )
    assert extract[["ticker", "date"]].to_dict("records") == (
        extract.sort_values(["ticker", "date"])[["ticker", "date"]]
        .to_dict("records")
    )


def test_consumer_derived_price_requirement_ledger_is_complete():
    bundle = load_security_identity_bundle(DEFAULT_CONFIG.paths.project_root)
    observations = load_price_observation_catalog(
        DEFAULT_CONFIG.paths,
        bundle=bundle,
    )
    ledger = price_requirement_ledger(
        bundle=bundle,
    )
    records = price_readiness_records(
        ledger=ledger,
        observations=observations,
        mappings=bundle.provider_mappings,
        checked_at_utc="2026-08-31T00:00:00Z",
    )
    providers = bundle.provider_mappings.set_index("Mapping_ID")["Provider"]
    expected = {
        f"{role}::{mapping_id}": (
            len(rows),
            str(providers.loc[mapping_id]),
        )
        for (mapping_id, role), rows in ledger.groupby(["Mapping_ID", "Role"])
    }
    by_set = {record.requirement_set: record for record in records}

    assert set(by_set) == set(expected)
    assert ledger["Mapping_ID"].ne("").all()
    for requirement_set, (required_count, provider) in expected.items():
        record = by_set[requirement_set]
        assert record.required_count == required_count
        assert record.covered_count == required_count
        assert record.status.value == "complete"
        assert record.missing_dates == ()
        assert record.contributing_sources == (provider,)


@pytest.mark.parametrize(
    ("observation_mapping", "effective_end"),
    (("MAP-LATER-REUSE", "2014-03-01"), ("MAP-CURRENT", "2014-02-01")),
)
def test_price_readiness_rejects_wrong_or_out_of_bounds_identity(
    observation_mapping,
    effective_end,
):
    ledger = _readiness_requirement()
    mappings = pd.DataFrame([_mapping(
        Mapping_ID="MAP-CURRENT",
        Effective_End=effective_end,
    )])
    observations = _price_observations(
        Observation_Date="2014-02-01",
        Month_End=pd.Timestamp("2014-02-28"),
        Mapping_ID=observation_mapping,
    )

    record = price_readiness_records(
        ledger,
        observations,
        mappings=mappings,
        checked_at_utc="2026-09-01T00:00:00Z",
    )[0]

    assert record.status is ReadinessStatus.MISSING
    assert record.missing_dates == ("2014-01-31",)


def test_exact_readiness_validation_rejects_false_zero_of_zero_complete():
    ledger = _readiness_requirement(
        Role="monthly_valuation",
        Match_Kind="month",
    )
    mappings = pd.DataFrame([_mapping(Mapping_ID="MAP-CURRENT")])
    observations = _price_observations()
    stale = ReadinessRecord(
        scope="backtest",
        dataset="prices",
        asset_id="AAA",
        requirement_set="monthly_valuation::MAP-CURRENT",
        required_count=0,
        covered_count=0,
        status=ReadinessStatus.COMPLETE,
        checked_at_utc="2026-09-01T00:00:00Z",
    )

    with pytest.raises(ValueError, match="does not exactly match"):
        validate_price_readiness_records(
            ledger,
            observations,
            [stale],
            mappings=mappings,
        )


def test_price_requirement_ledger_derives_extended_monthly_coverage():
    market = replace(DEFAULT_CONFIG.market, end_date=date(2026, 2, 28))

    ledger = price_requirement_ledger(market_config=market)

    sndk = ledger.loc[
        ledger["Mapping_ID"].eq("MAP-BACKTEST-REUTERS-SNDK-20250224")
        & ledger["Role"].eq("monthly_valuation")
    ]
    assert len(sndk) == 4
    assert sndk["Required_Date"].iloc[-1] == "2026-02-28"
