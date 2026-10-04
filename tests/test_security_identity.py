"""Fail-closed tests for canonical security identity provenance."""

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest

from portfolio_core.security_identity import (
    SecurityIdentityError,
    load_security_identity_bundle,
    parse_exact_decimal,
    parse_exact_ratio,
    validate_security_identity_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = PROJECT_ROOT / "data/shared/provenance/security_identity"


def _bundle():
    return load_security_identity_bundle(PROJECT_ROOT)


def test_unsupported_normalization_provider_is_rejected():
    bundle = _bundle()
    legs = bundle.legs.copy()
    legs.loc[
        legs["Event_Leg_ID"].eq("LEG-20160201-BRCM-AVGO-STOCK-01"),
        "Normalization_Provider",
    ] = "synthetic_provider"

    with pytest.raises(
        SecurityIdentityError,
        match="unsupported normalization provider 'synthetic_provider'",
    ):
        validate_security_identity_bundle(replace(bundle, legs=legs))


def test_manual_live_events_have_exact_structured_terms_and_sources():
    bundle = _bundle()
    events = bundle.events.set_index("Event_ID")
    legs = bundle.legs

    assert events.loc[
        "EVT-20260521-BK-BNY-TICKER-CHANGE", "Continuity_Class"
    ] == "same_security"
    sats_id = "EVT-20260624-SATS-ECHO-TICKER-CHANGE"
    assert events.loc[sats_id, "Event_Type"] == "identity_continuity"
    assert events.loc[sats_id, "Continuity_Class"] == "same_security"
    sats = legs.loc[legs["Event_ID"].eq(sats_id)].iloc[0]
    assert (
        sats["From_Ticker"],
        sats["To_Ticker"],
        sats["Leg_Type"],
        sats["Share_Ratio"],
        sats["Tradable"],
    ) == ("SATS", "ECHO", "relabel", "1", "True")
    assert events.loc[
        "EVT-20260507-CTRA-DVN-STOCK-EXCHANGE", "Event_Type"
    ] == "stock_exchange"
    ctra = legs.loc[
        legs["Event_ID"].eq("EVT-20260507-CTRA-DVN-STOCK-EXCHANGE")
    ].iloc[0]
    assert (ctra["From_Ticker"], ctra["To_Ticker"], ctra["Share_Ratio"]) == (
        "CTRA",
        "DVN",
        "0.70",
    )

    day = legs.loc[
        legs["Event_ID"].eq("EVT-20260204-DAY-CASH-ACQUISITION")
    ].iloc[0]
    assert (day["Leg_Type"], day["Cash_Amount"], day["Currency"]) == (
        "cash",
        "70",
        "USD",
    )

    holx = legs.loc[
        legs["Event_ID"].eq("EVT-20260407-HOLX-CASH-CVR-ACQUISITION")
    ].set_index("Leg_Type")
    assert events.loc[
        "EVT-20260407-HOLX-CASH-CVR-ACQUISITION", "Event_Type"
    ] == "cash_settlement"
    assert holx.loc["cash", "Cash_Amount"] == "76"
    assert holx.loc["cvr", "CVR_Units"] == "1"
    assert holx.loc["cvr", "CVR_Base_Value_Per_Unit"] == "0"
    assert holx.loc["cvr", "CVR_Max_Value_Per_Unit"] == "3"
    assert holx.loc["cvr", "Currency"] == "USD"
    assert set(legs["Source_ID"]).issubset(set(bundle.sources["Source_ID"]))

    bk_source = bundle.sources.loc[
        bundle.sources["Event_ID"].eq(
            "EVT-20260521-BK-BNY-TICKER-CHANGE"
        )
    ].iloc[0]
    assert bk_source["Document_Date"] == "2026-05-11"
    sats_source = bundle.sources.loc[
        bundle.sources["Event_ID"].eq(sats_id)
    ].iloc[0]
    assert sats_source["Publisher"] == "EchoStar"
    assert sats_source["Document_Date"] == "2026-06-22"
    assert sats_source["Source_URL"].startswith("https://ir.echostar.com/")


def test_extension_period_transitions_keep_reviewed_accounting_statuses():
    bundle = _bundle()
    events = bundle.events.set_index("Event_ID")

    assert events.loc[
        [
            "EVT-20240201-CDAY-DAY-IDENTITY-CONTINUITY",
            "EVT-20240301-PEAK-DOC-IDENTITY-CONTINUITY",
            "EVT-20240325-FLT-CPAY-IDENTITY-CONTINUITY",
            "EVT-20260114-MMC-MRSH-IDENTITY-CONTINUITY",
        ],
        "Accounting_Status",
    ].eq("executable").all()
    assert events.loc[
        "EVT-20250807-PARA-PSKY-STOCK-EXCHANGE",
        "Accounting_Status",
    ] == "executable"


def test_wba_event_has_exact_cash_dap_terms_and_primary_evidence():
    bundle = _bundle()
    event_id = "EVT-20250828-WBA-CASH-DAP-ACQUISITION"
    event = bundle.events.set_index("Event_ID").loc[event_id]
    legs = bundle.legs.loc[bundle.legs["Event_ID"].eq(event_id)].set_index(
        "Leg_Type"
    )
    sources = bundle.sources.loc[bundle.sources["Event_ID"].eq(event_id)]

    assert event["Effective_Date"] == "2025-08-28"
    assert event["Event_Type"] == "cash_settlement"
    assert event["Continuity_Class"] == "predecessor_extinguished"
    assert event["Accounting_Status"] == "executable"
    assert set(legs.index) == {"cash", "cvr"}
    assert legs.loc["cash", "From_Ticker"] == "WBA"
    assert legs.loc["cash", "Cash_Amount"] == "11.45"
    assert legs.loc["cash", "Currency"] == "USD"
    assert legs.loc["cvr", "To_Ticker"] == "WBA-DAP"
    assert legs.loc["cvr", "CVR_Units"] == "1"
    assert legs.loc["cvr", "CVR_Base_Value_Per_Unit"] == "0"
    assert legs.loc["cvr", "CVR_Max_Value_Per_Unit"] == "3"
    assert legs.loc["cvr", "Tradable"] == "False"
    assert set(sources["Source_Role"]) == {"primary", "supplemental"}
    primary = sources.loc[sources["Source_Role"].eq("primary")].iloc[0]
    assert primary["Publisher"] == "U.S. Securities and Exchange Commission"
    assert primary["Document_Date"] == "2025-08-28"
    assert primary["Source_URL"] == (
        "https://www.sec.gov/Archives/edgar/data/1618921/"
        "000119312525190603/d87240dex991.htm"
    )


def test_lsi_wiki_mapping_is_bound_to_sec_validated_corporation_identity():
    bundle = _bundle()
    mapping = bundle.provider_mappings.loc[
        bundle.provider_mappings["Mapping_ID"].eq(
            "MAP-BACKTEST-WIKI-LSI-20140101"
        )
    ].iloc[0]
    source = bundle.sources.loc[
        bundle.sources["Source_ID"].eq(
            "SRC-20140224-LSI-IDENTITY-SUPPLEMENTAL"
        )
    ].iloc[0]

    assert mapping["Provider_Symbol"] == "LSI"
    assert mapping["Asset_ID"] == "LSI"
    assert mapping["Effective_End"] == "2014-05-07"
    assert mapping["Event_ID"] == "EVT-20140506-LSI-CASH-ACQUISITION"
    assert mapping["Primary_Source_ID"] == source["Source_ID"]
    assert source["Publisher"] == "U.S. Securities and Exchange Commission"
    assert source["Document_Date"] == "2014-02-24"
    assert source["Source_URL"] == (
        "https://www.sec.gov/Archives/edgar/data/703360/"
        "000119312514069522/d628854d10k.htm"
    )
    assert "stock symbol LSI" in source["Evidence_Claim"]


@pytest.mark.parametrize(
    ("event_id", "date", "from_ticker", "to_ticker", "share_ratio"),
    [
        (
            "EVT-20220228-INFO-SPGI-STOCK-EXCHANGE",
            "2022-02-28",
            "INFO",
            "SPGI",
            "0.2838",
        ),
        (
            "EVT-20220401-PBCT-MTB-STOCK-EXCHANGE",
            "2022-04-01",
            "PBCT",
            "MTB",
            "0.118",
        ),
    ],
)
def test_stock_exchange_events_close_discovered_backtest_lifecycle_gaps(
    event_id,
    date,
    from_ticker,
    to_ticker,
    share_ratio,
):
    bundle = _bundle()
    event = bundle.events.set_index("Event_ID").loc[event_id]
    leg = bundle.legs.loc[bundle.legs["Event_ID"].eq(event_id)].iloc[0]
    source = bundle.sources.loc[
        bundle.sources["Event_ID"].eq(event_id)
    ].iloc[0]

    assert event["Effective_Date"] == date
    assert event["Event_Type"] == "stock_exchange"
    assert event["Continuity_Class"] == "predecessor_extinguished"
    assert (leg["From_Ticker"], leg["To_Ticker"], leg["Share_Ratio"]) == (
        from_ticker,
        to_ticker,
        share_ratio,
    )
    assert source["Document_Date"] == date
    assert source["Publisher"] == "U.S. Securities and Exchange Commission"


@pytest.mark.parametrize(
    ("event_id", "date", "from_ticker", "to_ticker", "leg_type", "term"),
    [
        (
            "EVT-20220930-CTXS-CASH-ACQUISITION",
            "2022-09-30",
            "CTXS",
            "",
            "cash",
            "104",
        ),
        (
            "EVT-20221003-DRE-PLD-STOCK-EXCHANGE",
            "2022-10-03",
            "DRE",
            "PLD",
            "stock",
            "0.475",
        ),
        (
            "EVT-20221027-TWTR-CASH-ACQUISITION",
            "2022-10-27",
            "TWTR",
            "",
            "cash",
            "54.20",
        ),
    ],
)
def test_discovered_2022_events_have_official_executable_terms(
    event_id,
    date,
    from_ticker,
    to_ticker,
    leg_type,
    term,
):
    bundle = _bundle()
    event = bundle.events.set_index("Event_ID").loc[event_id]
    leg = bundle.legs.loc[bundle.legs["Event_ID"].eq(event_id)].iloc[0]
    source = bundle.sources.loc[
        bundle.sources["Event_ID"].eq(event_id)
    ].iloc[0]

    assert event["Effective_Date"] == date
    assert event["Accounting_Status"] == "executable"
    assert (leg["From_Ticker"], leg["To_Ticker"], leg["Leg_Type"]) == (
        from_ticker,
        to_ticker,
        leg_type,
    )
    if leg_type == "cash":
        assert leg["Cash_Amount"] == term
        assert leg["Currency"] == "USD"
    else:
        assert leg["Share_Ratio"] == term
    assert source["Document_Date"] == date
    assert source["Publisher"] == "U.S. Securities and Exchange Commission"


def test_jci_tyco_and_utx_events_have_exact_holder_specific_terms():
    bundle = _bundle()

    jci_id = "EVT-20160906-JCI-TYC-JCI-CASH-AND-STOCK"
    jci = bundle.legs.loc[bundle.legs["Event_ID"].eq(jci_id)]
    assert set(zip(
        jci["From_Ticker"],
        jci["Leg_Type"],
        jci["Share_Ratio"],
        jci["Cash_Amount"],
        strict=True,
    )) == {
        ("TYC", "stock", "0.955", ""),
        ("JCI", "stock", "0.8357", ""),
        ("JCI", "cash", "", "5.7293"),
    }

    utx_id = "EVT-20200403-UTX-RTX-DISTRIBUTION"
    utx = bundle.legs.loc[bundle.legs["Event_ID"].eq(utx_id)]
    assert set(zip(
        utx["To_Ticker"],
        utx["Leg_Type"],
        utx["Share_Ratio"],
        strict=True,
    )) == {
        ("CARR", "distribution", "1"),
        ("OTIS", "distribution", "0.5"),
        ("RTX", "relabel", "1"),
    }
    supplemental = bundle.sources.loc[
        bundle.sources["Event_ID"].isin([jci_id, utx_id])
        & bundle.sources["Source_Role"].eq("supplemental")
    ]
    assert set(supplemental["Document_Date"]) == {"2016-09-06", "2020-04-03"}


def test_provider_mapping_authorizes_only_same_security_historical_aliases():
    bundle = _bundle()
    live = bundle.provider_mappings.loc[
        bundle.provider_mappings["Scope"].eq("live")
    ]
    bk = live.loc[live["Source_Ticker"].eq("BK")].iloc[0]
    sats = live.loc[live["Source_Ticker"].eq("SATS")].iloc[0]
    ctra = live.loc[live["Source_Ticker"].eq("CTRA")].iloc[0]
    assert (bk["Provider_Symbol"], bk["Resolution_Method"]) == (
        "BNY",
        "reviewed_yahoo_alias",
    )
    assert (sats["Provider_Symbol"], sats["Resolution_Method"]) == (
        "ECHO",
        "reviewed_yahoo_alias",
    )
    assert ctra["Provider_Symbol"] == "CTRA"
    assert not (
        live["Source_Ticker"].eq("CTRA") & live["Provider_Symbol"].eq("DVN")
    ).any()


def test_distribution_events_encode_atomic_predecessor_consumption():
    legs = _bundle().legs

    pure = legs.loc[
        legs["Event_ID"].eq("EVT-20190402-DWDP-DOW-DISTRIBUTION")
    ].iloc[0]
    assert pure["Leg_Type"] == "distribution"
    assert pure["Share_Ratio"] == "1/3"
    assert pure["Retain_Predecessor"] == "True"

    expected = {
        "EVT-20200401-ARNC-HWM-DISTRIBUTION": ("HWM", "ARNC", "1/4"),
        "EVT-20210803-LB-BBWI-DISTRIBUTION": (
            "BBWI",
            "VSCO",
            "1/3",
        ),
        "EVT-20221215-FBHS-FBIN-DISTRIBUTION": ("FBIN", "MBC", "1"),
    }
    for event_id, (survivor, child, ratio) in expected.items():
        event_legs = legs.loc[legs["Event_ID"].eq(event_id)].set_index("Leg_Type")
        assert event_legs.loc["relabel", "To_Ticker"] == survivor
        assert event_legs.loc["distribution", "To_Ticker"] == child
        assert event_legs.loc["distribution", "Share_Ratio"] == ratio
        assert set(event_legs["Retain_Predecessor"]) == {"False"}


def test_exact_ratio_parser_preserves_fractional_terms():
    assert parse_exact_ratio("1/3", required=True) == Fraction(1, 3)
    assert parse_exact_ratio("0.70", required=True) == Fraction(7, 10)
    assert parse_exact_ratio("") is None
    with pytest.raises(SecurityIdentityError, match="positive N/D ratio"):
        parse_exact_ratio("1/0", required=True)


def test_exact_decimal_parser_preserves_optional_and_invalid_behavior():
    assert parse_exact_decimal("", field="Cash_Amount") is None
    assert parse_exact_decimal("0", field="Cash_Amount") == Decimal("0")
    assert parse_exact_decimal("76.500", field="Cash_Amount") == Decimal(
        "76.500"
    )
    with pytest.raises(
        SecurityIdentityError,
        match="must be a nonnegative plain decimal",
    ):
        parse_exact_decimal("1e3", field="Cash_Amount")


def test_validator_rejects_orphan_event_source_and_unstructured_execution():
    bundle = _bundle()
    mappings = bundle.provider_mappings.copy()
    mappings.loc[mappings.index[0], "Event_ID"] = "EVT-20990101-ORPHAN"
    with pytest.raises(SecurityIdentityError, match="orphan Event_ID"):
        validate_security_identity_bundle(
            replace(bundle, provider_mappings=mappings)
        )

    events = bundle.events.copy()
    event_id = events.loc[events["Accounting_Status"].eq("documented_not_executable")].iloc[0][
        "Event_ID"
    ]
    events.loc[events["Event_ID"].eq(event_id), "Accounting_Status"] = "executable"
    with pytest.raises(SecurityIdentityError, match="no structured legs"):
        validate_security_identity_bundle(replace(bundle, events=events))


def test_validator_rejects_leg_source_from_another_event():
    bundle = _bundle()
    legs = bundle.legs.copy()
    leg_event = legs.iloc[0]["Event_ID"]
    wrong = bundle.sources.loc[bundle.sources["Event_ID"].ne(leg_event)].iloc[0]
    legs.loc[legs.index[0], "Source_ID"] = wrong["Source_ID"]
    with pytest.raises(SecurityIdentityError, match="same event"):
        validate_security_identity_bundle(replace(bundle, legs=legs))


def test_validator_rejects_ctra_dvn_equivalence_and_overlapping_mapping():
    bundle = _bundle()
    equivalence = bundle.provider_row_equivalence.copy()
    row = equivalence["Predecessor_Symbol"].eq("CTRA")
    equivalence.loc[row, "Review_Result"] = "equivalent"
    equivalence.loc[row, "Compared_Fields"] = '["Close"]'
    equivalence.loc[row, "Absolute_Tolerance"] = "0"
    equivalence.loc[row, "Relative_Tolerance"] = "0"
    equivalence.loc[row, "Common_Start"] = "2026-05-01"
    equivalence.loc[row, "Common_End"] = "2026-05-06"
    equivalence.loc[row, "Common_Row_Count"] = "4"
    equivalence.loc[row, "Predecessor_Artifact_SHA256"] = "a" * 64
    equivalence.loc[row, "Successor_Response_SHA256"] = "b" * 64
    with pytest.raises(SecurityIdentityError, match="different securities"):
        validate_security_identity_bundle(
            replace(bundle, provider_row_equivalence=equivalence)
        )

    mappings = bundle.provider_mappings.copy()
    bk = mappings.loc[
        mappings["Mapping_ID"].eq("MAP-LIVE-YAHOO-BK-20240101-BNY")
    ].iloc[0].copy()
    bk["Mapping_ID"] = "MAP-LIVE-YAHOO-BK-20250101-BNY-DUPLICATE"
    bk["Effective_Start"] = "2025-01-01"
    mappings = pd.concat([mappings, bk.to_frame().T], ignore_index=True)
    with pytest.raises(SecurityIdentityError, match="Overlapping provider mappings"):
        validate_security_identity_bundle(
            replace(bundle, provider_mappings=mappings)
        )


def test_non_alias_successor_backfill_requires_approved_row_equivalence():
    bundle = _bundle()
    mappings = bundle.provider_mappings.copy()
    bk_mapping = mappings["Mapping_ID"].eq(
        "MAP-LIVE-YAHOO-BK-20240101-BNY"
    )
    mappings.loc[bk_mapping, "Resolution_Method"] = "reviewed_effective_symbol"

    with pytest.raises(
        SecurityIdentityError,
        match="without approved provider row equivalence",
    ):
        validate_security_identity_bundle(
            replace(bundle, provider_mappings=mappings)
        )

    equivalence = bundle.provider_row_equivalence.copy()
    bk_equivalence = equivalence["Predecessor_Symbol"].eq("BK")
    equivalence.loc[bk_equivalence, "Review_Result"] = "equivalent"
    equivalence.loc[bk_equivalence, "Compared_Fields"] = '["Open","Close"]'
    equivalence.loc[bk_equivalence, "Absolute_Tolerance"] = "0"
    equivalence.loc[bk_equivalence, "Relative_Tolerance"] = "0"
    equivalence.loc[bk_equivalence, "Common_Start"] = "2026-05-01"
    equivalence.loc[bk_equivalence, "Common_End"] = "2026-05-20"
    equivalence.loc[bk_equivalence, "Common_Row_Count"] = "14"
    equivalence.loc[
        bk_equivalence, "Predecessor_Artifact_SHA256"
    ] = "a" * 64
    equivalence.loc[
        bk_equivalence, "Successor_Response_SHA256"
    ] = "b" * 64

    validate_security_identity_bundle(replace(
        bundle,
        provider_mappings=mappings,
        provider_row_equivalence=equivalence,
    ))


def test_reviewed_yahoo_alias_requires_a_strict_same_security_relabel():
    bundle = _bundle()

    legs = bundle.legs.copy()
    bk_leg = legs["Event_ID"].eq("EVT-20260521-BK-BNY-TICKER-CHANGE")
    legs.loc[bk_leg, "Tradable"] = "False"
    with pytest.raises(
        SecurityIdentityError,
        match="without approved provider row equivalence",
    ):
        validate_security_identity_bundle(replace(bundle, legs=legs))

    mappings = bundle.provider_mappings.copy()
    ctra = mappings["Mapping_ID"].eq("MAP-LIVE-YAHOO-CTRA-20240101-CTRA")
    mappings.loc[ctra, "Provider_Symbol"] = "DVN"
    mappings.loc[ctra, "Resolution_Method"] = "reviewed_yahoo_alias"
    with pytest.raises(
        SecurityIdentityError,
        match="without approved provider row equivalence",
    ):
        validate_security_identity_bundle(replace(bundle, provider_mappings=mappings))


def test_approved_otc_alias_keeps_the_same_share_and_rejects_conversion():
    bundle = _bundle()
    mappings = bundle.provider_mappings.copy()
    row = mappings.loc[
        mappings.Scope.eq("backtest") & mappings.Source_Ticker.eq("FRC")
    ].iloc[0].copy()
    row.update({
        "Mapping_ID": "MAP-TEST-YAHOO-FRC-FRCB", "Scope": "live",
        "Provider": "yahoo", "Provider_Symbol": "FRCB", "Asset_ID": "FRC",
        "Effective_Start": "2023-01-01", "Effective_End": "",
        "Resolution_Method": "reviewed_yahoo_alias",
    })
    mappings = mappings.loc[~(mappings.Scope.eq("live") & mappings.Source_Ticker.eq("FRC"))]
    mappings = pd.concat([mappings, row.to_frame().T], ignore_index=True)
    candidate = replace(bundle, provider_mappings=mappings)
    validate_security_identity_bundle(candidate)

    legs = bundle.legs.copy()
    frc = legs.Event_ID.eq("EVT-20230503-FRC-FRCB-OTC-TRANSITION")
    legs.loc[frc, "Share_Ratio"] = "2"
    with pytest.raises(SecurityIdentityError, match="relabel requires To_Ticker, ratio 1"):
        validate_security_identity_bundle(replace(candidate, legs=legs))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Scope", "backtest"),
        ("Provider", "yahoo"),
        ("Provider_Source_URL", ""),
        ("Provider_Evidence_Status", "reviewed_provider_symbol"),
        ("Review_Status", "unresolved"),
    ],
)
def test_reviewed_wikipedia_mapping_is_provider_specific_and_evidenced(
    column,
    value,
):
    bundle = _bundle()
    mappings = bundle.provider_mappings.copy()
    row = mappings["Mapping_ID"].eq(
        "MAP-SHARED-WIKIPEDIA-BKNG-20150131-PCLN"
    )
    mappings.loc[row, column] = value
    with pytest.raises(
        SecurityIdentityError,
        match="invalid reviewed Wikipedia mapping",
    ):
        validate_security_identity_bundle(
            replace(bundle, provider_mappings=mappings)
        )


def test_loader_rejects_manifest_hash_drift(tmp_path):
    directory = tmp_path / "security_identity"
    directory.mkdir()
    for path in CANONICAL_DIR.iterdir():
        (directory / path.name).write_bytes(path.read_bytes())
    with (directory / "security_events.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        load_security_identity_bundle(directory)
