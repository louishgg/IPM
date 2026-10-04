"""Backtest-only resolution policies for canonical security events."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

import backtest.security_event_preparation as security_events
from backtest.config import BacktestMarketConfig
from portfolio_core.portfolio_lifecycle import finite_positive_price
from portfolio_core.security_identity import SecurityIdentityBundle


CONTINUOUS_EVENTS = (
    (
        "EVT-20200401-ARNC-HWM-DISTRIBUTION",
        "2020-04-01",
        "ARNC",
        "HWM",
        "ARNC",
        "HWM",
        "1/4",
    ),
    (
        "EVT-20210803-LB-BBWI-DISTRIBUTION",
        "2021-08-03",
        "LB",
        "BBWI",
        "VSCO",
        "BBWI.K",
        "1/3",
    ),
    (
        "EVT-20221215-FBHS-FBIN-DISTRIBUTION",
        "2022-12-15",
        "FBHS",
        "FBIN",
        "MBC",
        "FBIN.K",
        "1",
    ),
)
EXPLICIT_EVENT = "EVT-20190402-DWDP-DOW-DISTRIBUTION"


def test_security_event_prices_use_canonical_finite_positive_validation():
    date = pd.Timestamp("2024-01-31")
    prices = pd.DataFrame(
        {
            "NaN": [float("nan")],
            "Infinite": [float("inf")],
            "Zero": [0.0],
            "Negative": [-1.0],
            "Valid": [12.5],
        },
        index=[date],
    )

    assert finite_positive_price(prices, date, "Missing") is None
    assert finite_positive_price(
        prices,
        pd.Timestamp("2024-02-29"),
        "Valid",
    ) is None
    for asset_id in ("NaN", "Infinite", "Zero", "Negative"):
        assert finite_positive_price(prices, date, asset_id) is None
    assert finite_positive_price(prices, date, "Valid") == 12.5
    assert security_events.finite_positive_price is finite_positive_price


def _bundle() -> SecurityIdentityBundle:
    event_rows: list[dict[str, object]] = []
    leg_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    common_leg = {
        "Cash_Amount": "",
        "Currency": "",
        "CVR_Units": "",
        "CVR_Base_Value_Per_Unit": "",
        "CVR_Max_Value_Per_Unit": "",
        "Review_Status": "approved",
    }
    mapping_evidence = {
        "Scope": "backtest",
        "Provider": "reuters",
        "Local_First_Date": "2014-01-31",
        "Local_Last_Date": "2026-01-30",
        "Simultaneous_Symbol_Conflict": "False",
        "Provider_Evidence_Status": "local_reuters_observation",
        "Review_Status": "approved",
    }
    for (
        event_id,
        effective_date,
        predecessor,
        survivor,
        child,
        asset_id,
        child_ratio,
    ) in CONTINUOUS_EVENTS:
        event_rows.append(
            {
                "Event_ID": event_id,
                "Effective_Date": effective_date,
                "Event_Type": "distribution",
                "Continuity_Class": "predecessor_survives",
                "Accounting_Status": "executable",
                "Review_Status": "approved",
            }
        )
        leg_rows.extend(
            [
                {
                    **common_leg,
                    "Event_ID": event_id,
                    "Leg_Sequence": "1",
                    "From_Ticker": predecessor,
                    "To_Ticker": child,
                    "Leg_Type": "distribution",
                    "Share_Ratio": child_ratio,
                    "Retain_Predecessor": "False",
                },
                {
                    **common_leg,
                    "Event_ID": event_id,
                    "Leg_Sequence": "2",
                    "From_Ticker": predecessor,
                    "To_Ticker": survivor,
                    "Leg_Type": "relabel",
                    "Share_Ratio": "1",
                    "Retain_Predecessor": "False",
                },
            ]
        )
        mapping_rows.append(
            {
                **mapping_evidence,
                "Event_ID": event_id,
                "Source_Ticker": predecessor,
                "Asset_ID": asset_id,
                "Legacy_From_Ticker": predecessor,
                "Legacy_To_Ticker": survivor,
            }
        )
        source_rows.append(
            {
                "Event_ID": event_id,
                "Source_URL": f"https://example.test/{event_id}",
                "Review_Status": "approved",
            }
        )

    event_rows.append(
        {
            "Event_ID": EXPLICIT_EVENT,
            "Effective_Date": "2019-04-02",
            "Event_Type": "distribution",
            "Continuity_Class": "predecessor_survives",
            "Accounting_Status": "executable",
            "Review_Status": "approved",
        }
    )
    leg_rows.append(
        {
            **common_leg,
            "Event_ID": EXPLICIT_EVENT,
            "Leg_Sequence": "1",
            "From_Ticker": "DWDP",
            "To_Ticker": "DOW",
            "Leg_Type": "distribution",
            "Share_Ratio": "1/3",
            "Retain_Predecessor": "True",
        }
    )
    mapping_rows.extend(
        [
            {
                **mapping_evidence,
                "Event_ID": EXPLICIT_EVENT,
                "Source_Ticker": "DWDP",
                "Asset_ID": "DD",
                "Legacy_From_Ticker": "DWDP",
                "Legacy_To_Ticker": "",
            },
            {
                **mapping_evidence,
                "Event_ID": EXPLICIT_EVENT,
                "Source_Ticker": "DOW",
                "Asset_ID": "DOW",
                "Legacy_From_Ticker": "",
                "Legacy_To_Ticker": "DOW",
            },
        ]
    )
    source_rows.append(
        {
            "Event_ID": EXPLICIT_EVENT,
            "Source_URL": f"https://example.test/{EXPLICIT_EVENT}",
            "Review_Status": "approved",
        }
    )
    return SecurityIdentityBundle(
        events=pd.DataFrame(event_rows),
        legs=pd.DataFrame(leg_rows),
        sources=pd.DataFrame(source_rows),
        provider_mappings=pd.DataFrame(mapping_rows),
        provider_row_equivalence=pd.DataFrame(),
    )


def _prepare_policy_result(tmp_path, monkeypatch):
    bundle = _bundle()
    monkeypatch.setattr(
        security_events,
        "load_security_identity_bundle",
        lambda _root: bundle,
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    resolution_path = prepared / "ticker_ric_resolution.csv"
    price_path = prepared / "prices_monthly.csv"
    dates = pd.date_range("2019-03-31", "2022-12-31", freq="ME")
    predecessor_identities = {
        "DWDP": ("DD", pd.Timestamp("2019-04-02")),
        "ARNC": ("HWM", pd.Timestamp("2020-04-01")),
        "LB": ("BBWI.K", pd.Timestamp("2021-08-03")),
        "FBHS": ("FBIN.K", pd.Timestamp("2022-12-15")),
    }
    resolution_rows = [
        {
            "Date": observation_date,
            "Source_Ticker": ticker,
            "Asset_ID": asset_id,
            "Is_Priced": True,
        }
        for observation_date in dates
        for ticker, (asset_id, effective) in predecessor_identities.items()
        if observation_date < effective
    ]
    resolution_rows.extend(
        {
            "Date": observation_date,
            "Source_Ticker": "DOW",
            "Asset_ID": "DOW",
            "Is_Priced": True,
        }
        for observation_date in dates
    )
    pd.DataFrame(
        resolution_rows
    ).to_csv(resolution_path, index=False)
    price_assets = {"DD", "DOW", "HWM", "BBWI.K", "FBIN.K"}
    pd.DataFrame(
        [
            {
                "Date": observation_date,
                "Asset_ID": asset_id,
                "Price_Close": 100.0 + offset,
            }
            for offset, observation_date in enumerate(dates)
            for asset_id in price_assets
        ]
    ).to_csv(price_path, index=False)
    paths = SimpleNamespace(
        project_root=tmp_path,
        ticker_ric_resolution_csv=resolution_path,
        prices_monthly_csv=price_path,
        security_events_prepared_csv=prepared / "security_events.csv",
        security_event_legs_prepared_csv=prepared / "security_event_legs.csv",
        security_event_sources_prepared_csv=prepared / "security_event_sources.csv",
        security_event_crossing_audit_csv=prepared / "security_event_crossing_audit.csv",
    )
    return security_events.prepare_backtest_security_events(
        BacktestMarketConfig(
            start_date=date(2019, 3, 31),
            end_date=date(2022, 12, 31),
        ),
        paths,
    )


def test_prepared_backtest_applies_continuous_and_explicit_event_policies(
    tmp_path,
    monkeypatch,
):
    events, legs, sources, audit = _prepare_policy_result(tmp_path, monkeypatch)
    event_ids = set(events["Event_ID"])
    leg_event_ids = set(legs["Event_ID"])
    source_event_ids = set(sources["Event_ID"])

    for event_id, _, _, _, _, from_asset_id, _ in CONTINUOUS_EVENTS:
        assert event_id not in event_ids
        assert event_id not in leg_event_ids
        assert event_id not in source_event_ids
        row = audit.loc[
            audit["Event_ID"].eq(event_id)
            & audit["From_Asset_ID"].eq(from_asset_id)
        ].iloc[0]
        assert bool(row["Potential_Holding_Crossing"])
        assert row["Accounting_Treatment"] == "verified_continuous_price_ratio"
        assert row["End_Valuation_Status"] == "complete"
        assert "without applying the child leg again" in row["Notes"]

    assert EXPLICIT_EVENT in set(events["Event_ID"])
    assert EXPLICIT_EVENT in set(legs["Event_ID"])
    assert EXPLICIT_EVENT in set(sources["Event_ID"])
    row = audit.loc[audit["Event_ID"].eq(EXPLICIT_EVENT)].iloc[0]
    assert row["Accounting_Treatment"] == "explicit_structured_action"
    assert row["End_Valuation_Status"] == "complete"


def test_documentary_reorganization_links_to_adjacent_structured_action():
    documentary = "EVT-20190319-DOCUMENTARY"
    accounting = "EVT-20190320-STRUCTURED"
    bundle = SecurityIdentityBundle(
        events=pd.DataFrame([
            {
                "Event_ID": documentary,
                "Accounting_Status": "documented_not_executable",
            },
            {
                "Event_ID": accounting,
                "Accounting_Status": "executable",
            },
        ]),
        legs=pd.DataFrame(),
        sources=pd.DataFrame(),
        provider_mappings=pd.DataFrame(),
        provider_row_equivalence=pd.DataFrame(),
    )
    common = {
        "From_Ticker": "FOX",
        "From_Asset_ID": "TFCF.O^C19",
        "Previous_Decision_Date": pd.Timestamp("2019-02-28"),
        "Next_Valuation_Date": pd.Timestamp("2019-03-31"),
        "Potential_Holding_Crossing": True,
        "End_Valuation_Status": "complete",
        "Validation_Status": "approved",
    }
    audit = pd.DataFrame([
        {
            **common,
            "Event_ID": documentary,
            "Effective_Date": pd.Timestamp("2019-03-19"),
            "Accounting_Treatment": "verified_continuous_price_ratio",
            "Executing_Event_ID": "",
            "Source_URLs": "https://example.test/documentary",
            "Notes": "legacy fallback",
        },
        {
            **common,
            "Event_ID": accounting,
            "Effective_Date": pd.Timestamp("2019-03-20"),
            "Accounting_Treatment": "explicit_structured_action",
            "Executing_Event_ID": accounting,
            "Source_URLs": "https://example.test/terms",
            "Notes": "structured terms",
        },
    ], columns=security_events.AUDIT_COLUMNS)

    linked = security_events._link_adjacent_documentary_events(
        audit,
        bundle,
        {
            documentary: "https://example.test/documentary",
            accounting: "https://example.test/terms",
        },
    )

    row = linked.loc[linked["Event_ID"].eq(documentary)].iloc[0]
    assert row["Accounting_Treatment"] == "explicit_structured_action"
    assert row["Executing_Event_ID"] == accounting
    assert set(row["Source_URLs"].split(";")) == {
        "https://example.test/documentary",
        "https://example.test/terms",
    }
    assert "not executed separately" in row["Notes"]
