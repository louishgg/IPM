"""Tests for strategy-independent backtest sector requirements."""

from datetime import date

import numpy as np
import pandas as pd

import backtest.acquisition_planning as acquisition_planning
import backtest.sector_requirements as sector_requirements_module
from backtest.config import DEFAULT_CONFIG, BacktestMarketConfig
from backtest.preparation_builders import build_sector_assignment_requirements
from backtest.sector_requirements import (
    active_sector_requirements_from_resolution,
    consumer_sector_requirements_from_resolution,
)
from portfolio_core.sector_assignments import (
    load_sector_assignments,
    prior_sector_audit_fields,
)


def test_active_sector_requirements_are_a_conservative_acquisition_superset():
    config = BacktestMarketConfig(
        start_date=date(2014, 1, 31),
        end_date=date(2015, 2, 28),
    )
    resolution = pd.DataFrame(
        [
            ("2014-12-31", "EARLY", "EARLY", "eligible_priced"),
            ("2015-01-31", "CB", "CB", "eligible_priced"),
            ("2015-01-31", "CB^A16", "CB", "eligible_unavailable"),
            ("2015-01-31", "SKIP", "SKIP", "unresolved"),
            ("2015-02-28", "LATE", "LATE", "eligible_priced"),
        ],
        columns=("Date", "Asset_ID", "Source_Ticker", "Eligibility_Status"),
    )

    requirements = active_sector_requirements_from_resolution(resolution, config)

    assert requirements.to_dict("records") == [
        {
            "As_Of_Date": pd.Timestamp("2015-01-31"),
            "Asset_ID": "CB",
            "Source_Ticker": "CB",
        },
        {
            "As_Of_Date": pd.Timestamp("2015-01-31"),
            "Asset_ID": "CB^A16",
            "Source_Ticker": "CB",
        },
    ]


def test_consumer_requirements_exclude_assets_without_interval_valuation():
    config = BacktestMarketConfig(
        start_date=date(2014, 1, 31),
        end_date=date(2015, 2, 28),
    )
    dates = pd.date_range(config.start_date, config.end_date, freq="ME")
    close = pd.DataFrame(
        {
            "GOOD": np.linspace(10.0, 23.0, len(dates)),
            "NO_END": [*np.linspace(20.0, 32.0, len(dates) - 1), np.nan],
            "UNPRICED": [np.nan] * len(dates),
        },
        index=dates,
    )
    resolution = pd.DataFrame(
        [
            ("2015-01-31", "GOOD", "GOOD", "eligible_priced"),
            ("2015-01-31", "NO_END", "NO_END", "eligible_priced"),
            ("2015-01-31", "UNPRICED", "UNPRICED", "eligible_unavailable"),
        ],
        columns=("Date", "Asset_ID", "Source_Ticker", "Eligibility_Status"),
    )

    requirements = consumer_sector_requirements_from_resolution(
        resolution,
        close,
        market_config=config,
    )

    assert requirements.to_dict("records") == [
        {
            "As_Of_Date": pd.Timestamp("2015-01-31"),
            "Asset_ID": "GOOD",
            "Source_Ticker": "GOOD",
        }
    ]


def test_backtest_sector_requirements_add_scope_cutoff_and_schema(monkeypatch):
    active = pd.DataFrame(
        [
            {
                "As_Of_Date": pd.Timestamp("2020-01-31"),
                "Asset_ID": "AAA.N",
                "Source_Ticker": "AAA",
            }
        ]
    )
    monkeypatch.setattr(
        sector_requirements_module,
        "build_active_sector_requirements",
        lambda market_config, paths: active,
    )

    requirements = acquisition_planning.backtest_sector_requirements()

    assert list(requirements.columns) == [
        "Scope",
        "Requirement_Date",
        "Cutoff_UTC",
        "Asset_ID",
        "Source_Ticker",
    ]
    assert requirements.to_dict("records") == [
        {
            "Scope": acquisition_planning.SCOPE,
            "Requirement_Date": pd.Timestamp("2020-01-31"),
            "Cutoff_UTC": "2020-01-31T00:00:00Z",
            "Asset_ID": "AAA.N",
            "Source_Ticker": "AAA",
        }
    ]


def test_repository_assignments_exactly_cover_true_consumers():
    paths = DEFAULT_CONFIG.paths
    assignments = load_sector_assignments(paths.sector_assignments_csv)
    requirements = build_sector_assignment_requirements(
        DEFAULT_CONFIG.market,
        paths,
    )

    expected_pairs = set(
        requirements[["As_Of_Date", "Asset_ID"]].itertuples(
            index=False, name=None
        )
    )
    actual_pairs = set(
        assignments[["As_Of_Date", "Asset_ID"]].itertuples(
            index=False, name=None
        )
    )
    assert actual_pairs == expected_pairs
    assert not assignments.duplicated(["As_Of_Date", "Asset_ID"]).any()
    assert set(assignments["GICS_Sector_Code"]).issubset(
        {"10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60"}
    )
    deletion_fills = assignments.loc[
        assignments["Resolution_Method"].str.contains("deletion")
    ]
    assert deletion_fills[
        ["As_Of_Date", "Asset_ID", "GICS_Sector_Code", "Source_Reference"]
    ].to_dict("records") == [
        {
            "As_Of_Date": pd.Timestamp("2016-01-31"),
            "Asset_ID": "BRCM.O^B16",
            "GICS_Sector_Code": "45",
            "Source_Reference": "SPDJI-2016-01-22-BRCM",
        }
    ]
    assert {
        (pd.Timestamp("2016-01-31"), "PCP^B16"),
        (pd.Timestamp("2022-10-31"), "TWTR.K^J22"),
    }.isdisjoint(actual_pairs)
    assert (pd.Timestamp("2016-01-31"), "BRCM.O^B16") in actual_pairs

    fields = prior_sector_audit_fields(
        assignments,
        "PCP^B16",
        "2016-01-31",
        context="regression exit",
    )
    assert fields["Sector_As_Of_Date"] == pd.Timestamp("2015-12-31")
    assert fields["GICS_Sector_Code"] == "20"


def test_extended_acquisition_plan_has_exact_dates_and_asset_requirements():
    requirements = acquisition_planning.backtest_sector_requirements()

    assert len(requirements) == 66_524
    assert requirements["Requirement_Date"].nunique() == 132
    assert requirements["Requirement_Date"].min() == pd.Timestamp("2015-01-31")
    assert requirements["Requirement_Date"].max() == pd.Timestamp("2025-12-31")


def test_momentum_forced_exits_use_latest_prior_causal_sector():
    paths = DEFAULT_CONFIG.paths.for_strategy("momentum")
    assignments = load_sector_assignments(paths.sector_assignments_csv)
    trades = pd.read_csv(paths.strategy_trades_csv, keep_default_na=False)
    trades["Execution_Date"] = pd.to_datetime(trades["Execution_Date"])
    trades["Sector_As_Of_Date"] = pd.to_datetime(trades["Sector_As_Of_Date"])
    forced = trades.loc[
        trades["Sector_As_Of_Date"].lt(trades["Execution_Date"])
    ].copy()

    assert len(forced) == 39
    assert set(forced["Applied_Rule"]).issubset({"exit_long", "exit_short"})
    for row in forced.itertuples(index=False):
        prior = assignments.loc[
            assignments["Asset_ID"].eq(row.Asset_ID)
            & assignments["As_Of_Date"].lt(row.Execution_Date)
        ].sort_values("As_Of_Date", kind="stable").iloc[-1]
        assert row.Sector_As_Of_Date == prior["As_Of_Date"]
        assert str(row.GICS_Sector_Code) == str(prior["GICS_Sector_Code"])
        assert row.Sector_Source_Type == prior["Source_Type"]
        assert str(row.Sector_Source_Reference) == str(
            prior["Source_Reference"]
        )


def test_momentum_twtr_is_event_accounting_not_an_october_2022_trade():
    paths = DEFAULT_CONFIG.paths.for_strategy("momentum")
    trades = pd.read_csv(paths.strategy_trades_csv, keep_default_na=False)
    trades["Execution_Date"] = pd.to_datetime(trades["Execution_Date"])
    october_twtr = trades.loc[
        trades["Asset_ID"].eq("TWTR.K^J22")
        & trades["Execution_Date"].eq(pd.Timestamp("2022-10-31"))
    ]
    assert october_twtr.empty

    accounting = pd.read_csv(
        paths.security_event_accounting_audit_csv,
        keep_default_na=False,
    )
    event = accounting.loc[
        accounting["Event_ID"].eq(
            "EVT-20221027-TWTR-CASH-ACQUISITION"
        )
    ]
    assert len(event) == 1
    assert event.iloc[0]["Effective_Date"] == "2022-10-27"
    assert event.iloc[0]["Event_Type"] == "cash_settlement"
