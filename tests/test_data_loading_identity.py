"""Regression tests for S&P 500 membership and full-RIC price identity."""

from shutil import copy2, copytree

import pandas as pd
import pytest

from backtest.data_loading import (
    load_pit_universe,
    load_price_data,
)
from backtest.paths import DEFAULT_PATHS, BacktestPaths
from backtest.preparation_builders import (
    prepare_pit_membership,
    prepare_prices_monthly,
)
from backtest.security_event_preparation import prepare_backtest_security_events
from portfolio_core.artifacts import file_sha256


def _real_raw_paths(tmp_path) -> BacktestPaths:
    """Use real read-only sources while writing prepared CSVs in tmp_path."""
    paths = BacktestPaths(tmp_path / "backtest")
    paths.raw_data_dir.mkdir(parents=True)
    paths.shared_provenance_dir.mkdir(parents=True)
    if not DEFAULT_PATHS.prices_csv.exists():
        pytest.skip("Reuters raw data is unavailable")
    paths.prices_csv.parent.mkdir(parents=True)
    paths.prices_csv.symlink_to(DEFAULT_PATHS.prices_csv)
    copytree(
        DEFAULT_PATHS.price_sources.raw_dir,
        paths.price_sources.raw_dir,
    )
    paths.security_identity.directory.symlink_to(
        DEFAULT_PATHS.security_identity.directory,
        target_is_directory=True,
    )
    paths.membership.directory.symlink_to(
        DEFAULT_PATHS.membership.directory,
        target_is_directory=True,
    )
    paths.prepared_data_dir.mkdir(parents=True)
    for source, destination in (
        (
            DEFAULT_PATHS.ticker_ric_resolution_csv,
            paths.ticker_ric_resolution_csv,
        ),
        (
            DEFAULT_PATHS.security_event_crossing_audit_csv,
            paths.security_event_crossing_audit_csv,
        ),
        (
            DEFAULT_PATHS.security_event_legs_prepared_csv,
            paths.security_event_legs_prepared_csv,
        ),
        (DEFAULT_PATHS.asset_metadata_csv, paths.asset_metadata_csv),
    ):
        copy2(source, destination)
    return paths


def test_real_prepared_price_loader_keeps_colliding_histories_separate(tmp_path):
    paths = _real_raw_paths(tmp_path)
    prepared = prepare_prices_monthly(paths=paths)
    data_close, data_volume = load_price_data(paths=paths)

    assert len(prepared) == 145 * 742
    assert data_close.shape == (145, 742)
    assert data_close.columns.is_unique
    expected_closes = {
        ("2014-01-31", "AGN^C15"): 114.60,
        ("2014-01-31", "AGN^E20"): 188.98,
        ("2014-01-31", "CB"): 93.81,
        ("2014-01-31", "CB^A16"): 84.54,
        ("2014-01-31", "JCI"): 37.749126621,
        ("2014-01-31", "JCI^I16"): 40.49,
        ("2017-01-31", "FTI"): 26.51774138,
        ("2017-01-31", "FTI^A17"): 35.85,
    }
    for (date, asset_id), expected in expected_closes.items():
        assert data_close.loc[pd.Timestamp(date), asset_id] == pytest.approx(expected)

    assert data_close.loc[pd.Timestamp("2017-08-31"), "DD^I17"] == 83.93
    assert data_volume.loc[
        pd.Timestamp("2017-08-31"), "DD^I17"
    ] == 34_861_021


def test_real_membership_pit_reconciles_sources_and_is_read_only(
    tmp_path,
):
    paths = _real_raw_paths(tmp_path)
    read_only_sources = (
        paths.prices_csv,
        paths.membership.components_csv,
        paths.membership.changes_csv,
        paths.membership.intervals_csv,
        paths.membership.manifest_csv,
        *sorted(paths.security_identity.directory.glob("*.csv")),
    )
    source_hashes = {
        path: file_sha256(path)
        for path in read_only_sources
    }

    pit, historical_asset_ids = prepare_pit_membership(paths=paths)
    reloaded, reloaded_asset_ids = load_pit_universe(paths=paths)
    audit = pd.read_csv(paths.membership_coverage_audit_csv)

    assert pit.shape[0] == 145
    assert pit.columns.is_unique
    assert list(pit.columns) == sorted(pit.columns)
    assert historical_asset_ids == sorted(historical_asset_ids)
    assert audit["Source_Row_Reconciled"].all()
    assert (
        audit["Resolved_Asset_Count"]
        == audit["Source_Member_Count"]
        - audit["Evidence_Backed_Premature_Successor_Exclusion_Count"]
        + audit["Intentional_Collision_Expansion_Count"]
    ).all()
    excluded = audit.loc[
        audit["Evidence_Backed_Premature_Successor_Exclusion_Count"].gt(0),
        ["Date", "Excluded_Premature_Successor_Tickers"],
    ]
    excluded["Date"] = pd.to_datetime(excluded["Date"])
    assert set(excluded["Excluded_Premature_Successor_Tickers"]) == {
        "AVGO",
        "AVGO;CCEP;CPRI",
        "CCEP;CPRI",
        "CPRI",
    }
    assert excluded["Date"].min() == pd.Timestamp("2014-05-31")
    assert excluded["Date"].max() == pd.Timestamp("2018-12-31")
    assert audit["Unavailable_Source_Tickers"].fillna("").eq("").all()
    final = audit.loc[audit["Date"].eq("2026-01-31")].iloc[0]
    assert int(final["Resolved_Asset_Count"]) == 503
    assert int(final["Priced_Asset_Count"]) == 503
    assert pd.isna(final["Unavailable_Source_Tickers"])
    pd.testing.assert_frame_equal(reloaded, pit, check_freq=False)
    assert reloaded_asset_ids == historical_asset_ids

    simultaneous = [
        "AGN^C15",
        "AGN^E20",
        "CB",
        "CB^A16",
        "JCI",
        "JCI^I16",
    ]
    assert pit.loc[pd.Timestamp("2014-12-31"), simultaneous].all()
    assert {
        path: file_sha256(path)
        for path in read_only_sources
    } == source_hashes


def test_real_extension_events_use_reviewed_accounting_treatments(
    tmp_path,
):
    paths = _real_raw_paths(tmp_path)
    prepare_prices_monthly(paths=paths)
    prepare_pit_membership(paths=paths)

    _, _, _, audit = prepare_backtest_security_events(paths=paths)
    extension_events = {
        "EVT-20240201-CDAY-DAY-IDENTITY-CONTINUITY",
        "EVT-20240301-PEAK-DOC-IDENTITY-CONTINUITY",
        "EVT-20240325-FLT-CPAY-IDENTITY-CONTINUITY",
        "EVT-20250807-PARA-PSKY-STOCK-EXCHANGE",
        "EVT-20260114-MMC-MRSH-IDENTITY-CONTINUITY",
    }
    rows = audit.loc[audit["Event_ID"].isin(extension_events)]

    assert set(rows["Event_ID"]) == extension_events
    assert rows.groupby("Event_ID")["Potential_Holding_Crossing"].any().all()
    crossings = rows.loc[rows["Potential_Holding_Crossing"]]
    continuity = crossings.loc[
        crossings["Event_ID"].ne(
            "EVT-20250807-PARA-PSKY-STOCK-EXCHANGE"
        )
    ]
    assert continuity["Accounting_Treatment"].eq(
        "verified_continuous_price_ratio"
    ).all()
    para = crossings.loc[
        crossings["Event_ID"].eq(
            "EVT-20250807-PARA-PSKY-STOCK-EXCHANGE"
        )
    ]
    assert len(para) == 1
    assert para.iloc[0]["Accounting_Treatment"] == "explicit_structured_action"
    assert crossings["End_Valuation_Status"].eq("complete").all()
    assert crossings["Validation_Status"].eq("approved").all()

    hess = audit.loc[
        audit["Event_ID"].eq("EVT-20250718-HES-CVX-STOCK-EXCHANGE")
        & audit["Potential_Holding_Crossing"]
    ]
    assert len(hess) == 1
    assert hess.iloc[0]["Accounting_Treatment"] == "explicit_structured_action"
    assert hess.iloc[0]["End_Valuation_Status"] == "complete"
    assert hess.iloc[0]["Validation_Status"] == "approved"
