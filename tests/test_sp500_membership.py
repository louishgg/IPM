"""Source-of-truth tests for shared, effective-dated fja05680 provenance."""

from pathlib import Path

import pandas as pd
import pytest

from backtest.paths import DEFAULT_PATHS as BACKTEST_PATHS
from live.paths import DEFAULT_PATHS as LIVE_PATHS
from portfolio_core.artifacts import file_sha256
from portfolio_core.sp500_membership import (
    MEMBERSHIP_COMPONENTS_FILENAME,
    MembershipSourcePaths,
    membership_asof,
    validate_membership_source_manifest,
    validate_membership_sources,
)
from _membership_test_helpers import _write_membership_source_manifest


EXPECTED_HASHES = {
    MEMBERSHIP_COMPONENTS_FILENAME: (
        "39a9202c9ef69a74c0ff07e2113ad41fb6da7c8c5b6cd9541f0185fb4391e717"
    ),
    "sp500_changes_since_2019.csv": (
        "8a187dc1ed1d2bca9ea7a8408a8d7710401fe004705165895bac7aac2811efbc"
    ),
    "sp500_ticker_start_end.csv": (
        "39c488ebd6ce6838599e54751adbe8c8e4b68d5801dd77d29b6d137dd77388ac"
    ),
}


def _write_minimal_sources(directory: Path) -> MembershipSourcePaths:
    paths = MembershipSourcePaths(directory)
    directory.mkdir(parents=True)
    pd.DataFrame({
        "date": ["2019-01-01", "2019-02-01"],
        "tickers": ["AAA,BBB", "BBB,CCC"],
    }).to_csv(paths.components_csv, index=False)
    pd.DataFrame({
        "date": ["2019-02-01"],
        "add": ["CCC"],
        "remove": ["AAA"],
    }).to_csv(paths.changes_csv, index=False)
    pd.DataFrame([
        {"ticker": "AAA", "start_date": "2019-01-01", "end_date": "2019-02-01"},
        {"ticker": "BBB", "start_date": "2019-01-01", "end_date": ""},
        {"ticker": "CCC", "start_date": "2019-02-01", "end_date": ""},
    ]).to_csv(paths.intervals_csv, index=False)
    _write_membership_source_manifest(paths)
    return paths


def test_live_and_backtest_reference_one_shared_membership_directory():
    expected = LIVE_PATHS.project_root / "data/shared/provenance/sp500_membership"

    assert LIVE_PATHS.membership.directory == expected
    assert BACKTEST_PATHS.membership.directory == expected


def test_shared_real_sources_match_recorded_hashes_and_reconcile():
    paths = MembershipSourcePaths(LIVE_PATHS.membership.directory)
    history = validate_membership_sources(paths, required_through="2026-05-06")
    manifest = validate_membership_source_manifest(paths).set_index("File")

    assert len(history) == 2718
    assert history["date"].min() == pd.Timestamp("1996-01-02")
    assert history["date"].max() == pd.Timestamp("2026-06-30")
    assert not history["date"].duplicated().any()
    assert {_path.name: file_sha256(_path) for _path in paths.data_files} == EXPECTED_HASHES
    assert manifest["SHA256"].to_dict() == EXPECTED_HASHES


def test_membership_asof_never_uses_a_future_effective_row():
    history = validate_membership_sources(LIVE_PATHS.membership)

    effective_may_6, may_6 = membership_asof(history, "2026-05-06")
    effective_may_7, may_7 = membership_asof(history, "2026-05-07")

    assert effective_may_6 == pd.Timestamp("2026-04-09")
    assert effective_may_7 == pd.Timestamp("2026-05-07")
    assert "CTRA" in may_6 and "VEEV" not in may_6
    assert "VEEV" in may_7 and "CTRA" not in may_7


def test_shared_validator_rejects_interval_disagreement(tmp_path):
    paths = _write_minimal_sources(tmp_path / "sp500_membership")
    intervals = pd.read_csv(paths.intervals_csv, keep_default_na=False)
    intervals.loc[intervals["ticker"].eq("AAA"), "end_date"] = ""
    intervals.to_csv(paths.intervals_csv, index=False)

    with pytest.raises(ValueError, match="intervals do not reconcile"):
        validate_membership_sources(paths, validate_manifest=False)


def test_shared_validator_rejects_overlapping_open_intervals(tmp_path):
    paths = _write_minimal_sources(tmp_path / "sp500_membership")
    intervals = pd.read_csv(paths.intervals_csv, keep_default_na=False)
    intervals = pd.concat([
        intervals,
        pd.DataFrame([{
            "ticker": "BBB",
            "start_date": "2019-02-01",
            "end_date": "",
        }]),
    ], ignore_index=True)
    intervals.to_csv(paths.intervals_csv, index=False)

    with pytest.raises(ValueError, match="interval rows overlap"):
        validate_membership_sources(paths, validate_manifest=False)


def test_shared_validator_rejects_manifest_hash_drift(tmp_path):
    paths = _write_minimal_sources(tmp_path / "sp500_membership")
    with paths.components_csv.open("a") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="source hash mismatch"):
        validate_membership_source_manifest(paths)
