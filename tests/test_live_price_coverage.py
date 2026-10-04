"""Offline live price-requirement, supplement, and readiness tests."""

from __future__ import annotations

import shutil

import pandas as pd
import pytest

from live import acquisition_execution
from live.config import DEFAULT_CONFIG
from live.corporate_action_policy import load_corporate_actions
from live.paths import LivePaths
from live.price_coverage import (
    PRICE_REQUIREMENT_SET,
    SUPPLEMENT_COLUMNS,
    YAHOO_SUPPLEMENT_EXPECTED_ROWS,
    aggregate_price_readiness,
    build_coverage_ledger,
    build_price_requirements,
    read_yahoo_supplement,
    supplemental_available_dates,
)
from live.strategy_universe import evaluation_periods, load_validated_strategy_universe


def _requirements(schedule, membership):
    return build_price_requirements(
        schedule, membership,
        evaluation=evaluation_periods(
            schedule, evaluation_start=DEFAULT_CONFIG.market.competition_start,
            evaluation_end=DEFAULT_CONFIG.market.competition_end,
        ),
    )
from portfolio_core.artifacts import (
    SCHEMA_VERSION,
    ArtifactOrigin,
    read_manifests,
    validate_manifest_artifact,
)


def test_membership_and_possible_holdings_define_exact_price_cells():
    schedule, membership, _ = load_validated_strategy_universe()
    requirements = _requirements(schedule, membership)
    date_sets: dict[str, set[str]] = {}
    for requirement in requirements:
        date_sets.setdefault(requirement.asset_id, set()).add(
            requirement.requirement_date
        )
    dates = {
        asset_id: tuple(sorted(required_dates))
        for asset_id, required_dates in sorted(date_sets.items())
    }

    assert len(dates) == 508
    assert sum(map(len, dates.values())) == 4_034
    assert "DAY" not in dates
    assert dates["HOLX"] == (
        "2026-02-13",
        "2026-02-27",
        "2026-03-02",
        "2026-03-31",
        "2026-04-01",
        "2026-04-30",
        "2026-05-01",
    )
    assert dates["SATS"] == (
        "2026-03-31",
        "2026-04-01",
        "2026-04-30",
        "2026-05-01",
        "2026-05-06",
    )

    assert "2026-02-13" in dates["CIEN"]
    assert "2026-02-27" in dates["CIEN"]
    assert "2026-03-31" in dates["LW"]
    assert "2026-04-01" in dates["LW"]
    assert "2026-05-04" not in dates["LW"]


def test_authoritative_yahoo_supplement_is_hash_validated(tmp_path):
    source = read_yahoo_supplement(DEFAULT_CONFIG.paths)
    assert SUPPLEMENT_COLUMNS == (
        "Date",
        "Asset_ID",
        "Provider_Symbol",
        "Open",
        "Close",
        "Volume",
    )
    assert YAHOO_SUPPLEMENT_EXPECTED_ROWS == {
        "CTRA": 547,
        "HOLX": 526,
    }
    assert tuple(source.columns) == SUPPLEMENT_COLUMNS
    assert source.groupby("Asset_ID").size().to_dict() == (
        YAHOO_SUPPLEMENT_EXPECTED_ROWS
    )
    assert len(source) == 1_073

    paths = LivePaths(tmp_path / "live")
    paths.raw_price_supplemental_dir.mkdir(parents=True)
    shutil.copyfile(
        DEFAULT_CONFIG.paths.raw_price_supplemental_csv,
        paths.raw_price_supplemental_csv,
    )
    shutil.copyfile(
        DEFAULT_CONFIG.paths.raw_price_supplemental_manifest_csv,
        paths.raw_price_supplemental_manifest_csv,
    )
    manifests = read_manifests(paths.raw_price_supplemental_manifest_csv)
    assert len(manifests) == 1
    assert manifests[0].schema_version == SCHEMA_VERSION
    assert manifests[0].origin is ArtifactOrigin.MIGRATED
    validate_manifest_artifact(
        manifests[0], base_dir=paths.raw_price_supplemental_dir
    )
    pd.testing.assert_frame_equal(read_yahoo_supplement(paths), source)

    changed = pd.read_csv(paths.raw_price_supplemental_csv, keep_default_na=False)
    changed.loc[0, "Close"] += 1.0
    changed.to_csv(paths.raw_price_supplemental_csv, index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        read_yahoo_supplement(paths)


def test_composite_readiness_covers_every_required_price_cell():
    schedule, membership, metadata = load_validated_strategy_universe()
    requirements = _requirements(schedule, membership)
    paths = DEFAULT_CONFIG.paths
    panels = {
        "open": acquisition_execution._read_panel(paths.raw_price_open_csv),
        "close": acquisition_execution._read_panel(paths.raw_price_close_csv),
        "volume": acquisition_execution._read_panel(paths.raw_price_volume_csv),
    }
    available_by_symbol = acquisition_execution._available_price_dates(panels)
    symbol_by_asset = metadata.set_index("Asset_ID")["Yahoo_Ticker"].to_dict()
    yahoo_dates = {
        asset_id: available_by_symbol.get(symbol, set())
        for asset_id, symbol in symbol_by_asset.items()
    }
    supplement = read_yahoo_supplement(DEFAULT_CONFIG.paths)
    actions = load_corporate_actions(DEFAULT_CONFIG.paths)
    ledger = build_coverage_ledger(
        requirements,
        yahoo_dates=yahoo_dates,
        supplemental_dates=supplemental_available_dates(supplement),
        corporate_actions=actions,
    )
    readiness = aggregate_price_readiness(
        ledger,
        checked_at_utc="2026-07-30T00:00:00Z",
    )

    assert sum(item.required_count for item in readiness) == 4_034
    assert sum(item.covered_count for item in readiness) == 4_034
    assert [item for item in readiness if item.status.value != "complete"] == []
    sats = ledger.loc[ledger["Asset_ID"].eq("SATS")]
    assert set(sats["Requirement_Date"]) == {
        "2026-03-31",
        "2026-04-01",
        "2026-04-30",
        "2026-05-01",
        "2026-05-06",
    }
    assert sats["Coverage_Source"].eq("yahoo").all()
    assert ledger.loc[ledger["Asset_ID"].eq("DAY")].empty
    holx_late = ledger.loc[
        ledger["Asset_ID"].eq("HOLX")
        & ledger["Requirement_Date"].ge("2026-04-07")
    ]
    assert holx_late["Coverage_Source"].eq(
        "corporate_action_settlement"
    ).all()
    assert holx_late["Coverage_Reference"].eq(
        "EVT-20260407-HOLX-CASH-CVR-ACQUISITION"
    ).all()
    assert set(ledger["Requirement_Set"]) == {PRICE_REQUIREMENT_SET}
