"""Tests for causal sector identity and evidence resolution."""

from __future__ import annotations

import pandas as pd
import pytest

from data_acquisition.contracts import ReadinessStatus
from data_acquisition.sector_acquisition_planning import (
    SECTOR_ACQUISITION_REQUIREMENT_COLUMNS,
    build_sector_acquisition_readiness,
    validate_sector_acquisition_requirements,
)
from portfolio_core.sector_assignments import sector_rows_asof
from portfolio_core.security_identity import load_security_identity_bundle
from portfolio_core.sector_resolution import (
    WikipediaIdentityAmbiguityError,
    WikipediaIdentityMissingError,
    WikipediaIdentityResolver,
    build_sector_assignments,
)
from _sector_test_helpers import (
    PROJECT_ROOT,
    _notice,
    _notices,
    _requirements,
    _snapshots,
)


def test_synthetic_sector_change_does_not_leak_backward():
    dates = ["2020-01-31", "2020-02-29"]
    assignments = build_sector_assignments(
        _requirements(
            [(dates[0], "AAA.N", "AAA"), (dates[1], "AAA.N", "AAA")]
        ),
        _snapshots(
            [
                (dates[0], "AAA", "Energy"),
                (dates[1], "AAA", "Materials"),
            ]
        ),
        _notices(),
    )
    assert sector_rows_asof(assignments, dates[0])[
        "GICS_Sector_Code"
    ].to_dict() == {"AAA.N": "10"}
    assert sector_rows_asof(assignments, dates[1])[
        "GICS_Sector_Code"
    ].to_dict() == {"AAA.N": "15"}
    assert assignments["Source_Reference"].tolist() == ["101", "102"]

def test_missing_wikipedia_member_is_never_filled_by_a_prior_assignment():
    with pytest.raises(ValueError, match="No causal sector source"):
        build_sector_assignments(
            _requirements(
                [
                    ("2020-01-31", "OLD.N", "OLD"),
                    ("2020-02-29", "OLD.N", "OLD"),
                ]
            ),
            _snapshots(
                [
                    ("2020-01-31", "OLD", "Energy"),
                    ("2020-02-29", "OTHER", "Materials"),
                ]
            ),
            _notices(),
        )

def test_reviewed_notice_fills_only_a_missing_wikipedia_member():
    assignments = build_sector_assignments(
        _requirements([("2020-01-31", "NEW.N", "NEW")]),
        _snapshots([("2020-01-31", "AAA", "Energy")]),
        _notices([_notice()]),
    )
    assert assignments.loc[0, "GICS_Sector_Code"] == "20"
    assert assignments.loc[0, "Source_Type"] == "S&P Notice"
    assert assignments.loc[0, "Resolution_Method"] == (
        "approved_sp500_addition_fill"
    )

def test_readiness_and_assignments_reject_ambiguous_wikipedia_candidates():
    requirements = _requirements([("2020-01-31", "NEW.N", "NEW")])
    snapshots = _snapshots([
        ("2020-01-31", "N.EW", "Energy"), ("2020-01-31", "NE.W", "Industrials"),
    ])
    notices = _notices([_notice()])
    with pytest.raises(WikipediaIdentityAmbiguityError, match="punctuation normalization.*ambiguous"):
        build_sector_assignments(requirements, snapshots, notices)
    with pytest.raises(WikipediaIdentityAmbiguityError, match="punctuation normalization.*ambiguous"):
        build_sector_acquisition_readiness(
            _readiness_requirements(requirements), snapshots, notices,
            identity_resolver=WikipediaIdentityResolver(), checked_at_utc="2026-08-05T12:00:00Z",
        )


@pytest.mark.parametrize(
    "change",
    [
        {"Published_Date": "2020-01-31"},
        {"Index_Name": "S&P MidCap 400"},
        {"Action": "Replacement"},
        {"Review_Status": "unreviewed"},
    ],
)
def test_noncausal_or_ineligible_notice_cannot_fill(change):
    with pytest.raises(ValueError, match="No causal sector source"):
        build_sector_assignments(
            _requirements([("2020-01-31", "NEW.N", "NEW")]),
            _snapshots([("2020-01-31", "AAA", "Energy")]),
            _notices([_notice(**change)]),
        )


@pytest.mark.parametrize(
    ("change", "method"),
    [
        (
            {"Effective_Date": "2020-02-01"},
            "approved_sp500_announced_sector_fill",
        ),
        (
            {"Action": "Deletion"},
            "approved_sp500_deletion_boundary_fill",
        ),
    ],
)
def test_reviewed_notice_classification_can_fill_without_creating_membership(
    change,
    method,
):
    assignments = build_sector_assignments(
        _requirements([("2020-01-31", "NEW.N", "NEW")]),
        _snapshots([("2020-01-31", "AAA", "Energy")]),
        _notices([_notice(**change)]),
    )

    assert assignments.loc[0, "Resolution_Method"] == method

def test_notice_never_overrides_conflicting_wikipedia_sector():
    with pytest.raises(ValueError, match="Wikipedia/S&P notice sector conflict"):
        build_sector_assignments(
            _requirements([("2020-01-31", "NEW.N", "NEW")]),
            _snapshots([("2020-01-31", "NEW", "Energy")]),
            _notices([_notice(Sector="Industrials")]),
        )

def _readiness_requirements(assignments: pd.DataFrame) -> pd.DataFrame:
    return validate_sector_acquisition_requirements(
        pd.DataFrame(
            [
                (
                    "backtest",
                    row.As_Of_Date,
                    f"{row.As_Of_Date}T00:00:00Z",
                    row.Asset_ID,
                    row.Source_Ticker,
                )
                for row in assignments.itertuples(index=False)
            ],
            columns=SECTOR_ACQUISITION_REQUIREMENT_COLUMNS,
        )
    )


def test_readiness_and_assignments_share_addition_handoff_resolution():
    assignment_requirements = _requirements(
        [
            ("2020-01-31", "NEW.N", "NEW"),
            ("2020-02-29", "NEW.N", "NEW"),
        ]
    )
    snapshots = _snapshots(
        [
            ("2020-01-31", "OTHER", "Energy"),
            ("2020-02-29", "NEW", "Industrials"),
        ]
    )
    notices = _notices([_notice()])
    resolver = WikipediaIdentityResolver()

    readiness = build_sector_acquisition_readiness(
        _readiness_requirements(assignment_requirements),
        snapshots,
        notices,
        identity_resolver=resolver,
        checked_at_utc="2026-08-05T12:00:00Z",
    )
    assignments = build_sector_assignments(
        assignment_requirements,
        snapshots,
        notices,
        identity_resolver=resolver,
    )

    assert readiness[0].status is ReadinessStatus.COMPLETE
    assert readiness[0].covered_count == len(assignments) == 2
    assert readiness[0].contributing_sources == (
        "sp_global_notice",
        "wikipedia",
    )


def test_readiness_reports_missing_evidence_where_assignments_fail():
    assignment_requirements = _requirements(
        [("2020-01-31", "MISSING.N", "MISSING")]
    )
    snapshots = _snapshots([("2020-01-31", "OTHER", "Materials")])
    notices = _notices()
    resolver = WikipediaIdentityResolver()

    readiness = build_sector_acquisition_readiness(
        _readiness_requirements(assignment_requirements),
        snapshots,
        notices,
        identity_resolver=resolver,
        checked_at_utc="2026-08-05T12:00:00Z",
    )

    assert readiness[0].status is ReadinessStatus.MISSING
    assert readiness[0].missing_dates == ("2020-01-31",)
    with pytest.raises(ValueError, match="No causal sector source"):
        build_sector_assignments(
            assignment_requirements,
            snapshots,
            notices,
            identity_resolver=resolver,
        )


def test_notice_handoff_checks_first_wikipedia_row_then_stops_overriding_history():
    assignments = build_sector_assignments(
        _requirements(
            [
                ("2020-01-31", "NEW.N", "NEW"),
                ("2020-02-29", "NEW.N", "NEW"),
                ("2020-03-31", "NEW.N", "NEW"),
            ]
        ),
        _snapshots(
            [
                ("2020-01-31", "OTHER", "Energy"),
                ("2020-02-29", "NEW", "Industrials"),
                ("2020-03-31", "NEW", "Energy"),
            ]
        ),
        _notices([_notice()]),
    )
    assert assignments["GICS_Sector_Code"].tolist() == ["20", "20", "10"]
    assert assignments["Source_Type"].tolist() == [
        "S&P Notice",
        "Wikipedia",
        "Wikipedia",
    ]

def test_notice_handoff_uses_earlier_acquired_snapshot_outside_scope_requirements():
    assignments = build_sector_assignments(
        _requirements([("2020-03-31", "NEW.N", "NEW")]),
        _snapshots(
            [
                ("2020-01-31", "OTHER", "Energy"),
                ("2020-02-29", "NEW", "Industrials"),
                ("2020-03-31", "NEW", "Energy"),
            ]
        ),
        _notices([_notice()]),
    )

    assert assignments.loc[0, "GICS_Sector_Code"] == "10"
    assert assignments.loc[0, "Source_Type"] == "Wikipedia"

def test_readiness_and_assignments_use_effective_asset_specific_wikipedia_mapping():
    dates = ("2015-10-31", "2015-11-30", "2015-12-31")
    requirements = _requirements([(date, "DXC", "DXC") for date in dates])
    snapshots = _snapshots([
        (dates[0], "OTHER", "Energy"), (dates[1], "CSC", "Industrials"),
        (dates[2], "CSC", "Energy"),
    ])
    notices = _notices([_notice(
        Published_Date="2015-09-01", Effective_Date="2015-10-01", Ticker="DXC",
    )])
    resolver = WikipediaIdentityResolver(load_security_identity_bundle(PROJECT_ROOT))
    readiness = build_sector_acquisition_readiness(
        _readiness_requirements(requirements), snapshots, notices,
        identity_resolver=resolver, checked_at_utc="2026-08-05T12:00:00Z",
    )
    assignments = build_sector_assignments(requirements, snapshots, notices, identity_resolver=resolver)
    assert readiness[0].status is ReadinessStatus.COMPLETE
    assert readiness[0].covered_count == len(assignments) == 3
    assert readiness[0].contributing_sources == ("sp_global_notice", "wikipedia")
    assert assignments.GICS_Sector_Code.tolist() == ["20", "20", "10"]
    assert assignments.Source_Type.tolist() == ["S&P Notice", "Wikipedia", "Wikipedia"]
    assert assignments.loc[1, "Source_Symbol"] == "CSC"


def test_identity_resolver_never_guesses_from_company_name():
    snapshot = _snapshots([("2020-01-31", "OLD", "Energy")])
    snapshot.loc[0, "Company_Name"] = "New Holdings"
    with pytest.raises(WikipediaIdentityMissingError, match="No Wikipedia symbol"):
        WikipediaIdentityResolver().resolve("NEW", "2020-01-31", snapshot)

def test_reviewed_wikipedia_ticker_annotation_is_syntactic_not_an_alias():
    snapshot = _snapshots(
        [("2023-12-31", "RVTY (PREVIOUSLY PKI)", "Health Care")]
    )
    symbol, method = WikipediaIdentityResolver().resolve(
        "RVTY", "2023-12-31", snapshot
    )
    assert symbol == "RVTY (PREVIOUSLY PKI)"
    assert method == "normalized_wikipedia_presentation"

@pytest.mark.parametrize(
    ("source", "historical", "date"),
    [
        ("BKNG", "PCLN", "2018-02-28"),
        ("BHGE", "BHI", "2017-06-30"),
        ("DXC", "CSC", "2015-11-30"),
        ("CPRI", "KORS", "2018-12-31"),
        ("LHX", "HRS", "2019-06-30"),
        ("COR", "ABC", "2023-08-31"),
    ],
)
def test_reviewed_wikipedia_mappings_resolve_only_inside_their_intervals(
    source,
    historical,
    date,
):
    resolver = WikipediaIdentityResolver(
        load_security_identity_bundle(PROJECT_ROOT)
    )
    snapshot = _snapshots([(date, historical, "Industrials")])
    resolved, method = resolver.resolve(source, date, snapshot)
    assert resolved == historical
    assert method.startswith("reviewed_wikipedia_mapping:MAP-SHARED-WIKIPEDIA-")

def test_notice_matching_uses_an_exact_effective_wikipedia_mapping():
    date = "2018-02-28"
    assignments = build_sector_assignments(
        _requirements([(date, "BKNG.O", "BKNG")]),
        _snapshots([(date, "OTHER", "Energy")]),
        _notices(
            [
                _notice(
                    Published_Date="2018-02-01",
                    Effective_Date="2018-02-15",
                    Ticker="PCLN",
                )
            ]
        ),
        identity_resolver=WikipediaIdentityResolver(
            load_security_identity_bundle(PROJECT_ROOT)
        ),
    )

    assert assignments.loc[0, "Source_Type"] == "S&P Notice"
    assert assignments.loc[0, "Source_Symbol"] == "PCLN"

@pytest.mark.parametrize(
    ("source", "asset_id", "date"),
    [
        ("CB", "CB^A16", "2015-01-31"),
        ("JCI", "JCI^I16", "2015-01-31"),
    ],
)
def test_collision_leg_cannot_borrow_another_assets_wikipedia_mapping(
    source,
    asset_id,
    date,
):
    resolver = WikipediaIdentityResolver(
        load_security_identity_bundle(PROJECT_ROOT)
    )
    snapshot = _snapshots([(date, "OTHER", "Industrials")])

    with pytest.raises(WikipediaIdentityMissingError, match="found 0"):
        resolver.resolve(source, date, snapshot, asset_id=asset_id)

    assert resolver.notice_symbol_keys(
        source,
        date,
        asset_id=asset_id,
    ) == ()

    with pytest.raises(ValueError, match="No causal sector source"):
        build_sector_assignments(
            _requirements([(date, asset_id, source)]),
            snapshot,
            _notices(
                [
                    _notice(
                        Published_Date="2015-01-01",
                        Effective_Date="2015-01-15",
                        Ticker=source,
                    )
                ]
            ),
            identity_resolver=resolver,
        )

@pytest.mark.parametrize(
    ("source", "asset_id", "historical", "date"),
    [
        ("AGN", "AGN^E20", "ACT", "2015-01-31"),
        ("CB", "CB", "ACE", "2015-12-31"),
        ("JCI", "JCI", "TYC", "2016-08-31"),
    ],
)
def test_asset_specific_collision_mapping_precedes_direct_source_symbol(
    source,
    asset_id,
    historical,
    date,
):
    resolver = WikipediaIdentityResolver(
        load_security_identity_bundle(PROJECT_ROOT)
    )
    snapshot = _snapshots(
        [
            (date, source, "Consumer Discretionary"),
            (date, historical, "Industrials"),
        ]
    )

    resolved, method = resolver.resolve(
        source,
        date,
        snapshot,
        asset_id=asset_id,
    )

    assert resolved == historical
    assert method.startswith("reviewed_wikipedia_mapping:MAP-SHARED-WIKIPEDIA-")

def test_jci_collision_legs_receive_their_distinct_historical_sectors():
    date = "2016-08-31"
    snapshot = _snapshots(
        [
            (date, "JCI", "Consumer Discretionary"),
            (date, "TYC", "Industrials"),
        ]
    )
    snapshot.loc[:, "Revision_ID"] = "735661179"
    snapshot.loc[:, "Revision_Timestamp_UTC"] = "2016-08-30T12:00:00Z"
    assignments = build_sector_assignments(
        _requirements(
            [
                (date, "JCI", "JCI"),
                (date, "JCI^I16", "JCI"),
            ]
        ),
        snapshot,
        _notices(),
        identity_resolver=WikipediaIdentityResolver(
            load_security_identity_bundle(PROJECT_ROOT)
        ),
    ).set_index("Asset_ID")

    assert assignments.loc["JCI", "GICS_Sector_Code"] == "20"
    assert assignments.loc["JCI", "Source_Symbol"] == "TYC"
    assert assignments.loc["JCI^I16", "GICS_Sector_Code"] == "25"
    assert assignments.loc["JCI^I16", "Source_Symbol"] == "JCI"

def test_premature_cpri_successor_cannot_leak_past_the_kors_change():
    resolver = WikipediaIdentityResolver(
        load_security_identity_bundle(PROJECT_ROOT)
    )
    snapshot = _snapshots([("2019-01-31", "KORS", "Consumer Discretionary")])
    with pytest.raises(WikipediaIdentityMissingError, match="found 0"):
        resolver.resolve("CPRI", "2019-01-31", snapshot)
