"""Focused contracts for canonical share-interval projection."""

from dataclasses import replace

import pandas as pd
import pytest

from data_acquisition.contracts import AcquisitionStatus, ProviderStatus
from data_acquisition.share_checkpoints import (
    project_share_identity_interval, reconcile_share_checkpoints,
    validate_share_status_reconciliation,
)
from _acquisition_test_helpers import backtest_acquisition_request
from portfolio_core.shares import (
    CanonicalShareIdentity,
    RAW_SHARES_COLUMNS,
    SHARE_ASOF_COLUMNS,
    replace_canonical_share_observations,
    select_canonical_share_observations_asof,
    validate_raw_shares,
)


REQUESTED_START = "2021-01-31"
REQUESTED_END = "2024-03-31"


@pytest.fixture
def share_checkpoints():
    requests = [backtest_acquisition_request(asset) for asset in ("AAA", "BBB")]
    statuses = [AcquisitionStatus(
        identity=request.identity, status=ProviderStatus.OK,
        requested_start=request.requested_start, requested_end=request.requested_end,
        observation_count=2, observation_start="2024-01-10", observation_end="2024-01-20",
    ) for request in requests]
    raw = validate_raw_shares(pd.DataFrame([
        [asset, asset, "", "", date, 0, value]
        for asset, value in (("AAA", 100.0), ("BBB", 200.0))
        for date in ("2024-01-10", "2024-01-20")
    ], columns=RAW_SHARES_COLUMNS))
    return requests, statuses, raw


@pytest.mark.parametrize("fault,message", [
    ("duplicate", "Duplicate full acquisition identities"),
    ("collapsed", "collapse distinct acquisition identities"),
    ("orphan", "no status identity"),
    ("count", "checkpoint mismatch"),
    ("start", "checkpoint mismatch"),
    ("end", "checkpoint mismatch"),
    ("missing", "checkpoint mismatch"),
])
def test_share_reconciliation_rejects_corrupt_checkpoints(share_checkpoints, fault, message):
    _, statuses, raw = share_checkpoints
    validate_share_status_reconciliation(statuses, raw)
    if fault == "duplicate":
        statuses.append(statuses[0])
    elif fault == "collapsed":
        statuses.append(replace(statuses[0], identity=replace(statuses[0].identity, scope="live")))
    elif fault == "orphan":
        statuses = statuses[:1]
    elif fault == "missing":
        raw = raw.loc[raw.Asset_ID.eq("BBB")]
    else:
        changes = {"count": {"observation_count": 3},
                   "start": {"observation_start": "2024-01-09"},
                   "end": {"observation_end": "2024-01-21"}}
        statuses[0] = replace(statuses[0], **changes[fault])
    with pytest.raises(ValueError, match=message):
        validate_share_status_reconciliation(statuses, raw)


@pytest.mark.parametrize("fault", ["count", "start", "end", "missing", "absent-status"])
def test_invalid_requested_checkpoint_resets_without_changing_other_history(share_checkpoints, fault):
    requests, statuses, raw = share_checkpoints
    unrelated = raw.loc[raw.Asset_ID.eq("BBB")].reset_index(drop=True)
    if fault == "absent-status":
        statuses = statuses[1:]
    elif fault == "missing":
        raw = unrelated
    else:
        changes = {"count": {"observation_count": 3},
                   "start": {"observation_start": "2024-01-09"},
                   "end": {"observation_end": "2024-01-21"}}
        statuses[0] = replace(statuses[0], **changes[fault])
    actual, retained = reconcile_share_checkpoints(requests[:1], statuses, raw, reset_note="retry corrupt checkpoint")
    assert actual == ([] if fault == "absent-status" else [
        AcquisitionStatus.pending(requests[0], migration_note="retry corrupt checkpoint"),
    ])
    pd.testing.assert_frame_equal(retained, unrelated)


@pytest.mark.parametrize("status", list(ProviderStatus))
def test_valid_or_retained_requested_history_survives_reconciliation(share_checkpoints, status):
    requests, statuses, raw = share_checkpoints
    if status is not ProviderStatus.OK:
        statuses[0] = replace(
            AcquisitionStatus.pending(requests[0]), status=status,
            error_class="TimeoutError" if status is ProviderStatus.FAILED else "",
            error_message="retryable timeout" if status is ProviderStatus.FAILED else "",
        )
    validate_share_status_reconciliation(statuses, raw)
    actual, retained = reconcile_share_checkpoints(requests[:1], statuses, raw, reset_note="unused")
    assert actual == statuses[:1]
    pd.testing.assert_frame_equal(retained, raw)


def test_reconciliation_rejects_duplicate_and_corrupt_unrelated_statuses(share_checkpoints):
    requests, statuses, raw = share_checkpoints
    with pytest.raises(ValueError, match="Duplicate full acquisition identities"):
        reconcile_share_checkpoints(requests[:1], [*statuses, statuses[0]], raw, reset_note="unused")
    statuses[1] = replace(statuses[1], observation_count=3)
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        reconcile_share_checkpoints(requests[:1], statuses, raw, reset_note="unused")


@pytest.mark.parametrize("payload", [None, pd.Series(dtype="float64")])
def test_empty_share_replacement_preserves_history_and_other_identities(payload):
    raw = validate_raw_shares(pd.DataFrame(
        [
            ["AAA", "AAA", "", "", "2024-01-10", 0, 100.0],
            ["AAA", "AAA", "", "", "2024-01-10", 1, 110.0],
            ["BBB", "BBB", "", "", "2024-01-15", 0, 200.0],
        ],
        columns=RAW_SHARES_COLUMNS,
    ))

    actual = replace_canonical_share_observations(
        raw, CanonicalShareIdentity("AAA", "AAA"), payload,
    )

    pd.testing.assert_frame_equal(actual, raw)


@pytest.mark.parametrize(
    (
        "effective_start",
        "effective_end_exclusive",
        "expected_raw_key",
        "expected_requested_start",
        "expected_requested_end",
    ),
    (
        (
            None,
            None,
            ("AAA.O", "AAA", "", ""),
            REQUESTED_START,
            REQUESTED_END,
        ),
        (
            pd.Timestamp(REQUESTED_START),
            pd.Timestamp("2024-04-01"),
            ("AAA.O", "AAA", "", ""),
            REQUESTED_START,
            REQUESTED_END,
        ),
        (
            pd.Timestamp("2022-02-17"),
            pd.Timestamp("2022-06-09"),
            ("AAA.O", "AAA", "2022-02-17", "2022-06-08"),
            "2022-02-17",
            "2022-06-08",
        ),
        (
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2025-01-01"),
            ("AAA.O", "AAA", "", ""),
            REQUESTED_START,
            REQUESTED_END,
        ),
    ),
)
def test_share_projection_preserves_boundaries_and_durable_identity_keys(
    effective_start,
    effective_end_exclusive,
    expected_raw_key,
    expected_requested_start,
    expected_requested_end,
):
    identity, requested_start, requested_end = project_share_identity_interval(
        asset_id="AAA.O",
        provider_symbol="AAA",
        requested_start=REQUESTED_START,
        requested_end=REQUESTED_END,
        effective_start=effective_start,
        effective_end_exclusive=effective_end_exclusive,
    )

    assert identity.raw_key == expected_raw_key
    assert requested_start == expected_requested_start
    assert requested_end == expected_requested_end


def test_share_asof_selection_respects_adjacent_inclusive_intervals():
    predecessor, _, _ = project_share_identity_interval(
        asset_id="META.O",
        provider_symbol="FB",
        requested_start=REQUESTED_START,
        requested_end=REQUESTED_END,
        effective_start=None,
        effective_end_exclusive=pd.Timestamp("2022-06-09"),
    )
    successor, _, _ = project_share_identity_interval(
        asset_id="META.O",
        provider_symbol="META",
        requested_start=REQUESTED_START,
        requested_end=REQUESTED_END,
        effective_start=pd.Timestamp("2022-06-09"),
        effective_end_exclusive=None,
    )
    raw = pd.DataFrame(columns=RAW_SHARES_COLUMNS)
    raw = replace_canonical_share_observations(
        raw,
        predecessor,
        pd.Series([100.0], index=pd.DatetimeIndex(["2022-06-01"])),
    )
    raw = replace_canonical_share_observations(
        raw,
        successor,
        pd.Series([200.0], index=pd.DatetimeIndex(["2022-06-10"])),
    )

    selected = select_canonical_share_observations_asof(
        raw,
        (
            ("2022-06-08", "META.O"),
            ("2022-06-09", "META.O"),
            ("2022-06-10", "META.O"),
            ("2022-06-11", "META.O"),
        ),
    )

    expected = pd.DataFrame(
        [
            ("2022-06-08", "META.O", "2022-06-01", 100.0),
            ("2022-06-10", "META.O", "2022-06-10", 200.0),
            ("2022-06-11", "META.O", "2022-06-10", 200.0),
        ],
        columns=SHARE_ASOF_COLUMNS,
    )
    expected[["Date", "Observation_Date"]] = expected[
        ["Date", "Observation_Date"]
    ].apply(pd.to_datetime)
    pd.testing.assert_frame_equal(selected, expected)


def test_share_asof_selection_rejects_conflicting_overlapping_identities():
    raw = pd.DataFrame(
        [
            ["AAA", "OLD", "", "2022-06-30", "2022-06-01", 0, 100.0],
            ["AAA", "NEW", "2022-06-01", "", "2022-06-01", 0, 200.0],
        ],
        columns=RAW_SHARES_COLUMNS,
    )

    with pytest.raises(
        ValueError,
        match="Overlapping canonical share identities.*AAA",
    ):
        select_canonical_share_observations_asof(
            raw,
            (("2022-06-15", "AAA"),),
        )
