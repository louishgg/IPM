"""Acquisition-owned reconciliation for canonical raw-share checkpoints."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from portfolio_core.shares import (
    RAW_IDENTITY_COLUMNS,
    CanonicalShareIdentity,
    raw_rows_for_key,
    replace_canonical_share_observations,
    validate_raw_shares,
)

from .contracts import AcquisitionRequest, AcquisitionStatus, ProviderStatus


def project_share_identity_interval(
    *,
    asset_id: str,
    provider_symbol: str,
    requested_start: str,
    requested_end: str,
    effective_start: pd.Timestamp | None,
    effective_end_exclusive: pd.Timestamp | None,
) -> tuple[CanonicalShareIdentity, str, str]:
    """Project an end-exclusive provider identity onto inclusive share dates.

    Effective boundaries outside the requested window stay open so they do not
    fragment the durable share identity key.
    """

    canonical_start = (
        pd.Timestamp(effective_start).strftime("%Y-%m-%d")
        if effective_start is not None
        else ""
    )
    canonical_end = (
        (
            pd.Timestamp(effective_end_exclusive)
            - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")
        if effective_end_exclusive is not None
        else ""
    )
    identity = CanonicalShareIdentity(
        asset_id=str(asset_id),
        provider_symbol=str(provider_symbol),
        effective_start=(
            canonical_start
            if canonical_start and canonical_start > requested_start
            else ""
        ),
        effective_end=(
            canonical_end
            if canonical_end and canonical_end < requested_end
            else ""
        ),
    )
    return (
        identity,
        max(requested_start, canonical_start or requested_start),
        min(requested_end, canonical_end or requested_end),
    )


def acquisition_identity_to_raw_key(status_or_request) -> tuple[str, str, str, str]:
    """Project one full acquisition identity onto the canonical raw-share key."""
    identity = status_or_request.identity
    return (
        identity.asset_id,
        identity.provider_symbol,
        identity.effective_start,
        identity.effective_end,
    )


def replace_share_payload(
    raw: pd.DataFrame,
    status_or_request,
    payload: pd.Series | None,
) -> pd.DataFrame:
    """Apply a successful replacement while retaining history on empty outcomes."""
    return replace_canonical_share_observations(
        raw,
        CanonicalShareIdentity(
            *acquisition_identity_to_raw_key(status_or_request)
        ),
        payload,
    )


def _status_matches_raw(status: AcquisitionStatus, rows: pd.DataFrame) -> bool:
    if status.status is ProviderStatus.OK:
        return (
            status.observation_count > 0
            and len(rows) == status.observation_count
            and rows["Date"].min().strftime("%Y-%m-%d")
            == status.observation_start
            and rows["Date"].max().strftime("%Y-%m-%d")
            == status.observation_end
        )
    return len(rows) == 0 and status.observation_count == 0


def validate_share_status_reconciliation(
    statuses: Iterable[AcquisitionStatus],
    raw: pd.DataFrame,
) -> None:
    """Validate identities and successful payloads; allow retained observations."""
    status_values = tuple(statuses)
    by_key = {status.identity.key: status for status in status_values}
    if len(by_key) != len(status_values):
        raise ValueError("Duplicate full acquisition identities in shares status")
    raw_by_key = {
        key: group
        for key, group in raw.groupby(list(RAW_IDENTITY_COLUMNS), sort=False)
    }
    status_raw_keys = {
        acquisition_identity_to_raw_key(status): status for status in status_values
    }
    if len(status_raw_keys) != len(status_values):
        raise ValueError(
            "Shares statuses collapse distinct acquisition identities onto one "
            "raw identity"
        )
    orphan = sorted(set(raw_by_key) - set(status_raw_keys))
    if orphan:
        raise ValueError(f"Raw shares observations have no status identity: {orphan}")
    for key, status in status_raw_keys.items():
        rows = raw_by_key.get(key, raw.iloc[0:0])
        if status.status is ProviderStatus.OK and not _status_matches_raw(status, rows):
            raise ValueError(
                "Shares observation checkpoint mismatch for full identity "
                f"{status.identity.key!r}"
            )


def reconcile_share_checkpoints(
    requests: Iterable[AcquisitionRequest],
    statuses: Iterable[AcquisitionStatus],
    raw: pd.DataFrame,
    *,
    reset_note: str,
) -> tuple[list[AcquisitionStatus], pd.DataFrame]:
    """Reset invalid requested checkpoints without merging identity intervals."""
    request_values = tuple(requests)
    status_values = tuple(statuses)
    by_key = {status.identity.key: status for status in status_values}
    if len(by_key) != len(status_values):
        raise ValueError("Duplicate full acquisition identities in shares status")
    result = validate_raw_shares(raw)
    reconciled: list[AcquisitionStatus] = []
    requested_keys = {request.identity.key for request in request_values}
    for request in request_values:
        status = by_key.get(request.identity.key)
        rows = raw_rows_for_key(
            result,
            acquisition_identity_to_raw_key(request),
        )
        valid_terminal = (
            status is not None
            and status.status in {ProviderStatus.OK, ProviderStatus.NO_DATA}
            and (
                _status_matches_raw(status, rows)
                or (
                    status.status is ProviderStatus.NO_DATA
                    and not rows.empty
                )
            )
        )
        if valid_terminal:
            reconciled.append(status)
            continue
        if (
            status is not None
            and status.status in {ProviderStatus.PENDING, ProviderStatus.FAILED}
            and not rows.empty
        ):
            reconciled.append(status)
            continue
        if not rows.empty:
            result = result.drop(index=rows.index).reset_index(drop=True)
        if status is not None and status.status in {
            ProviderStatus.OK,
            ProviderStatus.NO_DATA,
        }:
            reconciled.append(
                AcquisitionStatus.pending(request, migration_note=reset_note)
            )
        elif status is not None:
            reconciled.append(status)

    unrelated = [
        status
        for status in status_values
        if status.identity.key not in requested_keys
    ]
    unrelated_raw_keys = {
        acquisition_identity_to_raw_key(status) for status in unrelated
    }
    if unrelated_raw_keys:
        raw_keys = pd.MultiIndex.from_frame(
            result.loc[:, RAW_IDENTITY_COLUMNS].astype(str)
        )
        unrelated_raw = result.loc[raw_keys.isin(unrelated_raw_keys)]
    else:
        unrelated_raw = result.iloc[0:0]
    validate_share_status_reconciliation(
        unrelated,
        unrelated_raw,
    )
    return reconciled, validate_raw_shares(result)


__all__ = [
    "acquisition_identity_to_raw_key",
    "project_share_identity_interval",
    "reconcile_share_checkpoints",
    "replace_share_payload",
    "validate_share_status_reconciliation",
]
