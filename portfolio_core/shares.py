"""Canonical raw-share schemas, validation, replacement, and alignment."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_SHARES_COLUMNS = (
    "Asset_ID",
    "Provider_Symbol",
    "Effective_Start",
    "Effective_End",
    "Date",
    "Observation_Sequence",
    "Shares_Outstanding",
)
RAW_IDENTITY_COLUMNS = (
    "Asset_ID",
    "Provider_Symbol",
    "Effective_Start",
    "Effective_End",
)
SHARE_ASOF_COLUMNS = (
    "Date",
    "Asset_ID",
    "Observation_Date",
    "Shares_Outstanding",
)


@dataclass(frozen=True, slots=True)
class CanonicalShareIdentity:
    """One inclusive acquisition interval for an economic asset."""

    asset_id: str
    provider_symbol: str
    effective_start: str = ""
    effective_end: str = ""

    @property
    def raw_key(self) -> tuple[str, str, str, str]:
        return (
            self.asset_id,
            self.provider_symbol,
            self.effective_start,
            self.effective_end,
        )


def validate_raw_shares(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize canonical identity-keyed raw observations."""
    if tuple(frame.columns) != RAW_SHARES_COLUMNS:
        raise ValueError(
            f"Raw shares must contain exactly {list(RAW_SHARES_COLUMNS)}; "
            f"found {list(frame.columns)}"
        )
    result = frame.copy()
    for column in RAW_IDENTITY_COLUMNS:
        result[column] = result[column].astype(str).str.strip()
    if result[["Asset_ID", "Provider_Symbol"]].eq("").any().any():
        raise ValueError("Raw shares contain an empty provider identity")

    starts = pd.to_datetime(
        result["Effective_Start"].replace("", pd.NA), errors="coerce"
    )
    ends = pd.to_datetime(
        result["Effective_End"].replace("", pd.NA), errors="coerce"
    )
    invalid_starts = result["Effective_Start"].ne("") & starts.isna()
    invalid_ends = result["Effective_End"].ne("") & ends.isna()
    if invalid_starts.any() or invalid_ends.any():
        raise ValueError("Raw shares contain an invalid effective boundary")
    if (starts.notna() & ends.notna() & starts.gt(ends)).any():
        raise ValueError("Raw shares effective start cannot follow effective end")

    result["Date"] = pd.to_datetime(result["Date"], errors="raise")
    normalized_dates = result["Date"].dt.tz_localize(None).dt.normalize()
    outside = (
        (starts.notna() & normalized_dates.lt(starts))
        | (ends.notna() & normalized_dates.gt(ends))
    )
    if outside.any():
        raise ValueError(
            "Raw shares contain observations outside their inclusive effective "
            "identity interval"
        )

    sequence = pd.to_numeric(result["Observation_Sequence"], errors="raise")
    if (sequence < 0).any() or (sequence % 1 != 0).any():
        raise ValueError(
            "Raw shares observation sequences must be nonnegative integers"
        )
    result["Observation_Sequence"] = sequence.astype("int64")
    values = pd.to_numeric(result["Shares_Outstanding"], errors="raise")
    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric <= 0.0).any():
        raise ValueError("Raw shares must be finite and positive")
    result["Shares_Outstanding"] = values.astype("float64")

    key = [*RAW_IDENTITY_COLUMNS, "Date", "Observation_Sequence"]
    if result.duplicated(key).any():
        raise ValueError("Raw shares contain duplicate observation identities")
    result = result.sort_values(key, kind="stable").reset_index(drop=True)
    expected = result.groupby(
        [*RAW_IDENTITY_COLUMNS, "Date"],
        sort=False,
        dropna=False,
    ).cumcount()
    if not result["Observation_Sequence"].equals(expected):
        raise ValueError(
            "Raw shares observation sequences must be contiguous and zero-based"
        )
    return result


def load_raw_shares(path: Path) -> pd.DataFrame:
    """Load and validate canonical identity-keyed raw observations."""

    return validate_raw_shares(pd.read_csv(path, keep_default_na=False))


def raw_rows_for_key(
    raw: pd.DataFrame,
    key: tuple[str, str, str, str],
) -> pd.DataFrame:
    """Return rows matching one complete canonical provider identity."""
    mask = pd.Series(True, index=raw.index)
    for column, value in zip(RAW_IDENTITY_COLUMNS, key, strict=True):
        mask &= raw[column].eq(value)
    return raw.loc[mask]


def replace_canonical_share_observations(
    raw: pd.DataFrame,
    identity: CanonicalShareIdentity,
    observations: pd.Series | None,
) -> pd.DataFrame:
    """Replace a nonempty payload; preserve history when no replacement arrives."""
    result = validate_raw_shares(raw)
    if observations is None or observations.empty:
        return result
    existing = raw_rows_for_key(result, identity.raw_key)
    result = result.drop(index=existing.index).reset_index(drop=True)

    series = pd.Series(observations, dtype="float64")
    addition = pd.DataFrame({
        "Asset_ID": identity.asset_id,
        "Provider_Symbol": identity.provider_symbol,
        "Effective_Start": identity.effective_start,
        "Effective_End": identity.effective_end,
        "Date": pd.to_datetime(series.index).tz_localize(None),
        "Shares_Outstanding": series.to_numpy(dtype=float),
    })
    addition["Observation_Sequence"] = addition.groupby(
        [*RAW_IDENTITY_COLUMNS, "Date"],
        sort=False,
        dropna=False,
    ).cumcount()
    return validate_raw_shares(pd.concat(
        [result, addition.loc[:, RAW_SHARES_COLUMNS]],
        ignore_index=True,
    ))


def _select_canonical_share_observations_asof(
    canonical: pd.DataFrame,
    requirements: Iterable[tuple[object, str]],
) -> pd.DataFrame:
    """Select covered requirements from already validated canonical rows."""
    starts = pd.to_datetime(
        canonical["Effective_Start"].replace("", pd.NA),
        errors="raise",
    )
    ends = pd.to_datetime(
        canonical["Effective_End"].replace("", pd.NA),
        errors="raise",
    )
    observation_dates = pd.to_datetime(canonical["Date"], errors="raise")
    if observation_dates.dt.tz is not None:
        observation_dates = observation_dates.dt.tz_localize(None)
    rows_by_asset = {
        str(asset_id): rows
        for asset_id, rows in canonical.groupby("Asset_ID", sort=False)
    }

    records: list[dict[str, object]] = []
    for value, asset_id in requirements:
        boundary = pd.Timestamp(value)
        if pd.isna(boundary):
            raise ValueError("Share as-of requirement dates cannot be missing")
        if boundary.tzinfo is not None:
            boundary = boundary.tz_localize(None)
        asset = str(asset_id)
        if not asset:
            raise ValueError("Share as-of requirement Asset_ID cannot be empty")

        asset_rows = rows_by_asset.get(asset)
        if asset_rows is None:
            continue
        row_index = asset_rows.index
        eligible = asset_rows.loc[
            observation_dates.loc[row_index].le(boundary)
            & (starts.loc[row_index].isna() | starts.loc[row_index].le(boundary))
            & (ends.loc[row_index].isna() | ends.loc[row_index].ge(boundary))
        ]
        if eligible.empty:
            continue
        candidates = eligible.groupby(
            list(RAW_IDENTITY_COLUMNS),
            sort=False,
            dropna=False,
        ).tail(1)
        values = candidates["Shares_Outstanding"].to_numpy(dtype=float)
        if len(values) > 1 and not np.isclose(
            values,
            values[0],
            rtol=0.0,
            atol=0.0,
        ).all():
            raise ValueError(
                "Overlapping canonical share identities provide conflicting "
                f"values for {asset}"
            )
        last = candidates.iloc[-1]
        records.append({
            "Date": boundary,
            "Asset_ID": asset,
            "Observation_Date": last["Date"],
            "Shares_Outstanding": float(last["Shares_Outstanding"]),
        })
    return pd.DataFrame(records, columns=SHARE_ASOF_COLUMNS)


def select_canonical_share_observations_asof(
    raw: pd.DataFrame,
    requirements: Iterable[tuple[object, str]],
) -> pd.DataFrame:
    """Return the latest interval-eligible observation for each requirement.

    Effective starts and ends are inclusive. Requirements without a causal
    observation are omitted so readiness and preparation can apply their own
    reporting and recovery policies to the same selected rows.
    """
    return _select_canonical_share_observations_asof(
        validate_raw_shares(raw),
        requirements,
    )


__all__ = [
    "CanonicalShareIdentity",
    "RAW_IDENTITY_COLUMNS",
    "RAW_SHARES_COLUMNS",
    "SHARE_ASOF_COLUMNS",
    "load_raw_shares",
    "raw_rows_for_key",
    "replace_canonical_share_observations",
    "select_canonical_share_observations_asof",
    "validate_raw_shares",
]
