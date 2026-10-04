"""Shared deterministic date-index helpers."""

from __future__ import annotations

import pandas as pd


def month_end_index(start: object, end: object) -> pd.DatetimeIndex:
    """Return inclusive calendar month ends."""
    start_text = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_text = pd.Timestamp(end).strftime("%Y-%m-%d")
    return pd.date_range(start_text, end_text, freq="ME")


__all__ = ["month_end_index"]
