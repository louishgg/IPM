"""Deterministic ranking shared by strategy searches."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def add_deterministic_ranks(
    frame: pd.DataFrame,
    rankings: Sequence[tuple[str, str]],
    *,
    tie_breaker: str,
) -> pd.DataFrame:
    """Rank metrics descending and resolve ties by one ascending stable key."""
    required = {tie_breaker, *(metric for metric, _ in rankings)}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Ranking input is missing columns {missing}")

    result = frame.copy()
    for metric, rank_column in rankings:
        order = result.sort_values(
            [metric, tie_breaker],
            ascending=[False, True],
            kind="stable",
        ).index
        ranks = pd.Series(range(1, len(order) + 1), index=order, dtype=int)
        result[rank_column] = ranks.reindex(result.index).astype(int)
    return result


__all__ = ["add_deterministic_ranks"]
