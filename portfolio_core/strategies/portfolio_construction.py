"""Deterministic buffered stock selection, global neutral repair and base sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stock_volatility(prices, profile):
    """Sample volatility of monthly returns, with linked finite-observation minima.

    Missing calendar months and prices are never filled. Selection accepts a
    valid zero estimate; inverse sizing applies its own strictly-positive gate.
    """
    calendar = pd.date_range(prices.index.min(), prices.index.max(), freq="ME")
    monthly = prices.reindex(calendar).where(np.isfinite(prices))
    returns = monthly.pct_change(fill_method=None)
    returns = returns.where(np.isfinite(returns))
    return returns.rolling(
        profile.window, min_periods=profile.minimum_observations
    ).std(ddof=1)


def rank_orders(scores):
    ids = sorted(scores.index)
    return (
        sorted(ids, key=lambda a: (-float(scores[a]), a)),
        sorted(ids, key=lambda a: (float(scores[a]), a)),
    )


def select_stocks(scores, previous, sectors, selection, buffer, neutral):
    """Protect both retained books, then fill longs and shorts in rank order.

    Neutral repair ensures paired sector representation before dollar-neutral
    sizing. Return selections and retained/provisional audit sets, with empty
    selections when the requested counts or sector pairing are infeasible.
    """
    long_order, short_order = rank_orders(scores)
    nl, ns = selection.n_long, selection.n_short
    if len(scores) < nl + ns:
        return (), (), {}, {}
    retained = {}
    for side, order, k in ((1, long_order, nl), (-1, short_order, ns)):
        cutoff = min(len(order), k + buffer.additional_pool(k))
        retained[side] = set(
            a
            for a in order[:cutoff]
            if buffer.enabled and side * previous.get(a, 0) > 0
        )
        # Accepted books have exact counts; guard externally supplied oversized state.
        if len(retained[side]) > k:
            retained[side] = set([a for a in order if a in retained[side]][:k])
    chosen = {
        side: [a for a in order if a in retained[side]]
        for side, order in ((1, long_order), (-1, short_order))
    }
    assigned = set(chosen[1] + chosen[-1])
    for side, order, k in ((1, long_order, nl), (-1, short_order, ns)):
        for a in order:
            if len(chosen[side]) == k:
                break
            if a not in assigned:
                chosen[side].append(a)
                assigned.add(a)
    provisional = {side: set(ids) for side, ids in chosen.items()}
    if neutral and {sectors[a] for a in chosen[1]} != {sectors[a] for a in chosen[-1]}:
        chosen = _neutral_repair(
            long_order, short_order, sectors, nl, ns, retained, provisional
        )
        if chosen is None:
            return (), (), retained, provisional
    return (
        tuple(a for a in long_order if a in chosen[1]),
        tuple(a for a in short_order if a in chosen[-1]),
        retained,
        provisional,
    )


def _neutral_repair(longs, shorts, sectors, nl, ns, retained, provisional):
    """Return disjoint, sector-paired books with exact counts, or None if infeasible.

    Maximize same-side retained count, then provisional count, then minimize
    summed side ranks. Final ties prefer inclusion in retained/provisional/other,
    rank, long-before-short, then Asset_ID order.
    """
    ranks = {
        1: {a: i + 1 for i, a in enumerate(longs)},
        -1: {a: i + 1 for i, a in enumerate(shorts)},
    }
    pairs = [(side, a) for side, ids in ((1, longs), (-1, shorts)) for a in ids]
    pairs.sort(
        key=lambda sa: (
            0 if sa[1] in retained[sa[0]] else 1 if sa[1] in provisional[sa[0]] else 2,
            ranks[sa[0]][sa[1]],
            0 if sa[0] == 1 else 1,
            sa[1],
        )
    )
    bit = {pair: 1 << (len(pairs) - i - 1) for i, pair in enumerate(pairs)}
    # For K=nl+ns and N=len(longs), inclusion bits sum below rank_base,
    # all K*N rank units fit below provisional_base, and all K provisional
    # units below retained_base. Lower priorities cannot outweigh higher ones.
    rank_base = 1 << len(pairs)
    provisional_base = rank_base * ((nl + ns) * len(longs) + 1)
    retained_base = provisional_base * (nl + ns + 1)
    reward = {
        (side, a): bit[side, a]
        + (len(longs) - ranks[side][a]) * rank_base
        + (a in provisional[side]) * provisional_base
        + (a in retained[side]) * retained_base
        for side, a in pairs
    }
    global_dp = {(0, 0): 0}
    # Local states assign each asset at most once; the global search combines
    # only unused sectors or sectors represented on both sides.
    for sector in sorted(set(sectors[a] for a in longs)):
        local = {(0, 0): 0}
        for a in sorted(a for a in longs if sectors[a] == sector):
            updated = dict(local)
            for (l, s), value in local.items():
                if l < nl:
                    key = (l + 1, s)
                    updated[key] = max(updated.get(key, -1), value + reward[1, a])
                if s < ns:
                    key = (l, s + 1)
                    updated[key] = max(updated.get(key, -1), value + reward[-1, a])
            local = updated
        options = [
            (l, s, v)
            for (l, s), v in local.items()
            if (l == 0 and s == 0) or (l > 0 and s > 0)
        ]
        combined = {}
        for (gl, gs), gv in global_dp.items():
            for l, s, value in options:
                if gl + l <= nl and gs + s <= ns:
                    key = (gl + l, gs + s)
                    combined[key] = max(combined.get(key, -1), gv + value)
        global_dp = combined
    value = global_dp.get((nl, ns))
    if value is None:
        return None
    mask = value % rank_base
    return {
        side: {a for s, a in pairs if s == side and mask & bit[s, a]}
        for side in (1, -1)
    }


def size_stocks(longs, shorts, volatility, sectors, sizing, long_share, neutral):
    """Size signed unit-gross weights using equal or inverse-volatility sizing.

    Initially allocate long_share to longs and 1-long_share to shorts.
    Neutral sizing requires paired sectors and long_share=0.5; it splits each
    sector's preliminary gross equally while preserving within-side ratios.
    """
    weights = pd.Series(index=pd.Index([*longs, *shorts], name="Asset_ID"), dtype=float)
    for ids, budget in ((longs, long_share), (shorts, -(1 - long_share))):
        base = (
            pd.Series(1.0, index=list(ids))
            if sizing.method == "equal"
            else 1 / volatility.loc[list(ids)]
        )
        if not np.isfinite(base).all() or (base <= 0).any():
            raise ValueError(
                "Sizing requires finite positive inverse-volatility inputs"
            )
        weights.loc[list(ids)] = budget * base / base.sum()
    if neutral:
        groups = pd.Series(sectors).reindex(weights.index)
        for sector in sorted(groups.unique()):
            ids = groups.index[groups == sector]
            ls = ids[weights.loc[ids] > 0]
            ss = ids[weights.loc[ids] < 0]
            l, s = weights.loc[ls].sum(), -weights.loc[ss].sum()
            if l <= 0 or s <= 0:
                raise ValueError("Neutral sizing requires paired sector representation")
            budget = (l + s) / 2
            weights.loc[ls] *= budget / l
            weights.loc[ss] *= budget / s
    return weights
