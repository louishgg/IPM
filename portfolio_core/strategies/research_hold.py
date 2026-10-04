"""Shared hard-constraint check for an incomplete monthly research decision."""

import numpy as np
import pandas as pd

from portfolio_core.accounting_ledger import InfeasibleRebalanceError


def cap_carried_targets(weights, strategy, accounting):
    """Reduce a drifted held target without changing its relative allocation."""
    if not np.isfinite(weights).all():
        raise ValueError("Prior targets must be finite")
    gross = float(weights.abs().sum())
    limit = min(strategy.parameters.exposure.gross, accounting.maximum_gross_exposure)
    scale = min(1.0, limit / gross) if gross > 0 else 1.0
    capped = weights * scale
    return capped, {
        "Carried_Gross_Before": gross,
        "Carried_Gross_After": float(capped.abs().sum()),
        "Carried_Gross_Scale": scale,
        "Carried_Gross_Override": "configured_gross_cap" if scale < 1 else "",
    }


def validate_hold(weights, eligible, sectors, strategy, date, *, context=None, selected=None):
    """Carry prior targets only when every selection constraint still holds."""
    weights = weights.loc[weights.index.isin(eligible)]
    if strategy.strategy_id == "sector_momentum":
        weights = strategy.validate_prior(weights, context, selected)
    elif (weights > 0).sum() != strategy.n_long or (
        weights < 0
    ).sum() != strategy.n_short:
        raise InfeasibleRebalanceError(
            f"{date}: incomplete momentum decision cannot preserve requested counts"
        )
    if not np.isfinite(weights).all():
        raise ValueError("Prior targets must be finite")
    if strategy.parameters.sector_neutral:
        net = weights.groupby(pd.Series(sectors).reindex(weights.index)).sum()
        if not np.allclose(net, 0, atol=1e-12, rtol=0):
            raise InfeasibleRebalanceError(
                f"{date}: incomplete momentum decision cannot preserve sector neutrality"
            )
    return weights
