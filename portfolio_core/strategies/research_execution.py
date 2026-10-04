"""Research constraints shared by monthly backtest and live execution."""

from dataclasses import replace

import numpy as np
import pandas as pd

from portfolio_core.accounting_ledger import (
    InfeasibleRebalanceError,
    _scaled_target_shares,
)


def require_positive_equity(equity, date):
    """Economic insolvency is candidate infeasibility; invalid values are errors."""
    if not np.isfinite(equity):
        raise ValueError(f"{date}: research equity must be finite")
    if equity <= 0:
        raise InfeasibleRebalanceError(f"{date}: nonpositive research equity {equity}")


def research_accounting_config(accounting, n_long, n_short):
    """Exact research holdings can strengthen, never weaken, account minimums."""
    return replace(
        accounting,
        minimum_positions=max(accounting.minimum_positions, n_long + n_short),
        minimum_long_positions=max(accounting.minimum_long_positions, n_long),
        minimum_short_positions=max(accounting.minimum_short_positions, n_short),
    )


def sector_execution_residuals(
    unsuppressed, current, applied, scale, prices, sectors, *, whole_share_orders,
):
    """Separate suppression from reachable rounding at a common price basis.

    ``current`` is the execution ledger's holding vector, including any event
    fractions acquired since sizing. Include zero-trade assets in both vectors.
    """
    mechanical = _scaled_target_shares(
        unsuppressed, current, scale, whole_share_orders=whole_share_orders,
    ).reindex(prices.index).fillna(0.0)
    applied = applied.reindex(prices.index).fillna(0.0)
    values = pd.DataFrame({
        "Applied_Net_Dollars": applied * prices,
        "Mechanical_Net_Dollars": mechanical * prices,
    }).groupby(sectors).sum()
    values["Suppression_Net_Dollars"] = (
        values.Applied_Net_Dollars - values.Mechanical_Net_Dollars
    )
    return values
