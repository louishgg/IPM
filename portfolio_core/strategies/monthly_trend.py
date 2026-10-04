"""Monthly price/SMA ranking, a project hypothesis with two mandatory books."""

import numpy as np
import pandas as pd

from .research_strategy import CONSTRUCTION_SIGNAL_COLUMNS, ResearchStrategy


def monthly_sma_scores(prices, window):
    """Include the current month and require every finite monthly observation.

    Reindex missing calendar months rather than treating L observed rows as L
    months. No forward filling or sign-based eligibility is applied.
    """
    calendar = pd.date_range(prices.index.min(), prices.index.max(), freq="ME")
    monthly = prices.reindex(calendar).where(np.isfinite(prices))
    average = monthly.rolling(window, min_periods=window).mean()
    return monthly / average - 1


class MonthlyTrendStrategy(ResearchStrategy):
    strategy_id = "monthly_trend"
    strategy_version = "2.0.0"
    return_column = "Price_To_SMA"
    signal_columns = (return_column, *CONSTRUCTION_SIGNAL_COLUMNS)

    def _signal(self, prices):
        return monthly_sma_scores(prices, self.parameters.signal.moving_average_months).iloc[-1]
