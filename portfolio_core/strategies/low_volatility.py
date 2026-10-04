"""Low-volatility long-short selection using the shared monthly estimator."""
from .research_strategy import CONSTRUCTION_SIGNAL_COLUMNS, ResearchStrategy
from .portfolio_construction import stock_volatility


class LowVolatilityStrategy(ResearchStrategy):
    strategy_id = "low_volatility"
    strategy_version = "2.0.0"
    return_column = "Selection_Volatility"
    score_direction = -1
    signal_columns = (return_column, *CONSTRUCTION_SIGNAL_COLUMNS)

    def _signal(self, prices):
        return stock_volatility(
            prices, self.parameters.signal.selection_volatility
        ).iloc[-1]
