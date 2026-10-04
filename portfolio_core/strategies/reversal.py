"""Return reversal: buy past losers and short past winners over the chosen window."""

from .research_strategy import CONSTRUCTION_SIGNAL_COLUMNS, ResearchStrategy


class ReversalStrategy(ResearchStrategy):
    """Reuse stock construction and execution with negative formation returns.

    The supplied JSON grid selects the one-month F=1/S=0 specification.
    The raw return stays in the audit; Strategy_Score is its negative so all
    shared ranking, buffering, neutral repair and execution refill paths agree.
    """

    strategy_id = "reversal"
    strategy_version = "2.0.0"
    score_direction = -1

    @property
    def return_column(self):
        # Keep the historical audit label only when it describes the signal.
        signal = self.parameters.signal
        if signal.formation_months == 1 and signal.skip_months == 0:
            return "Prior_Month_Return"
        return "Formation_Return"

    @property
    def signal_columns(self):
        return (self.return_column, *CONSTRUCTION_SIGNAL_COLUMNS)
