"""Monthly F/S momentum with shared stock construction and fixed exposure."""

from dataclasses import dataclass

from .research_strategy import CONSTRUCTION_SIGNAL_COLUMNS, ResearchStrategy


@dataclass(frozen=True)
class MomentumStrategy(ResearchStrategy):
    strategy_id = "momentum"
    return_column = "Momentum"
    signal_columns = (return_column, *CONSTRUCTION_SIGNAL_COLUMNS)
