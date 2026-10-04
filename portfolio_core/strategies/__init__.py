"""Canonical interchangeable strategy implementations."""

from .strategy_contract import (
    ExecutionDecision,
    Strategy,
    StrategyDecision,
    StrategyDecisionContext,
)
from .research_parameters import (
    BufferParameters,
    ExposureParameters,
    ResearchParameters,
    SignalParameters,
    SizingParameters,
    StockSelection,
    VolatilityProfile,
    WholeSectorSelection,
)
from .research_state import (
    SectorBasket,
    SelectedSectors,
)
from .registry import (
    StrategyDefinition,
    available_strategy_ids,
    build_registered_strategy,
    build_strategy_from_file,
    get_strategy_definition,
    load_parameter_packet,
    load_saved_strategy_parameters,
    read_saved_strategy_identity,
    save_strategy_parameters,
    strategy_parameters_payload,
)

__all__ = [
    "BufferParameters",
    "ExposureParameters",
    "ResearchParameters",
    "SectorBasket",
    "SelectedSectors",
    "SignalParameters",
    "SizingParameters",
    "StockSelection",
    "VolatilityProfile",
    "WholeSectorSelection",
    "ExecutionDecision",
    "Strategy",
    "StrategyDefinition",
    "StrategyDecision",
    "StrategyDecisionContext",
    "available_strategy_ids",
    "build_registered_strategy",
    "build_strategy_from_file",
    "get_strategy_definition",
    "load_parameter_packet",
    "load_saved_strategy_parameters",
    "read_saved_strategy_identity",
    "save_strategy_parameters",
    "strategy_parameters_payload",
]
