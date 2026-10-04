"""Strategy implementation registry and strict, lazy JSON configuration loading."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from portfolio_core.io import atomic_write_text

from . import configuration
from .configuration import read_json_object
from .low_volatility import LowVolatilityStrategy
from .momentum import MomentumStrategy
from .monthly_trend import MonthlyTrendStrategy
from .research_parameters import ResearchParameters
from .reversal import ReversalStrategy
from .sector_momentum import SectorMomentumStrategy
from .strategy_contract import Strategy


ParameterBuilder = Callable[[ResearchParameters], Strategy]


@dataclass(frozen=True, slots=True)
class StrategyDefinition:
    """Implementation metadata; registration does not read defaults or grids."""

    strategy_id: str
    builder: ParameterBuilder
    strategy_version: str
    default_available: bool = False

    def build(self, parameters: Mapping[str, object] | None = None) -> Strategy:
        if parameters is None:
            if not self.default_available:
                raise ValueError(
                    f"Strategy {self.strategy_id!r} has no default; supply a complete --params-file"
                )
            parameters = configuration.parameter_defaults(self.strategy_id)
        resolved = ResearchParameters.from_payload(parameters)
        if resolved.signal.family != self.strategy_id:
            raise ValueError(
                f"Strategy {self.strategy_id!r} requires signal.family={self.strategy_id!r}"
            )
        try:
            strategy = self.builder(resolved)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid parameters for {self.strategy_id}: {exc}") from exc
        if strategy.strategy_id != self.strategy_id or strategy.strategy_version != self.strategy_version:
            raise RuntimeError(f"Strategy builder identity mismatch for {self.strategy_id}")
        return strategy


_STRATEGY_REGISTRY: Mapping[str, StrategyDefinition] = MappingProxyType({
    "momentum": StrategyDefinition(
        "momentum", MomentumStrategy,
        MomentumStrategy.strategy_version, default_available=True,
    ),
    "reversal": StrategyDefinition(
        "reversal", ReversalStrategy,
        ReversalStrategy.strategy_version,
    ),
    "monthly_trend": StrategyDefinition(
        "monthly_trend", MonthlyTrendStrategy,
        MonthlyTrendStrategy.strategy_version,
    ),
    "low_volatility": StrategyDefinition(
        "low_volatility", LowVolatilityStrategy,
        LowVolatilityStrategy.strategy_version,
    ),
    "sector_momentum": StrategyDefinition(
        "sector_momentum", SectorMomentumStrategy,
        SectorMomentumStrategy.strategy_version,
    ),
})


def available_strategy_ids() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGY_REGISTRY))


def get_strategy_definition(strategy_id: str) -> StrategyDefinition:
    normalized = str(strategy_id).strip()
    try:
        return _STRATEGY_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown strategy {normalized!r}. Available strategies: "
            f"{list(available_strategy_ids())}"
        ) from exc


def build_registered_strategy(
    strategy_id: str, parameters: Mapping[str, object] | None = None,
) -> Strategy:
    return get_strategy_definition(strategy_id).build(parameters)


def load_parameter_packet(path: str | Path | None) -> Mapping[str, object] | None:
    """Read a complete raw packet or saved identity wrapper."""
    if path is None:
        return None
    parameter_path = Path(path)
    if not parameter_path.is_file():
        raise FileNotFoundError(f"Missing strategy parameter file: {parameter_path}")
    try:
        payload = read_json_object(parameter_path)
    except ValueError as exc:
        raise ValueError(f"Invalid strategy parameter file {parameter_path}: {exc}") from exc
    return payload


def _unwrap_parameters(strategy_id: str, packet: Mapping[str, object]) -> Mapping[str, object]:
    identity_keys = {"strategy_id", "strategy_version", "parameters"}
    if not identity_keys.intersection(packet):
        return packet
    if set(packet) != identity_keys or not isinstance(packet["parameters"], dict):
        raise ValueError(
            "Saved parameter wrapper requires exactly strategy_id, strategy_version, parameters"
        )
    definition = get_strategy_definition(strategy_id)
    if packet["strategy_id"] != strategy_id:
        raise ValueError(
            f"Parameter packet declares strategy {packet['strategy_id']!r}; expected {strategy_id!r}"
        )
    if packet["strategy_version"] != definition.strategy_version:
        from .research_parameters import SCHEMA_MIGRATION
        raise ValueError(SCHEMA_MIGRATION)
    return packet["parameters"]


def build_strategy_from_file(
    strategy_id: str, parameter_path: str | Path | None,
) -> Strategy:
    packet = load_parameter_packet(parameter_path)
    return build_registered_strategy(
        strategy_id, None if packet is None else _unwrap_parameters(strategy_id, packet),
    )


def strategy_parameters_payload(strategy: Strategy) -> dict[str, Any]:
    return {
        "strategy_id": strategy.strategy_id,
        "strategy_version": strategy.strategy_version,
        "parameters": strategy.parameters.payload(),
    }


def save_strategy_parameters(strategy: Strategy, path: str | Path) -> None:
    atomic_write_text(
        json.dumps(strategy_parameters_payload(strategy), indent=2, sort_keys=True) + "\n",
        Path(path),
    )


def load_saved_strategy_parameters(strategy_id: str, path: str | Path) -> dict[str, Any]:
    packet = read_saved_strategy_identity(strategy_id, path)
    strategy = build_registered_strategy(strategy_id, packet["parameters"])
    return strategy_parameters_payload(strategy)


def read_saved_strategy_identity(strategy_id: str, path: str | Path) -> Mapping[str, object]:
    """Read saved attribution metadata without loading defaults or building a strategy."""
    packet = load_parameter_packet(path)
    if packet is None or set(packet) != {"strategy_id", "strategy_version", "parameters"}:
        raise ValueError(f"Missing saved strategy identity: {path}")
    _unwrap_parameters(strategy_id, packet)
    return MappingProxyType(dict(packet))


__all__ = [
    "StrategyDefinition", "available_strategy_ids", "build_registered_strategy",
    "build_strategy_from_file", "get_strategy_definition", "load_parameter_packet",
    "load_saved_strategy_parameters", "read_saved_strategy_identity",
    "save_strategy_parameters", "strategy_parameters_payload",
]
