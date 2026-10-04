"""Versioned research packets with fixed portfolio exposure.

JSON grids own search choices. These schemas validate mathematical domains,
supported algorithms and compatibility, without duplicating candidate lists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
from math import isfinite
from typing import Mapping

SCHEMA_VERSION = "2.0.0"
SCHEMA_MIGRATION = (
    "Research schema version migration: 2.0.0 requires fixed exposure {gross, long_share}; "
    "remove mode, target_volatility, reference_window, reference_contract and "
    "startup_policy from packets, and exposure_profiles from grids. "
    "Use the retained (60,24) profile in supplied selection/sizing configurations."
)
SIGNAL_FAMILIES = (
    "momentum", "low_volatility", "reversal", "sector_momentum", "monthly_trend",
)


def _integer(value: int, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _choice(value: object, choices: tuple, name: str) -> None:
    if isinstance(value, bool) or value not in choices:
        raise ValueError(f"Unsupported {name}: {value!r}; expected {choices}")


def _number(value: object, name: str) -> None:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")


@dataclass(frozen=True, slots=True)
class StockSelection:
    """Exact stock counts. Small counts remain usable by historical fixtures."""

    n_long: int
    n_short: int

    def __post_init__(self) -> None:
        _integer(self.n_long, "n_long")
        _integer(self.n_short, "n_short")


@dataclass(frozen=True, slots=True)
class WholeSectorSelection:
    """Sector counts are independent; stock counts are variable with a 10/10 floor."""

    n_long_sectors: int
    n_short_sectors: int

    def __post_init__(self) -> None:
        _integer(self.n_long_sectors, "n_long_sectors")
        _integer(self.n_short_sectors, "n_short_sectors")


@dataclass(frozen=True, slots=True)
class VolatilityProfile:
    window: int
    minimum_observations: int

    def __post_init__(self) -> None:
        # Sample standard deviation needs at least two observations.
        _integer(self.window, "window", 2)
        _integer(self.minimum_observations, "minimum_observations", 2)
        if self.minimum_observations > self.window:
            raise ValueError("minimum_observations cannot exceed the volatility window")


@dataclass(frozen=True, slots=True)
class SignalParameters:
    family: str
    formation_months: int | None
    skip_months: int | None
    selection_volatility: VolatilityProfile | None
    moving_average_months: int | None

    def __post_init__(self) -> None:
        _choice(self.family, SIGNAL_FAMILIES, "family")
        if self.selection_volatility is not None and not isinstance(
            self.selection_volatility, VolatilityProfile
        ):
            raise ValueError("selection_volatility must be a linked VolatilityProfile")
        if self.family in ("momentum", "reversal", "sector_momentum"):
            _integer(self.formation_months, "formation_months")
            _integer(self.skip_months, "skip_months", 0)
            if (
                self.selection_volatility is not None
                or self.moving_average_months is not None
            ):
                raise ValueError("Inactive signal fields must be null")
        elif self.family == "low_volatility":
            if self.selection_volatility is None or any(
                v is not None
                for v in (
                    self.formation_months,
                    self.skip_months,
                    self.moving_average_months,
                )
            ):
                raise ValueError(
                    "Low-volatility selection requires only its linked profile"
                )
        else:
            # A one-price average would give the same zero score to every stock.
            _integer(self.moving_average_months, "moving_average_months", 2)
            if any(
                v is not None
                for v in (
                    self.formation_months,
                    self.skip_months,
                    self.selection_volatility,
                )
            ):
                raise ValueError("Moving-average ranking requires only its SMA window")


@dataclass(frozen=True, slots=True)
class SizingParameters:
    method: str
    volatility: VolatilityProfile | None

    def __post_init__(self) -> None:
        _choice(self.method, ("equal", "inverse_volatility"), "sizing method")
        if self.volatility is not None and not isinstance(
            self.volatility, VolatilityProfile
        ):
            raise ValueError("Sizing volatility must be a linked VolatilityProfile")
        if self.method == "inverse_volatility" and self.volatility is None:
            raise ValueError("Inverse-volatility sizing requires a profile")


@dataclass(frozen=True, slots=True)
class BufferParameters:
    enabled: bool
    exit_multiplier: float | None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("Buffer enabled must be boolean")
        if self.enabled or self.exit_multiplier is not None:
            _number(self.exit_multiplier, "exit multiplier")
            if self.exit_multiplier < 1:
                raise ValueError("exit multiplier must be >= 1")

    def additional_pool(self, count: int) -> int:
        """Uncapped extra ranks, using the approved ties-to-even convention."""
        _integer(count, "count")
        return (
            max(0, int(round((self.exit_multiplier - 1) * count)))
            if self.enabled
            else 0
        )


@dataclass(frozen=True, slots=True)
class ExposureParameters:
    """Fixed gross target and long-side share of that target."""

    gross: float
    long_share: float

    def __post_init__(self) -> None:
        _number(self.gross, "gross target")
        if self.gross <= 0:
            raise ValueError("gross target must be positive")
        _number(self.long_share, "long share")
        if not 0 < self.long_share < 1:
            raise ValueError("long share must be strictly between 0 and 1")


@dataclass(frozen=True, slots=True)
class ResearchParameters:
    """Complete requested packet with an explicit effective representation.

    `gross` is a fixed exposure target, subject to account execution constraints.
    No H, cash-timing, sign gates, or simultaneous stock/sector axes are accepted.
    """

    signal: SignalParameters
    selection: StockSelection | WholeSectorSelection
    sizing: SizingParameters
    buffer: BufferParameters
    exposure: ExposureParameters
    sector_neutral: bool
    turnover_threshold: float
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value, kind in (
            (self.signal, SignalParameters),
            (self.selection, (StockSelection, WholeSectorSelection)),
            (self.sizing, SizingParameters),
            (self.buffer, BufferParameters),
            (self.exposure, ExposureParameters),
        ):
            if not isinstance(value, kind):
                raise ValueError("Research parameters require typed component packets")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(SCHEMA_MIGRATION)
        if type(self.sector_neutral) is not bool:
            raise ValueError("sector_neutral must be boolean")
        _number(self.turnover_threshold, "turnover threshold")
        if self.turnover_threshold < 0:
            raise ValueError("turnover threshold cannot be negative")
        sector = isinstance(self.selection, WholeSectorSelection)
        if sector != (self.signal.family == "sector_momentum"):
            raise ValueError(
                "Sector momentum requires whole-sector selection; stock axes are inactive"
            )
        if self.sector_neutral and (sector or self.exposure.long_share != 0.5):
            raise ValueError(
                "Sector neutrality requires stock selection and long_share=0.5"
            )

    def effective(self) -> ResearchParameters:
        return replace(
            self,
            sizing=(
                replace(self.sizing, volatility=None)
                if self.sizing.method == "equal"
                else self.sizing
            ),
            buffer=(
                replace(self.buffer, exit_multiplier=None)
                if not self.buffer.enabled
                else self.buffer
            ),
            turnover_threshold=float(self.turnover_threshold),
            exposure=replace(
                self.exposure,
                gross=float(self.exposure.gross),
                long_share=float(self.exposure.long_share),
            ),
        )

    def payload(self, *, effective: bool = False) -> dict:
        result = asdict(self.effective() if effective else self)
        result["selection"]["kind"] = (
            "whole_sector"
            if isinstance(self.selection, WholeSectorSelection)
            else "stocks"
        )
        if effective and self.buffer.enabled:
            counts = (
                (self.selection.n_long, self.selection.n_short)
                if isinstance(self.selection, StockSelection)
                else (self.selection.n_long_sectors, self.selection.n_short_sectors)
            )
            result["buffer"] = {
                "enabled": True,
                "additional_long_pool": self.buffer.additional_pool(counts[0]),
                "additional_short_pool": self.buffer.additional_pool(counts[1]),
            }
        return result

    def canonical_json(self) -> str:
        """Effective parameter identity only; candidate IDs also need fingerprints."""
        return json.dumps(
            self.payload(effective=True),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_payload(cls, payload: Mapping) -> ResearchParameters:
        """Strict requested-packet reader: no promoted or implicit nested defaults."""

        if isinstance(payload, Mapping):
            exposure = payload.get("exposure", {})
            if (payload.get("schema_version") != SCHEMA_VERSION or
                    isinstance(exposure, Mapping) and set(exposure) - {"gross", "long_share"}):
                raise ValueError(SCHEMA_MIGRATION)

        def read(kind, data, nested=None, path="parameters"):
            if not isinstance(data, Mapping):
                raise ValueError(f"{path} must be a JSON object")
            expected = {f.name for f in fields(kind)}
            if set(data) != expected:
                raise ValueError(
                    f"{path} requires complete {kind.__name__} fields: "
                    f"missing={sorted(expected - set(data))}, "
                    f"unknown={sorted(set(data) - expected)}"
                )
            values = dict(data)
            for key, parser in (nested or {}).items():
                if values[key] is not None:
                    values[key] = parser(values[key])
            try:
                return kind(**values)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: {exc}") from exc

        def selection(data):
            if not isinstance(data, Mapping):
                raise ValueError("Selection must be an object")
            values = dict(data)
            kind = values.pop("kind", None)
            if kind not in ("stocks", "whole_sector"):
                raise ValueError("Unknown selection kind")
            return read(
                StockSelection if kind == "stocks" else WholeSectorSelection,
                values, path="parameters.selection",
            )

        signal_profile = lambda data: read(
            VolatilityProfile, data, path="parameters.signal.selection_volatility"
        )
        sizing_profile = lambda data: read(
            VolatilityProfile, data, path="parameters.sizing.volatility"
        )
        return read(
            cls,
            payload,
            {
                "signal": lambda data: read(
                    SignalParameters, data, {"selection_volatility": signal_profile},
                    "parameters.signal",
                ),
                "selection": selection,
                "sizing": lambda data: read(
                    SizingParameters, data, {"volatility": sizing_profile},
                    "parameters.sizing",
                ),
                "buffer": lambda data: read(
                    BufferParameters, data, path="parameters.buffer"
                ),
                "exposure": lambda data: read(
                    ExposureParameters, data, path="parameters.exposure"
                ),
            },
        )
