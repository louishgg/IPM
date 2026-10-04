"""Conditional research-family grids and frozen, effective candidate identities."""

from hashlib import sha256
from itertools import product
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_core.strategies.research_parameters import (
    ResearchParameters,
    SignalParameters,
    StockSelection,
    WholeSectorSelection,
    SizingParameters,
    VolatilityProfile,
    BufferParameters,
    ExposureParameters,
    SCHEMA_VERSION,
    SCHEMA_MIGRATION,
    SIGNAL_FAMILIES,
)
from portfolio_core.strategies.configuration import read_json_object
from portfolio_core.simulation_assumptions import build_simulation_assumptions
from .config import DEFAULT_CONFIG


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _grid_dimensions(strategy_id):
    sector = strategy_id == "sector_momentum"
    return (
        "signal_profiles",
        "n_long_sectors" if sector else "n_long",
        "n_short_sectors" if sector else "n_short",
        "sizing_profiles",
        "buffer_profiles",
        "turnover_threshold",
        "sector_neutral",
        "long_share",
        "gross",
    )


def read_grid(path):
    """Validate schema shape without supplying or restricting JSON choices."""
    grid = read_json_object(path)
    family = grid.get("strategy_id")
    dimensions = _grid_dimensions(family)
    if grid.get("schema_version") != SCHEMA_VERSION or any(
        key in grid for key in ("reference_contract", "startup_policy", "exposure_profiles")
    ):
        raise ValueError(SCHEMA_MIGRATION)
    expected = set(dimensions) | {"schema_version", "strategy_id"}
    if (set(grid) != expected or family not in SIGNAL_FAMILIES
            or grid["schema_version"] != SCHEMA_VERSION):
        raise ValueError("Research grid requires the complete versioned conditional schema")
    for name in dimensions:
        if not isinstance(grid[name], list) or not grid[name]:
            raise ValueError(f"Empty or invalid grid axis {name}")
    signal_fields = {
        "momentum": {"formation_months", "skip_months"},
        "reversal": {"formation_months", "skip_months"},
        "sector_momentum": {"formation_months", "skip_months"},
        "monthly_trend": {"moving_average_months"},
        "low_volatility": {"selection_volatility"},
    }[family]
    nested_shapes = {
        "signal_profiles": signal_fields,
        "sizing_profiles": {"method", "volatility"},
        "buffer_profiles": {"enabled", "exit_multiplier"},
    }
    for name, shape in nested_shapes.items():
        for number, value in enumerate(grid[name], 1):
            if not isinstance(value, dict) or set(value) != shape:
                raise ValueError(
                    f"Incomplete {name}[{number}]: expected exactly {sorted(shape)}"
                )
    for name, nested in (
        ("sizing_profiles", "volatility"),
        ("signal_profiles", "selection_volatility"),
    ):
        for number, value in enumerate(grid[name], 1):
            if nested in value and value[nested] is not None and (
                not isinstance(value[nested], dict)
                or set(value[nested]) != {"window", "minimum_observations"}
            ):
                raise ValueError(
                    f"Incomplete {name}[{number}].{nested}: expected window and minimum_observations"
                )
    return grid


def expand_grid(grid):
    """Separate rejected requests, duplicate mappings and effective candidates."""
    if grid.get("schema_version") != SCHEMA_VERSION or any(
        key in grid for key in ("exposure_profiles", "reference_contract", "startup_policy")
    ):
        raise ValueError(SCHEMA_MIGRATION)
    sector = grid["strategy_id"] == "sector_momentum"
    if sector and ("n_long" in grid or "n_short" in grid):
        raise ValueError("Whole-sector grids cannot activate fixed stock counts")
    dimensions = _grid_dimensions(grid["strategy_id"])
    candidates, rejected, duplicates, seen = [], [], [], {}
    for requested_number, values in enumerate(
        product(*(grid[k] for k in dimensions)), 1
    ):
        signal, nl, ns, sizing, buffer, threshold, neutral, p, gross = values
        requested = dict(zip(dimensions, values))
        try:
            if set(sizing) != {"method", "volatility"}:
                raise ValueError(
                    "Sizing profiles require exactly method and volatility"
                )
            signal_fields = dict(formation_months=None, skip_months=None,
                                 moving_average_months=None, selection_volatility=None)
            signal_fields.update(signal)
            if signal_fields["selection_volatility"] is not None:
                signal_fields["selection_volatility"] = VolatilityProfile(
                    **signal_fields["selection_volatility"])
            packet = ResearchParameters(
                SignalParameters(grid["strategy_id"], **signal_fields),
                WholeSectorSelection(nl, ns) if sector else StockSelection(nl, ns),
                SizingParameters(
                    sizing["method"],
                    (
                        VolatilityProfile(**sizing["volatility"])
                        if sizing["volatility"] is not None
                        else None
                    ),
                ),
                BufferParameters(**buffer),
                ExposureParameters(gross=gross, long_share=p),
                neutral,
                threshold,
            )
        except (ValueError, TypeError) as exc:
            rejected.append(
                dict(
                    Requested_Number=requested_number,
                    Reason=str(exc),
                    Requested_JSON=canonical(requested),
                )
            )
            continue
        key = packet.canonical_json()
        if key in seen:
            duplicates.append(
                dict(
                    Requested_Number=requested_number,
                    Candidate_Number=seen[key],
                    Requested_JSON=canonical(packet.payload()),
                )
            )
            continue
        number = len(candidates) + 1
        seen[key] = number
        inactive = []
        if packet.sizing.method == "equal":
            inactive.append("sizing.volatility")
        if not packet.buffer.enabled:
            inactive.append("buffer.exit_multiplier")
        candidates.append(
            dict(
                Candidate_Number=number,
                Requested_Number=requested_number,
                Parameters_JSON=key,
                Requested_JSON=canonical(packet.payload()),
                Inactive_Dimensions=",".join(inactive),
            )
        )
    return candidates, rejected, duplicates


def frozen_identity(data, accounting, root):
    """Hash actual in-memory inputs, relevant source bytes and frozen assumptions."""
    root = Path(root)
    sources = sorted(
        p
        for folder in ("backtest", "portfolio_core")
        for p in (root / folder).rglob("*.py")
    )
    code = sha256()
    for path in sources:
        code.update(str(path.relative_to(root)).encode())
        code.update(path.read_bytes())
    data_hash = sha256()
    for name in (
        "data_close",
        "data_volume",
        "pit_matrix",
        "sector_assignments",
        "rolling_dollar_vol",
        "security_events",
        "security_event_legs",
        "security_event_sources",
        "event_delivery_executions",
    ):
        frame = getattr(data, name)
        data_hash.update(name.encode())
        data_hash.update(canonical([str(c) for c in frame.columns]).encode())
        data_hash.update(
            pd.util.hash_pandas_object(frame.astype(str), index=True).values.tobytes()
        )
    assumptions = build_simulation_assumptions(accounting, data.price_basis)
    data_hash.update(
        canonical(
            dict(
                asset_to_ticker=data.asset_to_ticker,
                historical_asset_to_ticker=data.historical_asset_to_ticker,
                valid_trading_days=[str(d) for d in data.valid_trading_days],
            )
        ).encode()
    )
    return dict(
        schema_version=SCHEMA_VERSION,
        runtime_versions=dict(
            python=sys.version, numpy=np.__version__, pandas=pd.__version__
        ),
        code_fingerprint=code.hexdigest(),
        data_fingerprint=data_hash.hexdigest(),
        assumptions_fingerprint=assumptions["simulation_fingerprint"],
        assumptions=assumptions,
    )


def candidate_id(row, identity):
    return sha256(
        canonical(
            dict(effective_parameters=json.loads(row["Parameters_JSON"]), **identity)
        ).encode()
    ).hexdigest()


def readiness_bounds(packet, data, *, evaluation_config=DEFAULT_CONFIG.evaluation):
    """Earliest possible construction from actual signal/sizing coverage.

    This is a necessary history gate, never a claim that future buffered,
    neutral or rounded execution paths are feasible.
    """
    f, s = packet.signal.formation_months, packet.signal.skip_months
    prices = data.data_close.loc[:pd.Timestamp(evaluation_config.validation_end_date)]
    research_start = pd.Timestamp(evaluation_config.initial_research_start_date)
    from portfolio_core.strategies.portfolio_construction import stock_volatility
    if packet.signal.family == "sector_momentum":
        from .sector_returns import build_sector_returns
        history = data.sector_return_history or build_sector_returns(
            data, prices.index[-1], start=research_start,
        )
        returns = history.returns.reindex(prices.index)
        finite_sector = returns.where(np.isfinite(returns)).rolling(f, min_periods=f).count().shift(s).eq(f)
        ready_signal = finite_sector.sum(axis=1).ge(packet.selection.n_long_sectors + packet.selection.n_short_sectors)
        signal_dates = prices.index[ready_signal & (prices.index >= research_start)]
        enough = ready_signal.copy()
        if packet.sizing.method == 'inverse_volatility':
            vol = stock_volatility(prices, packet.sizing.volatility)
            membership = data.pit_matrix.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)
            enough &= (vol.gt(0) & np.isfinite(vol) & membership).sum(axis=1).ge(20)
        dates = prices.index[enough & (prices.index >= research_start)]
        return _bounds_result(signal_dates, dates, evaluation_config)
    if packet.signal.family == "low_volatility":
        signals = stock_volatility(prices, packet.signal.selection_volatility)
    elif packet.signal.family == "monthly_trend":
        from portfolio_core.strategies.monthly_trend import monthly_sma_scores
        signals = monthly_sma_scores(prices, packet.signal.moving_average_months)
    else:
        signals = prices.shift(s) / prices.shift(f + s) - 1
    finite = signals.notna() & signals.abs().lt(float("inf"))
    membership = (
        data.pit_matrix.reindex(index=prices.index, columns=prices.columns)
        .fillna(False)
        .astype(bool)
    )
    signal_ready = (finite & membership).sum(
        axis=1
    ) >= packet.selection.n_long + packet.selection.n_short
    signal_dates = prices.index[
        signal_ready & (prices.index >= research_start)
    ]
    if packet.sizing.method == "inverse_volatility":
        profile = packet.sizing.volatility
        vol = stock_volatility(prices, profile)
        finite &= vol.gt(0) & vol.lt(float("inf"))
    finite &= membership
    enough = finite.sum(axis=1) >= packet.selection.n_long + packet.selection.n_short
    dates = prices.index[
        enough
        & (prices.index >= research_start)
    ]
    return _bounds_result(signal_dates, dates, evaluation_config)


def _bounds_result(signal_dates, dates, evaluation):
    start = dates[0] if len(dates) else None
    return dict(
        First_Signal_Lower_Bound=(
            str(signal_dates[0].date()) if len(signal_dates) else None
        ),
        First_Construction_Lower_Bound=str(start.date()) if start is not None else None,
        History_Unready=start is None or start > (
            pd.Timestamp(evaluation.validation_start_date) - pd.offsets.MonthEnd(1)
        ),
        Readiness_Basis="necessary_history_bound_not_feasibility",
    )
