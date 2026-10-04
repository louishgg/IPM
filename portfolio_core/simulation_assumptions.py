"""Deterministic persistence and validation of simulation-accounting rules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .io import atomic_write_bytes
from .accounting_config import PortfolioAccountingConfig
from .price_basis import PriceBasisSpec, price_basis_spec


ASSUMPTIONS_SCHEMA_VERSION = 4
ACCOUNTING_MODEL_ID = "portfolio_accounting_v4"
SPREAD_MODEL_ID = "fixed_adv_half_spread_v2"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def build_simulation_assumptions(
    accounting: PortfolioAccountingConfig,
    price_basis: PriceBasisSpec,
) -> dict[str, object]:
    """Build complete rules plus a fingerprint independent of strategy choice."""

    if not isinstance(price_basis, PriceBasisSpec):
        raise TypeError("price_basis must be a validated PriceBasisSpec")
    accounting_values = asdict(accounting)
    transaction_costs = dict(accounting_values.pop("transaction_costs"))
    fingerprint_input = {
        "schema_version": ASSUMPTIONS_SCHEMA_VERSION,
        "accounting_model": ACCOUNTING_MODEL_ID,
        "rules": accounting_values,
        "execution_conventions": {
            "voluntary_share_rounding": (
                "nearest_half_away_from_zero" if accounting.whole_share_orders else "none"
            ),
            "whole_share_scope": (
                "voluntary_orders_only" if accounting.whole_share_orders else "disabled"
            ),
            "minimum_entry_price_rule": "omitted_for_adjusted_price_basis",
            "sign_flip_orders": 2,
            "feasibility_adjustment": "largest_common_scale_factor",
            "constraint_timing": "immediately_after_trades",
            "automatic_margin_calls": False,
        },
        "financing_conventions": {
            "day_count": f"actual/{accounting.day_count_days}",
            "compounding": f"daily_apr_over_{accounting.day_count_days}",
            "interval": "[start_date,next_event_date)",
            "restricted_short_proceeds": "historical_transactional_proceeds",
            "daily_cent_rounding": False,
        },
        "spread": {
            "model": SPREAD_MODEL_ID,
            "side": "one_way_half_spread",
            "buy_cover": "ask",
            "sell_short": "bid",
            "tick_size_floor": "omitted_for_adjusted_price_basis",
            "missing_liquidity_bps": transaction_costs["max_bps"],
            "parameters": transaction_costs,
        },
        "price_basis": price_basis.as_assumptions(),
    }
    fingerprint = hashlib.sha256(_canonical_bytes(fingerprint_input)).hexdigest()
    return {
        **fingerprint_input,
        "simulation_fingerprint": fingerprint,
    }


def write_simulation_assumptions(
    path: Path,
    accounting: PortfolioAccountingConfig,
    price_basis: PriceBasisSpec,
) -> dict[str, object]:
    payload = build_simulation_assumptions(
        accounting,
        price_basis,
    )
    content = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    atomic_write_bytes(content.encode("utf-8"), Path(path))
    return payload


def load_simulation_assumptions(path: Path) -> dict[str, object]:
    assumptions_path = Path(path)
    if not assumptions_path.is_file():
        raise FileNotFoundError(
            f"Missing simulation assumptions: {assumptions_path}"
        )
    try:
        payload = json.loads(assumptions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unreadable simulation assumptions: {assumptions_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Simulation assumptions must be a JSON object")
    fingerprint = payload.get("simulation_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("Simulation assumptions have an invalid fingerprint")
    fingerprint_input = dict(payload)
    del fingerprint_input["simulation_fingerprint"]
    expected = hashlib.sha256(_canonical_bytes(fingerprint_input)).hexdigest()
    if fingerprint != expected:
        raise ValueError("Simulation-assumption fingerprint does not reconcile")
    if payload.get("schema_version") != ASSUMPTIONS_SCHEMA_VERSION:
        raise ValueError("Unsupported simulation-assumptions schema version")
    basis = payload.get("price_basis")
    if not isinstance(basis, dict) or not isinstance(basis.get("domain"), str):
        raise ValueError("Simulation assumptions have an invalid price basis")
    expected_basis = price_basis_spec(basis["domain"]).as_assumptions()
    if basis != expected_basis:
        raise ValueError("Simulation assumptions have an inconsistent price basis")
    return payload


__all__ = [
    "ACCOUNTING_MODEL_ID",
    "ASSUMPTIONS_SCHEMA_VERSION",
    "SPREAD_MODEL_ID",
    "build_simulation_assumptions",
    "load_simulation_assumptions",
    "write_simulation_assumptions",
]
