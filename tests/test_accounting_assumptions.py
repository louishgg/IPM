"""Price-basis and deterministic accounting-assumption artifact contracts."""

from dataclasses import replace
import hashlib
import json

import pandas as pd
import pytest

from portfolio_core.accounting_config import DEFAULT_ACCOUNTING_CONFIG
from portfolio_core.accounting_ledger import project_financing_amounts
from portfolio_core.rebalance_planner import plan_rebalance_targets
from portfolio_core.price_basis import (
    price_basis_frame,
    price_basis_spec,
    validate_price_basis,
)
from portfolio_core.simulation_assumptions import (
    ACCOUNTING_MODEL_ID,
    ASSUMPTIONS_SCHEMA_VERSION,
    SPREAD_MODEL_ID,
    build_simulation_assumptions,
    load_simulation_assumptions,
    write_simulation_assumptions,
)


def test_prepared_price_basis_is_one_exact_domain_specific_row():
    backtest = price_basis_frame("backtest")
    live = price_basis_frame("live")

    assert len(backtest) == len(live) == 1
    assert (
        backtest.at[0, "Price_Basis_ID"]
        == "reuters_yahoo_wiki_price_return_v1"
    )
    assert validate_price_basis(backtest, "backtest") == price_basis_spec(
        "backtest"
    )
    assert validate_price_basis(live, "live") == price_basis_spec("live")
    assert backtest.at[0, "Ordinary_Dividend_Treatment"] == "omitted"
    assert (
        live.at[0, "Ordinary_Dividend_Treatment"]
        == "embedded once in adjusted closes or event-return factors"
    )
    with pytest.raises(ValueError, match="inconsistent"):
        validate_price_basis(
            live.assign(Price_Basis_ID="raw_yahoo_prices"),
            "live",
        )


def test_simulation_assumptions_are_complete_stable_and_tamper_evident(tmp_path):
    path = tmp_path / "simulation_assumptions.json"
    first = write_simulation_assumptions(
        path,
        DEFAULT_ACCOUNTING_CONFIG,
        price_basis_spec("backtest"),
    )
    second = build_simulation_assumptions(
        DEFAULT_ACCOUNTING_CONFIG,
        price_basis_spec("backtest"),
    )

    assert first == second == load_simulation_assumptions(path)
    assert ASSUMPTIONS_SCHEMA_VERSION == 4
    assert ACCOUNTING_MODEL_ID == "portfolio_accounting_v4"
    assert first["accounting_model"] == ACCOUNTING_MODEL_ID
    assert first["spread"]["model"] == SPREAD_MODEL_ID
    assert first["rules"]["initial_capital"] == 1_000_000.0
    assert first["rules"]["maximum_gross_exposure"] == 2.0
    assert "maximum_position_weight" not in first["rules"]
    assert first["financing_conventions"]["day_count"] == "actual/365"
    assert first["execution_conventions"]["sign_flip_orders"] == 2
    assert (
        first["execution_conventions"]["minimum_entry_price_rule"]
        == "omitted_for_adjusted_price_basis"
    )
    assert first["spread"]["parameters"]["liquidity_bps"] == 7.0
    assert (
        first["spread"]["tick_size_floor"]
        == "omitted_for_adjusted_price_basis"
    )

    low_engine = replace(
        DEFAULT_ACCOUNTING_CONFIG,
        transaction_costs=replace(
            DEFAULT_ACCOUNTING_CONFIG.transaction_costs,
            liquidity_bps=3.5,
        ),
    )
    low = build_simulation_assumptions(
        low_engine,
        price_basis_spec("backtest"),
    )
    assert low["simulation_fingerprint"] != first["simulation_fingerprint"]

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["rules"]["fee_per_trade"] = 3.0
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="does not reconcile"):
        load_simulation_assumptions(path)


@pytest.mark.parametrize("domain,expected", (
    ("backtest", "7726fbf8fb7bc1307ec83b7e4c1141aeaefe9ac486100bcdd3a6e31be7b42b70"),
    ("live", "978f9c08021695fe5a2159b434aa261ec61e7bd9d4fd863b4527935c561c0636"),
))
def test_default_assumption_payload_retains_its_existing_fingerprint(domain, expected):
    # Captured before configuration-derived descriptions replaced fixed labels.
    payload = build_simulation_assumptions(DEFAULT_ACCOUNTING_CONFIG, price_basis_spec(domain))
    assert payload["simulation_fingerprint"] == expected


@pytest.mark.parametrize("domain", ("backtest", "live"))
@pytest.mark.parametrize("day_count", (365, 360))
@pytest.mark.parametrize("whole_shares", (True, False))
def test_assumption_overrides_describe_actual_financing_and_sizing(
    tmp_path, domain, day_count, whole_shares,
):
    config = replace(DEFAULT_ACCOUNTING_CONFIG, day_count_days=day_count, whole_share_orders=whole_shares)
    path = tmp_path / "assumptions.json"
    payload = write_simulation_assumptions(path, config, price_basis_spec(domain))
    assert payload == load_simulation_assumptions(path)
    assert payload == build_simulation_assumptions(config, price_basis_spec(domain))
    assert payload["financing_conventions"]["day_count"] == f"actual/{day_count}"
    assert payload["financing_conventions"]["compounding"] == f"daily_apr_over_{day_count}"
    assert payload["execution_conventions"]["voluntary_share_rounding"] == (
        "nearest_half_away_from_zero" if whole_shares else "none"
    )
    assert payload["execution_conventions"]["whole_share_scope"] == (
        "voluntary_orders_only" if whole_shares else "disabled"
    )
    for cash, rate in ((1_000_000., .02), (-1_000_000., .08)):
        _, interest, _, _ = project_financing_amounts(cash, 0., "2020-01-01", "2020-01-31", config)
        assert interest == pytest.approx(cash * ((1 + rate / day_count) ** 30 - 1))
    targets = plan_rebalance_targets(
        pd.Series(dtype=float), pd.Series({"L": .5, "S": -.5}),
        pd.Series({"L": 3., "S": 3.}), 100., turnover_threshold=0.,
        whole_share_orders=config.whole_share_orders,
    )
    units = 17. if whole_shares else 50. / 3.
    assert targets.applied_shares.to_dict() == pytest.approx({"L": units, "S": -units})
    if day_count != 365 or not whole_shares:
        original = build_simulation_assumptions(DEFAULT_ACCOUNTING_CONFIG, price_basis_spec(domain))
        assert payload["simulation_fingerprint"] != original["simulation_fingerprint"]


def test_assumptions_require_typed_price_basis_and_serialize_dividend_treatment():
    with pytest.raises(TypeError, match="PriceBasisSpec"):
        build_simulation_assumptions(
            DEFAULT_ACCOUNTING_CONFIG,
            "reuters_price_return",  # type: ignore[arg-type]
        )

    backtest = build_simulation_assumptions(
        DEFAULT_ACCOUNTING_CONFIG,
        price_basis_spec("backtest"),
    )
    assert backtest["price_basis"]["ordinary_dividend_treatment"] == "omitted"


def test_unsupported_assumption_schema_is_rejected_with_a_valid_fingerprint(tmp_path):
    path = tmp_path / "simulation_assumptions.json"
    payload = build_simulation_assumptions(
        DEFAULT_ACCOUNTING_CONFIG,
        price_basis_spec("backtest"),
    )
    payload.pop("simulation_fingerprint")
    payload["schema_version"] = ASSUMPTIONS_SCHEMA_VERSION - 1
    payload["simulation_fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported simulation-assumptions schema"):
        load_simulation_assumptions(path)


@pytest.mark.parametrize("row_count", (0, 2), ids=("missing", "extra"))
def test_price_basis_validator_requires_exactly_one_row(row_count):
    frame = price_basis_frame("backtest")
    invalid = (
        frame.iloc[0:0].copy()
        if row_count == 0
        else pd.concat([frame] * row_count, ignore_index=True)
    )
    with pytest.raises(ValueError, match="exactly one"):
        validate_price_basis(invalid, "backtest")
