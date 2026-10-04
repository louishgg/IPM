"""Tests for prepared sector assignment contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from portfolio_core.sector_assignments import (
    ASSIGNMENT_COLUMNS,
    prior_sector_audit_fields,
    sector_rows_asof,
    trade_sector_audit_fields,
    validate_sector_assignments,
)


def test_assignment_contract_rejects_duplicates_and_unsupported_codes():
    row = {
        "As_Of_Date": "2020-01-31",
        "Asset_ID": "AAA.N",
        "GICS_Sector_Code": "10",
        "Sector": "Energy",
        "Source_Type": "Wikipedia",
        "Source_Reference": "123",
        "Source_Symbol": "AAA",
        "Resolution_Method": "exact_wikipedia_symbol",
    }
    duplicate = pd.DataFrame([row, row], columns=ASSIGNMENT_COLUMNS)
    with pytest.raises(ValueError, match="duplicate"):
        validate_sector_assignments(duplicate)
    unsupported = pd.DataFrame([{**row, "GICS_Sector_Code": "99"}])
    with pytest.raises(ValueError, match="Unsupported"):
        validate_sector_assignments(unsupported.loc[:, ASSIGNMENT_COLUMNS])


def test_prior_sector_lookup_is_strictly_causal_and_same_asset_only():
    rows = validate_sector_assignments(
        pd.DataFrame(
            [
                {
                    "As_Of_Date": date,
                    "Asset_ID": asset_id,
                    "GICS_Sector_Code": "10",
                    "Sector": "Energy",
                    "Source_Type": "Wikipedia",
                    "Source_Reference": reference,
                    "Source_Symbol": asset_id,
                    "Resolution_Method": "exact_wikipedia_symbol",
                }
                for date, asset_id, reference in (
                    ("2020-01-31", "AAA.N", "100"),
                    ("2020-02-29", "AAA.N", "101"),
                    ("2020-03-31", "BBB.N", "102"),
                )
            ],
            columns=ASSIGNMENT_COLUMNS,
        )
    )

    fields = prior_sector_audit_fields(
        rows,
        "AAA.N",
        "2020-03-31",
        context="test exit",
    )
    assert fields["Sector_As_Of_Date"] == pd.Timestamp("2020-02-29")
    assert fields["Sector_Source_Reference"] == "101"

    with pytest.raises(RuntimeError, match="Missing prior causal"):
        prior_sector_audit_fields(
            rows,
            "BBB.N",
            "2020-03-31",
            context="test exit",
        )
    with pytest.raises(RuntimeError, match="Missing prior causal"):
        prior_sector_audit_fields(
            rows,
            "MISSING.N",
            "2020-04-30",
            context="test exit",
        )


def _trade_sector_assignments() -> pd.DataFrame:
    return validate_sector_assignments(
        pd.DataFrame(
            [
                {
                    "As_Of_Date": date,
                    "Asset_ID": "AAA.N",
                    "GICS_Sector_Code": "10",
                    "Sector": "Energy",
                    "Source_Type": "Wikipedia",
                    "Source_Reference": reference,
                    "Source_Symbol": "AAA",
                    "Resolution_Method": "exact_wikipedia_symbol",
                }
                for date, reference in (
                    ("2020-01-31", "100"),
                    ("2020-02-29", "101"),
                    ("2020-03-31", "102"),
                )
            ],
            columns=ASSIGNMENT_COLUMNS,
        )
    )


def test_trade_sector_audit_uses_exact_execution_date_for_eligible_asset():
    assignments = _trade_sector_assignments()
    execution_date = pd.Timestamp("2020-03-31")

    fields = trade_sector_audit_fields(
        assignments=assignments,
        execution_rows=sector_rows_asof(assignments, execution_date),
        execution_date=execution_date,
        execution_eligible=True,
        asset_id="AAA.N",
        current_value=0.0,
        target_value=100.0,
        domain="backtest",
    )

    assert fields == {
        "GICS_Sector_Code": "10",
        "Sector": "Energy",
        "Sector_As_Of_Date": execution_date,
        "Sector_Source_Type": "Wikipedia",
        "Sector_Source_Reference": "102",
    }


def test_trade_sector_audit_full_forced_exit_uses_latest_strictly_prior_row():
    assignments = _trade_sector_assignments()
    execution_date = pd.Timestamp("2020-03-31")

    fields = trade_sector_audit_fields(
        assignments=assignments,
        execution_rows=sector_rows_asof(assignments, execution_date),
        execution_date=execution_date,
        execution_eligible=False,
        asset_id="AAA.N",
        current_value=100.0,
        target_value=0.0,
        domain="live",
    )

    assert fields == {
        "GICS_Sector_Code": "10",
        "Sector": "Energy",
        "Sector_As_Of_Date": pd.Timestamp("2020-02-29"),
        "Sector_Source_Type": "Wikipedia",
        "Sector_Source_Reference": "101",
    }


@pytest.mark.parametrize(
    ("current_value", "target_value"),
    (
        (100.0, 50.0),
        (0.0, 100.0),
        (100.0, 100.0),
        (100.0, -100.0),
    ),
    ids=("partial-exit", "opening", "retained", "sign-reversal"),
)
@pytest.mark.parametrize("domain", ("backtest", "live"))
def test_trade_sector_audit_rejects_invalid_ineligible_transition_exactly(
    current_value,
    target_value,
    domain,
):
    assignments = _trade_sector_assignments()
    execution_date = pd.Timestamp("2020-03-31")

    with pytest.raises(RuntimeError) as error:
        trade_sector_audit_fields(
            assignments=assignments,
            execution_rows=sector_rows_asof(assignments, execution_date),
            execution_date=execution_date,
            execution_eligible=False,
            asset_id="AAA.N",
            current_value=current_value,
            target_value=target_value,
            domain=domain,
        )

    assert str(error.value) == (
        f"An ineligible {domain} asset may only be closed completely: "
        f"AAA.N on 2020-03-31 has current={current_value}, "
        f"target={target_value}"
    )
