"""Clean-break tests for the supported strategy-explicit backtest API."""

import inspect
from typing import get_type_hints

import backtest
from backtest.data_loading import BacktestDataset
from backtest.engine import run_backtest
from portfolio_core.accounting_config import (
    DEFAULT_ACCOUNTING_CONFIG,
    PortfolioAccountingConfig,
)
from portfolio_core.strategies.strategy_contract import Strategy


def test_package_export_and_signature_require_an_explicit_strategy():
    assert backtest.__all__ == ["run_backtest"]
    assert backtest.run_backtest is run_backtest

    parameters = inspect.signature(backtest.run_backtest).parameters
    assert list(parameters) == [
        "backtest_data",
        "dates",
        "strategy",
        "accounting_config",
        "apply_turnover_threshold",
        "apply_fees",
        "apply_spread",
        "return_diagnostics",
        "return_holdings",
        "debug",
    ]
    assert [parameter.kind for parameter in parameters.values()] == [
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        *([inspect.Parameter.KEYWORD_ONLY] * 7),
    ]
    assert parameters["strategy"].default is inspect.Parameter.empty
    assert parameters["accounting_config"].default == DEFAULT_ACCOUNTING_CONFIG
    assert {
        name: parameters[name].default
        for name in (
            "apply_turnover_threshold",
            "apply_fees",
            "apply_spread",
            "return_diagnostics",
            "return_holdings",
            "debug",
        )
    } == {
        "apply_turnover_threshold": True,
        "apply_fees": True,
        "apply_spread": True,
        "return_diagnostics": False,
        "return_holdings": False,
        "debug": False,
    }
    assert get_type_hints(backtest.run_backtest) == {
        "backtest_data": BacktestDataset,
        "strategy": Strategy,
        "accounting_config": PortfolioAccountingConfig,
    }
