"""Tests for typed defaults, immutability, and validation."""

from dataclasses import FrozenInstanceError
from datetime import date

import pandas as pd
import pytest

from backtest.config import (
    BACKTEST_WARMUP_MONTHS,
    BacktestConfig,
    BacktestBrinsonConfig,
    BacktestMarketConfig,
    DEFAULT_CONFIG,
    BacktestEvaluationConfig,
)
from data_acquisition.engine import PRODUCTION_ACQUISITION_POLICY
from live.config import (
    DEFAULT_CONFIG as LIVE_DEFAULT_CONFIG,
    LiveBrinsonConfig,
    LiveConfig,
)
from portfolio_core.accounting_config import TransactionCostConfig
from _strategy_test_helpers import momentum_test_strategy


def test_default_values_match_the_extended_monthly_workflow():
    config = DEFAULT_CONFIG
    assert BACKTEST_WARMUP_MONTHS == 12
    assert config.market.start_date == date(2014, 1, 31)
    assert config.market.end_date == date(2026, 1, 31)
    assert config.evaluation.signal_warmup_start_date == date(2014, 1, 31)
    assert config.evaluation.signal_warmup_end_date == date(2014, 12, 31)
    assert config.evaluation.initial_research_start_date == date(2015, 1, 31)
    assert config.evaluation.initial_research_end_date == date(2017, 12, 31)
    assert config.evaluation.validation_start_date == date(2018, 1, 31)
    assert config.evaluation.validation_end_date == date(2023, 12, 31)
    assert config.evaluation.test_start_date == date(2024, 1, 31)
    assert config.benchmark.coverage_start_date == date(2014, 1, 1)
    assert config.benchmark.coverage_end_date == date(2026, 1, 31)
    assert len(config.benchmark.required_dates) == 145
    assert config.brinson.start_date == date(2024, 1, 31)
    assert config.brinson.end_date == date(2026, 1, 31)
    brinson_dates = pd.date_range(
        config.brinson.start_date,
        config.brinson.end_date,
        freq="ME",
    )
    assert len(brinson_dates) == 25
    assert PRODUCTION_ACQUISITION_POLICY.max_attempts == 3
    assert PRODUCTION_ACQUISITION_POLICY.item_budget_seconds == 90.0
    assert PRODUCTION_ACQUISITION_POLICY.initial_backoff_seconds == 2.0
    assert PRODUCTION_ACQUISITION_POLICY.max_backoff_seconds == 30.0
    assert PRODUCTION_ACQUISITION_POLICY.backoff_multiplier == 2.0
    assert PRODUCTION_ACQUISITION_POLICY.backoff_jitter_seconds == 1.0
    assert PRODUCTION_ACQUISITION_POLICY.pacing_min_seconds == 1.0
    assert PRODUCTION_ACQUISITION_POLICY.pacing_max_seconds == 3.0

    assert config.accounting.initial_capital == 1_000_000.0
    assert config.accounting.fee_per_trade == 2.0
    assert config.accounting.cash_interest_rate == 0.02
    assert config.accounting.loan_interest_rate == 0.08
    assert config.accounting.transaction_costs == TransactionCostConfig(
        base_bps=1.0,
        liquidity_bps=7.0,
        min_bps=1.0,
        max_bps=15.0,
    )


def test_backtest_and_live_share_canonical_strategy_and_accounting_defaults():
    assert DEFAULT_CONFIG.strategy is None
    assert LIVE_DEFAULT_CONFIG.strategy is None
    assert LIVE_DEFAULT_CONFIG.brinson.yahoo_lookback_months == 12
    assert DEFAULT_CONFIG.accounting == LIVE_DEFAULT_CONFIG.accounting


def test_configuration_is_deeply_immutable():
    with pytest.raises(FrozenInstanceError):
        DEFAULT_CONFIG.market.end_date = date(2025, 1, 31)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TransactionCostConfig(min_bps=16.0, max_bps=15.0),
        lambda: BacktestMarketConfig(
            start_date=date(2024, 1, 31),
            end_date=date(2014, 1, 31),
        ),
        lambda: BacktestEvaluationConfig(
            validation_end_date=date(2022, 1, 31),
            test_start_date=date(2022, 1, 31),
        ),
        lambda: BacktestEvaluationConfig(
            validation_start_date=date(2022, 1, 31),
            validation_end_date=date(2021, 12, 31),
        ),
        lambda: BacktestBrinsonConfig(yahoo_lookback_months=0),
        lambda: LiveBrinsonConfig(yahoo_lookback_months=0),
    ],
)
def test_invalid_configuration_is_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_cross_section_date_validation():
    with pytest.raises(ValueError):
        BacktestConfig(
            market=BacktestMarketConfig(
                start_date=date(2014, 1, 31),
                end_date=date(2020, 1, 31),
            )
        )


def test_live_config_rejects_empty_schedule():
    with pytest.raises(ValueError, match="schedule cannot be empty"):
        LiveConfig(schedule=())


def test_strategy_specific_configuration_requires_matching_strategy_paths():
    strategy = momentum_test_strategy()

    with pytest.raises(ValueError, match="must match"):
        BacktestConfig(strategy=strategy)
    assert BacktestConfig(paths=DEFAULT_CONFIG.paths.for_strategy("momentum")).strategy is None
    with pytest.raises(ValueError, match="must match"):
        BacktestConfig(
            strategy=strategy,
            paths=DEFAULT_CONFIG.paths.for_strategy(
                "reversal"
            ),
        )

    with pytest.raises(ValueError, match="must match"):
        LiveConfig(strategy=strategy)
    assert LiveConfig(paths=LIVE_DEFAULT_CONFIG.paths.for_strategy("momentum")).strategy is None
