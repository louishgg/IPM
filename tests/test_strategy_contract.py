"""Validation and immutability tests for the canonical strategy contract."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.engine import decision_audit_columns, trade_audit_columns
from live.analysis import live_decision_audit_columns, live_trade_audit_columns
from portfolio_core.strategies import (
    ExecutionDecision,
    StrategyDecisionContext,
    available_strategy_ids,
    build_registered_strategy,
)
from portfolio_core.strategies.research_strategy import ResearchStrategy
from portfolio_core.strategies.strategy_contract import RESERVED_AUDIT_COLUMNS

from portfolio_core.strategies.momentum import MomentumStrategy
from _strategy_test_helpers import (
    monthly_history, momentum_test_parameters,
    momentum_test_strategy, research_test_parameters, strategy_context,
)


_AUDIT_BUILDERS = (
    decision_audit_columns, trade_audit_columns,
    live_decision_audit_columns, live_trade_audit_columns,
)
_FIXED_AUDIT_COLUMNS = frozenset(
    column
    for builder in _AUDIT_BUILDERS
    for column in builder(SimpleNamespace(signal_columns=()))
)


def test_every_fixed_engine_audit_column_is_reserved():
    assert _FIXED_AUDIT_COLUMNS <= RESERVED_AUDIT_COLUMNS


@pytest.mark.parametrize("strategy_id", available_strategy_ids())
def test_registered_strategies_have_unique_engine_audit_columns(strategy_id):
    parameters = research_test_parameters(strategy_id).payload()
    strategy = build_registered_strategy(strategy_id, parameters)
    for builder in _AUDIT_BUILDERS:
        columns = builder(strategy)
        assert len(columns) == len(set(columns))


@pytest.mark.parametrize("strategy_id", available_strategy_ids())
def test_registered_strategies_preserve_frozen_parameters(strategy_id):
    parameters = research_test_parameters(strategy_id).payload()
    strategy = build_registered_strategy(strategy_id, parameters)
    assert isinstance(strategy, ResearchStrategy)
    with pytest.raises(FrozenInstanceError):
        strategy._parameters = replace(strategy.parameters, turnover_threshold=0.1)


def test_concrete_strategy_and_parameter_objects_are_immutable():
    strategy = momentum_test_strategy()
    with pytest.raises(FrozenInstanceError):
        strategy._parameters = replace(strategy.parameters, turnover_threshold=0.01)
    with pytest.raises(FrozenInstanceError):
        strategy.parameters.turnover_threshold = 0.01


@pytest.mark.parametrize(
    "mutation",
    [
        lambda decision: replace(
            decision,
            signal_audit=decision.signal_audit.assign(Undeclared=1.0),
        ),
        lambda decision: replace(
            decision,
            raw_target_weights=decision.raw_target_weights.mask(
                decision.raw_target_weights.index
                == decision.raw_target_weights.index[0],
                np.inf,
            ),
        ),
        lambda decision: replace(
            decision,
            raw_target_weights=decision.raw_target_weights.mask(
                decision.raw_target_weights.index
                == decision.original_long_asset_ids[0],
                -0.05,
            ),
        ),
        lambda decision: replace(
            decision,
            raw_target_weights=decision.raw_target_weights.mask(
                decision.raw_target_weights.index
                == decision.original_long_asset_ids[0],
                0.0,
            ),
        ),
        lambda decision: replace(
            decision,
            original_short_asset_ids=(
                decision.original_long_asset_ids[0],
                *decision.original_short_asset_ids[1:],
            ),
        ),
        lambda decision: replace(
            decision,
            raw_target_weights=decision.raw_target_weights.drop(
                decision.original_long_asset_ids[-1]
            ),
            original_long_asset_ids=decision.original_long_asset_ids[:-1],
        ),
    ],
)
def test_strategy_contract_rejects_malformed_decisions(mutation):
    close, volume, assets, sectors = monthly_history()
    valid_strategy = momentum_test_strategy()
    context = strategy_context(close, volume, assets, sectors)
    invalid = mutation(valid_strategy.decide(context))

    class InvalidDecisionStrategy(MomentumStrategy):
        def _decide(self, ignored_context):
            return invalid

    with pytest.raises(ValueError):
        InvalidDecisionStrategy(momentum_test_parameters()).decide(context)


def test_strategy_contract_rejects_duplicate_and_unsorted_candidates():
    close, volume, assets, sectors = monthly_history()
    with pytest.raises(ValueError, match="unique"):
        strategy_context(close, volume, (*assets[:-1], assets[0]), sectors)
    with pytest.raises(ValueError, match="deterministically ordered"):
        strategy_context(close, volume, tuple(reversed(assets)), sectors)


@pytest.mark.parametrize("future_field", ["source", "close"])
def test_strategy_contract_rejects_each_post_cutoff_source(future_field):
    close, volume, assets, sectors = monthly_history()
    cutoff = close.index[-2]
    with pytest.raises(ValueError, match="after the signal cutoff"):
        StrategyDecisionContext(
            close_history=close if future_field == "close" else close.loc[:cutoff],
            volume_history=volume if future_field == "close" else volume.loc[:cutoff],
            candidate_asset_ids=assets,
            sector_code_by_asset_id=sectors,
            previous_target_weights=pd.Series(dtype=float),
            signal_cutoff=cutoff,
            signal_source_max_date=close.index[-1] if future_field == "source" else cutoff,
        )


@pytest.mark.parametrize("column", sorted(_FIXED_AUDIT_COLUMNS))
def test_strategy_contract_rejects_colliding_declared_signal_columns(column):
    close, volume, assets, sectors = monthly_history()
    context = strategy_context(close, volume, assets, sectors)
    decision = momentum_test_strategy().decide(context)

    class CollidingStrategy(MomentumStrategy):
        signal_columns = (column,)

        def _decide(self, ignored_context):
            return decision

    with pytest.raises(ValueError, match="collide"):
        CollidingStrategy(momentum_test_parameters()).decide(context)


def test_strategy_contract_rejects_ineligible_final_holdings():
    close, volume, assets, sectors = monthly_history()
    context = strategy_context(close, volume, assets, sectors)

    class IneligibleFinalStrategy(MomentumStrategy):
        def _finalize_for_execution(self, decision, execution_eligible_asset_ids):
            return ExecutionDecision(
                final_target_weights=decision.raw_target_weights,
                final_long_asset_ids=decision.original_long_asset_ids,
                final_short_asset_ids=decision.original_short_asset_ids,
            )

    strategy = IneligibleFinalStrategy(momentum_test_parameters())
    decision = strategy.decide(context)
    with pytest.raises(ValueError, match="ineligible"):
        strategy.finalize_for_execution(
            decision,
            set(assets) - {decision.original_long_asset_ids[0]},
        )
