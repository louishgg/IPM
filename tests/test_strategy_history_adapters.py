"""Cross-domain causal history normalization and decision parity tests."""

import numpy as np
import pandas as pd
import pytest

from live.monthly_history import aggregate_monthly_market, monthly_matrices
from portfolio_core.strategies import (
    StrategyDecisionContext,
)

from _strategy_test_helpers import momentum_test_strategy, monthly_history, strategy_context


def _prepare_and_read(daily, cutoff):
    monthly = aggregate_monthly_market(daily.assign(Price_Source="yahoo"), through=daily.Date.max())
    close, volume, calendar = monthly_matrices(monthly, cutoff=cutoff)
    return close, volume, calendar[close.index[-1]]


def test_live_history_adapter_uses_last_monthly_close_and_summed_volume():
    market = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2026-01-15", "2026-01-30", "2026-02-02", "2026-02-27"]
            ),
            "Asset_ID": ["AAA", "AAA", "AAA", "AAA"],
            "Close": [10.0, 12.0, 99.0, 14.0],
            "Volume": [100.0, 250.0, 1_000.0, 400.0],
        }
    )

    close, volume, source_max = _prepare_and_read(
        market,
        pd.Timestamp("2026-01-30"),
    )

    assert close.loc[pd.Timestamp("2026-01-31"), "AAA"] == 12.0
    assert volume.loc[pd.Timestamp("2026-01-31"), "AAA"] == 350.0
    assert source_max == pd.Timestamp("2026-01-30")


@pytest.mark.parametrize("values,expected", [
    ([np.nan, np.nan], np.nan),
    ([0.0, 0.0], 0.0),
    ([np.nan, 250.0], 250.0),
    ([100.0, np.nan], 100.0),
], ids=["missing", "zero", "missing-first", "missing-last"])
def test_monthly_volume_distinguishes_missing_zero_and_partial_observations(values, expected):
    daily = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-15", "2026-01-30"]),
        "Asset_ID": ["AAA", "AAA"], "Close": [10.0, 12.0], "Volume": values,
    })
    close, volume, source_max = _prepare_and_read(daily, pd.Timestamp("2026-01-30"))
    actual = volume.loc[pd.Timestamp("2026-01-31"), "AAA"]
    if pd.isna(expected):
        assert pd.isna(actual)
    else:
        assert actual == expected
    assert close.loc[pd.Timestamp("2026-01-31"), "AAA"] == 12.0
    assert source_max == pd.Timestamp("2026-01-30")


def test_equivalent_histories_produce_identical_decisions_and_refills():
    strategy = momentum_test_strategy()
    close, volume, assets, sectors = monthly_history()
    daily = (
        close.rename_axis("Date")
        .stack(future_stack=True)
        .rename("Close")
        .to_frame()
        .join(
            volume.rename_axis("Date")
            .stack(future_stack=True)
            .rename("Volume")
        )
        .reset_index(names=["Date", "Asset_ID"])
    )
    live_close, live_volume, source_max = _prepare_and_read(
        daily,
        close.index[-1],
    )
    backtest_decision = strategy.decide(
        strategy_context(close, volume, assets, sectors)
    )
    live_decision = strategy.decide(
        StrategyDecisionContext(
            close_history=live_close,
            volume_history=live_volume,
            candidate_asset_ids=assets,
            sector_code_by_asset_id=sectors,
            previous_target_weights=pd.Series(dtype=float),
            signal_cutoff=close.index[-1],
            signal_source_max_date=source_max,
        )
    )

    assert (
        live_decision.ranked_candidate_asset_ids
        == backtest_decision.ranked_candidate_asset_ids
    )
    assert (
        live_decision.original_long_asset_ids
        == backtest_decision.original_long_asset_ids
    )
    assert (
        live_decision.original_short_asset_ids
        == backtest_decision.original_short_asset_ids
    )
    pd.testing.assert_series_equal(
        live_decision.raw_target_weights,
        backtest_decision.raw_target_weights,
    )
    pd.testing.assert_frame_equal(
        live_decision.signal_audit.reset_index(drop=True),
        backtest_decision.signal_audit.reset_index(drop=True),
    )

    eligible = set(assets) - {
        backtest_decision.original_long_asset_ids[0],
        backtest_decision.original_short_asset_ids[0],
    }
    backtest_final = strategy.finalize_for_execution(
        backtest_decision, eligible
    )
    live_final = strategy.finalize_for_execution(live_decision, eligible)
    assert live_final.final_long_asset_ids == backtest_final.final_long_asset_ids
    assert live_final.final_short_asset_ids == backtest_final.final_short_asset_ids
    assert dict(live_final.eligibility_reasons) == dict(
        backtest_final.eligibility_reasons
    )
    pd.testing.assert_series_equal(
        live_final.final_target_weights,
        backtest_final.final_target_weights,
    )


@pytest.mark.parametrize("strategy", [momentum_test_strategy()], ids=["momentum"])
def test_post_cutoff_daily_mutations_cannot_change_either_strategy(strategy):
    close, volume, assets, sectors = monthly_history(asset_count=60)
    cutoff = close.index[-1]
    daily = (
        close.rename_axis("Date")
        .stack(future_stack=True)
        .rename("Close")
        .to_frame()
        .join(
            volume.rename_axis("Date")
            .stack(future_stack=True)
            .rename("Volume")
        )
        .reset_index(names=["Date", "Asset_ID"])
    )
    future = daily.loc[daily["Date"].eq(cutoff)].copy()
    future["Date"] = cutoff + pd.Timedelta(days=7)
    future["Close"] = future["Close"] * np.linspace(0.01, 100.0, len(future))
    future["Volume"] = future["Volume"] * 1_000.0
    mutated = pd.concat([daily, future], ignore_index=True)

    baseline_history = _prepare_and_read(daily, cutoff)
    mutated_history = _prepare_and_read(mutated, cutoff)
    baseline = strategy.decide(
        StrategyDecisionContext(
            close_history=baseline_history[0],
            volume_history=baseline_history[1],
            candidate_asset_ids=assets,
            sector_code_by_asset_id=sectors,
            previous_target_weights=pd.Series(dtype=float),
            signal_cutoff=cutoff,
            signal_source_max_date=baseline_history[2],
        )
    )
    changed = strategy.decide(
        StrategyDecisionContext(
            close_history=mutated_history[0],
            volume_history=mutated_history[1],
            candidate_asset_ids=assets,
            sector_code_by_asset_id=sectors,
            previous_target_weights=pd.Series(dtype=float),
            signal_cutoff=cutoff,
            signal_source_max_date=mutated_history[2],
        )
    )
    assert changed.original_long_asset_ids == baseline.original_long_asset_ids
    assert changed.original_short_asset_ids == baseline.original_short_asset_ids
    pd.testing.assert_series_equal(
        changed.raw_target_weights,
        baseline.raw_target_weights,
    )
    pd.testing.assert_frame_equal(changed.signal_audit, baseline.signal_audit)
