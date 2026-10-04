"""Momentum mechanics, fixed exposure, exact neutral selection and checkpoints."""

from dataclasses import replace
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from portfolio_core.strategies.momentum import MomentumStrategy
from portfolio_core.strategies.portfolio_construction import (
    select_stocks,
    rank_orders,
    size_stocks,
)
from portfolio_core.strategies.research_parameters import (
    SignalParameters,
    StockSelection,
    SizingParameters,
    BufferParameters,
    VolatilityProfile,
)
from backtest.engine import run_backtest
import _research_test_helpers as shared
from _research_test_helpers import packet, tiny_grid
from backtest.research_grid import expand_grid, candidate_id
from _strategy_test_helpers import monthly_history, strategy_context
from test_engine_offline import make_synthetic_backtest_data


@pytest.mark.parametrize("f", [3, 6, 9, 11, 12])
def test_momentum_uses_f_plus_skip_endpoints_and_keeps_intermediate_gaps(f):
    close, volume, assets, sectors = monthly_history(asset_count=60)
    # Add one earlier price for the F=12 endpoint; an intermediate gap is not
    # an additional momentum eligibility rule.
    earlier = close.index[0] - pd.offsets.MonthEnd(1)
    close.loc[earlier] = 99
    volume.loc[earlier] = 1e6
    close, volume = close.sort_index(), volume.sort_index()
    close.iloc[-3, 0] = np.nan
    context = strategy_context(close, volume, assets, sectors)
    new = MomentumStrategy(
        packet(signal=SignalParameters("momentum", f, 1, None, None))
    ).decide(context)
    expected = (close.shift(1) / close.shift(f + 1) - 1).iloc[-1].reindex(assets)
    pd.testing.assert_series_equal(
        new.signal_audit.Momentum, expected, check_exact=True,
        check_names=False,
    )
    assert len(new.original_long_asset_ids) == 10
    assert len(new.original_short_asset_ids) == 10


def test_ties_buffers_retain_both_sides_before_refill():
    scores = pd.Series(1.0, index=list("abcdef"))
    previous = pd.Series({"b": 0.5, "a": -0.5})
    l, s, _, _ = select_stocks(
        scores,
        previous,
        dict.fromkeys(scores.index, "10"),
        StockSelection(1, 1),
        BufferParameters(True, 1.5),
        False,
    )
    assert l == ("b",) and s == ("a",)
    l, s, _, _ = select_stocks(
        scores,
        previous,
        dict.fromkeys(scores.index, "10"),
        StockSelection(2, 2),
        BufferParameters(False, None),
        False,
    )
    assert l == ("a", "b") and s == ("c", "d")


@pytest.mark.parametrize("seed", range(8))
def test_global_neutral_repair_matches_exhaustive_lexicographic_solution(seed):
    rng = np.random.default_rng(seed)
    assets = list("abcdefg")
    scores = pd.Series(rng.integers(0, 5, len(assets)), index=assets)
    sectors = dict(zip(assets, rng.choice(["10", "20", "30"], len(assets))))
    previous = pd.Series({"a": 0.5, "g": -0.5})
    selection, buffer = StockSelection(2, 2), BufferParameters(True, 1.5)
    l, s, retained, provisional = select_stocks(
        scores, previous, sectors, selection, buffer, True
    )
    lo, so = rank_orders(scores)
    ranks = {
        1: {a: i + 1 for i, a in enumerate(lo)},
        -1: {a: i + 1 for i, a in enumerate(so)},
    }
    pairs = [(side, a) for side in (1, -1) for a in assets]
    pairs.sort(
        key=lambda sa: (
            0 if sa[1] in retained[sa[0]] else 1 if sa[1] in provisional[sa[0]] else 2,
            ranks[sa[0]][sa[1]],
            0 if sa[0] == 1 else 1,
            sa[1],
        )
    )
    feasible = []
    for longs in combinations(assets, 2):
        for shorts in combinations([a for a in assets if a not in longs], 2):
            if {sectors[a] for a in longs} != {sectors[a] for a in shorts}:
                continue
            chosen = {1: set(longs), -1: set(shorts)}
            objective = (
                -sum(len(chosen[side] & retained[side]) for side in (1, -1)),
                -sum(len(chosen[side] & provisional[side]) for side in (1, -1)),
                sum(ranks[side][a] for side in (1, -1) for a in chosen[side]),
                tuple(-int(a in chosen[side]) for side, a in pairs),
            )
            feasible.append((objective, chosen))
    if feasible:
        expected = min(feasible, key=lambda item: item[0])[1]
        assert set(l) == expected[1] and set(s) == expected[-1]
    else:
        assert not l and not s


def test_neutral_budgets_preserve_preliminary_sector_gross_and_relative_weights():
    longs, shorts = ("a", "b", "c"), ("d", "e")
    sectors = dict(a="10", b="10", c="20", d="10", e="20")
    vol = pd.Series(dict(a=0.1, b=0.2, c=0.3, d=0.4, e=0.5))
    sizing = SizingParameters("inverse_volatility", VolatilityProfile(60, 24))
    base = size_stocks(longs, shorts, vol, sectors, sizing, 0.5, False)
    neutral = size_stocks(longs, shorts, vol, sectors, sizing, 0.5, True)
    for code in ("10", "20"):
        ids = [a for a in sectors if sectors[a] == code]
        assert neutral.loc[ids].sum() == pytest.approx(0, abs=1e-15)
        assert neutral.loc[ids].abs().sum() == pytest.approx(base.loc[ids].abs().sum())
    assert neutral.a / neutral.b == pytest.approx(2)


def test_sizing_profiles_have_linked_minima_no_fill_and_no_zero_fallback():
    shared.assert_sizing_contract(MomentumStrategy, packet())


def test_neutral_execution_residual_and_threshold_override_with_post_cost_caps():
    shared.assert_neutral_execution_contract(MomentumStrategy, packet())


def test_incomplete_after_activation_values_held_book_instead_of_freezing_nav():
    data, dates = make_synthetic_backtest_data()

    class MissingSignal(MomentumStrategy):
        def _decide(self, context):
            decision = super()._decide(context)
            if context.signal_cutoff == dates[14]:
                return replace(
                    decision,
                    raw_target_weights=pd.Series(dtype=float),
                    original_long_asset_ids=(),
                    original_short_asset_ids=(),
                    ranked_candidate_asset_ids=(),
                    signal_audit=decision.signal_audit.assign(
                        Strategy_Eligible=False, Strategy_Rank=np.nan
                    ),
                    is_complete=False,
                )
            return decision

    strategy = MissingSignal(packet())
    nav, diag = run_backtest(data, dates[12:17], strategy, return_diagnostics=True)
    assert diag.loc[dates[15], "Used_Hold_Logic"]
    assert nav.loc[dates[15]] != nav.loc[dates[14]]
    assert diag.loc[dates[15], "Interest"] != 0


def test_conditional_grid_rejections_duplicates_and_stable_ids():
    grid = tiny_grid()
    grid["n_long"] = [10, 10]
    grid["sector_neutral"] = [False, True]
    grid["long_share"] = [0.5, 0.6]
    rows, rejected, duplicates = expand_grid(grid)
    assert (len(rows), len(rejected), len(duplicates)) == (3, 2, 3)
    identity = {"code": "a", "data": "b", "assumptions": "c"}
    assert candidate_id(rows[0], identity) == candidate_id(
        {**rows[0], "Candidate_Number": 999}, identity
    )
    assert candidate_id(rows[0], identity) != candidate_id(
        rows[0], {**identity, "data": "new"}
    )


def test_checkpoint_resume_verifies_content_and_identity(tmp_path):
    from backtest.research_search import _load_checkpoint, _write_json
    from portfolio_core.artifacts import file_sha256
    from portfolio_core.io import atomic_write_dataframe

    directory = tmp_path
    artifact = directory / "nav.csv.gz"
    atomic_write_dataframe(pd.DataFrame({"NAV": [100]}), artifact)
    row = dict(
        Candidate_ID="a", Candidate_Number=1, Requested_Number=1, Parameters_JSON="{}"
    )
    identity = {"code": "x"}
    _write_json(
        dict(
            identity=identity,
            row=row,
            folds=[],
            artifacts={"nav.csv.gz": file_sha256(artifact)},
        ),
        directory / "checkpoint.json",
    )
    assert _load_checkpoint(directory, row, identity)["row"] == row
    with pytest.raises(ValueError, match="identity"):
        _load_checkpoint(directory, row, {"code": "changed"})
    atomic_write_dataframe(pd.DataFrame({"NAV": [999]}), artifact)
    with pytest.raises(ValueError, match="modified"):
        _load_checkpoint(directory, row, identity)


def test_search_serial_parallel_resume_and_complete_validation(tmp_path):
    shared.assert_search_serial_parallel_resume(tmp_path, tiny_grid())


def test_unready_branch_remains_declared_and_unranked(tmp_path):
    shared.assert_unready_branch(tmp_path, tiny_grid())


def test_unknown_sector_is_a_data_contract_error():
    close, volume, assets, sectors = monthly_history()
    sectors[assets[0]] = "Unknown"
    with pytest.raises(ValueError, match="known GICS"):
        MomentumStrategy(packet()).decide(
            strategy_context(close, volume, assets, sectors)
        )


def test_grid_deduplicates_integer_and_float_gross_values():
    grid = tiny_grid()
    grid["gross"] = [1, 1.0]
    rows, rejected, duplicates = expand_grid(grid)
    assert len(rows) == 1 and len(duplicates) == 1 and not rejected
