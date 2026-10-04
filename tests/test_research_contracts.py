"""Boundary tests for dormant research schemas and shared run-local contracts."""

from dataclasses import replace

import pandas as pd
import pytest

from portfolio_core.strategies import ExecutionDecision
from portfolio_core.strategies.momentum import MomentumStrategy
from portfolio_core.strategies.research_parameters import (
    BufferParameters,
    ExposureParameters,
    ResearchParameters,
    SignalParameters,
    SizingParameters,
    StockSelection,
    VolatilityProfile,
    WholeSectorSelection,
)
from portfolio_core.strategies.research_state import (
    SectorBasket,
    SelectedSectors,
)
from _strategy_test_helpers import (
    monthly_history, research_contract_parameters, strategy_context,
)


def test_complete_packet_roundtrip_and_no_implicit_defaults():
    value = research_contract_parameters()
    assert ResearchParameters.from_payload(value.payload()) == value
    for key in value.payload():
        partial = value.payload()
        partial.pop(key)
        with pytest.raises(ValueError, match="complete|migration"):
            ResearchParameters.from_payload(partial)
    for key in ("H", "fixed_gross_multiplier", "score_sign_gate"):
        with pytest.raises(ValueError, match="complete|migration"):
            ResearchParameters.from_payload({**value.payload(), key: 1})
    bad = value.payload()
    bad["selection"]["n_long_sectors"] = 1
    with pytest.raises(ValueError, match="complete|migration"):
        ResearchParameters.from_payload(bad)


def test_inactive_fields_and_linked_profiles_are_independent():
    base = research_contract_parameters()
    requested = replace(
        base,
        sizing=SizingParameters("equal", VolatilityProfile(60, 24)),
        buffer=BufferParameters(False, 1.5),
        exposure=ExposureParameters(1.5, 0.5),
    )
    assert requested.canonical_json() == base.canonical_json()
    assert requested.payload() != base.payload()
    low = replace(
        base,
        signal=SignalParameters(
            "low_volatility", None, None, VolatilityProfile(36, 36), None
        ),
        sizing=SizingParameters("inverse_volatility", VolatilityProfile(60, 24)),
    )
    assert low.effective().signal.selection_volatility == VolatilityProfile(36, 36)
    assert low.effective().sizing.volatility == VolatilityProfile(60, 24)
    with pytest.raises(ValueError):
        VolatilityProfile(24, 36)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: replace(
            p, sector_neutral=True, exposure=replace(p.exposure, long_share=0.6)
        ),
        lambda p: replace(p, selection=WholeSectorSelection(1, 1)),
        lambda p: replace(p, signal=SignalParameters("reversal", 0, 0, None, None)),
        lambda p: replace(p, selection=StockSelection(True, 20)),
        lambda p: replace(p, turnover_threshold=float("nan")),
    ],
)
def test_static_rejections(change):
    with pytest.raises(ValueError):
        change(research_contract_parameters())


def test_sector_packet_and_rounding_equivalence_preserve_enabled_status():
    sector = replace(
        research_contract_parameters(),
        signal=SignalParameters("sector_momentum", 6, 1, None, None),
        selection=WholeSectorSelection(1, 1),
    )
    a = replace(sector, buffer=BufferParameters(True, 1.25))
    b = replace(sector, buffer=BufferParameters(True, 1.5))
    assert a.canonical_json() == b.canonical_json()
    assert a.canonical_json() != sector.canonical_json()
    assert BufferParameters(True, 1.25).additional_pool(10) == 2
    assert BufferParameters(True, 1.5).additional_pool(1) == 0
    with pytest.raises(ValueError, match="neutrality"):
        replace(sector, sector_neutral=True)
    assert ResearchParameters.from_payload(sector.payload()) == sector


def test_whole_sector_contract_accepts_variable_counts_and_rejects_truncation():
    close, volume, assets, _ = monthly_history(asset_count=30)
    sectors = {a: "10" if i < 12 else "20" for i, a in enumerate(assets)}
    context = strategy_context(close, volume, assets, sectors)
    stock = MomentumStrategy(research_contract_parameters())
    raw = stock.decide(context)
    longs, shorts = assets[:12], assets[12:]
    basket = SectorBasket(
        {"10": longs, "20": shorts}, SelectedSectors(("10",), ("20",))
    )
    decision = replace(
        raw,
        raw_target_weights=pd.Series(
            [0.5 / 12] * 12 + [-0.5 / 18] * 18, index=pd.Index(assets, name="Asset_ID")
        ),
        original_long_asset_ids=longs,
        original_short_asset_ids=shorts,
        sector_basket=basket,
    )

    class BasketStrategy(MomentumStrategy):
        @property
        def selection(self):
            return WholeSectorSelection(1, 1)

        def _decide(self, context):
            return decision

        def _finalize_for_execution(self, decision, eligible):
            ls = tuple(a for a in longs if a in eligible)
            ss = tuple(a for a in shorts if a in eligible)
            return ExecutionDecision(
                decision.raw_target_weights.loc[[*ls, *ss]],
                ls,
                ss,
                sector_basket=basket,
            )

    strategy = BasketStrategy(research_contract_parameters())
    assert strategy.decide(context).is_complete
    assert (
        len(
            strategy.finalize_for_execution(
                decision, set(assets) - {longs[0]}
            ).final_long_asset_ids
        )
        == 11
    )
    with pytest.raises(ValueError, match="ten stocks"):
        strategy.finalize_for_execution(decision, set(assets) - set(longs[:3]))
    decision = replace(
        decision,
        raw_target_weights=decision.raw_target_weights.iloc[1:],
        original_long_asset_ids=longs[1:],
    )
    with pytest.raises(ValueError, match="every eligible"):
        strategy.decide(context)
    with pytest.raises(ValueError, match="Stock selections"):
        stock._validate_basket(basket, longs, shorts, True)




def test_numeric_json_representations_have_one_effective_identity():
    base = research_contract_parameters()
    integer = replace(
        base, turnover_threshold=0, exposure=replace(base.exposure, gross=1)
    )
    floating = replace(
        base, turnover_threshold=0.0, exposure=replace(base.exposure, gross=1.0)
    )
    assert integer.canonical_json() == floating.canonical_json()
