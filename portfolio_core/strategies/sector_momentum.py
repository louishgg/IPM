"""Historical equal-weight sector momentum with whole, current sector books."""
from dataclasses import replace

import numpy as np
import pandas as pd

from portfolio_core.accounting_ledger import InfeasibleRebalanceError
from portfolio_core.sector_evidence import GICS_SECTOR_LABELS
from .research_strategy import CONSTRUCTION_SIGNAL_COLUMNS, ResearchStrategy
from .portfolio_construction import rank_orders, select_stocks, stock_volatility
from .research_parameters import StockSelection, WholeSectorSelection
from .research_state import SectorBasket, SelectedSectors
from .strategy_contract import ExecutionDecision, empty_audit


def sector_scores(returns, cutoff, formation, skip):
    """Compound exactly F consecutive returns ending S months before cutoff."""
    end = pd.Timestamp(cutoff) - pd.offsets.MonthEnd(skip)
    dates = pd.date_range(end=end, periods=formation, freq="ME")
    window = returns.reindex(dates)
    return (1 + window).prod(min_count=formation).sub(1).where(
        np.isfinite(window).all())


class SectorMomentumStrategy(ResearchStrategy):
    strategy_id = "sector_momentum"
    signal_columns = ("Momentum", *CONSTRUCTION_SIGNAL_COLUMNS, "Sector_Long_Rank",
                      "Sector_Short_Rank", "Prior_Sector_Side")

    summary_columns = (*ResearchStrategy.summary_columns, "Basket_Long_Count", "Basket_Short_Count")

    def __post_init__(self):
        if self.parameters.signal.family != self.strategy_id or not isinstance(
                self.selection, WholeSectorSelection):
            raise ValueError("Sector momentum requires whole-sector selection")

    @property
    def n_long(self):
        raise TypeError("Whole-sector construction has no fixed stock count")

    @property
    def n_short(self):
        raise TypeError("Whole-sector construction has no fixed stock count")

    def _construct_baskets(self, audit, eligible):
        """Build whole eligible-sector baskets with equal sector budgets per side.

        Size within sectors using equal or inverse-volatility weights from the
        saved audit. Return empty weights and no basket unless each side has
        at least ten stocks.
        """
        rows = audit.loc[audit.index.isin(eligible) & audit.Strategy_Eligible.astype(bool)]
        pool = {str(code): tuple(sorted(group.index))
                for code, group in rows.groupby('Construction_Sector')}
        scores = rows.groupby('Construction_Sector').Strategy_Score.first()
        previous = rows.groupby('Construction_Sector').Prior_Sector_Side.first()
        selection = StockSelection(self.selection.n_long_sectors, self.selection.n_short_sectors)
        longs, shorts, retained, provisional = select_stocks(
            scores, previous, {}, selection, self.parameters.buffer, False)
        selected = SelectedSectors(longs, shorts)
        weights = {}
        for codes, budget in ((longs, self.parameters.exposure.long_share),
                              (shorts, -(1-self.parameters.exposure.long_share))):
            for code in codes:
                ids = list(pool[code])
                base = (1/rows.loc[ids, 'Sizing_Volatility'] if
                        self.parameters.sizing.method == 'inverse_volatility'
                        else pd.Series(1., index=ids))
                weights.update((base/base.sum()*budget/len(codes)).to_dict())
        weights = pd.Series(weights, dtype=float)
        if (weights > 0).sum() < 10 or (weights < 0).sum() < 10:
            return pd.Series(dtype=float), (), (), None, retained, provisional
        return (weights, tuple(weights.index[weights > 0]), tuple(weights.index[weights < 0]),
                SectorBasket(pool, selected), retained, provisional)

    def _decide(self, context):
        p = self.parameters
        audit = empty_audit(context.candidate_asset_ids, self.signal_columns)
        codes = pd.Series(context.sector_code_by_asset_id).reindex(audit.index)
        if not codes.isin(GICS_SECTOR_LABELS).all():
            raise ValueError("Sector momentum requires exact-date known GICS sectors")
        scores = sector_scores(context.sector_returns, context.signal_cutoff,
                               p.signal.formation_months, p.signal.skip_months)
        audit['Construction_Sector'] = codes
        audit['Momentum'] = codes.map(scores)
        audit['Prior_Target'] = context.previous_target_weights.reindex(audit.index).fillna(0)
        prior = {c: side for side, group in ((1, context.previous_selected_sectors.longs),
                                           (-1, context.previous_selected_sectors.shorts)) for c in group}
        audit['Prior_Sector_Side'] = codes.map(prior).fillna(0)
        valid = np.isfinite(audit.Momentum)
        if p.sizing.method == 'inverse_volatility':
            vol = stock_volatility(context.close_history, p.sizing.volatility).iloc[-1].reindex(audit.index)
            audit['Sizing_Volatility'] = vol
            usable = np.isfinite(vol) & vol.gt(0)
            audit.loc[valid & ~usable, 'Strategy_Exclusion_Reason'] = 'unavailable_sizing_volatility'
            valid &= usable
        audit.loc[valid, 'Strategy_Eligible'] = True
        audit.loc[valid, 'Strategy_Exclusion_Reason'] = ''
        audit.loc[valid, 'Strategy_Score'] = audit.loc[valid, 'Momentum']
        sector_long, sector_short = rank_orders(audit.loc[valid].groupby('Construction_Sector').Strategy_Score.first())
        audit['Sector_Long_Rank'] = codes.map({c:i+1 for i,c in enumerate(sector_long)})
        audit['Sector_Short_Rank'] = codes.map({c:i+1 for i,c in enumerate(sector_short)})
        # Common audit is asset-level; sector rank and code ties remain explicit.
        order = sorted(audit.index[valid], key=lambda a: (-audit.at[a,'Strategy_Score'], codes[a], a))
        audit.loc[order, 'Strategy_Rank'] = np.arange(1, len(order)+1)
        audit['Long_Rank'], audit['Short_Rank'] = audit.Sector_Long_Rank, audit.Sector_Short_Rank
        weights, longs, shorts, basket, retained, provisional = self._construct_baskets(audit, set(order))
        for name, side in (('Long',1),('Short',-1)):
            audit[f'Buffered_{name}'] = codes.isin(retained.get(side, ()))
            audit[f'Provisional_{name}'] = codes.isin(provisional.get(side, ()))
        audit['Neutral_Substitution'] = ''
        decision = self._complete_decision(audit, weights, longs, shorts, order, basket)
        return replace(decision, summary_metrics={**decision.summary_metrics,
            'Basket_Long_Count': float(audit.Provisional_Long.sum()),
            'Basket_Short_Count': float(audit.Provisional_Short.sum())})

    def _finalize_for_execution(self, decision, execution_eligible_asset_ids):
        """Rebuild whole baskets from the saved audit and execution eligibility.

        Preserve the causal constituent pool and frozen gross exposure.
        Incomplete decisions stay empty; failure to restore complete baskets
        with at least ten stocks per side raises InfeasibleRebalanceError.
        """
        if not decision.is_complete:
            return ExecutionDecision(pd.Series(dtype=float), (), (), is_complete=False)
        weights, longs, shorts, basket, _, _ = self._construct_baskets(decision.signal_audit, execution_eligible_asset_ids)
        if basket is None:
            raise InfeasibleRebalanceError("Execution eligibility cannot restore whole-sector baskets and 10/10 floor")
        # Retain the saved causal pool; contract intersects it with execution eligibility.
        basket = SectorBasket(decision.sector_basket.eligible_members, basket.selected)
        reasons = {a: 'departed_before_execution' for a in decision.raw_target_weights.index
                   if a not in execution_eligible_asset_ids}
        reasons.update({a:'execution_eligibility_refill' for a in weights.index
                        if a not in decision.raw_target_weights.index})
        return ExecutionDecision(weights*decision.summary_metrics['Target_Gross'], longs, shorts,
                                 reasons, sector_basket=basket)

    def validate_prior(self, weights, context, selected):
        """A carry is legal only if it is still the complete current sized basket."""
        pool = set(context.candidate_asset_ids)
        if self.parameters.sizing.method == 'inverse_volatility':
            vol = stock_volatility(context.close_history, self.parameters.sizing.volatility).iloc[-1]
            pool &= set(vol.index[np.isfinite(vol) & vol.gt(0)])
        codes = context.sector_code_by_asset_id
        for side, sectors, count in ((1, selected.longs, self.selection.n_long_sectors),
                                     (-1, selected.shorts, self.selection.n_short_sectors)):
            expected = {a for a in pool if codes[a] in sectors}
            actual = set(weights.index[weights*side > 0])
            if len(sectors) != count or len(expected) < 10 or actual != expected or any(
                    not any(codes[a] == c for a in expected) for c in sectors):
                raise InfeasibleRebalanceError(f"{context.signal_cutoff}: incomplete decision cannot preserve whole-sector baskets")
        return weights
