"""Shared monthly research construction, fixed exposure, and execution."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_core.sector_evidence import GICS_SECTOR_LABELS

from .portfolio_construction import rank_orders, select_stocks, size_stocks, stock_volatility
from .research_parameters import ResearchParameters, StockSelection
from .strategy_contract import (
    Strategy,
    StrategyDecision,
    ExecutionDecision,
    empty_audit,
)


CONSTRUCTION_SIGNAL_COLUMNS = (
    "Sizing_Volatility",
    "Construction_Sector",
    "Prior_Target",
    "Long_Rank",
    "Short_Rank",
    "Buffered_Long",
    "Buffered_Short",
    "Provisional_Long",
    "Provisional_Short",
    "Neutral_Substitution",
)


@dataclass(frozen=True)
class ResearchStrategy(Strategy):
    """Common research workflow; families own their signal and audit labels."""

    _parameters: ResearchParameters
    strategy_version = "2.0.0"
    score_direction = 1
    summary_columns = (
        "Target_Gross",
        "Construction_Ready",
    )

    def __post_init__(self):
        if self.parameters.signal.family != self.strategy_id or not isinstance(
            self.parameters.selection, StockSelection
        ):
            raise ValueError(
                f"{self.strategy_id} requires its matching signal and stock selection"
            )

    @property
    def parameters(self):
        return self._parameters

    @property
    def selection(self):
        return self.parameters.selection

    @property
    def n_long(self):
        return self.selection.n_long

    @property
    def n_short(self):
        return self.selection.n_short

    @property
    def turnover_threshold(self):
        return self.parameters.turnover_threshold

    def _construct(self, audit, eligible):
        rows = audit.loc[
            audit.index.isin(eligible) & audit.Strategy_Eligible.astype(bool)
        ]
        scores = rows.Strategy_Score
        sectors = rows.Construction_Sector.to_dict()
        longs, shorts, retained, provisional = select_stocks(
            scores,
            rows.Prior_Target,
            sectors,
            self.selection,
            self.parameters.buffer,
            self.parameters.sector_neutral,
        )
        if not longs:
            return pd.Series(dtype=float), longs, shorts, retained, provisional
        weights = size_stocks(
            longs,
            shorts,
            rows.Sizing_Volatility,
            sectors,
            self.parameters.sizing,
            self.parameters.exposure.long_share,
            self.parameters.sector_neutral,
        )
        return weights, longs, shorts, retained, provisional

    def _signal(self, prices):
        """Endpoint return shared by momentum and reversal; other signals override."""
        f, s = self.parameters.signal.formation_months, self.parameters.signal.skip_months
        return (prices.shift(s) / prices.shift(s + f) - 1).iloc[-1]

    def _decide(self, context):
        p = self.parameters
        prices = context.close_history
        signal_values = self._signal(prices).reindex(context.candidate_asset_ids)
        audit = empty_audit(context.candidate_asset_ids, self.signal_columns)
        audit[self.return_column] = signal_values
        scores = self.score_direction * signal_values
        audit["Construction_Sector"] = pd.Series(
            context.sector_code_by_asset_id
        ).reindex(audit.index)
        if not audit.Construction_Sector.isin(GICS_SECTOR_LABELS).all():
            raise ValueError(f"{self.strategy_id} requires exact-date known GICS sectors")
        audit["Prior_Target"] = context.previous_target_weights.reindex(
            audit.index
        ).fillna(0)
        eligible = signal_values.notna() & np.isfinite(signal_values)
        if p.sizing.method == "inverse_volatility":
            profile = p.sizing.volatility
            vol = stock_volatility(prices, profile).iloc[-1].reindex(audit.index)
            valid = np.isfinite(vol) & vol.gt(0)
            audit.loc[eligible & ~valid, "Strategy_Exclusion_Reason"] = (
                "unavailable_sizing_volatility"
            )
            eligible &= valid
            audit["Sizing_Volatility"] = vol
        else:
            audit["Sizing_Volatility"] = np.nan
        audit.loc[eligible, "Strategy_Eligible"] = True
        audit.loc[eligible, "Strategy_Exclusion_Reason"] = ""
        audit.loc[eligible, "Strategy_Score"] = scores[eligible]
        long_order, short_order = rank_orders(scores[eligible])
        audit.loc[long_order, "Strategy_Rank"] = np.arange(1, len(long_order) + 1)
        audit.loc[long_order, "Long_Rank"] = np.arange(1, len(long_order) + 1)
        audit.loc[short_order, "Short_Rank"] = np.arange(1, len(short_order) + 1)
        weights, longs, shorts, retained, provisional = self._construct(
            audit, set(long_order)
        )
        for name, side in (("Long", 1), ("Short", -1)):
            audit[f"Buffered_{name}"] = audit.index.isin(retained.get(side, ()))
            audit[f"Provisional_{name}"] = audit.index.isin(provisional.get(side, ()))
        selected = set(longs + shorts)
        audit["Neutral_Substitution"] = [
            (
                "added_for_paired_sectors"
                if a in selected
                and a not in provisional.get(1, set()) | provisional.get(-1, set())
                else (
                    "dropped_for_paired_sectors"
                    if a not in selected
                    and a in provisional.get(1, set()) | provisional.get(-1, set())
                    else ""
                )
            )
            for a in audit.index
        ]
        audit.loc[
            audit.index.isin(set(longs) & provisional.get(-1, set())),
            "Neutral_Substitution",
        ] = "short_to_long_for_paired_sectors"
        audit.loc[
            audit.index.isin(set(shorts) & provisional.get(1, set())),
            "Neutral_Substitution",
        ] = "long_to_short_for_paired_sectors"
        return self._complete_decision(audit, weights, longs, shorts, long_order)

    def _complete_decision(self, audit, weights, longs, shorts, long_order, basket=None):
        """Apply the configured fixed gross to a feasible unit-gross book."""
        gross = self.parameters.exposure.gross
        complete = bool(longs)
        return StrategyDecision(
            weights * gross if complete else pd.Series(dtype=float),
            audit,
            tuple(long_order),
            longs if complete else (),
            shorts if complete else (),
            complete,
            dict(
                Target_Gross=gross,
                Construction_Ready=float(complete),
            ),
            sector_basket=basket if complete else None,
        )

    def _finalize_for_execution(self, decision, execution_eligible_asset_ids):
        """Rebuild selection and sizing on eligible rows of the saved signal audit.

        Reuse frozen scores, volatility, prior holdings and gross exposure.
        Incomplete decisions stay empty; failed reconstruction of a complete
        decision raises InfeasibleRebalanceError.
        """
        if not decision.is_complete:
            return ExecutionDecision(pd.Series(dtype=float), (), (), is_complete=False)
        if set(decision.ranked_candidate_asset_ids).issubset(
            execution_eligible_asset_ids
        ):
            return ExecutionDecision(
                decision.raw_target_weights.copy(),
                decision.original_long_asset_ids,
                decision.original_short_asset_ids,
            )
        weights, longs, shorts, _, _ = self._construct(
            decision.signal_audit, execution_eligible_asset_ids
        )
        if not longs:
            from portfolio_core.accounting_ledger import InfeasibleRebalanceError

            raise InfeasibleRebalanceError(
                "Execution eligibility cannot restore the configured momentum book"
            )
        weights *= decision.summary_metrics["Target_Gross"]
        reasons = {
            a: "execution_eligibility_refill"
            for a in longs + shorts
            if a
            not in decision.original_long_asset_ids + decision.original_short_asset_ids
        }
        reasons.update(
            {
                a: "departed_before_execution"
                for a in decision.original_long_asset_ids
                + decision.original_short_asset_ids
                if a not in execution_eligible_asset_ids
            }
        )
        return ExecutionDecision(weights, longs, shorts, reasons)
