"""Strategy-neutral helpers shared by backtest and live audit builders."""

from __future__ import annotations

import numpy as np

from .accounting_ledger import TradeExecution
from .strategies import ExecutionDecision, Strategy, StrategyDecision


TRADE_EXECUTION_AUDIT_COLUMNS = (
    "Applied_Rule",
    "Current_Shares",
    "Requested_Target_Shares",
    "Applied_Target_Shares",
    "Trade_Shares",
    "Execution_Price",
    "Effective_Execution_Price",
    "Trade_Notional",
    "Order_Count",
    "Fixed_Fee",
    "Spread_Rate",
    "Spread_Cost",
    "Cash_Effect",
    "Restricted_Proceeds_Change",
)
STRATEGY_SIGNAL_AUDIT_COLUMNS = (
    "Strategy_Score",
    "Strategy_Rank",
    "Strategy_Eligible",
    "Strategy_Exclusion_Reason",
)
STRATEGY_SELECTION_AUDIT_COLUMNS = (
    "Signal_Selected_Side",
    "Signal_Raw_Target_Weight",
    "Execution_Eligible",
    "Final_Selected_Side",
    "Final_Target_Weight",
    "Execution_Adjustment",
)


def trade_execution_audit_fields(
    execution: TradeExecution,
) -> dict[str, object]:
    """Serialize the canonical per-security execution fields once."""

    return {
        "Applied_Rule": execution.applied_rule,
        "Current_Shares": execution.current_shares,
        "Requested_Target_Shares": execution.requested_target_shares,
        "Applied_Target_Shares": execution.applied_target_shares,
        "Trade_Shares": execution.trade_shares,
        "Execution_Price": execution.reference_price,
        "Effective_Execution_Price": execution.effective_execution_price,
        "Trade_Notional": execution.reference_notional,
        "Order_Count": execution.order_count,
        "Fixed_Fee": execution.fixed_fee,
        "Spread_Rate": execution.spread_rate,
        "Spread_Cost": execution.spread_cost,
        "Cash_Effect": execution.cash_effect,
        "Restricted_Proceeds_Change": execution.restricted_proceeds_change,
    }


def _selected_side(
    asset_id: str,
    long_asset_ids: tuple[str, ...],
    short_asset_ids: tuple[str, ...],
) -> str:
    """Return the selected portfolio side for one asset, if any."""

    if asset_id in long_asset_ids:
        return "Long"
    if asset_id in short_asset_ids:
        return "Short"
    return ""


def strategy_selection_fields(
    decision: StrategyDecision,
    execution_decision: ExecutionDecision,
    asset_id: str,
    *,
    execution_eligible: bool,
    execution_adjustment: str,
) -> dict[str, object]:
    """Serialize strategy selection before and after execution eligibility."""

    return {
        "Signal_Selected_Side": _selected_side(
            asset_id,
            decision.original_long_asset_ids,
            decision.original_short_asset_ids,
        ),
        "Signal_Raw_Target_Weight": float(
            decision.raw_target_weights.get(asset_id, 0.0)
        ),
        "Execution_Eligible": bool(execution_eligible),
        "Final_Selected_Side": _selected_side(
            asset_id,
            execution_decision.final_long_asset_ids,
            execution_decision.final_short_asset_ids,
        ),
        "Final_Target_Weight": float(
            execution_decision.final_target_weights.get(asset_id, 0.0)
        ),
        "Execution_Adjustment": str(execution_adjustment),
    }


def strategy_signal_fields(
    strategy: Strategy,
    decision: StrategyDecision,
    asset_id: str,
    *,
    require_ranked: bool = False,
) -> dict[str, object]:
    """Return common and declared signals, leaving forced exits blank."""

    if asset_id not in decision.signal_audit.index or (
        require_ranked and asset_id not in decision.ranked_candidate_asset_ids
    ):
        return {
            "Strategy_Score": np.nan,
            "Strategy_Rank": np.nan,
            "Strategy_Eligible": False,
            "Strategy_Exclusion_Reason": "forced_exit_not_in_current_decision",
            **{column: np.nan for column in strategy.signal_columns},
        }
    signal = decision.signal_audit.loc[asset_id]
    return {
        "Strategy_Score": signal["Strategy_Score"],
        "Strategy_Rank": signal["Strategy_Rank"],
        "Strategy_Eligible": bool(signal["Strategy_Eligible"]),
        "Strategy_Exclusion_Reason": str(
            signal["Strategy_Exclusion_Reason"]
        ),
        **{column: signal[column] for column in strategy.signal_columns},
    }


__all__ = [
    "STRATEGY_SELECTION_AUDIT_COLUMNS",
    "STRATEGY_SIGNAL_AUDIT_COLUMNS",
    "TRADE_EXECUTION_AUDIT_COLUMNS",
    "strategy_selection_fields",
    "strategy_signal_fields",
    "trade_execution_audit_fields",
]
