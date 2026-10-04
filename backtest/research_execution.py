"""Research hard constraints around the shared planner and immutable ledger."""

import numpy as np

from portfolio_core.accounting_ledger import (
    LedgerRebalancePlan,
    execute_rebalance,
    InfeasibleRebalanceError,
    InvalidRebalanceError,
)
from portfolio_core.rebalance_planner import plan_rebalance_targets
from portfolio_core.strategies.research_execution import (
    require_positive_equity,
    research_accounting_config,
    sector_execution_residuals,
)


def execute_research_rebalance(context, state, period, ledger_state, prices, before):
    """Return targets and execution while preserving the selected long/short names.

    Retry without turnover suppression after a ledger constraint failure, or
    when suppressed targets need scaling or disturb sector neutrality.
    Raise InfeasibleRebalanceError if that retry fails or selected holdings are lost.
    """
    strategy = context.strategy
    p = strategy.parameters
    require_positive_equity(before.equity, period.t_0)
    nl = int((state.target_weights > 0).sum()) if strategy.strategy_id == "sector_momentum" else strategy.n_long
    ns = int((state.target_weights < 0).sum()) if strategy.strategy_id == "sector_momentum" else strategy.n_short
    config = research_accounting_config(context.accounting_config, nl, ns)
    sectors = period.execution_sector_rows.GICS_Sector_Code.astype(str).reindex(
        prices.index
    )
    # Departing names need only prior provenance; their applied target is zero.
    sectors = sectors.fillna("departed")
    weights = state.target_weights.reindex(prices.index).fillna(0)
    if p.sector_neutral and not np.allclose(
        weights.groupby(sectors).sum(), 0, atol=1e-12, rtol=0
    ):
        raise InfeasibleRebalanceError(
            f"{period.t_0}: ideal sector targets are not neutral"
        )

    def plan(suppress):
        return plan_rebalance_targets(
            ledger_state.shares,
            weights,
            prices,
            before.equity,
            turnover_threshold=strategy.turnover_threshold,
            apply_turnover_threshold=suppress,
            whole_share_orders=config.whole_share_orders,
        )

    def execute(targets):
        return execute_rebalance(
            ledger_state,
            LedgerRebalancePlan.from_series(
                targets.applied_shares,
                targets.target_weights,
                prices,
                period.dollar_volume.reindex(prices.index),
                period.interval_eligible_asset_ids,
                period.t_0,
            ),
            config,
            apply_fees=context.apply_fees,
            apply_spread=context.apply_spread,
        )

    unsuppressed = plan(False)
    targets = plan(context.apply_turnover_threshold)
    override = ""
    try:
        result = execute(targets)
        if result.feasibility_scale < 1 and not targets.applied_shares.equals(
            unsuppressed.applied_shares
        ):
            override = "hard_constraint_suppression_override"
        if p.sector_neutral:
            residuals = sector_execution_residuals(
                unsuppressed.applied_shares, ledger_state.shares,
                result.applied_shares, result.feasibility_scale, prices, sectors,
                whole_share_orders=config.whole_share_orders,
            )
            if not np.allclose(residuals.Suppression_Net_Dollars, 0, atol=1e-7, rtol=0):
                override = "neutrality_suppression_override"
    except (InfeasibleRebalanceError, InvalidRebalanceError):
        override = "hard_constraint_suppression_override"
    if override:
        targets = unsuppressed
        try:
            result = execute(targets)
        except (InfeasibleRebalanceError, InvalidRebalanceError) as exc:
            raise InfeasibleRebalanceError(
                f"{period.t_0}: {strategy.strategy_id} unsuppressed execution infeasible: {exc}"
            ) from exc
    applied = result.applied_shares.reindex(prices.index).fillna(0)
    if set(applied.index[applied > 0]) != set(weights.index[weights > 0]) or set(
        applied.index[applied < 0]
    ) != set(weights.index[weights < 0]):
        raise InfeasibleRebalanceError(
            f"{period.t_0}: execution lost configured holdings"
        )
    if context.strategy_audit is not None:
        context.strategy_audit.research_records.append(
            dict(
                Date=period.t_0,
                Status="executed",
                Constraint_Override=override,
                Post_Trade_Gross=result.after.gross_exposure,
                Post_Trade_Equity=result.after.equity,
                Feasibility_Scale=result.feasibility_scale,
                Fees=result.fixed_fees,
                Spread_Cost=result.spread_cost,
            )
        )
        if p.sector_neutral:
            ideal = weights * before.equity
            rounded = (
                unsuppressed.applied_shares.reindex(prices.index).fillna(0) * prices
            )
            requested = targets.applied_shares.reindex(prices.index).fillna(0) * prices
            final = applied * prices
            residuals = sector_execution_residuals(
                unsuppressed.applied_shares, ledger_state.shares,
                applied, result.feasibility_scale, prices, sectors,
                whole_share_orders=config.whole_share_orders,
            )
            for code in sorted(set(sectors[weights != 0])):
                group = sectors == code
                residual = float(final[group].sum())
                expected = float(residuals.loc[code, "Mechanical_Net_Dollars"])
                if not np.isclose(residual, expected, atol=1e-7, rtol=0):
                    raise RuntimeError(
                        "Neutral residual does not reconcile to mechanical rounding"
                    )
                context.strategy_audit.sector_records.append(
                    dict(
                        Date=period.t_0,
                        GICS_Sector_Code=code,
                        Target_Long_Dollars=float(ideal[group & (ideal > 0)].sum()),
                        Target_Short_Dollars=float(-ideal[group & (ideal < 0)].sum()),
                        Target_Net_Dollars=float(ideal[group].sum()),
                        Scaled_Ideal_Net_Dollars=float(
                            ideal[group].sum() * result.feasibility_scale
                        ),
                        Unsuppressed_Rounded_Net_Dollars=float(rounded[group].sum()),
                        Requested_Net_Dollars=float(requested[group].sum()),
                        Applied_Long_Dollars=float(final[group & (final > 0)].sum()),
                        Applied_Short_Dollars=float(-final[group & (final < 0)].sum()),
                        Applied_Net_Dollars=residual,
                        Mechanical_Net_Dollars=expected,
                        Rerounding_Effect=residual
                        - float(rounded[group].sum()) * result.feasibility_scale,
                        Residual_Bps=1e4 * residual / result.after.equity,
                        Feasibility_Scale=result.feasibility_scale,
                        Post_Trade_Equity=result.after.equity,
                        Fees=result.fixed_fees,
                        Spread_Cost=result.spread_cost,
                        Constraint_Override=override,
                    )
                )
    return targets, result
