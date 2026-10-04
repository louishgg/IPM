"""Daily live valuation and performance from in-memory executed account records.

Normal analysis passes its current DataFrames directly. The disposable archive
repair loads the same records first. This module neither reads outputs nor
selects, sizes, or executes trades; valuation states never feed the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_core.accounting_ledger import LedgerState, account_snapshot, project_financing_amounts
from portfolio_core.interval_valuation import has_any_interval_event
from portfolio_core.performance_metrics import calculate_risk_metrics
from portfolio_core.portfolio_lifecycle import advance_ledger
from portfolio_core.price_basis import PriceBasisSpec
from .corporate_action_policy import LiveCorporateActionBundle


METRICS_METHODOLOGY = "daily_close_excess_capm_v1"
PERFORMANCE_METADATA_COLUMNS = (
    "Metrics_Methodology", "Return_Frequency", "Periods_Per_Year",
    "Risk_Start", "Risk_End", "Risk_Observation_Count", "Risk_Free_Convention",
)
RECONCILIATION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class DailyValuationData:
    """Prepared daily panels shared across candidates, with exact price lookups."""

    market: pd.DataFrame
    closes: pd.DataFrame
    benchmark: pd.DataFrame
    actions: LiveCorporateActionBundle

    @classmethod
    def from_frames(cls, market, benchmark, actions, *, start, end):
        market = market.loc[pd.to_datetime(market.Date).between(start, end)].copy()
        benchmark = benchmark.loc[pd.to_datetime(benchmark.Date).between(start, end)].copy()
        market["Date"] = pd.to_datetime(market.Date)
        benchmark["Date"] = pd.to_datetime(benchmark.Date)
        if market.duplicated(["Date", "Asset_ID"]).any() or benchmark.Date.duplicated().any():
            raise ValueError("Daily valuation requires unique asset/date and benchmark dates")
        sessions = pd.DatetimeIndex(sorted(market.Date.unique()))
        benchmark = benchmark.set_index("Date").sort_index()
        if not sessions.equals(benchmark.index):
            raise ValueError("Prepared equity sessions and benchmark dates must match for daily valuation")
        if sessions.empty or (sessions.dayofweek >= 5).any():
            raise ValueError("Daily valuation requires nonempty trading-session observations")
        if not np.isfinite(benchmark[["Open", "Close"]]).all().all() or (benchmark[["Open", "Close"]] <= 0).any().any():
            raise ValueError("Daily benchmark prices must be finite and positive")
        return cls(
            market.set_index(["Date", "Asset_ID"]).sort_index(),
            market.pivot(index="Date", columns="Asset_ID", values="Close"),
            benchmark, actions,
        )

    def prices(self, date, field, assets):
        index = pd.MultiIndex.from_product([[pd.Timestamp(date)], assets])
        values = self.market[field].reindex(index)
        if not np.isfinite(values).all() or values.le(0).any():
            missing = [str(key[1]) for key in values.index[~np.isfinite(values) | values.le(0)]]
            raise ValueError(f"Missing prepared {field} prices for daily valuation on {pd.Timestamp(date).date()}: {missing}")
        return pd.Series(values.to_numpy(), index=assets, dtype=float)

    def has_position_event(self, state, end):
        return has_any_interval_event(
            self.actions.events, self.actions.legs,
            (asset_id for asset_id, _ in state.position_items), state.state_date, end,
        )

    def advance(self, state, end, accounting):
        if self.has_position_event(state, end):
            return advance_ledger(
                state, end, self.actions.events, self.actions.legs,
                self.actions.sources, accounting_config=accounting,
            ).state
        # Same financing kernel, without rebuilding unrelated action audits on
        # every daily mark. Historical restricted proceeds are never repriced.
        cash, _, _, _ = project_financing_amounts(
            state.cash, state.restricted_total, state.state_date, end, accounting,
        )
        return LedgerState._from_series(state.shares, cash, state.restricted_short_proceeds, end)


def build_daily_nav(nav, holdings, trades, data: DailyValuationData, accounting):
    """Value executed positions daily and reconcile every saved period boundary.

Cash effects and restricted-proceeds changes come from actual trade records.
The original monthly NAV audit remains authoritative and is not modified.
    """
    from .analysis_data import validate_account_nav

    nav = validate_account_nav(nav)
    if nav.empty:
        raise ValueError("Cannot value an empty account history")
    errors = []

    def reconcile(actual, expected, label):
        if not np.isfinite(actual) or not np.isfinite(expected):
            raise ValueError(f"Nonfinite daily valuation reconciliation: {label}")
        error = abs(float(actual) - float(expected))
        errors.append(error)
        if error > RECONCILIATION_TOLERANCE:
            raise ValueError(f"Daily valuation does not reconcile {label}: {actual} != {expected}")

    start, end = nav.Period_Start.iloc[0], nav.Period_End.iloc[-1]
    sessions = data.benchmark.index[(data.benchmark.index >= start) & (data.benchmark.index <= end)]
    if start not in sessions or end not in sessions:
        raise ValueError("Daily valuation is missing an evaluation boundary")
    if holdings.duplicated(["Rebalance_ID", "Asset_ID"]).any() or trades.duplicated(["Rebalance_ID", "Asset_ID"]).any():
        raise ValueError("Duplicate executed holding or trade")
    valid_ids = set(nav.loc[nav.Period_Type.eq("invested"), "Rebalance_ID"])
    if not set(holdings.Rebalance_ID).issubset(valid_ids) or not set(trades.Rebalance_ID).issubset(valid_ids):
        raise ValueError("Executed records reference an unknown rebalance")
    state = LedgerState.initial(accounting, state_date=start)
    daily = {}
    for period in nav.itertuples(index=False):
        benchmark_return = (
            data.benchmark.at[period.Period_End, period.End_Field]
            / data.benchmark.at[period.Period_Start, period.Start_Field] - 1
        )
        if not np.isclose(benchmark_return, period.Benchmark_Return, rtol=0, atol=1e-12):
            raise ValueError("Daily valuation does not reconcile the saved benchmark return")
        state = data.advance(state, period.Period_Start, accounting)
        before = account_snapshot(state, data.prices(period.Period_Start, period.Start_Field, state.shares.index))
        reconcile(before.equity, period.Start_NAV, "pre-trade NAV")
        h = holdings.loc[holdings.Rebalance_ID.eq(period.Rebalance_ID)].set_index("Asset_ID")
        t = trades.loc[trades.Rebalance_ID.eq(period.Rebalance_ID)]
        quantities, restricted = state.shares.copy(), state.restricted_short_proceeds.copy()
        for trade in t.itertuples(index=False):
            reconcile(float(quantities.get(trade.Asset_ID, 0)), trade.Current_Shares, "pre-trade shares")
            reconcile(trade.Applied_Target_Shares - trade.Current_Shares, trade.Trade_Shares, "trade shares")
            reconcile(trade.Cash_Effect, -trade.Trade_Shares * trade.Execution_Price - trade.Fixed_Fee - trade.Spread_Cost, "trade cash effect")
            quantities.loc[trade.Asset_ID] = trade.Applied_Target_Shares
            restricted.loc[trade.Asset_ID] = restricted.get(trade.Asset_ID, 0.0) + trade.Restricted_Proceeds_Change
        quantities = quantities.loc[quantities.ne(0)]
        for asset in quantities.index.union(h.index):
            reconcile(float(quantities.get(asset, 0)), float(h.Shares.get(asset, 0)), "post-trade shares")
        # Cancellation of saved decimal trade amounts can leave tiny residuals.
        restricted = restricted.mask(restricted.abs() < 1e-7, 0.0)
        cash = state.cash + float(t.Cash_Effect.sum())
        reconcile(cash, period.Post_Trade_Cash, "post-trade cash")
        reconcile(float(t.Fixed_Fee.sum()), period.Fixed_Fees, "fixed fees")
        reconcile(float(t.Spread_Cost.sum()), period.Spread_Cost, "spread costs")
        anchor = LedgerState._from_series(quantities, cash, restricted, period.Period_Start)
        reconcile(anchor.restricted_total, period.Restricted_Short_Proceeds, "restricted proceeds")
        post = account_snapshot(anchor, data.prices(period.Period_Start, period.Start_Field, quantities.index))
        reconcile(post.equity, period.Post_Trade_NAV, "post-trade NAV")
        dates = sessions[(sessions >= period.Period_Start) & (
            (sessions < period.Period_End) if period.End_Field == "Open" else (sessions <= period.Period_End)
        )]
        if not data.has_position_event(anchor, period.Period_End):
            prices = data.closes.reindex(index=dates, columns=quantities.index)
            if not np.isfinite(prices).all().all() or prices.le(0).any().any():
                # Exact diagnostic rather than forward-filling a held price.
                for date in dates:
                    data.prices(date, "Close", quantities.index)
            values = prices.to_numpy() @ quantities.to_numpy()
            for date, value in zip(dates, values):
                marked_cash, _, _, _ = project_financing_amounts(
                    anchor.cash, anchor.restricted_total, anchor.state_date, date, accounting,
                )
                daily[date] = (marked_cash + value, marked_cash, anchor.restricted_total)
        else:
            for date in dates:
                marked = data.advance(anchor, date, accounting)
                snapshot = account_snapshot(marked, data.prices(date, "Close", marked.shares.index))
                daily[date] = (snapshot.equity, marked.cash, marked.restricted_total)
        state = data.advance(anchor, period.Period_End, accounting)
        endpoint = account_snapshot(state, data.prices(period.Period_End, period.End_Field, state.shares.index))
        reconcile(state.cash, period.End_Cash, "end cash")
        reconcile(state.restricted_total, period.End_Restricted_Short_Proceeds, "end restricted proceeds")
        reconcile(endpoint.equity, period.End_NAV, "end NAV")
    rows = [{"Date": start, "Field": nav.Start_Field.iloc[0], "Observation": "inception",
             "NAV": accounting.initial_capital, "Cash": accounting.initial_capital, "Restricted_Short_Proceeds": 0.0}]
    rows.extend({"Date": date, "Field": "Close", "Observation": "session_close", "NAV": value,
                 "Cash": cash, "Restricted_Short_Proceeds": restricted}
                for date, (value, cash, restricted) in sorted(daily.items()))
    if nav.End_Field.iloc[-1] == "Open":
        rows.append({"Date": end, "Field": "Open", "Observation": "final_open", "NAV": state.cash + endpoint.position_values.sum(),
                     "Cash": state.cash, "Restricted_Short_Proceeds": state.restricted_total})
    result = pd.DataFrame(rows)
    if not np.isfinite(result.NAV).all() or result.NAV.le(0).any():
        raise ValueError("Daily performance requires finite positive account NAV")
    result["Benchmark_Level"] = [float(data.benchmark.at[r.Date, r.Field]) for r in result.itertuples()]
    result["Portfolio_Return"] = result.NAV.pct_change(fill_method=None)
    result["Benchmark_Return"] = result.Benchmark_Level.pct_change(fill_method=None)
    days = result.Date.diff().dt.days
    result["Risk_Free_Return"] = (1 + accounting.cash_interest_rate / accounting.day_count_days) ** days - 1
    result["Include_In_Risk_Metrics"] = (
        result.Observation.eq("session_close") & result.Observation.shift().eq("session_close") & days.gt(0)
    )
    reconcile(result.NAV.iloc[-1], nav.End_NAV.iloc[-1], "final daily NAV")
    result.attrs["maximum_reconciliation_error_dollars"] = max(errors, default=0.0)
    return result


def performance_summary(nav, daily_nav, accounting, simulation_label, price_basis: PriceBasisSpec, simulation_fingerprint):
    """Full-window geometric growth plus complete-session arithmetic risk."""
    eligible = daily_nav.Include_In_Risk_Metrics.astype(bool)
    risk = daily_nav.loc[eligible].copy()
    risk_start = daily_nav.loc[risk.index[0] - 1, "Date"] if len(risk) else daily_nav.Date.iloc[0]
    metadata = {
        "Metrics_Methodology": METRICS_METHODOLOGY, "Return_Frequency": "daily_close_to_close",
        "Periods_Per_Year": 252, "Risk_Start": risk_start,
        "Risk_End": risk.Date.iloc[-1] if len(risk) else risk_start,
        "Risk_Observation_Count": len(risk),
        "Risk_Free_Convention": f"cash_apr_actual_{accounting.day_count_days}",
    }
    start, end = pd.Timestamp(nav.Period_Start.iloc[0]), pd.Timestamp(nav.Period_End.iloc[-1])
    annualizer = 365.0 / max(1, (end - start).days)
    cumulative = float(nav.End_NAV.iloc[-1] / accounting.initial_capital - 1)
    benchmark_cumulative = float(np.prod(1 + nav.Benchmark_Return.astype(float)) - 1)
    annual_return = (1 + cumulative) ** annualizer - 1
    benchmark_annual = (1 + benchmark_cumulative) ** annualizer - 1
    rows = []
    for is_benchmark in (False, True):
        returns = risk.Benchmark_Return if is_benchmark else risk.Portfolio_Return
        metrics = calculate_risk_metrics(returns, risk.Benchmark_Return, risk.Risk_Free_Return, periods_per_year=252, tolerance=1e-12)
        rows.append({
            "Series": "S&P 500 Total Return" if is_benchmark else "Live_Strategy",
            "Evaluation_Start": start, "Evaluation_End": end,
            "Cumulative_Return": benchmark_cumulative if is_benchmark else cumulative,
            "Annualized_Return": benchmark_annual if is_benchmark else annual_return,
            "Annualized_Volatility": metrics["Portfolio_Ann_Vol"],
            "Sharpe_Ratio": metrics["Sharpe"],
            "Excess_Return_Versus_Benchmark": 0.0 if is_benchmark else annual_return - benchmark_annual,
            "Excess_Return_Cumulative": 0.0 if is_benchmark else cumulative - benchmark_cumulative,
            "Active_Return_Ann_Arith": metrics["Active_Return_Ann_Arith"],
            "Tracking_Error": metrics["Tracking_Error_Ann"], "Information_Ratio": metrics["Information_Ratio"],
            "Beta": metrics["Beta"], "Treynor_Ratio": metrics["Treynor"], "CAPM_Alpha": metrics["CAPM_Alpha_OLS_Ann"],
            "Alpha_P_Value": metrics["Alpha_P_Value"], "Beta_P_Value": metrics["Beta_P_Value"],
            "Label": "Prepared S&P 500 Total Return benchmark" if is_benchmark else simulation_label,
            "Price_Basis": "sp500_total_return_benchmark" if is_benchmark else price_basis.price_basis_id,
            "Simulation_Fingerprint": simulation_fingerprint, **metadata,
        })
    return pd.DataFrame(rows)
