"""Read saved formation targets and causal, identity-preserving live history."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from live.analysis_data import LiveAnalysisInputs, load_analysis_inputs
from live.paths import LivePaths
from live.research_history import interval_holding_returns
from portfolio_core.artifacts import file_sha256
from portfolio_core.interval_valuation import has_interval_event
from portfolio_core.price_basis import price_basis_spec
from portfolio_core.simulation_assumptions import load_simulation_assumptions
from portfolio_core.strategies import load_saved_strategy_parameters

from .contracts import HistorySample, PortfolioSnapshot


def snapshot_from_tables(decisions, nav, packet, assumptions, run_dir) -> PortfolioSnapshot:
    """Select the earliest nonzero formation, never a later successful substitute."""
    required = {"Asset_ID", "Source_Ticker", "Yahoo_Ticker", "GICS_Sector_Code", "Sector",
                "Rebalance_ID", "Signal_Cutoff", "Signal_Source_Max_Date", "Sizing_Date",
                "Execution_Date", "Strategy_ID", "Strategy_Version", "Signal_Raw_Target_Weight",
                "Sector_As_Of_Date", "Sector_Source_Type", "Sector_Source_Reference"}
    if missing := required - set(decisions):
        raise ValueError(f"Saved decisions missing columns: {sorted(missing)}")
    decisions = decisions.copy()
    for column in ("Signal_Cutoff", "Signal_Source_Max_Date", "Sizing_Date", "Execution_Date", "Sector_As_Of_Date"):
        decisions[column] = pd.to_datetime(decisions[column], format="%Y-%m-%d", errors="raise")
    weights = pd.to_numeric(decisions.Signal_Raw_Target_Weight, errors="raise")
    if not np.isfinite(weights).all() or decisions.Signal_Cutoff.isna().any():
        raise ValueError("Saved decisions contain nonfinite weights or undated formations")
    decisions["Signal_Raw_Target_Weight"] = weights
    nonzero = decisions.loc[weights.ne(0)]
    if nonzero.empty:
        raise ValueError("Saved run has no nonzero formation targets")
    formation = nonzero.Signal_Cutoff.min()
    rows = decisions.loc[decisions.Signal_Cutoff.eq(formation)].copy()
    for column in ("Rebalance_ID", "Sizing_Date", "Execution_Date", "Strategy_ID", "Strategy_Version"):
        if rows[column].isna().any() or rows[column].astype(str).eq("").any() or rows[column].nunique() != 1:
            raise ValueError(f"First formation has inconsistent {column}")
    if rows.Asset_ID.eq("").any() or rows.Asset_ID.isna().any() or rows.Asset_ID.duplicated().any():
        raise ValueError("First formation has blank or duplicate canonical Asset_ID values")
    if rows.Strategy_ID.iloc[0] != packet["strategy_id"] or rows.Strategy_Version.iloc[0] != packet["strategy_version"]:
        raise ValueError("Saved strategy identity disagrees with decisions")
    positions = rows.loc[rows.Signal_Raw_Target_Weight.ne(0)].sort_values("Asset_ID").reset_index(drop=True)
    if (positions.Signal_Source_Max_Date.isna().any()
            or positions.Signal_Source_Max_Date.gt(formation).any()
            or positions.Sector_As_Of_Date.isna().any()
            or positions.Sector_As_Of_Date.gt(formation).any()):
        raise ValueError("First formation contains unavailable signal or sector evidence")
    sizing, execution = rows.Sizing_Date.iloc[0], rows.Execution_Date.iloc[0]
    if sizing != formation or execution <= formation:
        raise ValueError("First formation requires same-day sizing and subsequent execution")
    if not {"Rebalance_ID", "Sizing_NAV", "Period_Start", "Period_Type"}.issubset(nav):
        raise ValueError("Saved NAV lacks sizing-NAV provenance")
    rebalance = str(rows.Rebalance_ID.iloc[0])
    nav_row = nav.loc[nav.Rebalance_ID.eq(rebalance)]
    if len(nav_row) != 1 or nav_row.Period_Type.iloc[0] != "invested" or pd.Timestamp(nav_row.Period_Start.iloc[0]) != execution:
        raise ValueError(f"Sizing NAV must match exactly one invested row for {rebalance}")
    sizing_nav = float(nav_row.Sizing_NAV.iloc[0])
    if not np.isfinite(sizing_nav) or sizing_nav <= 0:
        raise ValueError("Sizing NAV must be finite and positive")
    sector_neutral = packet["parameters"].get("sector_neutral", False)
    if not isinstance(sector_neutral, bool):
        raise ValueError("Saved sector_neutral must be a boolean")
    if positions.GICS_Sector_Code.isna().any() or positions.GICS_Sector_Code.astype(str).eq("").any():
        raise ValueError("First formation has missing GICS sectors")
    return PortfolioSnapshot(packet["strategy_id"], packet["strategy_version"], rebalance,
                             formation, execution, formation, sizing, sizing_nav, positions,
                             sector_neutral, packet, assumptions, Path(run_dir))


def load_inputs(run_dir: Path, paths: LivePaths | None = None):
    """Reuse the live validators, including reviewed event/dividend contracts."""
    paths = paths or LivePaths.from_package()
    run_dir = Path(run_dir).resolve()
    parameter_file = run_dir / "strategy_parameters.json"
    raw = json.loads(parameter_file.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("strategy_id"), str):
        raise ValueError(f"Saved strategy parameters must declare a strategy_id: {parameter_file}")
    packet = load_saved_strategy_parameters(raw["strategy_id"], parameter_file)
    assumptions_path = run_dir / "simulation_assumptions.json"
    try:
        assumptions = load_simulation_assumptions(assumptions_path)
    except ValueError as exc:
        if "inconsistent price basis" not in str(exc):
            raise
        saved_basis = json.loads(assumptions_path.read_text()).get("price_basis", {}).get("price_basis_id")
        raise ValueError(f"Saved run {run_dir.name!r} declares price basis {saved_basis!r}; "
                         f"current live history requires {price_basis_spec('live').price_basis_id!r}. "
                         "Rerun the live strategy with the current pipeline; do not merely relabel its metadata.") from exc
    inputs = load_analysis_inputs(paths)
    if assumptions["price_basis"] != inputs.price_basis.as_assumptions():
        raise ValueError("Saved run and prepared live history use different price bases")
    decisions_path = run_dir / "tables/strategy/decisions.csv"
    nav_path = run_dir / "tables/strategy/nav.csv"
    snapshot = snapshot_from_tables(pd.read_csv(decisions_path, keep_default_na=False),
                                    pd.read_csv(nav_path, keep_default_na=False), packet, assumptions, run_dir)
    # Keep provenance independent of the checkout location. External runs use
    # saved_run/ relative to --run-dir; file hashes retain the inspected vintage.
    project_root = paths.project_root.resolve()
    run_reference = (run_dir.relative_to(project_root) if run_dir.is_relative_to(project_root)
                     else Path("saved_run"))
    sources = {(run_reference / p.relative_to(run_dir)).as_posix(): file_sha256(p)
               for p in (parameter_file, assumptions_path, decisions_path, nav_path)}
    sources.update({p.relative_to(paths.project_root).as_posix(): file_sha256(p)
                    for p in sorted(paths.prepared_data_dir.glob("*.csv"))})
    return snapshot, inputs, sources


def build_history(inputs: LiveAnalysisInputs, snapshot: PortfolioSnapshot, lookback: int) -> HistorySample:
    """Compute adjacent returns before complete-case selection; no filling or stitching."""
    cutoff = snapshot.information_cutoff
    end = cutoff + pd.offsets.MonthEnd(0)
    labels = pd.date_range(end=end, periods=lookback + 1, freq="ME", name="Month")
    # Calendar excludes future observations even within the cutoff's month label.
    monthly = inputs.market_monthly
    known = monthly.loc[monthly.Month.le(end) & monthly.Observation_Date.le(cutoff)]
    calendar = known.groupby("Month").Observation_Date.max().to_dict()
    indexed = monthly.set_index(["Month", "Asset_ID"])
    assets = snapshot.assets
    returns = pd.DataFrame(np.nan, index=labels[1:], columns=assets)
    missing = []
    actions = inputs.corporate_actions

    def price_reason(asset, label):
        if (label, asset) not in indexed.index:
            return "missing_month"
        row = indexed.loc[(label, asset)]
        if pd.isna(row.Observation_Date) or row.Observation_Date > cutoff:
            return "future_observation"
        if pd.isna(row.Available_Date) or row.Available_Date > cutoff:
            return "unavailable_evidence"
        if row.Observation_Date != calendar.get(label):
            return "stale_observation"
        if not np.isfinite(row.Close) or row.Close <= 0:
            return "missing_positive_close"
        return None

    for start, finish in zip(labels, labels[1:]):
        for asset in assets:
            reason = price_reason(asset, start)
            endpoint = start
            actual_start, actual_end = calendar.get(start), calendar.get(finish)
            event = (actual_start is not None and actual_end is not None
                     and has_interval_event(actions.events, actions.legs, asset, actual_start, actual_end))
            if reason is None and event:
                # An extinguished predecessor need not have an end price. Use the
                # reviewed one-share valuation, never an ordinary ratio across it.
                value = interval_holding_returns(inputs, [asset], start, finish, as_of=cutoff,
                                                monthly_index=indexed, calendar=calendar).loc[asset]
            else:
                if reason is None:
                    reason, endpoint = price_reason(asset, finish), finish
                if reason is None:
                    value = indexed.loc[(finish, asset), "Close"] / indexed.loc[(start, asset), "Close"] - 1
            if reason is not None:
                missing.append({"Month": finish, "Asset_ID": asset, "Endpoint": endpoint, "Reason": reason})
            elif not np.isfinite(value):
                raise ValueError(f"Nonfinite historical return for {asset} at {finish.date()}")
            else:
                returns.loc[finish, asset] = float(value)
    coverage = []
    for asset in assets:
        valid = returns[asset].dropna()
        coverage.append({"Asset_ID": asset, "Valid_Returns": len(valid),
                         "First_Return": None if valid.empty else valid.index.min(),
                         "Last_Return": None if valid.empty else valid.index.max(),
                         "Missing_Intervals": int(returns[asset].isna().sum())})
    return HistorySample(returns.dropna(how="any"),
                         pd.DataFrame(missing, columns=["Month", "Asset_ID", "Endpoint", "Reason"]),
                         pd.DataFrame(coverage), labels[-1])
