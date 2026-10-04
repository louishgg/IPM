"""Dated equal-weight historical sector returns, separate from Brinson and NAV."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_core.portfolio_lifecycle import (
    eligible_assets_for_interval, event_delivery_execution_candidates,
    event_valuation_result, asset_has_interval_event, finite_positive_price,
)
from portfolio_core.sector_assignments import sector_rows_asof
from .data_loading import active_pit_asset_ids
from .config import DEFAULT_CONFIG

SECTOR_RETURN_VERSION = 'historical_equal_weight_sector_v1'


def interval_holding_returns(data, assets, start, end):
    """Canonical one-share returns, including delivered-successor forced closes."""
    deliveries = event_delivery_execution_candidates(data, start, end)
    overrides = {str(r.Asset_ID): float(r.Reference_Close) for r in deliveries.itertuples()}
    result = {}
    for asset in assets:
        initial = finite_positive_price(data.data_close, start, asset)
        if asset_has_interval_event(data, asset, start, end):
            valued = event_valuation_result(data, asset, start, end, successor_price_overrides=overrides)
            terminal = float(valued[1]) if valued is not None else None
        else:
            terminal = finite_positive_price(data.data_close, end, asset)
        if initial is None or terminal is None or not np.isfinite(terminal):
            raise RuntimeError(f'Unvalueable eligible holding {asset} over {start} -> {end}')
        result[asset] = terminal/initial-1
    return pd.Series(result, dtype=float)


@dataclass
class SectorReturnHistory:
    returns: pd.DataFrame
    constituents: pd.DataFrame
    summary: pd.DataFrame
    end: pd.Timestamp


def build_sector_returns(data, end, *, start=DEFAULT_CONFIG.evaluation.initial_research_start_date):
    """No classifications before January 2015; no intervals beyond requested end.

    Excluded rows retain available provenance but require no classification.
    Every included row requires the exact starting snapshot. No stale fallback.
    """
    end = pd.Timestamp(end)
    records, summaries = [], []
    dates = pd.date_range(start, end, freq='ME')
    for start, finish in zip(dates, dates[1:]):
        members = set(active_pit_asset_ids(data, start))
        eligible = eligible_assets_for_interval(data, members, start, finish)
        sectors = sector_rows_asof(data.sector_assignments, start)
        missing = eligible - set(sectors.index)
        if missing:
            raise ValueError(f'Missing exact-date sector basket classifications at {start}: {sorted(missing)}')
        values = interval_holding_returns(data, sorted(eligible), start, finish)
        groups = {}
        for asset in sorted(members):
            row = sectors.loc[asset].to_dict() if asset in sectors.index else {}
            code = str(row.get('GICS_Sector_Code', ''))
            included = asset in eligible
            reason = '' if included else (
                'missing_start_price' if finite_positive_price(data.data_close, start, asset) is None
                else 'canonical_lifecycle_or_terminal_valuation_exclusion')
            row.pop("Asset_ID", None)
            record = dict(Start=start, End=finish, Asset_ID=asset, **row,
                          Included=included, Exclusion_Reason=reason,
                          Holding_Return=values.get(asset, np.nan))
            records.append(record)
            if code:
                groups.setdefault(code, []).append(record)
        for code, rows in sorted(groups.items()):
            included = [r for r in rows if r['Included']]
            count = len(included)
            for row in rows:
                row['Basket_Weight'] = 1/count if row['Included'] else 0.
                row['Return_Contribution'] = row['Holding_Return']/count if row['Included'] else np.nan
            summaries.append(dict(Start=start, End=finish, GICS_Sector_Code=code,
                                  Member_Count=len(rows), Included_Count=count,
                                  Excluded_Count=len(rows)-count,
                                  Sector_Return=sum(r['Holding_Return'] for r in included)/count if count else np.nan))
    summary = pd.DataFrame(summaries)
    returns = (summary.pivot(index='End', columns='GICS_Sector_Code', values='Sector_Return')
               if not summary.empty else pd.DataFrame(index=pd.DatetimeIndex([])))
    returns = returns.reindex(dates[1:])
    return SectorReturnHistory(returns, pd.DataFrame(records), summary, end)
