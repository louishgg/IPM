"""Validated causal contract shared by every portfolio strategy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from .research_parameters import StockSelection, WholeSectorSelection
from .research_state import (
    SelectedSectors, SectorBasket,
)


COMMON_SIGNAL_COLUMNS = (
    "Strategy_Eligible",
    "Strategy_Exclusion_Reason",
    "Strategy_Score",
    "Strategy_Rank",
)

RESERVED_AUDIT_COLUMNS = frozenset({
    "Rebalance_ID",
    "Membership_Effective_Date",
    "Asset_ID",
    "Ticker",
    "Source_Ticker",
    "Yahoo_Ticker",
    "GICS_Sector_Code",
    "Sector",
    "Sector_As_Of_Date",
    "Sector_Source_Type",
    "Sector_Source_Reference",
    "Signal_Cutoff",
    "Signal_Source_Max_Date",
    "Sizing_Date",
    "Execution_Date",
    "Valuation_End",
    "Strategy_ID",
    "Strategy_Version",
    *COMMON_SIGNAL_COLUMNS,
    "Signal_Selected_Side",
    "Signal_Raw_Target_Weight",
    "Execution_Eligible",
    "Final_Selected_Side",
    "Final_Target_Weight",
    "Execution_Adjustment",
    "Execution_Reason",
    "Trigger_Event_ID",
    "Applied_Target_Weight",
    "Decision_Complete",
    "Used_Hold_Logic",
    "Applied_Rule",
    "Current_Capital",
    "Current_Weight",
    "Target_Weight",
    "Weight_Drift",
    "Turnover_Threshold",
    "Current_Position_Value",
    "Requested_Target_Position_Value",
    "Target_Position_Value",
    "Execution_Price",
    "Effective_Execution_Price",
    "Order_Count",
    "Fixed_Fee",
    "Spread_Rate",
    "Spread_Cost",
    "Turnover_Contribution",
    "Current_Shares",
    "Requested_Target_Shares",
    "Applied_Target_Shares",
    "Trade_Shares",
    "Sizing_Price",
    "Prior_Position_Value_At_Sizing",
    "Target_Position_Value_At_Sizing",
    "Trade_Notional",
    "Sizing_Weight_Drift",
    "Overnight_Price_Return",
    "Sizing_Rounding_Share_Delta",
    "Turnover_Suppression_Share_Delta",
    "Feasibility_Share_Delta",
    "Constraint_Override",
    "Cash_Effect",
    "Restricted_Proceeds_Change",
})


@dataclass(frozen=True)
class StrategyDecisionContext:
    """Causal monthly history and universe available at one signal cutoff."""

    close_history: pd.DataFrame
    volume_history: pd.DataFrame
    candidate_asset_ids: tuple[str, ...]
    sector_code_by_asset_id: Mapping[str, str]
    previous_target_weights: pd.Series
    signal_cutoff: pd.Timestamp
    signal_source_max_date: pd.Timestamp
    previous_selected_sectors: SelectedSectors = field(default_factory=SelectedSectors)
    sector_returns: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        close = self.close_history.copy()
        volume = self.volume_history.copy()
        if close.empty:
            raise ValueError("Strategy close history cannot be empty")
        if (
            not close.index.is_monotonic_increasing
            or close.index.has_duplicates
        ):
            raise ValueError("Strategy close-history dates must be unique and ordered")
        if not volume.index.equals(close.index) or not volume.columns.equals(
            close.columns
        ):
            raise ValueError("Strategy close and volume histories must be aligned")
        if close.columns.has_duplicates:
            raise ValueError("Strategy market history contains duplicate assets")
        close.columns = pd.Index(
            close.columns.astype(str),
            name="Asset_ID",
        )
        volume.columns = pd.Index(
            volume.columns.astype(str),
            name="Asset_ID",
        )
        if close.columns.has_duplicates:
            raise ValueError(
                "Strategy market history contains duplicate normalized assets"
            )
        candidates = tuple(str(value) for value in self.candidate_asset_ids)
        if len(candidates) != len(set(candidates)):
            raise ValueError("Strategy candidates must be unique")
        if candidates != tuple(sorted(candidates)):
            raise ValueError("Strategy candidates must be deterministically ordered")
        signal_cutoff = pd.Timestamp(self.signal_cutoff).tz_localize(None)
        signal_source_max_date = pd.Timestamp(
            self.signal_source_max_date
        ).tz_localize(None)
        returns = self.sector_returns.copy()
        if not returns.empty and (not returns.index.is_monotonic_increasing
                or returns.index.has_duplicates or returns.columns.has_duplicates
                or any(pd.Timestamp(d) > signal_cutoff for d in returns.index)
                or not returns.index.is_month_end.all()):
            raise ValueError("Sector returns must have unique, ordered, causal month ends")
        object.__setattr__(self, "sector_returns", returns)
        if not isinstance(self.previous_selected_sectors, SelectedSectors):
            raise ValueError("Invalid previous selected-sector state")
        if signal_source_max_date > signal_cutoff:
            raise ValueError("Strategy context contains data after the signal cutoff")
        if isinstance(close.index, pd.DatetimeIndex):
            observed_max = pd.Timestamp(close.index.max()).tz_localize(None)
            if observed_max > signal_cutoff:
                raise ValueError(
                    "Strategy context contains data after the signal cutoff"
                )
        missing_history = sorted(
            set(candidates) - set(close.columns.astype(str))
        )
        if missing_history:
            raise ValueError(
                f"Strategy candidates are missing from market history: {missing_history}"
            )
        missing_sectors = sorted(
            set(candidates) - set(map(str, self.sector_code_by_asset_id))
        )
        if missing_sectors:
            raise ValueError(
                f"Strategy candidates are missing exact-date sectors: {missing_sectors}"
            )
        previous = self.previous_target_weights.copy().astype(float)
        previous.index = previous.index.astype(str)
        previous.index.name = "Asset_ID"
        if previous.index.has_duplicates:
            raise ValueError("Previous target weights contain duplicate assets")
        if not np.isfinite(previous.to_numpy(dtype=float)).all():
            raise ValueError("Previous target weights must be finite")
        object.__setattr__(self, "close_history", close)
        object.__setattr__(self, "volume_history", volume)
        object.__setattr__(self, "candidate_asset_ids", candidates)
        object.__setattr__(
            self,
            "sector_code_by_asset_id",
            MappingProxyType({
                str(key): str(value)
                for key, value in self.sector_code_by_asset_id.items()
            }),
        )
        object.__setattr__(self, "previous_target_weights", previous)
        object.__setattr__(self, "signal_cutoff", signal_cutoff)
        object.__setattr__(
            self,
            "signal_source_max_date",
            signal_source_max_date,
        )


@dataclass(frozen=True)
class StrategyDecision:
    """Audited raw strategy decision before execution-date eligibility changes."""

    raw_target_weights: pd.Series
    signal_audit: pd.DataFrame
    ranked_candidate_asset_ids: tuple[str, ...]
    original_long_asset_ids: tuple[str, ...]
    original_short_asset_ids: tuple[str, ...]
    is_complete: bool
    summary_metrics: Mapping[str, float] = field(default_factory=dict)
    sector_basket: SectorBasket | None = None


@dataclass(frozen=True)
class ExecutionDecision:
    """Final strategy targets after applying execution-date eligibility."""

    final_target_weights: pd.Series
    final_long_asset_ids: tuple[str, ...]
    final_short_asset_ids: tuple[str, ...]
    eligibility_reasons: Mapping[str, str] = field(default_factory=dict)
    is_complete: bool = True
    sector_basket: SectorBasket | None = None


class Strategy(ABC):
    """Abstract, validated contract implemented by every strategy family."""

    strategy_id: str
    strategy_version: str
    signal_columns: tuple[str, ...]
    summary_columns: tuple[str, ...]

    @property
    @abstractmethod
    def parameters(self) -> object:
        """Return the immutable parameter object owned by this strategy."""

    @property
    @abstractmethod
    def turnover_threshold(self) -> float:
        """Return the engine drift threshold associated with the strategy."""

    @property
    def n_long(self) -> int:
        """Historical stock-count adapter; whole-sector strategies override selection."""
        raise TypeError("Whole-sector construction has no fixed stock count")

    @property
    def n_short(self) -> int:
        """Historical stock-count adapter; whole-sector strategies override selection."""
        raise TypeError("Whole-sector construction has no fixed stock count")

    @property
    def selection(self) -> StockSelection | WholeSectorSelection:
        """Construction contract; historical strategies retain exact stock counts."""
        return StockSelection(self.n_long, self.n_short)

    def decide(self, context: StrategyDecisionContext) -> StrategyDecision:
        decision = self._decide(context)
        self._validate_decision(context, decision)
        return decision

    def finalize_for_execution(
        self,
        decision: StrategyDecision,
        execution_eligible_asset_ids: set[str] | frozenset[str],
    ) -> ExecutionDecision:
        eligible = frozenset(str(value) for value in execution_eligible_asset_ids)
        finalized = self._finalize_for_execution(decision, eligible)
        self._validate_execution_decision(finalized, eligible)
        if isinstance(self.selection, WholeSectorSelection) and finalized.is_complete:
            if decision.sector_basket is None or dict(
                finalized.sector_basket.eligible_members
            ) != dict(decision.sector_basket.eligible_members):
                raise ValueError("Execution must reuse the saved causal basket pool")
        return finalized

    @abstractmethod
    def _decide(self, context: StrategyDecisionContext) -> StrategyDecision:
        """Build an unvalidated raw decision."""

    @abstractmethod
    def _finalize_for_execution(
        self,
        decision: StrategyDecision,
        execution_eligible_asset_ids: frozenset[str],
    ) -> ExecutionDecision:
        """Build final targets from only causal signal information."""

    def _validate_declared_columns(self) -> None:
        if not self.strategy_id.strip() or not self.strategy_version.strip():
            raise ValueError("Strategy identity and version cannot be blank")
        if re.fullmatch(r"\d+\.\d+\.\d+", self.strategy_version) is None:
            raise ValueError("Strategy version must use semantic x.y.z form")
        if len(self.signal_columns) != len(set(self.signal_columns)):
            raise ValueError("Strategy signal columns must be unique")
        collisions = sorted(set(self.signal_columns) & RESERVED_AUDIT_COLUMNS)
        if collisions:
            raise ValueError(
                f"Strategy signal columns collide with reserved audit columns: {collisions}"
            )
        if len(self.summary_columns) != len(set(self.summary_columns)):
            raise ValueError("Strategy summary columns must be unique")

    def _validate_decision(
        self,
        context: StrategyDecisionContext,
        decision: StrategyDecision,
    ) -> None:
        self._validate_declared_columns()
        expected_columns = [*COMMON_SIGNAL_COLUMNS, *self.signal_columns]
        if list(decision.signal_audit.columns) != expected_columns:
            raise ValueError(
                "Strategy audit columns do not match the declared contract: "
                f"expected={expected_columns}, "
                f"actual={list(decision.signal_audit.columns)}"
            )
        if decision.signal_audit.index.has_duplicates:
            raise ValueError("Strategy audit contains duplicate assets")
        if (
            tuple(decision.signal_audit.index.astype(str))
            != context.candidate_asset_ids
        ):
            raise ValueError("Strategy audit must cover candidates in deterministic order")
        if tuple(decision.ranked_candidate_asset_ids) != tuple(
            dict.fromkeys(decision.ranked_candidate_asset_ids)
        ):
            raise ValueError("Strategy ranking contains duplicate assets")
        if not set(decision.ranked_candidate_asset_ids).issubset(
            set(context.candidate_asset_ids)
        ):
            raise ValueError("Strategy ranking contains an unknown candidate")
        if set(decision.original_long_asset_ids) & set(
            decision.original_short_asset_ids
        ):
            raise ValueError("Strategy selected an asset on both sides")
        if len(decision.original_long_asset_ids) != len(
            set(decision.original_long_asset_ids)
        ):
            raise ValueError("Strategy selected duplicate long assets")
        if len(decision.original_short_asset_ids) != len(
            set(decision.original_short_asset_ids)
        ):
            raise ValueError("Strategy selected duplicate short assets")
        ranked_rows = decision.signal_audit.loc[
            list(decision.ranked_candidate_asset_ids)
        ]
        expected_ranks = np.arange(1, len(ranked_rows) + 1, dtype=float)
        if not np.array_equal(
            pd.to_numeric(ranked_rows["Strategy_Rank"]).to_numpy(dtype=float),
            expected_ranks,
        ):
            raise ValueError("Strategy ranking and common rank audit disagree")
        if not ranked_rows["Strategy_Eligible"].astype(bool).all():
            raise ValueError("Ranked strategy assets must be signal-eligible")
        if not np.isfinite(
            pd.to_numeric(ranked_rows["Strategy_Score"]).to_numpy(dtype=float)
        ).all():
            raise ValueError("Ranked strategy scores must be finite")
        self._validate_weights(
            decision.raw_target_weights,
            decision.original_long_asset_ids,
            decision.original_short_asset_ids,
            decision.is_complete,
        )
        self._validate_basket(
            decision.sector_basket, decision.original_long_asset_ids,
            decision.original_short_asset_ids, decision.is_complete,
        )
        if decision.sector_basket is not None:
            pool = {
                a for members in decision.sector_basket.eligible_members.values()
                for a in members
            }
            audited_pool = set(decision.signal_audit.index[
                decision.signal_audit["Strategy_Eligible"].astype(bool)
            ])
            if pool != audited_pool:
                raise ValueError("Basket pool must cover all audited eligible assets")
            for code, members in decision.sector_basket.eligible_members.items():
                if any(
                    a not in context.candidate_asset_ids
                    or context.sector_code_by_asset_id[a] != code
                    for a in members
                ):
                    raise ValueError("Basket disagrees with current candidates or exact-date sectors")
        if tuple(decision.summary_metrics) != self.summary_columns:
            raise ValueError("Strategy summary metrics do not match declared columns")
        if any(
            not np.isfinite(float(value)) and not np.isnan(float(value))
            for value in decision.summary_metrics.values()
        ):
            raise ValueError("Strategy summary metrics contain an invalid value")

    def _validate_execution_decision(
        self,
        decision: ExecutionDecision,
        eligible: frozenset[str],
    ) -> None:
        selected = set(decision.final_long_asset_ids) | set(
            decision.final_short_asset_ids
        )
        if not selected.issubset(eligible):
            raise ValueError("Final strategy decision contains an ineligible asset")
        if set(decision.final_long_asset_ids) & set(
            decision.final_short_asset_ids
        ):
            raise ValueError("Final strategy decision overlaps long and short sides")
        if len(decision.final_long_asset_ids) != len(
            set(decision.final_long_asset_ids)
        ):
            raise ValueError("Final strategy decision contains duplicate longs")
        if len(decision.final_short_asset_ids) != len(
            set(decision.final_short_asset_ids)
        ):
            raise ValueError("Final strategy decision contains duplicate shorts")
        self._validate_weights(
            decision.final_target_weights,
            decision.final_long_asset_ids,
            decision.final_short_asset_ids,
            decision.is_complete,
        )
        self._validate_basket(
            decision.sector_basket, decision.final_long_asset_ids,
            decision.final_short_asset_ids, decision.is_complete, eligible=eligible,
        )

    def _validate_basket(self, basket, longs, shorts, complete, *, eligible=None) -> None:
        selection = self.selection
        if isinstance(selection, StockSelection):
            if basket is not None:
                raise ValueError("Stock selections cannot declare whole-sector baskets")
        elif complete:
            if not isinstance(basket, SectorBasket):
                raise ValueError("Whole-sector decisions require basket metadata")
            if (
                len(basket.selected.longs) != selection.n_long_sectors
                or len(basket.selected.shorts) != selection.n_short_sectors
            ):
                raise ValueError("Complete strategy decision has the wrong sector counts")
            basket.validate_holdings(longs, shorts, eligible=eligible)
        elif basket is not None:
            raise ValueError("Incomplete decisions cannot declare accepted baskets")

    def _validate_weights(
        self,
        weights: pd.Series,
        long_asset_ids: tuple[str, ...],
        short_asset_ids: tuple[str, ...],
        is_complete: bool,
    ) -> None:
        selected = weights.copy().astype(float)
        selected.index = selected.index.astype(str)
        if selected.index.has_duplicates:
            raise ValueError("Strategy weights contain duplicate assets")
        if not np.isfinite(selected.to_numpy(dtype=float)).all():
            raise ValueError("Strategy weights must be finite")
        if (selected.abs() <= 0.0).any():
            raise ValueError("Strategy weights cannot contain zero entries")
        long_set = set(str(value) for value in long_asset_ids)
        short_set = set(str(value) for value in short_asset_ids)
        if tuple(selected.index) != (*long_asset_ids, *short_asset_ids):
            raise ValueError(
                "Strategy weights must follow deterministic long-then-short order"
            )
        if set(selected.index[selected > 0]) != long_set:
            raise ValueError("Positive weights do not match selected longs")
        if set(selected.index[selected < 0]) != short_set:
            raise ValueError("Negative weights do not match selected shorts")
        if is_complete:
            selection = self.selection
            if isinstance(selection, StockSelection) and (
                len(long_set) != selection.n_long or len(short_set) != selection.n_short
            ):
                raise ValueError("Complete strategy decision has the wrong side counts")
            if isinstance(selection, WholeSectorSelection) and (
                len(long_set) < 10 or len(short_set) < 10
            ):
                raise ValueError("Whole-sector decisions require at least ten stocks per side")
        elif not selected.empty or long_set or short_set:
            raise ValueError("Incomplete strategy decisions must have empty targets")


def empty_audit(
    candidate_asset_ids: tuple[str, ...],
    signal_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Return a deterministic candidate audit initialized as ineligible."""

    frame = pd.DataFrame(index=pd.Index(candidate_asset_ids, name="Asset_ID"))
    frame["Strategy_Eligible"] = False
    frame["Strategy_Exclusion_Reason"] = "insufficient_signal_history"
    frame["Strategy_Score"] = np.nan
    frame["Strategy_Rank"] = np.nan
    for column in signal_columns:
        frame[column] = np.nan
    return frame[[*COMMON_SIGNAL_COLUMNS, *signal_columns]]


def empty_decision(
    signal_audit: pd.DataFrame,
    summary_columns: tuple[str, ...],
) -> StrategyDecision:
    return StrategyDecision(
        raw_target_weights=pd.Series(dtype=float),
        signal_audit=signal_audit,
        ranked_candidate_asset_ids=(),
        original_long_asset_ids=(),
        original_short_asset_ids=(),
        is_complete=False,
        summary_metrics=MappingProxyType({
            column: np.nan for column in summary_columns
        }),
    )


__all__ = [
    "COMMON_SIGNAL_COLUMNS",
    "ExecutionDecision",
    "RESERVED_AUDIT_COLUMNS",
    "Strategy",
    "StrategyDecision",
    "StrategyDecisionContext",
    "empty_audit",
    "empty_decision",
]
