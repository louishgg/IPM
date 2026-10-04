"""Immutable selected-sector and whole-basket state."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


def _sectors(values) -> tuple[str, ...]:
    result = tuple(values)
    if len(set(result)) != len(result) or any(
        not isinstance(v, str) or len(v) != 2 or not v.isdigit() for v in result
    ):
        raise ValueError("Sector IDs must be unique two-digit GICS codes")
    return result


@dataclass(frozen=True, slots=True)
class SelectedSectors:
    longs: tuple[str, ...] = ()
    shorts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "longs", _sectors(self.longs))
        object.__setattr__(self, "shorts", _sectors(self.shorts))
        if set(self.longs) & set(self.shorts):
            raise ValueError("Selected sectors overlap")


@dataclass(frozen=True, slots=True)
class SectorBasket:
    """Full current pool after active signal/sizing eligibility, before selection."""

    eligible_members: Mapping[str, tuple[str, ...]]
    selected: SelectedSectors

    def __post_init__(self) -> None:
        _sectors(self.eligible_members)
        copied = {code: tuple(assets) for code, assets in self.eligible_members.items()}
        flat = [asset for assets in copied.values() for asset in assets]
        if any(not isinstance(a, str) or not a for a in flat) or len(flat) != len(
            set(flat)
        ):
            raise ValueError("Basket assets must be nonempty unique Asset_IDs")
        if any(tuple(sorted(assets)) != assets for assets in copied.values()):
            raise ValueError("Basket constituents must be sorted")
        if not isinstance(self.selected, SelectedSectors):
            raise ValueError("Basket requires selected-sector state")
        if any(
            not copied.get(code)
            for code in (*self.selected.longs, *self.selected.shorts)
        ):
            raise ValueError("Selected sector has no eligible basket")
        object.__setattr__(self, "eligible_members", MappingProxyType(copied))

    def validate_holdings(self, longs, shorts, *, eligible=None) -> None:
        for codes, holdings in (
            (self.selected.longs, longs),
            (self.selected.shorts, shorts),
        ):
            expected = {
                asset
                for code in codes
                for asset in self.eligible_members[code]
                if eligible is None or asset in eligible
            }
            if set(holdings) != expected:
                raise ValueError(
                    "Whole-sector decision must hold every eligible basket constituent"
                )
            if any(
                not any(
                    eligible is None or a in eligible for a in self.eligible_members[c]
                )
                for c in codes
            ):
                raise ValueError("Selected sector is empty at execution")
