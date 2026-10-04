"""Explicit prepared price and ordinary-dividend treatment metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


PRICE_BASIS_COLUMNS = (
    "Domain",
    "Price_Basis_ID",
    "Price_Source",
    "Price_Treatment",
    "Ordinary_Dividend_Treatment",
)
BACKTEST_PRICE_BASIS_ID = "reuters_yahoo_wiki_price_return_v1"
LIVE_PRICE_BASIS_ID = "yahoo_adjusted_with_reviewed_monthly_reconstruction_v2"

@dataclass(frozen=True, slots=True)
class PriceBasisSpec:
    """Validated price and ordinary-dividend treatment for one data domain."""

    domain: str
    price_basis_id: str
    price_source: str
    price_treatment: str
    ordinary_dividend_treatment: str

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            normalized = str(value).strip()
            if not normalized:
                raise ValueError(f"{field_name} cannot be empty")
            object.__setattr__(self, field_name, normalized)

    def as_prepared_record(self) -> dict[str, str]:
        return {
            "Domain": self.domain,
            "Price_Basis_ID": self.price_basis_id,
            "Price_Source": self.price_source,
            "Price_Treatment": self.price_treatment,
            "Ordinary_Dividend_Treatment": (
                self.ordinary_dividend_treatment
            ),
        }

    def as_assumptions(self) -> dict[str, str]:
        return asdict(self)


_SPECS = {
    "backtest": PriceBasisSpec(
        domain="backtest",
        price_basis_id=BACKTEST_PRICE_BASIS_ID,
        price_source=(
            "Reuters supplied Price Close, Yahoo Close with auto_adjust=False, "
            "and hash-pinned WIKI close"
        ),
        price_treatment=(
            "effective-dated Reuters-first deterministic monthly merge; "
            "unadjusted price-return series with reviewed corporate actions"
        ),
        ordinary_dividend_treatment="omitted",
    ),
    "live": PriceBasisSpec(
        domain="live",
        price_basis_id=LIVE_PRICE_BASIS_ID,
        price_source="Yahoo Finance adjusted OHLC; reviewed Reuters monthly reconstruction",
        price_treatment=(
            "daily execution remains Yahoo adjusted; monthly history composes adjusted "
            "Yahoo closes with reviewed reconstructions and explicit event valuations"
        ),
        ordinary_dividend_treatment="embedded once in adjusted closes or event-return factors",
    ),
}


def price_basis_spec(domain: str) -> PriceBasisSpec:
    """Return the canonical typed specification for one domain."""

    normalized = str(domain).strip().lower()
    if normalized not in _SPECS:
        raise ValueError(f"Unsupported price-basis domain: {domain!r}")
    return _SPECS[normalized]


def price_basis_frame(domain: str) -> pd.DataFrame:
    """Return the canonical one-row prepared artifact for one domain."""

    spec = price_basis_spec(domain)
    return pd.DataFrame([spec.as_prepared_record()], columns=PRICE_BASIS_COLUMNS)


def validate_price_basis(frame: pd.DataFrame, domain: str) -> PriceBasisSpec:
    """Validate exact metadata and return the canonical typed specification."""

    spec = price_basis_spec(domain)
    expected = price_basis_frame(spec.domain)
    actual = frame.copy().fillna("")
    if list(actual.columns) != list(PRICE_BASIS_COLUMNS) or len(actual) != 1:
        raise ValueError(
            "Prepared price basis must contain exactly one canonical metadata row"
        )
    actual = actual.reset_index(drop=True).astype(str)
    if not actual.equals(expected.astype(str)):
        raise ValueError(
            f"Prepared {domain} price-basis metadata is inconsistent"
        )
    return spec


__all__ = [
    "BACKTEST_PRICE_BASIS_ID",
    "LIVE_PRICE_BASIS_ID",
    "PRICE_BASIS_COLUMNS",
    "PriceBasisSpec",
    "price_basis_frame",
    "price_basis_spec",
    "validate_price_basis",
]
