"""Canonical Yahoo symbol normalization for backtest consumers."""

from pathlib import Path

import pytest

from portfolio_core.provider_identity import YahooIdentityResolver
from portfolio_core.security_identity import load_security_identity_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_DIR = PROJECT_ROOT / "data/shared/provenance/security_identity"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("BRKb", "BRK-B"),
        ("BFb", "BF-B"),
        ("RDSb", "RDS-B"),
        ("BRKb^legacy", "BRK-B"),
        ("META.O", "META-O"),
    ],
)
def test_yahoo_symbols_use_manifested_aliases(source, expected):
    resolver = YahooIdentityResolver(
        load_security_identity_bundle(IDENTITY_DIR),
        scope="backtest",
    )
    assert resolver.resolve(
        source,
        purpose="historical_prices",
    ).provider_symbol == expected


def test_blank_ticker_is_rejected():
    resolver = YahooIdentityResolver(
        load_security_identity_bundle(IDENTITY_DIR),
        scope="backtest",
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        resolver.resolve("  ", purpose="historical_prices")
