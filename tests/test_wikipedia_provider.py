"""Offline tests for the historical Wikipedia sector provider."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

import data_acquisition.providers.wikipedia as wikipedia_provider
from data_acquisition.errors import RetryableProviderError
from data_acquisition.provider_clients import ClientInfo
from data_acquisition.providers.wikipedia import (
    fetch_wikipedia_sector_snapshot_once,
    make_wikipedia_sectors_adapter,
    parse_wikipedia_sector_tables,
)
from _acquisition_test_helpers import (
    _Response,
    _Session,
    backtest_acquisition_request as _request,
)


def test_historical_wikipedia_adapter_uses_request_identity_for_page_title(
    monkeypatch,
):
    api = {
        "query": {
            "pages": {
                "1": {
                    "revisions": [
                        {"revid": 123, "timestamp": "2023-12-31T23:45:00Z"}
                    ]
                }
            }
        }
    }
    html = """
    <table>
      <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
      <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td></tr>
      <tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td></tr>
    </table>
    """
    session = _Session(
        [_Response(200, payload=api), _Response(200, text=html)]
    )
    page_title = "S&P 500 test / archive"
    request = _request(
        dataset="sectors",
        provider="wikipedia",
        provider_symbol=page_title,
        start="2024-01-01",
        end="2024-01-01",
    )
    monkeypatch.setattr(wikipedia_provider.requests, "Session", lambda: session)
    adapter = make_wikipedia_sectors_adapter(
        ClientInfo(name="requests", version="fixed")
    )
    result = adapter.fetch(request)
    assert result.payload["Revision_ID"].unique().tolist() == ["123"]
    assert result.payload["Revision_Timestamp_UTC"].unique().tolist() == [
        "2023-12-31T23:45:00Z"
    ]
    assert result.payload["Wikipedia_Ticker"].tolist() == ["AAPL", "BRK.B"]
    assert result.payload["Requirement_Date"].unique().tolist() == [
        "2024-01-01"
    ]
    expected_source_url = (
        "https://en.wikipedia.org/w/index.php?"
        "title=S%26P_500_test_%2F_archive&oldid=123"
    )
    assert result.payload["Source_URL"].unique().tolist() == [
        expected_source_url
    ]
    assert (
        session.calls[0][1]["params"]["rvstart"]
        == "2023-12-31T23:59:59Z"
    )
    assert session.calls[0][1]["params"]["rvprop"] == "ids|timestamp"
    assert session.calls[0][1]["params"]["titles"] == page_title
    assert session.calls[1][1]["params"]["title"] == "S&P_500_test_/_archive"
    assert session.calls[1][1]["params"]["oldid"] == 123
    assert request.identity.provider_symbol == page_title
    assert session.closed

def test_wikipedia_parser_uses_explicit_synonym_precedence():
    table = pd.DataFrame({
        "Symbol": ["AAPL"],
        "Ticker": ["AAPL"],
        "Ticker symbol": ["AAPL"],
        "Security": ["Apple Inc."],
        "Company": ["Apple Inc."],
        "Company name": ["Apple Inc."],
        "GICS Sector": ["Information Technology"],
        "Sector": ["Information Technology"],
    })

    parsed = parse_wikipedia_sector_tables(
        [table],
        requirement_date="2024-01-01",
        cutoff_utc="2024-01-01T00:00:00Z",
        revision_id=123,
        revision_timestamp_utc="2023-12-31T23:45:00Z",
        source_url=(
            "https://en.wikipedia.org/w/index.php?"
            "title=List_of_S%26P_500_companies&oldid=123"
        ),
    )

    assert parsed.loc[0, "Wikipedia_Ticker"] == "AAPL"
    assert parsed.loc[0, "Company_Name"] == "Apple Inc."
    assert parsed.loc[0, "Raw_Sector"] == "Information Technology"

def test_wikipedia_parser_rejects_conflicting_synonyms():
    table = pd.DataFrame({
        "Symbol": ["AAPL"],
        "Ticker": ["MSFT"],
        "Security": ["Apple Inc."],
        "Company": ["Apple Inc."],
        "GICS Sector": ["Information Technology"],
        "Sector": ["Information Technology"],
    })

    with pytest.raises(RetryableProviderError, match="conflicting ticker columns"):
        parse_wikipedia_sector_tables(
            [table],
            requirement_date="2024-01-01",
            cutoff_utc="2024-01-01T00:00:00Z",
            revision_id=123,
            revision_timestamp_utc="2023-12-31T23:45:00Z",
            source_url=(
                "https://en.wikipedia.org/w/index.php?"
                "title=List_of_S%26P_500_companies&oldid=123"
            ),
        )

def test_wikipedia_parser_is_independent_of_python_hash_seed():
    script = """
import pandas as pd
from data_acquisition.providers.wikipedia import parse_wikipedia_sector_tables

table = pd.DataFrame({
    'Ticker symbol': ['AAPL'],
    'Ticker': ['AAPL'],
    'Symbol': ['AAPL'],
    'Company name': ['Apple Inc.'],
    'Company': ['Apple Inc.'],
    'Security': ['Apple Inc.'],
    'Sector': ['Information Technology'],
    'GICS Sector': ['Information Technology'],
})
result = parse_wikipedia_sector_tables(
    [table],
    requirement_date='2024-01-01',
    cutoff_utc='2024-01-01T00:00:00Z',
    revision_id=123,
    revision_timestamp_utc='2023-12-31T23:45:00Z',
    source_url='https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid=123',
)
print(result.to_json(orient='records'))
"""
    outputs = []
    for seed in ("1", "999"):
        environment = dict(os.environ, PYTHONHASHSEED=seed)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout)

    assert outputs[0] == outputs[1]

def test_historical_wikipedia_refresh_verifies_exact_pinned_revision():
    api = {
        "query": {
            "pages": [
                {
                    "revisions": [
                        {"revid": 456, "timestamp": "2023-12-30T12:00:00Z"}
                    ]
                }
            ]
        }
    }
    html = """
    <table>
      <tr><th>Ticker symbol</th><th>Company</th><th>Sector</th></tr>
      <tr><td>BF.B</td><td>Brown-Forman</td><td>Consumer Staples</td></tr>
    </table>
    """
    session = _Session([_Response(200, payload=api), _Response(200, text=html)])
    page_title = "S&P 500 pinned / archive"
    result = fetch_wikipedia_sector_snapshot_once(
        _request(
            dataset="sectors",
            provider="wikipedia",
            provider_symbol=page_title,
            start="2024-01-01",
            end="2024-01-01",
        ),
        session=session,
        pinned_revision_id=456,
    )
    assert session.calls[0][1]["params"]["revids"] == 456
    assert "rvstart" not in session.calls[0][1]["params"]
    assert session.calls[1][1]["params"]["title"] == "S&P_500_pinned_/_archive"
    assert result.payload["Source_URL"].unique().tolist() == [
        "https://en.wikipedia.org/w/index.php?"
        "title=S%26P_500_pinned_%2F_archive&oldid=456"
    ]
    assert result.payload["Wikipedia_Ticker"].tolist() == ["BF.B"]

def test_historical_wikipedia_rejects_revision_at_cutoff():
    api = {
        "query": {
            "pages": [
                {
                    "revisions": [
                        {"revid": 789, "timestamp": "2024-01-01T00:00:00Z"}
                    ]
                }
            ]
        }
    }
    session = _Session([_Response(200, payload=api)])
    with pytest.raises(RetryableProviderError, match="not strictly before"):
        fetch_wikipedia_sector_snapshot_once(
            _request(
                dataset="sectors",
                provider="wikipedia",
                provider_symbol="S&P 500 cutoff / archive",
                start="2024-01-01",
                end="2024-01-01",
            ),
            session=session,
        )
