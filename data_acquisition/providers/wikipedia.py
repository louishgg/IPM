"""One-attempt Wikipedia request boundaries for sector snapshots."""

from __future__ import annotations

import io
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import re
from urllib.parse import urlencode

import pandas as pd
import requests

from ..provider_clients import ClientInfo
from ..errors import ConfirmedNoData, ProviderRateLimited, RetryableProviderError
from ..contracts import AcquisitionRequest
from .base import AcquisitionResult, ProviderAdapter
from portfolio_core.sector_evidence import (
    SECTOR_ACQUISITION_DATASET,
    SNAPSHOT_COLUMNS,
    WIKIPEDIA_API_URL,
    WIKIPEDIA_PAGE_URL,
    clean_wikipedia_ticker,
)


WIKIMEDIA_USER_AGENT = (
    "ipm-sector-history/1.0 "
    "(https://github.com/louishgg; educational portfolio project)"
)


def _checked_response(response, *, resource: str):
    status = int(response.status_code)
    if status == 404:
        raise ConfirmedNoData(f"Wikipedia target does not exist: {resource}", http_status=404)
    if status == 429:
        raise ProviderRateLimited("Wikipedia returned HTTP 429", http_status=429)
    if status >= 400:
        raise RetryableProviderError(
            f"Wikipedia returned HTTP {status}: {resource}", http_status=status
        )
    return response


def _pages_from_payload(payload: Mapping) -> list[Mapping]:
    try:
        pages = payload["query"]["pages"]
    except (KeyError, TypeError) as exc:
        raise RetryableProviderError("Wikipedia response has no query pages") from exc
    if isinstance(pages, Mapping):
        return list(pages.values())
    if isinstance(pages, list):
        return pages
    raise RetryableProviderError("Wikipedia query pages has an invalid shape")


def _revision_from_payload(payload: Mapping) -> tuple[int, str]:
    try:
        pages = _pages_from_payload(payload)
        revision = pages[0]["revisions"][0]
        revision_id = int(revision["revid"])
        timestamp = str(revision["timestamp"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RetryableProviderError(
            "Wikipedia returned no usable revision metadata"
        ) from exc
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RetryableProviderError(
            f"Wikipedia returned an invalid revision timestamp: {timestamp!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise RetryableProviderError("Wikipedia revision timestamp is not UTC")
    timestamp = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return revision_id, timestamp


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        flattened = []
        for column in result.columns:
            values = [
                str(value).strip()
                for value in column
                if str(value).strip() and not str(value).startswith("Unnamed:")
            ]
            flattened.append(values[-1] if values else str(column[-1]).strip())
        result.columns = flattened
    else:
        result.columns = [str(column).strip() for column in result.columns]
    return result


def _column_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _source_column(
    table: pd.DataFrame,
    columns_by_key: Mapping[str, tuple[int, ...]],
    candidates: tuple[str, ...],
    *,
    field_name: str,
    revision_id: int,
) -> int | None:
    """Select one source column deterministically and reject disagreements."""
    matches = [
        column
        for candidate in candidates
        for column in columns_by_key.get(candidate, ())
    ]
    if not matches:
        return None
    selected = matches[0]
    selected_values = (
        table.iloc[:, selected]
        .fillna("")
        .astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )
    for alternate in matches[1:]:
        alternate_values = (
            table.iloc[:, alternate]
            .fillna("")
            .astype(str)
            .str.replace("\xa0", " ", regex=False)
            .str.strip()
        )
        if not selected_values.equals(alternate_values):
            raise RetryableProviderError(
                f"Wikipedia revision {revision_id} has conflicting {field_name} "
                f"columns: {[str(table.columns[index]) for index in matches]}"
            )
    return selected


def parse_wikipedia_sector_tables(
    tables: list[pd.DataFrame],
    *,
    requirement_date: str,
    cutoff_utc: str,
    revision_id: int,
    revision_timestamp_utc: str,
    source_url: str,
) -> pd.DataFrame:
    """Select a historical constituent layout and preserve source punctuation."""
    ticker_candidates = ("symbol", "ticker", "ticker symbol")
    company_candidates = ("security", "company", "company name")
    sector_candidates = ("gics sector", "sector")
    selected: tuple[pd.DataFrame, int, int, int] | None = None
    for raw in tables:
        table = _flatten_columns(raw)
        columns_by_key: dict[str, list[int]] = {}
        for index, column in enumerate(table.columns):
            columns_by_key.setdefault(_column_key(column), []).append(index)
        normalized_columns = {
            key: tuple(columns) for key, columns in columns_by_key.items()
        }
        ticker = _source_column(
            table,
            normalized_columns,
            ticker_candidates,
            field_name="ticker",
            revision_id=revision_id,
        )
        company = _source_column(
            table,
            normalized_columns,
            company_candidates,
            field_name="company",
            revision_id=revision_id,
        )
        sector = _source_column(
            table,
            normalized_columns,
            sector_candidates,
            field_name="sector",
            revision_id=revision_id,
        )
        if ticker is not None and company is not None and sector is not None:
            selected = (table, ticker, company, sector)
            break
    if selected is None:
        raise RetryableProviderError(
            f"Wikipedia revision {revision_id} has no constituent table with "
            "ticker, company, and sector columns"
        )
    table, ticker_column, company_column, sector_column = selected
    rows = table.iloc[:, [ticker_column, company_column, sector_column]].copy()
    rows.columns = ["Wikipedia_Ticker", "Company_Name", "Raw_Sector"]
    rows = rows.dropna(subset=["Wikipedia_Ticker", "Company_Name", "Raw_Sector"])
    rows["Wikipedia_Ticker"] = rows["Wikipedia_Ticker"].map(
        clean_wikipedia_ticker
    )
    for column in ("Company_Name", "Raw_Sector"):
        rows[column] = (
            rows[column].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
        )
    rows = rows.loc[
        rows[["Wikipedia_Ticker", "Company_Name", "Raw_Sector"]]
        .ne("")
        .all(axis=1)
    ]
    duplicates = rows["Wikipedia_Ticker"].duplicated(keep=False)
    if duplicates.any():
        values = sorted(rows.loc[duplicates, "Wikipedia_Ticker"].unique())
        raise RetryableProviderError(
            f"Wikipedia revision {revision_id} has duplicate tickers: {values}"
        )
    if rows.empty:
        raise RetryableProviderError(
            f"Wikipedia revision {revision_id} has no usable sector rows"
        )
    rows.insert(0, "Requirement_Date", requirement_date)
    rows.insert(1, "Cutoff_UTC", cutoff_utc)
    rows.insert(2, "Revision_ID", str(revision_id))
    rows.insert(3, "Revision_Timestamp_UTC", revision_timestamp_utc)
    rows["Source_URL"] = source_url
    return rows.loc[:, SNAPSHOT_COLUMNS].sort_values(
        "Wikipedia_Ticker", kind="stable"
    ).reset_index(drop=True)


def fetch_wikipedia_sector_snapshot_once(
    request: AcquisitionRequest,
    *,
    session: requests.Session | None = None,
    timeout_seconds: float = 30.0,
    pinned_revision_id: int | None = None,
) -> AcquisitionResult[pd.DataFrame]:
    """Resolve or verify one causal revision, fetch its HTML, and parse sectors."""

    page_title = request.identity.provider_symbol
    snapshot_date = request.requested_end or request.requested_start
    if not snapshot_date:
        raise ValueError("historical Wikipedia acquisition requires a snapshot date")
    try:
        cutoff = datetime.strptime(snapshot_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError("historical Wikipedia snapshot date must use YYYY-MM-DD") from exc
    cutoff_utc = cutoff.isoformat().replace("+00:00", "Z")
    # MediaWiki start timestamps are inclusive. Query one second earlier so a
    # revision exactly at midnight can never enter the requirement date.
    selection_target = (cutoff - timedelta(seconds=1)).isoformat().replace(
        "+00:00", "Z"
    )
    owned_session = session is None
    session = session or requests.Session()
    if hasattr(session, "headers"):
        session.headers["User-Agent"] = WIKIMEDIA_USER_AGENT
    try:
        if pinned_revision_id is None:
            api_params = {
                "action": "query",
                "prop": "revisions",
                "titles": page_title,
                "rvprop": "ids|timestamp",
                "rvlimit": 1,
                "rvstart": selection_target,
                "rvdir": "older",
                "formatversion": 2,
                "format": "json",
            }
            resource = f"revision strictly before {cutoff_utc}"
        else:
            api_params = {
                "action": "query",
                "prop": "revisions",
                "revids": int(pinned_revision_id),
                "rvprop": "ids|timestamp",
                "formatversion": 2,
                "format": "json",
            }
            resource = f"pinned revision {int(pinned_revision_id)}"
        api_response = _checked_response(
            session.get(
                WIKIPEDIA_API_URL,
                params=api_params,
                timeout=timeout_seconds,
            ),
            resource=resource,
        )
        try:
            revision_id, revision_timestamp_utc = _revision_from_payload(
                api_response.json()
            )
        except (TypeError, ValueError) as exc:
            raise RetryableProviderError("Wikipedia returned invalid JSON") from exc
        if pinned_revision_id is not None and revision_id != int(pinned_revision_id):
            raise RetryableProviderError(
                f"Wikipedia returned revision {revision_id} while verifying "
                f"pinned revision {pinned_revision_id}"
            )
        revision_timestamp = datetime.fromisoformat(
            revision_timestamp_utc.replace("Z", "+00:00")
        )
        if revision_timestamp >= cutoff:
            raise RetryableProviderError(
                f"Wikipedia revision {revision_id} at {revision_timestamp_utc} "
                f"is not strictly before {cutoff_utc}"
            )

        source_query = urlencode(
            {
                "title": page_title.replace(" ", "_"),
                "oldid": revision_id,
            }
        )
        source_url = f"{WIKIPEDIA_PAGE_URL}?{source_query}"
        html_response = _checked_response(
            session.get(
                WIKIPEDIA_PAGE_URL,
                params={
                    "title": page_title.replace(" ", "_"),
                    "oldid": revision_id,
                },
                timeout=timeout_seconds,
            ),
            resource=f"revision {revision_id}",
        )
        try:
            tables = pd.read_html(io.StringIO(str(html_response.text)))
        except Exception as exc:
            raise RetryableProviderError(
                f"Wikipedia revision {revision_id} could not be parsed"
            ) from exc
        if not tables:
            raise RetryableProviderError(
                f"Wikipedia revision {revision_id} contains no tables"
            )
        rows = parse_wikipedia_sector_tables(
            tables,
            requirement_date=snapshot_date,
            cutoff_utc=cutoff_utc,
            revision_id=revision_id,
            revision_timestamp_utc=revision_timestamp_utc,
            source_url=source_url,
        )
        return AcquisitionResult(
            payload=rows,
            observation_count=len(rows),
            observation_start=snapshot_date,
            observation_end=snapshot_date,
            http_status=200,
        )
    finally:
        if owned_session:
            session.close()


def make_wikipedia_sectors_adapter(
    client: ClientInfo,
    *,
    pinned_revisions: Mapping[str, int] | None = None,
) -> ProviderAdapter[pd.DataFrame]:
    pins = dict(pinned_revisions or {})
    return ProviderAdapter(
        lambda request: fetch_wikipedia_sector_snapshot_once(
            request,
            pinned_revision_id=pins.get(
                request.requested_end or request.requested_start
            ),
        ),
        provider="wikipedia",
        dataset=SECTOR_ACQUISITION_DATASET,
        client_name=client.name,
        client_version=client.version,
    )


__all__ = [
    "fetch_wikipedia_sector_snapshot_once",
    "make_wikipedia_sectors_adapter",
    "parse_wikipedia_sector_tables",
]
