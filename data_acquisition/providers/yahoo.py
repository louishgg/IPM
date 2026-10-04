"""One-attempt Yahoo acquisition and validated source-specific parsing.

Retries belong to :class:`data_acquisition.engine.SerialAcquisitionEngine`.  These
functions deliberately do not infer ``no_data`` from ambiguous yfinance text
such as "possibly delisted" or from an empty response.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..provider_clients import ClientInfo
from ..errors import (
    AcquisitionError,
    ConfirmedNoData,
    ProviderRateLimited,
    RetryableProviderError,
)
from ..contracts import AcquisitionRequest
from .base import AcquisitionResult, ProviderAdapter


def _translate_yahoo_exception(exc: BaseException) -> AcquisitionError:
    """Classify explicit Yahoo throttles; keep all other failures retryable."""

    if isinstance(exc, AcquisitionError):
        return exc
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if value is None:
        value = getattr(exc, "status_code", None)
    try:
        status = int(value) if value is not None else None
    except (TypeError, ValueError):
        status = None
    if type(exc).__name__ == "YFRateLimitError" or status == 429:
        return ProviderRateLimited(str(exc) or repr(exc), http_status=status or 429)
    return RetryableProviderError(str(exc) or repr(exc), http_status=status)


def _timezone_free_index(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, errors="raise", utc=True)).tz_convert(None)


def _exclusive_yahoo_end(inclusive_end: str) -> str:
    """Translate the canonical inclusive request end to Yahoo's exclusive end."""

    if not inclusive_end:
        return ""
    return (date.fromisoformat(inclusive_end) + timedelta(days=1)).isoformat()


def _flatten_yahoo_columns(frame: pd.DataFrame, fields: tuple[str, ...]) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    for level in range(frame.columns.nlevels):
        values = frame.columns.get_level_values(level)
        if set(fields).issubset(set(values)):
            result = frame.copy()
            result.columns = values
            return result
    raise ValueError("Yahoo response has no recognizable price-field column level")


def _normalize_yahoo_prices(
    downloaded: pd.DataFrame | None,
    *,
    request: AcquisitionRequest,
    required_fields: tuple[str, ...],
    optional_zero_fields: tuple[str, ...] = (),
) -> pd.DataFrame:
    provider_symbol = request.identity.provider_symbol
    if downloaded is None or downloaded.empty:
        raise RetryableProviderError(
            f"Yahoo returned an empty response for {provider_symbol}; absence "
            "was not confirmed"
        )
    frame = _flatten_yahoo_columns(
        pd.DataFrame(downloaded).copy(),
        required_fields,
    )
    missing = set(required_fields) - set(frame.columns)
    if missing:
        raise RetryableProviderError(
            f"Yahoo response for {provider_symbol} is missing fields: "
            f"{sorted(missing)}"
        )
    output_fields = required_fields + optional_zero_fields
    result = pd.DataFrame(index=_timezone_free_index(frame.index))
    for field in output_fields:
        values = (
            frame[field].to_numpy()
            if field in frame
            else np.zeros(len(frame), dtype=float)
        )
        result[field] = pd.to_numeric(values, errors="coerce")
    result = result.loc[
        ~result.index.duplicated(keep="last")
    ].sort_index()
    if request.requested_start:
        result = result.loc[
            result.index >= pd.Timestamp(request.requested_start)
        ]
    if request.requested_end:
        result = result.loc[
            result.index <= pd.Timestamp(request.requested_end)
        ]
    values = result.to_numpy(dtype=float)
    if result.empty or not np.isfinite(values).all():
        raise RetryableProviderError(
            f"Yahoo response for {provider_symbol} has no finite complete rows"
        )
    price_fields = [
        field for field in required_fields if field != "Volume"
    ]
    if price_fields and (result[price_fields] <= 0.0).to_numpy().any():
        raise RetryableProviderError("Yahoo prices must be positive")
    nonnegative_fields = [
        field for field in output_fields if field not in price_fields
    ]
    if (
        nonnegative_fields
        and (result[nonnegative_fields] < 0.0).to_numpy().any()
    ):
        raise RetryableProviderError(
            "Yahoo volume and transformation metadata cannot be negative"
        )
    result.index.name = "Date"
    return result.astype(float)


def _fetch_yahoo_prices_once(
    request: AcquisitionRequest,
    *,
    required_fields: tuple[str, ...],
    optional_zero_fields: tuple[str, ...] = (),
    auto_adjust: bool = True,
    actions: bool = False,
    downloader: Callable[..., pd.DataFrame | None] | None = None,
    timeout_seconds: float = 30.0,
) -> AcquisitionResult[pd.DataFrame]:
    if downloader is None:
        import yfinance as yf

        downloader = yf.download
    kwargs: dict[str, object] = {
        "tickers": request.identity.provider_symbol,
        "auto_adjust": auto_adjust,
        "progress": False,
        "threads": False,
        "timeout": timeout_seconds,
    }
    if actions:
        kwargs["actions"] = True
    if request.requested_start:
        kwargs["start"] = request.requested_start
    if request.requested_end:
        kwargs["end"] = _exclusive_yahoo_end(request.requested_end)
    try:
        downloaded = downloader(**kwargs)
        frame = _normalize_yahoo_prices(
            downloaded,
            request=request,
            required_fields=required_fields,
            optional_zero_fields=optional_zero_fields,
        )
    except Exception as exc:
        translated = _translate_yahoo_exception(exc)
        raise translated from exc
    return AcquisitionResult(
        payload=frame,
        observation_count=len(frame),
        observation_start=frame.index.min().strftime("%Y-%m-%d"),
        observation_end=frame.index.max().strftime("%Y-%m-%d"),
        http_status=200,
    )


def fetch_yahoo_ohlcv_once(
    request: AcquisitionRequest,
    *,
    downloader: Callable[..., pd.DataFrame | None] | None = None,
    timeout_seconds: float = 30.0,
) -> AcquisitionResult[pd.DataFrame]:
    """Perform one yfinance OHLCV call and validate genuinely dated rows."""

    return _fetch_yahoo_prices_once(
        request,
        required_fields=("Open", "High", "Low", "Close", "Volume"),
        downloader=downloader,
        timeout_seconds=timeout_seconds,
    )


def fetch_yahoo_benchmark_once(
    request: AcquisitionRequest,
    *,
    downloader: Callable[..., pd.DataFrame | None] | None = None,
    timeout_seconds: float = 30.0,
) -> AcquisitionResult[pd.DataFrame]:
    """Perform one Yahoo benchmark OHLC call; no retry occurs here."""

    return _fetch_yahoo_prices_once(
        request,
        required_fields=("Open", "High", "Low", "Close"),
        downloader=downloader,
        timeout_seconds=timeout_seconds,
    )


def fetch_yahoo_unadjusted_close_once(
    request: AcquisitionRequest,
    *,
    downloader: Callable[..., pd.DataFrame | None] | None = None,
    timeout_seconds: float = 30.0,
) -> AcquisitionResult[pd.DataFrame]:
    """Fetch Yahoo ``Close`` plus transformation metadata without adjustment.

    Yahoo's end parameter is exclusive; the shared price normalizer enforces
    canonical inclusive request bounds locally.
    ``Adj Close`` is deliberately neither requested nor admitted.
    """
    return _fetch_yahoo_prices_once(
        request,
        required_fields=("Close", "Volume"),
        optional_zero_fields=(
            "Dividends",
            "Stock Splits",
            "Capital Gains",
        ),
        auto_adjust=False,
        actions=True,
        downloader=downloader,
        timeout_seconds=timeout_seconds,
    )


def _confirm_no_shares_with_price_history(
    client: object,
    request: AcquisitionRequest,
) -> None:
    """Confirm the symbol works over the interval while shares are absent."""

    kwargs: dict[str, object] = {
        "auto_adjust": False,
        "actions": False,
        "raise_errors": True,
    }
    if request.requested_start:
        kwargs["start"] = request.requested_start
    if request.requested_end:
        kwargs["end"] = _exclusive_yahoo_end(request.requested_end)
    history = client.history(**kwargs)
    if history is None or len(history) == 0:
        raise RetryableProviderError(
            f"Yahoo price-history probe was empty for "
            f"{request.identity.provider_symbol}; shares absence is unconfirmed"
        )
    frame = pd.DataFrame(history)
    close_values = frame.get("Close")
    if close_values is None:
        raise RetryableProviderError(
            f"Yahoo price-history probe had no valid Close for "
            f"{request.identity.provider_symbol}"
        )
    close = pd.to_numeric(close_values, errors="coerce")
    if close.dropna().empty:
        raise RetryableProviderError(
            f"Yahoo price-history probe had no valid Close for "
            f"{request.identity.provider_symbol}"
        )
    values = close.dropna().to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise RetryableProviderError("Yahoo price-history probe returned invalid prices")
def fetch_yahoo_shares_once(
    request: AcquisitionRequest,
    *,
    client_factory: Callable[[str], object] | None = None,
) -> AcquisitionResult[pd.Series]:
    """Perform one sparse-shares call.

    An empty shares result remains retryable unless price history confirms that
    the functioning Yahoo symbol has no shares series.
    """

    if client_factory is None:
        import yfinance as yf

        client_factory = yf.Ticker
    try:
        client = client_factory(request.identity.provider_symbol)
        shares = client.get_shares_full(
            start=request.requested_start or None,
            end=_exclusive_yahoo_end(request.requested_end) or None,
        )
    except Exception as exc:
        translated = _translate_yahoo_exception(exc)
        raise translated from exc
    if shares is None or len(shares) == 0:
        try:
            _confirm_no_shares_with_price_history(client, request)
        except Exception as exc:
            translated = _translate_yahoo_exception(exc)
            raise translated from exc
        raise ConfirmedNoData(
            f"Yahoo explicitly has no shares series for "
            f"{request.identity.provider_symbol}"
        )
    try:
        series = pd.Series(shares, dtype="float64").dropna()
        series.index = _timezone_free_index(series.index)
        series = series.sort_index(kind="stable")
    except Exception as exc:
        translated = _translate_yahoo_exception(exc)
        raise translated from exc
    normalized_dates = series.index.normalize()
    if request.requested_start:
        series = series.loc[
            normalized_dates >= pd.Timestamp(request.requested_start)
        ]
        normalized_dates = series.index.normalize()
    if request.requested_end:
        series = series.loc[
            normalized_dates <= pd.Timestamp(request.requested_end)
        ]
    if series.empty:
        try:
            _confirm_no_shares_with_price_history(client, request)
        except Exception as exc:
            translated = _translate_yahoo_exception(exc)
            raise translated from exc
        raise ConfirmedNoData(
            f"Yahoo explicitly has no shares series within the requested "
            f"window for {request.identity.provider_symbol}"
        )
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise RetryableProviderError("Yahoo shares must be finite and positive")
    series.name = request.identity.asset_id
    return AcquisitionResult(
        payload=series,
        observation_count=len(series),
        observation_start=series.index.min().strftime("%Y-%m-%d"),
        observation_end=series.index.max().strftime("%Y-%m-%d"),
        http_status=200,
    )


def make_yahoo_ohlcv_adapter(
    client: ClientInfo,
) -> ProviderAdapter[pd.DataFrame]:
    return ProviderAdapter(
        fetch_yahoo_ohlcv_once,
        provider="yahoo",
        dataset="prices",
        client_name=client.name,
        client_version=client.version,
    )


def make_yahoo_shares_adapter(
    client: ClientInfo,
) -> ProviderAdapter[pd.Series]:
    return ProviderAdapter(
        fetch_yahoo_shares_once,
        provider="yahoo",
        dataset="shares",
        client_name=client.name,
        client_version=client.version,
    )


def make_yahoo_benchmark_adapter(
    client: ClientInfo,
) -> ProviderAdapter[pd.DataFrame]:
    return ProviderAdapter(
        fetch_yahoo_benchmark_once,
        provider="yahoo",
        dataset="benchmark",
        client_name=client.name,
        client_version=client.version,
    )


def make_yahoo_unadjusted_close_adapter(
    client: ClientInfo,
) -> ProviderAdapter[pd.DataFrame]:
    return ProviderAdapter(
        fetch_yahoo_unadjusted_close_once,
        provider="yahoo",
        dataset="prices",
        client_name=client.name,
        client_version=client.version,
    )


__all__ = [
    "fetch_yahoo_benchmark_once",
    "fetch_yahoo_ohlcv_once",
    "fetch_yahoo_unadjusted_close_once",
    "fetch_yahoo_shares_once",
    "make_yahoo_benchmark_adapter",
    "make_yahoo_ohlcv_adapter",
    "make_yahoo_unadjusted_close_adapter",
    "make_yahoo_shares_adapter",
]
