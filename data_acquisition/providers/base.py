"""Provider adapter contracts; parsing remains provider-specific."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Generic, TypeVar

from ..contracts import AcquisitionRequest


PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True, slots=True)
class AcquisitionResult(Generic[PayloadT]):
    """A validated provider payload and its observation metadata."""

    payload: PayloadT
    observation_count: int
    observation_start: str
    observation_end: str
    http_status: int | None = None

    def __post_init__(self) -> None:
        if self.observation_count <= 0:
            raise ValueError("a successful acquisition result requires observations")
        if not str(self.observation_start).strip() or not str(self.observation_end).strip():
            raise ValueError("a successful acquisition result requires observation bounds")
        try:
            start = date.fromisoformat(str(self.observation_start))
            end = date.fromisoformat(str(self.observation_end))
        except ValueError as exc:
            raise ValueError("observation bounds must use YYYY-MM-DD") from exc
        if start > end:
            raise ValueError("observation_start must not be after observation_end")
        if self.http_status is not None and not 100 <= int(self.http_status) <= 599:
            raise ValueError("http_status must be between 100 and 599")


class ProviderAdapter(Generic[PayloadT]):
    """Validated wrapper around a source-specific fetch/parse function."""

    def __init__(
        self,
        fetcher: Callable[[AcquisitionRequest], AcquisitionResult[PayloadT]],
        *,
        provider: str,
        dataset: str,
        client_name: str,
        client_version: str,
    ) -> None:
        if not callable(fetcher):
            raise TypeError("fetcher must be callable")
        self._fetcher = fetcher
        self.provider = str(provider).strip()
        self.dataset = str(dataset).strip()
        self.client_name = str(client_name).strip()
        self.client_version = str(client_version).strip()
        if not self.provider:
            raise ValueError("provider adapters require a provider name")
        if not self.dataset:
            raise ValueError("provider adapters require a dataset name")
        if not self.client_name or not self.client_version:
            raise ValueError("provider adapters require a client name and version")

    def fetch(self, request: AcquisitionRequest) -> AcquisitionResult[PayloadT]:
        identity = request.identity
        if identity.provider.casefold() != self.provider.casefold():
            raise ValueError(
                f"{type(self).__name__} cannot serve provider "
                f"{identity.provider!r}; expected {self.provider!r}"
            )
        if identity.dataset != self.dataset:
            raise ValueError(
                f"{type(self).__name__} cannot serve dataset "
                f"{identity.dataset!r}; expected {self.dataset!r}"
            )
        result = self._fetcher(request)
        if not isinstance(result, AcquisitionResult):
            raise TypeError("provider fetcher must return AcquisitionResult")
        return result


__all__ = ["AcquisitionResult", "PayloadT", "ProviderAdapter"]
