"""Shared acquisition exceptions with provider semantics."""

from __future__ import annotations


class AcquisitionError(RuntimeError):
    """Base class for expected acquisition failures."""


class ConfirmedNoData(AcquisitionError):
    """The target resource was reached and explicitly has no usable data."""

    def __init__(self, message: str, *, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


class ProviderRateLimited(AcquisitionError):
    """The provider explicitly throttled the current acquisition process."""

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = 429,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status


class RetryableProviderError(AcquisitionError):
    """A transport/authentication/provider error that does not prove no-data."""

    def __init__(self, message: str, *, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


class AcquisitionTimeout(AcquisitionError):
    """One item exceeded its whole-item retry budget."""


class AcquisitionReviewRequired(AcquisitionError):
    """A candidate payload needs a human decision before replacing cached data."""


__all__ = [
    "AcquisitionError",
    "AcquisitionReviewRequired",
    "AcquisitionTimeout",
    "ConfirmedNoData",
    "ProviderRateLimited",
    "RetryableProviderError",
]
