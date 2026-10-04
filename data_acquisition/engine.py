"""One bounded, resumable serial acquisition engine for every provider."""

from __future__ import annotations

import random
import signal
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generic

from .errors import (
    AcquisitionReviewRequired as _AcquisitionReviewRequired,
    AcquisitionTimeout as _AcquisitionTimeout,
    ConfirmedNoData as _ConfirmedNoData,
    ProviderRateLimited as _ProviderRateLimited,
)
from .providers.base import (
    AcquisitionResult as _AcquisitionResult,
    PayloadT as _PayloadT,
    ProviderAdapter,
)
from .contracts import (
    AcquisitionRequest,
    AcquisitionStatus,
    ProviderStatus,
    utc_timestamp,
)


@dataclass(frozen=True, slots=True)
class AcquisitionPolicy:
    max_attempts: int = 3
    item_budget_seconds: float = 90.0
    initial_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 30.0
    backoff_multiplier: float = 2.0
    backoff_jitter_seconds: float = 1.0
    pacing_min_seconds: float = 0.0
    pacing_max_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.item_budget_seconds <= 0:
            raise ValueError("item_budget_seconds must be positive")
        if self.initial_backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("backoff durations must be nonnegative")
        if self.initial_backoff_seconds > self.max_backoff_seconds:
            raise ValueError("initial_backoff_seconds cannot exceed max_backoff_seconds")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least one")
        if self.backoff_jitter_seconds < 0:
            raise ValueError("backoff_jitter_seconds must be nonnegative")
        if self.pacing_min_seconds < 0 or self.pacing_max_seconds < self.pacing_min_seconds:
            raise ValueError("pacing bounds must be ordered and nonnegative")


PRODUCTION_ACQUISITION_POLICY = AcquisitionPolicy(
    pacing_min_seconds=1.0,
    pacing_max_seconds=3.0,
)


@dataclass(frozen=True, slots=True)
class AcquisitionOutcome(Generic[_PayloadT]):
    request: AcquisitionRequest
    result: _AcquisitionResult[_PayloadT] | None = None


@dataclass(frozen=True, slots=True)
class AcquisitionRun:
    statuses: tuple[AcquisitionStatus, ...]
    stopped_reason: str = ""
    remaining: tuple[AcquisitionRequest, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.remaining and not self.stopped_reason


@dataclass(frozen=True, slots=True)
class AcquisitionPlan:
    """Normalized statuses and the exact order a provider run will attempt."""

    requests: tuple[AcquisitionRequest, ...]
    statuses: tuple[AcquisitionStatus, ...]
    execution_order: tuple[AcquisitionRequest, ...]


def build_acquisition_plan(
    requests: Iterable[AcquisitionRequest],
    *,
    statuses: Iterable[AcquisitionStatus] = (),
    refresh: bool = False,
) -> AcquisitionPlan:
    """Return the status initialization and ordering shared by run and dry-run."""
    requested = tuple(sorted(requests, key=lambda item: item.identity.key))
    request_keys = [request.identity.key for request in requested]
    if len(request_keys) != len(set(request_keys)):
        raise ValueError("acquisition requests contain duplicate identities")

    status_by_key: dict[tuple[str, ...], AcquisitionStatus] = {}
    for status in statuses:
        key = status.identity.key
        if key in status_by_key:
            raise ValueError(f"duplicate acquisition status identity: {key!r}")
        status_by_key[key] = status
    for request in requested:
        key = request.identity.key
        existing = status_by_key.get(key)
        request_changed = existing is not None and (
            existing.requested_start != request.requested_start
            or existing.requested_end != request.requested_end
        )
        if existing is None or refresh or request_changed:
            note = existing.migration_note if existing is not None else ""
            status_by_key[key] = AcquisitionStatus.pending(
                request,
                migration_note=note,
            )

    retryable = (
        request
        for request in requested
        if status_by_key[request.identity.key].status
        in {ProviderStatus.PENDING, ProviderStatus.FAILED}
    )
    execution_order = tuple(
        sorted(
            retryable,
            key=lambda request: (
                status_by_key[request.identity.key].status
                is ProviderStatus.FAILED,
                request.identity.key,
            ),
        )
    )
    ordered_statuses = tuple(
        status_by_key[key] for key in sorted(status_by_key)
    )
    return AcquisitionPlan(
        requests=requested,
        statuses=ordered_statuses,
        execution_order=execution_order,
    )


Checkpoint = Callable[
    [tuple[AcquisitionStatus, ...], AcquisitionOutcome | None],
    None,
]


def hard_deadline_supported() -> bool:
    return (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
    )


@contextmanager
def item_deadline(seconds: float, label: str, *, enabled: bool = True):
    """Enforce a whole-item deadline on the supported Unix main thread."""

    if not enabled or not hard_deadline_supported():
        yield
        return

    def timeout_handler(signum, frame):
        del signum, frame
        raise _AcquisitionTimeout(
            f"acquisition exceeded the {seconds:.0f}s item budget for {label}"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        previous_delay, previous_interval = previous_timer
        if previous_delay > 0:
            elapsed = time.monotonic() - started
            if elapsed < previous_delay:
                restored = previous_delay - elapsed
            elif previous_interval > 0:
                restored = previous_interval - ((elapsed - previous_delay) % previous_interval)
            else:
                restored = 1e-6
            signal.setitimer(signal.ITIMER_REAL, restored, previous_interval)


class SerialAcquisitionEngine(Generic[_PayloadT]):
    """Run pending/failed requests serially and checkpoint after every item."""

    def __init__(
        self,
        adapter: ProviderAdapter[_PayloadT],
        *,
        policy: AcquisitionPolicy,
        reporter: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        random_value: Callable[[], float] = random.random,
        clock: Callable[[], datetime] | None = None,
        enforce_hard_deadline: bool = True,
    ) -> None:
        self.adapter = adapter
        self.policy = policy
        self.report = reporter or (lambda message: None)
        self.sleep = sleep
        self.monotonic = monotonic
        self.random_value = random_value
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.enforce_hard_deadline = enforce_hard_deadline

    def run(
        self,
        requests: Iterable[AcquisitionRequest],
        *,
        statuses: Iterable[AcquisitionStatus] = (),
        checkpoint: Checkpoint | None = None,
        refresh: bool = False,
    ) -> AcquisitionRun:
        """Acquire pending/failed requests; checkpoint callbacks own persistence.

        Callbacks receive initialization (outcome=None) and each item outcome.
        Ordinary provider failures continue, rate limits stop the run, and
        callback errors or interrupts propagate.
        """
        # Reuse terminal statuses unless refreshed or the requested dates changed.
        plan = build_acquisition_plan(
            requests,
            statuses=statuses,
            refresh=refresh,
        )
        requested = plan.requests
        status_by_key = {
            status.identity.key: status for status in plan.statuses
        }
        self._checkpoint(checkpoint, status_by_key, None)
        pending = plan.execution_order
        self.report(
            f"{self.adapter.provider} acquisition preflight: {len(requested)} "
            f"requested, {len(requested) - len(pending)} already terminal, "
            f"{len(pending)} pending/retryable."
        )

        stopped_reason = ""
        for position, request in enumerate(pending, start=1):
            identity = request.identity
            label = f"{identity.asset_id} ({identity.provider_symbol})"
            self.report(
                f"[{position}/{len(pending)}] Starting {label}; "
                f"{self.policy.item_budget_seconds:.0f}s whole-item budget."
            )
            rate_limited = False
            try:
                with item_deadline(
                    self.policy.item_budget_seconds,
                    label,
                    enabled=self.enforce_hard_deadline,
                ):
                    result = self._fetch_with_retries(request)
                status = self._ok_status(request, result)
                outcome = AcquisitionOutcome(request=request, result=result)
            except _ConfirmedNoData as exc:
                status = self._no_data_status(request, exc)
                outcome = AcquisitionOutcome(request=request)
            except _ProviderRateLimited as exc:
                rate_limited = True
                status = self._failed_status(request, exc)
                outcome = AcquisitionOutcome(request=request)
            except Exception as exc:
                status = self._failed_status(request, exc)
                outcome = AcquisitionOutcome(request=request)

            status_by_key[identity.key] = status
            self._checkpoint(checkpoint, status_by_key, outcome)

            if status.status is ProviderStatus.OK:
                self.report(
                    f"{identity.asset_id}: checkpointed ok with "
                    f"{status.observation_count} observations."
                )
            elif status.status is ProviderStatus.NO_DATA:
                self.report(f"{identity.asset_id}: checkpointed confirmed no_data.")
            else:
                self.report(
                    f"{identity.asset_id}: checkpointed failed: "
                    f"{status.error_class}: {status.error_message}"
                )

            more = position < len(pending)
            if rate_limited:
                stopped_reason = "rate_limited"
                if more:
                    self.report(
                        "Stopping immediately after an explicit provider rate limit; "
                        "remaining requests retain resumable state."
                    )
                    break

        final_statuses = self._ordered_statuses(status_by_key)
        remaining = tuple(
            request
            for request in requested
            if status_by_key[request.identity.key].status
            in {ProviderStatus.PENDING, ProviderStatus.FAILED}
        )
        summary: dict[str, int] = {}
        requested_keys = {request.identity.key for request in requested}
        for status in final_statuses:
            if status.identity.key in requested_keys:
                summary[status.status.value] = summary.get(status.status.value, 0) + 1
        self.report(f"{self.adapter.provider} acquisition finished with statuses: {summary}.")
        # Provider completion does not establish downstream data readiness.
        return AcquisitionRun(
            statuses=final_statuses,
            stopped_reason=stopped_reason,
            remaining=remaining,
        )

    def _fetch_with_retries(
        self,
        request: AcquisitionRequest,
    ) -> _AcquisitionResult[_PayloadT]:
        """Retry within one item budget shared by pacing, backoff and fetching.

        Ordinary errors use capped exponential backoff with jitter; no-data,
        rate-limit, timeout and review-required signals propagate without retry.
        """
        started = self.monotonic()
        delay = self.policy.initial_backoff_seconds
        identity = request.identity
        for attempt in range(1, self.policy.max_attempts + 1):
            elapsed = self.monotonic() - started
            if elapsed >= self.policy.item_budget_seconds:
                raise _AcquisitionTimeout(
                    f"item budget exhausted before attempt {attempt} for "
                    f"{identity.asset_id}"
                )
            pace = self._uniform(
                self.policy.pacing_min_seconds,
                self.policy.pacing_max_seconds,
            )
            if pace:
                if elapsed + pace >= self.policy.item_budget_seconds:
                    raise _AcquisitionTimeout(
                        f"item budget cannot accommodate pacing before attempt "
                        f"{attempt} for {identity.asset_id}"
                    )
                self.report(
                    f"{identity.asset_id}: attempt {attempt}/"
                    f"{self.policy.max_attempts} starts after {pace:.1f}s pacing."
                )
                self.sleep(pace)
            else:
                self.report(
                    f"{identity.asset_id}: attempt {attempt}/"
                    f"{self.policy.max_attempts}."
                )
            try:
                result = self.adapter.fetch(request)
                if self.monotonic() - started > self.policy.item_budget_seconds:
                    raise _AcquisitionTimeout(
                        f"item budget exhausted while fetching {identity.asset_id}"
                    )
                return result
            except (
                _ConfirmedNoData, _ProviderRateLimited, _AcquisitionTimeout,
                _AcquisitionReviewRequired,
            ):
                raise
            except Exception as exc:
                if attempt >= self.policy.max_attempts:
                    raise
                backoff = min(
                    delay + self.random_value() * self.policy.backoff_jitter_seconds,
                    self.policy.max_backoff_seconds,
                )
                elapsed = self.monotonic() - started
                remaining = self.policy.item_budget_seconds - elapsed
                if backoff >= remaining:
                    raise _AcquisitionTimeout(
                        f"item budget exhausted before retrying {identity.asset_id}"
                    ) from exc
                self.report(
                    f"{identity.asset_id}: {type(exc).__name__}; retrying after "
                    f"{backoff:.1f}s. {self._error_message(exc)}"
                )
                self.sleep(backoff)
                delay = min(
                    max(delay * self.policy.backoff_multiplier, 0.0),
                    self.policy.max_backoff_seconds,
                )
        raise RuntimeError(f"acquisition exhausted retries for {identity.asset_id}")

    def _ok_status(
        self,
        request: AcquisitionRequest,
        result: _AcquisitionResult,
    ) -> AcquisitionStatus:
        return AcquisitionStatus(
            identity=request.identity,
            status=ProviderStatus.OK,
            requested_start=request.requested_start,
            requested_end=request.requested_end,
            observation_count=result.observation_count,
            observation_start=result.observation_start,
            observation_end=result.observation_end,
            attempted_at_utc=utc_timestamp(self.clock()),
            client=self.adapter.client_name,
            client_version=self.adapter.client_version,
            http_status=str(result.http_status or ""),
        )

    def _no_data_status(
        self,
        request: AcquisitionRequest,
        exc: _ConfirmedNoData,
    ) -> AcquisitionStatus:
        return AcquisitionStatus(
            identity=request.identity,
            status=ProviderStatus.NO_DATA,
            requested_start=request.requested_start,
            requested_end=request.requested_end,
            attempted_at_utc=utc_timestamp(self.clock()),
            client=self.adapter.client_name,
            client_version=self.adapter.client_version,
            http_status=str(getattr(exc, "http_status", None) or ""),
            error_class=type(exc).__name__,
            error_message=self._error_message(exc),
        )

    def _failed_status(
        self,
        request: AcquisitionRequest,
        exc: BaseException,
    ) -> AcquisitionStatus:
        return AcquisitionStatus(
            identity=request.identity,
            status=ProviderStatus.FAILED,
            requested_start=request.requested_start,
            requested_end=request.requested_end,
            attempted_at_utc=utc_timestamp(self.clock()),
            client=self.adapter.client_name,
            client_version=self.adapter.client_version,
            http_status=str(getattr(exc, "http_status", None) or ""),
            error_class=type(exc).__name__,
            error_message=self._error_message(exc),
        )

    def _checkpoint(
        self,
        checkpoint: Checkpoint | None,
        statuses: dict[tuple[str, ...], AcquisitionStatus],
        outcome: AcquisitionOutcome | None,
    ) -> None:
        if checkpoint is not None:
            checkpoint(self._ordered_statuses(statuses), outcome)

    @staticmethod
    def _ordered_statuses(
        statuses: dict[tuple[str, ...], AcquisitionStatus],
    ) -> tuple[AcquisitionStatus, ...]:
        return tuple(statuses[key] for key in sorted(statuses))

    @staticmethod
    def _error_message(exc: BaseException) -> str:
        message = " ".join(str(exc).splitlines()).strip()
        if not message:
            message = repr(exc)
        return message[:500]

    def _uniform(self, lower: float, upper: float) -> float:
        if lower == upper:
            return lower
        return lower + self.random_value() * (upper - lower)


__all__ = [
    "AcquisitionPlan",
    "AcquisitionOutcome",
    "AcquisitionPolicy",
    "AcquisitionRun",
    "Checkpoint",
    "PRODUCTION_ACQUISITION_POLICY",
    "SerialAcquisitionEngine",
    "build_acquisition_plan",
    "hard_deadline_supported",
    "item_deadline",
]
