"""Full-window Yahoo replacement accepts small revisions and holds anomalies."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from data_acquisition.contracts import (
    AcquisitionIdentity, AcquisitionRequest, AcquisitionStatus, ProviderStatus,
)
from data_acquisition.engine import AcquisitionPolicy, SerialAcquisitionEngine
from data_acquisition.errors import AcquisitionReviewRequired
from data_acquisition.providers.base import AcquisitionResult, ProviderAdapter
from live.acquisition_execution import (
    _extension_price_adapter, _merge_price_payload, _price_status_matches,
    _reconcile_price_checkpoint,
)
from live.acquisition_planning import price_request_window_matches


def _fixture():
    dates = pd.bdate_range("2024-01-02", periods=25, name="Date")
    panels = {
        name: pd.DataFrame({"AAA": [base + n for n in range(25)]}, index=dates)
        for name, base in (("open", 100.), ("close", 101.), ("volume", 1000.))
    }
    fresh = pd.DataFrame({
        field: panels[name].AAA.copy()
        for name, field in (("open", "Open"), ("close", "Close"), ("volume", "Volume"))
    })
    fresh.loc[pd.Timestamp("2023-01-31")] = [90., 91., 900.]
    request = AcquisitionRequest(
        AcquisitionIdentity("live", "prices", "AAA", "yahoo", "AAA"),
        "2023-01-01", "2026-05-06",
    )
    return panels, fresh.sort_index(), request


def _adapter(fresh, calls):
    def fetch(request):
        calls.append(request)
        return AcquisitionResult(
            fresh.copy(), len(fresh), fresh.index.min().date().isoformat(),
            fresh.index.max().date().isoformat(), http_status=200,
        )
    return ProviderAdapter(
        fetch, provider="yahoo", dataset="prices", client_name="mock", client_version="1",
    )


@pytest.mark.parametrize("change", [0., -0.000862, 0.01])
def test_full_download_uses_provider_values_without_scaling(tmp_path, change):
    panels, fresh, request = _fixture()
    before = {name: frame.copy() for name, frame in panels.items()}
    fresh[["Open", "Close"]] *= 1 + change
    calls, notes = [], {}
    result = _extension_price_adapter(
        _adapter(fresh, calls), panels, [], evidence_dir=tmp_path,
        replacement_notes=notes,
    ).fetch(request)
    assert calls == [request]
    pd.testing.assert_frame_equal(result.payload, fresh, check_freq=False)
    for name in panels:
        pd.testing.assert_frame_equal(panels[name], before[name])
    note = json.loads(notes[request.identity.key])
    assert note["price_replacement"] == "accepted_provider_values"
    assert "price_multiplier" not in note
    assert (tmp_path / (note["provider_response_sha256"] + ".csv")).is_file()
    _merge_price_payload(panels, "AAA", result.payload, allow_reprice=True)
    assert panels["close"].at[fresh.index[-1], "AAA"] == fresh.Close.iloc[-1]
    status = SimpleNamespace(
        identity=request.identity, observation_count=len(fresh),
        observation_start="2023-01-31", observation_end=fresh.index.max().date().isoformat(),
    )
    assert _price_status_matches(status, panels)


@pytest.mark.parametrize("problem", ["large_price", "large_volume", "missing_date", "missing_field"])
def test_anomaly_holds_entire_cached_security_for_review(tmp_path, problem):
    panels, fresh, request = _fixture()
    before = {name: frame.copy() for name, frame in panels.items()}
    if problem == "large_price":
        fresh.loc[fresh.index[-1], "Close"] *= 1.0101
    elif problem == "large_volume":
        fresh.loc[fresh.index[-1], "Volume"] *= 2
    elif problem == "missing_date":
        fresh = fresh.drop(fresh.index[-1])
    else:
        fresh.loc[fresh.index[-1], "Open"] = float("nan")
    calls, notes = [], {}
    adapter = _extension_price_adapter(
        _adapter(fresh, calls), panels, [], evidence_dir=tmp_path,
        replacement_notes=notes,
    )
    with pytest.raises(AcquisitionReviewRequired, match="retained pending review"):
        adapter.fetch(request)
    with pytest.raises(AcquisitionReviewRequired):
        adapter.fetch(request)
    assert calls == [request]
    assert json.loads(notes[request.identity.key])["price_replacement"] == "review_required"
    for name in panels:
        pd.testing.assert_frame_equal(panels[name], before[name])


def test_review_is_not_retried_and_refresh_cannot_bypass_it():
    panels, fresh, request = _fixture()
    fresh["Close"] *= 0.5
    calls = []
    policy = AcquisitionPolicy(max_attempts=3, initial_backoff_seconds=0)
    run = SerialAcquisitionEngine(
        _extension_price_adapter(_adapter(fresh, calls), panels, []), policy=policy,
    ).run([request])
    assert len(calls) == 1
    assert run.statuses[0].error_class == "AcquisitionReviewRequired"
    assert run.statuses[0].status is ProviderStatus.FAILED
    resumed = SerialAcquisitionEngine(
        _extension_price_adapter(_adapter(fresh, calls), panels, run.statuses), policy=policy,
    ).run([request], statuses=run.statuses, refresh=True)
    assert len(calls) == 1
    assert resumed.statuses[0].error_class == "AcquisitionReviewRequired"


def test_checkpoint_mismatch_does_not_erase_cached_alias_series():
    panels, _, request = _fixture()
    before = {name: frame.copy() for name, frame in panels.items()}
    stale = AcquisitionStatus(
        identity=request.identity, status=ProviderStatus.OK,
        requested_start="2024-01-01", requested_end=request.requested_end,
        observation_count=2, observation_start="2024-01-02", observation_end="2024-01-03",
    )
    actual = _reconcile_price_checkpoint([request], [stale], panels)
    assert actual[0].status is ProviderStatus.PENDING
    for name in panels:
        pd.testing.assert_frame_equal(panels[name], before[name])
    assert not price_request_window_matches(request, stale)


@pytest.mark.parametrize("state,expected", [
    (ProviderStatus.OK, True), (ProviderStatus.NO_DATA, True),
    (ProviderStatus.PENDING, False), (ProviderStatus.FAILED, False),
])
def test_pending_or_failed_history_cannot_be_skipped_by_competition_readiness(state, expected):
    _, _, request = _fixture()
    status = SimpleNamespace(
        status=state, requested_start=request.requested_start,
        requested_end=request.requested_end,
    )
    assert price_request_window_matches(request, status) is expected


@pytest.mark.parametrize('problem', ['missing_date', 'large_revision'])
def test_replacement_reviews_cached_supplement_even_without_raw_provider_column(problem):
    panels, fresh, request = _fixture()
    supplement = pd.DataFrame({
        'Date': panels['close'].index, 'Asset_ID': 'AAA',
        'Open': panels['open'].AAA.to_numpy(), 'Close': panels['close'].AAA.to_numpy(),
        'Volume': panels['volume'].AAA.to_numpy(),
    })
    empty = {name: frame.drop(columns='AAA') for name, frame in panels.items()}
    if problem == 'missing_date':
        fresh = fresh.drop(fresh.index[-1])
    else:
        fresh['Close'] *= 0.5
    with pytest.raises(AcquisitionReviewRequired, match='retained pending review'):
        _extension_price_adapter(
            _adapter(fresh, []), empty, [], supplemental=supplement,
        ).fetch(request)
    assert all(frame.empty for frame in empty.values())


def test_price_extension_keeps_old_rows_and_adds_history():
    panels = {
        name: pd.DataFrame({'AAA': values}, index=pd.to_datetime(['2026-01-30', '2026-02-27']))
        for name, values in (('open', [10., 11.]), ('close', [10.2, 11.2]), ('volume', [100., 110.]))
    }
    payload = pd.DataFrame(
        {'Open': [9., 10.], 'Close': [9.2, 10.2], 'Volume': [90., 100.]},
        index=pd.to_datetime(['2025-12-31', '2026-01-30']),
    )
    _merge_price_payload(panels, 'AAA', payload)
    assert panels['close'].loc[pd.Timestamp('2025-12-31'), 'AAA'] == 9.2
    assert panels['close'].loc[pd.Timestamp('2026-02-27'), 'AAA'] == 11.2


def test_price_extension_rejects_changed_overlap():
    panels, fresh, _ = _fixture()
    fresh['Close'] *= 1.25
    with pytest.raises(ValueError, match='price basis changed'):
        _merge_price_payload(panels, 'AAA', fresh)
