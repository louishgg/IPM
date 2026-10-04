"""Shared deterministic acquisition helpers for tests."""

import json

from data_acquisition.contracts import AcquisitionIdentity, AcquisitionRequest


class _Response:
    def __init__(self, status_code, *, content=b"", payload=None, text=""):
        self.status_code = status_code
        self.content = content
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is not None:
            return self._payload
        return json.loads(self.content)


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def backtest_acquisition_request(
    asset_id: str = "AAA",
    *,
    dataset: str = "shares",
    provider: str = "yahoo",
    provider_symbol: str | None = None,
    start: str = "2024-01-01",
    end: str = "2024-01-31",
) -> AcquisitionRequest:
    return AcquisitionRequest(
        identity=AcquisitionIdentity(
            scope="backtest",
            dataset=dataset,
            asset_id=asset_id,
            provider=provider,
            provider_symbol=provider_symbol or asset_id,
        ),
        requested_start=start,
        requested_end=end,
    )
