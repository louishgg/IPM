"""Shared safeguards for the fully offline test suite."""

import http.client
import socket
import urllib.request

import pytest
import requests

try:
    from curl_cffi import requests as curl_requests
except ImportError:  # pragma: no cover - dependency validation tests cover this
    curl_requests = None


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Fail every test that attempts an outbound socket or HTTP request."""

    def blocked(*args, **kwargs):
        raise AssertionError("network access is forbidden in the test suite")

    class GuardedSocket(socket.socket):
        def connect(self, *args, **kwargs):
            blocked(*args, **kwargs)

        def connect_ex(self, *args, **kwargs):
            blocked(*args, **kwargs)

    monkeypatch.setattr(socket, "socket", GuardedSocket)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    if curl_requests is not None:
        monkeypatch.setattr(curl_requests.Session, "request", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    monkeypatch.setattr(http.client.HTTPConnection, "connect", blocked)
    monkeypatch.setattr(http.client.HTTPSConnection, "connect", blocked)
