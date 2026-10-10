"""Tests for the TermoWeb REST client (backend/rest_client.py)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import logging
from typing import Any, Callable
from unittest.mock import AsyncMock

import aiohttp
from aiohttp import ClientError
import pytest

from custom_components.termoweb.backend.ducaheat import (
    DucaheatRequestError,
    DucaheatRESTClient,
)
import custom_components.termoweb.backend.rest_client as api
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.backend.sanitize import mask_identifier
from custom_components.termoweb.codecs.termoweb_codec import decode_samples
from custom_components.termoweb.const import (
    BRAND_DUCAHEAT,
    BRAND_TERMOWEB,
    get_brand_requested_with,
    get_brand_user_agent,
)
from custom_components.termoweb.domain.state import GeoData
from custom_components.termoweb.inventory import AccumulatorNode
from tests.fakes.rest import FakeSession, LatchedResponse, MockResponse


def _patch_api_clock(
    monkeypatch: pytest.MonkeyPatch,
    *,
    wall: float | Callable[[], float],
    mono: float | Callable[[], float] | None = None,
) -> None:
    """Patch the monotonic timer used by the API client.

    The ``wall`` parameter is accepted for backward compatibility and used
    as the default for ``mono`` when ``mono`` is not provided.
    """

    if callable(wall):
        wall_func = wall
    else:
        wall_value = float(wall)

        def wall_func() -> float:
            return wall_value

    if mono is None:
        mono_func = wall_func
    elif callable(mono):
        mono_func = mono
    else:
        mono_value = float(mono)

        def mono_func() -> float:
            return mono_value

    monkeypatch.setattr(api, "time_mod", mono_func)


def _set_token_expiry_seconds(client: RESTClient, seconds: float) -> None:
    """Set token expiry relative to the current time providers."""

    now_mono = api.time_mod()
    client._token_expiry_monotonic = now_mono + seconds


def test_token_refresh(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "t1", "expires_in": 1},
                headers={"Content-Type": "application/json"},
            ),
            MockResponse(
                200,
                {"access_token": "t2", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            ),
        )

        client = RESTClient(session, "user", "pass")

        fake_time = 0.0

        def _fake_time() -> float:
            return fake_time

        _patch_api_clock(monkeypatch, wall=_fake_time)
        token1 = await client._ensure_token()
        assert token1 == "t1"

        fake_time = 2.0  # advance beyond expiry
        token2 = await client._ensure_token()
        assert token2 == "t2"
        assert len(session.post_calls) == 2

    asyncio.run(_run())


def test_ducaheat_token_headers() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(
            session,
            "user",
            "pass",
            api_base="https://api-tevolve.termoweb.net",
        )

        token = await client._ensure_token()
        assert token == "tok"

        assert session.post_calls
        headers = session.post_calls[0][2]["headers"]
        assert headers["X-SerialId"] == "15"
        assert headers["X-Requested-With"] == get_brand_requested_with(BRAND_DUCAHEAT)
        assert headers["User-Agent"] == get_brand_user_agent(BRAND_DUCAHEAT)

    asyncio.run(_run())


def test_ensure_token_401_raises_auth_error() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                401,
                {},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pass")

        with pytest.raises(api.BackendAuthError):
            await client._ensure_token()

    asyncio.run(_run())


def test_ensure_token_429_raises_rate_limit_error() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                429,
                {},
                headers={"Content-Type": "application/json"},
                text_data='{"error":"rate"}',
            )
        )

        client = RESTClient(session, "user", "pass")

        with pytest.raises(api.BackendRateLimitError):
            await client._ensure_token()

    asyncio.run(_run())


def test_ensure_token_missing_access_token() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"unexpected": True},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pass")

        with pytest.raises(api.BackendAuthError):
            await client._ensure_token()

    asyncio.run(_run())


def test_resolve_node_descriptor_validations() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    with pytest.raises(ValueError, match="Unsupported node descriptor"):
        client._resolve_node_descriptor("htr")

    with pytest.raises(ValueError, match="Invalid node type"):
        client._resolve_node_descriptor(("  ", "1"))

    with pytest.raises(ValueError, match="Invalid node address"):
        client._resolve_node_descriptor(("htr", ""))


def test_resolve_node_descriptor_normalises_values() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    node = AccumulatorNode(name=" Storage ", addr=" 007 ")
    assert client._resolve_node_descriptor(node) == ("acm", "007")

    assert client._resolve_node_descriptor(("HTR", " 08 ")) == ("htr", "08")


def test_ensure_token_non_numeric_expires_in(monkeypatch) -> None:
    fake_time = 1000.0

    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": "soon"},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pass")

        token = await client._ensure_token()
        assert token == "tok"
        assert client._token_expiry_monotonic == pytest.approx(fake_time + 3600)

    _patch_api_clock(monkeypatch, wall=lambda: fake_time)
    asyncio.run(_run())


def test_get_node_samples_success() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                200,
                {"samples": [{"t": 1000, "counter": "1.5"}]},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pass")
        samples = await client.get_node_samples("dev", ("htr", "A"), 0, 10)

        assert samples == [{"t": 1000, "counter": "1.5"}]
        assert len(session.request_calls) == 1
        params = session.request_calls[0][2]["params"]
        assert params == {"start": 0, "end": 10}

    asyncio.run(_run())


def test_get_node_samples_404() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                404,
                {},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pass")
        with pytest.raises(aiohttp.ClientResponseError) as err:
            await client.get_node_samples("dev", ("htr", "A"), 0, 10)
        assert err.value.status == 404

    asyncio.run(_run())


def test_request_ignore_status_returns_none() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                404,
                {},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pass")
        result = await client._request(
            "GET", "/missing", headers={}, ignore_statuses=(404,)
        )
        assert result is None

    asyncio.run(_run())


def test_request_refreshes_once_then_raises() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "initial", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            ),
            MockResponse(
                200,
                {"access_token": "refreshed", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            ),
        )
        session.queue_request(
            LatchedResponse(
                MockResponse(
                    401,
                    {"error": "invalid_token"},
                    headers={"Content-Type": "application/json"},
                    text_data='{"error":"invalid_token"}',
                )
            )
        )

        client = RESTClient(session, "user", "pass")
        headers = await client.authed_headers()
        session.clear_calls()

        with pytest.raises(api.BackendAuthError):
            await client._request("GET", "/api/test", headers=headers)

        assert len(session.request_calls) == 2
        assert len(session.post_calls) == 1
        first_auth = session.request_calls[0][2]["headers"]["Authorization"]
        second_auth = session.request_calls[1][2]["headers"]["Authorization"]
        assert first_auth == "Bearer initial"
        assert second_auth == "Bearer refreshed"

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_authed_headers_builds_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession()
    client = RESTClient(session, "user", "pass")
    ensure_mock = AsyncMock(return_value="token")
    monkeypatch.setattr(client, "_ensure_token", ensure_mock)

    headers = await client.authed_headers()

    assert ensure_mock.await_count == 1
    expected = {
        "Authorization": "Bearer token",
        "Accept": "application/json",
        "User-Agent": client._user_agent,
        "Accept-Language": api.ACCEPT_LANGUAGE,
    }
    if client._requested_with:
        expected["X-Requested-With"] = client._requested_with
    assert headers == expected


def test_request_rate_limit_error() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "token", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                429,
                {"error": "rate"},
                headers={"Content-Type": "application/json"},
                text_data='{"error":"rate"}',
            )
        )

        client = RESTClient(session, "user", "pass")
        headers = await client.authed_headers()
        session.clear_calls()

        with pytest.raises(api.BackendRateLimitError):
            await client._request("GET", "/api/rate", headers=headers)

        assert len(session.request_calls) == 1
        assert len(session.post_calls) == 0

    asyncio.run(_run())


def test_request_5xx_surfaces_client_error() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "token", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                503,
                {"error": "down"},
                headers={"Content-Type": "application/json"},
                text_data="Service unavailable",
            )
        )

        client = RESTClient(session, "user", "pass")
        headers = await client.authed_headers()
        session.clear_calls()

        with pytest.raises(aiohttp.ClientResponseError) as err:
            await client._request("GET", "/api/down", headers=headers)

        assert err.value.status == 503
        assert len(session.request_calls) == 1

    asyncio.run(_run())


def test_request_timeout_propagates() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "token", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(asyncio.TimeoutError())

        client = RESTClient(session, "user", "pass")
        headers = await client.authed_headers()
        session.clear_calls()

        with pytest.raises(asyncio.TimeoutError):
            await client._request("GET", "/api/slow", headers=headers)

        assert len(session.request_calls) == 1

    asyncio.run(_run())


def test_api_base_property_returns_sanitized() -> None:
    session = FakeSession()
    client = RESTClient(session, "user", "pw", api_base="https://api.example.com/")

    assert client.api_base == "https://api.example.com"


def test_request_text_exception_fallback() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                200,
                {"ok": True},
                headers={"Content-Type": "application/json"},
                text_exc=RuntimeError("boom"),
            )
        )

        client = RESTClient(session, "user", "pw")
        headers = await client.authed_headers()
        session.clear_calls()

        result = await client._request("GET", "/api/data", headers=headers)
        assert result == {"ok": True}

    asyncio.run(_run())


def test_request_returns_plain_text() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                200,
                "ignored",
                headers={"Content-Type": "text/plain"},
                text_data="hello world",
            )
        )

        client = RESTClient(session, "user", "pw")
        headers = await client.authed_headers()
        session.clear_calls()

        result = await client._request("GET", "/api/plain", headers=headers)
        assert result == "hello world"

    asyncio.run(_run())


def test_log_non_htr_payload_truncates_long_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession()
    client = RESTClient(session, "user", "pass")

    class StubLogger:
        def __init__(self) -> None:
            self.debug_calls: list[tuple[str, tuple[Any, ...]]] = []

        def debug(self, msg: str, *args: Any) -> None:
            self.debug_calls.append((msg, args))

    stub_logger = StubLogger()
    monkeypatch.setattr(api, "_LOGGER", stub_logger)

    payload = {"data": "x" * 600}
    client._log_non_htr_payload(
        node_type="acm",
        dev_id="device-12345",
        addr="001",
        stage="update",
        payload=payload,
    )

    assert len(stub_logger.debug_calls) == 1
    _, args = stub_logger.debug_calls[0]
    snippet = args[-1]
    assert isinstance(snippet, str)
    assert snippet.endswith("...")
    assert len(snippet) <= 500


def test_ensure_token_uses_cache_without_http() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = "cached"
        _set_token_expiry_seconds(client, 1000.0)

        token = await client._ensure_token()
        assert token == "cached"
        assert session.post_calls == []

    asyncio.run(_run())


def test_ensure_token_concurrent_calls_share_refresh() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pw")
        tokens = await asyncio.gather(client._ensure_token(), client._ensure_token())

        assert tokens == ["tok", "tok"]
        assert len(session.post_calls) == 1

    asyncio.run(_run())


def test_ensure_token_returns_cached_after_lock_entry() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = None
        client._token_expiry_monotonic = 0.0

        class FakeLock:
            def __init__(self, owner: RESTClient) -> None:
                self._owner = owner

            async def __aenter__(self) -> "FakeLock":
                self._owner._access_token = "cached"
                _set_token_expiry_seconds(self._owner, 100.0)
                return self

            async def __aexit__(self, *_exc: Any) -> bool:
                return False

        client._lock = FakeLock(client)  # type: ignore[assignment]

        token = await client._ensure_token()
        assert token == "cached"
        assert session.post_calls == []

    asyncio.run(_run())


def test_token_request_error_raises_client_response_error() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                500,
                {},
                headers={"Content-Type": "text/plain"},
                text_data="failure",
            )
        )

        client = RESTClient(session, "user", "pw")
        with pytest.raises(aiohttp.ClientResponseError) as err:
            await client._ensure_token()
        assert err.value.status == 500

    asyncio.run(_run())


def test_list_devices_handles_various_shapes() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                200,
                {"devs": [{"id": 1}, "bad"]},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                200,
                {"devices": [{"dev_id": "abc"}, 123]},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pw")
        first = await client.list_devices()
        second = await client.list_devices()

        assert first == [{"id": 1}]
        assert second == [{"dev_id": "abc"}]

    asyncio.run(_run())


def test_ducaheat_authed_request_headers() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(200, [], headers={"Content-Type": "application/json"})
        )

        client = RESTClient(
            session,
            "user",
            "pw",
            api_base="https://api-tevolve.termoweb.net",
        )

        await client.list_devices()

        assert session.request_calls
        method, url, kwargs = session.request_calls[0]
        assert method == "GET"
        assert url == "https://api-tevolve.termoweb.net/api/v2/devs/"
        headers = kwargs["headers"]
        assert headers["X-Requested-With"] == get_brand_requested_with(BRAND_DUCAHEAT)
        assert headers["User-Agent"] == get_brand_user_agent(BRAND_DUCAHEAT)
        assert headers["X-SerialId"] == "15"

    asyncio.run(_run())


def test_termoweb_authed_request_headers() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(200, [], headers={"Content-Type": "application/json"})
        )

        client = RESTClient(session, "user", "pw")

        await client.list_devices()

        assert session.request_calls
        method, url, kwargs = session.request_calls[0]
        assert method == "GET"
        assert url == "https://control.termoweb.net/api/v2/devs/"
        headers = kwargs["headers"]
        assert headers["User-Agent"] == get_brand_user_agent(BRAND_TERMOWEB)
        assert headers["X-Requested-With"] == get_brand_requested_with(BRAND_TERMOWEB)

    asyncio.run(_run())


def test_get_nodes_and_settings_use_expected_paths(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        _set_token_expiry_seconds(client, 1000.0)

        calls: list[tuple[str, str]] = []

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((method, path))
            return {"ok": True}

        monkeypatch.setattr(client, "_request", fake_request)

        await client.get_nodes("dev123")
        await client.get_node_settings("dev123", ("htr", "5"))
        await client.get_node_samples("dev123", ("htr", "5"), 0, 10)

        assert calls == [
            ("GET", api.NODES_PATH_FMT.format(dev_id="dev123")),
            ("GET", f"/api/v2/devs/dev123/htr/5/settings"),
            ("GET", f"/api/v2/devs/dev123/htr/5/samples"),
        ]

    asyncio.run(_run())


def test_set_node_lock_uses_lock_segment_for_ducaheat(monkeypatch) -> None:
    """Ducaheat child lock writes should target the segmented lock endpoint."""

    async def _run() -> None:
        session = FakeSession()

        ducaheat = DucaheatRESTClient(
            session,
            "user",
            "pw",
            api_base="https://api-tevolve.termoweb.net",
        )
        ducaheat._access_token = "tok"
        _set_token_expiry_seconds(ducaheat, 1000.0)

        calls: list[tuple[str, str, dict[str, Any]]] = []

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((method, path, kwargs.get("json", {})))
            return {"ok": True}

        monkeypatch.setattr(ducaheat, "_request", fake_request)

        await ducaheat.set_node_lock("dev123", ("htr", "5"), lock=False)

        assert calls == [("POST", "/api/v2/devs/dev123/htr/5/lock", {"lock": False})]

    asyncio.run(_run())


def test_get_rtc_time_uses_expected_path(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        _set_token_expiry_seconds(client, 1000.0)

        calls: list[tuple[str, str]] = []

        async def fake_request(method: str, path: str, **kwargs: Any) -> Any:
            calls.append((method, path))
            return {"status": "ok"}

        monkeypatch.setattr(client, "_request", fake_request)

        data = await client.get_rtc_time("dev123")

        assert data == {"status": "ok"}
        assert calls == [("GET", "/api/v2/devs/dev123/mgr/rtc/time")]

    asyncio.run(_run())


def test_get_rtc_time_handles_non_dict(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        _set_token_expiry_seconds(client, 1000.0)

        async def fake_request(method: str, path: str, **kwargs: Any) -> Any:
            return ["unexpected"]

        monkeypatch.setattr(client, "_request", fake_request)

        caplog.set_level(logging.DEBUG, logger=api.__name__)
        data = await client.get_rtc_time("dev456")

        assert data == {}
        assert any(
            "Unexpected RTC time payload" in record.getMessage()
            for record in caplog.records
        )

    asyncio.run(_run())


def test_get_node_settings_acm_logs(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {"status": {"mode": "auto"}, "capabilities": {"boost": {"max": 60}}},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        _set_token_expiry_seconds(client, 1000.0)

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer tok"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        caplog.set_level(logging.DEBUG, logger=api.__name__)
        node = AccumulatorNode(name="Store", addr="7")
        data = await client.get_node_settings("dev", node)

        assert data["mode"] == "auto"
        expected = (
            f"GET settings node {mask_identifier('dev')}/{mask_identifier('7')}"
            " (acm) payload"
        )
        assert any(expected in record.getMessage() for record in caplog.records)

    asyncio.run(_run())


def test_get_node_settings_pmo_uses_device_endpoint(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        _set_token_expiry_seconds(client, 1000.0)

        captured: dict[str, Any] = {}

        async def fake_request(method: str, path: str, **kwargs: Any) -> Any:
            captured["method"] = method
            captured["path"] = path
            captured["kwargs"] = kwargs
            return {"status": {"power": 0}}

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer tok"}

        monkeypatch.setattr(client, "_request", fake_request)
        monkeypatch.setattr(client, "authed_headers", fake_headers)

        payload = await client.get_node_settings("dev", ("pmo", "4"))

        assert payload == {"power": 0}
        assert captured == {
            "method": "GET",
            "path": "/api/v2/devs/dev/pmo/4",
            "kwargs": {"headers": {"Authorization": "Bearer tok"}},
        }

    asyncio.run(_run())


def test_get_node_samples_logs_for_unknown_type(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {"unexpected": True},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        _set_token_expiry_seconds(client, 1000.0)

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer tok"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        caplog.set_level(logging.DEBUG, logger=api.__name__)
        samples = await client.get_node_samples("dev", ("pmo", "4"), 0, 5)

        assert samples == []
        expected = (
            f"GET samples node {mask_identifier('dev')}/{mask_identifier('4')}"
            " (pmo) payload"
        )
        assert any(expected in record.getMessage() for record in caplog.records)

    asyncio.run(_run())


def test_set_node_settings_includes_prog_and_ptemp(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pw")
        client._access_token = "tok"
        _set_token_expiry_seconds(client, 1000.0)

        received: list[dict[str, Any]] = []

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            received.append(kwargs["json"])
            return {"ok": True}

        monkeypatch.setattr(client, "_request", fake_request)

        prog = [0, 1, 2] * 56
        ptemp = [18.0, 19.0, 20.0]
        await client.set_node_settings(
            "dev123",
            ("htr", "7"),
            prog=prog,
            ptemp=ptemp,
            units="f",
        )

        assert received == [
            {
                "units": "F",
                "prog": prog,
                "ptemp": ["18.0", "19.0", "20.0"],
            }
        ]

    asyncio.run(_run())


def test_request_cancelled_error_propagates() -> None:
    async def _run() -> None:
        session = FakeSession()

        def raise_cancelled() -> None:
            raise asyncio.CancelledError()

        session.queue_request(raise_cancelled)

        client = RESTClient(session, "user", "pass")

        with pytest.raises(asyncio.CancelledError):
            await client._request("GET", "/api/cancel", headers={})

    asyncio.run(_run())


def test_request_generic_exception_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=api.__name__)

    async def _run() -> None:
        session = FakeSession()
        session.queue_request(RuntimeError("upstream Bearer secret-token failure"))

        client = RESTClient(session, "user", "pass")

        headers = {"Authorization": "Bearer secret-token"}

        with pytest.raises(RuntimeError):
            await client._request("GET", "/api/fail", headers=headers)

    asyncio.run(_run())

    assert "Request GET" in caplog.text
    assert "Bearer ***" in caplog.text
    # Transient transport errors log once, below ERROR.
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_set_node_settings_invalid_units() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        with pytest.raises(ValueError, match="Invalid units"):
            await client.set_node_settings("dev", ("htr", "1"), units="kelvin")

        assert not session.request_calls
        assert not session.post_calls

    asyncio.run(_run())


def test_set_node_settings_invalid_program() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        with pytest.raises(ValueError, match="prog must be a list of 168"):
            await client.set_node_settings("dev", ("htr", "1"), prog=[0, 1, 2])

        with pytest.raises(ValueError, match="prog values must be 0, 1, or 2"):
            await client.set_node_settings("dev", ("htr", "1"), prog=[0] * 167 + [5])

        with pytest.raises(ValueError, match="prog contains non-integer value"):
            await client.set_node_settings(
                "dev", ("htr", "1"), prog=[0] * 167 + ["bad"]
            )

        assert not session.request_calls
        assert not session.post_calls

    asyncio.run(_run())


def test_set_node_settings_invalid_temperatures() -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        with pytest.raises(ValueError, match="Invalid stemp value"):
            await client.set_node_settings("dev", ("htr", "1"), stemp="warm")

        with pytest.raises(
            ValueError, match="ptemp must be a list of three numeric values"
        ):
            await client.set_node_settings("dev", ("htr", "1"), ptemp=[21.0, 19.0])

        with pytest.raises(ValueError, match="ptemp contains non-numeric value"):
            await client.set_node_settings(
                "dev",
                ("htr", "1"),
                ptemp=[21.0, "bad", 23.0],
            )

        assert not session.request_calls
        assert not session.post_calls

    asyncio.run(_run())


def test_get_node_samples_empty_payload() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                200,
                {"samples": []},
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pass")
        samples = await client.get_node_samples("dev", ("htr", "A"), 0, 10)

        assert samples == []

    asyncio.run(_run())


def test_get_node_samples_decreasing_counters() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "tok", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            )
        )
        session.queue_request(
            MockResponse(
                200,
                {
                    "samples": [
                        {"t": 1, "counter": "3.0"},
                        {"t": 2, "counter": "2.5"},
                    ]
                },
                headers={"Content-Type": "application/json"},
            )
        )

        client = RESTClient(session, "user", "pass")
        samples = await client.get_node_samples("dev", ("htr", "A"), 0, 10)

        assert samples == [
            {"t": 1, "counter": "3.0"},
            {"t": 2, "counter": "2.5"},
        ]

    asyncio.run(_run())


def test_get_node_samples_malformed_items(monkeypatch, caplog) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {}

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            return {
                "samples": [
                    123,
                    {"t": "bad"},
                    {"t": 5, "counter": None},
                ]
            }

        monkeypatch.setattr(client, "authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        with caplog.at_level("DEBUG"):
            samples = await client.get_node_samples("dev", ("htr", "A"), 0, 10)

        assert samples == []

    caplog.clear()
    asyncio.run(_run())
    messages = [rec.message for rec in caplog.records]
    assert any("Unexpected htr sample item" in msg for msg in messages)
    assert any("Unexpected htr sample shape" in msg for msg in messages)


def test_get_node_samples_unexpected_payload(monkeypatch, caplog) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {}

        async def fake_request(method: str, path: str, **kwargs: Any) -> Any:
            return "garbled"

        monkeypatch.setattr(client, "authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        with caplog.at_level("DEBUG"):
            samples = await client.get_node_samples("dev", ("htr", "A"), 0, 10)

        assert samples == []

    caplog.clear()
    asyncio.run(_run())
    assert any(
        "Unexpected htr samples payload" in rec.message for rec in caplog.records
    )


def test_request_recovers_after_token_refresh() -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_post(
            MockResponse(
                200,
                {"access_token": "old", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            ),
            MockResponse(
                200,
                {"access_token": "new", "expires_in": 3600},
                headers={"Content-Type": "application/json"},
            ),
        )
        session.queue_request(
            MockResponse(
                401,
                {"error": "expired"},
                headers={"Content-Type": "application/json"},
                text_data='{"error":"expired"}',
            ),
            MockResponse(
                200,
                [{"dev_id": "1"}],
                headers={"Content-Type": "application/json"},
            ),
        )

        client = RESTClient(session, "user", "pass")
        devices = await client.list_devices()

        assert devices == [{"dev_id": "1"}]
        assert len(session.request_calls) == 2
        assert len(session.post_calls) == 2
        refreshed_headers = session.request_calls[1][2]["headers"]
        assert refreshed_headers["Authorization"] == "Bearer new"

    asyncio.run(_run())


def test_list_devices_unexpected_dict_payload(monkeypatch, caplog) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {}

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            return {"weird": []}

        monkeypatch.setattr(client, "authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        with caplog.at_level("DEBUG"):
            devices = await client.list_devices()

        assert devices == []

    caplog.clear()
    asyncio.run(_run())
    assert any("Unexpected /devs shape" in rec.message for rec in caplog.records)


def test_list_devices_unexpected_string_payload(monkeypatch, caplog) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {}

        async def fake_request(method: str, path: str, **kwargs: Any) -> str:
            return "oops"

        monkeypatch.setattr(client, "authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        with caplog.at_level("DEBUG"):
            devices = await client.list_devices()

        assert devices == []

    caplog.clear()
    asyncio.run(_run())
    assert any("Unexpected /devs shape" in rec.message for rec in caplog.records)


def test_set_node_settings_translates_heat(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(session, "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {}

        captured: dict[str, Any] = {}

        async def fake_request(method: str, path: str, **kwargs: Any) -> Any:
            captured["json"] = kwargs.get("json")
            return {}

        monkeypatch.setattr(client, "authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)

        await client.set_node_settings("dev", ("htr", 1), mode="heat", stemp=21.0)

        assert captured["json"]["mode"] == "manual"
        assert captured["json"]["stemp"] == "21.0"

    asyncio.run(_run())


def test_build_acm_extra_options_payload_with_boost_time_only() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    payload = client._build_acm_extra_options_payload(180, None)

    assert payload == {"extra_options": {"boost_time": 180}}


def test_build_acm_extra_options_payload_with_boost_temp_only() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    payload = client._build_acm_extra_options_payload(None, 23.0)

    assert payload == {"extra_options": {"boost_temp": "23.0"}}


def test_build_acm_extra_options_payload_missing_inputs() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    with pytest.raises(ValueError, match="must be provided"):
        client._build_acm_extra_options_payload(None, None)


def test_build_acm_extra_options_payload_rejects_invalid_boost_temp() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    with pytest.raises(ValueError, match="Invalid boost_temp value"):
        client._build_acm_extra_options_payload(None, "invalid")


@pytest.mark.asyncio
async def test_set_acm_extra_options_forwards_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession()
    client = RESTClient(session, "user", "pass")

    sentinel_payload: dict[str, Any] = {"extra_options": {"boost_time": 120}}
    request_mock = AsyncMock(return_value={"ok": True})
    headers_mock = AsyncMock(return_value={"Authorization": "Bearer token"})

    monkeypatch.setattr(
        client, "_build_acm_extra_options_payload", lambda *args: sentinel_payload
    )
    monkeypatch.setattr(client, "_request", request_mock)
    monkeypatch.setattr(client, "authed_headers", headers_mock)
    monkeypatch.setattr(client, "_log_non_htr_payload", lambda **_: None)

    response = await client.set_acm_extra_options("dev123", 9, boost_time=120)

    assert response == {"ok": True}
    assert request_mock.await_count == 1
    await_call = request_mock.await_args
    assert await_call.args[0] == "POST"
    assert await_call.args[1] == "/api/v2/devs/dev123/acm/9/setup"
    assert await_call.kwargs["json"] is sentinel_payload
    assert await_call.kwargs["headers"] == {"Authorization": "Bearer token"}
    assert headers_mock.await_count == 1


@pytest.mark.asyncio
async def test_set_node_display_select_posts_select_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Display flash writes should post to the segmented select endpoint."""

    client = RESTClient(FakeSession(), "user", "pass")
    request_mock = AsyncMock(return_value={"ok": True})
    headers_mock = AsyncMock(return_value={"Authorization": "Bearer token"})

    monkeypatch.setattr(client, "_request", request_mock)
    monkeypatch.setattr(client, "authed_headers", headers_mock)

    response = await client.set_node_display_select("dev123", ("htr", "7"), select=True)

    assert response == {"ok": True}
    assert request_mock.await_count == 1
    await_call = request_mock.await_args
    assert await_call.args[0] == "POST"
    assert await_call.args[1] == "/api/v2/devs/dev123/htr/7/select"
    assert await_call.kwargs["json"] == {"select": True}
    assert await_call.kwargs["headers"] == {"Authorization": "Bearer token"}
    assert headers_mock.await_count == 1


@pytest.mark.asyncio
async def test_set_acm_boost_state_formats_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession()
    client = RESTClient(session, "user", "pass")

    request_mock = AsyncMock(return_value={"ok": True})
    headers_mock = AsyncMock(return_value={"Authorization": "Bearer token"})

    monkeypatch.setattr(client, "_request", request_mock)
    monkeypatch.setattr(client, "authed_headers", headers_mock)
    monkeypatch.setattr(client, "_log_non_htr_payload", lambda **_: None)

    response = await client.set_acm_boost_state(
        "dev123",
        "7",
        boost=True,
        boost_time=120,
        stemp=22.5,
        units=" f ",
    )

    assert response == {"ok": True}
    assert request_mock.await_count == 1
    await_call = request_mock.await_args
    assert await_call.args[0] == "POST"
    assert await_call.args[1] == "/api/v2/devs/dev123/acm/7/boost"
    assert await_call.kwargs["json"] == {
        "boost": True,
        "boost_time": 120,
        "stemp": "22.5",
        "units": "F",
    }
    assert await_call.kwargs["headers"] == {"Authorization": "Bearer token"}
    assert headers_mock.await_count == 1


@pytest.mark.asyncio
async def test_set_acm_boost_state_rejects_invalid_units() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    with pytest.raises(ValueError, match="Invalid units"):
        await client.set_acm_boost_state("dev456", "8", boost=True, units="kelvin")


@pytest.mark.asyncio
async def test_set_acm_boost_state_rejects_invalid_stemp() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    with pytest.raises(ValueError, match="Invalid stemp value"):
        await client.set_acm_boost_state("dev789", "9", boost=True, stemp="oops")


def test_ducaheat_get_node_settings_normalises_payload(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {
                    "status": {
                        "mode": "Manual",
                        "state": "heating",
                        "stemp": "21.0",
                        "temp": 20.5,
                        "units": "c",
                        "boost_active": True,
                    },
                    "setup": {
                        "extra_options": {"boost_temp": "23.0", "boost_time": 45}
                    },
                    "prog": {
                        "days": {
                            "mon": {"slots": [0, 1, 2, 0] * 6},
                            "tue": {"slots": [1] * 24},
                            "wed": {"slots": [2] * 24},
                            "thu": {"slots": [0] * 24},
                            "fri": {"slots": [1, 2] * 12},
                            "sat": {"slots": [2, 2, 1, 1] * 6},
                            "sun": {"slots": [0, 0, 1, 2] * 6},
                        }
                    },
                    "prog_temps": {
                        "comfort": "21.0",
                        "eco": "18.0",
                        "antifrost": "7.0",
                    },
                    "addr": "A1",
                },
                headers={"Content-Type": "application/json"},
            )
        )

        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        with caplog.at_level(
            logging.DEBUG, logger="custom_components.termoweb.backend.ducaheat"
        ):
            data = await client.get_node_settings("dev", ("htr", "A1"))

        assert data["mode"] == "manual"
        assert data["state"] == "heating"
        assert data["stemp"] == "21.0"
        assert data["mtemp"] == "20.5"
        assert data["units"] == "C"
        assert "boost_active" not in data
        assert "boost_time" not in data
        assert "boost_temp" not in data
        assert len(data["prog"]) == 168
        assert data["ptemp"] == ["7.0", "18.0", "21.0"]
        assert "raw" not in data

    asyncio.run(_run())


def test_ducaheat_get_node_settings_acm_strips_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {
                    "status": {
                        "mode": "auto",
                        "capabilities": {"boost": {"max": 90}},
                    },
                    "setup": {
                        "capabilities": {
                            "boost": {"min": 10},
                            "charge": {"modes": ["eco"]},
                        },
                        "extra_options": {"boost_temp": "25.0", "boost_time": 30},
                    },
                    "capabilities": {"charge": {"levels": [1, 2, 3]}},
                },
                headers={"Content-Type": "application/json"},
            )
        )

        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        data = await client.get_node_settings("dev", ("acm", "2"))

        assert data["mode"] == "auto"
        assert data["boost_temp"] == "25.0"
        assert data["boost_time"] == 30
        assert "capabilities" not in data
        assert session.request_calls[0][1] == (
            "https://api.termoweb.fake/api/v2/devs/dev/acm/2"
        )

    asyncio.run(_run())


def test_ducaheat_get_node_settings_acm_handles_half_hour_prog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        half_hour_prog = {
            str(idx): pattern
            for idx, pattern in enumerate(
                [
                    [0, 2] * 24,
                    [0, 0] * 24,
                    [2, 2] * 24,
                    [0, 1] * 24,
                    [1, 2] * 24,
                    [0, 2, 1, 0] * 12,
                    [0] * 48,
                ]
            )
        }

        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {
                    "status": {
                        "sync_status": "ok",
                        "mode": "off",
                        "heating": False,
                        "units": "C",
                        "stemp": "21.0",
                        "mtemp": 24.4,
                    },
                    "setup": {
                        "sync_status": "ok",
                        "operational_mode": 1,
                        "control_mode": 5,
                        "units": "C",
                        "offset": "0.0",
                        "priority": "medium",
                        "away_offset": "2.0",
                        "window_mode_enabled": False,
                        "prog_resolution": 1,
                        "charging_conf": {
                            "slot_1": {"start": 0, "end": 1430},
                            "slot_2": {"start": 0, "end": 0},
                            "active_days": [1, 1, 1, 1, 1, 1, 1],
                        },
                        "min_stemp": "5.0",
                        "max_stemp": "30.0",
                    },
                    "prog": {"sync_status": "ok", "prog": half_hour_prog},
                },
                headers={"Content-Type": "application/json"},
            )
        )

        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        data = await client.get_node_settings("dev", ("acm", "2"))

        assert data["mode"] == "off"
        assert data["stemp"] == "21.0"
        assert data["mtemp"] == "24.4"
        assert data["units"] == "C"
        assert len(data["prog"]) == 168
        assert data["prog"][:24] == [2] * 24
        assert data["prog"][24:48] == [0] * 24
        assert data["prog"][72:96] == [1] * 24
        assert data["prog"][96:120] == [2] * 24
        assert session.request_calls[0][1] == (
            "https://api.termoweb.fake/api/v2/devs/dev/acm/2"
        )

    asyncio.run(_run())


def test_ducaheat_set_node_settings_invalid_stemp(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        with pytest.raises(ValueError) as exc:
            await client.set_node_settings("dev", ("htr", "A1"), stemp="bad")

        assert "Invalid temperature value" in str(exc.value)

    asyncio.run(_run())


def test_ducaheat_set_node_settings_invalid_units(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(200, {}, headers={"Content-Type": "application/json"})
        )

        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        with pytest.raises(ValueError) as exc:
            await client.set_node_settings("dev", ("htr", "A1"), stemp=21.0, units="K")

        assert "Invalid units" in str(exc.value)

    asyncio.run(_run())


def test_ducaheat_set_acm_settings_short_prog(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        with pytest.raises(ValueError) as exc:
            await client.set_node_settings("dev", ("acm", "5"), prog=[0] * 24)

        assert "168" in str(exc.value)
        assert not session.request_calls

    asyncio.run(_run())


def test_ducaheat_set_acm_settings_client_error(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                400,
                {},
                headers={"Content-Type": "text/plain"},
                text_data="bad request",
            )
        )

        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        with pytest.raises(DucaheatRequestError) as exc:
            await client.set_node_settings("dev", ("acm", "5"), mode="boost")

        assert "bad request" in str(exc.value)

    asyncio.run(_run())


def test_rest_client_set_node_settings_rejects_boost_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        session = FakeSession()
        client = RESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        with pytest.raises(ValueError):
            await client.set_node_settings("dev", ("htr", "3"), boost_time=30)

    asyncio.run(_run())


def test_ducaheat_get_node_samples_converts_ms(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {"samples": [{"t": 1_700_000_000_500, "counter": 7.5}]},
                headers={"Content-Type": "application/json"},
            )
        )

        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        samples = await client.get_node_samples("dev", ("htr", "A"), 10, 20)
        assert samples == [{"t": 1_700_000_000, "counter": "7.5"}]

        call = session.request_calls[0]
        assert call[1] == "https://api.termoweb.fake/api/v2/devs/dev/htr/A/samples"
        assert call[2]["params"] == {"start": 10, "end": 20}

    asyncio.run(_run())


def test_ducaheat_get_node_samples_keeps_second_payload(monkeypatch) -> None:
    async def _run() -> None:
        session = FakeSession()
        session.queue_request(
            MockResponse(
                200,
                {"samples": [{"t": 1_700_000_010, "counter": 3}]},
                headers={"Content-Type": "application/json"},
            )
        )

        client = DucaheatRESTClient(
            session,
            "user",
            "pass",
            api_base="https://api.termoweb.fake",
        )

        async def fake_headers() -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        monkeypatch.setattr(client, "authed_headers", fake_headers)

        samples = await client.get_node_samples("dev", ("htr", "A"), 5, 30)
        assert samples == [{"t": 1_700_000_010, "counter": "3"}]

        call = session.request_calls[0]
        assert call[1] == "https://api.termoweb.fake/api/v2/devs/dev/htr/A/samples"
        assert call[2]["params"] == {"start": 5, "end": 30}

    asyncio.run(_run())


def test_rest_client_normalise_ws_nodes_passthrough() -> None:
    client = RESTClient(FakeSession(), "user", "pass", api_base="https://api.fake")
    payload = {"htr": {"settings": {"01": {}}}}
    assert client.normalise_ws_nodes(payload) is payload


def test_extract_samples_handles_list_payload() -> None:
    samples = decode_samples(
        [
            {"timestamp": 2000.0, "value": 5.5},
            {"t": "bad"},
            {"timestamp": 1000, "energy": 3},
        ]
    )

    assert samples == [{"t": 2000, "counter": "5.5"}, {"t": 1000, "counter": "3"}]


def test_extract_samples_preserves_min_max() -> None:
    samples = decode_samples(
        [
            {
                "t": 1000,
                "counter": {"value": 3_600_000, "min": 3_500_000, "max": 3_700_000},
            },
            {
                "t": 2000,
                "counter": 7_200_000,
                "counter_min": 7_100_000,
                "counter_max": 7_300_000,
            },
        ]
    )

    assert samples == [
        {
            "t": 1000,
            "counter": "3600000",
            "counter_min": "3500000",
            "counter_max": "3700000",
        },
        {
            "t": 2000,
            "counter": "7200000",
            "counter_min": "7100000",
            "counter_max": "7300000",
        },
    ]


def test_extract_samples_uses_counter_field_when_value_missing() -> None:
    samples = decode_samples(
        [
            {
                "t": 3000,
                "counter": {
                    "counter": 12_345,
                    "min": 12_000,
                    "max": 13_000,
                },
            }
        ]
    )

    assert samples == [
        {
            "t": 3000,
            "counter": "12345",
            "counter_min": "12000",
            "counter_max": "13000",
        }
    ]


@pytest.mark.asyncio
async def test_rest_client_rejects_cancel_boost_for_non_acm() -> None:
    client = RESTClient(FakeSession(), "user", "pass")

    with pytest.raises(ValueError, match="cancel_boost"):
        await client.set_node_settings(
            "dev",
            ("pmo", "1"),
            cancel_boost=True,
        )


def _json_response(body: Any) -> MockResponse:
    """Return a 200 JSON response carrying ``body``."""
    return MockResponse(200, body, headers={"Content-Type": "application/json"})


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"power_limit": "5000"}, 5000),
        ({"power_limit": "0"}, 0),
        ({"power_limit": "abc"}, None),
        ({}, None),
        (None, None),
    ],
)
async def test_get_power_limit_reads_the_htr_system_endpoint(
    body: Any, expected: int | None
) -> None:
    """The power limit is read as an int; junk and missing values are None."""
    session = FakeSession()
    session.queue_post(_json_response({"access_token": "tok", "expires_in": 3600}))
    session.queue_request(_json_response(body))
    client = RESTClient(session, "user", "pass")

    assert await client.get_power_limit("dev123") == expected
    method, url, _kwargs = session.request_calls[0]
    assert method == "GET"
    assert url.endswith("/api/v2/devs/dev123/htr_system/power_limit")


@pytest.mark.parametrize("limit", [5000, 0])
async def test_set_power_limit_posts_the_value_as_a_string(limit: int) -> None:
    """The backend expects the power limit as a string."""
    session = FakeSession()
    session.queue_post(_json_response({"access_token": "tok", "expires_in": 3600}))
    session.queue_request(_json_response({}))
    client = RESTClient(session, "user", "pass")

    await client.set_power_limit("dev123", power_limit=limit)

    method, url, kwargs = session.request_calls[0]
    assert method == "POST"
    assert url.endswith("/api/v2/devs/dev123/htr_system/power_limit")
    assert kwargs["json"] == {"power_limit": str(limit)}


def test_request_text_failure_logs_placeholder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ClientResponseError should use placeholder message when text() fails."""

    caplog.set_level(logging.DEBUG, logger=api.__name__)

    async def _run() -> None:
        session = FakeSession()
        response = MockResponse(
            500,
            {"detail": "boom"},
            headers={"Content-Type": "application/json"},
            text_exc=lambda: RuntimeError("text decode failed"),
        )
        session.queue_request(response)

        client = api.RESTClient(session, "user@example.com", "secret")

        with pytest.raises(aiohttp.ClientResponseError) as excinfo:
            await client._request("GET", "/api/v2/fail")

        err = excinfo.value
        message_attr = getattr(err, "message", None)
        if message_attr is not None:
            assert message_attr == "<no body>"
        else:
            assert "<no body>" in str(err)
        assert response.text_calls == 1

    asyncio.run(_run())

    # A 5xx is transient: exactly one WARNING line, no ERROR.
    failure_logs = [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING and record.name == api.__name__
    ]
    assert [r.levelno for r in failure_logs] == [logging.WARNING]
    assert "<no body>" in failure_logs[0].getMessage()


JSON = {"Content-Type": "application/json"}


class FakeClock:
    """Monotonic clock whose sleep advances time instantly."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay
        await asyncio.sleep(0)


class StampedSession(FakeSession):
    """FakeSession that records the fake time at which each call starts."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__()
        self.clock = clock
        self.starts: list[tuple[str, float]] = []

    def request(self, method, url, *args, **kwargs):
        self.starts.append((f"{method} {url}", self.clock.now))
        return super().request(method, url, *args, **kwargs)

    def post(self, url, *args, **kwargs):
        self.starts.append((f"POST {url}", self.clock.now))
        return super().post(url, *args, **kwargs)


def _token(value: str) -> MockResponse:
    return MockResponse(200, {"access_token": value, "expires_in": 3600}, headers=JSON)


def _ok(body: object | None = None) -> MockResponse:
    return MockResponse(200, body or {}, headers=JSON, text_data="{}")


def _client(
    monkeypatch: pytest.MonkeyPatch, cls: type[api.RESTClient] = api.RESTClient, **kw
) -> tuple[api.RESTClient, StampedSession, FakeClock]:
    """Build a client whose limiter uses the production spacing and a fake clock."""
    monkeypatch.undo()  # restore the real REST_MIN_INTERVAL_S (conftest zeroes it)
    clock = FakeClock()
    session = StampedSession(clock)
    client = cls(session, "user@example.com", "secret", **kw)
    client._limiter = api.RequestLimiter(clock=clock, sleep=clock.sleep)
    return client, session, clock


def _gaps(starts: list[tuple[str, float]]) -> list[float]:
    times = [t for _, t in starts]
    return [b - a for a, b in zip(times, times[1:], strict=False)]


def test_concurrent_requests_are_spaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """N concurrent calls (token POST included) start >= 0.5 s apart."""
    client, session, _ = _client(monkeypatch)
    session.queue_post(_token("T1"))
    session.queue_request(*[_ok({"y": 2020}) for _ in range(6)])

    async def _run() -> None:
        await asyncio.gather(*(client.get_rtc_time("dev") for _ in range(6)))

    asyncio.run(_run())

    assert len(session.starts) == 7  # 1 token POST + 6 GETs
    assert session.starts[0][0].startswith("POST ")
    assert all(gap >= 0.5 for gap in _gaps(session.starts))
    assert api.REST_MIN_INTERVAL_S == 0.5


def test_429_retry_after_blocks_following_requests(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A 429 with Retry-After holds every later request on the client."""
    client, session, clock = _client(monkeypatch)
    session.queue_post(_token("T1"))
    session.queue_request(
        MockResponse(429, {}, headers={**JSON, "Retry-After": "7"}, text_data="{}"),
        _ok({"y": 2020}),
    )

    async def _run() -> None:
        headers = await client.authed_headers()
        with pytest.raises(api.BackendRateLimitError) as excinfo:
            await client._request("GET", "/a", headers=dict(headers))
        assert excinfo.value.retry_after == 7.0
        limited_at = clock.now
        await client.get_rtc_time("dev")
        assert session.starts[-1][1] >= limited_at + 7.0

    with caplog.at_level(logging.DEBUG, logger=api.__name__):
        asyncio.run(_run())

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_429_without_retry_after_uses_default_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare 429 still backs off by the default pause."""
    client, session, clock = _client(monkeypatch)
    session.queue_post(_token("T1"))
    session.queue_request(
        MockResponse(429, {}, headers=JSON, text_data="{}"), _ok({"y": 1})
    )

    async def _run() -> None:
        headers = await client.authed_headers()
        with pytest.raises(api.BackendRateLimitError) as excinfo:
            await client._request("GET", "/a", headers=dict(headers))
        assert excinfo.value.retry_after == api.RATE_LIMIT_DEFAULT_PAUSE_S
        limited_at = clock.now
        await client.get_rtc_time("dev")
        assert session.starts[-1][1] >= limited_at + api.RATE_LIMIT_DEFAULT_PAUSE_S

    asyncio.run(_run())


def test_token_429_pauses_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 from the token endpoint pauses the client and carries retry_after."""
    client, session, clock = _client(monkeypatch)
    session.queue_post(
        MockResponse(429, {}, headers={**JSON, "Retry-After": "12"}, text_data=""),
        _token("T1"),
    )

    async def _run() -> None:
        with pytest.raises(api.BackendRateLimitError) as excinfo:
            await client._ensure_token()
        assert excinfo.value.retry_after == 12.0
        limited_at = clock.now
        assert await client._ensure_token() == "T1"
        assert session.starts[-1][1] >= limited_at + 12.0

    asyncio.run(_run())


def test_segmented_write_after_401_fetches_one_token(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Four stale-token requests after server-side expiry cost one token POST."""
    client, session, _ = _client(
        monkeypatch, DucaheatRESTClient, api_base=api.DUCAHEAT_API_BASE
    )
    session.queue_post(_token("T1"), _token("T2"), _token("T3"), _token("T4"))

    def _unauth() -> MockResponse:
        return MockResponse(401, {"e": 1}, headers=JSON, text_data='{"e":1}')

    # The server rejects T1 for the prog GET (slot-resolution echo) and for
    # each segment (status setpoint, prog, status presets).
    session.queue_request(
        _unauth(), _ok(), _unauth(), _ok(), _unauth(), _ok(), _unauth(), _ok()
    )

    async def _run() -> None:
        await client.set_node_settings(
            "dev", ("htr", "1"), stemp=21, prog=[0] * 168, ptemp=[5, 17, 21]
        )

    with caplog.at_level(logging.DEBUG, logger=api.__name__):
        asyncio.run(_run())

    assert len(session.post_calls) == 2  # initial token + exactly one refresh
    assert len(session.request_calls) == 8
    retried = [c for c in session.request_calls[1::2]]
    assert all(c[2]["headers"]["Authorization"] == "Bearer T2" for c in retried)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert all(gap >= 0.5 for gap in _gaps(session.starts))


def test_401_after_reauth_logs_one_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A 401 that survives re-auth raises BackendAuthError with one ERROR line."""
    session = FakeSession()
    session.queue_post(_token("T1"), _token("T2"))
    session.queue_request(
        MockResponse(401, {}, headers=JSON, text_data="{}"),
        MockResponse(401, {}, headers=JSON, text_data="{}"),
    )
    client = api.RESTClient(session, "user@example.com", "secret")

    async def _run() -> None:
        headers = await client.authed_headers()
        with pytest.raises(api.BackendAuthError):
            await client._request("GET", "/a", headers=dict(headers))

    with caplog.at_level(logging.DEBUG, logger=api.__name__):
        asyncio.run(_run())

    assert [r.levelno for r in caplog.records if r.levelno >= logging.WARNING] == [
        logging.ERROR
    ]


def test_ignored_status_and_client_error_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ignored statuses log at DEBUG; a non-ignored 4xx logs one ERROR."""
    session = FakeSession()
    session.queue_request(
        MockResponse(404, {}, headers=JSON, text_data="nope"),
        MockResponse(404, {}, headers=JSON, text_data="nope"),
    )
    client = api.RESTClient(session, "user@example.com", "secret")

    async def _run() -> None:
        assert await client._request("GET", "/a", ignore_statuses=(404,)) is None
        with pytest.raises(api.aiohttp.ClientResponseError):
            await client._request("GET", "/a")

    with caplog.at_level(logging.DEBUG, logger=api.__name__):
        asyncio.run(_run())

    levels = [r.levelno for r in caplog.records if r.levelno >= logging.WARNING]
    assert levels == [logging.ERROR]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("  ", None),
        ("soon", None),
        ("nan", None),
        ("5", 5.0),
        (" 2.5 ", 2.5),
        ("-3", 0.0),
        ("999999", api.RATE_LIMIT_MAX_PAUSE_S),
        ("Sat, 10 Oct 2026 12:00:30 GMT", 30.0),
        ("Sat, 10 Oct 2026 12:00:30 -0000", 30.0),
        ("Sat, 10 Oct 2026 11:00:00 GMT", 0.0),
    ],
)
def test_parse_retry_after(value: str | None, expected: float | None) -> None:
    """Retry-After accepts delta seconds or an HTTP date and is clamped."""
    now = datetime(2026, 10, 10, 12, 0, 0, tzinfo=UTC)
    assert api.parse_retry_after(value, now=now) == expected


def test_parse_retry_after_defaults_to_wall_clock() -> None:
    """Without ``now`` an HTTP date in the past yields zero."""
    assert api.parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0


def test_limiter_pause_never_shortens() -> None:
    """A shorter pause does not cut an existing longer one."""
    clock = FakeClock()
    limiter = api.RequestLimiter(0.5, clock=clock, sleep=clock.sleep)
    limiter.pause(10)
    limiter.pause(2)

    asyncio.run(limiter.acquire())

    assert clock.sleeps == [10.0]


def _geo_client(*responses: object) -> tuple[RESTClient, FakeSession]:
    """Return a client whose session answers the token POST and ``responses``."""
    session = FakeSession()
    session.queue_post(
        MockResponse(200, {"access_token": "tok", "expires_in": 3600}, headers=JSON)
    )
    session.queue_request(*responses)
    return RESTClient(session, "user", "pass"), session


async def test_geo_data_is_read_from_the_gateway_endpoint() -> None:
    """A JSON body becomes GeoData; the request names the gateway."""
    body = {
        "country": "Testland",
        "state": "North",
        "city": "Sampleton",
        "tz_code": "Europe/X",
        "zip": "12345",
    }
    client, session = _geo_client(MockResponse(200, body, headers=JSON))

    result = await client.get_geo_data("0123456789abcdef")

    assert result == GeoData(**body)
    [(method, url, _kwargs)] = session.request_calls
    assert method == "GET"
    assert url.endswith("/api/v2/devs/0123456789abcdef/geo_data")


@pytest.mark.parametrize(
    "response",
    [
        MockResponse(404, {}, headers=JSON, text_data="not found"),
        MockResponse(200, ["unexpected"], headers=JSON),
        ClientError("network down"),
    ],
    ids=["404", "not-a-mapping", "network-error"],
)
async def test_geo_data_is_none_when_unavailable(response: object) -> None:
    """A missing endpoint, odd body or network error gives None, never an error."""
    client, _session = _geo_client(response)

    assert await client.get_geo_data("0123456789abcdef") is None
