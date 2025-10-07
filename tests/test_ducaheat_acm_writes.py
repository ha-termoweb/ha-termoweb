from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientResponseError

from custom_components.termoweb.backend.ducaheat import (
    DucaheatRESTClient,
    DucaheatRequestError,
)
from custom_components.termoweb.const import BRAND_DUCAHEAT, get_brand_user_agent
from homeassistant.components.climate import HVACMode


def _setup_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    responses: list[dict[str, Any] | None] | None = None,
) -> tuple[
    DucaheatRESTClient,
    list[tuple[str, str, dict[str, Any]]],
    list[str],
]:
    """Create a REST client with fake request handling."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
    responses = list(responses or [])
    calls: list[tuple[str, str, dict[str, Any]]] = []
    rtc_calls: list[str] = []

    headers = {
        "Authorization": "Bearer token",
        "X-SerialId": "15",
        "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
    }

    def _hvac_mode_str(self: HVACMode) -> str:
        """Return the enum value for consistent serialization."""

        return str(self.value)

    monkeypatch.setattr(HVACMode, "__str__", _hvac_mode_str, raising=False)

    async def fake_headers() -> dict[str, str]:
        return dict(headers)

    async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((method, path, kwargs))
        if responses:
            return responses.pop(0) or {}
        return {}

    async def fake_rtc(dev_id: str) -> dict[str, Any]:
        rtc_calls.append(dev_id)
        return {"y": 2024, "n": 1, "d": 1, "h": 0, "m": 0, "s": 0}
    monkeypatch.setattr(client, "_authed_headers", fake_headers)
    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr(client, "get_rtc_time", fake_rtc)
    return client, calls, rtc_calls


def test_ducaheat_acm_mode_boost_includes_duration_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        client, calls, rtc_calls = _setup_client(monkeypatch, responses=[{"ok": True}])

        result = await client.set_node_settings(
            "dev", ("acm", "9"), mode="boost", boost_time=45
        )

        assert calls == [
            (
                "POST",
                "/api/v2/devs/dev/acm/9/mode",
                {
                    "headers": {
                        "Authorization": "Bearer token",
                        "X-SerialId": "15",
                        "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
                    },
                    "json": {"mode": "boost", "boost_time": 45},
                },
            )
        ]
        assert rtc_calls == ["dev"]
        boost_state = result.get("boost_state")
        assert boost_state["boost_active"] is True
        assert boost_state["boost_minutes_delta"] == 45
        assert boost_state["boost_end_day"] == 1
        assert boost_state["boost_end_min"] == 45

    asyncio.run(_run())


def test_ducaheat_acm_mode_cancel_posts_status(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client, calls, rtc_calls = _setup_client(monkeypatch, responses=[{"ok": True}])

        result = await client.set_node_settings("dev", ("acm", "9"), mode=HVACMode.AUTO)

        assert calls == [
            (
                "POST",
                "/api/v2/devs/dev/acm/9/status",
                {
                    "headers": {
                        "Authorization": "Bearer token",
                        "X-SerialId": "15",
                        "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
                    },
                    "json": {"boost": False},
                },
            ),
            (
                "POST",
                "/api/v2/devs/dev/acm/9/mode",
                {
                    "headers": {
                        "Authorization": "Bearer token",
                        "X-SerialId": "15",
                        "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
                    },
                    "json": {"mode": "auto"},
                },
            ),
        ]
        assert rtc_calls == ["dev"]
        boost_state = result.get("boost_state")
        assert boost_state["boost_active"] is False
        assert boost_state["boost_end"] is None
        assert boost_state["boost_minutes_delta"] == 0

    asyncio.run(_run())


def test_ducaheat_acm_set_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client, calls, rtc_calls = _setup_client(
            monkeypatch, responses=[{"status": "ok"}]
        )

        await client.set_node_settings("dev", ("acm", "2"), stemp=19.5, units="c")

        assert calls == [
            (
                "POST",
                "/api/v2/devs/dev/acm/2/status",
                {
                    "headers": {
                        "Authorization": "Bearer token",
                        "X-SerialId": "15",
                        "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
                    },
                    "json": {"stemp": "19.5", "units": "C"},
                },
            )
        ]
        assert rtc_calls == []

    asyncio.run(_run())


def test_ducaheat_acm_program_write(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client, calls, rtc_calls = _setup_client(
            monkeypatch, responses=[{"saved": True}]
        )
        prog = [0, 1, 2] * 56

        await client.set_node_settings("dev", ("acm", "4"), prog=list(prog))

        assert calls == [
            (
                "POST",
                "/api/v2/devs/dev/acm/4/prog",
                {
                    "headers": {
                        "Authorization": "Bearer token",
                        "X-SerialId": "15",
                        "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
                    },
                    "json": {"prog": list(prog)},
                },
            )
        ]
        assert rtc_calls == []

        with pytest.raises(ValueError):
            await client.set_node_settings("dev", ("acm", "4"), prog=[0] * 24)

    asyncio.run(_run())


def test_ducaheat_acm_program_temps(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client, calls, rtc_calls = _setup_client(
            monkeypatch, responses=[{"saved": True}]
        )

        await client.set_node_settings(
            "dev", ("acm", "8"), ptemp=[18.0, 20.0, 22.0]
        )

        assert calls == [
            (
                "POST",
                "/api/v2/devs/dev/acm/8/prog_temps",
                {
                    "headers": {
                        "Authorization": "Bearer token",
                        "X-SerialId": "15",
                        "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
                    },
                    "json": {"ptemp": ["18.0", "20.0", "22.0"]},
                },
            )
        ]
        assert rtc_calls == []

    asyncio.run(_run())


def test_ducaheat_acm_request_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {
                "Authorization": "Bearer token",
                "X-SerialId": "15",
                "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
            }

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            raise ClientResponseError(
                request_info=None,
                history=(),
                status=400,
                message="malformed",
            )

        monkeypatch.setattr(client, "_authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)
        mock_rtc = AsyncMock(return_value={})
        monkeypatch.setattr(client, "get_rtc_time", mock_rtc)

        with pytest.raises(DucaheatRequestError) as exc:
            await client.set_node_settings("dev", ("acm", "1"), mode="boost")

        assert "malformed" in str(exc.value)
        mock_rtc.assert_not_awaited()

    asyncio.run(_run())


def test_ducaheat_acm_mode_invalid_boost_time(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client, _, _ = _setup_client(monkeypatch)

        with pytest.raises(ValueError):
            await client.set_node_settings(
                "dev", ("acm", "2"), mode="auto", boost_time=15
            )

    asyncio.run(_run())


def test_ducaheat_acm_mode_boost_invalid_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client, _, _ = _setup_client(monkeypatch)

        with pytest.raises(ValueError):
            await client.set_node_settings(
                "dev", ("acm", "2"), mode="boost", boost_time=0
            )

    asyncio.run(_run())


def test_ducaheat_acm_mode_boost_invalid_minutes_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        client, _, _ = _setup_client(monkeypatch)

        with pytest.raises(ValueError):
            await client.set_node_settings(
                "dev", ("acm", "2"), mode="boost", boost_time="abc"
            )

    asyncio.run(_run())


def test_ducaheat_collect_boost_metadata_rtc_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        client, calls, _ = _setup_client(
            monkeypatch, responses=[{"ok": True}, {"ok": True}]
        )

        async def failing_rtc(dev_id: str) -> dict[str, Any]:
            raise RuntimeError("boom")

        monkeypatch.setattr(client, "get_rtc_time", failing_rtc)

        result = await client.set_node_settings("dev", ("acm", "9"), mode="auto")
        assert len(calls) == 2
        metadata = result["boost_state"]
        assert metadata["boost_active"] is False
        assert metadata["boost_end"] is None
        assert metadata["boost_minutes_delta"] == 0

    asyncio.run(_run())


def test_ducaheat_collect_boost_metadata_invalid_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        client, calls, _ = _setup_client(
            monkeypatch, responses=[{"ok": True}, {"ok": True}]
        )

        async def bad_rtc(dev_id: str) -> dict[str, Any]:
            return {"y": "bad"}

        monkeypatch.setattr(client, "get_rtc_time", bad_rtc)

        result = await client.set_node_settings("dev", ("acm", "9"), mode="auto")
        assert len(calls) == 2
        metadata = result["boost_state"]
        assert metadata["boost_end"] is None
        assert metadata["boost_minutes_delta"] == 0

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_ducaheat_set_acm_boost_state_non_dict_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_headers() -> dict[str, str]:
        return {"Authorization": "Bearer"}

    async def fake_post(*args: Any, **kwargs: Any) -> Any:
        return "ok"

    rtc_mock = AsyncMock(return_value={"y": 2024, "n": 1, "d": 1, "h": 0, "m": 0, "s": 0})

    monkeypatch.setattr(client, "_authed_headers", fake_headers)
    monkeypatch.setattr(client, "_post_acm_endpoint", fake_post)
    monkeypatch.setattr(client, "get_rtc_time", rtc_mock)

    result = await client.set_acm_boost_state("dev", "5", boost=False)
    assert result["response"] == "ok"
    assert result["boost_state"]["boost_active"] is False
    rtc_mock.assert_awaited_once_with("dev")


def test_ducaheat_collect_boost_metadata_rtc_failure_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        client, calls, _ = _setup_client(
            monkeypatch, responses=[{"ok": True}]
        )

        async def failing_rtc(dev_id: str) -> dict[str, Any]:
            raise RuntimeError("boom")

        monkeypatch.setattr(client, "get_rtc_time", failing_rtc)

        result = await client.set_node_settings(
            "dev", ("acm", "9"), mode="boost", boost_time=30
        )
        assert calls == [
            (
                "POST",
                "/api/v2/devs/dev/acm/9/mode",
                {
                    "headers": {
                        "Authorization": "Bearer token",
                        "X-SerialId": "15",
                        "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
                    },
                    "json": {"mode": "boost", "boost_time": 30},
                },
            )
        ]
        metadata = result["boost_state"]
        assert metadata["boost_end"] is None
        assert metadata["boost_minutes_delta"] == 30

    asyncio.run(_run())


def test_ducaheat_collect_boost_metadata_invalid_payload_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        client, calls, _ = _setup_client(
            monkeypatch, responses=[{"ok": True}]
        )

        async def bad_rtc(dev_id: str) -> Any:
            return "bad"

        monkeypatch.setattr(client, "get_rtc_time", bad_rtc)

        result = await client.set_node_settings(
            "dev", ("acm", "9"), mode="boost", boost_time=30
        )
        assert calls == [
            (
                "POST",
                "/api/v2/devs/dev/acm/9/mode",
                {
                    "headers": {
                        "Authorization": "Bearer token",
                        "X-SerialId": "15",
                        "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
                    },
                    "json": {"mode": "boost", "boost_time": 30},
                },
            )
        ]
        metadata = result["boost_state"]
        assert metadata["boost_end"] is None
        assert metadata["boost_minutes_delta"] == 30

    asyncio.run(_run())


def test_ducaheat_collect_boost_metadata_invalid_datetime_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        client, calls, _ = _setup_client(
            monkeypatch, responses=[{"ok": True}]
        )

        async def bad_rtc(dev_id: str) -> dict[str, Any]:
            return {"y": 2024, "n": 13, "d": 1, "h": 0, "m": 0, "s": 0}

        monkeypatch.setattr(client, "get_rtc_time", bad_rtc)

        result = await client.set_node_settings(
            "dev", ("acm", "9"), mode="boost", boost_time=30
        )
        assert calls == [
            (
                "POST",
                "/api/v2/devs/dev/acm/9/mode",
                {
                    "headers": {
                        "Authorization": "Bearer token",
                        "X-SerialId": "15",
                        "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
                    },
                    "json": {"mode": "boost", "boost_time": 30},
                },
            )
        ]
        metadata = result["boost_state"]
        assert metadata["boost_end"] is None
        assert metadata["boost_minutes_delta"] == 30

    asyncio.run(_run())
