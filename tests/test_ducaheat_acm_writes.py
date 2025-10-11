from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientResponseError

from custom_components.termoweb.backend.ducaheat import (
    DucaheatRESTClient,
    DucaheatRequestError,
)
from custom_components.termoweb.const import BRAND_DUCAHEAT, get_brand_user_agent
from homeassistant.components.climate import HVACMode




def test_ducaheat_acm_mode_cancel_when_boost_active(
    ducaheat_rest_harness: Callable[..., Any]
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness(responses=[{"ok": True}])

        result = await harness.client.set_node_settings(
            "dev", ("acm", "9"), mode=HVACMode.AUTO, cancel_boost=True
        )

        assert harness.requests == [
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
            (
                "POST",
                "/api/v2/devs/dev/acm/9/boost",
                {
                    "headers": {
                        "Authorization": "Bearer token",
                        "X-SerialId": "15",
                        "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
                    },
                    "json": {"boost": False},
                },
            ),
        ]
        assert harness.rtc_calls == ["dev"]
        boost_state = result.get("boost_state")
        assert boost_state["boost_active"] is False
        assert boost_state["boost_end"] is None
        assert boost_state["boost_minutes_delta"] == 0

    asyncio.run(_run())


def test_ducaheat_acm_mode_change_without_boost_skips_status(
    ducaheat_rest_harness: Callable[..., Any]
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness(responses=[{"ok": True}])

        result = await harness.client.set_node_settings(
            "dev", ("acm", "9"), mode=HVACMode.AUTO
        )

        assert harness.requests == [
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
        assert harness.rtc_calls == []
        assert result == {"mode": {"ok": True}}

    asyncio.run(_run())


def test_ducaheat_acm_set_temperature(
    ducaheat_rest_harness: Callable[..., Any]
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness(responses=[{"status": "ok"}])

        await harness.client.set_node_settings(
            "dev", ("acm", "2"), stemp=19.5, units="c"
        )

        assert harness.requests == [
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
        assert harness.rtc_calls == []

    asyncio.run(_run())


def test_ducaheat_acm_program_write(
    ducaheat_rest_harness: Callable[..., Any]
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness(responses=[{"saved": True}])
        prog = [0, 1, 2] * 56

        await harness.client.set_node_settings("dev", ("acm", "4"), prog=list(prog))

        assert harness.requests == [
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
        assert harness.rtc_calls == []

        with pytest.raises(ValueError):
            await harness.client.set_node_settings(
                "dev", ("acm", "4"), prog=[0] * 24
            )

    asyncio.run(_run())


def test_ducaheat_acm_program_temps(
    ducaheat_rest_harness: Callable[..., Any]
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness(responses=[{"saved": True}])

        await harness.client.set_node_settings(
            "dev", ("acm", "8"), ptemp=[18.0, 20.0, 22.0]
        )

        assert harness.requests == [
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
        assert harness.rtc_calls == []

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


def test_ducaheat_acm_mode_invalid_boost_time(
    ducaheat_rest_harness: Callable[..., Any]
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness()

        with pytest.raises(ValueError):
            await harness.client.set_node_settings(
                "dev", ("acm", "2"), mode="auto", boost_time=15
            )

    asyncio.run(_run())


def test_ducaheat_acm_mode_boost_invalid_minutes(
    ducaheat_rest_harness: Callable[..., Any]
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness()

        with pytest.raises(ValueError):
            await harness.client.set_node_settings(
                "dev", ("acm", "2"), mode="boost", boost_time=0
            )

    asyncio.run(_run())


def test_ducaheat_acm_mode_boost_invalid_minutes_type(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness()

        with pytest.raises(ValueError):
            await harness.client.set_node_settings(
                "dev", ("acm", "2"), mode="boost", boost_time="abc"
            )

    asyncio.run(_run())


def test_ducaheat_collect_boost_metadata_rtc_failure(
    monkeypatch: pytest.MonkeyPatch,
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness(responses=[{"ok": True}, {"ok": True}])

        async def failing_rtc(dev_id: str) -> dict[str, Any]:
            raise RuntimeError("boom")

        monkeypatch.setattr(harness.client, "get_rtc_time", failing_rtc)

        result = await harness.client.set_node_settings("dev", ("acm", "9"), mode="auto")
        assert len(harness.requests) == 2
        metadata = result["boost_state"]
        assert metadata["boost_active"] is False
        assert metadata["boost_end"] is None
        assert metadata["boost_minutes_delta"] == 0

    asyncio.run(_run())


def test_ducaheat_collect_boost_metadata_invalid_payload(
    monkeypatch: pytest.MonkeyPatch,
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness(responses=[{"ok": True}, {"ok": True}])

        async def bad_rtc(dev_id: str) -> dict[str, Any]:
            return {"y": "bad"}

        monkeypatch.setattr(harness.client, "get_rtc_time", bad_rtc)

        result = await harness.client.set_node_settings(
            "dev", ("acm", "9"), mode="auto"
        )
        assert len(harness.requests) == 2
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
    select_mock = AsyncMock(side_effect=[{"select": True}, {"select": False}])

    monkeypatch.setattr(client, "_authed_headers", fake_headers)
    monkeypatch.setattr(client, "_post_acm_endpoint", fake_post)
    monkeypatch.setattr(client, "_select_segmented_node", select_mock)
    monkeypatch.setattr(client, "get_rtc_time", rtc_mock)

    result = await client.set_acm_boost_state("dev", "5", boost=False)
    assert result["response"] == "ok"
    assert result["boost_state"]["boost_active"] is False
    rtc_mock.assert_awaited_once_with("dev")
    assert select_mock.await_count == 2
    first_call = select_mock.await_args_list[0]
    assert first_call.kwargs == {
        "dev_id": "dev",
        "node_type": "acm",
        "addr": "5",
        "headers": {"Authorization": "Bearer"},
        "select": True,
    }
    second_call = select_mock.await_args_list[1]
    assert second_call.kwargs == {
        "dev_id": "dev",
        "node_type": "acm",
        "addr": "5",
        "headers": {"Authorization": "Bearer"},
        "select": False,
    }


