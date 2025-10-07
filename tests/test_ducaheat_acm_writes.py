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


def test_ducaheat_acm_mode_boost_includes_duration_and_metadata(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness(responses=[{"ok": True}])

        result = await harness.client.set_node_settings(
            "dev", ("acm", "9"), mode="boost", boost_time=45
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
                    "json": {"mode": "boost", "boost_time": 45},
                },
            )
        ]
        assert harness.rtc_calls == ["dev"]
        boost_state = result.get("boost_state")
        assert boost_state["boost_active"] is True
        assert boost_state["boost_minutes_delta"] == 45
        assert boost_state["boost_end_day"] == 1
        assert boost_state["boost_end_min"] == 45

    asyncio.run(_run())


def test_ducaheat_acm_mode_cancel_posts_status(
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
        assert harness.rtc_calls == ["dev"]
        boost_state = result.get("boost_state")
        assert boost_state["boost_active"] is False
        assert boost_state["boost_end"] is None
        assert boost_state["boost_minutes_delta"] == 0

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

    monkeypatch.setattr(client, "_authed_headers", fake_headers)
    monkeypatch.setattr(client, "_post_acm_endpoint", fake_post)
    monkeypatch.setattr(client, "get_rtc_time", rtc_mock)

    result = await client.set_acm_boost_state("dev", "5", boost=False)
    assert result["response"] == "ok"
    assert result["boost_state"]["boost_active"] is False
    rtc_mock.assert_awaited_once_with("dev")


@pytest.mark.parametrize(
    ("boost", "boost_time", "expected_payload"),
    [
        (True, 45, {"boost": True, "boost_time": 45}),
        (False, None, {"boost": False}),
    ],
)
@pytest.mark.asyncio
async def test_ducaheat_set_acm_boost_state_posts_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
    boost: bool,
    boost_time: int | None,
    expected_payload: dict[str, Any],
) -> None:
    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
    headers = {
        "Authorization": "Bearer token",
        "X-SerialId": "15",
        "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
    }
    headers_mock = AsyncMock(return_value=headers)
    monkeypatch.setattr(client, "_authed_headers", headers_mock)

    post_mock = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(client, "_post_acm_endpoint", post_mock)

    base_rtc = {"y": 2024, "n": 5, "d": 10, "h": 12, "m": 0, "s": 0}
    rtc_mock = AsyncMock(return_value=base_rtc)
    monkeypatch.setattr(client, "get_rtc_time", rtc_mock)

    result = await client.set_acm_boost_state(
        "dev", 5, boost=boost, boost_time=boost_time
    )

    post_mock.assert_awaited_once_with(
        "/api/v2/devs/dev/acm/5/status",
        headers,
        expected_payload,
        dev_id="dev",
        addr="5",
    )
    headers_mock.assert_awaited_once()
    rtc_mock.assert_awaited_once_with("dev")

    metadata = result["boost_state"]
    assert metadata["boost_active"] is boost

    if boost:
        assert boost_time is not None
        assert metadata["boost_minutes_delta"] == boost_time
        expected_end = datetime(2024, 5, 10, 12, 0) + timedelta(minutes=boost_time)
        assert metadata["boost_end_day"] == expected_end.timetuple().tm_yday
        assert metadata["boost_end_min"] == expected_end.hour * 60 + expected_end.minute
        assert metadata["boost_end"] == {
            "day": expected_end.timetuple().tm_yday,
            "minute": expected_end.hour * 60 + expected_end.minute,
        }
        assert metadata["boost_end_timestamp"] == expected_end.isoformat()
    else:
        assert metadata["boost_minutes_delta"] == 0
        assert metadata["boost_end"] is None
        assert metadata["boost_end_day"] is None
        assert metadata["boost_end_min"] is None
        assert metadata["boost_end_timestamp"] is None


@pytest.mark.parametrize(
    ("boost", "boost_time"),
    [
        (True, 30),
        (False, None),
    ],
)
@pytest.mark.asyncio
async def test_ducaheat_set_acm_boost_state_client_error(
    monkeypatch: pytest.MonkeyPatch,
    boost: bool,
    boost_time: int | None,
) -> None:
    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    headers = {
        "Authorization": "Bearer token",
        "X-SerialId": "15",
        "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
    }
    headers_mock = AsyncMock(return_value=headers)
    monkeypatch.setattr(client, "_authed_headers", headers_mock)

    async def failing_post_segmented(*args: Any, **kwargs: Any) -> Any:
        raise ClientResponseError(
            request_info=None,
            history=(),
            status=422,
            message="bad request",
        )

    monkeypatch.setattr(client, "_post_segmented", failing_post_segmented)

    rtc_mock = AsyncMock(return_value={"ignored": True})
    monkeypatch.setattr(client, "get_rtc_time", rtc_mock)

    with pytest.raises(DucaheatRequestError) as exc:
        await client.set_acm_boost_state(
            "dev", 7, boost=boost, boost_time=boost_time
        )

    assert exc.value.status == 422
    assert "bad request" in str(exc.value)
    rtc_mock.assert_not_awaited()


def test_ducaheat_collect_boost_metadata_rtc_failure_active(
    monkeypatch: pytest.MonkeyPatch,
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness(responses=[{"ok": True}])

        async def failing_rtc(dev_id: str) -> dict[str, Any]:
            raise RuntimeError("boom")

        monkeypatch.setattr(harness.client, "get_rtc_time", failing_rtc)

        result = await harness.client.set_node_settings(
            "dev", ("acm", "9"), mode="boost", boost_time=30
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
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness(responses=[{"ok": True}])

        async def bad_rtc(dev_id: str) -> Any:
            return "bad"

        monkeypatch.setattr(harness.client, "get_rtc_time", bad_rtc)

        result = await harness.client.set_node_settings(
            "dev", ("acm", "9"), mode="boost", boost_time=30
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
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness(responses=[{"ok": True}])

        async def bad_rtc(dev_id: str) -> dict[str, Any]:
            return {"y": 2024, "n": 13, "d": 1, "h": 0, "m": 0, "s": 0}

        monkeypatch.setattr(harness.client, "get_rtc_time", bad_rtc)

        result = await harness.client.set_node_settings(
            "dev", ("acm", "9"), mode="boost", boost_time=30
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
                    "json": {"mode": "boost", "boost_time": 30},
                },
            )
        ]
        metadata = result["boost_state"]
        assert metadata["boost_end"] is None
        assert metadata["boost_minutes_delta"] == 30

    asyncio.run(_run())
