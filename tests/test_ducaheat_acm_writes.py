from __future__ import annotations

import asyncio
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

        monkeypatch.setattr(client, "authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)
        mock_rtc = AsyncMock(return_value={})
        monkeypatch.setattr(client, "get_rtc_time", mock_rtc)

        with pytest.raises(DucaheatRequestError) as exc:
            await client.set_node_settings("dev", ("acm", "1"), mode="boost")

        assert "malformed" in str(exc.value)
        mock_rtc.assert_not_awaited()

    asyncio.run(_run())


def test_ducaheat_acm_mode_invalid_boost_time(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness()

        with pytest.raises(ValueError):
            await harness.client.set_node_settings(
                "dev", ("acm", "2"), mode="auto", boost_time=15
            )

    asyncio.run(_run())


def test_ducaheat_acm_mode_boost_invalid_minutes(
    ducaheat_rest_harness: Callable[..., Any],
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


@pytest.mark.asyncio
async def test_ducaheat_acm_extra_options_segmented_post(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Ensure extra options payload uses segmented POST with formatted fields."""

    harness = ducaheat_rest_harness()

    result = await harness.client.set_acm_extra_options(
        "dev", "3", boost_time=180, boost_temp=55.55
    )

    assert result == {"ok": True}
    setup_calls = [
        call for call in harness.segmented_calls if call["path"].endswith("/setup")
    ]
    assert len(setup_calls) == 1
    setup_payload = setup_calls[0]["payload"]
    assert setup_payload == {"extra_options": {"boost_time": 180, "boost_temp": "55.5"}}


@pytest.mark.asyncio
async def test_ducaheat_acm_settings_boost_flow(
    ducaheat_rest_harness: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover the boost branch of segmented ACM writes."""

    harness = ducaheat_rest_harness()

    monkeypatch.setattr(harness.client, "_ensure_units", lambda units: units.upper())

    responses = await harness.client.set_node_settings(
        "dev",
        ("acm", "6"),
        mode="boost",
        stemp=22,
        prog=[1] * 168,
        ptemp=[10.0, 15.0, 20.0],
        units="c",
        boost_time=60,
    )

    assert responses.keys() == {"status", "prog"}
    status_call = next(
        call for call in harness.segmented_calls if call["path"].endswith("/status")
    )
    assert status_call["payload"] == {
        "stemp": "22.0",
        "units": "C",
        "mode": "boost",
        "boost_time": 60,
    }
    assert not any(call["path"].endswith("/mode") for call in harness.segmented_calls)


@pytest.mark.asyncio
async def test_ducaheat_acm_settings_cancel_only(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Validate the standalone cancel boost path without status payload."""

    harness = ducaheat_rest_harness()

    responses = await harness.client.set_node_settings(
        "dev",
        ("acm", "8"),
        cancel_boost=True,
    )

    assert responses == {"boost": {"ok": True}}


@pytest.mark.asyncio
async def test_ducaheat_acm_settings_cancel_with_units(
    ducaheat_rest_harness: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel requests with explicit units should emit a status payload."""

    harness = ducaheat_rest_harness()

    responses = await harness.client.set_node_settings(
        "dev",
        ("acm", "16"),
        units="F",
        cancel_boost=True,
    )

    assert responses.keys() == {"status", "boost"}
    status_call = next(
        call
        for call in harness.segmented_calls
        if call["path"].endswith("/status") and call["addr"] == "16"
    )
    assert status_call["payload"] == {"units": "F"}


@pytest.mark.asyncio
async def test_ducaheat_acm_settings_boost_mode_segment(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Ensure boost mode writes include a dedicated mode payload."""

    harness = ducaheat_rest_harness()

    responses = await harness.client.set_node_settings(
        "dev",
        ("acm", "9"),
        mode="boost",
        boost_time=120,
    )

    assert "mode" in responses
    mode_call = next(
        call for call in harness.segmented_calls if call["path"].endswith("/mode")
    )
    assert mode_call["payload"] == {"mode": "boost", "boost_time": 120}


@pytest.mark.asyncio
async def test_post_acm_endpoint_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Client errors should translate into ``DucaheatRequestError`` instances."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_post_segmented(path: str, **_: Any) -> None:
        raise ClientResponseError(
            request_info=None, history=(), status=422, message="bad request"
        )

    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    with pytest.raises(DucaheatRequestError) as err:
        await client._post_acm_endpoint(
            "/api/v2/devs/dev/acm/1/status",
            {"Authorization": "token"},
            {"mode": "auto"},
            dev_id="dev",
            addr="1",
        )

    assert "bad request" in str(err.value)


@pytest.mark.asyncio
async def test_post_acm_endpoint_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server errors should bubble up without translation."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_post_segmented(path: str, **_: Any) -> None:
        raise ClientResponseError(
            request_info=None, history=(), status=500, message="boom"
        )

    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    with pytest.raises(ClientResponseError):
        await client._post_acm_endpoint(
            "/api/v2/devs/dev/acm/1/status",
            {"Authorization": "token"},
            {"mode": "auto"},
            dev_id="dev",
            addr="1",
        )


@pytest.mark.asyncio
async def test_set_node_display_select_posts_to_select_segment(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Display flash writes should target the segmented /select endpoint."""

    harness = ducaheat_rest_harness()

    await harness.client.set_node_display_select("dev", ("acm", "1"), select=True)

    assert harness.segmented_calls[-1]["path"] == "/api/v2/devs/dev/acm/1/select"
    assert harness.segmented_calls[-1]["payload"] == {"select": True}


@pytest.mark.asyncio
async def test_select_segmented_node_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection failures should surface as ``DucaheatRequestError``."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_post_segmented(path: str, **_: Any) -> None:
        raise ClientResponseError(
            request_info=None, history=(), status=409, message="conflict"
        )

    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    with pytest.raises(DucaheatRequestError) as err:
        await client._select_segmented_node(
            dev_id="dev",
            node_type="acm",
            addr="1",
            headers={"Authorization": "token"},
            select=True,
        )

    assert "conflict" in str(err.value)


@pytest.mark.asyncio
async def test_select_segmented_node_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server-side failures should propagate to the caller."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_post_segmented(path: str, **_: Any) -> None:
        raise ClientResponseError(
            request_info=None, history=(), status=502, message="upstream"
        )

    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    with pytest.raises(ClientResponseError):
        await client._select_segmented_node(
            dev_id="dev",
            node_type="acm",
            addr="1",
            headers={"Authorization": "token"},
            select=True,
        )


@pytest.mark.asyncio
async def test_set_acm_boost_state_invalid_stemp(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Invalid stemp values should raise immediately."""

    harness = ducaheat_rest_harness()

    with pytest.raises(ValueError) as err:
        await harness.client.set_acm_boost_state("dev", "12", boost=True, stemp="bad")

    assert "Invalid stemp value" in str(err.value)


def test_ensure_units_blank_defaults() -> None:
    """Empty unit strings should normalise to Celsius."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    assert client._ensure_units("") == "C"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_boost", [False, True])
async def test_acm_mode_write_makes_no_rtc_get_and_no_extra_boost_write(
    ducaheat_rest_harness: Callable[..., Any], cancel_boost: bool
) -> None:
    """A plain acm mode write must not GET the RTC or POST an unrequested boost=false.

    Regression: a failing RTC GET used to trigger a ``status_refresh`` POST of
    ``{"boost": false}``, which cancels a running boost.
    """

    harness = ducaheat_rest_harness()
    rtc = AsyncMock(side_effect=RuntimeError("rtc down"))
    harness.client.get_rtc_time = rtc

    await harness.client.set_node_settings(
        "dev", ("acm", "7"), mode="auto", cancel_boost=cancel_boost
    )

    rtc.assert_not_awaited()
    assert all(method != "GET" for method, _path, _kw in harness.requests)
    # Only an explicitly requested cancel may touch /boost.
    assert [call["path"].rsplit("/", 1)[-1] for call in harness.segmented_calls] == (
        ["mode", "boost"] if cancel_boost else ["mode"]
    )


@pytest.mark.asyncio
async def test_acm_boost_state_write_makes_no_rtc_get(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Starting a boost posts once to /boost and returns the server response."""

    harness = ducaheat_rest_harness()
    rtc = AsyncMock(side_effect=RuntimeError("rtc down"))
    harness.client.get_rtc_time = rtc

    result = await harness.client.set_acm_boost_state(
        "dev", "4", boost=True, boost_time=120, stemp=21.5, units="C"
    )

    rtc.assert_not_awaited()
    assert result == {"ok": True}
    assert [call["path"] for call in harness.segmented_calls] == [
        "/api/v2/devs/dev/acm/4/boost"
    ]


@pytest.mark.asyncio
async def test_acm_boost_with_stemp_sends_boost_time(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """mode=boost with stemp and boost_time must send both fields (B7)."""

    harness = ducaheat_rest_harness()

    await harness.client.set_node_settings(
        "dev", ("acm", "6"), mode="boost", stemp=22, boost_time=60
    )

    assert [call["path"] for call in harness.segmented_calls] == [
        "/api/v2/devs/dev/acm/6/status"
    ]
    assert harness.segmented_calls[0]["payload"] == {
        "mode": "boost",
        "stemp": "22.0",
        "units": "C",
        "boost_time": 60,
    }


@pytest.mark.asyncio
async def test_acm_boost_with_stemp_validates_boost_time(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """An invalid boost_time is rejected even when stemp is also supplied."""

    harness = ducaheat_rest_harness()

    with pytest.raises(ValueError, match="boost_time"):
        await harness.client.set_node_settings(
            "dev", ("acm", "6"), mode="boost", stemp=22, boost_time=45
        )

    assert harness.segmented_calls == []
