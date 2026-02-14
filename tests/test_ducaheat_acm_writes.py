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
async def test_ducaheat_acm_boost_metadata_fallback(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Verify boost metadata falls back when RTC collection fails."""

    harness = ducaheat_rest_harness()
    harness.client.get_rtc_time = AsyncMock(side_effect=RuntimeError("rtc down"))

    result = await harness.client.set_acm_boost_state(
        "dev", "4", boost=True, boost_time=120
    )

    assert result == {
        "ok": True,
        "boost_state": {
            "boost_active": True,
            "_fallback": True,
            "boost_minutes_delta": 120,
            "boost_end_day": None,
            "boost_end_min": None,
            "boost_end_timestamp": None,
        },
    }

    boost_calls = [
        call for call in harness.segmented_calls if call["path"].endswith("/boost")
    ]
    assert len(boost_calls) == 1


@pytest.mark.asyncio
async def test_ducaheat_acm_boost_metadata_happy_path(
    ducaheat_rest_harness: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure boost writes validate minutes and merge metadata."""

    minutes_calls: list[int | None] = []

    def fake_validate(value: int | None) -> int:
        minutes_calls.append(value)
        return 180

    monkeypatch.setattr(
        "custom_components.termoweb.backend.sanitize.validate_boost_minutes",
        fake_validate,
    )
    monkeypatch.setattr(
        "custom_components.termoweb.backend.ducaheat.validate_boost_minutes",
        fake_validate,
    )

    harness = ducaheat_rest_harness(
        rtc_payload={"y": 2024, "n": 1, "d": 1, "h": 5, "m": 30, "s": 0}
    )

    result = await harness.client.set_acm_boost_state(
        "dev", "5", boost=True, boost_time=600, stemp=21.5, units="C"
    )

    assert minutes_calls == [600, 600]
    boost_call = next(
        call for call in harness.segmented_calls if call["path"].endswith("/boost")
    )
    assert boost_call["payload"] == {
        "boost": True,
        "boost_time": 180,
        "stemp": "21.5",
        "units": "C",
    }
    assert harness.rtc_calls == ["dev"]

    assert result["ok"] is True
    boost_state = result.get("boost_state")
    assert boost_state == {
        "boost_active": True,
        "boost_end_day": 1,
        "boost_end_min": 510,
        "boost_minutes_delta": 180,
        "boost_end_timestamp": "2024-01-01T08:30:00",
    }


@pytest.mark.asyncio
async def test_ducaheat_acm_settings_boost_flow(
    ducaheat_rest_harness: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover the boost branch of segmented ACM writes including metadata collection."""

    harness = ducaheat_rest_harness()

    monkeypatch.setattr(harness.client, "_format_temp", lambda value: "22.0")
    monkeypatch.setattr(harness.client, "_ensure_units", lambda units: units.upper())
    monkeypatch.setattr(harness.client, "_ensure_prog", lambda prog: list(prog))
    monkeypatch.setattr(
        harness.client,
        "_ensure_ptemp",
        lambda values: ("10.0", "15.0", "20.0"),
    )

    responses = await harness.client.set_node_settings(
        "dev",
        ("acm", "6"),
        mode="boost",
        stemp=19.4,
        prog=[1] * 168,
        ptemp=[10.0, 15.0, 20.0],
        units="c",
        boost_time=45,
    )

    assert {"status", "prog", "prog_temps", "boost_state"} <= responses.keys()
    status_call = next(
        call for call in harness.segmented_calls if call["path"].endswith("/status")
    )
    assert status_call["payload"] == {
        "stemp": "22.0",
        "units": "C",
        "mode": "boost",
    }
    assert not any(call["path"].endswith("/mode") for call in harness.segmented_calls)
    assert responses["boost_state"]["boost_active"] is True


@pytest.mark.asyncio
async def test_ducaheat_acm_settings_mode_only_collects_custom_rtc(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Ensure non-boost mode writes run metadata collection when RTC is patched."""

    harness = ducaheat_rest_harness()

    async def custom_rtc(dev_id: str) -> dict[str, Any]:
        return {"y": 2024, "n": 2, "d": 3, "h": 4, "m": 5, "s": 6}

    harness.client.get_rtc_time = custom_rtc

    responses = await harness.client.set_node_settings(
        "dev", ("acm", "17"), mode="auto"
    )

    assert "mode" in responses
    assert responses["boost_state"]["boost_active"] is False


@pytest.mark.asyncio
async def test_ducaheat_acm_settings_cancel_boost_status_refresh(
    ducaheat_rest_harness: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure cancel boost requests emit status, mode, boost, and refresh calls."""

    harness = ducaheat_rest_harness()

    monkeypatch.setattr(harness.client, "_ensure_units", lambda units: f"unit:{units}")

    async def fake_collect(
        dev_id: str,
        addr: str,
        *,
        boost_active: bool,
        minutes: int | None,
    ) -> dict[str, Any]:
        """Return fallback metadata to trigger refresh dispatch."""

        assert boost_active is False
        assert minutes == 0
        return {"_fallback": True, "boost_active": False, "boost_minutes_delta": 0}

    monkeypatch.setattr(harness.client, "_collect_boost_metadata", fake_collect)

    responses = await harness.client.set_node_settings(
        "dev",
        ("acm", "7"),
        mode="auto",
        units="C",
        cancel_boost=True,
    )

    assert {"mode", "boost", "status_refresh", "boost_state"} <= responses.keys()
    mode_call = next(
        call
        for call in harness.segmented_calls
        if call["path"].endswith("/mode") and call["addr"] == "7"
    )
    assert mode_call["payload"] == {"mode": "auto"}
    boost_call = next(
        call
        for call in harness.segmented_calls
        if call["path"].endswith("/boost") and call["addr"] == "7"
    )
    assert boost_call["payload"] == {"boost": False}
    assert responses["boost_state"] == {"boost_active": False, "boost_minutes_delta": 0}


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

    assert responses["boost"] == {"ok": True}
    assert responses["boost_state"]["boost_active"] is False


@pytest.mark.asyncio
async def test_ducaheat_acm_settings_cancel_with_units(
    ducaheat_rest_harness: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel requests with explicit units should emit a status payload."""

    harness = ducaheat_rest_harness()
    monkeypatch.setattr(harness.client, "_ensure_units", lambda units: f"unit:{units}")

    responses = await harness.client.set_node_settings(
        "dev",
        ("acm", "16"),
        units="F",
        cancel_boost=True,
    )

    assert {"status", "boost", "boost_state"} <= responses.keys()
    status_call = next(
        call
        for call in harness.segmented_calls
        if call["path"].endswith("/status") and call["addr"] == "16"
    )
    assert status_call["payload"] == {"units": "unit:F"}


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
async def test_collect_boost_metadata_rtc_exception_inactive(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """RTC errors for inactive boosts should use fallback defaults."""

    harness = ducaheat_rest_harness()
    harness.client.get_rtc_time = AsyncMock(side_effect=RuntimeError("fail"))

    metadata = await harness.client._collect_boost_metadata(
        "dev", "9", boost_active=False, minutes=None
    )

    assert metadata == {
        "boost_active": False,
        "_fallback": True,
        "boost_minutes_delta": 0,
        "boost_end_day": None,
        "boost_end_min": None,
        "boost_end_timestamp": None,
    }


@pytest.mark.asyncio
async def test_collect_boost_metadata_invalid_payload(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Invalid RTC payloads should mark metadata as fallback with minutes."""

    harness = ducaheat_rest_harness(rtc_payload={"y": 2024})

    metadata = await harness.client._collect_boost_metadata(
        "dev", "10", boost_active=True, minutes=30
    )

    assert metadata["_fallback"] is True
    assert metadata["boost_minutes_delta"] == 30


@pytest.mark.asyncio
async def test_collect_boost_metadata_invalid_payload_inactive(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Inactive boosts with invalid payloads should default minutes to zero."""

    harness = ducaheat_rest_harness(rtc_payload={"y": 2024})

    metadata = await harness.client._collect_boost_metadata(
        "dev", "18", boost_active=False, minutes=None
    )

    assert metadata["_fallback"] is True
    assert metadata["boost_minutes_delta"] == 0


@pytest.mark.asyncio
async def test_collect_boost_metadata_inactive_valid(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Inactive boosts should derive zeroed end metadata from RTC payloads."""

    harness = ducaheat_rest_harness(
        rtc_payload={"y": 2024, "n": 2, "d": 3, "h": 4, "m": 5, "s": 6}
    )

    metadata = await harness.client._collect_boost_metadata(
        "dev", "11", boost_active=False, minutes=15
    )

    assert metadata == {
        "boost_active": False,
        "boost_end_day": None,
        "boost_end_min": None,
        "boost_minutes_delta": 15,
        "boost_end_timestamp": None,
    }


@pytest.mark.asyncio
async def test_ducaheat_acm_settings_non_mapping_metadata(
    ducaheat_rest_harness: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-dict metadata should be attached verbatim to the response."""

    harness = ducaheat_rest_harness()

    async def fake_collect(*_: Any, **__: Any) -> list[str]:
        return ["unexpected"]

    monkeypatch.setattr(harness.client, "_collect_boost_metadata", fake_collect)

    responses = await harness.client.set_node_settings(
        "dev", ("acm", "19"), cancel_boost=True
    )

    assert responses["boost_state"] == ["unexpected"]


def test_rtc_payload_to_datetime_non_mapping() -> None:
    """Non-mapping RTC payloads should return ``None``."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    assert client._rtc_payload_to_datetime(None) is None


def test_rtc_payload_to_datetime_invalid_values() -> None:
    """String RTC fields should be rejected when parsing the timestamp."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    assert client._rtc_payload_to_datetime({"y": "bad"}) is None


def test_rtc_payload_to_datetime_invalid_date() -> None:
    """Impossible calendar dates should be ignored."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    assert (
        client._rtc_payload_to_datetime(
            {"y": 2024, "n": 2, "d": 30, "h": 0, "m": 0, "s": 0}
        )
        is None
    )


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


@pytest.mark.asyncio
async def test_set_acm_boost_state_cancel_minutes(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Cancelling boosts should log zero minutes and collect metadata."""

    harness = ducaheat_rest_harness()

    result = await harness.client.set_acm_boost_state("dev", "13", boost=False)

    assert result["boost_state"]["boost_active"] is False


@pytest.mark.asyncio
async def test_set_acm_boost_state_metadata_wrapped_response(
    ducaheat_rest_harness: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-mapping responses should be wrapped alongside boost metadata."""

    harness = ducaheat_rest_harness()

    async def fake_post_acm(*args: Any, **kwargs: Any) -> bool:
        """Return a sentinel to force the non-dict branch."""

        harness.segmented_calls.append(
            {"path": args[0], "payload": kwargs.get("payload", {})}
        )
        return True

    monkeypatch.setattr(harness.client, "_post_acm_endpoint", fake_post_acm)

    result = await harness.client.set_acm_boost_state("dev", "14", boost=True)

    assert result["response"] is True
    assert result["boost_state"]["boost_active"] is True


def test_ensure_units_blank_defaults() -> None:
    """Empty unit strings should normalise to Celsius."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    assert client._ensure_units("") == "C"
