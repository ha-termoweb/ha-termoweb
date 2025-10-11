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


@pytest.mark.asyncio
async def test_ducaheat_acm_extra_options_segmented_post(
    ducaheat_rest_harness: Callable[..., Any]
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
    assert setup_payload == {
        "extra_options": {"boost_time": 180, "boost_temp": "55.5"}
    }


@pytest.mark.asyncio
async def test_ducaheat_acm_boost_metadata_fallback_releases_selection(
    ducaheat_rest_harness: Callable[..., Any]
) -> None:
    """Verify boost metadata falls back and selection is released when RTC fails."""

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
            "boost_end": None,
            "boost_end_day": None,
            "boost_end_min": None,
            "boost_end_timestamp": None,
        },
    }

    assert harness.segmented_calls[0]["path"].endswith("/select")
    assert harness.segmented_calls[0]["payload"] == {"select": True}
    assert harness.segmented_calls[-1]["path"].endswith("/select")
    assert harness.segmented_calls[-1]["payload"] == {"select": False}


@pytest.mark.asyncio
async def test_ducaheat_acm_boost_metadata_happy_path(
    ducaheat_rest_harness: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure boost writes validate minutes, merge metadata, and release selection."""

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
    assert harness.segmented_calls[0]["payload"] == {"select": True}
    boost_call = next(
        call for call in harness.segmented_calls if call["path"].endswith("/boost")
    )
    assert boost_call["payload"] == {
        "boost": True,
        "boost_time": 180,
        "stemp": "21.5",
        "units": "C",
    }
    assert harness.segmented_calls[-1]["payload"] == {"select": False}

    assert harness.rtc_calls == ["dev"]

    assert result["ok"] is True
    boost_state = result.get("boost_state")
    assert boost_state == {
        "boost_active": True,
        "boost_end": {"day": 1, "minute": 510},
        "boost_end_day": 1,
        "boost_end_min": 510,
        "boost_minutes_delta": 180,
        "boost_end_timestamp": "2024-01-01T08:30:00",
    }

