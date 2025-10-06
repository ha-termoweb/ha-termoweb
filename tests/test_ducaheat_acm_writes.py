from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

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
) -> tuple[DucaheatRESTClient, list[tuple[str, str, dict[str, Any]]]]:
    """Create a REST client with fake request handling."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
    responses = list(responses or [])
    calls: list[tuple[str, str, dict[str, Any]]] = []

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

    monkeypatch.setattr(client, "_authed_headers", fake_headers)
    monkeypatch.setattr(client, "_request", fake_request)
    return client, calls


@pytest.mark.parametrize(
    "input_mode,expected",
    [
        (HVACMode.AUTO, "auto"),
        (HVACMode.OFF, "off"),
        ("boost", "boost"),
        ("manual", "manual"),
    ],
)
def test_ducaheat_acm_mode_requests(
    monkeypatch: pytest.MonkeyPatch, input_mode: HVACMode | str, expected: str
) -> None:
    async def _run() -> None:
        client, calls = _setup_client(monkeypatch, responses=[{"ok": True}])

        await client.set_node_settings("dev", ("acm", "9"), mode=input_mode)

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
                    "json": {"mode": expected},
                },
            )
        ]

    asyncio.run(_run())


def test_ducaheat_acm_set_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client, calls = _setup_client(monkeypatch, responses=[{"status": "ok"}])

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

    asyncio.run(_run())


def test_ducaheat_acm_program_write(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client, calls = _setup_client(monkeypatch, responses=[{"saved": True}])
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

        with pytest.raises(ValueError):
            await client.set_node_settings("dev", ("acm", "4"), prog=[0] * 24)

    asyncio.run(_run())


def test_ducaheat_acm_program_temps(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client, calls = _setup_client(monkeypatch, responses=[{"saved": True}])

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

        with pytest.raises(DucaheatRequestError) as exc:
            await client.set_node_settings("dev", ("acm", "1"), mode="boost")

        assert "malformed" in str(exc.value)

    asyncio.run(_run())
