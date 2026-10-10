# ruff: noqa: D100,D101,D102,D103,D105,D107,INP001,E402
from __future__ import annotations

import asyncio

import pytest
from conftest import _install_stubs

_install_stubs()

import custom_components.termoweb.config_flow as config_flow
from custom_components.termoweb.const import (
    BRAND_DUCAHEAT,
    BRAND_TERMOWEB,
    BRAND_TEVOLVE,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant


def test_get_version_reads_integration_version() -> None:
    hass = HomeAssistant()

    result = asyncio.run(config_flow._get_version(hass))

    assert result == "test-version"


def test_get_version_returns_unknown_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = HomeAssistant()

    class DummyIntegration:
        def __init__(self, version: str) -> None:
            self.version = version

    async def fake_get_integration(
        _hass: HomeAssistant, _domain: str
    ) -> DummyIntegration:
        return DummyIntegration("")

    monkeypatch.setattr(
        "custom_components.termoweb.utils.async_get_integration",
        fake_get_integration,
    )

    result = asyncio.run(config_flow._get_version(hass))

    assert result == "unknown"


def test_validate_login_uses_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    hass = HomeAssistant()
    created: list[tuple[Any, str, str, str]] = []
    listed: list[Any] = []
    dummy_client = object()

    def fake_create(
        hass_in: HomeAssistant, username: str, password: str, brand: str
    ) -> Any:
        created.append((hass_in, username, password, brand))
        return dummy_client

    async def fake_list(client: Any) -> list[Any]:
        listed.append(client)
        return []

    monkeypatch.setattr(config_flow, "create_rest_client", fake_create)
    monkeypatch.setattr(config_flow, "async_list_devices", fake_list)

    asyncio.run(
        config_flow._validate_login(
            hass, "user@example.com", "pw", BRAND_DUCAHEAT
        )
    )

    assert created == [(hass, "user@example.com", "pw", BRAND_DUCAHEAT)]
    assert listed == [dummy_client]


@pytest.mark.parametrize(
    ("brand", "supported"),
    [
        (BRAND_TERMOWEB, False),
        (BRAND_DUCAHEAT, False),
        (BRAND_TEVOLVE, False),
        (config_flow.BRAND_RADIO_MONITOR, False),
        (config_flow.BRAND_RADIO, True),
    ],
)
def test_options_flow_offered_only_for_radio_entries(
    brand: str, supported: bool
) -> None:
    """Cloud and listen-only entries have no options, so HA must not offer a form."""
    entry = ConfigEntry("entry-id", data={"brand": brand})

    assert (
        config_flow.TermoWebConfigFlow.async_supports_options_flow(entry) is supported
    )
