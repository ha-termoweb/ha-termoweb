# ruff: noqa: D100,D101,D102,D103,D105,D107,INP001,E402
from __future__ import annotations

import asyncio
from typing import Any

import pytest
import voluptuous as vol
from conftest import _install_stubs

_install_stubs()

import custom_components.termoweb.config_flow as config_flow
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant


def _schema_default(schema: vol.Schema, field: str) -> Any:
    for key in getattr(schema, "schema", {}):
        name = getattr(key, "schema", key)
        if name == field:
            return getattr(key, "default", None)
    raise AssertionError(f"Missing default for {field}")


def _create_flow(hass: HomeAssistant) -> config_flow.TermoWebConfigFlow:
    flow = config_flow.TermoWebConfigFlow()
    flow.hass = hass
    flow.context = {}
    return flow


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
            hass, "user@example.com", "pw", config_flow.BRAND_DUCAHEAT
        )
    )

    assert created == [(hass, "user@example.com", "pw", config_flow.BRAND_DUCAHEAT)]
    assert listed == [dummy_client]


def test_async_step_reconfigure_missing_entry_aborts() -> None:
    hass = HomeAssistant()
    flow = _create_flow(hass)
    flow.context["entry_id"] = "missing"

    result = asyncio.run(flow.async_step_reconfigure())

    assert result == {"type": "abort", "reason": "no_config_entry"}


def test_async_step_reconfigure_invalid_brand_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = HomeAssistant()
    entry = ConfigEntry(
        "entry-id",
        data={
            "username": "existing",
            "poll_interval": 150,
            config_flow.CONF_BRAND: "legacy",
        },
        options={"poll_interval": 150},
    )
    hass.config_entries.add_entry(entry)

    flow = _create_flow(hass)
    flow.context = {"entry_id": entry.entry_id}

    async def fake_version(_hass: HomeAssistant) -> str:
        return "7.7.7"

    monkeypatch.setattr(config_flow, "_get_version", fake_version)

    result = asyncio.run(flow.async_step_reconfigure())

    assert result["type"] == "form"
    schema = result["data_schema"]
    assert _schema_default(schema, "brand") == config_flow.DEFAULT_BRAND


@pytest.mark.parametrize(
    ("brand", "supported"),
    [
        (config_flow.BRAND_TERMOWEB, False),
        (config_flow.BRAND_DUCAHEAT, False),
        (config_flow.BRAND_TEVOLVE, False),
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
