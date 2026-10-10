# ruff: noqa: D103,INP001,E402
"""Tests for the listen-only ("monitor") choice of the radio config flow."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import _install_stubs

_install_stubs()

from custom_components.termoweb import config_flow
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

DEV_ID = "0a0b0c0d0e0f"
PORT = config_flow.SerialPort("/dev/serial/by-id/usb-nanoCUL-if00", "nanoCUL", "X1")


def _flow(hass: HomeAssistant, **context: Any) -> config_flow.TermoWebConfigFlow:
    flow = config_flow.TermoWebConfigFlow()
    flow.hass = hass
    flow.context = dict(context)
    return flow


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Patch the gateway and stick probes; discovery and pairing must not run."""
    calls: list[tuple[Any, ...]] = []

    async def fake_gateway(host: str, port: int) -> str:
        calls.append(("gateway", host, port))
        return DEV_ID

    async def fake_stick(device: str, usb_serial: str | None = None):
        calls.append(("stick", device, usb_serial))
        return "nanocul-x1", False  # stock firmware: dialect A only

    async def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a listen-only setup must not discover or pair")

    monkeypatch.setattr(config_flow, "probe_gateway", fake_gateway)
    monkeypatch.setattr(config_flow, "probe_nanocul", fake_stick)
    monkeypatch.setattr(config_flow, "list_serial_ports", lambda: [PORT])
    monkeypatch.setattr(config_flow, "discover_radio", forbidden)
    monkeypatch.setattr(config_flow, "pair_radio", forbidden)
    return calls


@pytest.mark.asyncio
async def test_esp32_listen_only_entry(probes) -> None:
    flow = _flow(HomeAssistant())
    menu = await flow.async_step_radio(
        {"host": "10.0.0.5", "port": 2323, "dialect": "auto", "network_id": ""}
    )
    assert menu["menu_options"] == ["radio_discover", "radio_pair", "radio_monitor"]
    assert flow._unique_id == f"radio:{DEV_ID}"

    result = await flow.async_step_radio_monitor()

    assert flow._unique_id == f"radio_monitor:{DEV_ID}"  # a normal entry stays free
    assert result["type"] == "create_entry"
    assert result["title"] == "Radio monitor (10.0.0.5)"
    assert result["data"] == {
        "brand": "radio_monitor",
        "radio_type": "esp32",
        "host": "10.0.0.5",
        "port": 2323,
        "supports_diagnostics": True,
    }
    assert probes == [("gateway", "10.0.0.5", 2323)]


@pytest.mark.asyncio
async def test_nanocul_listen_only_entry_on_dialect_a_firmware(probes) -> None:
    flow = _flow(HomeAssistant())
    menu = await flow.async_step_nanocul(
        {"device": PORT.device, "dialect": "auto", "network_id": ""}
    )
    assert menu["step_id"] == "radio_method"

    result = await flow.async_step_radio_monitor()

    assert flow._unique_id == "radio_monitor:nanocul-x1"
    assert result["title"] == f"Radio monitor ({PORT.device})"
    assert result["data"] == {
        "brand": "radio_monitor",
        "radio_type": "nanocul",
        "device": PORT.device,
        "radio_device_id": "nanocul-x1",
        "supports_diagnostics": True,
    }
    assert probes == [("stick", PORT.device, "X1")]


@pytest.mark.asyncio
async def test_listen_only_entry_cannot_be_reconfigured() -> None:
    hass = HomeAssistant()
    entry = ConfigEntry(
        "monitor",
        data={"brand": "radio_monitor", "host": "10.0.0.5", "port": 2323},
    )
    hass.config_entries.add_entry(entry)
    flow = _flow(hass, entry_id=entry.entry_id, source="reconfigure")
    result = await flow.async_step_reconfigure()
    assert result == {"type": "abort", "reason": "monitor_reconfigure"}


@pytest.mark.asyncio
async def test_listen_only_options_have_no_pairing() -> None:
    hass = HomeAssistant()
    entry = ConfigEntry("monitor", data={"brand": "radio_monitor"})
    hass.config_entries.add_entry(entry)
    flow = config_flow.TermoWebOptionsFlow(entry)
    flow.hass = hass
    form = await flow.async_step_init()
    assert form["type"] == "form" and form["step_id"] == "init"
    assert form["description_placeholders"]["heaters"] == ""
