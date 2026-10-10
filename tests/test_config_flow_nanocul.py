# ruff: noqa: D103,INP001,E402
"""Tests for the nanoCUL USB stick branch of the config flow."""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest
from conftest import _install_stubs
from tests_ha.fakes.radio_link import ProbeLink

_install_stubs()

from custom_components.termoweb import config_flow
from custom_components.termoweb.backend.radio import (
    DIALECT_A,
    DIALECT_B,
    GatewayInfo,
    RadioLink,
    RadioLinkError,
)
from custom_components.termoweb.backend.radio.discovery import NetworkSighting
from homeassistant.core import HomeAssistant

NET = bytes.fromhex("1234")  # synthetic network id
PORT = config_flow.SerialPort(
    "/dev/serial/by-id/usb-SHK_NANO_CUL_868-if00-port0", "nanoCUL 868", "A1B2C3"
)
FORM = {"device": PORT.device, "dialect": "auto", "network_id": ""}


def _flow(hass: HomeAssistant, **context: Any) -> config_flow.TermoWebConfigFlow:
    flow = config_flow.TermoWebConfigFlow()
    flow.hass = hass
    flow.context = dict(context)
    return flow


@pytest.fixture
def stick(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch port listing, the stick probe and discovery; record calls."""
    state: dict[str, Any] = {
        "ports": [PORT],
        "probe": ("nanocul-a1b2c3", False),
        "discover": (NetworkSighting(DIALECT_A, DIALECT_A.network_id), {6: object()}),
        "calls": [],
    }

    monkeypatch.setattr(config_flow, "list_serial_ports", lambda: state["ports"])

    async def fake_probe(device, usb_serial=None):
        state["calls"].append(("probe", device, usb_serial))
        if isinstance(state["probe"], Exception):
            raise state["probe"]
        return state["probe"]

    async def fake_discover(host, port, dialect, network_id, **kwargs):
        state["calls"].append(("discover", host, port, dialect, network_id, kwargs))
        if isinstance(state["discover"], Exception):
            raise state["discover"]
        return state["discover"]

    monkeypatch.setattr(config_flow, "probe_nanocul", fake_probe)
    monkeypatch.setattr(config_flow, "discover_radio", fake_discover)
    return state


async def _finish(flow: config_flow.TermoWebConfigFlow, first: Any) -> Any:
    """Choose discovery, drive the progress step and return the finish result."""
    if first["type"] == "menu":
        assert first["step_id"] == "radio_method"
        first = await flow.async_step_radio_discover()
    assert first["type"] == "progress"
    await asyncio.wait([first["progress_task"]])
    assert (await flow.async_step_radio_discover())["type"] == "progress_done"
    return await flow.async_step_radio_finish()


@pytest.mark.asyncio
async def test_port_choice_then_discovery_creates_entry(stick) -> None:
    flow = _flow(HomeAssistant())
    form = await flow.async_step_nanocul()
    assert form["type"] == "form" and form["step_id"] == "nanocul"

    result = await _finish(flow, await flow.async_step_nanocul(dict(FORM)))

    assert flow._unique_id == "radio:nanocul-a1b2c3"
    assert result["type"] == "create_entry"
    assert result["title"] == f"nanoCUL ({PORT.device})"
    assert result["data"]["radio_type"] == "nanocul"
    assert result["data"]["device"] == PORT.device
    assert result["data"]["radio_device_id"] == "nanocul-a1b2c3"
    assert result["data"]["dialect"] == "A" and "host" not in result["data"]
    assert stick["calls"][0] == ("probe", PORT.device, "A1B2C3")
    discover = stick["calls"][1]
    assert discover[1:3] == (PORT.device, 0)
    assert discover[5]["dialect_capable"] is False


@pytest.mark.asyncio
async def test_manual_path_when_no_port_is_detected(stick) -> None:
    stick["ports"] = []
    flow = _flow(HomeAssistant())
    first = await flow.async_step_nanocul({**FORM, "device": "manual"})
    assert first["step_id"] == "nanocul_manual"
    assert first["data_schema"]({"device": "socket://gw:2323"})["device"]
    result = await _finish(
        flow,
        await flow.async_step_nanocul_manual({**FORM, "device": " socket://gw:2323 "}),
    )
    assert result["data"]["device"] == "socket://gw:2323"
    assert stick["calls"][0] == ("probe", "socket://gw:2323", None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("form", "probe", "error"),
    [
        (FORM, RadioLinkError("busy"), {"base": "cannot_connect_nanocul"}),
        (
            {**FORM, "dialect": "B"},
            ("id", False),
            {"base": "dialect_unsupported_firmware"},
        ),
        (
            {**FORM, "network_id": "XYZ"},
            ("id", False),
            {"network_id": "invalid_network_id"},
        ),
    ],
)
async def test_stick_errors_stay_on_the_form(stick, form, probe, error) -> None:
    stick["probe"] = probe
    result = await _flow(HomeAssistant()).async_step_nanocul(dict(form))
    assert result["type"] == "form" and result["errors"] == error


@pytest.mark.asyncio
async def test_dialect_b_on_capable_firmware_is_accepted(stick) -> None:
    stick["probe"] = ("0a0b0c0d0e0f", True)
    stick["discover"] = (NetworkSighting(DIALECT_B, NET), {6: object()})
    flow = _flow(HomeAssistant())
    result = await _finish(
        flow, await flow.async_step_nanocul({**FORM, "dialect": "B"})
    )
    assert result["data"]["dialect"] == "B" and result["data"]["network_id"] == "1234"
    assert stick["calls"][1][5]["dialect_capable"] is True


@pytest.mark.asyncio
async def test_discovery_failure_returns_to_the_manual_form(stick) -> None:
    stick["discover"] = RadioLinkError("unplugged")
    flow = _flow(HomeAssistant())
    result = await _finish(flow, await flow.async_step_nanocul(dict(FORM)))
    assert result["step_id"] == "nanocul_manual"
    assert result["errors"] == {"base": "cannot_connect_nanocul"}


@pytest.mark.asyncio
async def test_probe_nanocul(monkeypatch) -> None:
    monkeypatch.setattr(config_flow, "RadioLink", ProbeLink)
    monkeypatch.setattr(ProbeLink, "created", [])
    monkeypatch.setattr(
        ProbeLink, "info", GatewayInfo("3.5", "869.525", "2DE5", False, 1, None, "")
    )
    dev_id, capable = await config_flow.probe_nanocul("/dev/ttyUSB0", "A1B2C3")
    assert (dev_id, capable) == ("nanocul-a1b2c3", False)
    made = ProbeLink.created[-1]
    assert made["auto_ack"] is False and made["dialect"] is DIALECT_A
    assert callable(made["open_connection"])

    monkeypatch.setattr(
        ProbeLink,
        "info",
        GatewayInfo("3.6", "869.525", "2DE5", True, 1, "0A:0B:0C:0D:0E:0F", "", "A"),
    )
    assert await config_flow.probe_nanocul("socket://gw:1") == ("0a0b0c0d0e0f", True)


def test_radio_link_factory_and_address() -> None:
    nanocul = {"radio_type": "nanocul", "device": "/dev/ttyUSB0"}
    factory = config_flow.radio_link_factory(nanocul)
    assert (
        factory.func.__name__ == "RadioLink" and "open_connection" in factory.keywords
    )
    assert config_flow.radio_address(nanocul) == ("/dev/ttyUSB0", 0)
    esp32 = {"host": "gw", "port": 2323}
    assert config_flow.radio_link_factory(esp32).__name__ == "RadioLink"
    assert config_flow.radio_address(esp32) == ("gw", 2323)


@pytest.mark.asyncio
async def test_discover_radio_dialect_a_only_firmware(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_discover(host, port, *, dialects, link_factory):
        seen["dialects"] = [d.name for d in dialects]
        return NetworkSighting(DIALECT_A, DIALECT_A.network_id)

    async def fake_probe(host, port, dialect, network_id, candidates, link_factory):
        return {6: "s"}

    monkeypatch.setattr(config_flow, "discover_network", fake_discover)
    monkeypatch.setattr(config_flow, "probe_heaters", fake_probe)
    await config_flow.discover_radio("dev", 0, "auto", None, dialect_capable=False)
    assert seen["dialects"] == ["A"]


def test_list_serial_ports_prefers_by_id_paths(monkeypatch, tmp_path) -> None:
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    tty = tmp_path / "ttyUSB0"
    tty.write_text("")
    (by_id / "usb-SHK_NANO_CUL-port0").symlink_to(tty)
    (by_id / "usb-other").symlink_to(tmp_path / "ttyUSB9")
    monkeypatch.setattr(config_flow, "SERIAL_BY_ID_DIR", str(by_id))
    ports = [
        types.SimpleNamespace(
            device=str(tty), description="nanoCUL", serial_number="X1"
        ),
        types.SimpleNamespace(
            device="/dev/ttyS0", description=None, serial_number=None
        ),
    ]
    list_ports = types.SimpleNamespace(comports=lambda: ports)
    tools = types.ModuleType("serial.tools")
    tools.list_ports = list_ports
    monkeypatch.setitem(sys.modules, "serial", types.ModuleType("serial"))
    monkeypatch.setitem(sys.modules, "serial.tools", tools)
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", list_ports)

    result = config_flow.list_serial_ports()

    assert result[0] == config_flow.SerialPort(
        str(by_id / "usb-SHK_NANO_CUL-port0"), "nanoCUL", "X1"
    )
    assert result[1] == config_flow.SerialPort("/dev/ttyS0", "/dev/ttyS0", None)
    monkeypatch.setattr(config_flow, "SERIAL_BY_ID_DIR", str(tmp_path / "missing"))
    assert config_flow._by_id_path("/dev/ttyACM0") == "/dev/ttyACM0"


def test_nanocul_schema_defaults() -> None:
    schema = config_flow._nanocul_schema([PORT], {"device": "/dev/gone"})
    assert schema({})["device"] == PORT.device
    schema = config_flow._nanocul_schema([], {})
    assert schema({})["device"] == "manual"
    manual = config_flow._nanocul_manual_schema({"device": "manual"})
    assert manual({"device": "x"})["dialect"] == "auto"
