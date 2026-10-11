"""Tests for the radio gateway and nanoCUL config flow steps and helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
import dataclasses
import hashlib
from ipaddress import ip_address
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.config_entries import SOURCE_USER, SOURCE_ZEROCONF, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType, InvalidData, section
from homeassistant.helpers import config_validation as cv, instance_id
from homeassistant.helpers.service_info.zeroconf import ZeroconfServiceInfo
import probatio
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
import voluptuous as vol

from custom_components.termoweb import config_flow_radio, radio_pairing
from custom_components.termoweb.backend.radio import (
    DIALECT_A,
    DIALECT_B,
    GatewayInfo,
    RadioLink,
    RadioLinkError,
)
from custom_components.termoweb.backend.radio.discovery import NetworkSighting
from custom_components.termoweb.backend.radio.link import parse_query_line
from custom_components.termoweb.backend.radio.survey import analyse
from custom_components.termoweb.const import (
    BRAND_RADIO,
    BRAND_RADIO_MONITOR,
    CONF_BRAND,
    DOMAIN,
)
from tests.fakes.radio_link import ProbeLink

# Synthetic identifiers only.
GATEWAY_ID = "0a0b0c0d0e0f"
OTHER_GATEWAY_ID = "0f0e0d0c0b0a"
STICK = "/dev/serial/by-id/usb-test-stick"
NET = bytes.fromhex("1234")
NODES = [{"type": "htr", "addr": "6", "name": "Heater 6"}]


class FakeRadio:
    """Records probes and serves canned probe and discovery results."""

    def __init__(self) -> None:
        """Start with a healthy gateway and a healthy dialect-capable stick."""
        self.gateway: Any = GATEWAY_ID
        self.stick: Any = (GATEWAY_ID, True)
        self.discover: Any = (NetworkSighting(DIALECT_B, NET), {6: None, 7: None})
        self.ports: list[config_flow_radio.SerialPort] = []
        self.calls: list[tuple[Any, ...]] = []
        self.kwargs: list[dict[str, Any]] = []
        self.serials: list[str | None] = []

    async def probe_gateway(self, host: str, port: int) -> str:
        """Return (or raise) the configured gateway probe result."""
        self.calls.append(("probe", host, port))
        if isinstance(self.gateway, Exception):
            raise self.gateway
        return self.gateway

    async def probe_nanocul(
        self, device: str, usb_serial: str | None = None
    ) -> tuple[str, bool]:
        """Return (or raise) the configured stick probe result."""
        self.calls.append(("probe", device))
        self.serials.append(usb_serial)
        if isinstance(self.stick, Exception):
            raise self.stick
        return self.stick

    async def discover_radio(self, *args: Any, **kwargs: Any) -> Any:
        """Return (or raise) the configured discovery result."""
        self.calls.append(("discover", *args[:2]))
        self.kwargs.append({"args": args[2:], **kwargs})
        if isinstance(self.discover, Exception):
            raise self.discover
        return self.discover


@pytest.fixture
def radio() -> Generator[FakeRadio]:
    """Patch the radio probes and discovery; never set up a real radio entry."""
    fake = FakeRadio()
    with (
        patch.object(config_flow_radio, "probe_gateway", fake.probe_gateway),
        patch.object(config_flow_radio, "probe_nanocul", fake.probe_nanocul),
        patch.object(config_flow_radio, "discover_radio", fake.discover_radio),
        patch.object(config_flow_radio, "list_serial_ports", lambda: fake.ports),
        patch(
            "custom_components.termoweb.async_setup_entry",
            AsyncMock(return_value=True),
        ),
    ):
        yield fake


def _esp32_entry(
    hass: HomeAssistant,
    brand: str = BRAND_RADIO,
    host: str = "10.0.0.5",
    **kwargs: Any,
) -> MockConfigEntry:
    """Add an ESP32 radio entry for gateway GATEWAY_ID."""
    data: dict[str, Any] = {CONF_BRAND: brand, "host": host, "port": 2323}
    if brand == BRAND_RADIO:
        data |= {"dialect": "B", "network_id": "1234", "nodes": NODES}
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id=f"{brand}:{GATEWAY_ID}",
        minor_version=2,
        data=data,
        **kwargs,
    )
    entry.add_to_hass(hass)
    return entry


def _stick_entry(hass: HomeAssistant, device_id: str = GATEWAY_ID) -> MockConfigEntry:
    """Add a nanoCUL radio entry on STICK."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id=f"{BRAND_RADIO}:{device_id}",
        minor_version=2,
        data={
            CONF_BRAND: BRAND_RADIO,
            "radio_type": "nanocul",
            "device": STICK,
            "radio_device_id": device_id,
            "dialect": "B",
            "network_id": "1234",
            "nodes": NODES,
        },
    )
    entry.add_to_hass(hass)
    return entry


async def _menu(hass: HomeAssistant, option: str) -> dict[str, Any]:
    """Start a user flow and pick ``option`` from the first menu."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    return await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": option}
    )


async def _finish_progress(hass: HomeAssistant, result: dict[str, Any]) -> Any:
    """Let the discovery task finish and return the flow's final result."""
    if result["type"] is not FlowResultType.SHOW_PROGRESS:
        return result  # the fake discovery finished before the first step returned
    await hass.async_block_till_done()
    return await hass.config_entries.flow.async_configure(result["flow_id"])


async def test_radio_step_takes_over_a_gateway_another_entry_uses(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A gateway another entry talks to is probed anyway: HA takes the link over."""
    _esp32_entry(hass, brand=BRAND_RADIO_MONITOR)
    form = await _menu(hass, "radio")

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"host": " 10.0.0.5 "}
    )

    assert result["type"] is FlowResultType.MENU
    assert radio.calls == [("probe", "10.0.0.5", 2323)]


async def test_nanocul_step_takes_over_a_stick_another_entry_reads(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A stick another entry reads is opened anyway: HA takes it over."""
    MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_BRAND: BRAND_RADIO_MONITOR,
            "radio_type": "nanocul",
            "device": STICK,
        },
    ).add_to_hass(hass)
    form = await _menu(hass, "nanocul")
    form = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"device": "manual"}
    )
    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"device": f" {STICK} "}
    )

    assert result["type"] is FlowResultType.MENU
    assert radio.calls == [("probe", STICK)]


async def test_reconfigure_radio_moves_address_and_reloads(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A new address of the same gateway is saved and the entry reloaded."""
    entry = _esp32_entry(hass)
    form = await entry.start_reconfigure_flow(hass)
    assert form["step_id"] == "reconfigure_radio"

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"host": "10.0.0.9", "port": 2424, "rescan": False}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert (entry.data["host"], entry.data["port"]) == ("10.0.0.9", 2424)
    assert entry.data["nodes"] == NODES
    assert radio.calls == [("probe", "10.0.0.9", 2424)]


async def test_reconfigure_radio_rescan_replaces_nodes(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """Scanning again stores the heaters found on the entry's own network."""
    entry = _esp32_entry(hass)
    form = await entry.start_reconfigure_flow(hass)

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"host": "10.0.0.5", "port": 2323, "rescan": True}
    )
    result = await _finish_progress(hass, result)

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert [node["addr"] for node in entry.data["nodes"]] == ["6", "7"]
    assert entry.data["network_id"] == "1234"


@pytest.mark.parametrize(
    ("make_entry", "user_input", "step_id"),
    [
        (
            _esp32_entry,
            {"host": "10.0.0.5", "port": 2323, "rescan": True},
            "reconfigure_radio",
        ),
        (_stick_entry, {"device": STICK, "rescan": True}, "reconfigure_nanocul"),
    ],
)
async def test_reconfigure_rescan_error_shows_form(
    hass: HomeAssistant,
    radio: FakeRadio,
    make_entry: Any,
    user_input: dict[str, Any],
    step_id: str,
) -> None:
    """A failed rescan returns to the reconfigure form and keeps the entry."""
    entry = make_entry(hass)
    before = dict(entry.data)
    radio.discover = config_flow_radio.RadioSetupError("no_heaters")
    form = await entry.start_reconfigure_flow(hass)

    result = await hass.config_entries.flow.async_configure(form["flow_id"], user_input)
    result = await _finish_progress(hass, result)

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == step_id
    assert result["errors"] == {"base": "no_heaters"}
    assert dict(entry.data) == before


async def test_reconfigure_radio_refuses_other_gateway(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """Pointing an entry at a different gateway aborts instead of orphaning devices."""
    entry = _esp32_entry(hass)
    before = dict(entry.data)
    radio.gateway = OTHER_GATEWAY_ID
    form = await entry.start_reconfigure_flow(hass)

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"host": "10.0.0.9", "port": 2323, "rescan": False}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "unique_id_mismatch"
    assert dict(entry.data) == before


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RadioLinkError("down"), "cannot_connect_radio"),
        (config_flow_radio.RadioSetupError("no_gateway_mac"), "no_gateway_mac"),
    ],
)
async def test_reconfigure_radio_probe_errors(
    hass: HomeAssistant, radio: FakeRadio, error: Exception, expected: str
) -> None:
    """An unreachable gateway re-shows the form with the error."""
    entry = _esp32_entry(hass)
    radio.gateway = error
    form = await entry.start_reconfigure_flow(hass)

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"host": "10.0.0.9", "port": 2323, "rescan": False}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": expected}


async def test_reconfigure_nanocul_moves_port_and_reloads(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """The same stick on a new port is saved and the entry reloaded."""
    entry = _stick_entry(hass)
    form = await entry.start_reconfigure_flow(hass)
    assert form["step_id"] == "reconfigure_nanocul"

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"device": "/dev/ttyUSB1", "rescan": False}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data["device"] == "/dev/ttyUSB1"


async def test_reconfigure_nanocul_rescan_replaces_nodes(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """Scanning again through the stick stores the heaters found."""
    entry = _stick_entry(hass)
    form = await entry.start_reconfigure_flow(hass)

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"device": STICK, "rescan": True}
    )
    result = await _finish_progress(hass, result)

    assert result["reason"] == "reconfigure_successful"
    assert [node["addr"] for node in entry.data["nodes"]] == ["6", "7"]


async def test_reconfigure_nanocul_refuses_other_stick(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A stick with a different MAC is a different device: abort."""
    entry = _stick_entry(hass)
    radio.stick = (OTHER_GATEWAY_ID, True)
    form = await entry.start_reconfigure_flow(hass)

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"device": "/dev/ttyUSB1", "rescan": False}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "unique_id_mismatch"
    assert entry.data["device"] == STICK


async def test_reconfigure_nanocul_without_mac_cannot_be_checked(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A stock stick is known only by its port, so a new port is accepted."""
    entry = _stick_entry(hass, device_id="nanocul-a1b2c3")
    radio.stick = ("nanocul-ffffffffffff", True)
    form = await entry.start_reconfigure_flow(hass)

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"device": "/dev/ttyUSB1", "rescan": False}
    )

    assert result["reason"] == "reconfigure_successful"
    assert entry.unique_id == f"{BRAND_RADIO}:nanocul-a1b2c3"


@pytest.mark.parametrize(
    ("stick", "expected"),
    [
        (RadioLinkError("gone"), "cannot_connect_nanocul"),
        ((GATEWAY_ID, False), "dialect_unsupported_firmware"),
    ],
)
async def test_reconfigure_nanocul_errors(
    hass: HomeAssistant, radio: FakeRadio, stick: Any, expected: str
) -> None:
    """An unreadable stick, or one that cannot speak dialect B, shows an error."""
    entry = _stick_entry(hass)
    radio.stick = stick
    form = await entry.start_reconfigure_flow(hass)

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"device": "/dev/ttyUSB1", "rescan": False}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure_nanocul"
    assert result["errors"] == {"base": expected}


async def test_listen_only_entry_cannot_be_reconfigured(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A monitor entry has nothing to change."""
    entry = _esp32_entry(hass, brand=BRAND_RADIO_MONITOR)

    result = await entry.start_reconfigure_flow(hass)

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "monitor_reconfigure"


# --- setting up a radio entry ----------------------------------------------------

FORM = {"host": "10.0.0.5"}  # the manual form needs only the address
PORT = config_flow_radio.SerialPort(
    "/dev/serial/by-id/usb-SHK_NANO_CUL_868-if00-port0", "nanoCUL 868", "A1B2C3"
)
STICK_FORM = {"device": PORT.device}


async def _gateway_menu(hass: HomeAssistant, form: dict[str, Any]) -> dict[str, Any]:
    """Submit the ESP32 gateway form and return the method menu (or error)."""
    first = await _menu(hass, "radio")
    assert first["step_id"] == "radio"
    return await hass.config_entries.flow.async_configure(first["flow_id"], form)


async def _stick_menu(hass: HomeAssistant, form: dict[str, Any]) -> dict[str, Any]:
    """Submit the nanoCUL port form and return the method menu (or error)."""
    first = await _menu(hass, "nanocul")
    assert first["step_id"] == "nanocul"
    return await hass.config_entries.flow.async_configure(first["flow_id"], form)


async def _choose(
    hass: HomeAssistant, menu: dict[str, Any], option: str
) -> dict[str, Any]:
    """Pick ``option`` from the method menu and let any progress finish."""
    assert menu["type"] is FlowResultType.MENU
    assert menu["menu_options"] == ["radio_discover", "radio_monitor"]
    result = await hass.config_entries.flow.async_configure(
        menu["flow_id"], {"next_step_id": option}
    )
    return await _finish_progress(hass, result)


async def test_user_step_offers_cloud_or_radio(hass: HomeAssistant) -> None:
    """The first step is a menu: cloud account, ESP32 gateway or nanoCUL."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.MENU
    assert result["menu_options"] == ["cloud", "nanocul", "radio"]


async def test_radio_gateway_entry_is_created_without_waiting_for_heaters(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """Gateway form, then the entry at once: no progress step, no heaters yet."""
    menu = await _gateway_menu(hass, FORM)

    result = await hass.config_entries.flow.async_configure(
        menu["flow_id"], {"next_step_id": "radio_discover"}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Radio gateway (10.0.0.5)"
    assert result["result"].unique_id == f"radio:{GATEWAY_ID}"
    assert result["data"] == {
        "brand": "radio",
        "radio_type": "esp32",
        "host": "10.0.0.5",
        "port": 2323,
        "nodes": [],
    }
    assert radio.calls == [("probe", "10.0.0.5", 2323)]  # nothing listened to


@pytest.mark.parametrize("menu_of", ["gateway", "stick"])
async def test_setup_menu_offers_only_add_or_listen(
    hass: HomeAssistant, radio: FakeRadio, menu_of: str
) -> None:
    """Pairing is automatic after setup: the menu has no separate pairing choice."""
    radio.ports = [PORT]
    if menu_of == "gateway":
        menu = await _gateway_menu(hass, FORM)
    else:
        menu = await _stick_menu(hass, STICK_FORM)
    assert menu["menu_options"] == ["radio_discover", "radio_monitor"]
    assert not hasattr(config_flow_radio.RadioConfigFlow, "async_step_radio_pair")
    with pytest.raises(InvalidData):
        await hass.config_entries.flow.async_configure(
            menu["flow_id"], {"next_step_id": "radio_pair"}
        )


async def test_dialect_a_entry_starts_on_its_fixed_network(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """Dialect A has one network id, so the entry knows it from the start."""
    menu = await _gateway_menu(hass, {**FORM, "advanced": {"dialect": "A"}})
    result = await _choose(hass, menu, "radio_discover")
    assert result["data"]["dialect"] == "A"
    assert result["data"]["network_id"] == DIALECT_A.network_id.hex().upper()
    assert result["data"]["nodes"] == []


async def test_radio_form_asks_only_for_the_address(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """Port and dialect sit in a collapsed section; no network id field."""
    form = await _menu(hass, "radio")
    fields = {str(key): value for key, value in form["data_schema"].schema.items()}
    assert set(fields) == {"host", "advanced"}
    assert isinstance(fields["advanced"], section)
    assert fields["advanced"].options == {"collapsed": True}
    assert {str(key) for key in fields["advanced"].schema.schema} == {
        "port",
        "dialect",
    }
    fields_json = probatio.to_field_list(
        form["data_schema"], custom_serializer=cv.custom_serializer
    )
    assert [field["name"] for field in fields_json] == ["host", "advanced"]
    assert fields_json[1]["type"] == "expandable"


async def test_radio_advanced_port_and_dialect_are_used(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A port and dialect from the advanced section reach the probe and the entry."""
    menu = await _gateway_menu(
        hass, {**FORM, "advanced": {"port": 2424, "dialect": "B"}}
    )
    result = await _choose(hass, menu, "radio_discover")
    assert result["data"]["port"] == 2424
    assert result["data"]["dialect"] == "B"
    assert "network_id" not in result["data"]  # the network is learned later
    assert radio.calls == [("probe", "10.0.0.5", 2424)]


async def test_radio_step_aborts_for_a_configured_gateway(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """The same gateway at a new address is already configured: abort."""
    _esp32_entry(hass, host="10.0.0.9")
    result = await _gateway_menu(hass, FORM)
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


@pytest.mark.parametrize(
    ("probe", "error"),
    [
        (RadioLinkError("down"), "cannot_connect_radio"),
        (config_flow_radio.RadioSetupError("no_gateway_mac"), "no_gateway_mac"),
    ],
)
async def test_radio_form_gateway_errors(
    hass: HomeAssistant, radio: FakeRadio, probe: Exception, error: str
) -> None:
    """An unreachable gateway, or one without a MAC, keeps the form open."""
    radio.gateway = probe
    result = await _gateway_menu(hass, FORM)
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": error}


@pytest.mark.parametrize(
    ("raised", "error"),
    [
        (RadioLinkError("lost"), "cannot_connect_radio"),
        (RuntimeError("boom"), "unknown"),
    ],
)
async def test_rescan_errors_return_to_the_reconfigure_form(
    hass: HomeAssistant, radio: FakeRadio, raised: Exception, error: str
) -> None:
    """A rescan that fails shows the reconfigure form again with the reason."""
    entry = _esp32_entry(hass)
    radio.discover = raised
    form = await entry.start_reconfigure_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"host": "10.0.0.5", "port": 2323, "rescan": True}
    )
    result = await _finish_progress(hass, result)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure_radio"
    assert result["errors"] == {"base": error}


async def test_rescan_shows_progress_until_the_scan_finishes(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A slow rescan of the stored network shows progress, then saves the heaters."""
    release = asyncio.Event()
    seen: list[tuple[Any, ...]] = []

    async def slow(*args: Any, **_kwargs: Any) -> Any:
        seen.append(args)
        await release.wait()
        return NetworkSighting(DIALECT_B, NET), {6: None}

    entry = _esp32_entry(hass)
    form = await entry.start_reconfigure_flow(hass)
    with patch.object(config_flow_radio, "discover_radio", slow):
        result = await hass.config_entries.flow.async_configure(
            form["flow_id"], {"host": "10.0.0.5", "port": 2323, "rescan": True}
        )
        assert result["type"] is FlowResultType.SHOW_PROGRESS
        assert result["progress_action"] == "radio_rescan"
        again = await hass.config_entries.flow.async_configure(result["flow_id"])
        assert again["type"] is FlowResultType.SHOW_PROGRESS
        release.set()
        result = await _finish_progress(hass, again)
    assert result["reason"] == "reconfigure_successful"
    assert seen == [("10.0.0.5", 2323, "B", NET)]  # the stored network: no listening


@pytest.mark.parametrize(
    ("make_entry", "user_input"),
    [
        (_esp32_entry, {"host": "10.0.0.9", "port": 2323, "rescan": True}),
        (_stick_entry, {"device": "/dev/ttyUSB1", "rescan": True}),
    ],
)
async def test_rescan_of_an_entry_still_learning_its_network_just_saves(
    hass: HomeAssistant, radio: FakeRadio, make_entry: Any, user_input: dict
) -> None:
    """Without a network there is nothing to scan: the address is saved, no scan."""
    entry = make_entry(hass)
    data = {k: v for k, v in entry.data.items() if k not in ("dialect", "network_id")}
    hass.config_entries.async_update_entry(entry, data={**data, "nodes": []})
    radio.stick = (GATEWAY_ID, False)  # stock firmware: auto is still fine
    form = await entry.start_reconfigure_flow(hass)

    result = await hass.config_entries.flow.async_configure(form["flow_id"], user_input)
    await hass.async_block_till_done()

    assert result["reason"] == "reconfigure_successful"
    assert not [call for call in radio.calls if call[0] == "discover"]
    assert "network_id" not in entry.data


# --- the site network ------------------------------------------------------------


async def test_site_network_id_hashes_instance_and_gateway(
    hass: HomeAssistant,
) -> None:
    """The site network id is stable per installation and gateway."""
    seed = f"{await instance_id.async_get(hass)}:{GATEWAY_ID}".encode()
    assert (
        await radio_pairing.async_site_network_id(hass, GATEWAY_ID)
        == (hashlib.sha256(seed).digest()[:2])
    )


# --- listen-only entries -----------------------------------------------------------


async def test_esp32_listen_only_entry(hass: HomeAssistant, radio: FakeRadio) -> None:
    """A monitor entry has its own unique id and never discovers or pairs."""
    result = await _choose(hass, await _gateway_menu(hass, FORM), "radio_monitor")

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["result"].unique_id == f"radio_monitor:{GATEWAY_ID}"
    assert result["title"] == "Radio monitor (10.0.0.5)"
    assert result["data"] == {
        "brand": "radio_monitor",
        "radio_type": "esp32",
        "host": "10.0.0.5",
        "port": 2323,
    }
    assert radio.calls == [("probe", "10.0.0.5", 2323)]


async def test_nanocul_listen_only_entry(hass: HomeAssistant, radio: FakeRadio) -> None:
    """A stock nanoCUL can be a monitor too."""
    radio.ports = [PORT]
    radio.stick = ("nanocul-x1", False)
    result = await _choose(hass, await _stick_menu(hass, STICK_FORM), "radio_monitor")

    assert result["result"].unique_id == "radio_monitor:nanocul-x1"
    assert result["title"] == f"Radio monitor ({PORT.device})"
    assert result["data"] == {
        "brand": "radio_monitor",
        "radio_type": "nanocul",
        "device": PORT.device,
        "radio_device_id": "nanocul-x1",
    }
    assert radio.calls == [("probe", PORT.device)]


# --- nanoCUL -------------------------------------------------------------------


@pytest.fixture
def stick(radio: FakeRadio) -> FakeRadio:
    """A detected stock nanoCUL hearing a dialect-A installation."""
    radio.ports = [PORT]
    radio.stick = ("nanocul-a1b2c3", False)
    radio.discover = (NetworkSighting(DIALECT_A, DIALECT_A.network_id), {6: None})
    return radio


async def test_port_choice_then_discovery_creates_entry(
    hass: HomeAssistant, stick: FakeRadio
) -> None:
    """A detected port is probed with its USB serial; the entry needs no scan."""
    result = await _choose(hass, await _stick_menu(hass, STICK_FORM), "radio_discover")

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["result"].unique_id == "radio:nanocul-a1b2c3"
    assert result["title"] == f"nanoCUL ({PORT.device})"
    assert result["data"] == {
        "brand": "radio",
        "radio_type": "nanocul",
        "device": PORT.device,
        "radio_device_id": "nanocul-a1b2c3",
        "nodes": [],
    }
    assert stick.calls == [("probe", PORT.device)]
    assert stick.serials == ["A1B2C3"]


async def test_manual_path_when_no_port_is_detected(
    hass: HomeAssistant, stick: FakeRadio
) -> None:
    """Choosing "manual" asks for a path or pyserial URL."""
    stick.ports = []
    manual = await _stick_menu(hass, {"device": "manual"})
    assert manual["step_id"] == "nanocul_manual"
    menu = await hass.config_entries.flow.async_configure(
        manual["flow_id"], {**STICK_FORM, "device": " socket://gw:2323 "}
    )
    result = await _choose(hass, menu, "radio_discover")
    assert result["data"]["device"] == "socket://gw:2323"
    assert stick.calls[0] == ("probe", "socket://gw:2323")
    assert stick.serials == [None]


@pytest.mark.parametrize(
    ("form", "probe", "error"),
    [
        (STICK_FORM, RadioLinkError("busy"), {"base": "cannot_connect_nanocul"}),
        (
            {**STICK_FORM, "advanced": {"dialect": "B"}},
            ("id", False),
            {"base": "dialect_unsupported_firmware"},
        ),
    ],
)
async def test_stick_errors_stay_on_the_form(
    hass: HomeAssistant, stick: FakeRadio, form: dict, probe: Any, error: dict
) -> None:
    """A busy stick, or a dialect its firmware lacks, keeps the form open."""
    stick.stick = probe
    result = await _stick_menu(hass, form)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "nanocul"
    assert result["errors"] == error


async def test_stick_step_aborts_for_a_configured_stick(
    hass: HomeAssistant, stick: FakeRadio
) -> None:
    """The same stick on another port is already configured: abort."""
    _stick_entry(hass, device_id="nanocul-a1b2c3")
    result = await _stick_menu(hass, STICK_FORM)
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_dialect_b_on_capable_firmware_is_accepted(
    hass: HomeAssistant, stick: FakeRadio
) -> None:
    """Firmware that can switch dialects may use dialect B."""
    stick.stick = (GATEWAY_ID, True)
    menu = await _stick_menu(hass, {**STICK_FORM, "advanced": {"dialect": "B"}})
    result = await _choose(hass, menu, "radio_discover")
    assert result["data"]["dialect"] == "B"
    assert "network_id" not in result["data"]  # learned after setup


async def test_stick_lost_during_a_rescan_shows_the_reconfigure_form(
    hass: HomeAssistant, stick: FakeRadio
) -> None:
    """A stick lost while scanning again shows its reconfigure form with the error."""
    entry = _stick_entry(hass, device_id="nanocul-a1b2c3")
    stick.discover = RadioLinkError("unplugged")
    stick.stick = ("nanocul-a1b2c3", True)
    form = await entry.start_reconfigure_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"device": STICK, "rescan": True}
    )
    result = await _finish_progress(hass, result)
    assert result["step_id"] == "reconfigure_nanocul"
    assert result["errors"] == {"base": "cannot_connect_nanocul"}


# --- radio helpers used by the flow ---------------------------------------------------


@pytest.fixture
def probe_link(monkeypatch: pytest.MonkeyPatch) -> type[ProbeLink]:
    """Let the flow's probes open ProbeLinks instead of real radio links."""
    monkeypatch.setattr(config_flow_radio, "RadioLink", ProbeLink)
    monkeypatch.setattr(ProbeLink, "created", [])
    return ProbeLink


async def test_probe_gateway_listens_without_acking(
    probe_link: type[ProbeLink], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gateway probe never acks and needs the gateway's MAC."""
    assert await config_flow_radio.probe_gateway("gw", 2323) == GATEWAY_ID
    assert probe_link.created[-1]["auto_ack"] is False

    monkeypatch.setattr(
        probe_link, "info", dataclasses.replace(probe_link.info, mac=None)
    )
    with pytest.raises(config_flow_radio.RadioSetupError, match="no_gateway_mac"):
        await config_flow_radio.probe_gateway("gw", 2323)


async def test_probe_nanocul(
    probe_link: type[ProbeLink], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stock firmware is named by its USB serial; newer firmware by its MAC."""
    monkeypatch.setattr(
        probe_link, "info", GatewayInfo("3.5", "869.525", "2DE5", False, 1, None, "")
    )
    assert await config_flow_radio.probe_nanocul("/dev/ttyUSB0", "A1B2C3") == (
        "nanocul-a1b2c3",
        False,
    )
    made = probe_link.created[-1]
    assert made["auto_ack"] is False and made["dialect"] is DIALECT_A
    assert callable(made["open_connection"])

    monkeypatch.setattr(
        probe_link,
        "info",
        GatewayInfo("3.6", "869.525", "2DE5", True, 1, "0A:0B:0C:0D:0E:0F", "", "A"),
    )
    assert await config_flow_radio.probe_nanocul("socket://gw:1") == (GATEWAY_ID, True)


async def test_probe_nanocul_with_this_repos_firmware(
    probe_link: type[ProbeLink], monkeypatch: pytest.MonkeyPatch
) -> None:
    """firmware/nanocul (3.7-nanocul) has no MAC but reports dialects: capable."""
    info = parse_query_line(
        "# Q termoweb_rx 3.7-nanocul freq=869.525 pa=C0 sync=2DE5 mode=dynamic "
        "autoack=off id=01 dialect=A net=1B30"
    )
    monkeypatch.setattr(probe_link, "info", info)
    assert await config_flow_radio.probe_nanocul("/dev/ttyUSB0", "A1B2C3") == (
        "nanocul-a1b2c3",
        True,
    )


def test_radio_link_factory_and_address() -> None:
    """Sticks open over serial on port 0; gateways over TCP."""
    nanocul = {"radio_type": "nanocul", "device": "/dev/ttyUSB0"}
    factory = config_flow_radio.radio_link_factory(nanocul)
    assert factory.func is RadioLink and "open_connection" in factory.keywords
    assert config_flow_radio.radio_address(nanocul) == ("/dev/ttyUSB0", 0)
    esp32 = {"host": "gw", "port": 2323}
    assert config_flow_radio.radio_link_factory(esp32) is RadioLink
    assert config_flow_radio.radio_address(esp32) == ("gw", 2323)


def test_serial_ports_resolve_to_stable_by_id_links(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A port is shown by its /dev/serial/by-id link when it has one."""
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    tty = tmp_path / "ttyUSB0"
    tty.write_text("")
    (by_id / "usb-SHK_NANO_CUL-port0").symlink_to(tty)
    (by_id / "usb-other").symlink_to(tmp_path / "ttyUSB9")
    monkeypatch.setattr(config_flow_radio, "SERIAL_BY_ID_DIR", str(by_id))
    assert config_flow_radio._by_id_path(str(tty)) == str(
        by_id / "usb-SHK_NANO_CUL-port0"
    )
    assert config_flow_radio._by_id_path("/dev/ttyS0") == "/dev/ttyS0"
    monkeypatch.setattr(
        config_flow_radio, "SERIAL_BY_ID_DIR", str(tmp_path / "missing")
    )
    assert config_flow_radio._by_id_path("/dev/ttyACM0") == "/dev/ttyACM0"


class DiscoveryRadio:
    """Scripted listen, probe and survey steps behind discover_radio."""

    def __init__(self) -> None:
        """Hear dialect B on NET from source 40; heater 6 answers."""
        self.sighting: Any = NetworkSighting(DIALECT_B, NET, frozenset({40}))
        self.heaters: dict[int, Any] = {6: "s"}
        self.surveyed: Any = None
        self.calls: list[tuple] = []

    async def discover_network(self, host, port, *, dialects, link_factory) -> Any:
        """Record the dialects listened for."""
        self.calls.append(("listen", tuple(d.name for d in dialects)))
        return self.sighting

    async def probe_heaters(
        self, host, port, dialect, network_id, candidates, link_factory
    ) -> Any:
        """Record the network probed and whether heard sources were included."""
        self.calls.append(
            ("probe", dialect.name, network_id, 40 in candidates, 2 in candidates)
        )
        return self.heaters

    async def survey_sighting(self, host, port, **kwargs) -> Any:
        """Return the scripted survey result."""
        self.calls.append(("survey",))
        return self.surveyed


@pytest.fixture
def air(monkeypatch: pytest.MonkeyPatch) -> DiscoveryRadio:
    """Patch the radio discovery primitives used by discover_radio."""
    fake = DiscoveryRadio()
    monkeypatch.setattr(config_flow_radio, "discover_network", fake.discover_network)
    monkeypatch.setattr(config_flow_radio, "probe_heaters", fake.probe_heaters)
    monkeypatch.setattr(config_flow_radio, "survey_sighting", fake.survey_sighting)
    return fake


@pytest.mark.parametrize(
    ("dialect", "network_id", "capable", "calls"),
    [
        ("auto", None, True, [("listen", ("B", "A")), ("probe", "B", NET, True, True)]),
        ("auto", None, False, [("listen", ("A",)), ("probe", "B", NET, True, True)]),
        ("B", None, True, [("listen", ("B",)), ("probe", "B", NET, True, True)]),
        ("A", None, True, [("probe", "A", DIALECT_A.network_id, False, True)]),
        ("B", NET, True, [("probe", "B", NET, False, True)]),
    ],
)
async def test_discover_radio_listens_only_when_it_must(
    air: DiscoveryRadio, dialect: str, network_id: Any, capable: bool, calls: list
) -> None:
    """A known network is probed directly; otherwise the air is listened to."""
    await config_flow_radio.discover_radio(
        "gw", 1, dialect, network_id, dialect_capable=capable
    )
    assert air.calls == calls


async def test_discover_radio_surveys_silent_air(air: DiscoveryRadio) -> None:
    """Nothing heard: a survey decides between no traffic and a known network."""
    air.sighting = None
    with pytest.raises(config_flow_radio.RadioSetupError, match="no_traffic"):
        await config_flow_radio.discover_radio("gw", 1, "auto", None)
    air.surveyed = NetworkSighting(DIALECT_B, NET)
    sighting, _ = await config_flow_radio.discover_radio("gw", 1, "auto", None)
    assert sighting.network_id == NET
    assert air.calls[-1] == ("probe", "B", NET, False, True)


async def test_discover_radio_without_heaters(air: DiscoveryRadio) -> None:
    """A network without answering heaters is an error."""
    air.heaters = {}
    with pytest.raises(config_flow_radio.RadioSetupError, match="no_heaters"):
        await config_flow_radio.discover_radio("gw", 1, "auto", None)


async def test_survey_is_analysed_inline_by_default() -> None:
    """Without Home Assistant's executor the survey is analysed on the loop."""
    assert (await config_flow_radio._analyse_inline([])).verdict == "silent"


def _report(verdict: str, dialect: str | None = None, nets: tuple[str, ...] = ()):
    """Return a survey report with the given outcome and no bursts."""
    return dataclasses.replace(
        analyse([]), verdict=verdict, dialect=dialect, network_ids=nets
    )


@pytest.mark.parametrize(
    ("bursts", "report", "expected"),
    [
        (None, None, None),
        (["b"], _report("silent"), None),
        (["b"], _report("known", "B", ("1234", "5678")), (DIALECT_B, NET)),
        (["b"], _report("known", "A"), (DIALECT_A, DIALECT_A.network_id)),
        (["b"], _report("known", "B"), None),
        (["b"], _report("known", "Z"), "raise"),
        (["b"], _report("undecodable"), "raise"),
        (["b"], _report("candidate"), "raise"),
    ],
)
async def test_survey_sighting_outcomes(
    monkeypatch: pytest.MonkeyPatch, bursts: Any, report: Any, expected: Any
) -> None:
    """Survey verdicts map to a known network, nothing, or an unknown dialect."""
    seen: list[Any] = []

    async def fake_survey(host, port, *, link_factory):
        seen.append((host, port, link_factory))
        return bursts

    async def fake_analyse(raw):
        seen.append(raw)
        return report

    monkeypatch.setattr(config_flow_radio, "survey_network", fake_survey)
    if expected == "raise":
        with pytest.raises(config_flow_radio.UnknownDialectError) as err:
            await config_flow_radio.survey_sighting(
                "gw", 1, analyse_survey=fake_analyse
            )
        assert err.value.reason == "unknown_dialect" and err.value.report is report
        return
    sighting = await config_flow_radio.survey_sighting(
        "gw", 1, analyse_survey=fake_analyse
    )
    if expected is None:
        assert sighting is None
    else:
        assert (sighting.dialect, sighting.network_id) == expected
    assert seen[0] == ("gw", 1, RadioLink)
    assert seen[1:] == ([] if bursts is None else [bursts])


# --- zeroconf discovery of the ESP32 gateway -----------------------------------------


def _zeroconf(
    host: str = "10.0.0.7", mac: str | None = "0A:0B:0C:0D:0E:0F", port: int = 2323
) -> ZeroconfServiceInfo:
    """Return the mDNS advertisement of the synthetic gateway."""
    properties = {"version": "3.7-esp32"}
    if mac is not None:
        properties["mac"] = mac.lower().replace(":", "")
    return ZeroconfServiceInfo(
        ip_address=ip_address(host),
        ip_addresses=[ip_address(host)],
        port=port,
        hostname="termoweb-radio.local.",
        type="_termoweb-radio._tcp.local.",
        name="termoweb-radio._termoweb-radio._tcp.local.",
        properties=properties,
    )


async def _discovered(
    hass: HomeAssistant, info: ZeroconfServiceInfo | None = None
) -> dict[str, Any]:
    """Start a zeroconf flow for ``info`` (the synthetic gateway by default)."""
    return await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_ZEROCONF}, data=info or _zeroconf()
    )


async def test_zeroconf_shows_confirm_without_connecting(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """The advertisement's MAC names the gateway; nothing is opened before confirm."""
    result = await _discovered(hass)

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "zeroconf_confirm"
    assert result["description_placeholders"] == {"host": "10.0.0.7"}
    assert radio.calls == []
    flow = hass.config_entries.flow.async_get(result["flow_id"])
    assert flow["context"]["unique_id"] == f"radio:{GATEWAY_ID}"
    assert flow["context"]["title_placeholders"] == {"name": "Radio gateway (10.0.0.7)"}
    assert flow["context"]["confirm_only"] is True


async def test_zeroconf_confirm_probes_then_offers_the_heater_menu(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """Confirming probes the gateway, then the entry is created at once."""
    form = await _discovered(hass)

    menu = await hass.config_entries.flow.async_configure(form["flow_id"], {})
    result = await _choose(hass, menu, "radio_discover")

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["result"].unique_id == f"radio:{GATEWAY_ID}"
    assert result["title"] == "Radio gateway (10.0.0.7)"
    assert (result["data"]["host"], result["data"]["port"]) == ("10.0.0.7", 2323)
    assert "network_id" not in result["data"]  # learned after setup
    assert radio.calls == [("probe", "10.0.0.7", 2323)]


async def test_zeroconf_moves_a_configured_gateway_to_its_new_address(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A known gateway at a new IP updates its entries and aborts, unprobed."""
    entry = _esp32_entry(hass)
    monitor = _esp32_entry(hass, brand=BRAND_RADIO_MONITOR)

    result = await _discovered(hass, _zeroconf(host="10.0.0.8", port=2424))

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"
    assert (entry.data["host"], entry.data["port"]) == ("10.0.0.8", 2424)
    assert entry.data["nodes"] == NODES
    assert (monitor.data["host"], monitor.data["port"]) == ("10.0.0.8", 2424)
    assert radio.calls == []


async def test_zeroconf_reloads_loaded_entries_at_a_new_address(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """Loaded entries are reloaded so they reconnect to the new address."""
    entry = _esp32_entry(hass)
    monitor = _esp32_entry(hass, brand=BRAND_RADIO_MONITOR)
    for each in (entry, monitor):
        each.mock_state(hass, ConfigEntryState.LOADED)
    with patch.object(hass.config_entries, "async_schedule_reload") as reload:
        result = await _discovered(hass, _zeroconf(host="10.0.0.8"))
        unchanged = await _discovered(hass, _zeroconf(host="10.0.0.8"))

    assert result["reason"] == unchanged["reason"] == "already_configured"
    assert sorted(call.args[0] for call in reload.call_args_list) == sorted(
        [entry.entry_id, monitor.entry_id]
    )


async def test_zeroconf_without_mac_probes_for_the_gateway_id(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """An advertisement without a MAC is identified by probing the gateway."""
    result = await _discovered(hass, _zeroconf(mac=None))

    assert result["step_id"] == "zeroconf_confirm"
    assert radio.calls == [("probe", "10.0.0.7", 2323)]
    flow = hass.config_entries.flow.async_get(result["flow_id"])
    assert flow["context"]["unique_id"] == f"radio:{GATEWAY_ID}"


@pytest.mark.parametrize(
    ("probe", "reason"),
    [
        (RadioLinkError("down"), "cannot_connect_radio"),
        (config_flow_radio.RadioSetupError("no_gateway_mac"), "no_gateway_mac"),
    ],
)
async def test_zeroconf_without_mac_aborts_when_the_probe_fails(
    hass: HomeAssistant, radio: FakeRadio, probe: Exception, reason: str
) -> None:
    """A gateway that cannot be identified is not offered."""
    radio.gateway = probe
    result = await _discovered(hass, _zeroconf(mac=None))
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == reason


async def test_zeroconf_ignores_ipv6_advertisements(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """Only an IPv4 address can be used to reach the gateway."""
    result = await _discovered(hass, _zeroconf(host="fe80::1"))
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "not_ipv4_address"
    assert radio.calls == []


@pytest.mark.parametrize(
    ("probe", "error"),
    [
        (RadioLinkError("down"), "cannot_connect_radio"),
        (config_flow_radio.RadioSetupError("no_gateway_mac"), "no_gateway_mac"),
    ],
)
async def test_zeroconf_confirm_probe_errors_stay_on_confirm(
    hass: HomeAssistant, radio: FakeRadio, probe: Exception, error: str
) -> None:
    """A gateway that stops answering keeps the confirm step open with the error."""
    form = await _discovered(hass)
    radio.gateway = probe
    result = await hass.config_entries.flow.async_configure(form["flow_id"], {})
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "zeroconf_confirm"
    assert result["errors"] == {"base": error}


async def test_zeroconf_confirm_trusts_the_probed_gateway_id(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A probe naming another (configured) gateway aborts instead of duplicating."""
    _esp32_entry(hass)  # GATEWAY_ID is configured
    form = await _discovered(hass, _zeroconf(mac="0F:0E:0D:0C:0B:0A"))
    assert form["step_id"] == "zeroconf_confirm"

    result = await hass.config_entries.flow.async_configure(form["flow_id"], {})

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_zeroconf_confirm_adopts_a_new_probed_gateway_id(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A probe naming an unconfigured gateway continues under its id."""
    form = await _discovered(hass, _zeroconf(mac="0F:0E:0D:0C:0B:0A"))
    menu = await hass.config_entries.flow.async_configure(form["flow_id"], {})
    result = await _choose(hass, menu, "radio_monitor")
    assert result["result"].unique_id == f"radio_monitor:{GATEWAY_ID}"


# --- nanoCUL port pre-selection ---------------------------------------------------


def _port(device: str, description: str = "USB Serial", **kwargs: Any) -> Any:
    """Return a detected serial port (USB unless told otherwise)."""
    kwargs.setdefault("usb", True)
    return config_flow_radio.SerialPort(device, description, None, **kwargs)


ZIGBEE = _port("/dev/serial/by-id/usb-ITead_Sonoff_Zigbee_3.0-if00-port0")
CH340 = _port("/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0")
FTDI = _port(
    "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A1B2C3-if00-port0",
    manufacturer="FTDI",
    product="FT232R USB UART",
)
UART = _port("/dev/ttyAMA0", "ttyAMA0", usb=False)


@pytest.mark.parametrize(
    ("ports", "used", "expected"),
    [
        ([ZIGBEE, PORT], set(), PORT.device),  # "NANO_CUL" in the by-id name
        ([ZIGBEE, _port("/dev/ttyUSB1", "nanoCUL 868")], set(), "/dev/ttyUSB1"),
        ([CH340, _port("/dev/ttyACM0", product="culfw CUL868")], set(), "/dev/ttyACM0"),
        ([PORT, CH340], {PORT.device}, CH340.device),  # the CUL is taken
        ([FTDI, UART], set(), FTDI.device),  # the only USB serial port
        ([ZIGBEE, FTDI], set(), None),  # two candidates: let the user choose
        ([ZIGBEE, FTDI], {ZIGBEE.device}, FTDI.device),
        ([_port("/dev/ttyUSB0", "Circular buffer"), CH340], set(), None),
        ([UART], set(), None),
        ([], set(), None),
    ],
)
def test_likely_nanocul_port(
    ports: list[Any], used: set[str], expected: str | None
) -> None:
    """A CUL-named port wins, else the only free USB serial port, else none."""
    assert config_flow_radio.likely_nanocul_port(ports, used) == expected


def _default_device(form: dict[str, Any]) -> Any:
    """Return the pre-selected port of a nanoCUL form (None when blank)."""
    for key in form["data_schema"].schema:
        if str(key) == "device":
            default = key.default
            return None if default is vol.UNDEFINED else default()
    raise AssertionError("no device field")


@pytest.mark.parametrize(
    ("ports", "expected"),
    [
        ([ZIGBEE, PORT], PORT.device),
        ([FTDI, UART], FTDI.device),
        ([ZIGBEE, CH340], None),
        ([], "manual"),
    ],
)
async def test_nanocul_form_preselects_the_likely_port(
    hass: HomeAssistant, radio: FakeRadio, ports: list[Any], expected: str | None
) -> None:
    """The form opens with the stick chosen when it can tell; no port is opened."""
    radio.ports = ports
    form = await _menu(hass, "nanocul")
    assert form["step_id"] == "nanocul"
    assert _default_device(form) == expected
    assert {str(key) for key in form["data_schema"].schema} == {"device", "advanced"}
    assert radio.calls == []


async def test_nanocul_form_skips_the_port_of_a_configured_stick(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A port another TermoWeb entry uses is not pre-selected."""
    _esp32_entry(hass)  # an ESP32 entry has no port to exclude
    MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_BRAND: BRAND_RADIO_MONITOR,
            "radio_type": "nanocul",
            "device": PORT.device,
        },
    ).add_to_hass(hass)
    radio.ports = [PORT, FTDI]
    form = await _menu(hass, "nanocul")
    assert _default_device(form) == FTDI.device


def test_list_serial_ports_reads_usb_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detected ports carry their USB strings and whether they are USB at all."""
    usb = SimpleNamespace(
        device="/dev/ttyUSB0",
        description="FT232R USB UART",
        serial_number="A1B2C3",
        manufacturer="FTDI",
        product="FT232R USB UART",
        vid=0x0403,
    )
    builtin = SimpleNamespace(
        device="/dev/ttyS0",
        description=None,
        serial_number=None,
        manufacturer=None,
        product=None,
        vid=None,
    )
    tools = SimpleNamespace(list_ports=SimpleNamespace(comports=lambda: [usb, builtin]))
    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(tools=tools))
    monkeypatch.setitem(sys.modules, "serial.tools", tools)
    monkeypatch.setattr(config_flow_radio, "SERIAL_BY_ID_DIR", "/nonexistent")

    assert config_flow_radio.list_serial_ports() == [
        config_flow_radio.SerialPort(
            "/dev/ttyUSB0", "FT232R USB UART", "A1B2C3", "FTDI", "FT232R USB UART", True
        ),
        config_flow_radio.SerialPort(
            "/dev/ttyS0", "/dev/ttyS0", None, None, None, False
        ),
    ]
