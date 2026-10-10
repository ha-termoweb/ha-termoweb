"""Radio and nanoCUL config and reconfigure flows on real Home Assistant."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.config_entries import SOURCE_USER, ConfigEntryDisabler
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb import config_flow
from custom_components.termoweb.backend.radio import DIALECT_B, RadioLinkError
from custom_components.termoweb.backend.radio.discovery import NetworkSighting
from custom_components.termoweb.const import (
    BRAND_RADIO,
    BRAND_RADIO_MONITOR,
    CONF_BRAND,
    DOMAIN,
)

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
        self.calls: list[tuple[Any, ...]] = []

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
        if isinstance(self.stick, Exception):
            raise self.stick
        return self.stick

    async def discover_radio(self, *args: Any, **kwargs: Any) -> Any:
        """Return (or raise) the configured discovery result."""
        self.calls.append(("discover", *args[:2]))
        if isinstance(self.discover, Exception):
            raise self.discover
        return self.discover


@pytest.fixture
def radio() -> Generator[FakeRadio]:
    """Patch the radio probes and discovery; never set up a real radio entry."""
    fake = FakeRadio()
    with (
        patch.object(config_flow, "probe_gateway", fake.probe_gateway),
        patch.object(config_flow, "probe_nanocul", fake.probe_nanocul),
        patch.object(config_flow, "discover_radio", fake.discover_radio),
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


@pytest.mark.parametrize(
    ("brand", "host"),
    [(BRAND_RADIO, "10.0.0.5"), (BRAND_RADIO_MONITOR, "10.0.0.5")],
)
async def test_radio_step_refuses_gateway_in_use(
    hass: HomeAssistant, radio: FakeRadio, brand: str, host: str
) -> None:
    """A gateway another entry talks to is not probed (the bridge would drop it)."""
    _esp32_entry(hass, brand=brand, host=host)
    form = await _menu(hass, "radio")

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"host": " 10.0.0.5 ", "port": 2323, "dialect": "auto"}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_in_use"
    assert radio.calls == []


async def test_radio_step_ignores_disabled_entry_and_other_port(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A disabled entry, or one on another port, does not block the probe."""
    _esp32_entry(hass, disabled_by=ConfigEntryDisabler.USER)
    form = await _menu(hass, "radio")
    radio.gateway = OTHER_GATEWAY_ID

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"host": "10.0.0.5", "port": 2323, "dialect": "auto"}
    )

    assert result["type"] is FlowResultType.MENU
    assert radio.calls == [("probe", "10.0.0.5", 2323)]


async def test_nanocul_step_refuses_stick_in_use(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A stick another entry reads is not opened a second time."""
    _stick_entry(hass)
    with patch.object(config_flow, "list_serial_ports", return_value=[]):
        form = await _menu(hass, "nanocul")
        form = await hass.config_entries.flow.async_configure(
            form["flow_id"], {"device": "manual"}
        )
        result = await hass.config_entries.flow.async_configure(
            form["flow_id"], {"device": f" {STICK} ", "dialect": "auto"}
        )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_in_use"
    assert radio.calls == []


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
    radio.discover = config_flow.RadioSetupError("no_heaters")
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


async def test_reconfigure_radio_refuses_address_of_other_entry(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """The new address must not be one that another entry already uses."""
    entry = _esp32_entry(hass)
    _esp32_entry(hass, brand=BRAND_RADIO_MONITOR, host="10.0.0.9")
    form = await entry.start_reconfigure_flow(hass)

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"host": "10.0.0.9", "port": 2323, "rescan": False}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_in_use"
    assert radio.calls == []


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RadioLinkError("down"), "cannot_connect_radio"),
        (config_flow.RadioSetupError("no_gateway_mac"), "no_gateway_mac"),
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


async def test_reconfigure_nanocul_refuses_port_of_other_entry(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """The new port must not be one that another entry already reads."""
    entry = _stick_entry(hass)
    MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_BRAND: BRAND_RADIO_MONITOR,
            "radio_type": "nanocul",
            "device": "/dev/ttyUSB1",
        },
    ).add_to_hass(hass)
    form = await entry.start_reconfigure_flow(hass)

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"device": "/dev/ttyUSB1", "rescan": False}
    )

    assert result["reason"] == "already_in_use"
    assert radio.calls == []


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
