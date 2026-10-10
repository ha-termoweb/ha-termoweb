"""Tests for the config and options flows."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from aiohttp import ClientError
from homeassistant.config_entries import (
    SOURCE_USER,
    ConfigEntryDisabler,
    ConfigEntryState,
)
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType, InvalidData
from homeassistant.helpers import instance_id
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb import (
    async_migrate_entry,
    config_flow,
    radio_pairing,
    radio_survey,
)
from custom_components.termoweb.backend.radio import (
    DIALECT_A,
    DIALECT_B,
    GatewayInfo,
    RadioLink,
    RadioLinkError,
)
from custom_components.termoweb.backend.radio.discovery import NetworkSighting
from custom_components.termoweb.backend.radio.pairing import (
    PAIRING_DIALECTS,
    NoFreeAddressError,
    PairedHeater,
)
from custom_components.termoweb.backend.radio.survey import analyse
from custom_components.termoweb.backend.rest_client import (
    BackendAuthError,
    BackendRateLimitError,
)
from custom_components.termoweb.const import (
    BRAND_DUCAHEAT,
    BRAND_RADIO,
    BRAND_RADIO_MONITOR,
    BRAND_TERMOWEB,
    BRAND_TEVOLVE,
    CONF_BRAND,
    DEFAULT_BRAND,
    DOMAIN,
)
from tests.fakes.cloud import PASSWORD, USERNAME, VERSION, FakeCloud
from tests.fakes.radio_link import ProbeLink
from tests.fakes.radio_setup import record_reloads
from tests.fakes.runtime import build_entry_runtime

MINOR_VERSION = config_flow.TermoWebConfigFlow.MINOR_VERSION


async def _cloud_form(hass: HomeAssistant) -> dict[str, Any]:
    """Start a user flow and pick the cloud menu option."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.MENU
    assert result["menu_options"] == ["cloud", "radio", "nanocul"]
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": "cloud"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "cloud"
    assert result["errors"] is None
    assert result["description_placeholders"] == {"version": VERSION}
    assert _defaults(result) == {
        CONF_BRAND: DEFAULT_BRAND,
        "username": "",
        "password": None,
    }
    return result


def _defaults(result: dict[str, Any]) -> dict[str, Any]:
    """Return the form defaults by field name (None when a field has none)."""
    defaults: dict[str, Any] = {}
    for key in result["data_schema"].schema:
        default = getattr(key, "default", None)
        defaults[str(key)] = default() if callable(default) else None
    return defaults


def _login(brand: str = BRAND_TERMOWEB, username: str = USERNAME) -> dict[str, str]:
    """Return the cloud login form input."""
    return {CONF_BRAND: brand, "username": username, "password": PASSWORD}


@pytest.mark.parametrize(
    ("brand", "title", "unique_id"),
    [
        (BRAND_TERMOWEB, f"TermoWeb ({USERNAME})", USERNAME),
        (BRAND_DUCAHEAT, f"Ducaheat ({USERNAME})", f"ducaheat:{USERNAME}"),
        (BRAND_TEVOLVE, f"Tevolve ({USERNAME})", f"ducaheat:{USERNAME}"),
    ],
)
async def test_cloud_step_creates_entry(
    hass: HomeAssistant, cloud: FakeCloud, brand: str, title: str, unique_id: str
) -> None:
    """Valid credentials create an entry keyed by backend and username."""
    form = await _cloud_form(hass)

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], _login(brand, f"  {USERNAME}  ")
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == title
    assert result["data"]["username"] == USERNAME
    assert result["data"]["password"] == PASSWORD
    assert result["data"][CONF_BRAND] == brand
    entry = result["result"]
    assert entry.unique_id == unique_id
    assert cloud.list_devices.await_count >= 1


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (BackendAuthError("bad credentials"), "invalid_auth"),
        (BackendRateLimitError("slow down"), "rate_limited"),
        (ClientError("offline"), "cannot_connect"),
        (RuntimeError("boom"), "unknown"),
        (TimeoutError(), "cannot_connect"),
    ],
)
async def test_cloud_step_errors_then_recovers(
    hass: HomeAssistant, cloud: FakeCloud, error: Exception, expected: str
) -> None:
    """A failed login re-shows the form with an error; a retry can succeed."""
    form = await _cloud_form(hass)
    cloud.list_devices.side_effect = error

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], _login(BRAND_DUCAHEAT, "  trouble  ")
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "cloud"
    assert result["errors"] == {"base": expected}
    assert result["description_placeholders"] == {"version": VERSION}
    defaults = _defaults(result)
    assert defaults["username"] == "trouble"
    assert defaults[CONF_BRAND] == BRAND_DUCAHEAT
    assert not hass.config_entries.async_entries(DOMAIN)

    cloud.list_devices.side_effect = None
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _login(BRAND_DUCAHEAT)
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY


async def test_cloud_step_aborts_when_account_already_configured(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Adding the same account twice aborts with already_configured."""
    config_entry.add_to_hass(hass)
    form = await _cloud_form(hass)

    result = await hass.config_entries.flow.async_configure(form["flow_id"], _login())

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"
    assert len(hass.config_entries.async_entries(DOMAIN)) == 1


async def test_cloud_step_aborts_on_differently_cased_username(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """The same account typed with different case is still a duplicate."""
    config_entry.add_to_hass(hass)
    form = await _cloud_form(hass)

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], _login(username=USERNAME.upper())
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"
    assert len(hass.config_entries.async_entries(DOMAIN)) == 1


async def test_cloud_entry_unique_id_is_case_folded(
    hass: HomeAssistant, cloud: FakeCloud
) -> None:
    """A new entry stores the case-folded account as its unique id."""
    form = await _cloud_form(hass)

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], _login(BRAND_DUCAHEAT, "User@Example.COM")
    )

    assert result["result"].unique_id == "ducaheat:user@example.com"
    assert result["result"].minor_version == MINOR_VERSION


async def test_cloud_step_stores_only_credentials(
    hass: HomeAssistant, cloud: FakeCloud
) -> None:
    """The entry data holds the login fields and nothing else."""
    form = await _cloud_form(hass)

    result = await hass.config_entries.flow.async_configure(form["flow_id"], _login())

    assert result["data"] == _login()


async def test_reconfigure_shows_current_values(
    hass: HomeAssistant, cloud: FakeCloud
) -> None:
    """The reconfigure form defaults to the entry's username and brand."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id=f"ducaheat:{USERNAME}",
        data={"username": USERNAME, "password": PASSWORD, CONF_BRAND: BRAND_DUCAHEAT},
    )
    entry.add_to_hass(hass)

    result = await entry.start_reconfigure_flow(hass)

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    assert _defaults(result) == {
        CONF_BRAND: BRAND_DUCAHEAT,
        "username": USERNAME,
        "password": None,
    }


async def test_reconfigure_updates_entry(hass: HomeAssistant, cloud: FakeCloud) -> None:
    """Reconfigure saves a new password for the same account and keeps other data."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id=f"ducaheat:{USERNAME}",
        minor_version=2,
        data={
            "username": USERNAME,
            "password": "old",
            CONF_BRAND: BRAND_DUCAHEAT,
            "other": "keep",
        },
        options={"extra": True},
    )
    entry.add_to_hass(hass)
    form = await entry.start_reconfigure_flow(hass)

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], _login(BRAND_DUCAHEAT, USERNAME.upper())
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data["username"] == USERNAME.upper()
    assert entry.data["password"] == PASSWORD
    assert entry.data[CONF_BRAND] == BRAND_DUCAHEAT
    assert entry.data["other"] == "keep"
    assert entry.options == {"extra": True}
    assert entry.unique_id == f"ducaheat:{USERNAME}"


@pytest.mark.parametrize(
    ("brand", "username"),
    [
        (BRAND_TERMOWEB, "other@example.com"),
        (BRAND_DUCAHEAT, USERNAME),
        (BRAND_TEVOLVE, USERNAME),
    ],
)
async def test_reconfigure_to_another_account_aborts(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    brand: str,
    username: str,
) -> None:
    """Reconfigure cannot move an entry to a different account or brand."""
    config_entry.add_to_hass(hass)
    before = dict(config_entry.data)
    form = await config_entry.start_reconfigure_flow(hass)
    cloud.list_devices.reset_mock()

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], _login(brand, username)
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "unique_id_mismatch"
    assert dict(config_entry.data) == before
    assert config_entry.unique_id == USERNAME
    cloud.list_devices.assert_not_awaited()


async def test_reconfigure_defaults_unknown_brand(
    hass: HomeAssistant, cloud: FakeCloud
) -> None:
    """An entry with an unknown stored brand offers the default brand."""
    entry = MockConfigEntry(
        domain=DOMAIN, unique_id=USERNAME, data={"username": USERNAME, CONF_BRAND: "x"}
    )
    entry.add_to_hass(hass)

    result = await entry.start_reconfigure_flow(hass)

    assert _defaults(result)[CONF_BRAND] == DEFAULT_BRAND


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (BackendAuthError("bad credentials"), "invalid_auth"),
        (BackendRateLimitError("slow down"), "rate_limited"),
        (ClientError("offline"), "cannot_connect"),
        (TimeoutError(), "cannot_connect"),
        (RuntimeError("boom"), "unknown"),
    ],
)
async def test_reconfigure_error_keeps_entry(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    error: Exception,
    expected: str,
) -> None:
    """A failed login during reconfigure re-shows the form, entry unchanged."""
    config_entry.add_to_hass(hass)
    before = dict(config_entry.data)
    form = await config_entry.start_reconfigure_flow(hass)
    cloud.list_devices.side_effect = error

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], _login(BRAND_TERMOWEB, f" {USERNAME.upper()} ")
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    assert result["errors"] == {"base": expected}
    assert result["description_placeholders"] == {"version": VERSION}
    defaults = _defaults(result)
    assert defaults["username"] == USERNAME.upper()
    assert defaults[CONF_BRAND] == BRAND_TERMOWEB
    assert dict(config_entry.data) == before


async def test_reconfigure_reloads_loaded_entry(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Saving new credentials reloads the running entry."""
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    runtime_before = config_entry.runtime_data
    form = await config_entry.start_reconfigure_flow(hass)

    await hass.config_entries.flow.async_configure(
        form["flow_id"], _login(username=USERNAME)
    )
    await hass.async_block_till_done()

    assert config_entry.state is ConfigEntryState.LOADED
    assert config_entry.runtime_data is not runtime_before


async def _start_reauth(
    hass: HomeAssistant, cloud: FakeCloud, entry: MockConfigEntry
) -> dict[str, Any]:
    """Fail setup with rejected credentials and return the reauth form."""
    cloud.list_devices.side_effect = BackendAuthError("bad credentials")
    entry.add_to_hass(hass)
    assert not await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    [flow] = hass.config_entries.flow.async_progress_by_handler(DOMAIN)
    result = await hass.config_entries.flow.async_configure(flow["flow_id"])
    assert result["step_id"] == "reauth_confirm"
    placeholders = result["description_placeholders"]
    assert placeholders["username"] == USERNAME
    assert placeholders["brand"] == "TermoWeb"
    return result


async def test_reauth_saves_password_and_reloads(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """A new password after an auth failure is saved and the entry loads."""
    form = await _start_reauth(hass, cloud, config_entry)
    cloud.list_devices.side_effect = None

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"password": "new-secret"}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert config_entry.data["password"] == "new-secret"
    assert config_entry.data["username"] == USERNAME
    assert config_entry.state is ConfigEntryState.LOADED


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (BackendAuthError("still bad"), "invalid_auth"),
        (TimeoutError(), "cannot_connect"),
    ],
)
async def test_reauth_error_shows_form(
    hass: HomeAssistant,
    cloud: FakeCloud,
    config_entry: MockConfigEntry,
    error: Exception,
    expected: str,
) -> None:
    """A failed reauth login re-shows the form and keeps the old password."""
    form = await _start_reauth(hass, cloud, config_entry)
    cloud.list_devices.side_effect = error

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], {"password": "wrong"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"
    assert result["errors"] == {"base": expected}
    assert config_entry.data["password"] == PASSWORD


async def test_migration_case_folds_unique_id_and_drops_poll_interval(
    hass: HomeAssistant, cloud: FakeCloud
) -> None:
    """Version 1.1 entries get a case-folded unique id and lose poll_interval."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id="User@Example.com",
        data={
            "username": "User@Example.com",
            "password": PASSWORD,
            CONF_BRAND: BRAND_TERMOWEB,
            "poll_interval": 90,
        },
        options={"poll_interval": 120, "keep": 1},
    )
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert (entry.version, entry.minor_version) == (1, MINOR_VERSION)
    assert entry.unique_id == "user@example.com"
    assert entry.data["username"] == "User@Example.com"
    assert "poll_interval" not in entry.data
    assert entry.options == {"keep": 1}


async def test_migration_keeps_unique_id_on_collision(
    hass: HomeAssistant, cloud: FakeCloud, caplog: pytest.LogCaptureFixture
) -> None:
    """Two old entries for one account keep distinct ids and the user is told."""
    entries = [
        MockConfigEntry(
            domain=DOMAIN,
            title=f"TermoWeb ({name})",
            unique_id=name,
            data={"username": name, "password": PASSWORD, CONF_BRAND: BRAND_TERMOWEB},
        )
        for name in (USERNAME, USERNAME.upper())
    ]
    for entry in entries:
        entry.add_to_hass(hass)

    # Setting up the integration sets up (and migrates) both entries in order.
    await hass.config_entries.async_setup(entries[0].entry_id)
    await hass.async_block_till_done()

    assert [entry.unique_id for entry in entries] == [USERNAME, USERNAME.upper()]
    assert [entry.minor_version for entry in entries] == [MINOR_VERSION] * 2
    assert "is the same account as entry" in caplog.text


@pytest.mark.parametrize(
    ("existing", "added"),
    [(BRAND_DUCAHEAT, BRAND_TEVOLVE), (BRAND_TEVOLVE, BRAND_DUCAHEAT)],
)
async def test_ducaheat_and_tevolve_are_one_account(
    hass: HomeAssistant, cloud: FakeCloud, existing: str, added: str
) -> None:
    """Ducaheat and Tevolve share one backend: one account cannot be added twice."""
    form = await _cloud_form(hass)
    await hass.config_entries.flow.async_configure(form["flow_id"], _login(existing))
    await hass.async_block_till_done()
    form = await _cloud_form(hass)

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], _login(added, USERNAME.upper())
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"
    assert len(hass.config_entries.async_entries(DOMAIN)) == 1


async def test_migration_scopes_tevolve_unique_id_by_backend(
    hass: HomeAssistant,
) -> None:
    """Version 1.3 Tevolve entries move to the shared ``ducaheat:`` unique id."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        minor_version=3,
        unique_id=f"tevolve:{USERNAME}",
        data={"username": USERNAME, "password": PASSWORD, CONF_BRAND: BRAND_TEVOLVE},
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry)

    assert entry.unique_id == f"ducaheat:{USERNAME}"
    assert entry.minor_version == MINOR_VERSION


async def test_migration_keeps_tevolve_unique_id_when_ducaheat_has_the_account(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """The same account under both brands keeps both ids and the user is told."""
    ducaheat = MockConfigEntry(
        domain=DOMAIN,
        minor_version=3,
        title=f"Ducaheat ({USERNAME})",
        unique_id=f"ducaheat:{USERNAME}",
        data={"username": USERNAME, "password": PASSWORD, CONF_BRAND: BRAND_DUCAHEAT},
    )
    tevolve = MockConfigEntry(
        domain=DOMAIN,
        minor_version=3,
        title=f"Tevolve ({USERNAME})",
        unique_id=f"tevolve:{USERNAME}",
        data={"username": USERNAME, "password": PASSWORD, CONF_BRAND: BRAND_TEVOLVE},
    )
    ducaheat.add_to_hass(hass)
    tevolve.add_to_hass(hass)

    assert await async_migrate_entry(hass, tevolve)
    assert await async_migrate_entry(hass, ducaheat)

    assert tevolve.unique_id == f"tevolve:{USERNAME}"
    assert ducaheat.unique_id == f"ducaheat:{USERNAME}"
    assert [tevolve.minor_version, ducaheat.minor_version] == [MINOR_VERSION] * 2
    assert (
        f"Entry 'Tevolve ({USERNAME})' is the same account as entry "
        f"'Ducaheat ({USERNAME})'" in caplog.text
    )


@pytest.mark.parametrize("brand", [BRAND_RADIO, BRAND_RADIO_MONITOR])
async def test_migration_leaves_radio_unique_id(
    hass: HomeAssistant, brand: str
) -> None:
    """Radio entries keep their unique id; only the version moves on."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id=f"{brand}:0A0B0C0D0E0F",
        data={CONF_BRAND: brand, "username": "Not-An-Account"},
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry)

    assert entry.unique_id == f"{brand}:0A0B0C0D0E0F"
    assert entry.minor_version == MINOR_VERSION


async def test_migration_of_current_entry_is_a_no_op(hass: HomeAssistant) -> None:
    """An entry already at the current version is left alone."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        minor_version=MINOR_VERSION,
        unique_id="Mixed",
        data={"username": "Mixed"},
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry)

    assert entry.unique_id == "Mixed"


@pytest.mark.parametrize("minor_version", [1, 2])
async def test_migration_drops_supports_diagnostics(
    hass: HomeAssistant, minor_version: int
) -> None:
    """F2: the fictional supports_diagnostics flag is removed from entry data."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        minor_version=minor_version,
        unique_id=f"radio:{'0a' * 6}",
        data={CONF_BRAND: BRAND_RADIO, "host": "h", "supports_diagnostics": True},
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry)

    assert dict(entry.data) == {CONF_BRAND: BRAND_RADIO, "host": "h"}
    assert entry.minor_version == MINOR_VERSION


async def test_migration_refuses_newer_version(hass: HomeAssistant) -> None:
    """An entry written by a newer major version is not touched."""
    entry = MockConfigEntry(domain=DOMAIN, version=2, data={CONF_BRAND: BRAND_RADIO})
    entry.add_to_hass(hass)

    assert not await async_migrate_entry(hass, entry)


@pytest.mark.parametrize(
    ("brand", "supported"),
    [
        (BRAND_TERMOWEB, False),
        (BRAND_DUCAHEAT, False),
        (BRAND_TEVOLVE, False),
        (BRAND_RADIO_MONITOR, False),
        (BRAND_RADIO, True),
    ],
)
async def test_options_flow_offered_only_for_radio_entries(
    hass: HomeAssistant, brand: str, supported: bool
) -> None:
    """Cloud and listen-only entries have no options, so HA hides "Configure"."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_BRAND: brand})
    entry.add_to_hass(hass)

    assert entry.supports_options is supported


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
        self.paired: Any = (DIALECT_B, [_paired(2), _paired(3)])
        self.ports: list[config_flow.SerialPort] = []
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

    async def pair_new_network(
        self, host: str, port: int, network_id: bytes, **kwargs: Any
    ) -> Any:
        """Return (or raise) the configured pairing result."""
        self.calls.append(("pair", host, port, network_id))
        self.kwargs.append(kwargs)
        if isinstance(self.paired, Exception):
            raise self.paired
        return self.paired


def _paired(node_id: int) -> PairedHeater:
    """Return a heater paired as ``node_id``."""
    return PairedHeater(node_id, None)


@pytest.fixture
def radio() -> Generator[FakeRadio]:
    """Patch the radio probes and discovery; never set up a real radio entry."""
    fake = FakeRadio()
    with (
        patch.object(config_flow, "probe_gateway", fake.probe_gateway),
        patch.object(config_flow, "probe_nanocul", fake.probe_nanocul),
        patch.object(config_flow, "discover_radio", fake.discover_radio),
        patch.object(config_flow, "pair_new_network", fake.pair_new_network),
        patch.object(config_flow, "list_serial_ports", lambda: fake.ports),
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


# --- setting up a radio entry ----------------------------------------------------

FORM = {"host": "10.0.0.5", "port": 2323, "dialect": "auto", "network_id": ""}
PORT = config_flow.SerialPort(
    "/dev/serial/by-id/usb-SHK_NANO_CUL_868-if00-port0", "nanoCUL 868", "A1B2C3"
)
STICK_FORM = {"device": PORT.device, "dialect": "auto", "network_id": ""}


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
    assert menu["menu_options"] == ["radio_discover", "radio_pair", "radio_monitor"]
    result = await hass.config_entries.flow.async_configure(
        menu["flow_id"], {"next_step_id": option}
    )
    return await _finish_progress(hass, result)


async def _pair(hass: HomeAssistant, menu: dict[str, Any]) -> dict[str, Any]:
    """Choose pairing, confirm the explanation and return the final result."""
    form = await _choose(hass, menu, "radio_pair")
    assert form["step_id"] == "radio_pair"
    result = await hass.config_entries.flow.async_configure(form["flow_id"], {})
    return await _finish_progress(hass, result)


async def test_user_step_offers_cloud_or_radio(hass: HomeAssistant) -> None:
    """The first step is a menu: cloud account, ESP32 gateway or nanoCUL."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": SOURCE_USER}
    )
    assert result["type"] is FlowResultType.MENU
    assert result["menu_options"] == ["cloud", "radio", "nanocul"]


async def test_radio_discovery_creates_entry(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """Gateway form, discovery, then an entry with the heaters found."""
    radio.discover = (
        NetworkSighting(DIALECT_B, NET, frozenset({6})),
        {6: None, 3: None},
    )
    menu = await _gateway_menu(hass, FORM)

    result = await _choose(hass, menu, "radio_discover")

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Radio gateway (10.0.0.5)"
    assert result["result"].unique_id == f"radio:{GATEWAY_ID}"
    assert result["data"] == {
        "brand": "radio",
        "radio_type": "esp32",
        "host": "10.0.0.5",
        "port": 2323,
        "dialect": "B",
        "network_id": "1234",
        "nodes": [
            {"type": "htr", "addr": "3", "name": "Heater 3"},
            {"type": "htr", "addr": "6", "name": "Heater 6"},
        ],
    }
    assert radio.calls == [("probe", "10.0.0.5", 2323), ("discover", "10.0.0.5", 2323)]
    assert radio.kwargs[0]["args"] == ("auto", None)


async def test_radio_discovery_uses_a_manual_network_id(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A typed network id (any separators) is passed to discovery."""
    menu = await _gateway_menu(hass, {**FORM, "dialect": "B", "network_id": "12 34"})
    result = await _choose(hass, menu, "radio_discover")
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert radio.kwargs[0]["args"] == ("B", NET)


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
        (config_flow.RadioSetupError("no_gateway_mac"), "no_gateway_mac"),
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


async def test_radio_form_rejects_bad_network_id(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A malformed network id is rejected before the gateway is probed."""
    result = await _gateway_menu(hass, {**FORM, "network_id": "XYZ"})
    assert result["errors"] == {"network_id": "invalid_network_id"}
    assert radio.calls == []


@pytest.mark.parametrize(
    ("raised", "error"),
    [
        (config_flow.RadioSetupError("no_traffic"), "no_traffic"),
        (config_flow.RadioSetupError("no_heaters"), "no_heaters"),
        (RadioLinkError("lost"), "cannot_connect_radio"),
        (RuntimeError("boom"), "unknown"),
    ],
)
async def test_discovery_errors_return_to_the_form(
    hass: HomeAssistant, radio: FakeRadio, raised: Exception, error: str
) -> None:
    """A failed discovery shows the gateway form again with the reason."""
    radio.discover = raised
    result = await _choose(hass, await _gateway_menu(hass, FORM), "radio_discover")
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "radio"
    assert result["errors"] == {"base": error}


async def test_discovery_shows_progress_until_the_task_finishes(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A slow discovery shows progress, then the entry is created."""
    release = asyncio.Event()

    async def slow(*_args: Any, **_kwargs: Any) -> Any:
        await release.wait()
        return NetworkSighting(DIALECT_B, NET), {6: None}

    menu = await _gateway_menu(hass, FORM)
    with patch.object(config_flow, "discover_radio", slow):
        result = await hass.config_entries.flow.async_configure(
            menu["flow_id"], {"next_step_id": "radio_discover"}
        )
        assert result["type"] is FlowResultType.SHOW_PROGRESS
        assert result["progress_action"] == "radio_discover"
        again = await hass.config_entries.flow.async_configure(result["flow_id"])
        assert again["type"] is FlowResultType.SHOW_PROGRESS
        release.set()
        result = await _finish_progress(hass, again)
    assert result["type"] is FlowResultType.CREATE_ENTRY


async def test_unknown_dialect_saves_a_report_and_links_it(
    hass: HomeAssistant, radio: FakeRadio, tmp_path: Path
) -> None:
    """An unknown dialect saves a redacted survey report and names the file."""
    hass.config.config_dir = str(tmp_path)
    radio.discover = config_flow.UnknownDialectError(_report("undecodable"))

    result = await _choose(hass, await _gateway_menu(hass, FORM), "radio_discover")

    assert result["errors"] == {"base": "unknown_dialect"}
    placeholders = result["description_placeholders"]
    assert placeholders["issue_url"] == radio_survey.ISSUE_URL
    files = list(tmp_path.glob("termoweb_radio_survey_setup_*.json"))
    assert [str(f) for f in files] == [placeholders["report"]]
    saved = json.loads(files[0].read_text())
    assert saved["integration_version"] == VERSION
    assert saved["radio_type"] == "esp32" and saved["configured_dialect"] == "auto"
    assert saved["report"]["verdict"] == "undecodable"
    assert saved["report"]["redacted"] is True
    analyse_survey = radio.kwargs[0]["analyse_survey"]
    assert (await analyse_survey([])).verdict == "silent"  # runs in the executor


async def test_unknown_dialect_report_that_cannot_be_saved(
    hass: HomeAssistant, radio: FakeRadio, tmp_path: Path
) -> None:
    """When the report cannot be written the error text says so."""
    hass.config.config_dir = str(tmp_path / "missing")
    radio.discover = config_flow.UnknownDialectError(_report("undecodable"))
    result = await _choose(hass, await _gateway_menu(hass, FORM), "radio_discover")
    assert "could not be saved" in result["description_placeholders"]["report"]


# --- pairing new heaters from the config flow -------------------------------------


async def test_pairing_creates_an_entry_on_the_site_network(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """Without a typed network id, heaters are paired into the site network."""
    site_net = await radio_pairing.async_site_network_id(hass, GATEWAY_ID)

    result = await _pair(hass, await _gateway_menu(hass, FORM))

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"]["dialect"] == "B"
    assert result["data"]["network_id"] == site_net.hex().upper()
    assert result["data"]["nodes"] == [
        {"type": "htr", "addr": "2", "name": "Heater 2"},
        {"type": "htr", "addr": "3", "name": "Heater 3"},
    ]
    assert radio.calls[-1] == ("pair", "10.0.0.5", 2323, site_net)
    assert radio.kwargs[-1]["dialects"] == PAIRING_DIALECTS


async def test_site_network_id_hashes_instance_and_gateway(
    hass: HomeAssistant,
) -> None:
    """The site network id is stable per installation and gateway."""
    seed = f"{await instance_id.async_get(hass)}:{GATEWAY_ID}".encode()
    assert (
        await radio_pairing.async_site_network_id(hass, GATEWAY_ID)
        == (hashlib.sha256(seed).digest()[:2])
    )


@pytest.mark.parametrize(
    ("dialect", "dialects"), [("auto", PAIRING_DIALECTS), ("B", (DIALECT_B,))]
)
async def test_pairing_uses_a_network_id_the_user_entered(
    hass: HomeAssistant, radio: FakeRadio, dialect: str, dialects: tuple
) -> None:
    """A typed network id and dialect are used for pairing."""
    radio.paired = (DIALECT_A, [_paired(2)])
    menu = await _gateway_menu(hass, {**FORM, "dialect": dialect, "network_id": "1234"})
    result = await _pair(hass, menu)
    assert result["data"]["network_id"] == "1234"
    assert result["data"]["dialect"] == "A"
    assert radio.kwargs[-1]["dialects"] == dialects


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        ((None, []), "no_heaters_paired"),
        (RadioLinkError("closed"), "cannot_connect_radio"),
        (RuntimeError("boom"), "unknown"),
    ],
)
async def test_pairing_errors_return_to_the_pairing_form(
    hass: HomeAssistant, radio: FakeRadio, error: Any, reason: str
) -> None:
    """A failed pairing run shows the pairing form again with the reason."""
    radio.paired = error
    result = await _pair(hass, await _gateway_menu(hass, FORM))
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "radio_pair"
    assert result["errors"] == {"base": reason}


async def test_nanocul_pairing_listens_for_dialect_a_only(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """Stock nanoCUL firmware pairs dialect-A heaters only."""
    radio.ports = [PORT]
    radio.stick = ("nanocul-a1b2c3", False)
    radio.paired = (DIALECT_A, [_paired(2)])

    result = await _pair(hass, await _stick_menu(hass, STICK_FORM))

    assert result["data"]["radio_type"] == "nanocul"
    assert radio.calls[-1][2] == 0  # serial: no port number
    assert radio.kwargs[-1]["dialects"] == (DIALECT_A,)


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
    """A detected port is probed with its USB serial, then discovered."""
    result = await _choose(hass, await _stick_menu(hass, STICK_FORM), "radio_discover")

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["result"].unique_id == "radio:nanocul-a1b2c3"
    assert result["title"] == f"nanoCUL ({PORT.device})"
    assert result["data"]["radio_type"] == "nanocul"
    assert result["data"]["device"] == PORT.device
    assert result["data"]["radio_device_id"] == "nanocul-a1b2c3"
    assert result["data"]["dialect"] == "A" and "host" not in result["data"]
    assert stick.calls == [("probe", PORT.device), ("discover", PORT.device, 0)]
    assert stick.serials == ["A1B2C3"]
    assert stick.kwargs[0]["dialect_capable"] is False


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
            {**STICK_FORM, "dialect": "B"},
            ("id", False),
            {"base": "dialect_unsupported_firmware"},
        ),
        (
            {**STICK_FORM, "network_id": "XYZ"},
            ("id", False),
            {"network_id": "invalid_network_id"},
        ),
    ],
)
async def test_stick_errors_stay_on_the_form(
    hass: HomeAssistant, stick: FakeRadio, form: dict, probe: Any, error: dict
) -> None:
    """A busy stick, a dialect its firmware lacks or a bad id keep the form open."""
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
    stick.discover = (NetworkSighting(DIALECT_B, NET), {6: None})
    menu = await _stick_menu(hass, {**STICK_FORM, "dialect": "B"})
    result = await _choose(hass, menu, "radio_discover")
    assert result["data"]["dialect"] == "B" and result["data"]["network_id"] == "1234"
    assert stick.kwargs[0]["dialect_capable"] is True


async def test_stick_discovery_failure_returns_to_the_manual_form(
    hass: HomeAssistant, stick: FakeRadio
) -> None:
    """A stick lost during discovery shows the manual form with the error."""
    stick.discover = RadioLinkError("unplugged")
    result = await _choose(hass, await _stick_menu(hass, STICK_FORM), "radio_discover")
    assert result["step_id"] == "nanocul_manual"
    assert result["errors"] == {"base": "cannot_connect_nanocul"}


# --- options flow ------------------------------------------------------------------


def _options_entry(
    hass: HomeAssistant, nodes: list[dict[str, Any]] | None = None, **options: Any
) -> MockConfigEntry:
    """Add a radio entry with ``options``."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_BRAND: BRAND_RADIO,
            "network_id": "1234",
            "nodes": NODES if nodes is None else nodes,
        },
        options=options,
    )
    entry.add_to_hass(hass)
    return entry


async def _options_step(hass: HomeAssistant, entry: MockConfigEntry, step: str) -> Any:
    """Open the options menu and pick ``step``."""
    menu = await hass.config_entries.options.async_init(entry.entry_id)
    assert menu["menu_options"] == ["settings", "pair_heaters", "rehome"]
    return await hass.config_entries.options.async_configure(
        menu["flow_id"], {"next_step_id": step}
    )


async def test_options_store_heater_rated_power(hass: HomeAssistant) -> None:
    """Rated power is stored per heater; other options are kept."""
    entry = _options_entry(
        hass,
        radio_power={"power_limit": 2000, "rated_power": {"6": 1200}},
        energy_history_progress={"htr:6": 1_700_000_000},
        energy_history_imported=True,
    )
    form = await _options_step(hass, entry, "settings")
    assert form["step_id"] == "settings"
    assert form["data_schema"]({}) == {"rated_power_6": 1200}
    assert "rated_power_6 = heater 6" in form["description_placeholders"]["heaters"]

    result = await hass.config_entries.options.async_configure(
        form["flow_id"], {"rated_power_6": 1500}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert dict(entry.options) == {
        "radio_power": {"power_limit": 2000, "rated_power": {"6": 1500}},
        "energy_history_progress": {"htr:6": 1_700_000_000},
        "energy_history_imported": True,
    }


async def test_options_reject_out_of_range_power(hass: HomeAssistant) -> None:
    """Rated power is validated by the form schema."""
    entry = _options_entry(hass)
    form = await _options_step(hass, entry, "settings")
    with pytest.raises(InvalidData):
        await hass.config_entries.options.async_configure(
            form["flow_id"], {"rated_power_6": -1}
        )


async def test_options_without_heaters_keep_existing_options(
    hass: HomeAssistant,
) -> None:
    """An entry without heaters has nothing to set and keeps its options."""
    entry = _options_entry(hass, nodes=[], energy_history_imported=True)
    form = await _options_step(hass, entry, "settings")
    result = await hass.config_entries.options.async_configure(form["flow_id"], {})
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert dict(entry.options) == {"energy_history_imported": True}


class PairingClient:
    """The loaded entry's radio client; its pairing result is scripted."""

    def __init__(self) -> None:
        """Pair heater 2 by default."""
        self.result: Any = [_paired(2)]
        self.calls: list[Any] = []

    async def async_pair(self, window_s: float, **kwargs: Any) -> Any:
        """Return (or raise) the scripted pairing result."""
        self.calls.append((window_s, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def _options_pair(hass: HomeAssistant, entry: MockConfigEntry) -> Any:
    """Run the options pairing steps and return the final result."""
    form = await _options_step(hass, entry, "pair_heaters")
    assert form["step_id"] == "pair_heaters" and form["errors"] == {}
    result = await hass.config_entries.options.async_configure(form["flow_id"], {})
    if result["type"] is not FlowResultType.SHOW_PROGRESS:
        return result
    await hass.async_block_till_done()
    return await hass.config_entries.options.async_configure(result["flow_id"])


@pytest.fixture
def pairing_entry(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> tuple[MockConfigEntry, PairingClient, list[str]]:
    """A loaded radio entry with a scripted pairing client; reloads recorded."""
    client = PairingClient()
    entry = _options_entry(hass, energy_history_imported=True)
    build_entry_runtime(
        hass=hass,
        entry_id=entry.entry_id,
        client=client,
        config_entry=entry,
        brand=BRAND_RADIO,
    )
    return entry, client, record_reloads(monkeypatch, hass)


async def test_options_pairing_adds_heaters_and_reloads(
    hass: HomeAssistant, pairing_entry: tuple
) -> None:
    """Paired heaters are added to the entry, which then reloads."""
    entry, client, reloads = pairing_entry
    result = await _options_pair(hass, entry)
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert dict(entry.options) == {"energy_history_imported": True}
    assert client.calls == [(300.0, {"idle_stop_s": 60.0})]
    assert entry.data["nodes"][-1] == {"type": "htr", "addr": "2", "name": "Heater 2"}
    assert reloads == [entry.entry_id]


async def test_options_pairing_keeps_heaters_paired_before_ids_ran_out(
    hass: HomeAssistant, pairing_entry: tuple
) -> None:
    """Heaters paired before the radio ids ran out are still added."""
    entry, client, _reloads = pairing_entry
    client.result = NoFreeAddressError("no free radio id", [_paired(0x41)])
    result = await _options_pair(hass, entry)
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.data["nodes"][-1]["addr"] == "65"


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        ([], "no_heaters_paired"),
        (NoFreeAddressError("no free radio id"), "no_free_address"),
        (RadioLinkError("closed"), "cannot_connect_radio"),
        (RuntimeError("boom"), "unknown"),
    ],
)
async def test_options_pairing_errors_show_the_form_again(
    hass: HomeAssistant, pairing_entry: tuple, error: Any, reason: str
) -> None:
    """A failed run shows the pairing form with the reason; nothing reloads."""
    entry, client, reloads = pairing_entry
    client.result = error
    result = await _options_pair(hass, entry)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pair_heaters"
    assert result["errors"] == {"base": reason}
    assert reloads == []


async def test_options_pairing_needs_a_loaded_entry(hass: HomeAssistant) -> None:
    """Without a running radio, pairing cannot start."""
    entry = _options_entry(hass)
    result = await _options_pair(hass, entry)
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "not_loaded"


# --- radio helpers used by the flow ---------------------------------------------------


def test_parse_network_id() -> None:
    """Network ids are four hex digits with optional separators."""
    assert config_flow.parse_network_id("") is None
    assert config_flow.parse_network_id(None) is None
    assert config_flow.parse_network_id(" 12:34 ") == NET
    for bad in ("123", "12345", "GGGG"):
        with pytest.raises(ValueError):
            config_flow.parse_network_id(bad)


@pytest.fixture
def probe_link(monkeypatch: pytest.MonkeyPatch) -> type[ProbeLink]:
    """Let the flow's probes open ProbeLinks instead of real radio links."""
    monkeypatch.setattr(config_flow, "RadioLink", ProbeLink)
    monkeypatch.setattr(ProbeLink, "created", [])
    return ProbeLink


async def test_probe_gateway_listens_without_acking(
    probe_link: type[ProbeLink], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gateway probe never acks and needs the gateway's MAC."""
    assert await config_flow.probe_gateway("gw", 2323) == GATEWAY_ID
    assert probe_link.created[-1]["auto_ack"] is False

    monkeypatch.setattr(
        probe_link, "info", dataclasses.replace(probe_link.info, mac=None)
    )
    with pytest.raises(config_flow.RadioSetupError, match="no_gateway_mac"):
        await config_flow.probe_gateway("gw", 2323)


async def test_probe_nanocul(
    probe_link: type[ProbeLink], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stock firmware is named by its USB serial; newer firmware by its MAC."""
    monkeypatch.setattr(
        probe_link, "info", GatewayInfo("3.5", "869.525", "2DE5", False, 1, None, "")
    )
    assert await config_flow.probe_nanocul("/dev/ttyUSB0", "A1B2C3") == (
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
    assert await config_flow.probe_nanocul("socket://gw:1") == (GATEWAY_ID, True)


def test_radio_link_factory_and_address() -> None:
    """Sticks open over serial on port 0; gateways over TCP."""
    nanocul = {"radio_type": "nanocul", "device": "/dev/ttyUSB0"}
    factory = config_flow.radio_link_factory(nanocul)
    assert factory.func is RadioLink and "open_connection" in factory.keywords
    assert config_flow.radio_address(nanocul) == ("/dev/ttyUSB0", 0)
    esp32 = {"host": "gw", "port": 2323}
    assert config_flow.radio_link_factory(esp32) is RadioLink
    assert config_flow.radio_address(esp32) == ("gw", 2323)


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
    monkeypatch.setattr(config_flow, "SERIAL_BY_ID_DIR", str(by_id))
    assert config_flow._by_id_path(str(tty)) == str(by_id / "usb-SHK_NANO_CUL-port0")
    assert config_flow._by_id_path("/dev/ttyS0") == "/dev/ttyS0"
    monkeypatch.setattr(config_flow, "SERIAL_BY_ID_DIR", str(tmp_path / "missing"))
    assert config_flow._by_id_path("/dev/ttyACM0") == "/dev/ttyACM0"


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
    monkeypatch.setattr(config_flow, "discover_network", fake.discover_network)
    monkeypatch.setattr(config_flow, "probe_heaters", fake.probe_heaters)
    monkeypatch.setattr(config_flow, "survey_sighting", fake.survey_sighting)
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
    await config_flow.discover_radio(
        "gw", 1, dialect, network_id, dialect_capable=capable
    )
    assert air.calls == calls


async def test_discover_radio_surveys_silent_air(air: DiscoveryRadio) -> None:
    """Nothing heard: a survey decides between no traffic and a known network."""
    air.sighting = None
    with pytest.raises(config_flow.RadioSetupError, match="no_traffic"):
        await config_flow.discover_radio("gw", 1, "auto", None)
    air.surveyed = NetworkSighting(DIALECT_B, NET)
    sighting, _ = await config_flow.discover_radio("gw", 1, "auto", None)
    assert sighting.network_id == NET
    assert air.calls[-1] == ("probe", "B", NET, False, True)


async def test_discover_radio_without_heaters(air: DiscoveryRadio) -> None:
    """A network without answering heaters is an error."""
    air.heaters = {}
    with pytest.raises(config_flow.RadioSetupError, match="no_heaters"):
        await config_flow.discover_radio("gw", 1, "auto", None)


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

    monkeypatch.setattr(config_flow, "survey_network", fake_survey)
    if expected == "raise":
        with pytest.raises(config_flow.UnknownDialectError) as err:
            await config_flow.survey_sighting("gw", 1, analyse_survey=fake_analyse)
        assert err.value.reason == "unknown_dialect" and err.value.report is report
        return
    sighting = await config_flow.survey_sighting("gw", 1, analyse_survey=fake_analyse)
    if expected is None:
        assert sighting is None
    else:
        assert (sighting.dialect, sighting.network_id) == expected
    assert seen[0] == ("gw", 1, RadioLink)
    assert seen[1:] == ([] if bursts is None else [bursts])


async def test_pairing_shows_progress_until_the_run_ends(
    hass: HomeAssistant, radio: FakeRadio
) -> None:
    """A pairing run that is still listening shows progress."""
    release = asyncio.Event()

    async def slow(*_args: Any, **_kwargs: Any) -> Any:
        await release.wait()
        return DIALECT_B, [_paired(2)]

    form = await _choose(hass, await _gateway_menu(hass, FORM), "radio_pair")
    with patch.object(config_flow, "pair_new_network", slow):
        result = await hass.config_entries.flow.async_configure(form["flow_id"], {})
        assert result["type"] is FlowResultType.SHOW_PROGRESS
        assert result["progress_action"] == "radio_pair"
        again = await hass.config_entries.flow.async_configure(result["flow_id"])
        assert again["type"] is FlowResultType.SHOW_PROGRESS
        release.set()
        result = await _finish_progress(hass, again)
    assert result["type"] is FlowResultType.CREATE_ENTRY


async def test_options_pairing_shows_progress_until_the_run_ends(
    hass: HomeAssistant, pairing_entry: tuple
) -> None:
    """Options pairing shows progress while the gateway listens."""
    entry, client, reloads = pairing_entry
    release = asyncio.Event()

    async def slow(window_s: float, **kwargs: Any) -> Any:
        await release.wait()
        return [_paired(2)]

    client.async_pair = slow
    form = await _options_step(hass, entry, "pair_heaters")
    result = await hass.config_entries.options.async_configure(form["flow_id"], {})
    assert result["type"] is FlowResultType.SHOW_PROGRESS
    assert result["progress_action"] == "pair_heaters"
    release.set()
    await hass.async_block_till_done()
    result = await hass.config_entries.options.async_configure(result["flow_id"])
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert reloads == [entry.entry_id]
