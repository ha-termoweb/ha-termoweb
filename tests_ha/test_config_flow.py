"""Config, reconfigure and options flows on real Home Assistant."""

from __future__ import annotations

from typing import Any

from aiohttp import ClientError
from homeassistant.config_entries import SOURCE_USER, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend.rest_client import (
    BackendAuthError,
    BackendRateLimitError,
)
from custom_components.termoweb.const import (
    BRAND_DUCAHEAT,
    BRAND_TERMOWEB,
    BRAND_TEVOLVE,
    CONF_BRAND,
    DEFAULT_BRAND,
    DOMAIN,
)
from custom_components.termoweb.energy import (
    OPTION_ENERGY_HISTORY_IMPORTED,
    OPTION_ENERGY_HISTORY_PROGRESS,
)

from .conftest import PASSWORD, USERNAME, VERSION, FakeCloud


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
        (BRAND_TEVOLVE, f"Tevolve ({USERNAME})", f"tevolve:{USERNAME}"),
    ],
)
async def test_cloud_step_creates_entry(
    hass: HomeAssistant, cloud: FakeCloud, brand: str, title: str, unique_id: str
) -> None:
    """Valid credentials create an entry keyed by brand and username."""
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
        pytest.param(
            TimeoutError(),
            "cannot_connect",
            marks=pytest.mark.xfail(
                strict=True,
                reason="A login timeout is reported as 'unknown' because the flow "
                "only maps ClientError to cannot_connect (new finding, PLAN Phase 3)",
            ),
        ),
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


@pytest.mark.xfail(
    strict=True,
    reason="B8: unique_id is not case-folded, so one account can be added twice "
    "(PLAN Phase 3, setup_energy.md B8)",
)
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


@pytest.mark.xfail(
    strict=True,
    reason="F2: the flow writes a fictional 'supports_diagnostics' key into "
    "entry.data (PLAN Phase 2.4/3, setup_energy.md F2)",
)
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
    """Reconfigure saves new credentials and drops the legacy poll interval."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id=USERNAME,
        data={
            "username": USERNAME,
            "password": "old",
            CONF_BRAND: BRAND_TERMOWEB,
            "poll_interval": 90,
            "other": "keep",
        },
        options={"poll_interval": 120, "extra": True},
    )
    entry.add_to_hass(hass)
    form = await entry.start_reconfigure_flow(hass)

    result = await hass.config_entries.flow.async_configure(
        form["flow_id"], _login(BRAND_DUCAHEAT, "updated@example.com")
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data["username"] == "updated@example.com"
    assert entry.data["password"] == PASSWORD
    assert entry.data[CONF_BRAND] == BRAND_DUCAHEAT
    assert entry.data["other"] == "keep"
    assert "poll_interval" not in entry.data
    assert entry.options == {"extra": True}


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (BackendAuthError("bad credentials"), "invalid_auth"),
        (BackendRateLimitError("slow down"), "rate_limited"),
        (ClientError("offline"), "cannot_connect"),
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
        form["flow_id"], _login(BRAND_DUCAHEAT, " candidate ")
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    assert result["errors"] == {"base": expected}
    assert result["description_placeholders"] == {"version": VERSION}
    defaults = _defaults(result)
    assert defaults["username"] == "candidate"
    assert defaults[CONF_BRAND] == BRAND_DUCAHEAT
    assert dict(config_entry.data) == before


@pytest.mark.xfail(
    strict=True,
    reason="B7: reconfigure updates the entry but never reloads it, so the new "
    "credentials are unused until restart (PLAN Phase 3, setup_energy.md B7)",
)
async def test_reconfigure_reloads_loaded_entry(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """Saving new credentials reloads the running entry."""
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    runtime_before = hass.data[DOMAIN][config_entry.entry_id]
    form = await config_entry.start_reconfigure_flow(hass)

    await hass.config_entries.flow.async_configure(
        form["flow_id"], _login(username=USERNAME)
    )
    await hass.async_block_till_done()

    assert config_entry.state is ConfigEntryState.LOADED
    assert hass.data[DOMAIN][config_entry.entry_id] is not runtime_before


async def test_options_flow_keeps_unknown_keys(
    hass: HomeAssistant, config_entry: MockConfigEntry
) -> None:
    """Saving the options form keeps option keys the form does not own."""
    progress = {"htr:1": 1_700_000_000}
    config_entry.add_to_hass(hass)
    hass.config_entries.async_update_entry(
        config_entry,
        options={
            "debug": False,
            OPTION_ENERGY_HISTORY_PROGRESS: progress,
            OPTION_ENERGY_HISTORY_IMPORTED: True,
            "future_option": "keep-me",
        },
    )

    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    assert _defaults(result) == {"debug": False}
    assert result["description_placeholders"] == {"version": VERSION, "heaters": ""}

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"debug": True}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert config_entry.options == {
        "debug": True,
        OPTION_ENERGY_HISTORY_PROGRESS: progress,
        OPTION_ENERGY_HISTORY_IMPORTED: True,
        "future_option": "keep-me",
    }
