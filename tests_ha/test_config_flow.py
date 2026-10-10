"""Config, reconfigure and options flows on real Home Assistant."""

from __future__ import annotations

from typing import Any

from aiohttp import ClientError
from homeassistant.config_entries import SOURCE_USER, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb import (
    async_migrate_entry,
    config_flow,  # noqa: F401 (registers handler)
)
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
    assert result["result"].minor_version == 2


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

    assert (entry.version, entry.minor_version) == (1, 2)
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
    assert [entry.minor_version for entry in entries] == [2, 2]
    assert "is the same account as entry" in caplog.text


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
    assert entry.minor_version == 2


async def test_migration_of_current_entry_is_a_no_op(hass: HomeAssistant) -> None:
    """An entry already at the current version is left alone."""
    entry = MockConfigEntry(
        domain=DOMAIN, minor_version=2, unique_id="Mixed", data={"username": "Mixed"}
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry)

    assert entry.unique_id == "Mixed"


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
