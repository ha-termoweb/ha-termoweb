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

    async def fake_get_integration(_hass: HomeAssistant, _domain: str) -> DummyIntegration:
        return DummyIntegration("")

    monkeypatch.setattr(
        config_flow, "async_get_integration", fake_get_integration
    )

    result = asyncio.run(config_flow._get_version(hass))

    assert result == "unknown"


def test_validate_login_uses_brand_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = HomeAssistant()
    calls: list[tuple[Any, str, str, str, str]] = []

    class DummyClient:
        def __init__(
            self,
            session: object,
            username: str,
            password: str,
            *,
            api_base: str,
            basic_auth_b64: str,
        ) -> None:
            calls.append((session, username, password, api_base, basic_auth_b64))

        async def list_devices(self) -> None:
            calls.append(("listed",))

    monkeypatch.setattr(config_flow, "TermoWebClient", DummyClient)

    asyncio.run(
        config_flow._validate_login(
            hass, "user@example.com", "pw", config_flow.BRAND_DUCAHEAT
        )
    )

    assert calls
    _session, username, password, api_base, basic_auth = calls[0]
    assert username == "user@example.com"
    assert password == "pw"
    assert api_base == config_flow.get_brand_api_base(config_flow.BRAND_DUCAHEAT)
    assert basic_auth == config_flow.get_brand_basic_auth(config_flow.BRAND_DUCAHEAT)
    assert calls[-1] == ("listed",)


def test_async_step_reconfigure_missing_entry_aborts() -> None:
    hass = HomeAssistant()
    flow = _create_flow(hass)
    flow.context["entry_id"] = "missing"

    result = asyncio.run(flow.async_step_reconfigure())

    assert result == {"type": "abort", "reason": "no_config_entry"}


def test_async_step_user_initial_form(monkeypatch: pytest.MonkeyPatch) -> None:
    hass = HomeAssistant()
    flow = _create_flow(hass)

    async def fake_version(_hass: HomeAssistant) -> str:
        return "1.2.3"

    monkeypatch.setattr(config_flow, "_get_version", fake_version)

    result = asyncio.run(flow.async_step_user())

    assert result["type"] == "form"
    assert result["step_id"] == "user"
    assert result["errors"] == {}
    assert result["description_placeholders"] == {"version": "1.2.3"}

    schema = result["data_schema"]
    assert _schema_default(schema, "username") == ""
    assert _schema_default(schema, "poll_interval") == config_flow.DEFAULT_POLL_INTERVAL
    assert _schema_default(schema, "brand") == config_flow.DEFAULT_BRAND


def test_async_step_user_success(monkeypatch: pytest.MonkeyPatch) -> None:
    hass = HomeAssistant()
    flow = _create_flow(hass)

    async def fake_version(_hass: HomeAssistant) -> str:
        return "9.9.9"

    calls: list[tuple[str, str, str]] = []

    async def fake_validate(
        _hass: HomeAssistant, username: str, password: str, brand: str
    ) -> None:
        calls.append((username, password, brand))

    monkeypatch.setattr(config_flow, "_get_version", fake_version)
    monkeypatch.setattr(config_flow, "_validate_login", fake_validate)

    result = asyncio.run(
        flow.async_step_user(
            {
                "brand": config_flow.BRAND_DUCAHEAT,
                "username": "  new_user  ",
                "password": "pw",
                "poll_interval": 200,
            }
        )
    )

    assert calls == [("new_user", "pw", config_flow.BRAND_DUCAHEAT)]
    assert result["type"] == "create_entry"
    assert result["title"] == "Ducaheat (new_user)"
    assert result["data"] == {
        "username": "new_user",
        "password": "pw",
        "poll_interval": 200,
        config_flow.CONF_BRAND: config_flow.BRAND_DUCAHEAT,
    }


@pytest.mark.parametrize(
    ("raised_factory", "expected"),
    [
        (lambda: config_flow.TermoWebAuthError(), "invalid_auth"),
        (lambda: config_flow.TermoWebRateLimitError(), "rate_limited"),
        (lambda: config_flow.ClientError(), "cannot_connect"),
        (lambda: RuntimeError("boom"), "unknown"),
    ],
)
def test_async_step_user_errors(
    monkeypatch: pytest.MonkeyPatch, raised_factory: Any, expected: str
) -> None:
    hass = HomeAssistant()
    flow = _create_flow(hass)

    async def fake_version(_hass: HomeAssistant) -> str:
        return "2.0.0"

    async def fake_validate(
        _hass: HomeAssistant, username: str, password: str, brand: str
    ) -> None:
        raise raised_factory()

    monkeypatch.setattr(config_flow, "_get_version", fake_version)
    monkeypatch.setattr(config_flow, "_validate_login", fake_validate)

    user_input = {
        "brand": config_flow.BRAND_DUCAHEAT,
        "username": "  trouble  ",
        "password": "pw",
        "poll_interval": 321,
    }
    result = asyncio.run(flow.async_step_user(user_input))

    assert result["type"] == "form"
    assert result["errors"] == {"base": expected}
    assert result["description_placeholders"] == {"version": "2.0.0"}

    schema = result["data_schema"]
    assert _schema_default(schema, "username") == "trouble"
    assert _schema_default(schema, "poll_interval") == 321
    assert _schema_default(schema, "brand") == config_flow.BRAND_DUCAHEAT


def test_async_step_reconfigure_initial_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = HomeAssistant()
    entry = ConfigEntry(
        "entry-id",
        data={
            "username": "existing",
            "poll_interval": 150,
            config_flow.CONF_BRAND: config_flow.BRAND_DUCAHEAT,
        },
        options={"poll_interval": 180},
    )
    hass.config_entries.add_entry(entry)

    flow = _create_flow(hass)
    flow.context = {"entry_id": entry.entry_id}

    async def fake_version(_hass: HomeAssistant) -> str:
        return "3.3.3"

    monkeypatch.setattr(config_flow, "_get_version", fake_version)

    result = asyncio.run(flow.async_step_reconfigure())

    assert result["type"] == "form"
    assert result["step_id"] == "reconfigure"
    assert result["errors"] == {}
    assert result["description_placeholders"] == {"version": "3.3.3"}

    schema = result["data_schema"]
    assert _schema_default(schema, "username") == "existing"
    assert _schema_default(schema, "poll_interval") == 180
    assert _schema_default(schema, "brand") == config_flow.BRAND_DUCAHEAT


def test_async_step_reconfigure_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = HomeAssistant()
    entry = ConfigEntry(
        "entry-id",
        data={
            "username": "old",
            "password": "old-pass",
            "poll_interval": 90,
            "other": "keep",
            config_flow.CONF_BRAND: config_flow.BRAND_TERMOWEB,
        },
        options={"poll_interval": 120, "extra": True},
    )
    hass.config_entries.add_entry(entry)

    flow = _create_flow(hass)
    flow.context = {"entry_id": entry.entry_id}

    async def fake_version(_hass: HomeAssistant) -> str:
        return "4.4.4"

    async def fake_validate(
        _hass: HomeAssistant, username: str, password: str, brand: str
    ) -> None:
        assert username == "updated"
        assert password == "new-pass"
        assert brand == config_flow.BRAND_DUCAHEAT

    monkeypatch.setattr(config_flow, "_get_version", fake_version)
    monkeypatch.setattr(config_flow, "_validate_login", fake_validate)

    result = asyncio.run(
        flow.async_step_reconfigure(
            {
                "brand": config_flow.BRAND_DUCAHEAT,
                "username": "  updated  ",
                "password": "new-pass",
                "poll_interval": 300,
            }
        )
    )

    assert result["type"] == "abort"
    assert result["reason"] == "reconfigure_successful"

    expected_data = {
        "username": "updated",
        "password": "new-pass",
        "poll_interval": 300,
        "other": "keep",
        config_flow.CONF_BRAND: config_flow.BRAND_DUCAHEAT,
    }
    expected_options = {"poll_interval": 300, "extra": True}

    assert hass.config_entries.updated_entries == [
        (entry, expected_data, expected_options)
    ]
    assert entry.data == expected_data
    assert entry.options == expected_options


@pytest.mark.parametrize(
    ("raised_factory", "expected"),
    [
        (lambda: config_flow.TermoWebAuthError(), "invalid_auth"),
        (lambda: config_flow.TermoWebRateLimitError(), "rate_limited"),
        (lambda: config_flow.ClientError(), "cannot_connect"),
        (lambda: RuntimeError("fail"), "unknown"),
    ],
)
def test_async_step_reconfigure_errors(
    monkeypatch: pytest.MonkeyPatch, raised_factory: Any, expected: str
) -> None:
    hass = HomeAssistant()
    entry = ConfigEntry(
        "entry-id",
        data={
            "username": "original",
            "poll_interval": 110,
            config_flow.CONF_BRAND: config_flow.BRAND_TERMOWEB,
        },
        options={"poll_interval": 140},
    )
    hass.config_entries.add_entry(entry)

    flow = _create_flow(hass)
    flow.context = {"entry_id": entry.entry_id}

    async def fake_version(_hass: HomeAssistant) -> str:
        return "5.5.5"

    async def fake_validate(
        _hass: HomeAssistant, username: str, password: str, brand: str
    ) -> None:
        raise raised_factory()

    monkeypatch.setattr(config_flow, "_get_version", fake_version)
    monkeypatch.setattr(config_flow, "_validate_login", fake_validate)

    user_input = {
        "brand": config_flow.BRAND_DUCAHEAT,
        "username": " candidate ",
        "password": "pw",
        "poll_interval": 210,
    }
    result = asyncio.run(flow.async_step_reconfigure(user_input))

    assert result["type"] == "form"
    assert result["errors"] == {"base": expected}
    assert result["description_placeholders"] == {"version": "5.5.5"}

    schema = result["data_schema"]
    assert _schema_default(schema, "username") == "candidate"
    assert _schema_default(schema, "poll_interval") == 210
    assert _schema_default(schema, "brand") == config_flow.BRAND_DUCAHEAT
    assert hass.config_entries.updated_entries == []


def test_options_flow_init_and_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = HomeAssistant()
    entry = ConfigEntry(
        "entry-id",
        data={"poll_interval": 60},
        options={"poll_interval": 10},
    )

    options_flow = asyncio.run(config_flow.async_get_options_flow(entry))
    options_flow.hass = hass

    async def fake_version(_hass: HomeAssistant) -> str:
        return "6.6.6"

    monkeypatch.setattr(config_flow, "_get_version", fake_version)

    initial = asyncio.run(options_flow.async_step_init())
    assert initial["type"] == "form"
    assert initial["description_placeholders"] == {"version": "6.6.6"}

    schema = initial["data_schema"]
    assert _schema_default(schema, "poll_interval") == config_flow.MIN_POLL_INTERVAL

    created = asyncio.run(options_flow.async_step_init({"poll_interval": 240}))
    assert created["type"] == "create_entry"
    assert created["data"] == {"poll_interval": 240}
