# ruff: noqa: D100,D101,D102,D103,D105,D107,INP001,E402
from __future__ import annotations

import asyncio
import importlib
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any

import pytest

# --- Minimal third-party stubs -------------------------------------------------

aiohttp_stub = sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))


class ClientSession:  # pragma: no cover - placeholder
    pass


class ClientTimeout:  # pragma: no cover - placeholder
    def __init__(self, total: int | None = None) -> None:
        self.total = total


class ClientResponseError(Exception):  # pragma: no cover - placeholder
    pass


class ClientError(Exception):  # pragma: no cover - placeholder
    pass


aiohttp_stub.ClientSession = getattr(aiohttp_stub, "ClientSession", ClientSession)
aiohttp_stub.ClientTimeout = getattr(aiohttp_stub, "ClientTimeout", ClientTimeout)
aiohttp_stub.ClientResponseError = getattr(
    aiohttp_stub, "ClientResponseError", ClientResponseError
)
aiohttp_stub.ClientError = ClientError


vol = sys.modules.setdefault("voluptuous", types.ModuleType("voluptuous"))


class Required:  # pragma: no cover - minimal stub
    def __init__(self, schema: Any, *, default: Any | None = None) -> None:
        self.schema = schema
        self.default = default

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Required({self.schema!r})"


class All:  # pragma: no cover - minimal stub
    def __init__(self, *validators: Any) -> None:
        self.validators = validators


class Range:  # pragma: no cover - minimal stub
    def __init__(self, *, min: int | None = None, max: int | None = None) -> None:
        self.min = min
        self.max = max


class In:  # pragma: no cover - minimal stub
    def __init__(self, container: Any) -> None:
        self.container = container


class Schema:  # pragma: no cover - minimal stub
    def __init__(self, schema: dict[Any, Any]) -> None:
        self.schema = schema

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, validator in self.schema.items():
            if isinstance(key, Required):
                name = key.schema
                if name in data:
                    value = data[name]
                elif key.default is not None:
                    value = key.default
                else:
                    raise KeyError(name)
            else:
                name = key
                value = data[name]
            result[name] = _apply_validator(value, validator)
        return result


def _apply_validator(value: Any, validator: Any) -> Any:
    if isinstance(validator, All):
        for item in validator.validators:
            value = _apply_validator(value, item)
        return value
    if isinstance(validator, In):
        container = validator.container
        if isinstance(container, dict):
            if value not in container:
                raise ValueError(f"{value!r} not in {list(container)!r}")
            return value
        if value not in container:
            raise ValueError(f"{value!r} not in {container!r}")
        return value
    if isinstance(validator, Range):
        if validator.min is not None and value < validator.min:
            raise ValueError(f"{value} < {validator.min}")
        if validator.max is not None and value > validator.max:
            raise ValueError(f"{value} > {validator.max}")
        return value
    if validator is int:
        return int(value)
    if validator is str:
        return str(value)
    if callable(validator):
        return validator(value)
    return value


vol.Required = Required
vol.All = All
vol.Range = Range
vol.In = In
vol.Schema = Schema

# --- Minimal Home Assistant stubs ----------------------------------------------

homeassistant_pkg = sys.modules.setdefault(
    "homeassistant", types.ModuleType("homeassistant")
)
config_entries_mod = sys.modules.setdefault(
    "homeassistant.config_entries", types.ModuleType("homeassistant.config_entries")
)
core_mod = sys.modules.setdefault(
    "homeassistant.core", types.ModuleType("homeassistant.core")
)
helpers_mod = sys.modules.setdefault(
    "homeassistant.helpers", types.ModuleType("homeassistant.helpers")
)
aiohttp_client_mod = sys.modules.setdefault(
    "homeassistant.helpers.aiohttp_client",
    types.ModuleType("homeassistant.helpers.aiohttp_client"),
)
loader_mod = sys.modules.setdefault(
    "homeassistant.loader", types.ModuleType("homeassistant.loader")
)
data_entry_flow_mod = sys.modules.setdefault(
    "homeassistant.data_entry_flow", types.ModuleType("homeassistant.data_entry_flow")
)

homeassistant_pkg.config_entries = config_entries_mod
homeassistant_pkg.core = core_mod
homeassistant_pkg.helpers = helpers_mod
homeassistant_pkg.loader = loader_mod
homeassistant_pkg.data_entry_flow = data_entry_flow_mod
helpers_mod.aiohttp_client = aiohttp_client_mod


class FlowResult(dict):  # pragma: no cover - placeholder type alias
    pass


data_entry_flow_mod.FlowResult = FlowResult


class ConfigEntry:  # pragma: no cover - simplified stand-in
    def __init__(
        self,
        entry_id: str,
        data: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.entry_id = entry_id
        self.data = dict(data or {})
        self.options = dict(options or {})
        self.unique_id: str | None = None
        self.title = ""


class ConfigEntriesManager:  # pragma: no cover - minimal registry
    def __init__(self) -> None:
        self._entries: dict[str, ConfigEntry] = {}
        self.updated_entries: list[tuple[ConfigEntry, dict[str, Any] | None, dict[str, Any] | None]] = []

    def add_entry(self, entry: ConfigEntry) -> None:
        self._entries[entry.entry_id] = entry

    def async_get_entry(self, entry_id: str) -> ConfigEntry | None:
        return self._entries.get(entry_id)

    def async_update_entry(
        self,
        entry: ConfigEntry,
        *,
        data: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        if data is not None:
            entry.data = data
        if options is not None:
            entry.options = options
        self.updated_entries.append((entry, data, options))


class HomeAssistant:  # pragma: no cover - minimal hass
    def __init__(self) -> None:
        self.config_entries = ConfigEntriesManager()


class ConfigFlow:  # pragma: no cover - simplified ConfigFlow base
    def __init_subclass__(cls, *, domain: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.DOMAIN = domain

    def __init__(self) -> None:
        self.hass: HomeAssistant | None = None
        self.context: dict[str, Any] = {}
        self._unique_id: str | None = None

    async def async_set_unique_id(self, unique_id: str) -> None:
        self._unique_id = unique_id

    def _abort_if_unique_id_configured(self) -> None:
        return None

    def async_show_form(
        self,
        *,
        step_id: str,
        data_schema: Any,
        errors: dict[str, str] | None = None,
        description_placeholders: dict[str, Any] | None = None,
    ) -> FlowResult:
        return FlowResult(
            {
                "type": "form",
                "step_id": step_id,
                "data_schema": data_schema,
                "errors": errors or {},
                "description_placeholders": description_placeholders or {},
            }
        )

    def async_create_entry(self, *, title: str, data: dict[str, Any]) -> FlowResult:
        return FlowResult({"type": "create_entry", "title": title, "data": data})

    def async_abort(self, *, reason: str) -> FlowResult:
        return FlowResult({"type": "abort", "reason": reason})


class OptionsFlow:  # pragma: no cover - simplified OptionsFlow base
    def async_show_form(
        self,
        *,
        step_id: str,
        data_schema: Any,
        errors: dict[str, str] | None = None,
        description_placeholders: dict[str, Any] | None = None,
    ) -> FlowResult:
        return FlowResult(
            {
                "type": "form",
                "step_id": step_id,
                "data_schema": data_schema,
                "errors": errors or {},
                "description_placeholders": description_placeholders or {},
            }
        )

    def async_create_entry(self, *, title: str, data: dict[str, Any]) -> FlowResult:
        return FlowResult({"type": "create_entry", "title": title, "data": data})


config_entries_mod.ConfigEntry = ConfigEntry
config_entries_mod.ConfigEntriesManager = ConfigEntriesManager
config_entries_mod.ConfigFlow = ConfigFlow
config_entries_mod.OptionsFlow = OptionsFlow
core_mod.HomeAssistant = HomeAssistant


def async_get_clientsession(_hass: HomeAssistant) -> object:  # pragma: no cover - stub
    return object()


aiohttp_client_mod.async_get_clientsession = async_get_clientsession


class _Integration:  # pragma: no cover - minimal integration info
    def __init__(self, version: str) -> None:
        self.version = version


async def async_get_integration(_hass: HomeAssistant, _domain: str) -> _Integration:
    return _Integration("0.0-test")


loader_mod.async_get_integration = async_get_integration

# --- Import module under test ---------------------------------------------------

custom_components_pkg = sys.modules.setdefault(
    "custom_components", types.ModuleType("custom_components")
)
custom_components_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "custom_components")]

CONFIG_FLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "termoweb"
    / "config_flow.py"
)
package_name = "custom_components.termoweb"
module_name = f"{package_name}.config_flow"

termoweb_pkg = types.ModuleType(package_name)
termoweb_pkg.__path__ = [str(CONFIG_FLOW_PATH.parent)]
sys.modules[package_name] = termoweb_pkg
setattr(custom_components_pkg, "termoweb", termoweb_pkg)

spec = importlib.util.spec_from_file_location(module_name, CONFIG_FLOW_PATH)
config_flow = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[module_name] = config_flow
spec.loader.exec_module(config_flow)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant


def _schema_default(schema: Schema, field: str) -> Any:
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
    ("raised", "expected"),
    [
        (config_flow.TermoWebAuthError(), "invalid_auth"),
        (config_flow.TermoWebRateLimitError(), "rate_limited"),
        (aiohttp_stub.ClientError(), "cannot_connect"),
        (RuntimeError("boom"), "unknown"),
    ],
)
def test_async_step_user_errors(
    monkeypatch: pytest.MonkeyPatch, raised: Exception, expected: str
) -> None:
    hass = HomeAssistant()
    flow = _create_flow(hass)

    async def fake_version(_hass: HomeAssistant) -> str:
        return "2.0.0"

    async def fake_validate(
        _hass: HomeAssistant, username: str, password: str, brand: str
    ) -> None:
        raise raised

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
    ("raised", "expected"),
    [
        (config_flow.TermoWebAuthError(), "invalid_auth"),
        (config_flow.TermoWebRateLimitError(), "rate_limited"),
        (aiohttp_stub.ClientError(), "cannot_connect"),
        (RuntimeError("fail"), "unknown"),
    ],
)
def test_async_step_reconfigure_errors(
    monkeypatch: pytest.MonkeyPatch, raised: Exception, expected: str
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
        raise raised

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
