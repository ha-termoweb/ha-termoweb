# ruff: noqa: D100,D101,D102,D103,D104,D105,D106,D107,INP001,E402
from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any, Callable


ConfigEntryAuthFailedStub = type(
    "ConfigEntryAuthFailed",
    (Exception,),
    {"__module__": "homeassistant.exceptions"},
)
ConfigEntryNotReadyStub = type(
    "ConfigEntryNotReady",
    (Exception,),
    {"__module__": "homeassistant.exceptions"},
)


def _install_stubs() -> None:
    # --- aiohttp ---------------------------------------------------------------
    aiohttp_stub = types.ModuleType("aiohttp")

    class ClientSession:  # pragma: no cover - placeholder
        pass

    class ClientTimeout:  # pragma: no cover - placeholder
        def __init__(self, total: int | None = None) -> None:
            self.total = total

    class ClientResponseError(Exception):  # pragma: no cover - placeholder
        def __init__(
            self,
            request_info: Any,
            history: Any,
            *,
            status: int | None = None,
            message: str | None = None,
            headers: Any | None = None,
        ) -> None:
            super().__init__(message)
            self.request_info = request_info
            self.history = history
            self.status = status
            self.headers = headers

    class ClientError(Exception):  # pragma: no cover - placeholder
        pass

    aiohttp_stub.ClientSession = ClientSession
    aiohttp_stub.ClientTimeout = ClientTimeout
    aiohttp_stub.ClientResponseError = ClientResponseError
    aiohttp_stub.ClientError = ClientError
    sys.modules["aiohttp"] = aiohttp_stub

    # --- voluptuous ------------------------------------------------------------
    vol = types.ModuleType("voluptuous")

    class Required:  # pragma: no cover - minimal stub
        def __init__(self, schema: Any, *, default: Any | None = None) -> None:
            self.schema = schema
            self.default = default

    class All:  # pragma: no cover - minimal stub
        def __init__(self, *validators: Any) -> None:
            self.validators = validators

    class Range:  # pragma: no cover - minimal stub
        def __init__(self, *, min: int | None = None, max: int | None = None) -> None:
            self.min = min
            self.max = max

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
    vol.Schema = Schema
    sys.modules["voluptuous"] = vol

    # --- custom_components -----------------------------------------------------
    custom_components_pkg = types.ModuleType("custom_components")
    custom_components_pkg.__path__ = [
        str(Path(__file__).resolve().parents[1] / "custom_components")
    ]
    sys.modules["custom_components"] = custom_components_pkg

    # --- homeassistant ---------------------------------------------------------
    homeassistant_pkg = types.ModuleType("homeassistant")
    config_entries_mod = types.ModuleType("homeassistant.config_entries")
    const_mod = types.ModuleType("homeassistant.const")
    core_mod = types.ModuleType("homeassistant.core")
    exceptions_mod = types.ModuleType("homeassistant.exceptions")
    helpers_mod = types.ModuleType("homeassistant.helpers")
    aiohttp_client_mod = types.ModuleType("homeassistant.helpers.aiohttp_client")
    entity_registry_mod = types.ModuleType("homeassistant.helpers.entity_registry")
    dispatcher_mod = types.ModuleType("homeassistant.helpers.dispatcher")
    loader_mod = types.ModuleType("homeassistant.loader")
    update_coordinator_mod = types.ModuleType(
        "homeassistant.helpers.update_coordinator"
    )

    sys.modules["homeassistant"] = homeassistant_pkg
    sys.modules["homeassistant.config_entries"] = config_entries_mod
    sys.modules["homeassistant.const"] = const_mod
    sys.modules["homeassistant.core"] = core_mod
    sys.modules["homeassistant.exceptions"] = exceptions_mod
    sys.modules["homeassistant.helpers"] = helpers_mod
    sys.modules["homeassistant.helpers.aiohttp_client"] = aiohttp_client_mod
    sys.modules["homeassistant.helpers.entity_registry"] = entity_registry_mod
    sys.modules["homeassistant.helpers.dispatcher"] = dispatcher_mod
    sys.modules["homeassistant.helpers.update_coordinator"] = update_coordinator_mod
    sys.modules["homeassistant.loader"] = loader_mod

    homeassistant_pkg.config_entries = config_entries_mod
    homeassistant_pkg.const = const_mod
    homeassistant_pkg.core = core_mod
    homeassistant_pkg.exceptions = exceptions_mod
    homeassistant_pkg.helpers = helpers_mod
    homeassistant_pkg.loader = loader_mod
    def async_get_clientsession(hass: Any) -> ClientSession:
        if hasattr(hass, "client_session_calls"):
            hass.client_session_calls += 1
        return ClientSession()

    aiohttp_client_mod.async_get_clientsession = async_get_clientsession

    helpers_mod.aiohttp_client = aiohttp_client_mod
    helpers_mod.entity_registry = entity_registry_mod
    helpers_mod.dispatcher = dispatcher_mod
    helpers_mod.update_coordinator = update_coordinator_mod

    const_mod.EVENT_HOMEASSISTANT_STARTED = "homeassistant_started"

    class ConfigEntry:
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
            self.title: str = ""

    class _SimpleConfigEntries:
        def __init__(self) -> None:
            self._entries: dict[str, ConfigEntry] = {}
            self.updated_entries: list[
                tuple[ConfigEntry, dict[str, Any] | None, dict[str, Any] | None]
            ] = []

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

    class HomeAssistant:
        def __init__(self) -> None:
            self.config_entries = _SimpleConfigEntries()
            self.dispatcher_connections: list[
                tuple[str, Callable[[Any], None]]
            ] = []

    config_entries_mod.ConfigEntry = ConfigEntry
    core_mod.HomeAssistant = HomeAssistant
    exceptions_mod.ConfigEntryAuthFailed = ConfigEntryAuthFailedStub
    exceptions_mod.ConfigEntryNotReady = ConfigEntryNotReadyStub

    class DataUpdateCoordinator:
        def __class_getitem__(cls, _item: Any) -> type:
            return cls

        def __init__(
            self,
            hass: Any,
            *,
            logger: Any | None = None,
            name: str | None = None,
            update_interval: Any | None = None,
        ) -> None:
            self.hass = hass
            self.logger = logger
            self.name = name
            self.update_interval = update_interval
            self.data: Any = None

        async def async_config_entry_first_refresh(self) -> None:
            return None

        def async_add_listener(self, _listener: Any) -> None:
            return None

    class UpdateFailed(Exception):
        pass

    update_coordinator_mod.DataUpdateCoordinator = DataUpdateCoordinator
    update_coordinator_mod.UpdateFailed = UpdateFailed

    def async_dispatcher_connect(
        hass: Any, signal: str, callback: Callable[[Any], None]
    ) -> Callable[[], None]:
        if hasattr(hass, "dispatcher_connections"):
            hass.dispatcher_connections.append((signal, callback))
        return lambda: None

    def async_dispatcher_send(hass: Any, signal: str, *args: Any) -> None:
        if hasattr(hass, "dispatcher_connections"):
            for sig, callback in hass.dispatcher_connections:
                if sig == signal:
                    callback(*args if args else {})

    dispatcher_mod.async_dispatcher_connect = async_dispatcher_connect
    dispatcher_mod.async_dispatcher_send = async_dispatcher_send

    class _StubIntegration:
        def __init__(self, domain: str) -> None:
            self.domain = domain
            self.version = "test-version"

    async def async_get_integration(hass: Any, domain: str) -> _StubIntegration:
        if hasattr(hass, "integration_requests"):
            hass.integration_requests.append(domain)
        return _StubIntegration(domain)

    loader_mod.async_get_integration = async_get_integration


_install_stubs()


def pytest_runtest_setup(item: Any) -> None:  # pragma: no cover - ensure isolation
    _install_stubs()
