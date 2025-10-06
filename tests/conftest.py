# ruff: noqa: D100,D101,D102,D103,D104,D105,D106,D107,INP001,E402
from __future__ import annotations

import asyncio
import inspect
import datetime as dt
import enum
import time
from pathlib import Path
import sys
import types
from typing import Any, Callable, Iterable
from unittest.mock import AsyncMock

import pytest


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


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers used across the suite."""

    if not config.pluginmanager.hasplugin("pytest_asyncio"):
        config.addinivalue_line(
            "markers", "asyncio: mark test as requiring asyncio event loop support."
        )


def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Execute async tests when pytest-asyncio is unavailable."""

    if pyfuncitem.config.pluginmanager.hasplugin("pytest_asyncio"):
        return None

    testfunction = pyfuncitem.obj
    if not inspect.iscoroutinefunction(testfunction):
        return None

    marker = pyfuncitem.get_closest_marker("asyncio")
    if marker is None:
        return None

    with asyncio.Runner(debug=False) as runner:
        runner.run(testfunction(**pyfuncitem.funcargs))
    return True


def _install_stubs() -> None:
    # --- aiohttp ---------------------------------------------------------------
    aiohttp_stub = sys.modules.get("aiohttp") or types.ModuleType("aiohttp")

    if not hasattr(aiohttp_stub, "ClientSession"):

        class ClientSession:  # pragma: no cover - placeholder
            pass

        aiohttp_stub.ClientSession = ClientSession

    if not hasattr(aiohttp_stub, "ClientTimeout"):

        class ClientTimeout:  # pragma: no cover - placeholder
            def __init__(self, total: int | None = None) -> None:
                self.total = total

        aiohttp_stub.ClientTimeout = ClientTimeout

    if not hasattr(aiohttp_stub, "ClientResponseError"):

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

        aiohttp_stub.ClientResponseError = ClientResponseError

    if not hasattr(aiohttp_stub, "ClientError"):

        class ClientError(Exception):  # pragma: no cover - placeholder
            pass

        aiohttp_stub.ClientError = ClientError

    if not hasattr(aiohttp_stub, "WSMsgType"):

        class WSMsgType(enum.IntEnum):
            TEXT = 1
            BINARY = 2
            CLOSE = 3
            CLOSED = 4
            ERROR = 5

        aiohttp_stub.WSMsgType = WSMsgType
    else:
        WSMsgType = aiohttp_stub.WSMsgType

    if not hasattr(aiohttp_stub, "WSCloseCode"):

        class WSCloseCode(types.SimpleNamespace):
            GOING_AWAY = 1001

        aiohttp_stub.WSCloseCode = WSCloseCode
    else:
        WSCloseCode = aiohttp_stub.WSCloseCode

    testing_ns = getattr(aiohttp_stub, "testing", None)
    defaults = getattr(testing_ns, "_defaults", None) if testing_ns else None
    if defaults is None:
        defaults = types.SimpleNamespace(get_responses=[], ws_connect_results=[])

    class FakeHTTPResponse:
        def __init__(
            self,
            status: int,
            body: Any,
            *,
            headers: dict[str, Any] | None = None,
        ) -> None:
            self.status = status
            self._body = body
            self.headers = headers or {}
            self.request_info = None
            self.history: tuple[Any, ...] = ()

        async def text(self) -> str:
            body = self._body
            if asyncio.iscoroutine(body):
                body = await body
            if callable(body):
                body = body()
            if isinstance(body, bytes):
                return body.decode("utf-8", "ignore")
            return str(body or "")

    class FakeGetContext:
        def __init__(self, response: FakeHTTPResponse) -> None:
            self._response = response

        async def __aenter__(self) -> FakeHTTPResponse:
            return self._response

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeWebSocket:
        def __init__(self, messages: Iterable[Any] | None = None) -> None:
            self._messages = list(messages or [])
            self.sent: list[str] = []
            self.close_code: int | None = None
            self._exception: BaseException | None = None

        def queue_message(self, message: Any) -> None:
            self._messages.append(message)

        async def receive(self) -> Any:
            if not self._messages:
                return types.SimpleNamespace(
                    type=WSMsgType.CLOSED, data=None, extra=None
                )
            msg = self._messages.pop(0)
            if callable(msg):
                msg = msg()
            if asyncio.iscoroutine(msg):
                msg = await msg
            if isinstance(msg, dict):
                return types.SimpleNamespace(
                    type=msg.get("type", WSMsgType.TEXT),
                    data=msg.get("data"),
                    extra=msg.get("extra"),
                )
            return msg

        async def send_str(self, data: str) -> None:
            self.sent.append(data)

        async def close(
            self, code: int | None = None, message: bytes | None = None
        ) -> None:
            self.close_code = code

        def exception(self) -> BaseException | None:
            return self._exception

        def set_exception(self, exc: BaseException | None) -> None:
            self._exception = exc

    class FakeClientSession:
        def __init__(
            self,
            *,
            get_responses: Iterable[Any] | None = None,
            ws_connect_results: Iterable[Any] | None = None,
        ) -> None:
            self._get_script = list(
                defaults.get_responses if get_responses is None else get_responses
            )
            self._ws_script = list(
                defaults.ws_connect_results
                if ws_connect_results is None
                else ws_connect_results
            )
            self.get_calls: list[dict[str, Any]] = []
            self.ws_connect_calls: list[dict[str, Any]] = []

        def queue_get(self, response: Any) -> None:
            self._get_script.append(response)

        def queue_ws(self, result: Any) -> None:
            self._ws_script.append(result)

        def get(self, url: str, *, timeout: Any | None = None) -> FakeGetContext:
            if not self._get_script:
                raise AssertionError("No scripted GET response available")
            entry = self._get_script.pop(0)
            if callable(entry):
                entry = entry(url=url, timeout=timeout)
            response = _coerce_aiohttp_response(entry)
            self.get_calls.append({"url": url, "timeout": timeout})
            return FakeGetContext(response)

        async def ws_connect(self, url: str, **kwargs: Any) -> Any:
            if not self._ws_script:
                raise AssertionError("No scripted ws_connect result available")
            entry = self._ws_script.pop(0)
            if callable(entry):
                entry = entry(url=url, **kwargs)
            if asyncio.iscoroutine(entry):
                entry = await entry
            if isinstance(entry, dict) and "messages" in entry:
                entry = FakeWebSocket(entry["messages"])
            self.ws_connect_calls.append({"url": url, "kwargs": kwargs})
            return entry

    def _coerce_aiohttp_response(entry: Any) -> FakeHTTPResponse:
        if isinstance(entry, FakeHTTPResponse):
            return entry
        if isinstance(entry, tuple):
            status = int(entry[0])
            body = entry[1] if len(entry) > 1 else ""
            headers = entry[2] if len(entry) > 2 else None
            return FakeHTTPResponse(status, body, headers=headers)
        if isinstance(entry, dict):
            return FakeHTTPResponse(
                int(entry.get("status", 200)),
                entry.get("body", ""),
                headers=entry.get("headers"),
            )
        return FakeHTTPResponse(200, entry)

    aiohttp_stub.ClientSession = FakeClientSession
    aiohttp_stub.testing = types.SimpleNamespace(
        FakeClientSession=FakeClientSession,
        FakeWebSocket=FakeWebSocket,
        FakeHTTPResponse=FakeHTTPResponse,
        _defaults=defaults,
    )

    if not hasattr(aiohttp_stub, "ClientWebSocketResponse"):
        aiohttp_stub.ClientWebSocketResponse = FakeWebSocket

    ClientSession = aiohttp_stub.ClientSession
    ClientTimeout = aiohttp_stub.ClientTimeout
    ClientResponseError = aiohttp_stub.ClientResponseError
    ClientError = aiohttp_stub.ClientError
    sys.modules["aiohttp"] = aiohttp_stub

    # --- voluptuous ------------------------------------------------------------
    vol = sys.modules.get("voluptuous") or types.ModuleType("voluptuous")

    class Required:  # pragma: no cover - minimal stub
        def __init__(self, schema: Any, *, default: Any | None = None) -> None:
            self.schema = schema
            self.default = default

    class Optional:  # pragma: no cover - minimal stub
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

    class In:  # pragma: no cover - minimal stub
        def __init__(self, container: list[Any]) -> None:
            self.container = container

    class Coerce:  # pragma: no cover - minimal stub
        def __init__(self, func: Callable[[Any], Any]) -> None:
            self.func = func

    class Length:  # pragma: no cover - minimal stub
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
                elif isinstance(key, Optional):
                    name = key.schema
                    if name in data:
                        value = data[name]
                    elif key.default is not None:
                        value = key.default
                    else:
                        continue
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
        if isinstance(validator, Coerce):
            return validator.func(value)
        if isinstance(validator, In):
            if value not in validator.container:
                raise ValueError(f"{value} not in {validator.container}")
            return value
        if isinstance(validator, Length):
            length = len(value)
            if validator.min is not None and length < validator.min:
                raise ValueError(f"len {length} < {validator.min}")
            if validator.max is not None and length > validator.max:
                raise ValueError(f"len {length} > {validator.max}")
            return value
        if validator is int:
            return int(value)
        if validator is str:
            return str(value)
        if callable(validator):
            return validator(value)
        return value

    vol.Required = Required
    vol.Optional = Optional
    vol.All = All
    vol.Range = Range
    vol.In = In
    vol.Coerce = Coerce
    vol.Length = Length
    vol.Schema = Schema
    sys.modules["voluptuous"] = vol

    # --- custom_components -----------------------------------------------------
    custom_components_pkg = sys.modules.get("custom_components") or types.ModuleType(
        "custom_components"
    )
    custom_components_pkg.__path__ = [
        str(Path(__file__).resolve().parents[1] / "custom_components")
    ]
    sys.modules["custom_components"] = custom_components_pkg

    # --- homeassistant ---------------------------------------------------------
    homeassistant_pkg = sys.modules.get("homeassistant") or types.ModuleType(
        "homeassistant"
    )
    config_entries_mod = sys.modules.get(
        "homeassistant.config_entries"
    ) or types.ModuleType("homeassistant.config_entries")
    const_mod = sys.modules.get("homeassistant.const") or types.ModuleType(
        "homeassistant.const"
    )
    core_mod = sys.modules.get("homeassistant.core") or types.ModuleType(
        "homeassistant.core"
    )
    exceptions_mod = sys.modules.get("homeassistant.exceptions") or types.ModuleType(
        "homeassistant.exceptions"
    )
    helpers_mod = sys.modules.get("homeassistant.helpers") or types.ModuleType(
        "homeassistant.helpers"
    )
    aiohttp_client_mod = sys.modules.get(
        "homeassistant.helpers.aiohttp_client"
    ) or types.ModuleType("homeassistant.helpers.aiohttp_client")
    entity_registry_mod = sys.modules.get(
        "homeassistant.helpers.entity_registry"
    ) or types.ModuleType("homeassistant.helpers.entity_registry")
    dispatcher_mod = sys.modules.get(
        "homeassistant.helpers.dispatcher"
    ) or types.ModuleType("homeassistant.helpers.dispatcher")
    entity_mod = sys.modules.get("homeassistant.helpers.entity") or types.ModuleType(
        "homeassistant.helpers.entity"
    )
    loader_mod = sys.modules.get("homeassistant.loader") or types.ModuleType(
        "homeassistant.loader"
    )
    data_entry_flow_mod = sys.modules.get(
        "homeassistant.data_entry_flow"
    ) or types.ModuleType("homeassistant.data_entry_flow")
    update_coordinator_mod = sys.modules.get(
        "homeassistant.helpers.update_coordinator"
    ) or types.ModuleType("homeassistant.helpers.update_coordinator")
    components_mod = sys.modules.get("homeassistant.components") or types.ModuleType(
        "homeassistant.components"
    )
    binary_sensor_mod = sys.modules.get(
        "homeassistant.components.binary_sensor"
    ) or types.ModuleType("homeassistant.components.binary_sensor")
    button_mod = sys.modules.get("homeassistant.components.button") or types.ModuleType(
        "homeassistant.components.button"
    )
    sensor_mod = sys.modules.get("homeassistant.components.sensor") or types.ModuleType(
        "homeassistant.components.sensor"
    )
    climate_mod = sys.modules.get(
        "homeassistant.components.climate"
    ) or types.ModuleType("homeassistant.components.climate")
    entity_platform_mod = sys.modules.get(
        "homeassistant.helpers.entity_platform"
    ) or types.ModuleType("homeassistant.helpers.entity_platform")

    sys.modules["homeassistant"] = homeassistant_pkg
    sys.modules["homeassistant.config_entries"] = config_entries_mod
    sys.modules["homeassistant.const"] = const_mod
    sys.modules["homeassistant.core"] = core_mod
    sys.modules["homeassistant.exceptions"] = exceptions_mod
    sys.modules["homeassistant.helpers"] = helpers_mod
    sys.modules["homeassistant.helpers.aiohttp_client"] = aiohttp_client_mod
    sys.modules["homeassistant.data_entry_flow"] = data_entry_flow_mod
    sys.modules["homeassistant.helpers.entity"] = entity_mod
    sys.modules["homeassistant.helpers.entity_registry"] = entity_registry_mod
    sys.modules["homeassistant.helpers.dispatcher"] = dispatcher_mod
    sys.modules["homeassistant.helpers.update_coordinator"] = update_coordinator_mod
    sys.modules["homeassistant.helpers.entity_platform"] = entity_platform_mod
    sys.modules["homeassistant.loader"] = loader_mod
    sys.modules["homeassistant.components"] = components_mod
    sys.modules["homeassistant.components.binary_sensor"] = binary_sensor_mod
    sys.modules["homeassistant.components.button"] = button_mod
    sys.modules["homeassistant.components.sensor"] = sensor_mod
    sys.modules["homeassistant.components.climate"] = climate_mod

    homeassistant_pkg.config_entries = config_entries_mod
    homeassistant_pkg.const = const_mod
    homeassistant_pkg.core = core_mod
    homeassistant_pkg.exceptions = exceptions_mod
    homeassistant_pkg.helpers = helpers_mod
    homeassistant_pkg.data_entry_flow = data_entry_flow_mod
    homeassistant_pkg.components = components_mod
    homeassistant_pkg.loader = loader_mod

    def async_get_clientsession(hass: Any) -> ClientSession:
        if hasattr(hass, "client_session_calls"):
            hass.client_session_calls += 1
        return ClientSession()

    aiohttp_client_mod.async_get_clientsession = async_get_clientsession

    helpers_mod.aiohttp_client = aiohttp_client_mod
    helpers_mod.entity = entity_mod
    helpers_mod.entity_registry = entity_registry_mod
    helpers_mod.dispatcher = dispatcher_mod
    helpers_mod.update_coordinator = update_coordinator_mod
    components_mod.binary_sensor = binary_sensor_mod
    components_mod.button = button_mod
    components_mod.sensor = sensor_mod

    const_mod.EVENT_HOMEASSISTANT_STARTED = "homeassistant_started"

    class FlowResult(dict):
        pass

    data_entry_flow_mod.FlowResult = FlowResult

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
            self._on_unload: list[Callable[[], None]] = []

        def async_on_unload(self, func: Callable[[], None]) -> Callable[[], None]:
            self._on_unload.append(func)
            return func

        def _call_unload_callbacks(self) -> None:
            while self._on_unload:
                callback = self._on_unload.pop()
                try:
                    callback()
                except Exception:  # noqa: BLE001 - defensive logging
                    pass

    class _SimpleConfigEntries:
        def __init__(self) -> None:
            self._entries: dict[str, ConfigEntry] = {}
            self.updated_entries: list[
                tuple[ConfigEntry, dict[str, Any] | None, dict[str, Any] | None]
            ] = []
            self.forwarded: list[tuple[ConfigEntry, tuple[str, ...]]] = []
            self.unloaded: list[tuple[ConfigEntry, tuple[str, ...]]] = []

        def add_entry(self, entry: ConfigEntry) -> None:
            self._entries[entry.entry_id] = entry

        def add(self, entry: ConfigEntry) -> None:
            self.add_entry(entry)

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

        async def async_forward_entry_setups(
            self, entry: ConfigEntry, platforms: list[str] | tuple[str, ...]
        ) -> None:
            self.forwarded.append((entry, tuple(platforms)))

        async def async_unload_platforms(
            self, entry: ConfigEntry, platforms: list[str] | tuple[str, ...]
        ) -> bool:
            self.unloaded.append((entry, tuple(platforms)))
            entry._call_unload_callbacks()
            return True

    class _ServiceRegistry:
        def __init__(self) -> None:
            self._services: dict[tuple[str, str], Callable[..., Any]] = {}

        def has_service(self, domain: str, service: str) -> bool:
            return (domain, service) in self._services

        def async_register(
            self, domain: str, service: str, handler: Callable[..., Any]
        ) -> None:
            self._services[(domain, service)] = handler

        def async_remove(self, domain: str, service: str) -> None:
            self._services.pop((domain, service), None)

        def get(self, domain: str, service: str) -> Callable[..., Any] | None:
            return self._services.get((domain, service))

    class _EventBus:
        def __init__(self) -> None:
            self.listeners: list[tuple[str, Callable[[Any], Any]]] = []

        def async_listen_once(
            self, event: str, callback: Callable[[Any], Any]
        ) -> Callable[[], None]:
            def _remove() -> None:
                try:
                    self.listeners.remove((event, wrapped))
                except ValueError:
                    pass

            async def wrapped(payload: Any) -> Any:
                _remove()
                result = callback(payload)
                if inspect.isawaitable(result):
                    return await result
                return result

            self.listeners.append((event, wrapped))
            return _remove

    class HomeAssistant:
        def __init__(self) -> None:
            self.config_entries = _SimpleConfigEntries()
            self.dispatcher_connections: list[tuple[str, Callable[[Any], None]]] = []
            self.data: dict[str, Any] = {}
            self.integration_requests: list[str] = []
            self.is_running = True
            self.is_stopping = False
            self.client_session_calls = 0
            self.services = _ServiceRegistry()
            self.bus = _EventBus()
            self.tasks: list[asyncio.Task[Any]] = []

        def async_create_task(self, coro: Any) -> asyncio.Task[Any]:
            task = asyncio.create_task(coro)
            self.tasks.append(task)
            return task

    class ConfigFlow:
        def __init_subclass__(cls, *, domain: str | None = None, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)
            if domain is not None:
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

    class OptionsFlow:
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
    config_entries_mod.ConfigFlow = ConfigFlow
    config_entries_mod.OptionsFlow = OptionsFlow
    core_mod.HomeAssistant = HomeAssistant
    core_mod.callback = lambda func: func

    class ServiceCall:
        def __init__(self, data: dict[str, Any] | None = None) -> None:
            self.data = data or {}

    core_mod.ServiceCall = ServiceCall
    exceptions_mod.ConfigEntryAuthFailed = ConfigEntryAuthFailedStub
    exceptions_mod.ConfigEntryNotReady = ConfigEntryNotReadyStub

    if not hasattr(update_coordinator_mod, "DataUpdateCoordinator"):

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
                self._listeners: list[Callable[[], None]] = []

            async def _async_update_data(self) -> Any:
                raise NotImplementedError

            async def async_refresh(self) -> None:
                self.data = await self._async_update_data()
                for listener in list(self._listeners):
                    listener()

            async def async_config_entry_first_refresh(self) -> None:
                await self.async_refresh()

            async def async_request_refresh(self) -> None:
                await self.async_refresh()

            def async_set_updated_data(self, data: Any) -> None:
                self.data = data
                for listener in list(self._listeners):
                    listener()

            def async_add_listener(
                self, listener: Callable[[], None]
            ) -> Callable[[], None]:
                if callable(listener):
                    self._listeners.append(listener)

                def _unsub() -> None:
                    if listener in self._listeners:
                        self._listeners.remove(listener)

                return _unsub

        update_coordinator_mod.DataUpdateCoordinator = DataUpdateCoordinator
    else:
        DataUpdateCoordinator = update_coordinator_mod.DataUpdateCoordinator

    if not hasattr(update_coordinator_mod, "CoordinatorEntity"):

        class CoordinatorEntity:
            def __init__(self, coordinator: Any) -> None:
                self.coordinator = coordinator
                self.hass = getattr(coordinator, "hass", None)
                self._remove_callbacks: list[Callable[[], None]] = []

            async def async_added_to_hass(self) -> None:
                return None

            def async_on_remove(self, func: Callable[[], None]) -> None:
                self._remove_callbacks.append(func)

            def schedule_update_ha_state(self) -> None:
                return None

            @classmethod
            def __class_getitem__(cls, _item: Any) -> type:
                return cls

            async def async_will_remove_from_hass(self) -> None:
                for callback in list(self._remove_callbacks):
                    callback()

        update_coordinator_mod.CoordinatorEntity = CoordinatorEntity
    else:
        CoordinatorEntity = update_coordinator_mod.CoordinatorEntity

    if not hasattr(update_coordinator_mod, "UpdateFailed"):

        class UpdateFailed(Exception):
            pass

        update_coordinator_mod.UpdateFailed = UpdateFailed
    else:
        UpdateFailed = update_coordinator_mod.UpdateFailed

    dispatch_map: dict[str, list[Callable[[Any], None]]] = getattr(
        dispatcher_mod, "_dispatch_map", {}
    )
    dispatcher_mod._dispatch_map = dispatch_map

    def async_dispatcher_connect(
        hass: Any, signal: str, callback: Callable[[Any], None]
    ) -> Callable[[], None]:
        dispatch_map.setdefault(signal, []).append(callback)
        if hasattr(hass, "dispatcher_connections"):
            hass.dispatcher_connections.append((signal, callback))

        def _unsubscribe() -> None:
            callbacks = dispatch_map.get(signal, [])
            if callback in callbacks:
                callbacks.remove(callback)

        return _unsubscribe

    def _dispatch(signal: str, payload_args: tuple[Any, ...]) -> None:
        callbacks = list(dispatch_map.get(signal, []))
        for callback in callbacks:
            callback(*payload_args if payload_args else ({},))

    def async_dispatcher_send(
        first: Any, second: Any | None = None, *args: Any
    ) -> None:
        if isinstance(first, str) and (second is None or not isinstance(second, str)):
            actual_signal = first
            payload_args = (() if second is None else (second,)) + args
        else:
            actual_signal = str(second if second is not None else first)
            payload_args = args
        _dispatch(actual_signal, payload_args)

    dispatcher_mod.async_dispatcher_connect = async_dispatcher_connect
    dispatcher_mod.async_dispatcher_send = async_dispatcher_send
    dispatcher_mod.dispatcher_send = async_dispatcher_send

    def make_ws_payload(
        dev_id: str,
        addr: Any | None = None,
        *,
        node_type: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Return a websocket payload for tests with optional node type."""

        payload: dict[str, Any] = {"dev_id": dev_id}
        if addr is not None:
            payload["addr"] = addr
        if node_type is not None:
            payload["node_type"] = node_type
        payload.update(extra)
        return payload

    dispatcher_mod.make_ws_payload = make_ws_payload
    globals()["make_ws_payload"] = make_ws_payload

    class DeviceInfo(dict):
        pass

    entity_mod.DeviceInfo = DeviceInfo

    class EntityPlatform:
        def __init__(self) -> None:
            self.registered: list[tuple[str, Any, Callable[..., Any]]] = []

        def async_register_entity_service(
            self, name: str, schema: Any, func: Callable[..., Any]
        ) -> None:
            self.registered.append((name, schema, func))

    def async_get_current_platform() -> EntityPlatform:
        platform = getattr(entity_platform_mod, "_current_platform", None)
        if platform is None:
            platform = EntityPlatform()
            entity_platform_mod._current_platform = platform
        return platform

    def _set_current_platform(platform: EntityPlatform) -> None:
        entity_platform_mod._current_platform = platform

    entity_platform_mod.EntityPlatform = EntityPlatform
    entity_platform_mod.async_get_current_platform = async_get_current_platform
    entity_platform_mod._set_current_platform = _set_current_platform

    class ClimateEntity:
        def __init__(self) -> None:
            self.hass: Any | None = None
            self._on_remove: Callable[[], None] | None = None
            self._attr_preset_modes: list[str] | None = None

        async def async_added_to_hass(self) -> None:
            return None

        async def async_will_remove_from_hass(self) -> None:
            if self._on_remove is not None:
                self._on_remove()

        def async_on_remove(self, func: Callable[[], None]) -> None:
            self._on_remove = func

        def schedule_update_ha_state(self) -> None:
            return None

        def async_write_ha_state(self) -> None:
            return None

        @property
        def preset_modes(self) -> list[str] | None:
            return self._attr_preset_modes

    class ClimateEntityFeature(enum.IntFlag):
        TARGET_TEMPERATURE = 1
        PRESET_MODE = 2

    class HVACMode(str, enum.Enum):
        OFF = "off"
        HEAT = "heat"
        AUTO = "auto"

    class HVACAction(str, enum.Enum):
        OFF = "off"
        IDLE = "idle"
        HEATING = "heating"

    climate_mod.ClimateEntity = ClimateEntity
    climate_mod.ClimateEntityFeature = ClimateEntityFeature
    climate_mod.HVACMode = HVACMode
    climate_mod.HVACAction = HVACAction
    components_mod.climate = climate_mod

    class UnitOfTemperature:
        CELSIUS = "C"

    const_mod.UnitOfTemperature = UnitOfTemperature
    const_mod.ATTR_TEMPERATURE = "temperature"

    class SensorEntity:
        def __init__(self) -> None:
            self.hass: Any | None = None
            self._remove_callbacks: list[Callable[[], None]] = []

        async def async_added_to_hass(self) -> None:
            return None

        async def async_will_remove_from_hass(self) -> None:
            for callback in list(self._remove_callbacks):
                callback()

        def async_on_remove(self, func: Callable[[], None]) -> None:
            self._remove_callbacks.append(func)

        def schedule_update_ha_state(self) -> None:
            return None

        @property
        def device_class(self) -> str | None:
            return getattr(self, "_attr_device_class", None)

        @property
        def state_class(self) -> str | None:
            return getattr(self, "_attr_state_class", None)

        @property
        def native_unit_of_measurement(self) -> str | None:
            return getattr(self, "_attr_native_unit_of_measurement", None)

    class SensorDeviceClass:
        ENERGY = "energy"
        POWER = "power"
        TEMPERATURE = "temperature"

    class SensorStateClass:
        MEASUREMENT = "measurement"
        TOTAL_INCREASING = "total_increasing"

    sensor_mod.SensorEntity = SensorEntity
    sensor_mod.SensorDeviceClass = SensorDeviceClass
    sensor_mod.SensorStateClass = SensorStateClass

    class BinarySensorEntity:
        def __init__(self) -> None:
            self.hass: Any | None = None

        async def async_added_to_hass(self) -> None:
            return None

        def async_on_remove(self, func: Callable[[], Any]) -> None:
            self._on_remove = func

        def schedule_update_ha_state(self) -> None:
            return None

    class BinarySensorDeviceClass:
        CONNECTIVITY = "connectivity"

    binary_sensor_mod.BinarySensorEntity = BinarySensorEntity
    binary_sensor_mod.BinarySensorDeviceClass = BinarySensorDeviceClass

    class ButtonEntity:
        def __init__(self) -> None:
            self.hass: Any | None = None

        async def async_added_to_hass(self) -> None:
            return None

    button_mod.ButtonEntity = ButtonEntity

    class _StubIntegration:
        def __init__(self, domain: str) -> None:
            self.domain = domain
            self.version = "test-version"

    async def async_get_integration(hass: Any, domain: str) -> _StubIntegration:
        if hasattr(hass, "integration_requests"):
            hass.integration_requests.append(domain)
        return _StubIntegration(domain)

    loader_mod.async_get_integration = async_get_integration

    util_mod = sys.modules.get("homeassistant.util") or types.ModuleType(
        "homeassistant.util"
    )
    dt_mod = sys.modules.get("homeassistant.util.dt") or types.ModuleType(
        "homeassistant.util.dt"
    )
    dt_mod.NOW = getattr(
        dt_mod,
        "NOW",
        dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc),
    )

    def now() -> dt.datetime:
        return dt_mod.NOW

    dt_mod.now = now
    util_mod.dt = dt_mod
    sys.modules["homeassistant.util"] = util_mod
    sys.modules["homeassistant.util.dt"] = dt_mod


_install_stubs()


class FakeCoordinator:
    """Reusable coordinator stub shared across tests."""

    instances: list["FakeCoordinator"] = []

    def __init__(
        self,
        hass: Any,
        client: Any | None = None,
        base_interval: int = 0,
        dev_id: str = "dev",
        dev: dict[str, Any] | None = None,
        nodes: dict[str, Any] | None = None,
        node_inventory: Iterable[Any] | None = None,
        *,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.hass = hass
        self.client = client
        self.base_interval = base_interval
        self.dev_id = dev_id
        self.dev = dev or {}
        self.nodes = nodes or {}
        self.node_inventory = list(node_inventory or [])
        self.update_interval = dt.timedelta(seconds=base_interval or 0)
        if data is not None:
            self.data = data
        elif dev_id:
            self.data = {dev_id: self.dev}
        else:
            self.data = {}
        self.listeners: list[Callable[[], None]] = []
        self.refresh_calls = 0
        self.async_request_refresh = AsyncMock()
        self.async_refresh_heater = AsyncMock()
        self.pending_settings: dict[tuple[str, str], dict[str, Any]] = {}
        type(self).instances.append(self)

    async def async_config_entry_first_refresh(self) -> None:
        self.refresh_calls += 1

    def async_add_listener(self, listener: Callable[[], None]) -> None:
        if callable(listener):
            self.listeners.append(listener)

    def update_nodes(
        self,
        nodes: dict[str, Any],
        node_inventory: Iterable[Any] | None = None,
    ) -> None:
        self.nodes = nodes
        if node_inventory is not None:
            self.node_inventory = list(node_inventory)

    def register_pending_setting(
        self,
        node_type: str,
        addr: str,
        *,
        mode: str | None,
        stemp: float | None,
        ttl: float = 0.0,
    ) -> None:
        """Record pending settings for verification in tests."""

        key = (str(node_type), str(addr))
        self.pending_settings[key] = {
            "mode": mode,
            "stemp": stemp,
            "expires_at": time.time() + max(ttl, 0.0),
        }


def pytest_runtest_setup(item: Any) -> None:  # pragma: no cover - ensure isolation
    _install_stubs()
