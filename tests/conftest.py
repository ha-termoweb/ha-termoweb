# ruff: noqa: D100,D101,D102,D103,D104,D105,D106,D107,INP001,E402
from __future__ import annotations

import asyncio
import datetime as dt
import enum
import inspect
import time
from pathlib import Path
import sys
import types
import threading
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping
from unittest.mock import AsyncMock

from dataclasses import dataclass
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _coerce_inventory(inventory: Any) -> tuple["Inventory" | None, list[Any]]:
    """Return an ``Inventory`` and cached node list when available."""

    try:
        from custom_components.termoweb.inventory import Inventory as InventoryType
    except ImportError:
        InventoryType = None  # type: ignore[assignment]

    if InventoryType is None:
        return None, []

    if isinstance(inventory, InventoryType):
        return inventory, list(inventory.nodes)

    return None, []


_frame_module: Any | None = None


@pytest.fixture
def inventory_builder() -> Callable[
    [str, Mapping[str, Any] | None, Iterable[Any] | None], "Inventory"
]:
    """Return helper to construct Inventory containers for tests."""

    from custom_components.termoweb.inventory import Inventory

    def _factory(
        dev_id: str,
        payload: Mapping[str, Any] | None = None,
        nodes: Iterable[Any] | None = None,
    ) -> "Inventory":
        payload_value: Any
        if payload is None:
            payload_value = {}
        elif isinstance(payload, Mapping):
            payload_value = dict(payload)
        else:
            payload_value = payload
        node_list = list(nodes or [])
        return Inventory(dev_id, payload_value, node_list)

    return _factory


@pytest.fixture
def inventory_from_map(
    inventory_builder: Callable[
        [str, Mapping[str, Any] | None, Iterable[Any] | None], "Inventory"
    ],
) -> Callable[[Mapping[str, Iterable[str]] | None, str], "Inventory"]:
    """Return helper to build Inventory objects from address maps."""

    from custom_components.termoweb.inventory import build_node_inventory

    def _factory(
        mapping: Mapping[str, Iterable[str]] | None,
        dev_id: str = "dev",
    ) -> "Inventory":
        payload_nodes: list[dict[str, Any]] = []
        if mapping:
            for raw_type, values in mapping.items():
                if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
                    continue
                for addr in values:
                    payload_nodes.append({"type": raw_type, "addr": addr})
        payload = {"nodes": payload_nodes}
        node_list = list(build_node_inventory(payload))
        return inventory_builder(dev_id, payload, node_list)

    return _factory


def build_coordinator_device_state(
    *,
    nodes: Mapping[str, Any] | None = None,
    settings: Mapping[str, Mapping[str, Any]] | None = None,
    addresses: Mapping[str, Iterable[Any]] | None = None,
    sections: Mapping[str, Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a coordinator device record with normalised heater metadata."""

    from custom_components.termoweb.inventory import (
        build_node_inventory,
        normalize_node_addr,
        normalize_node_type,
    )

    record: dict[str, Any] = {}
    if nodes is not None:
        record["inventory_payload"] = nodes
        record["node_inventory"] = list(build_node_inventory(nodes))
    if extra:
        record.update(dict(extra))

    normalised_settings: dict[str, dict[str, Any]] = {}
    if settings:
        for raw_type, raw_bucket in settings.items():
            node_type = normalize_node_type(raw_type, use_default_when_falsey=True)
            if not node_type or not isinstance(raw_bucket, Mapping):
                continue
            bucket = normalised_settings.setdefault(node_type, {})
            for addr, data in raw_bucket.items():
                normalised_addr = normalize_node_addr(
                    addr, use_default_when_falsey=True
                )
                if not normalised_addr:
                    continue
                if isinstance(data, dict):
                    bucket[normalised_addr] = data
                elif isinstance(data, Mapping):
                    bucket[normalised_addr] = dict(data)
                elif data is None:
                    bucket[normalised_addr] = {}
                else:
                    bucket[normalised_addr] = {"value": data}

    normalised_addresses: dict[str, list[str]] = {}
    if addresses:
        for raw_type, raw_addrs in addresses.items():
            node_type = normalize_node_type(raw_type, use_default_when_falsey=True)
            if not node_type:
                continue
            bucket = normalised_addresses.setdefault(node_type, [])
            if isinstance(raw_addrs, Iterable) and not isinstance(
                raw_addrs, (str, bytes)
            ):
                seen: set[str] = set(bucket)
                for candidate in raw_addrs:
                    normalised_addr = normalize_node_addr(
                        candidate, use_default_when_falsey=True
                    )
                    if not normalised_addr or normalised_addr in seen:
                        continue
                    seen.add(normalised_addr)
                    bucket.append(normalised_addr)

    normalised_sections: dict[str, dict[str, Any]] = {}
    if sections:
        for raw_type, raw_section in sections.items():
            node_type = normalize_node_type(raw_type, use_default_when_falsey=True)
            if not node_type or not isinstance(raw_section, Mapping):
                continue
            normalised_sections[node_type] = dict(raw_section)

    type_keys = (
        set(normalised_settings) | set(normalised_addresses) | set(normalised_sections)
    )
    nodes_by_type: dict[str, dict[str, Any]] = {}

    for node_type in sorted(type_keys):
        section = dict(normalised_sections.get(node_type, {}))
        settings_bucket = normalised_settings.setdefault(node_type, {})
        if not isinstance(settings_bucket, dict):
            settings_bucket = normalised_settings[node_type] = dict(settings_bucket)

        existing_settings = section.get("settings")
        if isinstance(existing_settings, Mapping):
            for addr, data in existing_settings.items():
                normalised_addr = normalize_node_addr(
                    addr, use_default_when_falsey=True
                )
                if not normalised_addr:
                    continue
                if isinstance(data, dict):
                    settings_bucket.setdefault(normalised_addr, data)
                elif isinstance(data, Mapping):
                    settings_bucket.setdefault(normalised_addr, dict(data))
                elif data is None:
                    settings_bucket.setdefault(normalised_addr, {})
        section["settings"] = settings_bucket

        addr_list = normalised_addresses.get(node_type)
        if addr_list is None:
            existing_addrs = section.get("addrs")
            addr_list = []
            seen_addrs: set[str] = set()
            if isinstance(existing_addrs, Iterable) and not isinstance(
                existing_addrs, (str, bytes)
            ):
                for candidate in existing_addrs:
                    normalised_addr = normalize_node_addr(
                        candidate, use_default_when_falsey=True
                    )
                    if not normalised_addr or normalised_addr in seen_addrs:
                        continue
                    seen_addrs.add(normalised_addr)
                    addr_list.append(normalised_addr)
        else:
            addr_list = list(addr_list)

        for addr in settings_bucket:
            if addr not in addr_list:
                addr_list.append(addr)

        normalised_addresses[node_type] = addr_list
        section["addrs"] = addr_list
        nodes_by_type[node_type] = section

    if nodes_by_type:
        record["nodes_by_type"] = nodes_by_type
        record["settings"] = normalised_settings
        if "htr" in nodes_by_type:
            record["htr"] = nodes_by_type["htr"]

    return record


def _setup_frame_for_hass(hass: Any) -> None:
    """Ensure frame helpers are initialised for a HomeAssistant instance."""

    frame_mod = _frame_module
    if frame_mod is None:
        return
    setup = getattr(frame_mod, "async_setup", None)
    if setup is None:
        return
    result = setup(hass)
    if inspect.isawaitable(result):
        loop = getattr(hass, "loop", None)
        if loop is None:
            return
        if loop.is_running():
            loop.create_task(result)
        else:
            loop.run_until_complete(result)


if TYPE_CHECKING:
    from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient
    from custom_components.termoweb.inventory import Inventory
    from homeassistant.components.climate import HVACMode


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

    base_client_response_error = getattr(aiohttp_stub, "ClientResponseError", Exception)

    class ClientResponseError(
        base_client_response_error
    ):  # pragma: no cover - placeholder
        def __init__(
            self,
            request_info: Any,
            history: Any,
            *,
            status: int | None = None,
            message: str | None = None,
            headers: Any | None = None,
        ) -> None:
            try:
                super().__init__(
                    request_info,
                    history,
                    status=status,
                    message=message,
                    headers=headers,
                )
            except Exception:  # pragma: no cover - compatibility shim
                try:
                    super().__init__(message)
                except Exception:  # pragma: no cover - fallback
                    Exception.__init__(self, message)
            self.request_info = request_info
            self.history = history
            self.status = status
            self.headers = headers

        def __str__(self) -> str:
            return f"{self.status}, message={self.args[0]!r}, url=<stubbed>"

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
    service_mod = sys.modules.get("homeassistant.helpers.service") or types.ModuleType(
        "homeassistant.helpers.service"
    )

    class _ReferencedEntities:
        def __init__(
            self,
            referenced_entity_ids: Iterable[str] | None = None,
            indirectly_referenced_entity_ids: Iterable[str] | None = None,
        ) -> None:
            self.referenced_entity_ids = list(referenced_entity_ids or [])
            self.indirectly_referenced_entity_ids = list(
                indirectly_referenced_entity_ids or []
            )

    if not hasattr(service_mod, "async_extract_referenced_entity_ids"):
        service_mod.async_extract_referenced_entity_ids = AsyncMock(
            return_value=_ReferencedEntities()
        )
    if not hasattr(service_mod, "ReferencedEntities"):
        service_mod.ReferencedEntities = _ReferencedEntities  # type: ignore[attr-defined]

    helpers_mod.service = service_mod
    try:
        from homeassistant.helpers import frame as frame_mod  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - fallback when HA not installed
        frame_mod = sys.modules.get("homeassistant.helpers.frame") or types.ModuleType(
            "homeassistant.helpers.frame"
        )

        async def async_setup(_hass: Any) -> None:
            return None

        frame_mod.async_setup = async_setup  # type: ignore[attr-defined]
    else:  # pragma: no cover - use existing module when available
        frame_mod = frame_mod
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
    translation_mod = sys.modules.get(
        "homeassistant.helpers.translation"
    ) or types.ModuleType("homeassistant.helpers.translation")
    loader_mod = sys.modules.get("homeassistant.loader") or types.ModuleType(
        "homeassistant.loader"
    )
    data_entry_flow_mod = sys.modules.get(
        "homeassistant.data_entry_flow"
    ) or types.ModuleType("homeassistant.data_entry_flow")
    update_coordinator_mod = sys.modules.get(
        "homeassistant.helpers.update_coordinator"
    ) or types.ModuleType("homeassistant.helpers.update_coordinator")
    restore_state_mod = sys.modules.get(
        "homeassistant.helpers.restore_state"
    ) or types.ModuleType("homeassistant.helpers.restore_state")
    components_mod = sys.modules.get("homeassistant.components") or types.ModuleType(
        "homeassistant.components"
    )
    http_mod = sys.modules.get("homeassistant.components.http") or types.ModuleType(
        "homeassistant.components.http"
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
    select_mod = sys.modules.get("homeassistant.components.select") or types.ModuleType(
        "homeassistant.components.select"
    )
    number_mod = sys.modules.get("homeassistant.components.number") or types.ModuleType(
        "homeassistant.components.number"
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
    sys.modules["homeassistant.helpers.frame"] = frame_mod
    sys.modules["homeassistant.helpers.aiohttp_client"] = aiohttp_client_mod
    sys.modules["homeassistant.data_entry_flow"] = data_entry_flow_mod
    sys.modules["homeassistant.helpers.entity"] = entity_mod
    sys.modules["homeassistant.helpers.entity_registry"] = entity_registry_mod
    sys.modules["homeassistant.helpers.service"] = service_mod
    sys.modules["homeassistant.helpers.dispatcher"] = dispatcher_mod
    sys.modules["homeassistant.helpers.translation"] = translation_mod
    sys.modules["homeassistant.helpers.update_coordinator"] = update_coordinator_mod
    sys.modules["homeassistant.helpers.restore_state"] = restore_state_mod
    sys.modules["homeassistant.helpers.entity_platform"] = entity_platform_mod
    sys.modules["homeassistant.loader"] = loader_mod
    sys.modules["homeassistant.components"] = components_mod
    sys.modules["homeassistant.components.http"] = http_mod
    sys.modules["homeassistant.components.binary_sensor"] = binary_sensor_mod
    sys.modules["homeassistant.components.button"] = button_mod
    sys.modules["homeassistant.components.sensor"] = sensor_mod
    sys.modules["homeassistant.components.select"] = select_mod
    sys.modules["homeassistant.components.number"] = number_mod
    sys.modules["homeassistant.components.climate"] = climate_mod

    homeassistant_pkg.config_entries = config_entries_mod
    homeassistant_pkg.const = const_mod
    homeassistant_pkg.core = core_mod
    homeassistant_pkg.exceptions = exceptions_mod
    homeassistant_pkg.helpers = helpers_mod
    helpers_mod.frame = frame_mod
    global _frame_module
    _frame_module = frame_mod
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
    helpers_mod.translation = translation_mod
    helpers_mod.update_coordinator = update_coordinator_mod
    helpers_mod.restore_state = restore_state_mod
    components_mod.http = http_mod
    components_mod.binary_sensor = binary_sensor_mod
    components_mod.button = button_mod
    components_mod.sensor = sensor_mod
    components_mod.select = select_mod
    components_mod.number = number_mod

    const_mod.EVENT_HOMEASSISTANT_STARTED = "homeassistant_started"
    const_mod.STATE_UNKNOWN = "unknown"
    state_unknown_value = const_mod.STATE_UNKNOWN

    if not hasattr(translation_mod, "async_get_exception_message"):

        async def async_get_exception_message(*_: Any, **__: Any) -> str:
            return "unknown_error"

        translation_mod.async_get_exception_message = async_get_exception_message

    if not hasattr(const_mod, "UnitOfTemperature"):

        class UnitOfTemperature(str, enum.Enum):
            CELSIUS = "°C"

        const_mod.UnitOfTemperature = UnitOfTemperature

    else:
        UnitOfTemperature = const_mod.UnitOfTemperature

    if not hasattr(http_mod, "HomeAssistantApplication"):

        class HomeAssistantApplication(dict):
            """Lightweight aiohttp-style application used in tests."""

            def __init__(
                self,
                *args: Any,
                middlewares: Iterable[Any] | None = None,
                **kwargs: Any,
            ) -> None:
                super().__init__()
                self.on_cleanup: list[Callable[..., Any]] = []
                self.on_shutdown: list[Callable[..., Any]] = []
                self.cleanup_ctx: list[Callable[..., Any]] = []
                self.middlewares = list(middlewares or [])

            def freeze(self) -> None:
                return None

        http_mod.HomeAssistantApplication = HomeAssistantApplication

    else:
        HomeAssistantApplication = http_mod.HomeAssistantApplication

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
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            self.loop = loop
            self.loop_thread_id = threading.get_ident()
            self._data: dict[str, Any] = {
                "integrations": {},
                "components": {},
                "missing_platforms": {},
                "preload_platforms": set(),
            }
            self.integration_requests: list[str] = []
            self.is_running = True
            self.is_stopping = False
            self.client_session_calls = 0
            self.services = _ServiceRegistry()
            self.bus = _EventBus()
            self.tasks: list[asyncio.Task[Any]] = []
            self.config = types.SimpleNamespace(recovery_mode=False, safe_mode=False)
            _setup_frame_for_hass(self)

        def async_create_task(self, coro: Any) -> asyncio.Task[Any]:
            task = asyncio.create_task(coro)
            self.tasks.append(task)
            return task

        async def async_add_executor_job(
            self, func: Callable[..., Any], *args: Any
        ) -> Any:
            """Run ``func`` in a background thread for the test harness."""

            return await asyncio.to_thread(func, *args)

        def async_run_hass_job(
            self, job: Any, *args: Any, background: bool = False
        ) -> asyncio.Future[Any] | None:
            """Execute ``job`` immediately, mimicking Home Assistant's scheduler."""

            target = getattr(job, "target", job)
            result = target(*args)
            if inspect.isawaitable(result):
                task = asyncio.tasks.Task(
                    result,
                    loop=self.loop,
                    eager_start=True,  # type: ignore[arg-type]
                )
                self.tasks.append(task)
                return task
            return None

        def verify_event_loop_thread(self, what: str) -> None:
            return None

        @property
        def data(self) -> dict[str, Any]:
            """Return the Home Assistant data registry for tests."""

            return self._data

        @data.setter
        def data(self, value: Any) -> None:
            """Store core data while preserving required integration caches."""

            base = {
                "integrations": {},
                "components": {},
                "missing_platforms": {},
                "preload_platforms": set(),
            }
            if isinstance(value, dict):
                merged = dict(base)
                merged.update(value)
                self._data = merged
            else:
                self._data = value

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
                self.last_update_success = True
                self.last_exception: Exception | None = None

            async def _async_update_data(self) -> Any:
                raise NotImplementedError

            async def async_refresh(self) -> None:
                try:
                    result = await self._async_update_data()
                except Exception as exc:
                    self.last_update_success = False
                    self.last_exception = exc
                else:
                    self.data = result
                    self.last_update_success = True
                    self.last_exception = None
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
                self._attr_unique_id: str | None = None

            async def async_added_to_hass(self) -> None:
                return None

            def async_on_remove(self, func: Callable[[], None]) -> None:
                self._remove_callbacks.append(func)

            def schedule_update_ha_state(self) -> None:
                return None

            @property
            def unique_id(self) -> str | None:
                return getattr(self, "_attr_unique_id", None)

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
        settings: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Return a websocket payload for tests with optional node type."""

        payload: dict[str, Any] = {"dev_id": dev_id}
        if addr is not None:
            payload["addr"] = addr
        if node_type is not None:
            payload["node_type"] = node_type

        provided_settings: Mapping[str, Any] | None
        if settings is not None:
            provided_settings = settings
        else:
            raw_settings = extra.pop("settings", None)
            provided_settings = (
                raw_settings if isinstance(raw_settings, Mapping) else None
            )

        if provided_settings is None and addr is not None:
            key = str(addr)
            provided_settings = {key: {}}

        if provided_settings is not None:
            payload["settings"] = dict(provided_settings)

        payload.update(extra)
        return payload

    dispatcher_mod.make_ws_payload = make_ws_payload
    globals()["make_ws_payload"] = make_ws_payload

    class DeviceInfo(dict):
        pass

    class EntityCategory(str, enum.Enum):
        CONFIG = "config"
        DIAGNOSTIC = "diagnostic"
        SYSTEM = "system"

    entity_mod.DeviceInfo = DeviceInfo
    entity_mod.EntityCategory = EntityCategory

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

        @property
        def native_value(self) -> Any:
            return getattr(self, "_attr_native_value", None)

        @property
        def state(self) -> Any:
            if hasattr(self, "_attr_state"):
                return getattr(self, "_attr_state")

            value = self.native_value
            device_class = self.device_class
            if value is None:
                return None

            timestamp_class = getattr(SensorDeviceClass, "TIMESTAMP", "timestamp")
            if device_class == timestamp_class and isinstance(value, dt.datetime):
                if value.tzinfo is not None and value.tzinfo != dt.timezone.utc:
                    value = value.astimezone(dt.timezone.utc)
                return state_unknown_value

            date_class = getattr(SensorDeviceClass, "DATE", "date")
            if device_class == date_class and isinstance(value, dt.date):
                return value.isoformat()

            return value

    class SensorDeviceClass:
        ENERGY = "energy"
        POWER = "power"
        TEMPERATURE = "temperature"
        TIMESTAMP = "timestamp"

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

    class SelectEntity:
        def __init__(self) -> None:
            self.hass: Any | None = None

        async def async_added_to_hass(self) -> None:
            return None

        async def async_will_remove_from_hass(self) -> None:
            return None

        def schedule_update_ha_state(self) -> None:
            return None

        def async_write_ha_state(self) -> None:
            return None

        @property
        def current_option(self) -> Any:
            return getattr(self, "_attr_current_option", None)

    select_mod.SelectEntity = SelectEntity

    class NumberMode:
        SLIDER = "slider"

    class NumberEntity:
        def __init__(self) -> None:
            self.hass: Any | None = None

        async def async_added_to_hass(self) -> None:
            return None

        async def async_will_remove_from_hass(self) -> None:
            return None

        def schedule_update_ha_state(self) -> None:
            return None

        def async_write_ha_state(self) -> None:
            return None

    number_mod.NumberEntity = NumberEntity
    number_mod.NumberMode = NumberMode

    class RestoreEntity:
        async def async_added_to_hass(self) -> None:
            return None

        async def async_get_last_state(self) -> Any:
            return None

    restore_state_mod.RestoreEntity = RestoreEntity

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


from custom_components.termoweb.const import DOMAIN  # noqa: E402


@pytest.fixture
def heater_hass_data() -> Callable[..., dict[str, Any]]:
    """Return helper to attach TermoWeb domain data to a Home Assistant stub."""

    def _factory(
        hass: Any,
        entry_id: str,
        dev_id: str,
        coordinator: Any,
        *,
        boost_runtime: Mapping[str, Mapping[str, int]] | None = None,
        ws_state: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        inventory: "Inventory" | None = None,
    ) -> dict[str, Any]:
        from custom_components.termoweb.inventory import Inventory

        entry_data: dict[str, Any] = {
            "coordinator": coordinator,
            "dev_id": dev_id,
        }
        if boost_runtime is not None:
            entry_data["boost_runtime"] = {
                key: dict(value) for key, value in boost_runtime.items()
            }
        if ws_state is not None:
            entry_data["ws_state"] = dict(ws_state)
        if extra:
            entry_data.update(dict(extra))
        container = inventory
        if container is None:
            candidate = entry_data.get("inventory")
            if isinstance(candidate, Inventory):
                container = candidate
        if container is None:
            candidate = getattr(coordinator, "inventory", None)
            if isinstance(candidate, Inventory):
                container = candidate
        if container is not None:
            entry_data["inventory"] = container
        hass.data.setdefault(DOMAIN, {})[entry_id] = entry_data
        return entry_data

    return _factory


class FakeCoordinator:
    """Reusable coordinator stub shared across tests."""

    instances: list["FakeCoordinator"] = []

    @staticmethod
    def _normalise_device_record(record: Mapping[str, Any] | None) -> dict[str, Any]:
        """Return a copy of ``record`` with normalized heater metadata."""

        base: dict[str, Any]
        if isinstance(record, Mapping):
            base = dict(record)
        else:
            base = {}

        settings_source = base.get("settings") if isinstance(record, Mapping) else None
        normalised_settings: dict[str, dict[str, Any]] = {}
        if isinstance(settings_source, Mapping):
            for node_type, bucket in settings_source.items():
                if isinstance(bucket, Mapping):
                    normalised_settings[node_type] = dict(bucket)

        addresses_source = (
            base.get("addresses_by_type") if isinstance(record, Mapping) else None
        )
        normalised_addresses: dict[str, list[str]] = {}
        if isinstance(addresses_source, Mapping):
            for node_type, addrs in addresses_source.items():
                if isinstance(addrs, Iterable) and not isinstance(addrs, (str, bytes)):
                    normalised_addresses[node_type] = list(addrs)

        nodes_by_type = base.get("nodes_by_type")
        if isinstance(nodes_by_type, Mapping):
            nodes_copy: dict[str, Any] = {}
            for node_type, section in nodes_by_type.items():
                if not isinstance(section, Mapping):
                    continue
                section_copy = dict(section)
                node_settings = section_copy.get("settings")
                if (
                    isinstance(node_settings, Mapping)
                    and node_type not in normalised_settings
                ):
                    if not isinstance(node_settings, dict):
                        node_settings = dict(node_settings)
                    section_copy["settings"] = node_settings
                    normalised_settings[node_type] = node_settings
                elif isinstance(node_settings, dict):
                    section_copy["settings"] = node_settings
                    normalised_settings.setdefault(node_type, node_settings)

                node_addrs = section_copy.get("addrs")
                if (
                    isinstance(node_addrs, Iterable)
                    and not isinstance(node_addrs, (str, bytes))
                    and node_type not in normalised_addresses
                ):
                    normalised_addresses[node_type] = list(node_addrs)

                nodes_copy[node_type] = section_copy

            base["nodes_by_type"] = nodes_copy

        base["settings"] = normalised_settings
        # addresses_by_type is no longer stored in normalized records

        legacy = base.get("htr")
        legacy_copy = dict(legacy) if isinstance(legacy, Mapping) else {}
        htr_settings = normalised_settings.get("htr")
        if htr_settings and "settings" not in legacy_copy:
            legacy_copy["settings"] = dict(htr_settings)
        htr_addrs = normalised_addresses.get("htr")
        if htr_addrs and "addrs" not in legacy_copy:
            legacy_copy["addrs"] = list(htr_addrs)
        if legacy_copy:
            base["htr"] = legacy_copy

        return base

    def __init__(
        self,
        hass: Any,
        client: Any | None = None,
        base_interval: int = 0,
        dev_id: str = "dev",
        dev: dict[str, Any] | None = None,
        nodes: dict[str, Any] | None = None,
        inventory: "Inventory" | None = None,
        *,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.hass = hass
        self.client = client
        self.base_interval = base_interval
        self.dev_id = dev_id
        self.dev = self._normalise_device_record(dev)
        self.nodes = nodes or {}
        inventory_obj, nodes_list = _coerce_inventory(inventory)
        self.inventory: "Inventory" | None = inventory_obj
        self.node_inventory = list(nodes_list)
        self.update_interval = dt.timedelta(seconds=base_interval or 0)
        if data is not None:
            self.data = {
                key: self._normalise_device_record(value) for key, value in data.items()
            }
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

    def async_add_listener(
        self,
        listener: Callable[[], None],
        context: Any | None = None,
    ) -> None:
        if callable(listener):
            self.listeners.append(listener)

    def update_nodes(
        self,
        nodes: dict[str, Any],
        inventory: "Inventory" | None = None,
    ) -> None:
        self.nodes = nodes
        inventory_obj, nodes_list = _coerce_inventory(inventory)
        if inventory_obj is not None:
            self.inventory = inventory_obj
        elif inventory is None:
            self.inventory = None
        self.node_inventory = list(nodes_list)

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

    def resolve_boost_end(
        self,
        boost_end_day: Any,
        boost_end_min: Any,
        *,
        now: dt.datetime | None = None,
    ) -> tuple[dt.datetime | None, int | None]:
        """Mirror coordinator helper for translating boost end fields."""

        from custom_components.termoweb.boost import resolve_boost_end_from_fields

        return resolve_boost_end_from_fields(
            boost_end_day,
            boost_end_min,
            now=now,
        )


@dataclass
class DucaheatClientHarness:
    """Container for a fake Ducaheat REST client and its call history."""

    client: "DucaheatRESTClient"
    requests: list[tuple[str, str, dict[str, Any]]]
    segmented_calls: list[dict[str, Any]]
    rtc_calls: list[str]


@pytest.fixture
def ducaheat_rest_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., DucaheatClientHarness]:
    """Provide a factory that builds a fake Ducaheat REST client harness."""

    def factory(
        *,
        responses: Iterable[dict[str, Any] | None] | None = None,
        segmented_side_effects: Mapping[str, Exception] | None = None,
        headers: Mapping[str, str] | None = None,
        rtc_payload: Mapping[str, int] | None = None,
    ) -> DucaheatClientHarness:
        """Create a Ducaheat REST client with predictable helpers for tests."""

        from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient
        from custom_components.termoweb.const import (
            BRAND_DUCAHEAT,
            get_brand_user_agent,
        )
        from homeassistant.components.climate import HVACMode

        client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
        request_calls: list[tuple[str, str, dict[str, Any]]] = []
        segmented_calls: list[dict[str, Any]] = []
        rtc_calls: list[str] = []
        pending_responses = list(responses or [])
        segmented_effects = dict(segmented_side_effects or {})
        base_headers = {
            "Authorization": "Bearer token",
            "X-SerialId": "15",
            "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
        }
        if headers is not None:
            base_headers = dict(headers)
        rtc_template = dict(
            rtc_payload or {"y": 2024, "n": 1, "d": 1, "h": 0, "m": 0, "s": 0}
        )

        def _hvac_mode_str(self: HVACMode) -> str:
            """Return the enum value for consistent serialization."""

            return str(self.value)

        monkeypatch.setattr(HVACMode, "__str__", _hvac_mode_str, raising=False)

        async def fake_headers() -> dict[str, str]:
            """Return static authentication headers for the fake client."""

            return dict(base_headers)

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            """Record REST requests and return queued responses."""

            request_calls.append((method, path, kwargs))
            if pending_responses:
                response = pending_responses.pop(0)
                return dict(response or {})
            return {}

        async def fake_post_segmented(
            path: str,
            *,
            headers: dict[str, str],
            payload: Mapping[str, Any],
            dev_id: str,
            addr: str,
            node_type: str,
            ignore_statuses: tuple[int, ...] | None = None,
        ) -> dict[str, Any]:
            """Record segmented POST calls and replay optional side effects."""

            payload_copy = dict(payload)
            mode_value = payload_copy.get("mode")
            if isinstance(mode_value, str) and mode_value.startswith("hvacmode."):
                payload_copy["mode"] = mode_value.split(".", 1)[1]

            record = {
                "path": path,
                "payload": payload_copy,
                "dev_id": dev_id,
                "addr": addr,
                "node_type": node_type,
                "ignore_statuses": tuple(ignore_statuses or ()),
                "headers": dict(headers),
            }
            segmented_calls.append(record)
            request_calls.append(
                (
                    "POST",
                    path,
                    {
                        "headers": dict(headers),
                        "json": payload_copy,
                    },
                )
            )
            effect = segmented_effects.get(path)
            if effect is not None:
                raise effect
            return {"ok": True}

        async def fake_rtc(dev_id: str) -> dict[str, Any]:
            """Capture RTC lookups and return the configured template."""

            rtc_calls.append(dev_id)
            return dict(rtc_template)

        monkeypatch.setattr(client, "authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)
        monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)
        monkeypatch.setattr(client, "get_rtc_time", fake_rtc)

        return DucaheatClientHarness(
            client=client,
            requests=request_calls,
            segmented_calls=segmented_calls,
            rtc_calls=rtc_calls,
        )

    return factory


def pytest_runtest_setup(item: Any) -> None:  # pragma: no cover - ensure isolation
    _install_stubs()
