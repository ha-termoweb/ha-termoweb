"""Tests for the abstract backend base helpers."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.termoweb.backend.base import Backend, BoostContext
from custom_components.termoweb.inventory import NodeDescriptor


class DummyWsClient:
    """Simple websocket client stub that tracks lifecycle calls."""

    def __init__(self, hass: SimpleNamespace, *, entry_id: str, dev_id: str) -> None:
        self.hass = hass
        self.entry_id = entry_id
        self.dev_id = dev_id
        self._loop = getattr(hass, "loop", asyncio.get_event_loop())
        self._task: asyncio.Task | None = None
        self.started = False
        self.stopped = False

    async def _runner(self) -> None:
        self.started = True
        await asyncio.sleep(0)

    def start(self) -> asyncio.Task:
        if self._task is None:
            self._task = self._loop.create_task(self._runner())
        return self._task

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        self.stopped = True


class ExampleBackend(Backend):
    """Concrete backend used to exercise the abstract base class."""

    def __init__(self, *, brand: str, client: Any) -> None:
        super().__init__(brand=brand, client=client)
        self.calls: list[dict[str, Any]] = []

    async def get_instant_power(self, node: NodeDescriptor) -> float | None:
        """Return ``None`` for the abstract backend helper."""

        return None

    def create_ws_client(
        self,
        hass: SimpleNamespace,
        entry_id: str,
        dev_id: str,
        coordinator: Any,
        *,
        inventory: Any | None = None,
    ) -> DummyWsClient:
        self.calls.append(
            {
                "hass": hass,
                "entry_id": entry_id,
                "dev_id": dev_id,
                "coordinator": coordinator,
                "inventory": inventory,
            }
        )
        return DummyWsClient(hass, entry_id=entry_id, dev_id=dev_id)

    async def fetch_hourly_samples(
        self,
        dev_id: str,
        nodes: Any,
        start_local: datetime,
        end_local: datetime,
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Return an empty sample mapping for the dummy backend."""

        return {}


class SettingsClient:
    """Client stub that records set_node_settings calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def set_node_settings(
        self,
        dev_id: str,
        node: NodeDescriptor,
        *,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
        units: str = "C",
    ) -> None:
        """Record the payload sent to the client."""

        self.calls.append(
            {
                "dev_id": dev_id,
                "node": node,
                "mode": mode,
                "stemp": stemp,
                "prog": prog,
                "ptemp": ptemp,
                "units": units,
            }
        )


@pytest.mark.asyncio
async def test_backend_properties_and_ws_creation() -> None:
    """The backend stores metadata and returns the websocket stub."""

    hass = SimpleNamespace(loop=asyncio.get_running_loop())
    client = object()
    coordinator = object()
    backend = ExampleBackend(brand="termoweb", client=client)

    assert backend.brand == "termoweb"
    assert backend.client is client

    inventory = object()
    ws_client = backend.create_ws_client(
        hass,
        entry_id="entry-1",
        dev_id="dev-1",
        coordinator=coordinator,
        inventory=inventory,
    )
    assert isinstance(ws_client, DummyWsClient)
    assert backend.calls == [
        {
            "hass": hass,
            "entry_id": "entry-1",
            "dev_id": "dev-1",
            "coordinator": coordinator,
            "inventory": inventory,
        }
    ]

    task = ws_client.start()
    assert isinstance(task, asyncio.Task)
    assert not task.done()

    await ws_client.stop()
    assert ws_client.stopped is True


@pytest.mark.asyncio
async def test_backend_set_node_settings_passes_through() -> None:
    """The base backend delegates settings writes to the client."""

    client = SettingsClient()
    backend = ExampleBackend(brand="termoweb", client=client)

    await backend.set_node_settings(
        "dev-1",
        ("htr", "4"),
        mode="auto",
        stemp=21.5,
        prog=[0] * 168,
        ptemp=[17.0, 19.0, 21.0],
        units="F",
        boost_context=BoostContext(active=True),
    )

    assert client.calls == [
        {
            "dev_id": "dev-1",
            "node": ("htr", "4"),
            "mode": "auto",
            "stemp": 21.5,
            "prog": [0] * 168,
            "ptemp": [17.0, 19.0, 21.0],
            "units": "F",
        }
    ]


# ---------------------------------------------------------------------------
# Backend delegation methods
# ---------------------------------------------------------------------------


class FullClient:
    """Client stub recording all delegation calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def set_node_settings(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("set_node_settings", {"args": args, "kwargs": kwargs}))

    async def set_acm_boost_state(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("set_acm_boost_state", {"args": args, "kwargs": kwargs}))

    async def set_node_lock(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("set_node_lock", {"args": args, "kwargs": kwargs}))

    async def set_node_display_select(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("set_node_display_select", {"args": args, "kwargs": kwargs}))

    async def set_node_priority(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("set_node_priority", {"args": args, "kwargs": kwargs}))

    async def get_power_limit(self, dev_id: str) -> int | None:
        self.calls.append(("get_power_limit", {"dev_id": dev_id}))
        return 3000

    async def set_power_limit(self, dev_id: str, *, power_limit: int) -> None:
        self.calls.append(("set_power_limit", {"dev_id": dev_id, "power_limit": power_limit}))

    async def get_node_samples(self, *args: Any, **kwargs: Any) -> list:
        self.calls.append(("get_node_samples", {"args": args}))
        return []

    async def list_devices(self) -> list:
        return []

    async def get_nodes(self, dev_id: str) -> Any:
        return None

    async def get_node_settings(self, dev_id: str, node: Any) -> Any:
        return None


@pytest.mark.asyncio
async def test_backend_set_acm_boost_state() -> None:
    """set_acm_boost_state delegates to client correctly."""

    client = FullClient()
    backend = ExampleBackend(brand="termoweb", client=client)
    await backend.set_acm_boost_state("dev-1", "2", boost=True, boost_time=120)

    assert len(client.calls) == 1
    assert client.calls[0][0] == "set_acm_boost_state"


@pytest.mark.asyncio
async def test_backend_set_node_lock() -> None:
    """set_node_lock delegates to client correctly."""

    client = FullClient()
    backend = ExampleBackend(brand="termoweb", client=client)
    await backend.set_node_lock("dev-1", ("htr", "1"), lock=True)

    assert len(client.calls) == 1
    assert client.calls[0][0] == "set_node_lock"


@pytest.mark.asyncio
async def test_backend_set_node_display_select() -> None:
    """set_node_display_select delegates to client."""

    client = FullClient()
    backend = ExampleBackend(brand="termoweb", client=client)
    await backend.set_node_display_select("dev-1", ("htr", "1"), select=True)

    assert len(client.calls) == 1
    assert client.calls[0][0] == "set_node_display_select"


@pytest.mark.asyncio
async def test_backend_set_node_priority() -> None:
    """set_node_priority delegates to client."""

    client = FullClient()
    backend = ExampleBackend(brand="termoweb", client=client)
    await backend.set_node_priority("dev-1", ("htr", "1"), priority=5)

    assert len(client.calls) == 1
    assert client.calls[0][0] == "set_node_priority"


@pytest.mark.asyncio
async def test_backend_get_power_limit() -> None:
    """get_power_limit delegates to client and returns result."""

    client = FullClient()
    backend = ExampleBackend(brand="termoweb", client=client)
    result = await backend.get_power_limit("dev-1")

    assert result == 3000
    assert client.calls[0][0] == "get_power_limit"


@pytest.mark.asyncio
async def test_backend_set_power_limit() -> None:
    """set_power_limit delegates to client."""

    client = FullClient()
    backend = ExampleBackend(brand="termoweb", client=client)
    await backend.set_power_limit("dev-1", power_limit=2500)

    assert client.calls[0][0] == "set_power_limit"


# ---------------------------------------------------------------------------
# _resolve_node_descriptor
# ---------------------------------------------------------------------------


def test_resolve_node_descriptor_tuple() -> None:
    """_resolve_node_descriptor handles (type, addr) tuples."""

    client = FullClient()
    backend = ExampleBackend(brand="termoweb", client=client)
    node_type, addr = backend._resolve_node_descriptor(("htr", "01"))

    assert node_type == "htr"
    assert addr == "01"


def test_resolve_node_descriptor_object() -> None:
    """_resolve_node_descriptor handles objects with type/addr attributes."""

    client = FullClient()
    backend = ExampleBackend(brand="termoweb", client=client)
    node = SimpleNamespace(type="htr", addr="02")
    node_type, addr = backend._resolve_node_descriptor(node)

    assert node_type == "htr"
    assert addr == "02"


def test_resolve_node_descriptor_invalid_raises() -> None:
    """_resolve_node_descriptor raises ValueError for invalid descriptors."""

    client = FullClient()
    backend = ExampleBackend(brand="termoweb", client=client)

    with pytest.raises(ValueError, match="Invalid node descriptor"):
        backend._resolve_node_descriptor(("", ""))


# ---------------------------------------------------------------------------
# normalise_sample_records
# ---------------------------------------------------------------------------

from custom_components.termoweb.backend.base import normalise_sample_records


def test_normalise_sample_records_basic() -> None:
    """normalise_sample_records converts raw records to sorted samples."""

    records = [
        {"t": 1000.0, "counter": 5000.0},
        {"t": 900.0, "counter": 4000.0},
    ]
    result = normalise_sample_records("htr", records)
    assert len(result) == 2
    assert result[0]["ts"] < result[1]["ts"]
    # htr scale = 1000, divider = 1.0
    assert result[0]["energy_wh"] == 4000.0


def test_normalise_sample_records_pmo_scale() -> None:
    """Power monitor records use a different scale factor."""

    records = [{"t": 1000.0, "counter": 3_600_000.0}]
    result = normalise_sample_records("pmo", records)
    assert len(result) == 1
    # pmo scale = 3_600_000, divider = 3600
    assert result[0]["energy_wh"] == 1000.0


def test_normalise_sample_records_missing_timestamp_skipped() -> None:
    """Records without valid timestamp are skipped."""

    records = [{"t": None, "counter": 1000.0}]
    result = normalise_sample_records("htr", records)
    assert len(result) == 0


def test_normalise_sample_records_missing_counter_uses_fallbacks() -> None:
    """Records with missing counter try counter_max, counter_min, value."""

    records = [
        {"t": 1000.0, "counter_max": 2000.0},
        {"t": 1001.0, "counter_min": 3000.0},
        {"t": 1002.0, "value": 4000.0},
    ]
    result = normalise_sample_records("htr", records)
    assert len(result) == 3


def test_normalise_sample_records_no_counter_at_all_skipped() -> None:
    """Records with no counter data are skipped."""

    records = [{"t": 1000.0}]
    result = normalise_sample_records("htr", records)
    assert len(result) == 0


def test_normalise_sample_records_non_mapping_skipped() -> None:
    """Non-mapping records are skipped."""

    records = ["not a record", 42, None]
    result = normalise_sample_records("htr", records)
    assert len(result) == 0


def test_normalise_sample_records_includes_power() -> None:
    """Power field is included when present in records."""

    records = [{"t": 1000.0, "counter": 5000.0, "power": 250.0}]
    result = normalise_sample_records("htr", records)
    assert len(result) == 1
    assert result[0]["power_w"] == 250.0


# ---------------------------------------------------------------------------
# fetch_normalised_hourly_samples
# ---------------------------------------------------------------------------

from custom_components.termoweb.backend.base import fetch_normalised_hourly_samples
from unittest.mock import AsyncMock as AsyncMockType
import datetime as dt_module
from homeassistant.util import dt as dt_util


@pytest.mark.asyncio
async def test_fetch_normalised_hourly_samples_end_before_start() -> None:
    """Returns empty when end <= start."""

    client = AsyncMockType()
    tz = dt_util.get_time_zone("Europe/Paris")
    original_tz = dt_util.DEFAULT_TIME_ZONE
    dt_util.set_default_time_zone(tz)

    try:
        now = dt_module.datetime(2023, 6, 1, 10, 0, tzinfo=tz)
        result = await fetch_normalised_hourly_samples(
            client=client,
            dev_id="dev",
            nodes=[("htr", "01")],
            start_local=now,
            end_local=now,
        )
        assert result == {}
    finally:
        dt_util.set_default_time_zone(original_tz)


@pytest.mark.asyncio
async def test_fetch_normalised_hourly_samples_skips_invalid_nodes() -> None:
    """Empty/invalid node types/addrs are skipped."""

    client = AsyncMockType()
    tz = dt_util.get_time_zone("Europe/Paris")
    original_tz = dt_util.DEFAULT_TIME_ZONE
    dt_util.set_default_time_zone(tz)

    try:
        now = dt_module.datetime(2023, 6, 1, 10, 0, tzinfo=tz)
        result = await fetch_normalised_hourly_samples(
            client=client,
            dev_id="dev",
            nodes=[("", ""), ("invalid", "01")],
            start_local=now - dt_module.timedelta(hours=1),
            end_local=now,
        )
        assert result == {}
    finally:
        dt_util.set_default_time_zone(original_tz)


@pytest.mark.asyncio
async def test_fetch_normalised_hourly_samples_client_error() -> None:
    """Client errors are logged and that node is skipped."""

    client = AsyncMockType()
    client.get_node_samples = AsyncMockType(side_effect=Exception("fail"))
    tz = dt_util.get_time_zone("Europe/Paris")
    original_tz = dt_util.DEFAULT_TIME_ZONE
    dt_util.set_default_time_zone(tz)

    try:
        now = dt_module.datetime(2023, 6, 1, 10, 0, tzinfo=tz)
        result = await fetch_normalised_hourly_samples(
            client=client,
            dev_id="dev",
            nodes=[("htr", "01")],
            start_local=now - dt_module.timedelta(hours=1),
            end_local=now,
        )
        assert result == {}
    finally:
        dt_util.set_default_time_zone(original_tz)


@pytest.mark.asyncio
async def test_fetch_normalised_hourly_samples_non_iterable_response() -> None:
    """Non-iterable client responses are skipped."""

    client = AsyncMockType()
    client.get_node_samples = AsyncMockType(return_value=42)
    tz = dt_util.get_time_zone("Europe/Paris")
    original_tz = dt_util.DEFAULT_TIME_ZONE
    dt_util.set_default_time_zone(tz)

    try:
        now = dt_module.datetime(2023, 6, 1, 10, 0, tzinfo=tz)
        result = await fetch_normalised_hourly_samples(
            client=client,
            dev_id="dev",
            nodes=[("htr", "01")],
            start_local=now - dt_module.timedelta(hours=1),
            end_local=now,
        )
        assert result == {}
    finally:
        dt_util.set_default_time_zone(original_tz)
