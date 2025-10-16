from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.termoweb.hourly_poller import HourlySamplesPoller
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util


@pytest.mark.asyncio
async def test_hourly_poller_runs_previous_hour(
    inventory_from_map,
) -> None:
    """The poller fetches the previous hour window and merges results once."""

    hass = HomeAssistant()
    inventory = inventory_from_map({"htr": ["A"], "pmo": ["M"]})
    backend = AsyncMock()
    backend.fetch_hourly_samples = AsyncMock(
        return_value={
            ("htr", "A"): [{"ts": datetime.now(timezone.utc), "energy_wh": 1_200.0}]
        }
    )
    coordinator = AsyncMock()
    coordinator.merge_samples_for_window = AsyncMock()
    poller = HourlySamplesPoller(hass, coordinator, backend, inventory)

    tz = dt_util.get_time_zone("Europe/Paris")
    now_local = datetime(2023, 3, 27, 10, 5, tzinfo=tz)
    original_tz = dt_util.DEFAULT_TIME_ZONE
    dt_util.set_default_time_zone(tz)
    try:
        await poller._run_for_previous_hour(now_local)

        call = backend.fetch_hourly_samples.await_args
        assert call.args[0] == inventory.dev_id
        nodes_arg = call.args[1]
        assert isinstance(nodes_arg, tuple)
        assert nodes_arg == (("htr", "A"), ("pmo", "M"))
        start_arg = call.args[2]
        end_arg = call.args[3]
        assert start_arg.tzinfo is not None
        assert end_arg.tzinfo is not None
        local_start = dt_util.as_local(start_arg)
        local_end = dt_util.as_local(end_arg)
        assert local_start.tzinfo == dt_util.DEFAULT_TIME_ZONE
        assert local_end.tzinfo == dt_util.DEFAULT_TIME_ZONE
        assert local_start.hour == 9 and local_start.minute == 0
        assert local_end.hour == 10 and local_end.minute == 0
        coordinator.merge_samples_for_window.assert_awaited_once_with(
            inventory.dev_id,
            backend.fetch_hourly_samples.return_value,
        )

        backend.fetch_hourly_samples.reset_mock()
        coordinator.merge_samples_for_window.reset_mock()
        await poller._run_for_previous_hour(now_local)
        backend.fetch_hourly_samples.assert_not_called()
        coordinator.merge_samples_for_window.assert_not_called()
    finally:
        dt_util.set_default_time_zone(original_tz)

    await poller.async_shutdown()


@pytest.mark.asyncio
async def test_hourly_poller_skips_inventories_without_targets(
    inventory_from_map,
) -> None:
    """Inventories without sample targets are ignored during polling."""

    hass = HomeAssistant()
    empty_inventory = inventory_from_map(None, dev_id="empty")
    populated_inventory = inventory_from_map({"htr": ["A"]}, dev_id="full")
    backend = AsyncMock()
    backend.fetch_hourly_samples = AsyncMock(return_value={})
    coordinator = AsyncMock()
    coordinator.merge_samples_for_window = AsyncMock()
    poller = HourlySamplesPoller(
        hass,
        coordinator,
        backend,
        (empty_inventory, populated_inventory),
    )

    tz = dt_util.get_time_zone("Europe/Paris")
    now_local = datetime(2023, 3, 27, 10, 5, tzinfo=tz)
    original_tz = dt_util.DEFAULT_TIME_ZONE
    dt_util.set_default_time_zone(tz)
    try:
        await poller._run_for_previous_hour(now_local)

        backend.fetch_hourly_samples.assert_awaited_once()
        call = backend.fetch_hourly_samples.await_args
        assert call.args[0] == populated_inventory.dev_id
        assert call.args[1] == (("htr", "A"),)
    finally:
        dt_util.set_default_time_zone(original_tz)

    await poller.async_shutdown()


def test_hourly_poller_on_time_threadsafe(inventory_from_map) -> None:
    """Ensure the time trigger schedules work on the event loop thread safely."""

    hass = HomeAssistant()
    hass.loop = MagicMock()
    captured: dict[str, Any] = {}

    def _capture(callback: Callable[[], None], /, *args: Any) -> None:
        captured["callback"] = callback
        captured["args"] = args

    hass.loop.call_soon_threadsafe.side_effect = _capture

    class FakeTask:
        def __init__(self, coro: Any) -> None:
            self.coro = coro
            self._callbacks: list[Callable[["FakeTask"], None]] = []

        def add_done_callback(self, callback: Callable[["FakeTask"], None]) -> None:
            self._callbacks.append(callback)

        def done(self) -> bool:
            return False

        def cancelled(self) -> bool:
            return False

        def exception(self) -> BaseException | None:
            return None

        def trigger(self) -> None:
            for callback in list(self._callbacks):
                callback(self)
            try:
                self.coro.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    created_tasks: list[FakeTask] = []

    def _make_task(coro: Any) -> FakeTask:
        task = FakeTask(coro)
        created_tasks.append(task)
        return task

    hass.loop.create_task = MagicMock(side_effect=_make_task)
    hass.async_create_task = MagicMock(side_effect=AssertionError())

    inventory = inventory_from_map({"htr": ["A"]})
    backend = AsyncMock()
    coordinator = AsyncMock()
    poller = HourlySamplesPoller(hass, coordinator, backend, inventory)

    poller._on_time(None)

    hass.loop.call_soon_threadsafe.assert_called_once()
    hass.loop.create_task.assert_not_called()
    callback = captured["callback"]
    args = captured["args"]

    callback(*args)

    hass.loop.create_task.assert_called_once()
    fake_task = created_tasks[0]
    assert poller._active_task is fake_task

    fake_task.trigger()
    assert poller._active_task is None
