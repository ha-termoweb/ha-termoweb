from datetime import datetime, timezone
from unittest.mock import AsyncMock

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
        return_value={("htr", "A"): [{"ts": datetime.now(timezone.utc), "energy_wh": 1_200.0}]}
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
        assert set(call.args[1]) == {("htr", "A"), ("pmo", "M")}
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
