"""REST polling suspension driven by websocket health (``__init__``)."""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.termoweb.backend.ws_health import WsHealthTracker
from custom_components.termoweb.const import signal_ws_status
from tests.fakes.cloud import DEV_ID, FakeCloud


async def test_fresh_payload_without_a_window_suspends_polling_without_a_timer(
    hass: HomeAssistant, cloud: FakeCloud, config_entry: MockConfigEntry
) -> None:
    """With no staleness window there is nothing to time out: no resume timer."""
    config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    runtime = config_entry.runtime_data
    tracker = WsHealthTracker(DEV_ID)
    tracker.update_status("healthy")
    tracker.mark_payload()  # no stale_after: the payload never goes stale
    runtime.ws_trackers[DEV_ID] = tracker

    async_dispatcher_send(
        hass, signal_ws_status(config_entry.entry_id), {"payload_changed": True}
    )

    assert runtime.poll_suspended
    assert runtime.coordinator.update_interval is None
    assert runtime.poll_resume_unsub is None
