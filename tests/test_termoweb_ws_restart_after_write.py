"""Tests for restarting the websocket after idle writes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from custom_components.termoweb.backend import termoweb_ws as module
from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient
from homeassistant.core import HomeAssistant

from tests.test_termoweb_ws_protocol import DummyREST


@pytest.fixture
def termoweb_client(monkeypatch: pytest.MonkeyPatch) -> TermoWebWSClient:
    """Create a ``TermoWebWSClient`` instance for idle restart tests."""

    hass = HomeAssistant()
    hass.loop = SimpleNamespace(
        call_soon_threadsafe=lambda cb, *args: cb(*args),
        create_task=lambda coro, **_: SimpleNamespace(done=lambda: False),
        is_running=lambda: False,
    )
    hass.loop_thread_id = 0
    hass.data.setdefault(module.DOMAIN, {})["entry"] = {}

    coordinator = SimpleNamespace(data={}, update_nodes=MagicMock())
    monkeypatch.setattr(TermoWebWSClient, "_install_write_hook", lambda self: None)

    client = TermoWebWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=coordinator,
        session=SimpleNamespace(closed=False),
    )
    return client


def _tracker_stub(*, payload: float | None, heartbeat: float | None) -> SimpleNamespace:
    """Return a tracker stub exposing the relevant timestamps."""

    return SimpleNamespace(last_payload_at=payload, last_heartbeat_at=heartbeat)


@pytest.mark.asyncio
async def test_maybe_restart_after_write_returns_without_timestamps(
    termoweb_client: TermoWebWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restart helper should exit immediately when idle timestamps are missing."""

    tracker = _tracker_stub(payload=None, heartbeat=None)
    monkeypatch.setattr(termoweb_client, "_ws_health_tracker", lambda: tracker)
    monkeypatch.setattr(module.time, "time", lambda: 10.0)
    restart_spy = MagicMock()
    monkeypatch.setattr(termoweb_client, "_schedule_idle_restart", restart_spy)
    termoweb_client._stats.last_event_ts = 0.0
    termoweb_client._last_event_at = None

    await termoweb_client.maybe_restart_after_write()

    restart_spy.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_restart_after_write_skips_when_window_not_elapsed(
    termoweb_client: TermoWebWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restart helper should ignore writes when they occur within the idle window."""

    tracker = _tracker_stub(payload=100.0, heartbeat=None)
    monkeypatch.setattr(termoweb_client, "_ws_health_tracker", lambda: tracker)
    monkeypatch.setattr(
        module.time,
        "time",
        lambda: 100.0 + termoweb_client._payload_idle_window - 1.0,
    )
    restart_spy = MagicMock()
    monkeypatch.setattr(termoweb_client, "_schedule_idle_restart", restart_spy)

    await termoweb_client.maybe_restart_after_write()

    restart_spy.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_restart_after_write_schedules_restart_when_idle_window_exceeded(
    termoweb_client: TermoWebWSClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restart helper should restart the websocket when the idle window elapses."""

    idle_gap = termoweb_client._payload_idle_window + 12.5
    tracker = _tracker_stub(payload=None, heartbeat=50.0)
    monkeypatch.setattr(termoweb_client, "_ws_health_tracker", lambda: tracker)
    monkeypatch.setattr(module.time, "time", lambda: 50.0 + idle_gap)
    restart_spy = MagicMock()
    monkeypatch.setattr(termoweb_client, "_schedule_idle_restart", restart_spy)

    await termoweb_client.maybe_restart_after_write()

    restart_spy.assert_called_once_with(
        idle_for=pytest.approx(idle_gap),
        source="write notification",
    )

