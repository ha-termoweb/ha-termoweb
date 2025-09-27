from __future__ import annotations

import logging
import types

import pytest

import custom_components.termoweb.ws_shared as ws_shared


class DummyWS(ws_shared.TermoWebWSShared):
    async def _runner(self) -> None:  # pragma: no cover - never awaited in tests
        raise AssertionError("_runner should not be executed during unit tests")


def _make_hass() -> types.SimpleNamespace:
    loop = types.SimpleNamespace(create_task=lambda *args, **kwargs: None)
    return types.SimpleNamespace(loop=loop, data={})


def test_update_status_initialises_state(monkeypatch: pytest.MonkeyPatch) -> None:
    hass = _make_hass()

    dispatched: list[tuple[str, dict[str, str]]] = []

    def fake_dispatcher(hass_obj, signal: str, payload: dict[str, str]) -> None:
        dispatched.append((signal, payload))

    monkeypatch.setattr(ws_shared, "async_dispatcher_send", fake_dispatcher)
    monkeypatch.setattr(ws_shared.time, "time", lambda: 220.0)

    client = DummyWS(hass, entry_id="entry", dev_id="dev")
    client._stats.frames_total = 4
    client._stats.events_total = 2
    client._stats.last_event_ts = 123.0
    client._healthy_since = 100.0

    client._update_status("connected")

    state = hass.data[ws_shared.DOMAIN]["entry"]["ws_state"]["dev"]

    assert state == {
        "status": "connected",
        "last_event_at": 123.0,
        "healthy_since": 100.0,
        "healthy_minutes": 2,
        "frames_total": 4,
        "events_total": 2,
    }

    assert dispatched == [
        (
            ws_shared.signal_ws_status("entry"),
            {"dev_id": "dev", "status": "connected"},
        )
    ]


def test_mark_event_tracks_health(monkeypatch: pytest.MonkeyPatch) -> None:
    hass = _make_hass()
    client = DummyWS(hass, entry_id="entry", dev_id="dev")

    statuses: list[str] = []

    def fake_update(status: str) -> None:
        statuses.append(status)

    client._update_status = fake_update  # type: ignore[assignment]
    client._stats.frames_total = 7
    client._connected_since = 100.0

    monkeypatch.setattr(ws_shared.time, "time", lambda: 401.0)
    monkeypatch.setattr(ws_shared._LOGGER, "isEnabledFor", lambda level: True)

    client._mark_event(paths=["/a", "/b", "/a"])

    state = hass.data[ws_shared.DOMAIN]["entry"]["ws_state"]["dev"]

    assert state["events_total"] == 1
    assert state["frames_total"] == 7
    assert state["last_event_at"] == 401.0
    # Healthy transition should trigger exactly once.
    assert statuses == ["healthy"]
    assert client._healthy_since == pytest.approx(401.0)
    assert client._stats.last_paths == ["/a", "/b"]

