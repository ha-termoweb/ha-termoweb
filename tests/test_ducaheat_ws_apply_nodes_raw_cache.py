"""Tests verifying the Ducaheat websocket does not cache node snapshots."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import AsyncMock

import aiohttp
import pytest

from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient

from tests.test_ducaheat_ws_protocol import (
    QueueWebSocket,
    _make_client,
    _run_read_loop,
)


@pytest.mark.asyncio
async def test_dev_data_snapshot_does_not_populate_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dev_data snapshots should rely on the inventory without caching nodes."""

    client = _make_client(monkeypatch)
    monkeypatch.setattr(client, "_subscribe_feeds", AsyncMock(return_value=0))
    dispatched: list[Mapping[str, Any]] = []
    client._dispatcher = lambda *_args: dispatched.append(_args[2])

    class DevDataWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__(
                [
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data='442["dev_data",{"nodes":{"htr":{"status":{"1":{"power":5}}}}}]',
                    )
                ]
            )

    client._ws = DevDataWS()  # type: ignore[assignment]

    await _run_read_loop(client)

    assert getattr(client, "_nodes_raw", None) is None
    assert dispatched
    payload = dispatched[-1]
    assert payload["inventory"] is client._inventory
    assert "addr_map" not in payload
    assert "addresses_by_type" not in payload


@pytest.mark.asyncio
async def test_update_settings_mirrors_dev_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accumulator charge fields should be mirrored into coordinator state."""

    client = _make_client(monkeypatch)
    normaliser = DucaheatRESTClient.__new__(DucaheatRESTClient)
    client._client.normalise_ws_nodes = normaliser.normalise_ws_nodes  # type: ignore[method-assign]
    dispatched: list[Mapping[str, Any]] = []
    client._dispatcher = lambda *_args: dispatched.append(_args[2])

    class UpdateWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__(
                [
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data='442["update",{"body":{"charging":"yes","current_charge_per":"12","target_charge_per":110},"path":"/api/v2/devs/device/acm/11/settings"}]',
                    )
                ]
            )

    client._ws = UpdateWS()  # type: ignore[assignment]

    await _run_read_loop(client)

    settings = client._coordinator.data.get("device", {}).get("settings", {})
    acm_settings = settings.get("acm") or {}
    assert acm_settings["11"]["charging"] is True
    assert acm_settings["11"]["current_charge_per"] == 12
    assert acm_settings["11"]["target_charge_per"] == 100
    assert dispatched


@pytest.mark.asyncio
async def test_update_events_do_not_cache_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Update events should flow through without retaining prior snapshots."""

    client = _make_client(monkeypatch)
    dispatched: list[Mapping[str, Any]] = []
    client._dispatcher = lambda *_args: dispatched.append(_args[2])

    class UpdateWS(QueueWebSocket):
        def __init__(self) -> None:
            super().__init__(
                [
                    SimpleNamespace(
                        type=aiohttp.WSMsgType.TEXT,
                        data='442["update",{"body":{"temp":19},"path":"/api/v2/devs/device/htr/1/status"}]',
                    )
                ]
            )

    client._ws = UpdateWS()  # type: ignore[assignment]

    await _run_read_loop(client)

    assert getattr(client, "_nodes_raw", None) is None
    assert dispatched
    payload = dispatched[-1]
    assert payload["inventory"] is client._inventory
    assert "addr_map" not in payload
    assert "addresses_by_type" not in payload
