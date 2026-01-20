"""Tests verifying the Ducaheat websocket does not cache node snapshots."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import aiohttp
import pytest

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


@pytest.mark.asyncio
async def test_update_events_do_not_cache_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Update events should flow through without retaining prior snapshots."""

    client = _make_client(monkeypatch)

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
