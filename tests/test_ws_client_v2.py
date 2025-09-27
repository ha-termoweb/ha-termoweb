from __future__ import annotations

import asyncio
import copy
import json
import logging
import types
from typing import Any, Iterable

import pytest

import custom_components.termoweb.ws_client as ws_core
import custom_components.termoweb.ws_client_v2 as ws_v2


def test_ducaheat_ws_client_flow(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    dispatcher_calls: list[tuple[str, dict[str, object]]] = []

    def fake_dispatcher(hass, signal: str, payload: dict[str, object]) -> None:
        dispatcher_calls.append((signal, payload))

    monkeypatch.setattr(ws_v2, "async_dispatcher_send", fake_dispatcher)
    monkeypatch.setattr(ws_core, "async_dispatcher_send", fake_dispatcher)

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        caplog.set_level(logging.DEBUG, logger=ws_core.__name__)
        caplog.set_level(logging.DEBUG, logger=ws_v2.__name__)
        node_updates: list[tuple[dict[str, Any], list[Any]]] = []
        energy_updates: list[Any] = []

        class RecordingCoordinator:
            def update_nodes(self, nodes: dict[str, Any], inventory: list[Any]) -> None:
                node_updates.append((copy.deepcopy(nodes), list(inventory)))

        class FakeEnergyCoordinator:
            def update_addresses(self, addrs: Iterable[str] | dict[str, Iterable[str]]) -> None:
                if isinstance(addrs, dict):
                    energy_updates.append({k: list(v) for k, v in addrs.items()})
                else:
                    energy_updates.append(list(addrs))

        hass = types.SimpleNamespace(
            loop=loop,
            data={
                ws_v2.DOMAIN: {
                    "entry": {
                        "ws_state": {},
                        "energy_coordinator": FakeEnergyCoordinator(),
                    }
                }
            },
        )
        coordinator = RecordingCoordinator()

        class FakeClient:
            api_base = "https://api-tevolve.termoweb.net/"

            async def _authed_headers(self) -> dict[str, str]:
                return {"Authorization": "Bearer tok"}

        client = ws_v2.DucaheatWSClient(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=FakeClient(),
            coordinator=coordinator,
        )

        url = await client.ws_url()
        assert url == "https://api-tevolve.termoweb.net/api/v2/socket_io?token=tok"

        task = client.start()
        assert isinstance(task, asyncio.Task)
        await asyncio.sleep(0)

        handshake_payload = {"devs": [{"id": "dev"}], "permissions": {"dev": ["read"]}}
        client._on_frame(
            json.dumps({"event": "dev_handshake", "data": handshake_payload})
        )
        await asyncio.sleep(0)
        handshake_payload["devs"][0]["id"] = "mutated"
        assert client._handshake is not None
        assert client._handshake["devs"][0]["id"] == "dev"

        initial_update = {
            "nodes": {
                "htr": {"status": {"01": {"temp": 20}}},
                "acm": {"status": {"A1": {"mode": "eco"}}},
            }
        }
        client._on_frame(json.dumps({"event": "update", "data": initial_update}))
        await asyncio.sleep(0)

        dev_data_payload = {
            "nodes": {
                "htr": {"status": {"01": {"temp": 20}}},
                "acm": {"status": {"A1": {"mode": "eco"}}},
                "raw": {"meta": {"foo": "bar"}},
            }
        }
        client._on_frame(json.dumps({"event": "dev_data", "data": dev_data_payload}))
        await asyncio.sleep(0)

        incremental_update = {
            "nodes": {
                "htr": {"status": {"02": {"temp": 21}}},
                "acm": {"status": {"A2": {"mode": "boost"}}},
                "raw": {"meta": {"foo": "bar", "extra": True}},
                "metrics": 3,
            }
        }
        client._on_frame(json.dumps({"event": "update", "data": incremental_update}))
        await asyncio.sleep(0)

        client._on_frame(json.dumps({"event": "update", "data": None}))
        client._on_frame(json.dumps({"event": "unknown", "data": {}}))
        client._on_frame(json.dumps("literal"))
        client._on_frame("not-json")
        await asyncio.sleep(0)

        await client.stop()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        status_signal = ws_v2.signal_ws_status("entry")
        data_signal = ws_v2.signal_ws_data("entry")

        status_updates = [
            payload["status"]
            for signal, payload in dispatcher_calls
            if signal == status_signal
        ]
        assert status_updates == [
            "starting",
            "connecting",
            "connected",
            "healthy",
            "healthy",
            "healthy",
            "stopped",
        ]

        data_payloads = [
            payload for signal, payload in dispatcher_calls if signal == data_signal
        ]
        assert len(data_payloads) == 3
        assert data_payloads[0]["nodes"] == initial_update["nodes"]
        assert data_payloads[0]["node_type"] is None
        assert data_payloads[1]["nodes"] == dev_data_payload["nodes"]
        assert data_payloads[1]["nodes_by_type"]["acm"]["status"] == {"A1": {"mode": "eco"}}
        assert data_payloads[2]["nodes"]["htr"]["status"] == {
            "01": {"temp": 20},
            "02": {"temp": 21},
        }
        assert data_payloads[2]["nodes"]["acm"]["status"] == {
            "A1": {"mode": "eco"},
            "A2": {"mode": "boost"},
        }
        assert data_payloads[2]["nodes"]["raw"] == {
            "meta": {"foo": "bar", "extra": True},
        }
        assert len(node_updates) == len(energy_updates) == 3
        assert node_updates[0][0] == initial_update["nodes"]
        assert node_updates[1][0] == dev_data_payload["nodes"]
        assert node_updates[2][0]["htr"]["status"] == {
            "01": {"temp": 20},
            "02": {"temp": 21},
        }
        assert node_updates[2][0]["acm"]["status"] == {
            "A1": {"mode": "eco"},
            "A2": {"mode": "boost"},
        }
        assert all(not update for update in energy_updates)
        assert data_payloads[2]["nodes"]["metrics"] == 3
        for payload in data_payloads:
            assert payload["dev_id"] == "dev"
            assert payload["nodes"] is not client._nodes["nodes"]

        assert client._nodes["nodes_by_type"]["htr"]["status"]["02"]["temp"] == 21
        assert client._nodes["nodes_by_type"]["acm"]["status"]["A2"]["mode"] == "boost"
        assert client._healthy_since is not None
        assert client._status == "stopped"
        assert "unknown node types" not in caplog.text

    asyncio.run(_run())


def test_ducaheat_ws_client_logs_unknown_types(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    dispatcher_calls: list[tuple[str, dict[str, object]]] = []

    def fake_dispatcher(hass, signal: str, payload: dict[str, object]) -> None:
        dispatcher_calls.append((signal, payload))

    monkeypatch.setattr(ws_v2, "async_dispatcher_send", fake_dispatcher)
    monkeypatch.setattr(ws_core, "async_dispatcher_send", fake_dispatcher)

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        caplog.set_level(logging.DEBUG, logger=ws_core.__name__)
        caplog.set_level(logging.DEBUG, logger=ws_v2.__name__)

        node_updates: list[tuple[dict[str, Any], list[Any]]] = []
        energy_updates: list[Any] = []

        class RecordingCoordinator:
            def update_nodes(self, nodes: dict[str, Any], inventory: list[Any]) -> None:
                node_updates.append((copy.deepcopy(nodes), list(inventory)))

        class FakeEnergyCoordinator:
            def update_addresses(self, addrs: Iterable[str] | dict[str, Iterable[str]]) -> None:
                if isinstance(addrs, dict):
                    energy_updates.append({k: list(v) for k, v in addrs.items()})
                else:
                    energy_updates.append(list(addrs))

        hass = types.SimpleNamespace(
            loop=loop,
            data={
                ws_v2.DOMAIN: {
                    "entry": {
                        "ws_state": {},
                        "energy_coordinator": FakeEnergyCoordinator(),
                    }
                }
            },
        )
        coordinator = RecordingCoordinator()

        class FakeClient:
            async def _authed_headers(self) -> dict[str, str]:
                return {"Authorization": "Bearer tok"}

        client = ws_v2.DucaheatWSClient(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=FakeClient(),
            coordinator=coordinator,
        )

        payload = {"nodes": {"nodes": [{"addr": "X", "type": "gizmo"}, {"addr": "01", "type": "htr"}]}}
        client._handle_dev_data(payload)
        await asyncio.sleep(0)

        assert any("unknown node types in inventory: gizmo" in rec.message for rec in caplog.records)
        assert energy_updates == [{"gizmo": ["X"], "htr": ["01"]}]
        assert node_updates

    asyncio.run(_run())


def test_ducaheat_ws_client_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    dispatcher_calls: list[tuple[str, dict[str, object]]] = []

    def fake_dispatcher(hass, signal: str, payload: dict[str, object]) -> None:
        dispatcher_calls.append((signal, payload))

    monkeypatch.setattr(ws_v2, "async_dispatcher_send", fake_dispatcher)
    monkeypatch.setattr(ws_core, "async_dispatcher_send", fake_dispatcher)

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(loop=loop, data={})

        class FakeClient:
            async def _authed_headers(self) -> dict[str, str]:
                return {"Authorization": "Bearer token"}

        client = ws_v2.DucaheatWSClient(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=FakeClient(),
            coordinator=types.SimpleNamespace(),
        )

        # Stop before the runner starts to exercise the early stop path.
        await client.stop()

        # Starting twice should reuse the same task and not spawn another runner.
        task = client.start()
        assert client.start() is task
        await asyncio.sleep(0)

        # Invalid payloads should be ignored without raising.
        client._on_frame(json.dumps({"event": "dev_handshake", "data": None}))
        client._on_frame(json.dumps({"event": "dev_data", "data": {}}))
        client._on_frame(
            json.dumps({"event": "dev_data", "data": {"nodes": ["bad"]}})
        )
        await asyncio.sleep(0)

        # Cancel the task so stop() hits the CancelledError branch.
        task.cancel()
        await asyncio.sleep(0)
        await client.stop()

        status_signal = ws_v2.signal_ws_status("entry")
        status_updates = [
            payload["status"]
            for signal, payload in dispatcher_calls
            if signal == status_signal
        ]
        assert status_updates[0] == "starting"
        assert status_updates.count("disconnected") == 1
        assert status_updates[-1] == "stopped"
        assert client.hass.data[ws_v2.DOMAIN]["entry"]["ws_state"]["dev"]["status"] == "stopped"
        assert client._status == "stopped"

    asyncio.run(_run())


def test_ws_url_requires_token() -> None:
    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(
            loop=loop,
            data={ws_v2.DOMAIN: {"entry": {"ws_state": {}}}},
        )
        coordinator = types.SimpleNamespace()

        class NoTokenClient:
            async def _authed_headers(self) -> dict[str, str]:
                return {}

        client = ws_v2.DucaheatWSClient(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=NoTokenClient(),
            coordinator=coordinator,
        )

        with pytest.raises(RuntimeError):
            await client.ws_url()

    asyncio.run(_run())


def test_stop_handles_cancelled_task(monkeypatch: pytest.MonkeyPatch) -> None:
    dispatcher_calls: list[tuple[str, dict[str, object]]] = []

    def fake_dispatcher(hass, signal: str, payload: dict[str, object]) -> None:
        dispatcher_calls.append((signal, payload))

    monkeypatch.setattr(ws_v2, "async_dispatcher_send", fake_dispatcher)
    monkeypatch.setattr(ws_core, "async_dispatcher_send", fake_dispatcher)

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        hass = types.SimpleNamespace(loop=loop, data={})

        class FakeClient:
            async def _authed_headers(self) -> dict[str, str]:
                return {"Authorization": "Bearer token"}

        client = ws_v2.DucaheatWSClient(
            hass,
            entry_id="entry",
            dev_id="dev",
            api_client=FakeClient(),
            coordinator=types.SimpleNamespace(),
        )

        async def hanging_runner(self: ws_v2.DucaheatWSClient) -> None:
            self._update_status("connecting")
            try:
                await asyncio.Future()
            finally:
                self._update_status("stopped")

        client._runner = types.MethodType(hanging_runner, client)

        task = client.start()
        await asyncio.sleep(0)
        assert not task.done()

        task.cancel()
        await asyncio.sleep(0)

        await client.stop()
        await asyncio.sleep(0)

        assert client._task is None
        status_signal = ws_v2.signal_ws_status("entry")
        status_updates = [
            payload["status"]
            for signal, payload in dispatcher_calls
            if signal == status_signal
        ]
        assert status_updates[-1] == "stopped"

    asyncio.run(_run())
