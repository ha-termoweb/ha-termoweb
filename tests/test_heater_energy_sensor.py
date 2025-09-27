from __future__ import annotations

import asyncio
import copy
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import _install_stubs, make_ws_payload

_install_stubs()

from aiohttp import ClientError
from custom_components.termoweb import coordinator as coordinator_module
from custom_components.termoweb import sensor as sensor_module
from custom_components.termoweb import const as const_module
from custom_components.termoweb.nodes import build_node_inventory
from homeassistant.components.sensor import SensorDeviceClass, SensorStateClass
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers import dispatcher as dispatcher_module
from homeassistant.helpers.dispatcher import dispatcher_send

EnergyStateCoordinator = coordinator_module.EnergyStateCoordinator
HeaterTemperatureSensor = sensor_module.HeaterTemperatureSensor
HeaterEnergyTotalSensor = sensor_module.HeaterEnergyTotalSensor
HeaterPowerSensor = sensor_module.HeaterPowerSensor
AccumulatorEnergyTotalSensor = sensor_module.AccumulatorEnergyTotalSensor
AccumulatorPowerSensor = sensor_module.AccumulatorPowerSensor
InstallationTotalEnergySensor = sensor_module.InstallationTotalEnergySensor
async_setup_sensor_entry = sensor_module.async_setup_entry
signal_ws_data = const_module.signal_ws_data
DOMAIN = const_module.DOMAIN

dispatch_map = dispatcher_module._dispatch_map

if hasattr(const_module, "signal_poll_refresh"):
    signal_poll_refresh = const_module.signal_poll_refresh
else:

    def signal_poll_refresh(entry_id: str) -> str:
        return f"{signal_ws_data(entry_id)}:poll"


def test_coordinator_and_sensors() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_htr_samples = AsyncMock(
            side_effect=[
                [{"t": 1000, "counter": "1.0"}],
                [{"t": 1900, "counter": "1.5"}],
            ]
        )
        client.get_node_samples = AsyncMock(
            side_effect=[
                [{"t": 1000, "counter": "1.0"}],
                [{"t": 1000, "counter": "3.0"}],
                [{"t": 1900, "counter": "1.5"}],
                [{"t": 1900, "counter": "3.5"}],
            ]
        )

        hass = HomeAssistant()
        coord = EnergyStateCoordinator(hass, client, "1", {"htr": ["A"], "acm": ["B"]})

        await coord.async_refresh()
        await coord.async_refresh()

        device_data = coord.data["1"]
        assert device_data["htr"]["energy"]["A"] == pytest.approx(0.0015)
        assert device_data["htr"]["power"]["A"] == pytest.approx(2.0, rel=1e-3)
        acm_section = device_data["nodes_by_type"]["acm"]
        assert acm_section["energy"]["B"] == pytest.approx(0.0035)
        assert acm_section["power"]["B"] == pytest.approx(2.0, rel=1e-3)

        sensors = {
            ("htr", "energy"): HeaterEnergyTotalSensor(
                coord,
                "entry",
                "1",
                "A",
                "Energy",
                f"{DOMAIN}:1:htr:A:energy",
                "Heater",
            ),
            ("htr", "power"): HeaterPowerSensor(
                coord,
                "entry",
                "1",
                "A",
                "Power",
                f"{DOMAIN}:1:htr:A:power",
                "Heater",
            ),
            ("acm", "energy"): AccumulatorEnergyTotalSensor(
                coord,
                "entry",
                "1",
                "B",
                "Accumulator Energy",
                f"{DOMAIN}:1:acm:B:energy",
                "Accumulator",
                node_type="acm",
            ),
            ("acm", "power"): AccumulatorPowerSensor(
                coord,
                "entry",
                "1",
                "B",
                "Accumulator Power",
                f"{DOMAIN}:1:acm:B:power",
                "Accumulator",
                node_type="acm",
            ),
        }

        for sensor in sensors.values():
            sensor.hass = hass
            await sensor.async_added_to_hass()

        energy_sensor = sensors[("htr", "energy")]
        assert energy_sensor.device_class == SensorDeviceClass.ENERGY
        assert energy_sensor.state_class == SensorStateClass.TOTAL_INCREASING
        assert energy_sensor.native_unit_of_measurement == "kWh"
        assert energy_sensor._attr_unique_id == f"{DOMAIN}:1:htr:A:energy"
        assert sensors[("acm", "energy")]._attr_unique_id == f"{DOMAIN}:1:acm:B:energy"
        assert sensors[("acm", "power")]._attr_unique_id == f"{DOMAIN}:1:acm:B:power"

        signal = signal_ws_data("entry")

        expected_initial = {
            ("htr", "energy"): pytest.approx(0.0015),
            ("htr", "power"): pytest.approx(2.0, rel=1e-3),
            ("acm", "energy"): pytest.approx(0.0035),
            ("acm", "power"): pytest.approx(2.0, rel=1e-3),
        }
        for key, sensor in sensors.items():
            assert sensor.native_value == expected_initial[key]

        energy_baselines = {
            key: sensor.native_value
            for key, sensor in sensors.items()
            if key[1] == "energy"
        }

        def _set_metric(node_type: str, metric: str, value: Any) -> None:
            section = coord.data["1"]["nodes_by_type"][node_type][metric]
            addr = "A" if node_type == "htr" else "B"
            section[addr] = value

        for node_type, metric in sensors:
            _set_metric(node_type, metric, "bad" if metric == "energy" else "oops")

        for sensor in sensors.values():
            assert sensor.native_value is None
            sensor.schedule_update_ha_state = MagicMock()

        _set_metric("htr", "energy", 0.002)
        _set_metric("htr", "power", 123.0)

        dispatcher_send(signal, make_ws_payload("1", "A"))

        for key, sensor in sensors.items():
            node_type, metric = key
            if node_type == "htr":
                sensor.schedule_update_ha_state.assert_called_once()
                expected = 0.002 if metric == "energy" else 123.0
                assert sensor.native_value == pytest.approx(expected)
            else:
                sensor.schedule_update_ha_state.assert_not_called()

        _set_metric("acm", "energy", 0.004)
        _set_metric("acm", "power", 456.0)

        dispatcher_send(signal, make_ws_payload("1", "B", node_type="acm"))

        for key, sensor in sensors.items():
            node_type, metric = key
            if node_type == "acm":
                sensor.schedule_update_ha_state.assert_called_once()
                expected = 0.004 if metric == "energy" else 456.0
                assert sensor.native_value == pytest.approx(expected)
            else:
                assert sensor.schedule_update_ha_state.call_count == 1

        for key, baseline in energy_baselines.items():
            new_value = sensors[key].native_value
            assert new_value is not None and new_value >= baseline  # type: ignore[operator]

        for sensor in sensors.values():
            sensor.schedule_update_ha_state.reset_mock()
            original_unsub = sensor._unsub_ws
            sensor._unsub_ws = MagicMock(side_effect=original_unsub)
            mock_unsub = sensor._unsub_ws
            await sensor.async_will_remove_from_hass()
            mock_unsub.assert_called_once()
            assert sensor._handle_ws_message not in dispatch_map.get(signal, [])

        dispatcher_send(signal, make_ws_payload("1", "A"))
        dispatcher_send(signal, make_ws_payload("1", "B", node_type="acm"))
        for sensor in sensors.values():
            sensor.schedule_update_ha_state.assert_not_called()

        assert client.get_htr_samples.await_count == 0
        assert client.get_node_samples.await_count == 4
        call_args = [call.args[:2] for call in client.get_node_samples.await_args_list]
        assert call_args[:4] == [
            ("1", ("htr", "A")),
            ("1", ("acm", "B")),
            ("1", ("htr", "A")),
            ("1", ("acm", "B")),
        ]

    asyncio.run(_run())


def test_sensor_async_setup_entry_defaults_and_skips_invalid() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-default")
        dev_id = "dev-default"
        raw_nodes = {
            "nodes": [
                {"type": "htr", "addr": "1"},
                {"type": "acm", "addr": "2"},
                {"type": "pmo", "addr": "P1"},
            ]
        }
        inventory = build_node_inventory(raw_nodes)
        inventory.append(types.SimpleNamespace(type=" ", addr="extra"))
        inventory.append(types.SimpleNamespace(type="htr", addr=" "))

        coordinator = types.SimpleNamespace(
            hass=hass,
            data={
                dev_id: {
                    "nodes": {},
                    "htr": {"settings": {}, "energy": {}, "power": {}},
                }
            },
        )

        energy_coord = types.SimpleNamespace(update_addresses=MagicMock())

        hass.data = {
            DOMAIN: {
                entry.entry_id: {
                    "coordinator": coordinator,
                    "client": types.SimpleNamespace(),
                    "dev_id": dev_id,
                    "nodes": {},
                    "node_inventory": inventory,
                    "energy_coordinator": energy_coord,
                }
            }
        }

        added: list = []

        def _add_entities(entities: list) -> None:
            added.extend(entities)

        await async_setup_sensor_entry(hass, entry, _add_entities)

        energy_coord.update_addresses.assert_called_once()

        names = sorted(
            getattr(ent, "_attr_name", getattr(ent, "name", None)) for ent in added
        )
        assert names == [
            "Accumulator 2 Energy",
            "Accumulator 2 Power",
            "Accumulator 2 Temperature",
            "Node 1 Energy",
            "Node 1 Power",
            "Node 1 Temperature",
            "Total Energy",
        ]

    asyncio.run(_run())


def test_sensor_async_setup_entry_ignores_blank_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-skip")
        dev_id = "dev-skip"

        coordinator = types.SimpleNamespace(hass=hass, data={dev_id: {}})

        hass.data = {
            DOMAIN: {
                entry.entry_id: {
                    "coordinator": coordinator,
                    "client": types.SimpleNamespace(),
                    "dev_id": dev_id,
                    "energy_coordinator": types.SimpleNamespace(
                        update_addresses=lambda addrs: None
                    ),
                }
            }
        }

        blank_node = types.SimpleNamespace(addr=" ")
        valid_node = types.SimpleNamespace(addr="9")

        def fake_prepare(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any, Any]:
            return (
                [],
                {"acm": [blank_node, valid_node]},
                {"acm": ["9"]},
                lambda *_: "Heater",
            )

        monkeypatch.setattr(
            sensor_module, "prepare_heater_platform_data", fake_prepare
        )

        added: list[Any] = []

        def _add_entities(entities: list[Any]) -> None:
            added.extend(entities)

        await async_setup_sensor_entry(hass, entry, _add_entities)

        ids = [
            getattr(sensor, "unique_id", getattr(sensor, "_attr_unique_id", ""))
            for sensor in added
        ]
        assert any(":acm:9" in uid for uid in ids)
        assert not any("::" in uid for uid in ids)

    asyncio.run(_run())


def test_sensor_async_setup_entry_creates_entities_and_reuses_coordinator() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-setup")
        dev_id = "dev-1"
        nodes_meta = {
            "nodes": [
                {"type": "HTR", "addr": "1", "name": "Living Room"},
                {"type": "htr", "addr": "2", "name": "Bedroom"},
            ]
        }
        coordinator = types.SimpleNamespace(
            hass=hass,
            data={
                dev_id: {
                    "nodes": nodes_meta,
                    "htr": {
                        "settings": {
                            "1": {"mtemp": "21.0", "units": "C"},
                            "2": {"mtemp": "19.5", "units": "C"},
                        },
                        "energy": {"1": 1.0, "2": 2.0},
                        "power": {"1": 100.0, "2": 150.0},
                    },
                }
            },
        )
        record: dict = {
            "coordinator": coordinator,
            "client": types.SimpleNamespace(),
            "dev_id": dev_id,
            "nodes": nodes_meta,
            "node_inventory": build_node_inventory(nodes_meta),
        }
        hass.data = {DOMAIN: {entry.entry_id: record}}

        add_calls: list[list] = []
        added_entities: list = []

        def _add_entities(entities):
            add_calls.append(list(entities))
            added_entities.extend(entities)

        refresh_mock = AsyncMock()
        logger_mock = MagicMock()

        expected_names = [
            "Living Room Temperature",
            "Living Room Energy",
            "Living Room Power",
            "Bedroom Temperature",
            "Bedroom Energy",
            "Bedroom Power",
            "Total Energy",
        ]

        with (
            patch.object(
                EnergyStateCoordinator,
                "async_config_entry_first_refresh",
                new=refresh_mock,
            ),
            patch.object(sensor_module, "_LOGGER", logger_mock),
        ):
            await async_setup_sensor_entry(hass, entry, _add_entities)

            assert "energy_coordinator" in record
            energy_coord = record["energy_coordinator"]
            assert isinstance(energy_coord, EnergyStateCoordinator)
            assert refresh_mock.await_count == 1

            heater_addrs = energy_coord._addresses_by_type
            expected_count = sum(len(addrs) for addrs in heater_addrs.values()) * 3 + 1
            assert len(add_calls) == 1
            assert len(add_calls[0]) == expected_count
            assert len(added_entities) == expected_count

            names = [
                getattr(ent, "_attr_name", getattr(ent, "name", None))
                for ent in add_calls[0]
            ]
            assert names == expected_names
            logger_mock.debug.assert_called_once_with(
                "Adding %d TermoWeb sensors", expected_count
            )

            logger_mock.debug.reset_mock()
            refresh_mock.reset_mock()

            await async_setup_sensor_entry(hass, entry, _add_entities)

            assert record["energy_coordinator"] is energy_coord
            assert refresh_mock.await_count == 0
            assert len(add_calls) == 2
            assert len(add_calls[1]) == expected_count
            assert len(added_entities) == expected_count * 2

            names_second = [
                getattr(ent, "_attr_name", getattr(ent, "name", None))
                for ent in add_calls[1]
            ]
            assert names_second == expected_names
            logger_mock.debug.assert_called_once_with(
                "Adding %d TermoWeb sensors", expected_count
            )

    asyncio.run(_run())


def test_sensor_async_setup_entry_rebuilds_inventory_when_missing() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        entry = types.SimpleNamespace(entry_id="entry-missing")
        dev_id = "dev-missing"
        nodes_meta = {
            "nodes": [
                {"type": "htr", "addr": "A1", "name": "Living"},
                {"type": "HTR", "addr": "B2", "name": "Bedroom"},
            ]
        }

        coordinator = types.SimpleNamespace(
            hass=hass,
            data={
                dev_id: {
                    "nodes": nodes_meta,
                    "htr": {
                        "settings": {"A1": {}, "B2": {}},
                        "energy": {},
                        "power": {},
                    },
                }
            },
        )

        record: dict[str, Any] = {
            "coordinator": coordinator,
            "client": types.SimpleNamespace(),
            "dev_id": dev_id,
            "nodes": nodes_meta,
        }
        hass.data = {DOMAIN: {entry.entry_id: record}}

        added: list = []

        def _add_entities(entities: list) -> None:
            added.extend(entities)

        refresh_mock = AsyncMock()
        with patch.object(
            EnergyStateCoordinator,
            "async_config_entry_first_refresh",
            new=refresh_mock,
        ):
            await async_setup_sensor_entry(hass, entry, _add_entities)

        assert refresh_mock.await_count == 1
        assert len(added) == 7
        stored_inventory = hass.data[DOMAIN][entry.entry_id]["node_inventory"]
        assert [node.addr for node in stored_inventory] == ["A1", "B2"]

    asyncio.run(_run())


def test_heater_temp_sensor() -> None:
    async def _run() -> None:
        hass = HomeAssistant()
        coordinator = types.SimpleNamespace(
            hass=hass,
            data={
                "dev1": {
                    "nodes": {"nodes": [{"type": "htr", "addr": "A"}]},
                    "htr": {
                        "settings": {
                            "A": {
                                "mtemp": "21.5",
                                "units": "C",
                                "timestamp": 1_700_000_000,
                            }
                        }
                    },
                }
            },
        )

        sensor = HeaterTemperatureSensor(
            coordinator,
            "entry",
            "dev1",
            "A",
            "Living Room Temperature",
            "temp1",
            "Living Room",
        )

        original_async_on_remove = sensor.async_on_remove
        sensor.async_on_remove = MagicMock(side_effect=original_async_on_remove)

        await sensor.async_added_to_hass()

        ws_signal = signal_ws_data("entry")
        poll_signal = signal_poll_refresh("entry")

        assert sensor._unsub_ws is not None
        assert sensor._handle_ws_message in dispatch_map[ws_signal]
        sensor.async_on_remove.assert_called_once()

        info = sensor.device_info
        assert info["identifiers"] == {(DOMAIN, "dev1", "A")}
        assert info["name"] == "Living Room"
        assert info["manufacturer"] == "TermoWeb"
        assert info["model"] == "Heater"
        assert info["via_device"] == (DOMAIN, "dev1")

        assert sensor.native_unit_of_measurement == UnitOfTemperature.CELSIUS
        assert sensor.available is True
        assert sensor.native_value == pytest.approx(21.5)
        assert sensor.extra_state_attributes == {
            "dev_id": "dev1",
            "addr": "A",
            "units": "C",
        }

        original_nodes = coordinator.data["dev1"]["nodes"]
        coordinator.data["dev1"]["nodes"] = None
        assert sensor.available is False
        coordinator.data["dev1"]["nodes"] = original_nodes
        assert sensor.available is True

        original_device = coordinator.data["dev1"]
        coordinator.data["dev1"] = {}
        assert sensor.available is False
        coordinator.data["dev1"] = original_device
        assert sensor.available is True

        settings = coordinator.data["dev1"]["htr"]["settings"]["A"]
        settings["mtemp"] = "bad"
        assert sensor.native_value is None

        settings["mtemp"] = "22.75"
        assert sensor.native_value == pytest.approx(22.75)

        sensor.schedule_update_ha_state = MagicMock()

        dispatcher_send(ws_signal, {"dev_id": "other", "addr": "A"})
        dispatcher_send(ws_signal, {"dev_id": "dev1", "addr": "B"})
        dispatcher_send(poll_signal, {"dev_id": "dev1", "addr": "A"})
        sensor.schedule_update_ha_state.assert_not_called()

        new_value = "23.5"
        settings["mtemp"] = new_value
        dispatcher_send(ws_signal, {"dev_id": "dev1", "addr": "A"})
        sensor.schedule_update_ha_state.assert_called_once()
        assert sensor.native_value == pytest.approx(float(new_value))

        sensor.schedule_update_ha_state.reset_mock()
        dispatcher_send(ws_signal, {"dev_id": "dev1"})
        sensor.schedule_update_ha_state.assert_called_once()

        sensor.schedule_update_ha_state.reset_mock()
        original_unsub = sensor._unsub_ws
        assert original_unsub is not None
        sensor._unsub_ws = MagicMock(side_effect=original_unsub)
        mock_unsub = sensor._unsub_ws
        await sensor.async_will_remove_from_hass()
        mock_unsub.assert_called_once()
        assert sensor._handle_ws_message not in dispatch_map.get(ws_signal, [])

        dispatcher_send(ws_signal, {"dev_id": "dev1", "addr": "A"})
        dispatcher_send(poll_signal, {"dev_id": "dev1", "addr": "A"})
        sensor.schedule_update_ha_state.assert_not_called()

    asyncio.run(_run())


def test_total_energy_sensor() -> None:
    async def _run() -> None:
        client = types.SimpleNamespace()
        client.get_node_samples = AsyncMock(
            side_effect=[
                [{"t": 1000, "counter": "1.0"}],
                [{"t": 1000, "counter": "2.0"}],
            ]
        )

        hass = HomeAssistant()
        coord = EnergyStateCoordinator(
            hass,
            client,
            "1",
            {"htr": ["A"], "acm": ["B"]},
        )

        await coord.async_refresh()

        total_sensor = InstallationTotalEnergySensor(
            coord, "entry", "1", "Total Energy", "tot"
        )
        total_sensor.hass = hass
        await total_sensor.async_added_to_hass()

        nodes_by_type = coord.data["1"]["nodes_by_type"]
        assert nodes_by_type["htr"] is coord.data["1"]["htr"]
        assert nodes_by_type["acm"] is coord.data["1"]["acm"]

        assert total_sensor.device_class == SensorDeviceClass.ENERGY
        assert total_sensor.state_class == SensorStateClass.TOTAL_INCREASING
        assert total_sensor.native_unit_of_measurement == "kWh"

        assert total_sensor.native_value == pytest.approx(0.003)

        first_value: float = total_sensor.native_value  # type: ignore[assignment]

        signal = signal_ws_data("entry")
        total_sensor.schedule_update_ha_state = MagicMock()

        coord.data["1"]["htr"]["energy"]["A"] = 0.0015
        coord.data["1"]["acm"]["energy"]["B"] = 0.0025
        coord.data["1"]["htr"]["energy"]["C"] = "bad"
        dispatcher_send(signal, {"dev_id": "1", "addr": "A"})

        total_sensor.schedule_update_ha_state.assert_called_once()
        assert total_sensor.native_value >= first_value
        assert total_sensor.native_value == pytest.approx(0.004)

        total_sensor.schedule_update_ha_state.reset_mock()
        original_unsub = total_sensor._unsub_ws
        total_sensor._unsub_ws = MagicMock(side_effect=original_unsub)
        mock_total_unsub = total_sensor._unsub_ws
        await total_sensor.async_will_remove_from_hass()
        mock_total_unsub.assert_called_once()
        assert total_sensor._on_ws_data not in dispatch_map.get(signal, [])
        dispatcher_send(signal, {"dev_id": "1", "addr": "B"})
        total_sensor.schedule_update_ha_state.assert_not_called()

    asyncio.run(_run())


def test_energy_and_power_sensor_properties() -> None:
    hass = HomeAssistant()
    base_data = {
        "dev": {
            "nodes": {"nodes": []},
            "htr": {
                "energy": {"A": 1500},
                "power": {"A": 250},
            },
        }
    }
    coordinator = types.SimpleNamespace(hass=hass, data=copy.deepcopy(base_data))

    sensors = {
        "energy": HeaterEnergyTotalSensor(
            coordinator,
            "entry",
            "dev",
            "A",
            "Energy",
            "uid_energy",
            "Node",
        ),
        "power": HeaterPowerSensor(
            coordinator,
            "entry",
            "dev",
            "A",
            "Power",
            "uid_power",
            "Node",
        ),
    }

    expected_values = {
        "energy": pytest.approx(1.5),
        "power": pytest.approx(250.0),
    }

    for metric, sensor in sensors.items():
        info = sensor.device_info
        assert info["identifiers"] == {(DOMAIN, "dev", "A")}
        assert sensor.available is True
        assert sensor.native_value == expected_values[metric]
        assert sensor.extra_state_attributes == {"dev_id": "dev", "addr": "A"}

        sensor._handle_ws_message({"dev_id": "other"})
        sensor._handle_ws_message({"dev_id": "dev", "addr": "B"})

        coordinator.data = {"dev": {"htr": {metric: {"A": None}}}}
        assert sensor.native_value is None

        coordinator.data = {"dev": None}
        assert sensor.available is False

        coordinator.data = copy.deepcopy(base_data)

    total_sensor = InstallationTotalEnergySensor(
        coordinator, "entry", "dev", "Total", "tot"
    )
    total_info = total_sensor.device_info
    assert total_info == {"identifiers": {(DOMAIN, "dev")}}
    assert total_sensor.available is True
    assert total_sensor.native_value == pytest.approx(1.5)
    assert total_sensor.extra_state_attributes == {"dev_id": "dev"}

    total_sensor._on_ws_data({"dev_id": "other"})

    coordinator.data = {"dev": {"htr": {"energy": {}}}}
    assert total_sensor.native_value is None
    coordinator.data = {"dev": None}
    assert total_sensor.available is False


def test_energy_sensor_respects_scale_metadata() -> None:
    hass = HomeAssistant()
    base = {
        "dev": {
            "nodes": {},
            "htr": {
                "energy": {"A": "1500", "B": "500"},
                "power": {},
            },
        }
    }
    coordinator = types.SimpleNamespace(hass=hass, data=copy.deepcopy(base))

    energy_sensor = HeaterEnergyTotalSensor(
        coordinator,
        "entry",
        "dev",
        "A",
        "Energy",
        "uid",
        "Node",
    )
    total_sensor = InstallationTotalEnergySensor(
        coordinator, "entry", "dev", "Total", "tot"
    )

    assert energy_sensor.native_value == pytest.approx(1.5)
    assert total_sensor.native_value == pytest.approx(2.0)

    coordinator.data["dev"]["htr"]["energy"]["B"] = 0

    coordinator.data["dev"]["htr"]["energy"]["A"] = True
    assert energy_sensor.native_value is None

    coordinator.data["dev"]["htr"]["energy"]["A"] = "nan"
    assert energy_sensor.native_value is None

    coordinator._termoweb_energy_scale = "Wh"
    coordinator.data["dev"]["htr"]["energy"]["A"] = 2000
    assert energy_sensor.native_value == pytest.approx(2.0)

    coordinator._termoweb_energy_scale = "kWh"
    coordinator.data["dev"]["htr"]["energy"]["A"] = "2.5"
    assert energy_sensor.native_value == pytest.approx(2.5)

    coordinator._termoweb_energy_scale = "0.5"
    coordinator.data["dev"]["htr"]["energy"]["A"] = 4
    assert energy_sensor.native_value == pytest.approx(2.0)

    coordinator._termoweb_energy_scale = 2.0
    coordinator.data["dev"]["htr"]["energy"]["A"] = 1.5
    assert energy_sensor.native_value == pytest.approx(3.0)

    coordinator._termoweb_energy_scale = "invalid"
    coordinator.data["dev"]["htr"]["energy"]["A"] = "2.0"
    assert energy_sensor.native_value == pytest.approx(2.0)


def test_looks_like_integer_string_handles_signs_and_whitespace() -> None:
    helper = sensor_module._looks_like_integer_string

    assert helper("+123") is True
    assert helper(" -42 ") is True
    assert helper("   ") is False
