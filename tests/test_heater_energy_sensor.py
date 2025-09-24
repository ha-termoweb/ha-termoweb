from __future__ import annotations

import asyncio
import copy
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import _install_stubs

_install_stubs()

from aiohttp import ClientError
from custom_components.termoweb import coordinator as coordinator_module
from custom_components.termoweb import sensor as sensor_module
from custom_components.termoweb import const as const_module
from homeassistant.components.sensor import SensorDeviceClass, SensorStateClass
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers import dispatcher as dispatcher_module
from homeassistant.helpers.dispatcher import dispatcher_send

TermoWebHeaterEnergyCoordinator = (
    coordinator_module.TermoWebHeaterEnergyCoordinator
)
TermoWebHeaterTemp = sensor_module.TermoWebHeaterTemp
TermoWebHeaterEnergyTotal = sensor_module.TermoWebHeaterEnergyTotal
TermoWebHeaterPower = sensor_module.TermoWebHeaterPower
TermoWebTotalEnergy = sensor_module.TermoWebTotalEnergy
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

        hass = HomeAssistant()
        coord = TermoWebHeaterEnergyCoordinator(hass, client, "1", ["A"])  # type: ignore[arg-type]

        await coord.async_refresh()
        await coord.async_refresh()

        assert coord.data["1"]["htr"]["energy"]["A"] == pytest.approx(0.0015)
        power = coord.data["1"]["htr"]["power"]["A"]
        assert power == pytest.approx(2.0, rel=1e-3)

        energy_sensor = TermoWebHeaterEnergyTotal(
            coord, "entry", "1", "A", "Energy", "e1", "Heater"
        )
        power_sensor = TermoWebHeaterPower(
            coord, "entry", "1", "A", "Power", "p1", "Heater"
        )
        energy_sensor.hass = power_sensor.hass = hass
        await energy_sensor.async_added_to_hass()
        await power_sensor.async_added_to_hass()

        assert energy_sensor.device_class == SensorDeviceClass.ENERGY
        assert energy_sensor.state_class == SensorStateClass.TOTAL_INCREASING
        assert energy_sensor.native_unit_of_measurement == "kWh"

        assert energy_sensor.native_value == pytest.approx(0.0015)
        assert power_sensor.native_value == pytest.approx(2.0, rel=1e-3)

        signal = signal_ws_data("entry")
        first_value: float = energy_sensor.native_value  # type: ignore[assignment]

        coord.data["1"]["htr"]["energy"]["A"] = "bad"
        coord.data["1"]["htr"]["power"]["A"] = "oops"
        assert energy_sensor.native_value is None
        assert power_sensor.native_value is None

        energy_sensor.schedule_update_ha_state = MagicMock()
        power_sensor.schedule_update_ha_state = MagicMock()

        coord.data["1"]["htr"]["energy"]["A"] = 0.002
        coord.data["1"]["htr"]["power"]["A"] = 123.0
        dispatcher_send(signal, {"dev_id": "1", "addr": "A"})

        energy_sensor.schedule_update_ha_state.assert_called_once()
        power_sensor.schedule_update_ha_state.assert_called_once()
        assert energy_sensor.native_value >= first_value
        assert energy_sensor.native_value == pytest.approx(0.002)
        assert power_sensor.native_value == pytest.approx(123.0)

        energy_sensor.schedule_update_ha_state.reset_mock()
        original_unsub_energy = energy_sensor._unsub_ws
        energy_sensor._unsub_ws = MagicMock(side_effect=original_unsub_energy)
        mock_energy_unsub = energy_sensor._unsub_ws
        await energy_sensor.async_will_remove_from_hass()
        mock_energy_unsub.assert_called_once()
        assert energy_sensor._handle_ws_message not in dispatch_map.get(signal, [])
        dispatcher_send(signal, {"dev_id": "1", "addr": "A"})
        energy_sensor.schedule_update_ha_state.assert_not_called()

        power_sensor.schedule_update_ha_state.reset_mock()
        original_unsub_power = power_sensor._unsub_ws
        power_sensor._unsub_ws = MagicMock(side_effect=original_unsub_power)
        mock_power_unsub = power_sensor._unsub_ws
        await power_sensor.async_will_remove_from_hass()
        mock_power_unsub.assert_called_once()

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
            "htr_addrs": ["1", "2"],
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

        with patch.object(
            TermoWebHeaterEnergyCoordinator,
            "async_config_entry_first_refresh",
            new=refresh_mock,
        ), patch.object(sensor_module, "_LOGGER", logger_mock):
            await async_setup_sensor_entry(hass, entry, _add_entities)

            assert "energy_coordinator" in record
            energy_coord = record["energy_coordinator"]
            assert isinstance(energy_coord, TermoWebHeaterEnergyCoordinator)
            assert refresh_mock.await_count == 1

            expected_count = len(record["htr_addrs"]) * 3 + 1
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

        sensor = TermoWebHeaterTemp(
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
        client.get_htr_samples = AsyncMock(
            side_effect=[
                [{"t": 1000, "counter": "1.0"}],
                [{"t": 1000, "counter": "2.0"}],
            ]
        )

        hass = HomeAssistant()
        coord = TermoWebHeaterEnergyCoordinator(hass, client, "1", ["A", "B"])  # type: ignore[arg-type]

        await coord.async_refresh()

        total_sensor = TermoWebTotalEnergy(coord, "entry", "1", "Total Energy", "tot")
        total_sensor.hass = hass
        await total_sensor.async_added_to_hass()

        assert total_sensor.device_class == SensorDeviceClass.ENERGY
        assert total_sensor.state_class == SensorStateClass.TOTAL_INCREASING
        assert total_sensor.native_unit_of_measurement == "kWh"

        assert total_sensor.native_value == pytest.approx(0.003)

        first_value: float = total_sensor.native_value  # type: ignore[assignment]

        signal = signal_ws_data("entry")
        total_sensor.schedule_update_ha_state = MagicMock()

        coord.data["1"]["htr"]["energy"]["A"] = 0.0015
        coord.data["1"]["htr"]["energy"]["B"] = 0.0025
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

    energy_sensor = TermoWebHeaterEnergyTotal(
        coordinator,
        "entry",
        "dev",
        "A",
        "Energy",
        "uid_energy",
        "Node",
    )
    power_sensor = TermoWebHeaterPower(
        coordinator,
        "entry",
        "dev",
        "A",
        "Power",
        "uid_power",
        "Node",
    )

    energy_info = energy_sensor.device_info
    assert energy_info["identifiers"] == {(DOMAIN, "dev", "A")}
    assert energy_sensor.available is True
    assert energy_sensor.native_value == pytest.approx(1.5)
    assert energy_sensor.extra_state_attributes == {"dev_id": "dev", "addr": "A"}

    energy_sensor._handle_ws_message({"dev_id": "other"})
    energy_sensor._handle_ws_message({"dev_id": "dev", "addr": "B"})

    coordinator.data = {"dev": {"htr": {"energy": {"A": None}}}}
    assert energy_sensor.native_value is None
    coordinator.data = {"dev": None}
    assert energy_sensor.available is False
    coordinator.data = copy.deepcopy(base_data)

    power_info = power_sensor.device_info
    assert power_info["identifiers"] == {(DOMAIN, "dev", "A")}
    assert power_sensor.available is True
    assert power_sensor.native_value == pytest.approx(250.0)
    assert power_sensor.extra_state_attributes == {"dev_id": "dev", "addr": "A"}

    power_sensor._handle_ws_message({"dev_id": "other"})
    power_sensor._handle_ws_message({"dev_id": "dev", "addr": "B"})

    coordinator.data = {"dev": {"htr": {"power": {"A": None}}}}
    assert power_sensor.native_value is None
    coordinator.data = {"dev": None}
    assert power_sensor.available is False
    coordinator.data = copy.deepcopy(base_data)

    total_sensor = TermoWebTotalEnergy(coordinator, "entry", "dev", "Total", "tot")
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

    energy_sensor = TermoWebHeaterEnergyTotal(
        coordinator,
        "entry",
        "dev",
        "A",
        "Energy",
        "uid",
        "Node",
    )
    total_sensor = TermoWebTotalEnergy(coordinator, "entry", "dev", "Total", "tot")

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
