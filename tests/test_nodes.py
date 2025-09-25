"""Unit tests for node models."""

from __future__ import annotations

import pytest

from custom_components.termoweb.const import BRAND_DUCAHEAT, BRAND_TERMOWEB
from custom_components.termoweb.nodes import (
    AccumulatorNode,
    DucaheatAccum,
    HeaterNode,
    PowerMonitorNode,
    ThermostatNode,
)


def test_heater_node_normalises_inputs() -> None:
    node = HeaterNode(name=" Living ", addr=2, brand=BRAND_TERMOWEB)

    assert node.name == "Living"
    assert node.addr == "2"
    assert node.type == "htr"
    assert node.brand == BRAND_TERMOWEB
    assert node.supports_boost() is False


def test_accumulator_node_defaults() -> None:
    node = AccumulatorNode(name=None, addr="007", brand=BRAND_TERMOWEB)

    assert node.name == ""
    assert node.addr == "007"
    assert node.type == "acm"
    assert node.supports_boost() is False


def test_ducaheat_accum_supports_boost() -> None:
    node = DucaheatAccum(name="Tank", addr="4")

    assert node.brand == BRAND_DUCAHEAT
    assert node.type == "acm"
    assert node.supports_boost() is True


def test_power_monitor_stub() -> None:
    node = PowerMonitorNode(name="Monitor", addr="P1", brand=BRAND_TERMOWEB)

    with pytest.raises(NotImplementedError):
        node.power_level()


def test_thermostat_stub() -> None:
    node = ThermostatNode(name="Thermostat", addr="T1", brand=BRAND_TERMOWEB)

    with pytest.raises(NotImplementedError):
        node.capabilities()


def test_node_requires_brand() -> None:
    with pytest.raises(ValueError):
        HeaterNode(name="Living", addr=1, brand="")
