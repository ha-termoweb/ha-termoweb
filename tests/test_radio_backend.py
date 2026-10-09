"""Tests for RadioBackend and its factory wiring."""

from __future__ import annotations

from datetime import datetime
import importlib
from types import SimpleNamespace

import pytest
from fake_radio_link import NET, FakeRadioLink

from custom_components.termoweb.const import BRAND_RADIO
from custom_components.termoweb.inventory import Inventory, build_node_inventory


def _mod(name: str):
    """Import a backend module at test time (other suites reload these modules)."""

    return importlib.import_module(f"custom_components.termoweb.backend{name}")


NODES = [{"type": "htr", "addr": "6", "name": "Living room"}]


def make_backend():
    """Return a radio backend over a fake-linked client."""

    client = _mod(".radio_client").RadioClient(
        "radio.local", 2323, "B", NODES, network_id=NET, link_factory=FakeRadioLink
    )
    backend = _mod(".factory").create_backend(brand=BRAND_RADIO, client=client)
    assert isinstance(backend, _mod(".radio_backend").RadioBackend)
    return backend


def test_brand_constant_and_backend_class() -> None:
    """The radio brand resolves to RadioBackend, also through the lazy export."""

    radio_backend = _mod(".radio_backend").RadioBackend
    assert BRAND_RADIO == "radio"
    assert _mod(".factory")._backend_class("radio") is radio_backend  # noqa: SLF001
    assert _mod("").RadioBackend is radio_backend


def test_capabilities_switch_off_cloud_only_features() -> None:
    """Radio: keypad lock, local power limit and priority; no energy history."""

    factory = _mod(".factory")
    assert factory.backend_capabilities(BRAND_RADIO) == _mod(
        ".base"
    ).BackendCapabilities(
        lock=True,
        power_limit=True,
        priority=True,
        energy_history=False,
        energy=True,
    )
    assert make_backend().capabilities is factory.backend_capabilities(BRAND_RADIO)


def test_create_radio_client_helper() -> None:
    """The factory helper builds a lazily connecting client for PR 4's setup."""

    client = _mod("").create_radio_client("10.0.0.5", 2323, "A", NODES, None)
    assert isinstance(client, _mod(".radio_client").RadioClient)
    assert client.dialect.name == "A"
    assert client.link is None
    client = _mod("").create_radio_client("10.0.0.5", 2323, "B", NODES, NET)
    assert client.dialect.name == "B"
    with pytest.raises(ValueError, match="explicit network_id"):
        _mod("").create_radio_client("10.0.0.5", 2323, "B", NODES, None)


def test_create_ws_client_returns_listener() -> None:
    """The websocket slot is filled by the radio listener."""

    backend = make_backend()
    inventory = Inventory("aabbcc001122", build_node_inventory(NODES))
    listener = backend.create_ws_client(
        SimpleNamespace(data={}),
        "entry",
        "aabbcc001122",
        SimpleNamespace(),
        inventory=inventory,
    )
    assert isinstance(listener, _mod(".radio_ws").RadioListener)
    assert listener.dev_id == "aabbcc001122" and listener.entry_id == "entry"


def test_create_ws_client_requires_radio_client() -> None:
    """A radio backend paired with another client type is a wiring bug."""

    radio_backend = _mod(".radio_backend").RadioBackend
    backend = radio_backend(brand=BRAND_RADIO, client=SimpleNamespace())
    with pytest.raises(TypeError, match="RadioClient"):
        backend.create_ws_client(SimpleNamespace(data={}), "e", "d", SimpleNamespace())


@pytest.mark.asyncio
async def test_fetch_hourly_samples_is_empty() -> None:
    """Radio heaters keep no energy history."""

    backend = make_backend()
    result = await backend.fetch_hourly_samples(
        "aabbcc001122", [("htr", "6")], datetime(2026, 1, 1), datetime(2026, 1, 2)
    )
    assert result == {}


@pytest.mark.asyncio
async def test_base_backend_delegates_writes_to_radio_client() -> None:
    """Backend.set_node_settings and friends reach the radio client."""

    backend = make_backend()
    link = await backend.client.async_connect()
    link.reply(0xB8, bytes.fromhex("B921252A02"))
    link.reply(0xB6, b"\xb7\x55")
    link.reply(0xBA, b"\xbb\x55")
    await backend.set_node_settings("dev", ("htr", "6"), mode="auto")
    await backend.set_node_lock("dev", ("htr", "6"), lock=True)
    assert link.payloads() == [
        b"\xb8",
        bytes.fromhex("B621252A01"),
        b"\xba\x01",
    ]
