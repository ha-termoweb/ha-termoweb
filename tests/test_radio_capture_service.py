# ruff: noqa: D103,INP001,E402
"""Tests for the radio_capture service: file contents, summary and redaction."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import _install_stubs, build_entry_runtime

_install_stubs()

from tests_ha.fakes.radio_link import (
    HEATER,
    IDENTITY_SHORT,
    FakeRadioLink,
    gateway_info,
    received,
    received_ack,
)

from custom_components.termoweb.backend.radio import protocol as p
from custom_components.termoweb.backend.radio.dialect import DIALECT_A, decode
from custom_components.termoweb.backend.radio.link import RadioLinkError, ReceivedFrame
from custom_components.termoweb.backend.radio_client import RadioClient
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.services import radio_capture as service
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError

NET = bytes.fromhex("1234")  # synthetic network id
ENTRY_ID = "radio-entry"
MAC = "AA:BB:CC:00:11:22"
Q_LINE = f"# Q termoweb_rx 3.7-esp32 freq=869.525 id=FE mac={MAC} net=1234"
STOCK = dataclasses.replace(gateway_info(version="3.6"), raw=Q_LINE)
ESP32 = dataclasses.replace(STOCK, dialect="B")


def _hass(tmp_path) -> HomeAssistant:
    hass = HomeAssistant()
    hass.config.path = lambda name: str(tmp_path / name)
    return hass


def _runtime(
    hass: HomeAssistant,
    *,
    listen_only: bool,
    data: dict[str, Any],
    info=STOCK,
    traffic=True,
) -> tuple[Any, list[FakeRadioLink], list[float]]:
    """Return a runtime whose capture window plays identify traffic."""
    links: list[FakeRadioLink] = []
    windows: list[float] = []

    def factory(host, port, dialect, **kwargs):
        link = FakeRadioLink(host, port, dialect, **kwargs)
        link.info = info
        links.append(link)
        return link

    client = RadioClient(
        "10.0.0.5",
        2323,
        "A" if listen_only else "B",
        [],
        network_id=b"\x00\x00" if listen_only else NET,
        station_id=0xFE if listen_only else 1,
        link_factory=factory,
        listen_only=listen_only,
    )
    original = client.async_capture

    async def capture(seconds: float, **_kwargs: Any):
        async def window(waited: float) -> None:
            windows.append(waited)
            if not traffic:
                return
            link = links[0]
            dialect = DIALECT_A if listen_only else link.dialect
            for src, dst, payload in (
                (1, HEATER, p.flash_display()),
                (HEATER, 1, bytes.fromhex("5F55")),
                (HEATER, 1, IDENTITY_SHORT),
                (1, HEATER, bytes.fromhex("E701")),
            ):
                link.deliver(received(src, payload, dst=dst, dialect=dialect))
            link.deliver(received_ack(HEATER))
            link.deliver(ReceivedFrame(decode(DIALECT_A, b"\x01\x02"), -90.0, 0, 7))
            link.deliver_line(f"# status mac={MAC} net=1234")

        return await original(seconds, sleep=window)

    client.async_capture = capture  # type: ignore[method-assign]
    entry = SimpleNamespace(entry_id=ENTRY_ID, data=data)
    runtime = build_entry_runtime(
        hass=hass,
        entry_id=ENTRY_ID,
        client=client,
        config_entry=entry,
        brand="radio_monitor" if listen_only else "radio",
    )
    runtime.version = "1.2.3"
    return runtime, links, windows


async def _handler(hass: HomeAssistant):
    await service.async_register_radio_capture_service(hass)
    return hass.services.get(DOMAIN, service.SERVICE_RADIO_CAPTURE)


@pytest.mark.asyncio
async def test_registers_once_with_schema_and_optional_response(tmp_path) -> None:
    hass = _hass(tmp_path)
    handler = await _handler(hass)
    await service.async_register_radio_capture_service(hass)
    assert hass.services.get(DOMAIN, service.SERVICE_RADIO_CAPTURE) is handler
    key = (DOMAIN, service.SERVICE_RADIO_CAPTURE)
    assert hass.services.supports_response[key] is SupportsResponse.OPTIONAL
    schema = hass.services.schemas[key]
    assert schema({"entry_id": "x"}) == {
        "entry_id": "x",
        "seconds": 120,
        "redact": False,
    }
    full = schema({"entry_id": "x", "seconds": "1800", "redact": True, "note": "hi"})
    assert full["seconds"] == 1800 and full["note"] == "hi"
    for bad in (
        {"entry_id": "x", "seconds": 9},
        {"entry_id": "x", "seconds": 1801},
        {"entry_id": "x", "seconds": "soon"},
        {},
    ):
        with pytest.raises((ValueError, KeyError, TypeError)):  # vol.Invalid in HA
            schema(bad)


@pytest.mark.asyncio
async def test_capture_on_monitor_entry_saves_every_frame(tmp_path) -> None:
    hass = _hass(tmp_path)
    data = {"brand": "radio_monitor", "radio_type": "nanocul", "device": "/dev/x"}
    runtime, links, windows = _runtime(hass, listen_only=True, data=data)
    handler = await _handler(hass)

    result = await handler(
        ServiceCall({"entry_id": ENTRY_ID, "seconds": 60, "note": "kitchen"})
    )

    assert windows == [60]
    assert links[0].sent == []  # listening only
    assert set(result) == {"file", "frames", "networks", "nodes", "opcodes"}
    assert result["frames"] == 5
    assert result["networks"] == ["1234", "1B30"]  # the ack is synthetic dialect B
    assert result["nodes"] == [1, HEATER]
    assert result["opcodes"]["5E"] == {"count": 1, "name": "flash display (identify)"}
    assert result["opcodes"]["5F"] == {"count": 1, "name": "flash display reply"}
    assert result["opcodes"]["E7"] == {"count": 1, "name": None}
    assert runtime.last_radio_capture == result
    name = result["file"].removeprefix(str(tmp_path / "termoweb_radio_capture_"))
    assert len(name) == len("20260102T030405Z.json") and name.endswith("Z.json")

    saved = json.loads((tmp_path / result["file"]).read_text())
    assert list(saved) == [
        "version",
        "started",
        "ended",
        "gateway",
        "dialects_listened",
        "note",
        "redacted",
        "summary",
        "frames",
        "raw",
    ]
    assert saved["version"] == 1 and saved["note"] == "kitchen"
    assert saved["started"] <= saved["ended"] and saved["started"].endswith("+00:00")
    assert saved["gateway"] == Q_LINE.replace(MAC, "XX")
    assert saved["dialects_listened"] == ["A"] and saved["redacted"] is False
    assert saved["summary"] == {
        "frames": 5,
        "acks": 1,
        "networks": result["networks"],
        "nodes": result["nodes"],
        "opcodes": result["opcodes"],
    }
    first = saved["frames"][0]
    assert first["kind"] == "data" and first["dialect"] == "A"
    assert first["net"] == "1B30" and first["src"] == 1 and first["dst"] == HEATER
    assert first["payload"] == "5E01" and first["name"] == "flash display (identify)"
    assert first["t"].endswith("+00:00") and len(first["t"]) == 29  # milliseconds
    assert first["rssi"] == -60.0 and first["air"] == first["air"].upper()
    assert saved["frames"][-1]["kind"] == "ack"
    assert [r["line"] for r in saved["raw"]] == [
        "RX 7 -90.0 0 0 0102",
        "# status mac=XX net=1234",
    ]
    assert MAC not in json.dumps(saved)  # the gateway MAC never leaves HA


@pytest.mark.asyncio
async def test_redacted_capture_masks_networks_and_serial(tmp_path) -> None:
    hass = _hass(tmp_path)
    _runtime(
        hass,
        listen_only=True,
        data={"brand": "radio_monitor", "host": "10.0.0.5", "port": 2323},
        info=ESP32,
    )
    handler = await _handler(hass)

    result = await handler(ServiceCall({"entry_id": ENTRY_ID, "redact": True}))

    assert result["networks"] == ["NET1", "NET2"]
    text = (tmp_path / result["file"]).read_text()
    saved = json.loads(text)
    assert saved["redacted"] is True and saved["note"] is None
    assert saved["dialects_listened"] == ["A", "B"]
    assert saved["gateway"].endswith("mac=XX net=XXXX")
    for secret in ("1234", "1B30", b"X123456".hex().upper(), MAC, "mac=AA"):
        assert secret not in text
    payloads = [r["payload"] for r in saved["frames"]]
    assert "5E01" in payloads and "5F55" in payloads and "E701" in payloads
    assert "5B55" + "XX" * 16 in payloads
    assert all("air" not in r for r in saved["frames"])


@pytest.mark.asyncio
async def test_capture_works_passively_on_a_normal_radio_entry(tmp_path) -> None:
    hass = _hass(tmp_path)
    _runtime(hass, listen_only=False, data={"brand": "radio"}, info=ESP32)
    handler = await _handler(hass)

    result = await handler(ServiceCall({"entry_id": ENTRY_ID, "seconds": 10}))

    assert result["networks"] == ["1234"]
    saved = json.loads((tmp_path / result["file"]).read_text())
    assert saved["dialects_listened"] == ["B"]  # the entry's own dialect only
    assert {r["dialect"] for r in saved["frames"]} == {"B"}


@pytest.mark.asyncio
async def test_capture_without_gateway_info(tmp_path, monkeypatch) -> None:
    hass = _hass(tmp_path)
    runtime, _links, _ = _runtime(
        hass, listen_only=True, data={"brand": "radio_monitor"}, traffic=False
    )
    monkeypatch.setattr(type(runtime.client), "gateway_info", None)
    handler = await _handler(hass)
    result = await handler(ServiceCall({"entry_id": ENTRY_ID}))
    saved = json.loads((tmp_path / result["file"]).read_text())
    assert saved["gateway"] is None and saved["dialects_listened"] == ["A"]


@pytest.mark.asyncio
async def test_capture_errors(tmp_path) -> None:
    hass = _hass(tmp_path)
    handler = await _handler(hass)
    with pytest.raises(ServiceValidationError, match="No loaded TermoWeb entry"):
        await handler(ServiceCall({"entry_id": "missing"}))

    build_entry_runtime(hass=hass, entry_id="cloud", client=object())
    with pytest.raises(ServiceValidationError, match="does not use a radio"):
        await handler(ServiceCall({"entry_id": "cloud"}))

    runtime, _links, _ = _runtime(
        hass, listen_only=True, data={"brand": "radio_monitor"}
    )

    def failing(host, port, dialect, **kwargs):
        link = FakeRadioLink(host, port, dialect, **kwargs)
        link.connect_errors.append(RadioLinkError("unreachable"))
        return link

    runtime.client._link_factory = failing  # noqa: SLF001
    with pytest.raises(HomeAssistantError, match="Radio capture failed"):
        await handler(ServiceCall({"entry_id": ENTRY_ID}))
    assert runtime.last_radio_capture is None


@pytest.mark.asyncio
async def test_unwritable_config_dir_still_returns_the_summary(tmp_path) -> None:
    hass = HomeAssistant()
    hass.config.path = lambda name: str(tmp_path / "missing" / name)
    _runtime(hass, listen_only=True, data={"brand": "radio_monitor"}, traffic=False)
    handler = await _handler(hass)
    result = await handler(ServiceCall({"entry_id": ENTRY_ID}))
    assert result["file"] is None and result["frames"] == 0
