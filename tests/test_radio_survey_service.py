# ruff: noqa: D103,INP001,E402
"""Tests for the radio_survey service and its report helpers."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import _install_stubs, build_entry_runtime

_install_stubs()

from fake_radio_link import FakeRadioLink, gateway_info

from custom_components.termoweb import radio_survey
from custom_components.termoweb.backend.radio.link import RadioLinkError
from custom_components.termoweb.backend.radio.survey import RawBurst, analyse
from custom_components.termoweb.backend.radio_client import RadioClient
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.services import radio_survey as service
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError

NET = bytes.fromhex("1234")  # synthetic network id
ENTRY_ID = "radio-entry"


def _hass(tmp_path) -> HomeAssistant:
    hass = HomeAssistant()
    hass.config.path = lambda name: str(tmp_path / name)
    return hass


def _radio_runtime(hass: HomeAssistant, version: str = "3.7-esp32") -> tuple[Any, list]:
    links: list[FakeRadioLink] = []

    def factory(host, port, dialect, **kwargs):
        link = FakeRadioLink(host, port, dialect, **kwargs)
        link.info = gateway_info(version=version)
        link.survey_bursts = [RawBurst(-80.0, ((1, 104), (0, 104)))]
        links.append(link)
        return link

    client = RadioClient(
        "10.0.0.5", 2323, "B", [], network_id=NET, link_factory=factory
    )
    entry = SimpleNamespace(
        entry_id=ENTRY_ID,
        data={"dialect": "B"},  # no radio_type: ESP32
    )
    runtime = build_entry_runtime(
        hass=hass, entry_id=ENTRY_ID, client=client, config_entry=entry, brand="radio"
    )
    runtime.version = "1.2.3"
    return runtime, links


async def _handler(hass: HomeAssistant):
    await service.async_register_radio_survey_service(hass)
    return hass.services.get(DOMAIN, service.SERVICE_RADIO_SURVEY)


@pytest.mark.asyncio
async def test_registers_once_with_schema_and_optional_response(tmp_path) -> None:
    hass = _hass(tmp_path)
    handler = await _handler(hass)
    await service.async_register_radio_survey_service(hass)
    assert hass.services.get(DOMAIN, service.SERVICE_RADIO_SURVEY) is handler
    key = (DOMAIN, service.SERVICE_RADIO_SURVEY)
    assert hass.services.supports_response[key] is SupportsResponse.OPTIONAL
    schema = hass.services.schemas[key]
    assert schema({"entry_id": "x"}) == {"entry_id": "x", "seconds": 120}
    assert schema({"entry_id": "x", "seconds": "600"})["seconds"] == 600
    for bad in ({"entry_id": "x", "seconds": 9}, {"entry_id": "x", "seconds": 601}, {}):
        with pytest.raises((ValueError, KeyError)):  # vol.Invalid in real HA
            schema(bad)


@pytest.mark.asyncio
async def test_survey_saves_redacted_report_and_returns_summary(tmp_path) -> None:
    hass = _hass(tmp_path)
    runtime, links = _radio_runtime(hass)
    handler = await _handler(hass)

    result = await handler(ServiceCall({"entry_id": ENTRY_ID, "seconds": 30}))

    assert links[0].surveys == [30]
    assert result["verdict"] == "silent"  # two runs: no preamble
    assert result["bursts"] == 1
    assert runtime.last_radio_survey == result
    saved = json.loads((tmp_path / result["file"]).read_text())
    assert result["file"].startswith(
        str(tmp_path / f"termoweb_radio_survey_{ENTRY_ID}_")
    )
    assert saved["format"] == radio_survey.REPORT_FORMAT
    assert saved["integration_version"] == "1.2.3"
    assert saved["radio_type"] == "esp32" and saved["configured_dialect"] == "B"
    assert saved["gateway"] == {
        "firmware": "3.7-esp32",
        "freq": "869.525",
        "sync": "2DD4",
    }
    assert saved["survey_seconds"] == 30
    assert saved["report"]["redacted"] is True
    assert "AA:BB" not in json.dumps(saved)  # the gateway MAC never leaves HA


@pytest.mark.asyncio
async def test_survey_errors(tmp_path) -> None:
    hass = _hass(tmp_path)
    handler = await _handler(hass)
    with pytest.raises(ServiceValidationError, match="No loaded TermoWeb entry"):
        await handler(ServiceCall({"entry_id": "missing"}))

    build_entry_runtime(hass=hass, entry_id="cloud", client=object())
    with pytest.raises(ServiceValidationError, match="does not use a radio"):
        await handler(ServiceCall({"entry_id": "cloud"}))

    old, _ = _radio_runtime(hass, version="3.6-esp32")
    with pytest.raises(HomeAssistantError, match="has no raw survey"):
        await handler(ServiceCall({"entry_id": ENTRY_ID}))
    assert old.last_radio_survey is None

    old.client._link_factory = _failing_factory  # noqa: SLF001
    old.client._link = None  # noqa: SLF001
    with pytest.raises(HomeAssistantError, match="Radio survey failed"):
        await handler(ServiceCall({"entry_id": ENTRY_ID}))


def _failing_factory(host, port, dialect, **kwargs):
    link = FakeRadioLink(host, port, dialect, **kwargs)
    link.connect_errors.append(RadioLinkError("unreachable"))
    return link


@pytest.mark.asyncio
async def test_save_report_failure_returns_none(tmp_path) -> None:
    hass = HomeAssistant()
    hass.config.path = lambda name: str(tmp_path / "missing" / name)
    assert await radio_survey.async_save_report(hass, "x", {}) is None


def test_payload_and_summary_without_gateway() -> None:
    report = dataclasses.replace(
        analyse([]), verdict="candidate", network_ids=("1234",), confidence=0.666
    )
    payload = radio_survey.report_payload(
        report, seconds=5, gateway=None, radio_type=None, dialect=None, version=None
    )
    assert payload["gateway"] == {"firmware": None, "freq": None, "sync": None}
    assert payload["report"]["network_ids"] == ["XXXX"]
    summary = radio_survey.report_summary(report, None)
    assert summary["confidence"] == 0.67 and summary["file"] is None
    assert summary["verdict"] == "candidate" and summary["bursts"] == 0
