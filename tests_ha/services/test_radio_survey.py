"""The radio_survey service and its report helpers."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
import pytest
import voluptuous as vol

from custom_components.termoweb import radio_survey
from custom_components.termoweb.backend.radio.link import RadioLinkError
from custom_components.termoweb.backend.radio.survey import RawBurst, analyse
from custom_components.termoweb.backend.radio_client import RadioClient
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.runtime import EntryRuntime
from custom_components.termoweb.services import radio_survey as service
from tests_ha.fakes.radio_link import FakeRadioLink, gateway_info
from tests_ha.fakes.radio_setup import add_radio_entry
from tests_ha.fakes.runtime import build_entry_runtime

NET = bytes.fromhex("1234")  # synthetic network id
ENTRY_ID = "radio-entry"


def _survey_entry(
    hass: HomeAssistant, version: str = "3.7-esp32", *, reachable: bool = True
) -> tuple[EntryRuntime, list[FakeRadioLink]]:
    """Add an ESP32 radio entry whose gateway runs firmware ``version``."""
    links: list[FakeRadioLink] = []

    def factory(host: str, port: int, dialect: Any, **kwargs: Any) -> Any:
        link = FakeRadioLink(host, port, dialect, **kwargs)
        link.info = gateway_info(version=version)
        link.survey_bursts = [RawBurst(-80.0, ((1, 104), (0, 104)))]
        if not reachable:
            link.connect_errors.append(RadioLinkError("unreachable"))
        links.append(link)
        return link

    client = RadioClient(
        "10.0.0.5", 2323, "B", [], network_id=NET, link_factory=factory
    )
    runtime = add_radio_entry(
        hass,
        client=client,
        entry_id=ENTRY_ID,
        data={"brand": "radio", "dialect": "B"},  # no radio_type: ESP32
    )
    runtime.version = "1.2.3"
    return runtime, links


async def _survey(hass: HomeAssistant, **data: Any) -> dict[str, Any]:
    """Call radio_survey through Home Assistant and return its response."""
    await service.async_register_radio_survey_service(hass)
    return await hass.services.async_call(
        DOMAIN, service.SERVICE_RADIO_SURVEY, data, blocking=True, return_response=True
    )


async def test_registers_once_with_optional_response(hass: HomeAssistant) -> None:
    """Registration is idempotent and the response is optional."""
    await service.async_register_radio_survey_service(hass)
    await service.async_register_radio_survey_service(hass)
    assert (
        hass.services.supports_response(DOMAIN, service.SERVICE_RADIO_SURVEY)
        is SupportsResponse.OPTIONAL
    )


@pytest.mark.parametrize("seconds", [9, 601])
async def test_schema_rejects_bad_durations(hass: HomeAssistant, seconds: int) -> None:
    """Survey length is bounded (10 s to 10 min)."""
    with pytest.raises(vol.Invalid):
        await _survey(hass, entry_id=ENTRY_ID, seconds=seconds)


async def test_survey_saves_redacted_report_and_returns_summary(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """The report is saved redacted next to the config; the summary is returned."""
    hass.config.config_dir = str(tmp_path)
    runtime, links = _survey_entry(hass)

    result = await _survey(hass, entry_id=ENTRY_ID, seconds="30")

    assert links[0].surveys == [30]
    assert result["verdict"] == "silent"  # two runs: no preamble
    assert result["bursts"] == 1
    assert runtime.last_radio_survey == result
    assert result["file"].startswith(
        str(tmp_path / f"termoweb_radio_survey_{ENTRY_ID}_")
    )
    saved = json.loads(Path(result["file"]).read_text())
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


async def test_survey_validates_the_entry(hass: HomeAssistant) -> None:
    """Unknown and cloud entries are user errors."""
    with pytest.raises(ServiceValidationError, match="No loaded TermoWeb entry"):
        await _survey(hass, entry_id="missing")
    build_entry_runtime(hass=hass, entry_id="cloud", client=object())
    with pytest.raises(ServiceValidationError, match="does not use a radio"):
        await _survey(hass, entry_id="cloud")


async def test_survey_needs_survey_firmware(hass: HomeAssistant) -> None:
    """Old gateway firmware has no raw survey; nothing is stored."""
    runtime, _links = _survey_entry(hass, version="3.6-esp32")
    with pytest.raises(HomeAssistantError, match="has no raw survey"):
        await _survey(hass, entry_id=ENTRY_ID)
    assert runtime.last_radio_survey is None


async def test_survey_link_failure_is_reported(hass: HomeAssistant) -> None:
    """A gateway that cannot be reached fails the call."""
    _survey_entry(hass, reachable=False)
    with pytest.raises(HomeAssistantError, match="Radio survey failed"):
        await _survey(hass, entry_id=ENTRY_ID)


async def test_save_report_failure_returns_none(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """An unwritable config directory yields no file instead of an error."""
    hass.config.config_dir = str(tmp_path / "missing")
    assert await radio_survey.async_save_report(hass, "x", {}) is None


def test_payload_and_summary_without_gateway() -> None:
    """Missing gateway details are reported as None; network ids are masked."""
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
