# ruff: noqa: D103,INP001,E402
"""Tests for the radio branch of the config flow."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any

import pytest
from conftest import _install_stubs

_install_stubs()

from custom_components.termoweb import config_flow, radio_survey
from custom_components.termoweb.backend.radio import (
    DIALECT_A,
    DIALECT_B,
    GatewayInfo,
    RadioLinkError,
)
from custom_components.termoweb.backend.radio.discovery import NetworkSighting
from custom_components.termoweb.backend.radio.survey import analyse
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

NET = bytes.fromhex("1234")  # synthetic network id
DEV_ID = "0a0b0c0d0e0f"
METHOD_MENU = {
    "type": "menu",
    "step_id": "radio_method",
    "menu_options": ["radio_discover", "radio_pair", "radio_monitor"],
}
FORM = {
    "host": "10.0.0.5",
    "port": 2323,
    "dialect": "auto",
    "network_id": "",
}


def _flow(hass: HomeAssistant, **context: Any) -> config_flow.TermoWebConfigFlow:
    flow = config_flow.TermoWebConfigFlow()
    flow.hass = hass
    flow.context = dict(context)
    return flow


def _radio_entry(hass: HomeAssistant) -> ConfigEntry:
    entry = ConfigEntry(
        "radio-entry",
        data={
            "brand": "radio",
            "host": "10.0.0.5",
            "port": 2323,
            "dialect": "B",
            "network_id": "1234",
            "nodes": [{"type": "htr", "addr": "6", "name": "Heater 6"}],
        },
    )
    hass.config_entries.add_entry(entry)
    return entry


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the gateway probe and discovery; record calls."""
    state: dict[str, Any] = {"probe": DEV_ID, "discover": None, "calls": []}

    async def fake_probe(host: str, port: int) -> str:
        state["calls"].append(("probe", host, port))
        result = state["probe"]
        if isinstance(result, Exception):
            raise result
        return result

    async def fake_discover(host, port, dialect, network_id, **kwargs):
        state["calls"].append(("discover", host, port, dialect, network_id))
        state["discover_kwargs"] = kwargs
        result = state["discover"]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(config_flow, "probe_gateway", fake_probe)
    monkeypatch.setattr(config_flow, "discover_radio", fake_discover)
    return state


async def _run_discovery(flow: config_flow.TermoWebConfigFlow, first: Any) -> Any:
    """Choose discovery, drive the progress step and return the finish result."""
    if first["type"] == "menu":
        assert first == METHOD_MENU
        first = await flow.async_step_radio_discover()
    assert first["type"] == "progress"
    assert first["progress_action"] == "radio_discover"
    await asyncio.wait([first["progress_task"]])
    done = await flow.async_step_radio_discover()
    assert done == {"type": "progress_done", "step_id": "radio_finish"}
    return await flow.async_step_radio_finish()


@pytest.mark.asyncio
async def test_user_step_offers_cloud_or_radio() -> None:
    result = await _flow(HomeAssistant()).async_step_user()
    assert result == {
        "type": "menu",
        "step_id": "user",
        "menu_options": ["cloud", "radio", "nanocul"],
    }


@pytest.mark.asyncio
async def test_radio_form_then_discovery_creates_entry(gateway) -> None:
    gateway["discover"] = (
        NetworkSighting(DIALECT_B, NET, frozenset({6})),
        {6: object(), 3: object()},
    )
    flow = _flow(HomeAssistant())
    form = await flow.async_step_radio()
    assert form["type"] == "form" and form["step_id"] == "radio"

    first = await flow.async_step_radio(dict(FORM))
    result = await _run_discovery(flow, first)

    assert flow._unique_id == f"radio:{DEV_ID}"
    assert result["type"] == "create_entry"
    assert result["title"] == "Radio gateway (10.0.0.5)"
    assert result["data"] == {
        "brand": "radio",
        "radio_type": "esp32",
        "host": "10.0.0.5",
        "port": 2323,
        "dialect": "B",
        "network_id": "1234",
        "nodes": [
            {"type": "htr", "addr": "3", "name": "Heater 3"},
            {"type": "htr", "addr": "6", "name": "Heater 6"},
        ],
        "supports_diagnostics": True,
    }
    assert gateway["calls"][1] == ("discover", "10.0.0.5", 2323, "auto", None)


@pytest.mark.asyncio
async def test_radio_form_passes_manual_network_id(gateway) -> None:
    gateway["discover"] = (NetworkSighting(DIALECT_B, NET), {6: object()})
    flow = _flow(HomeAssistant())
    first = await flow.async_step_radio({**FORM, "dialect": "B", "network_id": "12 34"})
    result = await _run_discovery(flow, first)
    assert result["type"] == "create_entry"
    assert gateway["calls"][1][3:] == ("B", NET)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("probe", "field", "error"),
    [
        (RadioLinkError("down"), "base", "cannot_connect_radio"),
        (config_flow.RadioSetupError("no_gateway_mac"), "base", "no_gateway_mac"),
    ],
)
async def test_radio_form_gateway_errors(gateway, probe, field, error) -> None:
    gateway["probe"] = probe
    result = await _flow(HomeAssistant()).async_step_radio(dict(FORM))
    assert result["type"] == "form"
    assert result["errors"] == {field: error}


@pytest.mark.asyncio
async def test_radio_form_rejects_bad_network_id(gateway) -> None:
    result = await _flow(HomeAssistant()).async_step_radio(
        {**FORM, "network_id": "XYZ"}
    )
    assert result["errors"] == {"network_id": "invalid_network_id"}
    assert gateway["calls"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised", "error"),
    [
        (config_flow.RadioSetupError("no_traffic"), "no_traffic"),
        (config_flow.RadioSetupError("no_heaters"), "no_heaters"),
        (RadioLinkError("lost"), "cannot_connect_radio"),
        (RuntimeError("boom"), "unknown"),
    ],
)
async def test_discovery_errors_return_to_form(gateway, raised, error) -> None:
    gateway["discover"] = raised
    flow = _flow(HomeAssistant())
    result = await _run_discovery(flow, await flow.async_step_radio(dict(FORM)))
    assert result["type"] == "form" and result["step_id"] == "radio"
    assert result["errors"] == {"base": error}


@pytest.mark.asyncio
async def test_progress_step_reports_running_task(gateway, monkeypatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(*_args, **_kwargs):
        started.set()
        await release.wait()
        return NetworkSighting(DIALECT_B, NET), {6: object()}

    monkeypatch.setattr(config_flow, "discover_radio", slow)
    flow = _flow(HomeAssistant())
    assert await flow.async_step_radio(dict(FORM)) == METHOD_MENU
    await flow.async_step_radio_discover()
    await started.wait()
    again = await flow.async_step_radio_discover()
    assert again["type"] == "progress"
    release.set()
    result = await _run_discovery(flow, again)
    assert result["type"] == "create_entry"


@pytest.mark.asyncio
async def test_reconfigure_radio_updates_address(gateway) -> None:
    hass = HomeAssistant()
    entry = _radio_entry(hass)
    flow = _flow(hass, entry_id=entry.entry_id, source="reconfigure")
    form = await flow.async_step_reconfigure()
    assert form["type"] == "form" and form["step_id"] == "reconfigure_radio"

    result = await flow.async_step_reconfigure_radio(
        {"host": "10.0.0.9", "port": 2424, "rescan": False}
    )
    assert result == {"type": "abort", "reason": "reconfigure_successful"}
    assert entry.data["host"] == "10.0.0.9" and entry.data["port"] == 2424
    assert entry.data["nodes"] == [{"type": "htr", "addr": "6", "name": "Heater 6"}]


@pytest.mark.asyncio
async def test_reconfigure_radio_rescan_replaces_nodes(gateway) -> None:
    gateway["discover"] = (NetworkSighting(DIALECT_B, NET), {6: object(), 7: object()})
    hass = HomeAssistant()
    entry = _radio_entry(hass)
    flow = _flow(hass, entry_id=entry.entry_id, source="reconfigure")
    first = await flow.async_step_reconfigure_radio(
        {"host": "10.0.0.5", "port": 2323, "rescan": True}
    )
    result = await _run_discovery(flow, first)
    assert result == {"type": "abort", "reason": "reconfigure_successful"}
    assert [n["addr"] for n in entry.data["nodes"]] == ["6", "7"]
    assert gateway["calls"][-1][3:] == ("B", NET)


@pytest.mark.asyncio
async def test_reconfigure_radio_errors(gateway) -> None:
    hass = HomeAssistant()
    entry = _radio_entry(hass)
    flow = _flow(hass, entry_id=entry.entry_id, source="reconfigure")
    gateway["probe"] = RadioLinkError("down")
    result = await flow.async_step_reconfigure_radio(
        {"host": "10.0.0.5", "port": 2323, "rescan": False}
    )
    assert result["errors"] == {"base": "cannot_connect_radio"}
    gateway["probe"] = config_flow.RadioSetupError("no_gateway_mac")
    result = await flow.async_step_reconfigure_radio(
        {"host": "10.0.0.5", "port": 2323, "rescan": False}
    )
    assert result["errors"] == {"base": "no_gateway_mac"}

    gateway["probe"] = DEV_ID
    gateway["discover"] = config_flow.RadioSetupError("no_heaters")
    first = await flow.async_step_reconfigure_radio(
        {"host": "10.0.0.5", "port": 2323, "rescan": True}
    )
    result = await _run_discovery(flow, first)
    assert result["step_id"] == "reconfigure_radio"
    assert result["errors"] == {"base": "no_heaters"}

    assert await _flow(hass).async_step_reconfigure_radio() == {
        "type": "abort",
        "reason": "no_config_entry",
    }


def test_parse_network_id() -> None:
    assert config_flow.parse_network_id("") is None
    assert config_flow.parse_network_id(None) is None
    assert config_flow.parse_network_id(" 12:34 ") == NET
    for bad in ("123", "12345", "GGGG"):
        with pytest.raises(ValueError):
            config_flow.parse_network_id(bad)


class FakeLink:
    """RadioLink stand-in that reports a configurable MAC."""

    mac: str | None = "0A:0B:0C:0D:0E:0F"
    created: list[dict[str, Any]] = []

    def __init__(self, host, port, dialect, **kwargs) -> None:
        FakeLink.created.append({"host": host, "port": port, **kwargs})

    async def connect(self) -> GatewayInfo:
        return GatewayInfo("3.6", "869.525", "2DE5", False, 1, FakeLink.mac, "")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_probe_gateway_listens_without_acking(monkeypatch) -> None:
    monkeypatch.setattr(config_flow, "RadioLink", FakeLink)
    monkeypatch.setattr(FakeLink, "created", [])
    assert await config_flow.probe_gateway("gw", 2323) == DEV_ID
    assert FakeLink.created[-1]["auto_ack"] is False

    monkeypatch.setattr(FakeLink, "mac", None)
    with pytest.raises(config_flow.RadioSetupError, match="no_gateway_mac"):
        await config_flow.probe_gateway("gw", 2323)


@pytest.mark.asyncio
async def test_discover_radio_paths(monkeypatch) -> None:
    calls: list[Any] = []
    sighting_b = NetworkSighting(DIALECT_B, NET, frozenset({40}))

    async def fake_discover(host, port, *, dialects, link_factory):
        calls.append(("listen", tuple(d.name for d in dialects)))
        return calls_result["sighting"]

    async def fake_probe(host, port, dialect, network_id, candidates, link_factory):
        calls.append(
            ("probe", dialect.name, network_id, 40 in candidates, 2 in candidates)
        )
        return calls_result["heaters"]

    calls_result: dict[str, Any] = {"sighting": sighting_b, "heaters": {6: "s"}}
    monkeypatch.setattr(config_flow, "discover_network", fake_discover)
    monkeypatch.setattr(config_flow, "probe_heaters", fake_probe)

    sighting, heaters = await config_flow.discover_radio("gw", 1, "auto", None)
    assert sighting is sighting_b and heaters == {6: "s"}
    assert calls == [("listen", ("B", "A")), ("probe", "B", NET, True, True)]

    calls.clear()
    await config_flow.discover_radio("gw", 1, "B", None)
    assert calls[0] == ("listen", ("B",))

    calls.clear()
    sighting, _ = await config_flow.discover_radio("gw", 1, "A", None)
    assert sighting.network_id == DIALECT_A.network_id
    assert calls == [("probe", "A", DIALECT_A.network_id, False, True)]

    calls.clear()
    await config_flow.discover_radio("gw", 1, "B", NET)
    assert calls == [("probe", "B", NET, False, True)]

    calls_result["sighting"] = None
    surveyed: list[Any] = []

    async def no_survey(host, port, **kwargs):
        surveyed.append(kwargs["analyse_survey"])
        return calls_result.get("surveyed")

    monkeypatch.setattr(config_flow, "survey_sighting", no_survey)
    with pytest.raises(config_flow.RadioSetupError, match="no_traffic"):
        await config_flow.discover_radio("gw", 1, "auto", None)
    assert surveyed == [config_flow._analyse_inline]

    calls.clear()
    calls_result["surveyed"] = NetworkSighting(DIALECT_B, NET)
    sighting, _ = await config_flow.discover_radio("gw", 1, "auto", None)
    assert sighting.network_id == NET
    assert calls[-1] == ("probe", "B", NET, False, True)
    calls_result["surveyed"] = None
    calls_result.update(sighting=sighting_b, heaters={})
    with pytest.raises(config_flow.RadioSetupError, match="no_heaters"):
        await config_flow.discover_radio("gw", 1, "auto", None)


@pytest.mark.asyncio
async def test_radio_options_store_heater_rated_power() -> None:
    hass = HomeAssistant()
    entry = _radio_entry(hass)
    entry.options = {
        "radio_power": {"power_limit": 2000, "rated_power": {"6": 1200}},
        "energy_history_progress": {"htr:6": 1_700_000_000},
        "energy_history_imported": True,
    }
    flow = config_flow.TermoWebConfigFlow.async_get_options_flow(entry)
    flow.hass = hass

    menu = await flow.async_step_init()
    assert menu["menu_options"] == ["settings", "pair_heaters", "rehome"]
    form = await flow.async_step_settings()
    assert form["step_id"] == "settings"
    fields = {str(getattr(k, "schema", k)): k for k in form["data_schema"].schema}
    default = fields["rated_power_6"].default
    assert (default() if callable(default) else default) == 1200
    assert "rated_power_6 = heater 6" in form["description_placeholders"]["heaters"]

    result = await flow.async_step_settings({"rated_power_6": 1500})
    assert result["data"] == {
        "radio_power": {"power_limit": 2000, "rated_power": {"6": 1500}},
        "energy_history_progress": {"htr:6": 1_700_000_000},
        "energy_history_imported": True,
    }


@pytest.mark.asyncio
async def test_radio_options_without_heaters_keep_existing_options() -> None:
    hass = HomeAssistant()
    entry = _radio_entry(hass)
    entry.data = {**entry.data, "nodes": []}
    entry.options = {"energy_history_imported": True}
    flow = config_flow.TermoWebOptionsFlow(entry)
    flow.hass = hass

    result = await flow.async_step_settings({})
    assert result["data"] == {"energy_history_imported": True}


def _report(verdict: str, dialect: str | None = None, nets: tuple[str, ...] = ()):
    """Return a survey report with the given outcome and no bursts."""
    return dataclasses.replace(
        analyse([]), verdict=verdict, dialect=dialect, network_ids=nets
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bursts", "report", "expected"),
    [
        (None, None, None),
        (["b"], _report("silent"), None),
        (["b"], _report("known", "B", ("1234", "5678")), (DIALECT_B, NET)),
        (["b"], _report("known", "A"), (DIALECT_A, DIALECT_A.network_id)),
        (["b"], _report("known", "B"), None),
        (["b"], _report("known", "Z"), "raise"),
        (["b"], _report("undecodable"), "raise"),
        (["b"], _report("candidate"), "raise"),
    ],
)
async def test_survey_sighting_outcomes(monkeypatch, bursts, report, expected) -> None:
    seen: list[Any] = []

    async def fake_survey(host, port, *, link_factory):
        seen.append((host, port, link_factory))
        return bursts

    async def fake_analyse(raw):
        seen.append(raw)
        return report

    monkeypatch.setattr(config_flow, "survey_network", fake_survey)
    if expected == "raise":
        with pytest.raises(config_flow.UnknownDialectError) as err:
            await config_flow.survey_sighting("gw", 1, analyse_survey=fake_analyse)
        assert err.value.reason == "unknown_dialect" and err.value.report is report
        return
    sighting = await config_flow.survey_sighting("gw", 1, analyse_survey=fake_analyse)
    if expected is None:
        assert sighting is None
    else:
        assert (sighting.dialect, sighting.network_id) == expected
    assert seen[0] == ("gw", 1, config_flow.RadioLink)
    assert seen[1:] == ([] if bursts is None else [bursts])


@pytest.mark.asyncio
async def test_analyse_inline_runs_the_analyser() -> None:
    assert (await config_flow._analyse_inline([])).verdict == "silent"


@pytest.mark.asyncio
@pytest.mark.parametrize("writable", [True, False])
async def test_unknown_dialect_saves_report_and_links_it(
    gateway, monkeypatch, tmp_path, writable
) -> None:
    gateway["discover"] = config_flow.UnknownDialectError(_report("undecodable"))
    monkeypatch.setattr(config_flow, "_get_version", _fake_version)
    hass = HomeAssistant()
    folder = tmp_path if writable else tmp_path / "missing"
    hass.config.path = lambda name: str(folder / name)
    flow = _flow(hass)
    result = await _run_discovery(flow, await flow.async_step_radio(dict(FORM)))
    assert result["errors"] == {"base": "unknown_dialect"}
    placeholders = result["description_placeholders"]
    assert placeholders["issue_url"] == radio_survey.ISSUE_URL
    analyse_survey = gateway["discover_kwargs"]["analyse_survey"]
    assert (await analyse_survey([])).verdict == "silent"
    if not writable:
        assert "could not be saved" in placeholders["report"]
        return
    files = list(tmp_path.glob("termoweb_radio_survey_setup_*.json"))
    assert [str(f) for f in files] == [placeholders["report"]]
    saved = json.loads(files[0].read_text())
    assert saved["integration_version"] == "9.9.9"
    assert saved["radio_type"] == "esp32" and saved["configured_dialect"] == "auto"
    assert saved["report"]["verdict"] == "undecodable"
    assert saved["report"]["redacted"] is True


async def _fake_version(_hass: Any) -> str:
    return "9.9.9"
