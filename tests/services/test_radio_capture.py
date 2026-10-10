"""The radio_capture service: file contents, summary and redaction."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import json
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
import pytest
import voluptuous as vol

from custom_components.termoweb.backend.radio import protocol as p
from custom_components.termoweb.backend.radio.dialect import DIALECT_A, decode
from custom_components.termoweb.backend.radio.link import RadioLinkError, ReceivedFrame
from custom_components.termoweb.backend.radio_client import RadioClient
from custom_components.termoweb.const import DOMAIN
from custom_components.termoweb.runtime import EntryRuntime
from custom_components.termoweb.services import radio_capture as service
from tests.fakes.radio_link import (
    HEATER,
    IDENTITY_SHORT,
    FakeRadioLink,
    gateway_info,
    received,
    received_ack,
)
from tests.fakes.radio_setup import add_radio_entry
from tests.fakes.runtime import build_entry_runtime

NET = bytes.fromhex("1234")  # synthetic network id
ENTRY_ID = "radio-entry"
MAC = "AA:BB:CC:00:11:22"
Q_LINE = f"# Q termoweb_rx 3.7-esp32 freq=869.525 id=FE mac={MAC} net=1234"
STOCK = dataclasses.replace(gateway_info(version="3.6"), raw=Q_LINE)
ESP32 = dataclasses.replace(STOCK, dialect="B")


@pytest.fixture
def config_dir(hass: HomeAssistant, tmp_path: Path) -> Path:
    """Point the Home Assistant config directory at ``tmp_path``."""
    hass.config.config_dir = str(tmp_path)
    return tmp_path


class Capture:
    """A radio entry whose capture window plays identify traffic."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        listen_only: bool,
        data: dict[str, Any],
        info: Any = STOCK,
        traffic: bool = True,
        link_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Build the client over fake links and add the entry."""
        self.links: list[FakeRadioLink] = []
        self.windows: list[float] = []
        self.listen_only = listen_only
        self.info = info
        self.traffic = traffic
        client = RadioClient(
            "10.0.0.5",
            2323,
            "A" if listen_only else "B",
            [],
            network_id=b"\x00\x00" if listen_only else NET,
            station_id=0xFE if listen_only else 1,
            link_factory=link_factory or self._factory,
            listen_only=listen_only,
        )
        real_capture = client.async_capture

        async def capture(seconds: float, **_kwargs: Any) -> Any:
            return await real_capture(seconds, sleep=self._window)

        client.async_capture = capture  # type: ignore[method-assign]
        self.runtime: EntryRuntime = add_radio_entry(
            hass,
            client=client,
            entry_id=ENTRY_ID,
            data=data,
            brand="radio_monitor" if listen_only else "radio",
        )
        self.runtime.version = "1.2.3"

    def _factory(self, host: str, port: int, dialect: Any, **kwargs: Any) -> Any:
        link = FakeRadioLink(host, port, dialect, **kwargs)
        link.info = self.info
        self.links.append(link)
        return link

    async def _window(self, waited: float) -> None:
        """Deliver identify traffic, a junk frame and a status line."""
        self.windows.append(waited)
        if not self.traffic:
            return
        link = self.links[0]
        dialect = DIALECT_A if self.listen_only else link.dialect
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


async def _capture(hass: HomeAssistant, **data: Any) -> dict[str, Any]:
    """Call radio_capture through Home Assistant and return its response."""
    await service.async_register_radio_capture_service(hass)
    return await hass.services.async_call(
        DOMAIN, service.SERVICE_RADIO_CAPTURE, data, blocking=True, return_response=True
    )


async def test_registers_once_with_optional_response(hass: HomeAssistant) -> None:
    """Registration is idempotent and the response is optional."""
    await service.async_register_radio_capture_service(hass)
    await service.async_register_radio_capture_service(hass)
    assert (
        hass.services.supports_response(DOMAIN, service.SERVICE_RADIO_CAPTURE)
        is SupportsResponse.OPTIONAL
    )


@pytest.mark.parametrize("seconds", [9, 1801, "soon"])
async def test_schema_rejects_bad_durations(hass: HomeAssistant, seconds: Any) -> None:
    """Capture length is bounded (10 s to 30 min)."""
    with pytest.raises(vol.Invalid):
        await _capture(hass, entry_id=ENTRY_ID, seconds=seconds)


async def test_capture_on_monitor_entry_saves_every_frame(
    hass: HomeAssistant, config_dir: Path
) -> None:
    """A listen-only capture saves every frame and line, MAC removed."""
    data = {"brand": "radio_monitor", "radio_type": "nanocul", "device": "/dev/x"}
    rig = Capture(hass, listen_only=True, data=data)

    result = await _capture(hass, entry_id=ENTRY_ID, seconds="60", note="kitchen")

    assert rig.windows == [60]
    assert rig.links[0].sent == []  # listening only
    assert set(result) == {"file", "frames", "networks", "nodes", "opcodes"}
    assert result["frames"] == 5
    assert result["networks"] == ["1234", "1B30"]  # the ack is synthetic dialect B
    assert result["nodes"] == [1, HEATER]
    assert result["opcodes"]["5E"] == {"count": 1, "name": "flash display (identify)"}
    assert result["opcodes"]["5F"] == {"count": 1, "name": "flash display reply"}
    assert result["opcodes"]["E7"] == {"count": 1, "name": None}
    assert rig.runtime.last_radio_capture == result
    name = result["file"].removeprefix(str(config_dir / "termoweb_radio_capture_"))
    assert len(name) == len("20260102T030405Z.json") and name.endswith("Z.json")

    saved = json.loads(Path(result["file"]).read_text())
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


async def test_redacted_capture_masks_networks_and_serial(
    hass: HomeAssistant, config_dir: Path
) -> None:
    """redact=true masks network ids, the heater serial and raw air bytes."""
    Capture(
        hass,
        listen_only=True,
        data={"brand": "radio_monitor", "host": "10.0.0.5", "port": 2323},
        info=ESP32,
    )

    result = await _capture(hass, entry_id=ENTRY_ID, redact=True)

    assert result["networks"] == ["NET1", "NET2"]
    text = Path(result["file"]).read_text()
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


async def test_capture_works_passively_on_a_normal_radio_entry(
    hass: HomeAssistant, config_dir: Path
) -> None:
    """A normal entry listens in its own dialect only and sends nothing."""
    rig = Capture(hass, listen_only=False, data={"brand": "radio"}, info=ESP32)

    result = await _capture(hass, entry_id=ENTRY_ID, seconds=10)

    assert result["networks"] == ["1234"]
    saved = json.loads(Path(result["file"]).read_text())
    assert saved["dialects_listened"] == ["B"]
    assert {r["dialect"] for r in saved["frames"]} == {"B"}
    assert rig.links[0].sent == []


async def test_capture_without_gateway_info(
    hass: HomeAssistant, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a gateway banner the file says so."""
    Capture(hass, listen_only=True, data={"brand": "radio_monitor"}, traffic=False)
    monkeypatch.setattr(RadioClient, "gateway_info", None)
    result = await _capture(hass, entry_id=ENTRY_ID)
    saved = json.loads(Path(result["file"]).read_text())
    assert saved["gateway"] is None and saved["dialects_listened"] == ["A"]


async def test_capture_validates_the_entry(hass: HomeAssistant) -> None:
    """Unknown and cloud entries are user errors."""
    with pytest.raises(ServiceValidationError, match="No loaded TermoWeb entry"):
        await _capture(hass, entry_id="missing")
    build_entry_runtime(hass=hass, entry_id="cloud", client=object())
    with pytest.raises(ServiceValidationError, match="does not use a radio"):
        await _capture(hass, entry_id="cloud")


async def test_capture_link_failure_is_reported(hass: HomeAssistant) -> None:
    """A gateway that cannot be reached fails the call and stores nothing."""

    def failing(host: str, port: int, dialect: Any, **kwargs: Any) -> Any:
        link = FakeRadioLink(host, port, dialect, **kwargs)
        link.connect_errors.append(RadioLinkError("unreachable"))
        return link

    rig = Capture(
        hass, listen_only=True, data={"brand": "radio_monitor"}, link_factory=failing
    )
    with pytest.raises(HomeAssistantError, match="Radio capture failed"):
        await _capture(hass, entry_id=ENTRY_ID)
    assert rig.runtime.last_radio_capture is None


async def test_unwritable_config_dir_still_returns_the_summary(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """A file that cannot be written does not lose the summary."""
    hass.config.config_dir = str(tmp_path / "missing")
    Capture(hass, listen_only=True, data={"brand": "radio_monitor"}, traffic=False)
    result = await _capture(hass, entry_id=ENTRY_ID)
    assert result["file"] is None and result["frames"] == 0
