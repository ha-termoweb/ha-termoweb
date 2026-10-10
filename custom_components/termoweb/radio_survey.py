"""Analyse raw radio surveys and save reports test users can submit.

A survey (``R<seconds>`` on the ESP32 gateway) captures every RF burst on the
heater band, whatever its dialect. The analysis recognises the known dialects
and otherwise proposes the framing a new dialect probably uses. The saved
report is redacted (network ids masked, identity payloads cut) so a user can
attach it to a GitHub issue. See ``docs/radio_protocol.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import logging
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .backend.radio.link import GatewayInfo
from .backend.radio.survey import RawBurst, SurveyReport, analyse, redact

_LOGGER = logging.getLogger(__name__)

REPORT_FORMAT = 1
REPORT_PREFIX = "termoweb_radio_survey"
ISSUE_URL = (
    "https://github.com/ha-termoweb/ha-termoweb/issues/new"
    "?title=New+radio+dialect&labels=radio-dialect"
)


async def async_analyse(hass: HomeAssistant, bursts: Sequence[RawBurst]) -> SurveyReport:
    """Analyse survey bursts in the executor: the framing search is CPU work."""

    return await hass.async_add_executor_job(analyse, list(bursts))


def report_payload(
    report: SurveyReport,
    *,
    seconds: int,
    gateway: GatewayInfo | None,
    radio_type: str | None,
    dialect: str | None,
    version: str | None,
) -> dict[str, Any]:
    """Return the redacted, JSON-able report file content (no gateway MAC)."""

    return {
        "format": REPORT_FORMAT,
        "created": dt_util.utcnow().isoformat(timespec="seconds"),
        "integration_version": version,
        "radio_type": radio_type,
        "configured_dialect": dialect,
        "gateway": {
            "firmware": None if gateway is None else gateway.version,
            "freq": None if gateway is None else gateway.freq,
            "sync": None if gateway is None else gateway.sync,
        },
        "survey_seconds": seconds,
        "report": redact(report).as_dict(),
    }


def report_summary(report: SurveyReport, path: str | None) -> dict[str, Any]:
    """Return the short outcome shown in service responses and diagnostics."""

    return {
        "created": dt_util.utcnow().isoformat(timespec="seconds"),
        "verdict": report.verdict,
        "dialect": report.dialect,
        "bursts": len(report.bursts),
        "confidence": round(report.confidence, 2),
        "suggestion": report.suggestion,
        "file": path,
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    """Write ``payload`` as indented JSON to ``path``."""

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)


async def async_save_report(
    hass: HomeAssistant, label: str, payload: dict[str, Any]
) -> str | None:
    """Save the report in the config directory; return its path, None on failure."""

    stamp = dt_util.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = hass.config.path(f"{REPORT_PREFIX}_{label}_{stamp}.json")
    try:
        await hass.async_add_executor_job(_write_json, path, payload)
    except OSError:
        _LOGGER.exception("Cannot write radio survey report %s", path)
        return None
    _LOGGER.info("Radio survey report saved to %s", path)
    return path


__all__ = [
    "ISSUE_URL",
    "async_analyse",
    "async_save_report",
    "report_payload",
    "report_summary",
]
