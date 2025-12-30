"""Tests for Ducaheat REST client settings payload handling."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient


def test_normalise_settings_omits_raw_by_default() -> None:
    """Normalised heater settings should exclude raw payload copies."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
    payload = {
        "status": {"mode": "Auto", "temp": 21},
        "setup": {"raw_only": True},
    }

    result = client._normalise_settings(payload)

    assert "raw" not in result
    assert result["mode"] == "auto"
    assert result["mtemp"] == "21.0"


@pytest.mark.asyncio
async def test_get_node_settings_includes_raw_when_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Raw payloads should be retained when debug logging is enabled."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
    payload = {"status": {"mode": "Manual"}}

    with (
        patch.object(client, "authed_headers", AsyncMock(return_value={})),
        patch.object(client, "_request", AsyncMock(return_value=payload)),
        caplog.at_level(
            logging.DEBUG, logger="custom_components.termoweb.backend.ducaheat"
        ),
    ):
        result = await client.get_node_settings("dev", ("htr", "01"))

    assert result["mode"] == "manual"
    assert result["raw"] == payload
