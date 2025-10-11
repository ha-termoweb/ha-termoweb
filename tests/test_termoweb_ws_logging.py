"""Tests for the legacy websocket logging helper."""

from __future__ import annotations

from unittest.mock import patch

import custom_components.termoweb.backend.termoweb_ws as termoweb_ws
from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient


def _call_log_legacy_update(**kwargs: object) -> None:
    """Invoke ``_log_legacy_update`` without constructing a client."""

    TermoWebWSClient._log_legacy_update(object(), **kwargs)


def test_log_legacy_update_lists_unique_addresses() -> None:
    """Ensure address updates are combined and logged at debug level."""

    expected_pairs = sorted(["acm/2", "htr/1", "thm/3"])
    with (
        patch.object(termoweb_ws._LOGGER, "isEnabledFor", return_value=True)
        as mock_enabled,
        patch.object(termoweb_ws._LOGGER, "debug") as mock_debug,
    ):
        _call_log_legacy_update(
            updated_nodes=False,
            updated_addrs=[("htr", "1"), ("acm", "2")],
            sample_addrs=[("thm", "3")],
        )
    mock_enabled.assert_called_once()
    mock_debug.assert_called_once_with(
        "WS: legacy update for %s", ", ".join(expected_pairs)
    )


def test_log_legacy_update_notes_node_refresh() -> None:
    """Ensure the legacy nodes refresh message is logged when required."""

    with (
        patch.object(termoweb_ws._LOGGER, "isEnabledFor", return_value=True)
        as mock_enabled,
        patch.object(termoweb_ws._LOGGER, "debug") as mock_debug,
    ):
        _call_log_legacy_update(
            updated_nodes=True,
            updated_addrs=[],
            sample_addrs=[],
        )
    mock_enabled.assert_called_once()
    mock_debug.assert_called_once_with("WS: legacy nodes refresh")


def test_log_legacy_update_skips_logging_when_debug_disabled() -> None:
    """Confirm no logging occurs if debug logging is disabled."""

    with (
        patch.object(termoweb_ws._LOGGER, "isEnabledFor", return_value=False)
        as mock_enabled,
        patch.object(termoweb_ws._LOGGER, "debug") as mock_debug,
    ):
        _call_log_legacy_update(
            updated_nodes=True,
            updated_addrs=[("htr", "1")],
            sample_addrs=[("thm", "3")],
        )
    mock_enabled.assert_called_once()
    mock_debug.assert_not_called()
