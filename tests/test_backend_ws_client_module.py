"""Smoke tests for the websocket client facade module."""

from __future__ import annotations

import importlib

import pytest

from custom_components.termoweb.backend import ducaheat_ws, termoweb_ws


def test_ws_client_exports_real_classes() -> None:
    """Ensure ``ws_client`` re-exports the real backend client classes."""

    module = importlib.import_module("custom_components.termoweb.backend.ws_client")
    assert getattr(module, "DucaheatWSClient") is ducaheat_ws.DucaheatWSClient
    assert getattr(module, "TermoWebWSClient") is termoweb_ws.TermoWebWSClient


def test_ws_client_missing_attribute_raises() -> None:
    """Verify unknown attributes raise ``AttributeError``."""

    module = importlib.import_module("custom_components.termoweb.backend.ws_client")
    with pytest.raises(AttributeError):
        getattr(module, "BogusClient")
