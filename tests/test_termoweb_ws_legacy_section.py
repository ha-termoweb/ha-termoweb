"""Tests for TermoWeb legacy websocket section mapping."""

from custom_components.termoweb.backend import termoweb_ws as module


def test_legacy_section_for_known_suffixes() -> None:
    """Ensure known path suffixes map to legacy section names."""

    assert module.TermoWebWSClient._legacy_section_for_path("/foo/settings") == "settings"
    assert module.TermoWebWSClient._legacy_section_for_path("/foo/advanced_setup") == "advanced"
    assert module.TermoWebWSClient._legacy_section_for_path("/foo/samples") == "samples"


def test_legacy_section_for_unknown_suffixes() -> None:
    """Ensure irrelevant paths do not produce a legacy section mapping."""

    assert module.TermoWebWSClient._legacy_section_for_path("/foo/unknown") is None
    assert module.TermoWebWSClient._legacy_section_for_path("/foo/bar") is None
    assert module.TermoWebWSClient._legacy_section_for_path("") is None
