"""Tests for fallback translation string helpers."""

from __future__ import annotations

from custom_components.termoweb.fallback_translations import (
    get_fallback_translations,
    language_candidates,
)


# ---------------------------------------------------------------------------
# language_candidates
# ---------------------------------------------------------------------------


class TestLanguageCandidates:
    def test_empty_string_returns_en(self) -> None:
        assert language_candidates("") == ["en"]

    def test_none_returns_en(self) -> None:
        assert language_candidates(None) == ["en"]

    def test_simple_language(self) -> None:
        result = language_candidates("fr")
        assert result == ["fr", "en"]

    def test_language_with_region(self) -> None:
        result = language_candidates("pt-PT")
        assert "pt-pt" in result
        assert "pt" in result
        assert "en" in result

    def test_underscore_normalised_to_dash(self) -> None:
        result = language_candidates("pt_PT")
        assert "pt-pt" in result

    def test_en_not_duplicated(self) -> None:
        result = language_candidates("en")
        assert result.count("en") == 1

    def test_whitespace_stripped(self) -> None:
        result = language_candidates("  fr  ")
        assert "fr" in result


# ---------------------------------------------------------------------------
# get_fallback_translations
# ---------------------------------------------------------------------------


class TestGetFallbackTranslations:
    def test_known_language_returns_translations(self) -> None:
        result = get_fallback_translations("en")
        assert "heater_name" in result
        assert result["heater_name"] == "Heater {addr}"

    def test_unknown_language_falls_back_to_en(self) -> None:
        result = get_fallback_translations("unknown")
        assert result["heater_name"] == "Heater {addr}"

    def test_regional_variant_falls_back_to_base(self) -> None:
        result = get_fallback_translations("pt-pt")
        assert "heater_name" in result
        # pt-pt is directly in FALLBACK_TRANSLATIONS
        assert "Aquecedor" in result["heater_name"]

    def test_empty_language_returns_en(self) -> None:
        result = get_fallback_translations("")
        assert result["heater_name"] == "Heater {addr}"

    def test_en_fallback_when_no_candidates_match(self) -> None:
        """When no candidates exist in FALLBACK_TRANSLATIONS, return en."""
        # 'xx' is not in the dict, and 'en' is appended as fallback
        # but if we test with 'en' stripped somehow, we still get en
        result = get_fallback_translations("xx-yy")
        # xx-yy -> candidates: [xx-yy, xx, en]
        # None of xx-yy or xx are in dict, so en is used
        assert result["heater_name"] == "Heater {addr}"
