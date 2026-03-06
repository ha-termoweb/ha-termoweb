"""Tests for the i18n translation helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.termoweb.i18n import (
    COORDINATOR_FALLBACK_ATTR,
    _tr,
    attach_fallbacks,
    async_get_fallback_translations,
    format_fallback,
)


# ---------------------------------------------------------------------------
# format_fallback
# ---------------------------------------------------------------------------


class TestFormatFallback:
    def test_uses_fallback_mapping_when_present(self) -> None:
        fallbacks = {"key1": "Hello {name}!"}
        result = format_fallback(fallbacks, "key1", "Default {name}", name="World")
        assert result == "Hello World!"

    def test_falls_back_to_default_template(self) -> None:
        result = format_fallback(None, "key1", "Default {name}", name="World")
        assert result == "Default World"

    def test_missing_key_uses_default(self) -> None:
        fallbacks = {"other_key": "Other"}
        result = format_fallback(fallbacks, "key1", "Default {name}", name="World")
        assert result == "Default World"

    def test_format_error_in_fallback_returns_template(self) -> None:
        fallbacks = {"key1": "Bad {missing}"}
        result = format_fallback(fallbacks, "key1", "Default", name="test")
        assert result == "Bad {missing}"

    def test_format_error_in_default_returns_default(self) -> None:
        result = format_fallback(None, "key1", "Bad {missing}", name="test")
        assert result == "Bad {missing}"

    def test_non_mapping_fallbacks_uses_default(self) -> None:
        result = format_fallback("not a mapping", "key1", "Default {name}", name="X")
        assert result == "Default X"


# ---------------------------------------------------------------------------
# attach_fallbacks
# ---------------------------------------------------------------------------


class TestAttachFallbacks:
    def test_attaches_to_regular_object(self) -> None:
        target = SimpleNamespace()
        fallbacks = {"key": "value"}
        attach_fallbacks(target, fallbacks)
        assert getattr(target, COORDINATOR_FALLBACK_ATTR) == fallbacks

    def test_non_mapping_skipped(self) -> None:
        target = SimpleNamespace()
        attach_fallbacks(target, None)
        assert not hasattr(target, COORDINATOR_FALLBACK_ATTR)

    def test_non_mapping_string_skipped(self) -> None:
        target = SimpleNamespace()
        attach_fallbacks(target, "not a mapping")
        assert not hasattr(target, COORDINATOR_FALLBACK_ATTR)

    def test_attach_to_object_with_dict(self) -> None:
        """Objects supporting __dict__ get fallbacks via dict when setattr fails."""

        class Restricted:
            __slots__ = ()

            def __init__(self) -> None:
                pass

        target = Restricted()
        fallbacks = {"key": "val"}
        # This should not raise even if setattr fails
        attach_fallbacks(target, fallbacks)


# ---------------------------------------------------------------------------
# async_get_fallback_translations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_get_fallback_translations_cached_in_runtime() -> None:
    """Cached translations in EntryRuntime are returned directly."""

    # Use a mock that acts like EntryRuntime
    from custom_components.termoweb.runtime import EntryRuntime

    hass = SimpleNamespace()
    hass.config = SimpleNamespace(language="en")

    runtime = EntryRuntime.__new__(EntryRuntime)
    runtime.fallback_translations = {"key": "cached_value"}

    result = await async_get_fallback_translations(hass, runtime)
    assert result == {"key": "cached_value"}


@pytest.mark.asyncio
async def test_async_get_fallback_translations_without_runtime() -> None:
    """Without runtime, translations are fetched fresh."""

    hass = SimpleNamespace()
    hass.config = SimpleNamespace(language="en")

    result = await async_get_fallback_translations(hass, None)
    assert isinstance(result, dict)
    assert "heater_name" in result


@pytest.mark.asyncio
async def test_async_get_fallback_translations_no_language() -> None:
    """Falls back to default_language or 'en' when language not set."""

    hass = SimpleNamespace()
    hass.config = SimpleNamespace(language=None, default_language="fr")

    result = await async_get_fallback_translations(hass, None)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _tr translation helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tr_returns_translated_string() -> None:
    """_tr returns the translated string for a known key."""

    hass = SimpleNamespace()
    hass.config = SimpleNamespace(language="en")

    with patch(
        "custom_components.termoweb.i18n.async_get_translations",
        new_callable=AsyncMock,
        return_value={"component.termoweb.test_key": "Hello {name}"},
    ):
        result = await _tr(hass, "test_key", name="World")
        assert result == "Hello World"


@pytest.mark.asyncio
async def test_tr_returns_key_when_not_found() -> None:
    """_tr returns the key itself when translation is missing."""

    hass = SimpleNamespace()
    hass.config = SimpleNamespace(language="en")

    with patch(
        "custom_components.termoweb.i18n.async_get_translations",
        new_callable=AsyncMock,
        return_value={},
    ):
        result = await _tr(hass, "missing_key")
        assert result == "missing_key"


@pytest.mark.asyncio
async def test_tr_handles_format_error() -> None:
    """_tr returns the raw template when placeholders mismatch."""

    hass = SimpleNamespace()
    hass.config = SimpleNamespace(language="en")

    with patch(
        "custom_components.termoweb.i18n.async_get_translations",
        new_callable=AsyncMock,
        return_value={"component.termoweb.key": "Hello {missing}"},
    ):
        result = await _tr(hass, "key", name="World")
        assert result == "Hello {missing}"


@pytest.mark.asyncio
async def test_tr_falls_back_to_default_language() -> None:
    """_tr uses default_language when language is not set."""

    hass = SimpleNamespace()
    hass.config = SimpleNamespace(language=None, default_language="fr")

    with patch(
        "custom_components.termoweb.i18n.async_get_translations",
        new_callable=AsyncMock,
        return_value={},
    ) as mock_get:
        await _tr(hass, "key")
        # Should have been called with "fr"
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][1] == "fr"


@pytest.mark.asyncio
async def test_tr_falls_back_to_en() -> None:
    """_tr uses 'en' when neither language nor default_language is set."""

    hass = SimpleNamespace()
    hass.config = SimpleNamespace(language=None)

    with patch(
        "custom_components.termoweb.i18n.async_get_translations",
        new_callable=AsyncMock,
        return_value={},
    ) as mock_get:
        await _tr(hass, "key")
        call_args = mock_get.call_args
        assert call_args[0][1] == "en"


@pytest.mark.asyncio
async def test_async_get_fallback_translations_caches_in_runtime() -> None:
    """Translations are cached in EntryRuntime when provided."""

    from custom_components.termoweb.runtime import EntryRuntime

    hass = SimpleNamespace()
    hass.config = SimpleNamespace(language="en")

    runtime = EntryRuntime.__new__(EntryRuntime)
    runtime.fallback_translations = None

    result = await async_get_fallback_translations(hass, runtime)
    assert isinstance(result, dict)
    assert runtime.fallback_translations == result
