"""Translation helpers for the TermoWeb integration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.translation import async_get_translations
from .const import DOMAIN
from .runtime import EntryRuntime
from .fallback_translations import get_fallback_translations

FALLBACK_TRANSLATIONS_KEY = "fallback_translations"
COORDINATOR_FALLBACK_ATTR = "_termoweb_fallback_translations"


async def _tr(hass: HomeAssistant, key: str, **placeholders: Any) -> str:
    """Return the translated string for ``key`` with ``placeholders``."""

    language = getattr(hass.config, "language", None)
    if not language:
        language = getattr(hass.config, "default_language", None) or "en"
    strings = await async_get_translations(hass, language, DOMAIN)
    template = strings.get(f"component.{DOMAIN}.{key}")
    if template:
        try:
            return template.format(**placeholders)
        except (KeyError, ValueError):
            return template
    return key


async def async_get_fallback_translations(
    hass: HomeAssistant,
    entry_data: EntryRuntime | None = None,
) -> dict[str, str]:
    """Return cached fallback translation templates for the current language."""

    if isinstance(entry_data, EntryRuntime):
        cached = entry_data.fallback_translations
        if isinstance(cached, dict):
            return cached

    language = getattr(hass.config, "language", None)
    if not language:
        language = getattr(hass.config, "default_language", None) or "en"

    fallbacks = get_fallback_translations(language)

    if isinstance(entry_data, EntryRuntime):
        entry_data.fallback_translations = fallbacks

    return fallbacks


def format_fallback(
    fallbacks: Mapping[str, str] | None,
    key: str,
    default_template: str,
    **placeholders: Any,
) -> str:
    """Return the formatted fallback string for ``key``."""

    template: str | None = None
    if isinstance(fallbacks, Mapping):
        template = fallbacks.get(key)
    if template:
        try:
            return template.format(**placeholders)
        except (KeyError, ValueError):
            return template
    try:
        return default_template.format(**placeholders)
    except (KeyError, ValueError):
        return default_template


def attach_fallbacks(target: Any, fallbacks: Mapping[str, str] | None) -> None:
    """Attach fallback translations to ``target`` when supported."""

    if not isinstance(fallbacks, Mapping):
        return
    try:
        setattr(target, COORDINATOR_FALLBACK_ATTR, fallbacks)
    except (AttributeError, TypeError):
        if hasattr(target, "__dict__"):
            target.__dict__[COORDINATOR_FALLBACK_ATTR] = fallbacks


__all__ = [
    "COORDINATOR_FALLBACK_ATTR",
    "FALLBACK_TRANSLATIONS_KEY",
    "_tr",
    "async_get_fallback_translations",
    "attach_fallbacks",
    "format_fallback",
]
