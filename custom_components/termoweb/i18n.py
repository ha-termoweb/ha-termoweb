"""Translation helpers for the TermoWeb integration."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from homeassistant.core import HomeAssistant
try:  # pragma: no cover - fallback for stripped Home Assistant stubs in tests
    from homeassistant.helpers.translation import async_get_translations
except ImportError:  # pragma: no cover - executed in unit test stubs

    async def async_get_translations(  # type: ignore[override]
        hass: HomeAssistant, language: str, domain: str
    ) -> dict[str, str]:
        """Return an empty translation mapping when helpers are unavailable."""

        return {}

from .const import DOMAIN

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
    entry_data: MutableMapping[str, Any] | None = None,
) -> dict[str, str]:
    """Return cached fallback translation templates for the current language."""

    if isinstance(entry_data, MutableMapping):
        cached = entry_data.get(FALLBACK_TRANSLATIONS_KEY)
        if isinstance(cached, dict):
            return cached

    language = getattr(hass.config, "language", None)
    if not language:
        language = getattr(hass.config, "default_language", None) or "en"

    strings = await async_get_translations(hass, language, DOMAIN)
    prefix = f"component.{DOMAIN}."
    fallbacks = {
        key[len(prefix) :]: value
        for key, value in strings.items()
        if key.startswith(f"{prefix}fallbacks.")
    }

    if isinstance(entry_data, MutableMapping):
        entry_data[FALLBACK_TRANSLATIONS_KEY] = fallbacks

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
    "attach_fallbacks",
    "COORDINATOR_FALLBACK_ATTR",
    "FALLBACK_TRANSLATIONS_KEY",
    "_tr",
    "async_get_fallback_translations",
    "format_fallback",
]
