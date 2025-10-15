"""Fallback translation strings for the TermoWeb integration."""

from __future__ import annotations

from collections.abc import Mapping

FALLBACK_TRANSLATIONS: dict[str, dict[str, str]] = {
    "cs": {
        "heater_name": "Topidlo {addr}",
        "node_name": "Uzel {addr}",
        "power_monitor_name": "Monitor energie {addr}",
        "never": "Nikdy",
    },
    "de": {
        "heater_name": "Heizgerät {addr}",
        "node_name": "Knoten {addr}",
        "power_monitor_name": "Leistungsmonitor {addr}",
        "never": "Nie",
    },
    "el": {
        "heater_name": "Θερμαντικό {addr}",
        "node_name": "Κόμβος {addr}",
        "power_monitor_name": "Μετρητής ενέργειας {addr}",
        "never": "Ποτέ",
    },
    "en": {
        "heater_name": "Heater {addr}",
        "node_name": "Node {addr}",
        "power_monitor_name": "Power Monitor {addr}",
        "never": "Never",
    },
    "es": {
        "heater_name": "Calefactor {addr}",
        "node_name": "Nodo {addr}",
        "power_monitor_name": "Monitor de potencia {addr}",
        "never": "Nunca",
    },
    "fr": {
        "heater_name": "Radiateur {addr}",
        "node_name": "Nœud {addr}",
        "power_monitor_name": "Surveillance d’énergie {addr}",
        "never": "Jamais",
    },
    "hr": {
        "heater_name": "Grijač {addr}",
        "node_name": "Čvor {addr}",
        "power_monitor_name": "Mjerač energije {addr}",
        "never": "Nikada",
    },
    "it": {
        "heater_name": "Riscaldatore {addr}",
        "node_name": "Nodo {addr}",
        "power_monitor_name": "Monitor di potenza {addr}",
        "never": "Mai",
    },
    "pl": {
        "heater_name": "Grzejnik {addr}",
        "node_name": "Węzeł {addr}",
        "power_monitor_name": "Monitor mocy {addr}",
        "never": "Nigdy",
    },
    "pt-pt": {
        "heater_name": "Aquecedor {addr}",
        "node_name": "Nó {addr}",
        "power_monitor_name": "Monitor de energia {addr}",
        "never": "Nunca",
    },
    "ro": {
        "heater_name": "Încălzitor {addr}",
        "node_name": "Nod {addr}",
        "power_monitor_name": "Monitor de energie {addr}",
        "never": "Niciodată",
    },
    "ru": {
        "heater_name": "Обогреватель {addr}",
        "node_name": "Узел {addr}",
        "power_monitor_name": "Монитор энергии {addr}",
        "never": "Никогда",
    },
    "sk": {
        "heater_name": "Ohrievač {addr}",
        "node_name": "Uzol {addr}",
        "power_monitor_name": "Monitor energie {addr}",
        "never": "Nikdy",
    },
    "tr": {
        "heater_name": "Isıtıcı {addr}",
        "node_name": "Düğüm {addr}",
        "power_monitor_name": "Güç izleyicisi {addr}",
        "never": "Asla",
    },
    "uk": {
        "heater_name": "Обігрівач {addr}",
        "node_name": "Вузол {addr}",
        "power_monitor_name": "Лічильник енергії {addr}",
        "never": "Ніколи",
    },
}


def language_candidates(language: str) -> list[str]:
    """Return ordered fallback language candidates for ``language``."""
    normalized = (language or "").strip()
    if not normalized:
        return ["en"]
    normalized = normalized.replace("_", "-").lower()
    candidates: list[str] = [normalized]
    if "-" in normalized:
        base = normalized.split("-", 1)[0]
        candidates.append(base)
    if "en" not in candidates:
        candidates.append("en")
    return candidates


def get_fallback_translations(language: str) -> Mapping[str, str]:
    """Return fallback translation strings for ``language``."""
    for candidate in language_candidates(language):
        data = FALLBACK_TRANSLATIONS.get(candidate)
        if data:
            return dict(data)
    return FALLBACK_TRANSLATIONS["en"]
