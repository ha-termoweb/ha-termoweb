from __future__ import annotations

from pathlib import Path


ENTITY_FILES = (
    "button.py",
    "binary_sensor.py",
    "climate.py",
    "heater.py",
    "number.py",
    "sensor.py",
)
FORBIDDEN_TERMS = ("state_to_dict", "heater_settings(")


def test_entities_avoid_dict_state_access() -> None:
    """Ensure entity modules do not reference dict-based state helpers."""

    component_root = (
        Path(__file__).resolve().parents[1] / "custom_components" / "termoweb"
    )
    missing: list[str] = []
    for relative in ENTITY_FILES:
        path = component_root / relative
        content = path.read_text(encoding="utf-8")
        for term in FORBIDDEN_TERMS:
            if term in content:
                missing.append(f"{path.name} -> {term}")
    assert not missing, f"Remove dict helpers from entities: {', '.join(missing)}"
