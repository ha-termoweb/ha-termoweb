"""Tests for constants, translations and the platform module layout."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

import custom_components.termoweb as termoweb
from custom_components.termoweb import const
from custom_components.termoweb.utils import _DEFAULT_DEVICE_NAMES


def test_uses_ducaheat_backend_aliases() -> None:
    """Brands mapped to Ducaheat share backend selection."""

    assert const.uses_ducaheat_backend(const.BRAND_DUCAHEAT) is True
    assert const.uses_ducaheat_backend(const.BRAND_TEVOLVE) is True
    assert const.uses_ducaheat_backend(const.BRAND_TERMOWEB) is False


def test_unknown_brand_uses_the_termoweb_api() -> None:
    """An unknown brand falls back to the TermoWeb API base."""
    assert const.get_brand_api_base("unknown-brand") == const.API_BASE


COMPONENT = Path(__file__).parents[1] / "custom_components" / "termoweb"
TRANSLATION_FILES = sorted((COMPONENT / "translations").glob("*.json"))


def _device_names(path: Path) -> dict[str, str]:
    """Return ``{translation_key: name}`` from the ``device`` section of ``path``."""

    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: value["name"] for key, value in data.get("device", {}).items()}


def test_strings_and_english_match_default_names() -> None:
    """strings.json and en.json carry exactly the English defaults in code."""

    assert _device_names(COMPONENT / "strings.json") == _DEFAULT_DEVICE_NAMES
    assert (
        _device_names(COMPONENT / "translations" / "en.json") == _DEFAULT_DEVICE_NAMES
    )


@pytest.mark.parametrize("path", TRANSLATION_FILES, ids=lambda path: path.stem)
def test_translated_device_names_format_with_addr(path: Path) -> None:
    """Every language formats with the ``addr`` placeholder HA substitutes.

    HA calls ``name.format(**translation_placeholders)`` and raises outside the
    stable channel when a placeholder is missing, so a bad template in any
    language would break device registration for users of that language.
    """

    names = _device_names(path)
    assert set(names) <= set(_DEFAULT_DEVICE_NAMES)
    for name in names.values():
        formatted = name.format(addr="12")
        assert "12" in formatted
        assert "{" not in formatted


PACKAGE_DIR = Path(termoweb.__file__).parent
PLATFORM_MODULES = ("binary_sensor", "button", "climate", "lock", "number", "sensor")


def test_entities_package_and_legacy_shims_are_gone() -> None:
    """The entities/ package and the heater/switch shims must not come back."""

    assert not (PACKAGE_DIR / "entities").exists()
    assert not (PACKAGE_DIR / "heater.py").exists()
    assert not (PACKAGE_DIR / "switch.py").exists()


@pytest.mark.parametrize("name", PLATFORM_MODULES)
def test_platform_module_defines_setup_and_no_private_reexports(name: str) -> None:
    """Each platform defines async_setup_entry and does not re-export private names."""

    module = importlib.import_module(f"custom_components.termoweb.{name}")
    assert callable(module.async_setup_entry)
    assert module.async_setup_entry.__module__ == module.__name__

    tree = ast.parse((PACKAGE_DIR / f"{name}.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            imported = [alias.name for alias in node.names]
            assert "*" not in imported, f"{name}.py uses a star import"
            private = [n for n in imported if n.startswith("_")]
            assert not private, f"{name}.py imports private names {private}"
        # ``_x = other_module._x`` is the shim-style private re-export.
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
            assert not node.value.attr.startswith("_"), (
                f"{name}.py re-exports private attribute {node.value.attr}"
            )
