"""Guard the platform module layout: entity code lives in the platform modules."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import custom_components.termoweb as termoweb

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
