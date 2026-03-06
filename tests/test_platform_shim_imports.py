"""Verify that platform shim modules import without error.

Each HA platform file (e.g. ``lock.py``, ``climate.py``) is a thin shim that
re-exports symbols from ``entities/<platform>.py``.  If a refactor renames or
removes a symbol in the entity module but forgets to update the shim, the
platform will crash at import time -- but only when Home Assistant actually
loads that platform for a matching brand.

These tests import every shim *through the same path HA uses* so that broken
re-exports are caught unconditionally, regardless of brand or test matrix.
"""

from __future__ import annotations

import importlib

import pytest

# Every platform shim under custom_components/termoweb/ that re-exports from
# entities/.  The tuple pairs are (shim_module, list_of_re_exported_names).
_SHIM_EXPECTATIONS: list[tuple[str, list[str]]] = [
    (
        "custom_components.termoweb.lock",
        ["async_setup_entry", "ChildLockEntity", "build_settings_resolver"],
    ),
    (
        "custom_components.termoweb.switch",
        ["build_settings_resolver"],
    ),
    (
        "custom_components.termoweb.binary_sensor",
        ["async_setup_entry"],
    ),
    (
        "custom_components.termoweb.button",
        ["async_setup_entry"],
    ),
    (
        "custom_components.termoweb.climate",
        ["async_setup_entry", "HeaterClimateEntity"],
    ),
    (
        "custom_components.termoweb.sensor",
        ["async_setup_entry"],
    ),
    (
        "custom_components.termoweb.number",
        ["async_setup_entry"],
    ),
    (
        "custom_components.termoweb.heater",
        ["HeaterNodeBase", "HeaterPlatformDetails"],
    ),
]


@pytest.mark.parametrize(
    ("module_path", "expected_names"),
    _SHIM_EXPECTATIONS,
    ids=[path.rsplit(".", 1)[-1] for path, _ in _SHIM_EXPECTATIONS],
)
def test_platform_shim_imports_and_exports(
    module_path: str, expected_names: list[str]
) -> None:
    """Importing the platform shim should succeed and expose expected symbols."""

    mod = importlib.import_module(module_path)
    for name in expected_names:
        attr = getattr(mod, name, None)
        assert attr is not None, (
            f"{module_path} is missing expected attribute '{name}'. "
            f"A refactor likely renamed or removed it in entities/ without "
            f"updating the platform shim."
        )
