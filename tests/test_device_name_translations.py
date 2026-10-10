"""Device-name translations must match the keys the integration emits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from custom_components.termoweb.utils import _DEFAULT_DEVICE_NAMES

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
