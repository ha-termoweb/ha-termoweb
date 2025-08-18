from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


UTILS_PATH = (
    Path(__file__).resolve().parents[1] / "custom_components" / "termoweb" / "utils.py"
)


def _load_utils():
    package = "custom_components.termoweb"
    sys.modules.setdefault("custom_components", types.ModuleType("custom_components"))
    termoweb_pkg = types.ModuleType(package)
    termoweb_pkg.__path__ = [str(UTILS_PATH.parent)]
    sys.modules[package] = termoweb_pkg
    spec = importlib.util.spec_from_file_location(f"{package}.utils", UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[f"{package}.utils"] = module
    spec.loader.exec_module(module)
    return module


def test_extract_heater_addrs() -> None:
    utils = _load_utils()
    extract = utils.extract_heater_addrs

    assert extract({}) == []

    nodes = {
        "nodes": [
            {"type": "htr", "addr": "A"},
            {"type": "foo", "addr": "B"},
            {"type": "HTR", "addr": 1},
        ]
    }

    assert extract(nodes) == ["A", "1"]


def test_extract_heater_addrs_deduplicates() -> None:
    utils = _load_utils()
    extract = utils.extract_heater_addrs

    nodes = {"nodes": [{"type": "htr", "addr": "A"}, {"type": "HTR", "addr": "A"}]}

    assert extract(nodes) == ["A"]
