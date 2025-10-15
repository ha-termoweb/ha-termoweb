"""Tests for recorder import resolution helpers."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest

from custom_components.termoweb import energy


def _install_fake_homeassistant(
    monkeypatch: "pytest.MonkeyPatch", recorder_module: ModuleType | None
) -> None:
    """Install fake Home Assistant modules for recorder tests."""

    homeassistant_mod = ModuleType("homeassistant")
    homeassistant_mod.__path__ = []  # type: ignore[attr-defined]
    homeassistant_mod.__spec__ = ModuleSpec(  # type: ignore[attr-defined]
        "homeassistant", loader=None, is_package=True
    )

    components_mod = ModuleType("homeassistant.components")
    components_mod.__path__ = []  # type: ignore[attr-defined]
    components_mod.__spec__ = ModuleSpec(  # type: ignore[attr-defined]
        "homeassistant.components", loader=None, is_package=True
    )

    setattr(homeassistant_mod, "components", components_mod)

    monkeypatch.setitem(sys.modules, "homeassistant", homeassistant_mod)
    monkeypatch.setitem(
        sys.modules, "homeassistant.components", components_mod
    )

    if recorder_module is not None:
        recorder_module.__spec__ = ModuleSpec(  # type: ignore[attr-defined]
            "homeassistant.components.recorder", loader=None, is_package=False
        )
        setattr(components_mod, "recorder", recorder_module)
        monkeypatch.setitem(
            sys.modules,
            "homeassistant.components.recorder",
            recorder_module,
        )
        return

    if "homeassistant.components.recorder" in sys.modules:
        monkeypatch.delitem(
            sys.modules, "homeassistant.components.recorder", raising=False
        )


def test_resolve_recorder_imports_missing_get_instance(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Cache recorder imports when ``get_instance`` is unavailable."""

    monkeypatch.setattr(energy, "_RECORDER_IMPORTS", None)

    recorder_mod = ModuleType("homeassistant.components.recorder")
    statistics_module = SimpleNamespace(marker="statistics")
    recorder_mod.statistics = statistics_module  # type: ignore[attr-defined]

    _install_fake_homeassistant(monkeypatch, recorder_mod)

    first = energy._resolve_recorder_imports()
    second = energy._resolve_recorder_imports()

    assert first is second
    assert first.get_instance is None
    assert first.statistics is statistics_module


def test_resolve_recorder_imports_module_missing(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Cache recorder imports when the recorder module is unavailable."""

    monkeypatch.setattr(energy, "_RECORDER_IMPORTS", None)

    _install_fake_homeassistant(monkeypatch, None)

    first = energy._resolve_recorder_imports()
    second = energy._resolve_recorder_imports()

    assert first is second
    assert first.get_instance is None
    assert first.statistics is None


@pytest.mark.asyncio
async def test_clear_statistics_compat_uses_instance_async(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """Use recorder instance async delete helper when available."""

    monkeypatch.setattr(energy, "_RECORDER_IMPORTS", None)

    calls: list[tuple[list[str], dict[str, datetime]]] = []

    class _Recorder:
        """Stub recorder with async deletion helpers."""

        async def async_delete_statistics(
            self,
            statistic_ids: list[str],
            *,
            start_time: datetime | None = None,
            end_time: datetime | None = None,
        ) -> None:
            calls.append(
                (
                    list(statistic_ids),
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                    },
                )
            )

        async def async_add_executor_job(self, *args, **kwargs):
            raise AssertionError("executor helper should not be used")

    recorder = _Recorder()

    recorder_mod = ModuleType("homeassistant.components.recorder")
    recorder_mod.get_instance = lambda hass: recorder  # type: ignore[attr-defined]
    recorder_mod.statistics = SimpleNamespace()  # type: ignore[attr-defined]

    _install_fake_homeassistant(monkeypatch, recorder_mod)

    hass = SimpleNamespace()
    start = datetime(2025, 10, 1, tzinfo=UTC)
    end = datetime(2025, 10, 2, tzinfo=UTC)

    result = await energy._clear_statistics_compat(
        hass,
        "sensor.energy_total",
        start_time=start,
        end_time=end,
    )

    assert result == "delete"
    assert calls == [
        (
            ["sensor.energy_total"],
            {"start_time": start, "end_time": end},
        )
    ]
