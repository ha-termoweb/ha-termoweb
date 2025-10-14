from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.termoweb import energy
from custom_components.termoweb.energy import OPTION_ENERGY_HISTORY_PROGRESS


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, str], int, int]] = []

    async def get_node_samples(
        self,
        dev_id: str,
        node: tuple[str, str],
        start: int,
        stop: int,
    ) -> list[dict[str, Any]]:
        self.calls.append((dev_id, node, start, stop))
        return []


class _ImmediateRateLimiter:
    async def async_throttle(self, on_wait=None):  # type: ignore[override]
        if callable(on_wait):
            on_wait(0.0)
        return 0.0


class _FixedDatetime(datetime):
    _NOW = datetime(2024, 1, 2, 15, 45, 33, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        base = cls._NOW
        if tz is None:
            tz = base.tzinfo
        return cls(
            base.year,
            base.month,
            base.day,
            base.hour,
            base.minute,
            base.second,
            base.microsecond,
            tzinfo=tz,
        )


class _StubConfigEntry:
    def __init__(self, entry_id: str) -> None:
        self.entry_id = entry_id
        self.data: dict[str, Any] = {}
        self.options: dict[str, Any] = {}


class _StubConfigEntries:
    def __init__(self) -> None:
        self._entries: dict[str, _StubConfigEntry] = {}
        self.updated_entries: list[
            tuple[_StubConfigEntry, dict[str, Any] | None, dict[str, Any] | None]
        ] = []

    def add(self, entry: _StubConfigEntry) -> None:
        self._entries[entry.entry_id] = entry

    def async_update_entry(
        self,
        entry: _StubConfigEntry,
        *,
        data: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        if data is not None:
            entry.data = data
        if options is not None:
            entry.options = options
        self.updated_entries.append((entry, data, options))


@pytest.fixture
def stub_hass() -> SimpleNamespace:
    """Return a minimal Home Assistant stub for energy history tests."""

    config_entries = _StubConfigEntries()
    return SimpleNamespace(data={}, config_entries=config_entries)


@pytest.mark.asyncio
async def test_import_energy_history_fetches_until_current_minute(
    monkeypatch: pytest.MonkeyPatch,
    stub_hass,
    inventory_from_map,
) -> None:
    """Importer should request history through the current minute."""

    entry = _StubConfigEntry("entry")
    stub_hass.config_entries.add(entry)

    inventory = inventory_from_map({"htr": ["A"]}, dev_id="dev-1")
    client = _RecordingClient()

    stub_hass.data.setdefault(energy.DOMAIN, {})[entry.entry_id] = {
        "client": client,
        "dev_id": "dev-1",
        "inventory": inventory,
    }

    monkeypatch.setattr(energy, "datetime", _FixedDatetime, raising=False)
    monkeypatch.setattr(energy.er, "async_get", lambda hass: None, raising=False)

    await energy.async_import_energy_history(
        stub_hass,
        entry,
        rate_limit=_ImmediateRateLimiter(),
        max_days=1,
    )

    assert client.calls
    _, node, start, stop = client.calls[0]
    assert node == ("htr", "A")

    expected_stop = int(
        _FixedDatetime._NOW.replace(second=0, microsecond=0).timestamp()
    )
    assert stop == expected_stop
    assert start == expected_stop - 24 * 3600

    assert stub_hass.config_entries.updated_entries
    last_update = stub_hass.config_entries.updated_entries[-1]
    _, _, options = last_update
    assert isinstance(options, dict)
    assert OPTION_ENERGY_HISTORY_PROGRESS in options
