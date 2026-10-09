"""Local power manager for the radio gateway: keep heaters under a power limit.

A heater's ``BE`` power request is only acknowledged by ``BF``: a dialect-B
heater heats whether the answer is ``BF 01`` or ``BF 00`` (checked against a
house meter). The lever that does stop heating is the mode, so this manager
sheds load the way a person would:

- no limit set: nothing happens, and any heater it switched off comes back;
- over the limit: the heating heaters with the lowest priority are switched
  off (their previous mode is stored) until the rest fit;
- room again: switched-off heaters get their previous mode back, highest
  priority first, while each one's power fits.

Higher priority numbers win. A heater's power is entered in the options (or
reported by dialect-A heaters); a heater whose power is unknown is never
switched off. Nothing here touches the radio or Home Assistant.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)

DEFAULT_PRIORITY = 0

KEY_POWER_LIMIT = "power_limit"
KEY_PRIORITY = "priority"
KEY_RATED_POWER = "rated_power"
KEY_SHED = "shed"  # heaters switched off by the manager -> mode to restore


class PowerManager:
    """Track limit, priorities and rated powers; decide each power request."""

    def __init__(
        self,
        load: Callable[[], Mapping[str, Any] | None] | None = None,
        save: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Read settings through ``load`` on every use; ``save`` stores changes.

        The settings live in one place (the config entry's options), so an
        options-flow edit applies at once. Without callbacks they live here.
        """

        self._memory: dict[str, Any] = {}
        self._load = load or (lambda: self._memory)
        self._save = save or self._memory.update
        self._learned: dict[int, float] = {}
        self._heating: dict[int, bool] = {}

    # --- settings ------------------------------------------------------------

    def _settings(self) -> Mapping[str, Any]:
        """Return the current stored settings."""

        return self._load() or {}

    @property
    def power_limit(self) -> int | None:
        """Return the installation power limit in watts, or None when unset."""

        try:
            return int(self._settings().get(KEY_POWER_LIMIT) or 0) or None
        except (TypeError, ValueError):
            return None

    def set_power_limit(self, watts: int) -> None:
        """Set the limit; 0 removes it."""

        if watts < 0:
            raise ValueError(f"power limit must be >= 0 W, got {watts}")
        self._store(KEY_POWER_LIMIT, int(watts) or None)

    def priority(self, addr: int) -> int:
        """Return a heater's priority (higher wins)."""

        return _int_map(self._settings().get(KEY_PRIORITY)).get(addr, DEFAULT_PRIORITY)

    def set_priority(self, addr: int, priority: int) -> None:
        """Set a heater's priority."""

        priorities = _int_map(self._settings().get(KEY_PRIORITY))
        priorities[addr] = int(priority)
        self._store(KEY_PRIORITY, {str(a): p for a, p in priorities.items()})

    def rated_power(self, addr: int) -> float | None:
        """Return a heater's power: configured, else reported by the heater."""

        configured = _int_map(self._settings().get(KEY_RATED_POWER)).get(addr)
        return float(configured) if configured else self._learned.get(addr)

    def note_reported_power(self, addr: int, watts: float) -> None:
        """Remember the full-load power a heater reported itself (dialect A)."""

        self._learned[addr] = watts

    def _store(self, key: str, value: Any) -> None:
        """Save the settings with ``key`` replaced."""

        self._save({**self._settings(), key: value})

    # --- live state and shedding -------------------------------------------

    def note_heating(self, addr: int, heating: bool) -> None:
        """Record whether a heater last reported heating."""

        self._heating[addr] = heating

    def shed(self) -> dict[int, int]:
        """Return the heaters switched off by the manager, with the mode to restore."""

        return _int_map(self._settings().get(KEY_SHED))

    def mark_shed(self, addr: int, previous_mode: int) -> None:
        """Remember that ``addr`` was switched off and which mode to restore."""

        shed = self.shed()
        shed[addr] = previous_mode
        self._store(KEY_SHED, {str(a): m for a, m in shed.items()})
        self._heating[addr] = False

    def clear_shed(self, addr: int) -> None:
        """Forget a shed heater (restored, or its mode was changed by the user)."""

        shed = self.shed()
        if shed.pop(addr, None) is not None:
            self._store(KEY_SHED, {str(a): m for a, m in shed.items()})

    def plan(self) -> tuple[list[int], list[tuple[int, int]]]:
        """Return (heaters to switch off, (heater, mode) pairs to restore).

        Over the limit, the heating heaters with the lowest priority are switched
        off until the rest fit. Otherwise shed heaters come back, highest
        priority first, while each one's power still fits; all come back when
        the limit is removed. A heater whose power is unknown is never shed.
        """

        limit = self.power_limit
        shed = self.shed()
        if limit is None:
            return [], sorted(shed.items())
        heating = [a for a, on in self._heating.items() if on and a not in shed]
        load = sum(self.rated_power(a) or 0.0 for a in heating)
        to_off: list[int] = []
        for addr in sorted(heating, key=self.priority):
            if load <= limit:
                break
            rated = self.rated_power(addr)
            if rated is None:
                continue
            _LOGGER.info(
                "Power limit: switching heater %s (%.0f W) off, load %.0f W > %d W",
                addr,
                rated,
                load,
                limit,
            )
            to_off.append(addr)
            load -= rated
        to_restore: list[tuple[int, int]] = []
        if not to_off:
            for addr in sorted(shed, key=self.priority, reverse=True):
                rated = self.rated_power(addr)
                if rated is not None and load + rated > limit:
                    continue
                to_restore.append((addr, shed[addr]))
                load += rated or 0.0
        return to_off, to_restore


MAX_INTEGRATION_GAP_S = 900.0  # never extrapolate a reading further than this


class EnergyEstimator:
    """Estimate each heater's energy in Wh from its heating flag and duty.

    Between two power records a heater is taken to draw rated power x duty
    while its heating flag is set (the duty byte matched a house meter's
    pulse ratio), else nothing. The counter starts at 0 when Home Assistant
    starts; the energy coordinator treats a lower counter as a reset.
    """

    def __init__(self, rated_power: Callable[[int], float | None]) -> None:
        """Use ``rated_power(addr)`` (watts or None) for every integration step."""

        self._rated_power = rated_power
        self._last: dict[int, tuple[float, float]] = {}  # addr -> (time, fraction)
        self._wh: dict[int, float] = {}

    def observe(self, addr: int, heating: bool, duty_pct: int, now: float) -> None:
        """Add the energy since the last observation, then store the new state."""

        previous = self._last.get(addr)
        rated = self._rated_power(addr)
        if previous is not None and rated is not None:
            since, fraction = previous
            seconds = min(max(now - since, 0.0), MAX_INTEGRATION_GAP_S)
            self._wh[addr] = self._wh.get(addr, 0.0) + rated * fraction * seconds / 3600
        elif rated is not None:
            self._wh.setdefault(addr, 0.0)
        duty = duty_pct / 100 if 0 < duty_pct <= 100 else 1.0
        self._last[addr] = (now, duty if heating else 0.0)

    def counter_wh(self, addr: int) -> float | None:
        """Return the estimated Wh counter, or None while the power is unknown."""

        return self._wh.get(addr)


def _int_map(raw: Any) -> dict[int, int]:
    """Return ``{"6": 1500}``-style mappings with integer keys and values."""

    if not isinstance(raw, Mapping):
        return {}
    result: dict[int, int] = {}
    for key, value in raw.items():
        try:
            result[int(key)] = int(value)
        except (TypeError, ValueError):
            _LOGGER.debug("Ignoring bad power setting %r=%r", key, value)
    return result


__all__ = ["EnergyEstimator", "PowerManager"]
