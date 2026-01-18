"""Pydantic models for Ducaheat read payloads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from custom_components.termoweb.boost import validate_boost_minutes
from custom_components.termoweb.codecs.common import format_temperature, validate_units

ACCUMULATOR_ONLY_FIELDS: set[str] = {
    "boost",
    "boost_active",
    "boost_end_day",
    "boost_end_min",
    "boost_end_datetime",
    "boost_minutes_delta",
    "boost_remaining",
    "boost_temp",
    "boost_time",
    "charge_level",
    "charging",
    "current_charge_per",
    "target_charge_per",
}


def _coerce_bool(value: Any) -> bool | None:
    """Coerce common truthy and falsy values to ``bool``."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _coerce_number(value: Any) -> float | int | None:
    """Return ``value`` as a number when possible."""

    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_percentage(value: Any) -> int | None:
    """Clamp percentage values to the 0-100 range."""

    number = _coerce_number(value)
    if number is None:
        return None
    try:
        as_int = int(number)
    except (TypeError, ValueError):
        return None
    return max(0, min(100, as_int))


def _coerce_int(value: Any) -> int | None:
    """Return ``value`` coerced to int when possible."""

    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _safe_temperature(value: Any) -> str | None:
    """Defensively format inbound temperature values."""

    if value is None:
        return None
    try:
        return format_temperature(value)
    except ValueError:
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        return None


def _normalise_prog(data: Any) -> list[int] | None:
    """Convert vendor programme payloads into a 168-slot list."""

    if isinstance(data, list):
        try:
            values = [int(v) for v in data]
        except (TypeError, ValueError):
            return None
        if len(values) == 168 and all(v in (0, 1, 2) for v in values):
            return values
        return None

    if not isinstance(data, Mapping):
        return None

    days_section: Mapping[str, Any] | None = None
    if isinstance(data.get("days"), Mapping):
        days_section = data["days"]
    else:
        candidate = {
            k: v
            for k, v in data.items()
            if str(k) in {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
            or str(k).isdigit()
        }
        if candidate:
            days_section = candidate

    if days_section is None and isinstance(data.get("prog"), Mapping):
        days_section = data["prog"]

    if not isinstance(days_section, Mapping):
        return None

    day_order = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

    def _coerce_slots(entry: Any) -> list[int] | None:
        """Convert a day's slots into a 24-value list."""

        candidate = entry
        if isinstance(candidate, Mapping):
            slots_candidate = candidate.get("slots")
            values_candidate = candidate.get("values")
            if isinstance(slots_candidate, list):
                candidate = slots_candidate
            else:
                candidate = values_candidate

        if candidate is None or not isinstance(candidate, list):
            return None

        try:
            values = [int(v) for v in candidate]
        except (TypeError, ValueError):
            return None

        if len(values) == 48:
            values = [max(values[i : i + 2]) for i in range(0, 48, 2)]

        if len(values) < 24:
            values = values + [0] * (24 - len(values))
        if len(values) > 24:
            values = values[:24]

        return values if len(values) == 24 else None

    values: list[int] = []
    for idx, day in enumerate(day_order):
        entry = days_section.get(day) if isinstance(days_section, Mapping) else None
        if entry is None and isinstance(days_section, Mapping):
            entry = days_section.get(str(idx)) or days_section.get(idx)

        if entry is None:
            day_values = [0] * 24
        else:
            day_values = _coerce_slots(entry)
            if day_values is None:
                return None
        values.extend(day_values)

    return values if len(values) == 168 else None


def _normalise_prog_temps(data: Any) -> list[str] | None:
    """Convert preset temperature payloads into stringified list."""

    if not isinstance(data, Mapping):
        return None
    antifrost = data.get("antifrost") or data.get("cold")
    eco = data.get("eco") or data.get("night")
    comfort = data.get("comfort") or data.get("day")
    temps = [antifrost, eco, comfort]
    formatted: list[str] = []
    for value in temps:
        if value is None:
            formatted.append("")
            continue
        safe = _safe_temperature(value)
        formatted.append(str(value) if safe is None else safe)
    return formatted


class DucaheatReadModel(BaseModel):
    """Base model for Ducaheat wire payloads."""

    model_config = ConfigDict(extra="ignore")


class DucaheatStatusSegment(DucaheatReadModel):
    """Inbound status segment for heaters and accumulators."""

    mode: str | None = None
    state: str | None = Field(
        default=None,
        validation_alias=AliasChoices("state", "heating_state", "output_state"),
    )
    units: str | None = None
    stemp: str | None = Field(
        default=None,
        validation_alias=AliasChoices("stemp", "set_temp", "target", "setpoint"),
    )
    mtemp: str | None = Field(
        default=None,
        validation_alias=AliasChoices("mtemp", "temp", "ambient", "room_temp"),
    )
    boost_active: bool | None = None
    boost_remaining: float | int | None = None
    lock: bool | None = None
    lock_active: bool | None = None
    max_power: float | int | None = None
    boost: bool | None = None
    boost_time: int | None = None
    boost_temp: str | None = None
    boost_end_day: int | None = None
    boost_end_min: int | None = None
    boost_end: Mapping[str, Any] | None = Field(default=None, exclude=True)
    charging: bool | None = None
    charge_level: float | int | None = None
    current_charge_per: int | None = None
    target_charge_per: int | None = None

    @field_validator("mode")
    @classmethod
    def _normalise_mode(cls, value: Any) -> str | None:
        """Lowercase mode strings when provided."""

        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        return cleaned.lower() if cleaned else None

    @field_validator("state")
    @classmethod
    def _normalise_state(cls, value: Any) -> str | None:
        """Lowercase state strings when provided."""

        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        return cleaned.lower() if cleaned else None

    @field_validator("units")
    @classmethod
    def _clean_units(cls, value: str | None) -> str | None:
        """Ensure units are uppercase when provided."""

        if value is None:
            return None
        try:
            return validate_units(value, trim=True)
        except ValueError:
            return None

    @field_validator("stemp", "mtemp", "boost_temp", mode="before")
    @classmethod
    def _format_temps(cls, value: Any) -> str | None:
        """Validate and format temperature strings."""

        return _safe_temperature(value)

    @field_validator(
        "boost_active",
        "lock",
        "lock_active",
        "boost",
        "charging",
        mode="before",
    )
    @classmethod
    def _coerce_booleans(cls, value: Any) -> bool | None:
        """Coerce truthy and falsy values."""

        return _coerce_bool(value)

    @field_validator("boost_time")
    @classmethod
    def _validate_boost_time(cls, value: int | None) -> int | None:
        """Validate boost duration values."""

        if value is None:
            return None
        try:
            minutes = int(value)
        except (TypeError, ValueError):
            return None
        try:
            return validate_boost_minutes(minutes)
        except ValueError:
            return minutes

    @field_validator(
        "boost_remaining",
        "max_power",
        "charge_level",
        mode="before",
    )
    @classmethod
    def _coerce_numbers(cls, value: Any) -> float | int | None:
        """Coerce numeric metadata safely."""

        return _coerce_number(value)

    @field_validator("boost_end_day", "boost_end_min", mode="before")
    @classmethod
    def _clean_boost_end(cls, value: Any) -> int | None:
        """Convert boost end metadata to integers when possible."""

        return _coerce_int(value)

    @field_validator("current_charge_per", "target_charge_per", mode="before")
    @classmethod
    def _clean_percentages(cls, value: Any) -> int | None:
        """Clamp charge percentage values."""

        return _coerce_percentage(value)

    @model_validator(mode="after")
    def _derive_boost_end(self) -> DucaheatStatusSegment:
        """Populate boost end fields from nested mappings when supplied."""

        if self.boost_end and isinstance(self.boost_end, Mapping):
            self.boost_end_day = self.boost_end_day or _coerce_int(
                self.boost_end.get("day")
            )
            self.boost_end_min = self.boost_end_min or _coerce_int(
                self.boost_end.get("minute")
            )
        return self


class DucaheatExtraOptions(DucaheatReadModel):
    """Nested setup.extra_options payload."""

    boost_time: int | None = None
    boost_temp: str | None = None
    boost_end_day: int | None = None
    boost_end_min: int | None = None
    boost_end: Mapping[str, Any] | None = Field(default=None, exclude=True)
    charging: bool | None = None
    current_charge_per: int | None = None
    target_charge_per: int | None = None

    @field_validator("boost_time")
    @classmethod
    def _validate_boost_time(cls, value: int | None) -> int | None:
        """Validate boost duration values."""

        if value is None:
            return None
        try:
            minutes = int(value)
        except (TypeError, ValueError):
            return None
        try:
            return validate_boost_minutes(minutes)
        except ValueError:
            return minutes

    @field_validator("boost_temp", mode="before")
    @classmethod
    def _format_boost_temp(cls, value: Any) -> str | None:
        """Format boost temperature strings."""

        return _safe_temperature(value)

    @field_validator("boost_end_day", "boost_end_min", mode="before")
    @classmethod
    def _clean_boost_end(cls, value: Any) -> int | None:
        """Convert boost end metadata to integers when possible."""

        return _coerce_int(value)

    @field_validator("charging", mode="before")
    @classmethod
    def _coerce_charging(cls, value: Any) -> bool | None:
        """Coerce charging metadata to booleans."""

        return _coerce_bool(value)

    @field_validator("current_charge_per", "target_charge_per", mode="before")
    @classmethod
    def _clean_percentages(cls, value: Any) -> int | None:
        """Clamp charge percentage values."""

        return _coerce_percentage(value)

    @model_validator(mode="after")
    def _derive_boost_end(self) -> DucaheatExtraOptions:
        """Populate boost end fields from nested mappings when supplied."""

        if self.boost_end and isinstance(self.boost_end, Mapping):
            self.boost_end_day = self.boost_end_day or _coerce_int(
                self.boost_end.get("day")
            )
            self.boost_end_min = self.boost_end_min or _coerce_int(
                self.boost_end.get("minute")
            )
        return self


class DucaheatSetupSegment(DucaheatReadModel):
    """Setup segment container for heaters and accumulators."""

    boost_time: int | None = None
    boost_temp: str | None = None
    boost_end_day: int | None = None
    boost_end_min: int | None = None
    boost_end: Mapping[str, Any] | None = Field(default=None, exclude=True)
    charging: bool | None = None
    current_charge_per: int | None = None
    target_charge_per: int | None = None
    extra_options: DucaheatExtraOptions | None = None

    @field_validator("boost_time")
    @classmethod
    def _validate_boost_time(cls, value: int | None) -> int | None:
        """Validate boost duration values."""

        if value is None:
            return None
        try:
            minutes = int(value)
        except (TypeError, ValueError):
            return None
        try:
            return validate_boost_minutes(minutes)
        except ValueError:
            return minutes

    @field_validator("boost_temp", mode="before")
    @classmethod
    def _format_boost_temp(cls, value: Any) -> str | None:
        """Format boost temperature strings."""

        return _safe_temperature(value)

    @field_validator("boost_end_day", "boost_end_min", mode="before")
    @classmethod
    def _clean_boost_end(cls, value: Any) -> int | None:
        """Convert boost end metadata to integers when possible."""

        return _coerce_int(value)

    @field_validator("charging", mode="before")
    @classmethod
    def _coerce_charging(cls, value: Any) -> bool | None:
        """Coerce charging metadata to booleans."""

        return _coerce_bool(value)

    @field_validator("current_charge_per", "target_charge_per", mode="before")
    @classmethod
    def _clean_percentages(cls, value: Any) -> int | None:
        """Clamp charge percentage values."""

        return _coerce_percentage(value)

    @model_validator(mode="after")
    def _derive_boost_end(self) -> DucaheatSetupSegment:
        """Populate boost end fields from nested mappings when supplied."""

        if self.boost_end and isinstance(self.boost_end, Mapping):
            self.boost_end_day = self.boost_end_day or _coerce_int(
                self.boost_end.get("day")
            )
            self.boost_end_min = self.boost_end_min or _coerce_int(
                self.boost_end.get("minute")
            )
        return self


class DucaheatThermostatSettings(DucaheatReadModel):
    """Inbound thermostat settings payload."""

    mode: str | None = None
    state: str | None = None
    stemp: float | None = Field(
        default=None,
        validation_alias=AliasChoices("stemp", "setpoint", "set_temp", "target"),
    )
    mtemp: float | None = Field(
        default=None,
        validation_alias=AliasChoices("mtemp", "temp", "ambient", "room_temp"),
    )
    units: str | None = None
    ptemp: list[float | str] | None = Field(
        default=None,
        validation_alias=AliasChoices("ptemp", "prog_temps"),
    )
    prog: list[int] | None = None
    batt_level: int | None = None

    @field_validator("mode")
    @classmethod
    def _normalise_mode(cls, value: Any) -> str | None:
        """Lowercase mode strings when provided."""

        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        return cleaned.lower() if cleaned else None

    @field_validator("state")
    @classmethod
    def _normalise_state(cls, value: Any) -> str | None:
        """Lowercase state strings when provided."""

        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        return cleaned.lower() if cleaned else None

    @field_validator("stemp", "mtemp", mode="before")
    @classmethod
    def _coerce_temps(cls, value: Any) -> float | None:
        """Coerce temperature values to floats."""

        number = _coerce_number(value)
        if number is None:
            return None
        try:
            return float(number)
        except (TypeError, ValueError):
            return None

    @field_validator("units")
    @classmethod
    def _clean_units(cls, value: str | None) -> str | None:
        """Ensure units are uppercase when provided."""

        if value is None:
            return None
        try:
            return validate_units(value, trim=True)
        except ValueError:
            return None

    @field_validator("ptemp", mode="before")
    @classmethod
    def _normalise_ptemp(cls, value: Any) -> list[float | str] | None:
        """Normalise preset temperatures."""

        if isinstance(value, Mapping):
            return _normalise_prog_temps(value)
        if isinstance(value, Iterable) and not isinstance(value, (bytes, str)):
            cleaned: list[float] = []
            for temp in value:
                number = _coerce_number(temp)
                if number is None:
                    return None
                cleaned.append(float(number))
            return cleaned
        return None

    @field_validator("prog", mode="before")
    @classmethod
    def _normalise_prog(cls, value: Any) -> list[int] | None:
        """Convert weekly programmes to canonical lists."""

        return _normalise_prog(value)

    @field_validator("batt_level", mode="before")
    @classmethod
    def _clean_battery(cls, value: Any) -> int | None:
        """Clamp battery level values."""

        try:
            battery = int(value)
        except (TypeError, ValueError):
            return None
        return max(0, min(5, battery))


class DucaheatSegmentedSettings(DucaheatReadModel):
    """Segmented heater/accumulator settings payload."""

    status: DucaheatStatusSegment | None = None
    setup: DucaheatSetupSegment | None = None
    prog: list[int] | None = Field(
        default=None,
        validation_alias=AliasChoices("prog", "program"),
    )
    prog_temps: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("prog_temps", "ptemp"),
    )

    @field_validator("prog", mode="before")
    @classmethod
    def _normalise_prog(cls, value: Any) -> list[int] | None:
        """Convert weekly programmes to canonical lists."""

        return _normalise_prog(value)

    @field_validator("prog_temps", mode="before")
    @classmethod
    def _normalise_prog_temps(cls, value: Any) -> list[str] | None:
        """Normalise preset temperature payloads."""

        return _normalise_prog_temps(value)

    def to_flat_dict(self, *, accumulator: bool) -> dict[str, Any]:
        """Return a canonicalised mapping for domain storage."""

        flattened: dict[str, Any] = {}
        if self.status:
            flattened.update(
                self.status.model_dump(
                    exclude_none=True,
                )
            )

        if accumulator and self.setup:
            if self.setup.extra_options:
                nested = self.setup.extra_options.model_dump(exclude_none=True)
                if "boost_time" in nested:
                    flattened["boost_time"] = nested["boost_time"]
                if "boost_temp" in nested:
                    flattened["boost_temp"] = nested["boost_temp"]
                _merge_boost_metadata(flattened, nested, prefer_existing=True)
                _merge_accumulator_charge_metadata(
                    flattened, nested, prefer_existing=True
                )

            setup_dump = self.setup.model_dump(
                exclude_none=True,
                exclude={"extra_options"},
            )
            _merge_boost_metadata(flattened, setup_dump, prefer_existing=True)
            _merge_accumulator_charge_metadata(
                flattened, setup_dump, prefer_existing=True
            )

        if self.prog is not None:
            flattened["prog"] = self.prog

        if self.prog_temps is not None:
            flattened["ptemp"] = self.prog_temps

        if not accumulator:
            flattened = {
                key: value
                for key, value in flattened.items()
                if key not in ACCUMULATOR_ONLY_FIELDS
            }

        return flattened


def _merge_boost_metadata(
    target: dict[str, Any],
    source: Mapping[str, Any] | None,
    *,
    prefer_existing: bool = False,
) -> None:
    """Copy boost metadata from ``source`` into ``target`` safely."""

    if not isinstance(source, Mapping):
        return

    def _assign(
        key: str,
        value: Any,
        *,
        prefer: bool | None = None,
        allow_none: bool = False,
    ) -> None:
        """Assign a metadata value while respecting preference rules."""

        if value is None and not allow_none:
            return

        prefer_flag = prefer_existing if prefer is None else prefer
        if prefer_flag and key in target and target[key] is not None:
            return
        if prefer_flag and key in target and target[key] is None and value is None:
            return

        target[key] = value

    for key in ("boost", "boost_end_day", "boost_end_min"):
        if key in source:
            _assign(key, source[key])

    if "boost_end" in source and isinstance(source["boost_end"], Mapping):
        boost_end = source["boost_end"]
        _assign("boost_end_day", boost_end.get("day"), prefer=True)
        _assign("boost_end_min", boost_end.get("minute"), prefer=True)


def _merge_accumulator_charge_metadata(
    target: dict[str, Any],
    source: Mapping[str, Any] | None,
    *,
    prefer_existing: bool = False,
) -> None:
    """Copy accumulator charge metadata from ``source`` into ``target``."""

    if not isinstance(source, Mapping):
        return

    def _should_assign(key: str) -> bool:
        if not prefer_existing:
            return True
        if key not in target:
            return True
        return target[key] is None

    charging_value = _coerce_bool(source.get("charging"))
    if charging_value is not None and _should_assign("charging"):
        target["charging"] = charging_value

    for key in ("current_charge_per", "target_charge_per"):
        if not _should_assign(key):
            continue
        coerced = _coerce_percentage(source.get(key))
        if coerced is None:
            continue
        target[key] = coerced


__all__ = [
    "DucaheatExtraOptions",
    "DucaheatReadModel",
    "DucaheatSegmentedSettings",
    "DucaheatSetupSegment",
    "DucaheatStatusSegment",
    "DucaheatThermostatSettings",
    "_merge_accumulator_charge_metadata",
    "_merge_boost_metadata",
    "_normalise_prog",
    "_normalise_prog_temps",
    "_safe_temperature",
]
