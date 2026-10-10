"""Pydantic models for Ducaheat backend payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from custom_components.termoweb.boost import validate_boost_minutes
from custom_components.termoweb.codecs.common import format_temperature, validate_units


class DucaheatModel(BaseModel):
    """Base model for Ducaheat wire payloads."""

    model_config = ConfigDict(extra="ignore")


class StatusWritePayload(DucaheatModel):
    """Status write payload covering mode, setpoint and presets."""

    mode: str | None = None
    stemp: str | None = None
    units: str | None = None
    ice_temp: str | None = None
    eco_temp: str | None = None
    comf_temp: str | None = None
    boost: bool | None = None
    boost_time: int | None = None

    @field_validator("mode")
    @classmethod
    def _normalise_mode(cls, value: str) -> str:
        """Lowercase mode strings."""

        return value.lower()

    @field_validator("stemp", "ice_temp", "eco_temp", "comf_temp", mode="before")
    @classmethod
    def _format_temps(cls, value: Any) -> str:
        """Validate and format temperature strings."""

        return format_temperature(value)

    @field_validator("units")
    @classmethod
    def _clean_units(cls, value: str) -> str:
        """Ensure units are uppercase."""

        return validate_units(value, trim=True)

    @field_validator("boost_time")
    @classmethod
    def _validate_boost_time(cls, value: int | None) -> int | None:
        """Clamp boost_time to supported options."""

        return validate_boost_minutes(value)


class ModeWritePayload(DucaheatModel):
    """Mode-only write payload."""

    mode: str
    boost_time: int | None = None

    @field_validator("mode")
    @classmethod
    def _normalise_mode(cls, value: Any) -> str:
        """Lowercase the supplied mode string."""

        return str(value).lower()

    @field_validator("boost_time")
    @classmethod
    def _validate_boost_time(cls, value: int | None) -> int | None:
        """Validate boost duration values."""

        return validate_boost_minutes(value)


class BoostPayload(DucaheatModel):
    """Accumulator boost payload."""

    boost: bool
    boost_time: int | None = None
    stemp: str | None = None
    units: str | None = None

    @field_validator("boost_time")
    @classmethod
    def _validate_boost_time(cls, value: int | None) -> int | None:
        """Validate boost duration values."""

        return validate_boost_minutes(value)

    @field_validator("stemp", mode="before")
    @classmethod
    def _format_boost_temp(cls, value: Any) -> str | None:
        """Format boost temperature values."""

        if value is None:
            return None
        return format_temperature(value)

    @field_validator("units")
    @classmethod
    def _validate_units(cls, value: str | None) -> str | None:
        """Ensure units are uppercase when provided."""

        if value is None:
            return None
        return validate_units(value, trim=True)


class LockWritePayload(DucaheatModel):
    """Child lock payload for the lock endpoint."""

    lock: bool


class PriorityWritePayload(DucaheatModel):
    """Priority write payload for the setup endpoint."""

    priority: int = Field(ge=0, le=30)


__all__ = [
    "BoostPayload",
    "LockWritePayload",
    "ModeWritePayload",
    "PriorityWritePayload",
    "StatusWritePayload",
]
