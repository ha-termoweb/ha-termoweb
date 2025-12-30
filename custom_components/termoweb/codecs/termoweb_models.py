"""Pydantic models for TermoWeb backend payloads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TokenResponse(BaseModel):
    """Bearer token payload returned by the password grant flow."""

    model_config = ConfigDict(extra="ignore")

    access_token: str
    token_type: str | None = None
    expires_in: int | float | None = None
    scope: str | None = None


class DevSummary(BaseModel):
    """Summary of a gateway returned by ``/api/v2/devs/``."""

    model_config = ConfigDict(extra="allow")

    dev_id: str | None = None
    id: str | int | None = None
    serial_id: str | None = None
    name: str | None = None


class DevListResponse(BaseModel):
    """Device list payload supporting multiple legacy shapes."""

    model_config = ConfigDict(extra="ignore")

    devs: list[DevSummary] | None = None
    devices: list[DevSummary] | None = None


class NodeSummary(BaseModel):
    """Minimal node descriptor returned by the manager endpoints."""

    model_config = ConfigDict(extra="allow")

    type: str | None = None
    node_type: str | None = Field(default=None, alias="node_type")
    addr: int | str | None = None
    address: int | str | None = None
    name: str | None = None
    title: str | None = None
    label: str | None = None

    @field_validator("addr", "address", mode="before")
    @classmethod
    def _stringify_numeric(cls, value: Any) -> Any:
        """Convert numeric addresses to strings for consistency."""

        if isinstance(value, (int, float)):
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            return str(value)
        return value


class NodesResponse(BaseModel):
    """Node inventory payload for a gateway."""

    model_config = ConfigDict(extra="allow")

    nodes: list[NodeSummary] | dict[str, dict[str, NodeSummary]] | None = None
    htr: dict[str, NodeSummary] | None = None
    thm: dict[str, NodeSummary] | None = None
    acm: dict[str, NodeSummary] | None = None
    pmo: dict[str, NodeSummary] | None = None


def _format_temperature(value: Any) -> Any:
    """Return a temperature value formatted with one decimal when numeric."""

    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return value


def normalise_temperature(value: Any) -> Any:
    """Format numeric temperatures into a one-decimal string when possible."""

    return _format_temperature(value)


def normalise_prog(value: Any) -> Any:
    """Coerce program values to integers when possible without raising."""

    if not isinstance(value, list):
        return value
    normalised: list[int | float | str] = []
    for item in value:
        try:
            normalised.append(int(item))
        except (TypeError, ValueError):
            normalised.append(item)
    return normalised


def normalise_ptemp(value: Any) -> Any:
    """Format preset temperatures while preserving the original length."""

    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return value
    return [_format_temperature(item) for item in value]


class HeaterStatusPayload(BaseModel):
    """Settings/status fields shared by heater and accumulator nodes."""

    model_config = ConfigDict(extra="allow")

    mode: str | None = None
    stemp: Any | None = None
    mtemp: Any | None = None
    temp: Any | None = None
    prog: list[int | float | str] | None = None
    ptemp: list[Any] | None = None
    units: str | None = None
    boost_active: bool | None = None
    boost_time: int | float | None = None
    boost_temp: Any | None = None

    @field_validator("stemp", "mtemp", "temp", "boost_temp", mode="before")
    @classmethod
    def _normalise_temperature(cls, value: Any) -> Any:
        """Format numeric temperatures to a one-decimal string."""

        return normalise_temperature(value)

    @field_validator("prog", mode="before")
    @classmethod
    def _normalise_prog(cls, value: Any) -> Any:
        """Coerce program values to integers when possible without raising."""

        return normalise_prog(value)

    @field_validator("ptemp", mode="before")
    @classmethod
    def _normalise_ptemp(cls, value: Any) -> Any:
        """Format preset temperatures without altering the original length."""

        return normalise_ptemp(value)


class HeaterSettingsPayload(BaseModel):
    """Top-level heater settings payload."""

    model_config = ConfigDict(extra="allow")

    mode: str | None = None
    stemp: Any | None = None
    mtemp: Any | None = None
    temp: Any | None = None
    prog: list[int | float | str] | None = None
    ptemp: list[Any] | None = None
    units: str | None = None
    status: HeaterStatusPayload | Mapping[str, Any] | None = None
    capabilities: dict[str, Any] | None = None

    @field_validator("stemp", "mtemp", "temp", mode="before")
    @classmethod
    def _normalise_temperature(cls, value: Any) -> Any:
        """Format numeric temperatures to a one-decimal string."""

        return normalise_temperature(value)

    @field_validator("prog", mode="before")
    @classmethod
    def _normalise_prog(cls, value: Any) -> Any:
        """Coerce program values to integers when possible without raising."""

        return normalise_prog(value)

    @field_validator("ptemp", mode="before")
    @classmethod
    def _normalise_ptemp(cls, value: Any) -> Any:
        """Format preset temperatures without altering the original length."""

        return normalise_ptemp(value)


class ThermostatSettingsPayload(BaseModel):
    """Thermostat settings payload."""

    model_config = ConfigDict(extra="allow")

    mode: str | None = None
    stemp: Any | None = None
    mtemp: Any | None = None
    temp: Any | None = None
    prog: list[int | float | str] | None = None
    ptemp: list[Any] | None = None
    units: str | None = None

    @field_validator("stemp", "mtemp", "temp", mode="before")
    @classmethod
    def _normalise_temperature(cls, value: Any) -> Any:
        """Format numeric temperatures to a one-decimal string."""

        return normalise_temperature(value)

    @field_validator("prog", mode="before")
    @classmethod
    def _normalise_prog(cls, value: Any) -> Any:
        """Coerce program values to integers when possible without raising."""

        return normalise_prog(value)

    @field_validator("ptemp", mode="before")
    @classmethod
    def _normalise_ptemp(cls, value: Any) -> Any:
        """Format preset temperatures without altering the original length."""

        return normalise_ptemp(value)


class PowerMonitorPayload(BaseModel):
    """Power monitor payload wrapper used by the REST endpoint."""

    model_config = ConfigDict(extra="allow")

    status: Mapping[str, Any] | None = None
    capabilities: Mapping[str, Any] | None = None


class SampleItem(BaseModel):
    """Flexible representation of an energy sample item."""

    model_config = ConfigDict(extra="ignore")

    t: float | int | None = None
    timestamp: float | int | None = None
    counter: Any | None = None
    counter_min: Any | None = None
    counter_max: Any | None = None
    value: Any | None = None
    energy: Any | None = None


class SamplesResponse(BaseModel):
    """Samples payload returned by the REST API."""

    model_config = ConfigDict(extra="ignore")

    samples: list[SampleItem] | None = None


class NodeSettingsWritePayload(BaseModel):
    """Payload for TermoWeb node settings writes."""

    model_config = ConfigDict(extra="ignore")

    mode: str | None = None
    stemp: str | None = None
    prog: list[int] | None = None
    ptemp: list[str] | None = None
    units: str | None = None

    @field_validator("mode", mode="before")
    @classmethod
    def _normalise_mode(cls, value: Any) -> Any:
        """Lower-case mode strings when provided."""

        if value is None:
            return value
        return str(value).lower()

    @field_validator("units", mode="before")
    @classmethod
    def _normalise_units(cls, value: Any) -> Any:
        """Upper-case temperature unit identifiers."""

        if value is None:
            return value
        return str(value).strip().upper()


class ExtraOptionsPayload(BaseModel):
    """Accumulator extra options payload."""

    model_config = ConfigDict(extra="ignore")

    boost_time: int | None = None
    boost_temp: str | None = None


class AcmExtraOptionsWritePayload(BaseModel):
    """Wrapper for accumulator extra options writes."""

    model_config = ConfigDict(extra="ignore")

    extra_options: ExtraOptionsPayload


class AcmBoostWritePayload(BaseModel):
    """Accumulator boost payload."""

    model_config = ConfigDict(extra="ignore")

    boost: bool
    boost_time: int | None = None
    stemp: str | None = None
    units: str | None = None

    @field_validator("units", mode="before")
    @classmethod
    def _normalise_units(cls, value: Any) -> Any:
        """Upper-case temperature units for boost writes."""

        if value is None:
            return value
        return str(value).strip().upper()
