"""Pydantic models for TermoWeb backend payloads."""

from __future__ import annotations

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
