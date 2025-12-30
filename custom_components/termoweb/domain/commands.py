"""Command placeholder types for future writes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BaseCommand:
    """Base type for node commands."""


@dataclass(slots=True)
class AccumulatorCommand(BaseCommand):
    """Command envelope for accumulator-specific actions."""

    boost: bool | None = None
