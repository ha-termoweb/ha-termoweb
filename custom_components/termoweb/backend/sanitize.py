"""Shared sanitisation helpers for backend clients."""

from __future__ import annotations

import re
from typing import Any

from custom_components.termoweb.boost import validate_boost_minutes

_BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)
_TOKEN_QUERY_RE = re.compile(r"(?i)(token|refresh_token|access_token)=([^&\s]+)")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def redact_text(value: str | None) -> str:
    """Return ``value`` with bearer tokens, emails and query tokens removed."""

    if not value:
        return ""
    text = str(value)
    if not text:
        return ""
    redacted = _BEARER_RE.sub("Bearer ***", text)
    redacted = _TOKEN_QUERY_RE.sub(lambda match: f"{match.group(1)}=***", redacted)
    redacted = _EMAIL_RE.sub("***@***", redacted)
    return redacted.replace("authorization", "auth").replace("Authorization", "Auth")


def redact_token_fragment(value: str | None) -> str:
    """Return a shortened representation of a token-like string."""

    if value is None:
        return ""
    trimmed = str(value).strip()
    if not trimmed:
        return ""
    if len(trimmed) <= 4:
        return "***"
    if len(trimmed) <= 8:
        return f"{trimmed[:2]}***{trimmed[-2:]}"
    return f"{trimmed[:4]}...{trimmed[-4:]}"


def mask_identifier(value: str | None) -> str:
    """Return a masked identifier suitable for log output."""

    if value is None:
        return ""
    trimmed = str(value).strip()
    if not trimmed:
        return ""
    if len(trimmed) <= 4:
        return "***"
    if len(trimmed) <= 8:
        return f"{trimmed[:2]}...{trimmed[-2:]}"
    prefix = trimmed[:6]
    suffix = trimmed[-4:]
    return f"{prefix}...{suffix}"


def build_acm_boost_payload(
    boost: bool,
    boost_time: int | None,
    *,
    stemp: str | None = None,
    units: str | None = None,
) -> dict[str, Any]:
    """Return a validated accumulator boost payload."""

    payload: dict[str, Any] = {"boost": bool(boost)}
    minutes = validate_boost_minutes(boost_time)
    if minutes is not None:
        payload["boost_time"] = minutes
    if stemp is not None:
        temp_str = str(stemp).strip()
        if not temp_str:
            raise ValueError("stemp must be a non-empty string when provided")
        payload["stemp"] = temp_str
    if units is not None:
        unit = str(units).strip().upper()
        if unit not in {"C", "F"}:
            raise ValueError(f"Invalid units: {units!r}")
        payload["units"] = unit
    return payload


__all__ = [
    "build_acm_boost_payload",
    "mask_identifier",
    "redact_text",
    "redact_token_fragment",
    "validate_boost_minutes",
]
