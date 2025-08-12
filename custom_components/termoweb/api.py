from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp

from .const import (
    ACCEPT_LANGUAGE,
    API_BASE,
    BASIC_AUTH_B64,
    DEVS_PATH,
    NODES_PATH_FMT,
    TOKEN_PATH,
    USER_AGENT,
)

_LOGGER = logging.getLogger(__name__)

# Toggle to preview bodies in debug logs (redacted). Leave False by default.
API_LOG_PREVIEW = False


class TermoWebAuthError(Exception):
    """Authentication with TermoWeb failed."""


class TermoWebRateLimitError(Exception):
    """Server rate-limited the client (HTTP 429)."""


def _redact_bearer(text: str | None) -> str:
    """Remove Bearer tokens from an arbitrary string (defensive)."""
    if not text:
        return ""
    # Very defensive scrubbing for 'Bearer <opaque>' and long hex strings.
    return (
        text.replace("Bearer ", "Bearer ***REDACTED*** ")
        .replace("authorization", "auth")
        .replace("Authorization", "Auth")
    )


class TermoWebClient:
    """Thin async client for the TermoWeb cloud (HA-safe)."""

    def __init__(self, session: aiohttp.ClientSession, username: str, password: str) -> None:
        self._session = session
        self._username = username
        self._password = password
        self._access_token: Optional[str] = None
        self._token_obtained_at: float = 0.0
        self._token_expiry: float = 0.0
        self._lock = asyncio.Lock()

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        """
        Perform an HTTP request; return JSON when possible, else text.
        Errors are logged WITHOUT secrets; callers receive raised exceptions.
        """
        headers = kwargs.pop("headers", {})
        headers.setdefault("User-Agent", USER_AGENT)
        headers.setdefault("Accept-Language", ACCEPT_LANGUAGE)
        timeout = kwargs.pop("timeout", aiohttp.ClientTimeout(total=25))

        url = path if path.startswith("http") else API_BASE + path
        _LOGGER.debug("HTTP %s %s", method, url)

        for attempt in range(2):
            try:
                async with self._session.request(method, url, headers=headers, timeout=timeout, **kwargs) as resp:
                    ctype = resp.headers.get("Content-Type", "")
                    body_text: Optional[str]
                    try:
                        body_text = await resp.text()
                    except Exception:
                        body_text = "<no body>"

                    if resp.status >= 400:
                        # Log a compact, redacted error; do not log repr(RequestInfo) which includes headers.
                        _LOGGER.error(
                            "HTTP error %s %s -> %s; body=%s",
                            method,
                            url,
                            resp.status,
                            _redact_bearer(body_text),
                        )
                    else:
                        if API_LOG_PREVIEW:
                            _LOGGER.debug(
                                "HTTP %s -> %s, ctype=%s, body[0:200]=%r",
                                url, resp.status, ctype, (_redact_bearer(body_text) or "")[:200],
                            )
                        else:
                            _LOGGER.debug("HTTP %s -> %s, ctype=%s", url, resp.status, ctype)

                    if resp.status == 401:
                        if attempt == 0:
                            self._access_token = None
                            self._token_expiry = 0.0
                            token = await self._ensure_token()
                            headers["Authorization"] = f"Bearer {token}"
                            continue
                        raise TermoWebAuthError("Unauthorized")
                    if resp.status == 429:
                        raise TermoWebRateLimitError("Rate limited")
                    if resp.status >= 400:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history,
                            status=resp.status, message=body_text, headers=resp.headers
                        )

                    # Try JSON first; fall back to text
                    if "application/json" in ctype or (body_text and body_text[:1] in ("{", "[")):
                        try:
                            return await resp.json(content_type=None)
                        except Exception:
                            return body_text
                    return body_text

            except (TermoWebAuthError, TermoWebRateLimitError):
                raise
            except Exception as e:
                _LOGGER.error("Request %s %s failed (sanitized): %s", method, url, _redact_bearer(str(e)))
                raise
        raise TermoWebAuthError("Unauthorized")

    async def _ensure_token(self) -> str:
        """Ensure a bearer token is present; fetch if missing."""
        if self._access_token and time.time() <= self._token_expiry:
            return self._access_token

        async with self._lock:
            if self._access_token and time.time() <= self._token_expiry:
                return self._access_token

            data = {
                "username": self._username,
                "password": self._password,
                "grant_type": "password",
            }
            headers = {
                "Authorization": f"Basic {BASIC_AUTH_B64}",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
                "Accept-Language": ACCEPT_LANGUAGE,
            }
            url = API_BASE + TOKEN_PATH
            _LOGGER.debug(
                "Token POST %s for user domain=%s",
                url,
                self._username.split("@")[-1] if "@" in self._username else "<no-domain>",
            )
            async with self._session.post(
                url, data=data, headers=headers, timeout=aiohttp.ClientTimeout(total=25)
            ) as resp:
                _LOGGER.debug("Token resp status=%s", resp.status)

                if resp.status in (400, 401):
                    raise TermoWebAuthError(f"Invalid credentials or client auth failed (status {resp.status})")
                if resp.status == 429:
                    raise TermoWebRateLimitError("Rate limited on token endpoint")
                if resp.status >= 400:
                    text = await resp.text()
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history,
                        status=resp.status, message=text, headers=resp.headers
                    )

                js = await resp.json(content_type=None)
                token = js.get("access_token")
                if not token:
                    _LOGGER.error("No access_token in response JSON")
                    raise TermoWebAuthError("No access_token in response")
                self._access_token = token
                self._token_obtained_at = time.time()
                expires_in = js.get("expires_in")
                if isinstance(expires_in, (int, float)):
                    self._token_expiry = self._token_obtained_at + float(expires_in)
                else:
                    self._token_expiry = self._token_obtained_at + 3600
                return token

    async def _authed_headers(self) -> Dict[str, str]:
        token = await self._ensure_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            "Accept-Language": ACCEPT_LANGUAGE,
        }

    # ----------------- Public API -----------------

    async def list_devices(self) -> List[Dict[str, Any]]:
        """Return normalized device list: [{'dev_id', ...}, ...]."""
        headers = await self._authed_headers()
        data = await self._request("GET", DEVS_PATH, headers=headers)

        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            if isinstance(data.get("devs"), list):
                return [d for d in data["devs"] if isinstance(d, dict)]
            if isinstance(data.get("devices"), list):
                return [d for d in data["devices"] if isinstance(d, dict)]
        _LOGGER.debug("Unexpected /devs shape (%s); returning empty list", type(data).__name__)
        return []

    async def device_connected(self, dev_id: str) -> Optional[bool]:
        """Deprecated: connected endpoint often 404s; return None."""
        return None

    async def get_nodes(self, dev_id: str) -> Any:
        """Return raw nodes payload for a device (shape varies by firmware)."""
        headers = await self._authed_headers()
        path = NODES_PATH_FMT.format(dev_id=dev_id)
        return await self._request("GET", path, headers=headers)

    async def get_htr_settings(self, dev_id: str, addr: str | int) -> Any:
        """Return heater settings/state for a node: GET /htr/{addr}/settings."""
        headers = await self._authed_headers()
        path = f"/api/v2/devs/{dev_id}/htr/{addr}/settings"
        return await self._request("GET", path, headers=headers)

    async def set_htr_settings(
        self,
        dev_id: str,
        addr: str | int,
        *,
        mode: Optional[str] = None,          # "auto" | "manual" | "off"
        stemp: Optional[float] = None,       # target setpoint (in current units)
        prog: Optional[List[int]] = None,    # full 168-element weekly program (0=cold,1=night,2=day)
        ptemp: Optional[List[float]] = None, # preset temperatures [cold, night, day] (in current units)
        units: str = "C",
    ) -> Any:
        """
        Update heater settings.

        Supported fields (all optional):

        * ``mode`` – "auto", "manual" or "off". When ``mode == 'manual'`` the server expects
          ``stemp`` to be provided.
        * ``stemp`` – target setpoint for manual mode. A number which will be formatted as a string
          with one decimal before being sent.
        * ``prog`` – list of 168 integers representing the weekly program. Each value must be one
          of ``0`` (cold), ``1`` (night) or ``2`` (day). Monday 00:00 is index 0, Tuesday 00:00
          is index 24, etc. When provided, this list is sent unchanged to the API.
        * ``ptemp`` – list of three floats representing the preset temperatures in the order
          [cold, night, day]. These values are formatted to one decimal and sent as strings.
        * ``units`` – either ``"C"`` or ``"F"``. This field is always included and indicates
          whether the numeric temperature values are in Celsius or Fahrenheit.

        The payload will only include keys for the parameters passed by the caller, to avoid
        overwriting unrelated settings on the device.
        """

        # Always include units
        payload: Dict[str, Any] = {"units": units}

        # Mode
        if mode is not None:
            payload["mode"] = mode

        # Manual setpoint – format as string with one decimal
        if stemp is not None:
            try:
                payload["stemp"] = f"{float(stemp):.1f}"
            except Exception:
                raise ValueError(f"Invalid stemp value: {stemp}")

        # Weekly program – validate length and values
        if prog is not None:
            if not isinstance(prog, list) or len(prog) != 168:
                raise ValueError("prog must be a list of 168 integers (0, 1, or 2)")
            normalized: List[int] = []
            for v in prog:
                try:
                    iv = int(v)
                except Exception:
                    raise ValueError(f"prog contains non-integer value: {v}")
                if iv not in (0, 1, 2):
                    raise ValueError(f"prog values must be 0, 1, or 2; got {iv}")
                normalized.append(iv)
            payload["prog"] = normalized

        # Preset temperatures – validate length and convert to strings
        if ptemp is not None:
            if not isinstance(ptemp, list) or len(ptemp) != 3:
                raise ValueError("ptemp must be a list of three numeric values [cold, night, day]")
            formatted: List[str] = []
            for v in ptemp:
                try:
                    formatted.append(f"{float(v):.1f}")
                except Exception:
                    raise ValueError(f"ptemp contains non-numeric value: {v}")
            payload["ptemp"] = formatted

        headers = await self._authed_headers()
        path = f"/api/v2/devs/{dev_id}/htr/{addr}/settings"
        return await self._request("POST", path, headers=headers, json=payload)

    async def get_pmo_samples(
        self,
        dev_id: str,
        addr: str | int,
        *,
        start: int | None = None,
        end: int | None = None,
    ) -> Any:
        """Return historical samples for a PMO node."""
        if end is None:
            end = int(time.time())
        if start is None:
            start = end - 86400
        headers = await self._authed_headers()
        path = (
            f"/api/v2/devs/{dev_id}/pmo/{addr}/samples?start={int(start)}&end={int(end)}"
        )
        return await self._request("GET", path, headers=headers)
