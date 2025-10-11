from __future__ import annotations

import asyncio
from collections.abc import Iterable
import logging
import time
from time import monotonic as time_mod
from typing import Any

import aiohttp

from .backend.sanitize import (
    build_acm_boost_payload,
    mask_identifier,
    redact_text,
    validate_boost_minutes,
)
from .const import (
    ACCEPT_LANGUAGE,
    API_BASE,
    BASIC_AUTH_B64,
    BRAND_API_BASES,
    BRAND_DUCAHEAT,
    BRAND_TERMOWEB,
    DEVS_PATH,
    NODE_SAMPLES_PATH_FMT,
    NODES_PATH_FMT,
    TOKEN_PATH,
    get_brand_requested_with,
    get_brand_user_agent,
)
from .inventory import Node, NodeDescriptor, normalize_node_addr, normalize_node_type

_LOGGER = logging.getLogger(__name__)

# Toggle to preview bodies in debug logs (redacted). Leave False by default.
API_LOG_PREVIEW = False

DUCAHEAT_API_BASE = BRAND_API_BASES[BRAND_DUCAHEAT].rstrip("/")
DUCAHEAT_SERIAL_ID = "15"


class BackendAuthError(Exception):
    """Authentication with TermoWeb failed."""


class BackendRateLimitError(Exception):
    """Server rate-limited the client (HTTP 429)."""


class RESTClient:
    """Thin async client for the TermoWeb cloud (HA-safe)."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        username: str,
        password: str,
        *,
        api_base: str = API_BASE,
        basic_auth_b64: str = BASIC_AUTH_B64,
    ) -> None:
        """Initialise the REST client with authentication context."""
        self._session = session
        self._username = username
        self._password = password
        self._api_base = api_base.rstrip("/") if api_base else API_BASE
        self._basic_auth_b64 = basic_auth_b64 or BASIC_AUTH_B64
        self._access_token: str | None = None
        self._token_obtained_at: float = 0.0
        self._token_expiry: float = 0.0
        self._token_obtained_monotonic: float = 0.0
        self._token_expiry_monotonic: float = 0.0
        self._lock = asyncio.Lock()
        self._is_ducaheat = self._api_base == DUCAHEAT_API_BASE
        self._brand = BRAND_DUCAHEAT if self._is_ducaheat else BRAND_TERMOWEB
        self._user_agent = get_brand_user_agent(self._brand)
        self._requested_with = get_brand_requested_with(self._brand)

    @property
    def api_base(self) -> str:
        """Expose API base for auxiliary clients (e.g. WS)."""

        return self._api_base

    async def _request(
        self,
        method: str,
        path: str,
        *,
        ignore_statuses: Iterable[int] = (),
        **kwargs,
    ) -> Any | None:
        """Perform an HTTP request.

        Return JSON when possible, otherwise text. HTTP statuses listed in
        ``ignore_statuses`` are logged and yield ``None`` instead of raising an
        exception. Errors are logged WITHOUT secrets.
        """
        headers = kwargs.pop("headers", {})
        ignore_statuses = set(ignore_statuses)
        headers.setdefault("User-Agent", self._user_agent)
        headers.setdefault("Accept-Language", ACCEPT_LANGUAGE)
        if self._requested_with:
            headers.setdefault("X-Requested-With", self._requested_with)
        timeout = kwargs.pop("timeout", aiohttp.ClientTimeout(total=25))

        url = path if path.startswith("http") else f"{self._api_base}{path}"
        _LOGGER.debug("HTTP %s %s", method, url)

        for attempt in range(2):
            try:
                async with self._session.request(
                    method, url, headers=headers, timeout=timeout, **kwargs
                ) as resp:
                    ctype = resp.headers.get("Content-Type", "")
                    body_text: str | None
                    try:
                        body_text = await resp.text()
                    except Exception:
                        body_text = "<no body>"

                    if resp.status >= 400:
                        # Log a compact, redacted error; do not log repr(RequestInfo) which includes headers.
                        log_fn = (
                            _LOGGER.debug
                            if resp.status in ignore_statuses
                            else _LOGGER.error
                        )
                        log_fn(
                            "HTTP error %s %s -> %s; body=%s",
                            method,
                            url,
                            resp.status,
                            redact_text(body_text),
                        )
                    elif API_LOG_PREVIEW:
                        _LOGGER.debug(
                            "HTTP %s -> %s, ctype=%s, body[0:200]=%r",
                            url,
                            resp.status,
                            ctype,
                            (redact_text(body_text) or "")[:200],
                        )
                    else:
                        _LOGGER.debug(
                            "HTTP %s -> %s, ctype=%s", url, resp.status, ctype
                        )

                    if resp.status == 401:
                        if attempt == 0:
                            self._access_token = None
                            self._token_expiry = 0.0
                            self._token_expiry_monotonic = 0.0
                            token = await self._ensure_token()
                            headers["Authorization"] = f"Bearer {token}"
                            continue
                        raise BackendAuthError("Unauthorized")
                    if resp.status == 429:
                        raise BackendRateLimitError("Rate limited")
                    if resp.status in ignore_statuses:
                        return None
                    if resp.status >= 400:
                        raise aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=resp.status,
                            message=body_text,
                            headers=resp.headers,
                        )

                    # Try JSON first; fall back to text
                    if "application/json" in ctype or (
                        body_text and body_text[:1] in ("{", "[")
                    ):
                        try:
                            return await resp.json(content_type=None)
                        except Exception:
                            return body_text
                    return body_text

            except (BackendAuthError, BackendRateLimitError):
                raise
            except aiohttp.ClientResponseError as e:
                if e.status not in ignore_statuses:
                    _LOGGER.error(
                        "Request %s %s failed (sanitized): %s",
                        method,
                        url,
                        redact_text(str(e)),
                    )
                raise
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _LOGGER.error(
                    "Request %s %s failed (sanitized): %s",
                    method,
                    url,
                    redact_text(str(e)),
                )
                raise
        raise BackendAuthError("Unauthorized")

    async def _ensure_token(self) -> str:
        """Ensure a bearer token is present; fetch if missing."""
        if self._access_token and time_mod() <= self._token_expiry_monotonic:
            return self._access_token

        async with self._lock:
            if self._access_token and time_mod() <= self._token_expiry_monotonic:
                return self._access_token

            data = {
                "username": self._username,
                "password": self._password,
                "grant_type": "password",
            }
            headers = {
                "Authorization": f"Basic {self._basic_auth_b64}",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Accept": "application/json",
                "User-Agent": self._user_agent,
                "Accept-Language": ACCEPT_LANGUAGE,
            }
            if self._requested_with:
                headers["X-Requested-With"] = self._requested_with
            if self._is_ducaheat:
                headers["X-SerialId"] = DUCAHEAT_SERIAL_ID
            url = f"{self._api_base}{TOKEN_PATH}"
            _LOGGER.debug(
                "Token POST %s for user domain=%s",
                url,
                (
                    self._username.split("@")[-1]
                    if "@" in self._username
                    else "<no-domain>"
                ),
            )
            async with self._session.post(
                url, data=data, headers=headers, timeout=aiohttp.ClientTimeout(total=25)
            ) as resp:
                _LOGGER.debug("Token resp status=%s", resp.status)

                if resp.status in (400, 401):
                    raise BackendAuthError(
                        f"Invalid credentials or client auth failed (status {resp.status})"
                    )
                if resp.status == 429:
                    raise BackendRateLimitError("Rate limited on token endpoint")
                if resp.status >= 400:
                    text = await resp.text()
                    raise aiohttp.ClientResponseError(
                        resp.request_info,
                        resp.history,
                        status=resp.status,
                        message=text,
                        headers=resp.headers,
                    )

                js = await resp.json(content_type=None)
                token = js.get("access_token")
                if not token:
                    _LOGGER.error("No access_token in response JSON")
                    raise BackendAuthError("No access_token in response")
                self._access_token = token
                now_wall = time.time()
                now_mono = time_mod()
                self._token_obtained_at = now_wall
                self._token_obtained_monotonic = now_mono
                expires_in = js.get("expires_in")
                if isinstance(expires_in, (int, float)):
                    ttl = max(float(expires_in), 0.0)
                else:
                    ttl = 3600.0
                self._token_expiry = now_wall + ttl
                self._token_expiry_monotonic = now_mono + ttl
                return token

    @property
    def user_agent(self) -> str:
        """Return the configured User-Agent string."""

        return self._user_agent

    @property
    def requested_with(self) -> str | None:
        """Return the configured X-Requested-With header."""

        return self._requested_with

    # ----------------- Public API -----------------

    async def authed_headers(self) -> dict[str, str]:
        """Return HTTP headers including a valid bearer token."""

        token = await self._ensure_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": self._user_agent,
            "Accept-Language": ACCEPT_LANGUAGE,
        }
        if self._requested_with:
            headers["X-Requested-With"] = self._requested_with
        if self._is_ducaheat:
            headers["X-SerialId"] = DUCAHEAT_SERIAL_ID
        return headers

    async def refresh_token(self) -> None:
        """Refresh the cached bearer token immediately."""

        async with self._lock:
            self._access_token = None
            self._token_obtained_at = 0.0
            self._token_obtained_monotonic = 0.0
            self._token_expiry = 0.0
            self._token_expiry_monotonic = 0.0
        await self._ensure_token()

    async def list_devices(self) -> list[dict[str, Any]]:
        """Return normalized device list: [{'dev_id', ...}, ...]."""
        headers = await self.authed_headers()
        data = await self._request("GET", DEVS_PATH, headers=headers)

        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            if isinstance(data.get("devs"), list):
                return [d for d in data["devs"] if isinstance(d, dict)]
            if isinstance(data.get("devices"), list):
                return [d for d in data["devices"] if isinstance(d, dict)]
        _LOGGER.debug(
            "Unexpected /devs shape (%s); returning empty list", type(data).__name__
        )
        return []

    async def get_nodes(self, dev_id: str) -> Any:
        """Return raw nodes payload for a device (shape varies by firmware)."""
        headers = await self.authed_headers()
        path = NODES_PATH_FMT.format(dev_id=dev_id)
        return await self._request("GET", path, headers=headers)

    async def get_node_settings(self, dev_id: str, node: NodeDescriptor) -> Any:
        """Return settings/state for a node."""

        node_type, addr = self._resolve_node_descriptor(node)
        headers = await self.authed_headers()
        if node_type == "pmo":
            path = f"/api/v2/devs/{dev_id}/{node_type}/{addr}"
        else:
            path = f"/api/v2/devs/{dev_id}/{node_type}/{addr}/settings"
        data = await self._request("GET", path, headers=headers)
        self._log_non_htr_payload(
            node_type=node_type,
            dev_id=dev_id,
            addr=addr,
            stage="GET settings",
            payload=data,
        )
        return data

    async def get_rtc_time(self, dev_id: str) -> dict[str, Any]:
        """Return RTC metadata for a device's manager endpoint."""

        headers = await self.authed_headers()
        path = f"/api/v2/devs/{dev_id}/mgr/rtc/time"
        data = await self._request("GET", path, headers=headers)
        if isinstance(data, dict):
            return data
        _LOGGER.debug(
            "Unexpected RTC time payload for dev %s (%s); returning empty dict",
            dev_id,
            type(data).__name__,
        )
        return {}

    def _ensure_temperature(self, value: Any) -> str:
        """Normalise a numeric temperature to a string with one decimal."""

        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError) as err:
            raise ValueError(f"Invalid temperature value: {value!r}") from err

    def _ensure_prog(self, prog: list[int]) -> list[int]:
        """Validate and normalise a weekly program list."""

        if not isinstance(prog, list) or len(prog) != 168:
            raise ValueError("prog must be a list of 168 integers (0, 1, or 2)")
        normalised: list[int] = []
        for value in prog:
            try:
                ivalue = int(value)
            except (TypeError, ValueError) as err:
                raise ValueError(f"prog contains non-integer value: {value!r}") from err
            if ivalue not in (0, 1, 2):
                raise ValueError(f"prog values must be 0, 1, or 2; got {ivalue}")
            normalised.append(ivalue)
        return normalised

    def _ensure_ptemp(self, ptemp: list[float]) -> list[str]:
        """Validate preset temperatures and return formatted strings."""

        if not isinstance(ptemp, list) or len(ptemp) != 3:
            raise ValueError(
                "ptemp must be a list of three numeric values [cold, night, day]"
            )
        formatted: list[str] = []
        for value in ptemp:
            try:
                formatted.append(self._ensure_temperature(value))
            except ValueError as err:
                raise ValueError(f"ptemp contains non-numeric value: {value}") from err
        return formatted

    def _extract_samples(
        self, data: Any, *, timestamp_divisor: float = 1.0
    ) -> list[dict[str, str | int]]:
        """Normalise heater samples payloads into {"t", "counter"} lists."""

        items: list[Any] | None = None
        if isinstance(data, dict) and isinstance(data.get("samples"), list):
            items = data["samples"]
        elif isinstance(data, list):
            items = data

        if items is None:
            _LOGGER.debug(
                "Unexpected htr samples payload (%s); returning empty list",
                type(data).__name__,
            )
            return []

        samples: list[dict[str, str | int]] = []
        for item in items:
            if not isinstance(item, dict):
                _LOGGER.debug("Unexpected htr sample item: %r", item)
                continue
            timestamp = item.get("t")
            if timestamp is None:
                timestamp = item.get("timestamp")
            if not isinstance(timestamp, (int, float)):
                _LOGGER.debug("Unexpected htr sample shape: %s", item)
                _LOGGER.debug("Unexpected htr sample timestamp: %r", timestamp)
                continue
            counter_value = item.get("counter")
            counter_min = item.get("counter_min")
            counter_max = item.get("counter_max")
            if isinstance(counter_value, dict):
                counter_min = counter_value.get("min", counter_min)
                counter_max = counter_value.get("max", counter_max)
                if "value" in counter_value:
                    counter_value = counter_value.get("value")
                elif "counter" in counter_value:
                    counter_value = counter_value.get("counter")
            if counter_value is None:
                counter_value = item.get("value")
            if counter_value is None:
                counter_value = item.get("energy")
            if counter_value is None:
                _LOGGER.debug("Unexpected htr sample shape: %s", item)
                _LOGGER.debug("Unexpected htr sample counter: %r", item)
                continue
            sample: dict[str, str | int] = {
                "t": int(float(timestamp) / timestamp_divisor),
                "counter": str(counter_value),
            }
            if counter_min is not None:
                sample["counter_min"] = str(counter_min)
            if counter_max is not None:
                sample["counter_max"] = str(counter_max)
            samples.append(sample)
        return samples

    async def set_node_settings(
        self,
        dev_id: str,
        node: NodeDescriptor,
        *,
        mode: str | None = None,  # "auto" | "manual" | "off"
        stemp: float | None = None,  # target setpoint (in current units)
        prog: list[int]
        | None = None,  # full 168-element weekly program (0=cold,1=night,2=day)
        ptemp: list[float]
        | None = None,  # preset temperatures [cold, night, day] (in current units)
        units: str = "C",
        boost_time: int | None = None,
        cancel_boost: bool = False,
    ) -> Any:
        """Update heater settings.

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
        * ``cancel_boost`` – When ``True`` the Ducaheat accumulator adapter will cancel any
          active boost session before applying the update. The base REST client does not
          implement this behaviour and therefore rejects ``True``.

        The payload will only include keys for the parameters passed by the caller, to avoid
        overwriting unrelated settings on the device.
        """

        if boost_time is not None:
            raise ValueError("boost_time is not supported for this node type")
        if cancel_boost:
            raise ValueError("cancel_boost is not supported for this node type")

        node_type, addr = self._resolve_node_descriptor(node)

        # Validate units
        unit_str: str = units.upper()
        if unit_str not in {"C", "F"}:
            raise ValueError(f"Invalid units: {units}")

        # Always include units
        payload: dict[str, Any] = {"units": unit_str}

        # Mode
        if mode is not None:
            mode_str = str(mode).lower()
            if mode_str == "heat":
                mode_str = "manual"
            payload["mode"] = mode_str

        # Manual setpoint – format as string with one decimal
        if stemp is not None:
            try:
                payload["stemp"] = self._ensure_temperature(stemp)
            except ValueError as err:
                raise ValueError(f"Invalid stemp value: {stemp}") from err

        # Weekly program – validate length and values
        if prog is not None:
            payload["prog"] = self._ensure_prog(prog)

        # Preset temperatures – validate length and convert to strings
        if ptemp is not None:
            payload["ptemp"] = self._ensure_ptemp(ptemp)

        headers = await self.authed_headers()
        path = f"/api/v2/devs/{dev_id}/{node_type}/{addr}/settings"
        self._log_non_htr_payload(
            node_type=node_type,
            dev_id=dev_id,
            addr=addr,
            stage="POST settings",  # request payload
            payload=payload,
        )
        response = await self._request("POST", path, headers=headers, json=payload)
        self._log_non_htr_payload(
            node_type=node_type,
            dev_id=dev_id,
            addr=addr,
            stage="POST settings response",
            payload=response,
        )
        return response

    def _build_acm_extra_options_payload(
        self,
        boost_time: int | None,
        boost_temp: float | None,
    ) -> dict[str, Any]:
        """Return a validated payload for accumulator extra options."""

        extra: dict[str, Any] = {}
        minutes = validate_boost_minutes(boost_time)
        if minutes is not None:
            extra["boost_time"] = minutes
        if boost_temp is not None:
            try:
                extra["boost_temp"] = self._ensure_temperature(boost_temp)
            except ValueError as err:
                raise ValueError(f"Invalid boost_temp value: {boost_temp!r}") from err
        if not extra:
            raise ValueError("boost_time or boost_temp must be provided")
        return {"extra_options": extra}

    async def set_acm_extra_options(
        self,
        dev_id: str,
        addr: str | int,
        *,
        boost_time: int | None = None,
        boost_temp: float | None = None,
    ) -> Any:
        """Write default boost settings for an accumulator."""

        node_type, addr_str = self._resolve_node_descriptor(("acm", addr))
        payload = self._build_acm_extra_options_payload(boost_time, boost_temp)

        headers = await self.authed_headers()
        path = f"/api/v2/devs/{dev_id}/{node_type}/{addr_str}/setup"
        self._log_non_htr_payload(
            node_type=node_type,
            dev_id=dev_id,
            addr=addr_str,
            stage="POST boost setup",
            payload=payload,
        )
        response = await self._request("POST", path, headers=headers, json=payload)
        self._log_non_htr_payload(
            node_type=node_type,
            dev_id=dev_id,
            addr=addr_str,
            stage="POST boost setup response",
            payload=response,
        )
        return response

    async def set_acm_boost_state(
        self,
        dev_id: str,
        addr: str | int,
        *,
        boost: bool,
        boost_time: int | None = None,
        stemp: float | None = None,
        units: str | None = None,
    ) -> Any:
        """Start or stop an accumulator boost session."""

        node_type, addr_str = self._resolve_node_descriptor(("acm", addr))
        formatted_temp: str | None = None
        if stemp is not None:
            try:
                formatted_temp = self._ensure_temperature(stemp)
            except ValueError as err:
                raise ValueError(f"Invalid stemp value: {stemp!r}") from err

        unit_value: str | None = None
        if units is not None:
            unit_value = str(units).strip().upper()
            if unit_value not in {"C", "F"}:
                raise ValueError(f"Invalid units: {units!r}")

        payload = build_acm_boost_payload(
            boost,
            boost_time,
            stemp=formatted_temp,
            units=unit_value,
        )

        headers = await self.authed_headers()
        path = f"/api/v2/devs/{dev_id}/{node_type}/{addr_str}/boost"
        self._log_non_htr_payload(
            node_type=node_type,
            dev_id=dev_id,
            addr=addr_str,
            stage="POST boost",
            payload=payload,
        )
        response = await self._request("POST", path, headers=headers, json=payload)
        self._log_non_htr_payload(
            node_type=node_type,
            dev_id=dev_id,
            addr=addr_str,
            stage="POST boost response",
            payload=response,
        )
        return response

    async def get_node_samples(
        self,
        dev_id: str,
        node: NodeDescriptor,
        start: float,
        end: float,
    ) -> list[dict[str, str | int]]:
        """Return heater samples as list of {"t", "counter"} dicts."""
        node_type, addr = self._resolve_node_descriptor(node)
        headers = await self.authed_headers()
        path = NODE_SAMPLES_PATH_FMT.format(
            dev_id=dev_id, node_type=node_type, addr=addr
        )
        params = {"start": int(start), "end": int(end)}
        data = await self._request("GET", path, headers=headers, params=params)
        self._log_non_htr_payload(
            node_type=node_type,
            dev_id=dev_id,
            addr=addr,
            stage="GET samples",
            payload=data,
        )
        return self._extract_samples(data)

    def _resolve_node_descriptor(self, node: NodeDescriptor) -> tuple[str, str]:
        """Return ``(node_type, addr)`` for the provided descriptor."""

        if isinstance(node, Node):
            node_type = node.type
            addr = node.addr
        else:
            if not isinstance(node, tuple) or len(node) != 2:
                msg = f"Unsupported node descriptor: {node!r}"
                raise ValueError(msg)
            node_type, addr = node

        node_type_str = normalize_node_type(node_type)
        if not node_type_str:
            msg = f"Invalid node type extracted from descriptor: {node!r}"
            raise ValueError(msg)

        addr_str = normalize_node_addr(addr)
        if not addr_str:
            msg = f"Invalid node address extracted from descriptor: {node!r}"
            raise ValueError(msg)

        return node_type_str, addr_str

    def _log_non_htr_payload(
        self,
        *,
        node_type: str,
        dev_id: str,
        addr: str,
        stage: str,
        payload: Any,
    ) -> None:
        """Log payloads for unsupported node types at DEBUG level."""

        if node_type == "htr":
            return

        try:
            preview = redact_text(repr(payload))
        except Exception:  # pragma: no cover - defensive
            preview = "<unreprable>"
        if len(preview) > 500:
            preview = f"{preview[:497]}..."
        _LOGGER.debug(
            "%s node %s/%s (%s) payload: %s",
            stage,
            mask_identifier(dev_id),
            mask_identifier(addr),
            node_type,
            preview,
        )

    def normalise_ws_nodes(self, nodes: dict[str, Any]) -> dict[str, Any]:
        """Return websocket node payloads unchanged by default."""

        return nodes
