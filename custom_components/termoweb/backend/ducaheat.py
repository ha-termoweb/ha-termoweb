"""Ducaheat backend implementation and HTTP adapter."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import logging
import re
from typing import Any

from aiohttp import ClientResponseError

from ..api import RESTClient
from ..const import BRAND_DUCAHEAT, WS_NAMESPACE
from ..nodes import NodeDescriptor
from .base import Backend, WsClientProto
from .ducaheat_ws import DucaheatWSClient

_LOGGER = logging.getLogger(__name__)

_DAY_ORDER = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)
_TOKEN_QUERY_RE = re.compile(r"(?i)(token|refresh_token|access_token)=([^&\s]+)")


def _redact_log_value(value: str) -> str:
    """Redact bearer tokens and email addresses from log strings."""

    if not value:
        return ""
    redacted = _BEARER_RE.sub("Bearer ***", value)
    redacted = _EMAIL_RE.sub("***@***", redacted)
    return _TOKEN_QUERY_RE.sub(lambda match: f"{match.group(1)}=***", redacted)

class DucaheatRequestError(Exception):
    """Raised when the Ducaheat API returns a client error."""

    def __init__(self, *, status: int, path: str, body: str) -> None:
        """Initialise error metadata for logging and diagnostics."""

        clean_body = _redact_log_value(body)
        clean_path = _redact_log_value(path)
        super().__init__(
            f"Ducaheat request failed ({status}) for {clean_path}: {clean_body}"
        )
        self.status = status
        self.path = clean_path
        self.body = clean_body


class DucaheatRESTClient(RESTClient):
    """HTTP adapter that speaks the segmented Ducaheat API."""

    async def _post_segmented(
        self,
        path: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        dev_id: str,
        addr: str,
        node_type: str,
        ignore_statuses: Iterable[int] | None = None,
    ) -> Any:
        """Log segmented POST requests before delegating to ``_request``."""

        self._log_segmented_post(
            path=path,
            node_type=node_type,
            dev_id=dev_id,
            addr=addr,
            payload=payload,
        )
        request_kwargs: dict[str, Any] = {
            "headers": dict(headers),
            "json": dict(payload),
        }
        if ignore_statuses:
            request_kwargs["ignore_statuses"] = tuple(ignore_statuses)
        return await self._request("POST", path, **request_kwargs)

    def _log_segmented_post(
        self,
        *,
        path: str,
        node_type: str,
        dev_id: str,
        addr: str,
        payload: Mapping[str, Any] | None,
    ) -> None:
        """Emit a debug log for segmented POST calls with sanitized metadata."""

        if not _LOGGER.isEnabledFor(logging.DEBUG):
            return
        if isinstance(payload, Mapping):
            body_keys = tuple(sorted(str(key) for key in payload))
        elif payload is None:
            body_keys = ()
        else:
            body_keys = ("<non-mapping>",)
        _LOGGER.debug(
            "POST %s (node_type=%s dev=%s addr=%s) body_keys=%s",
            _redact_log_value(path),
            node_type,
            _redact_log_value(dev_id),
            _redact_log_value(addr),
            body_keys,
        )

    async def get_node_settings(
        self, dev_id: str, node: NodeDescriptor
    ) -> dict[str, Any]:
        """Fetch and normalise node settings for the Ducaheat API."""

        node_type, addr = self._resolve_node_descriptor(node)
        headers = await self._authed_headers()
        path = f"/api/v2/devs/{dev_id}/{node_type}/{addr}"
        payload = await self._request("GET", path, headers=headers)

        if node_type in {"htr", "acm"}:
            if node_type != "htr":
                self._log_non_htr_payload(
                    node_type=node_type,
                    dev_id=dev_id,
                    addr=addr,
                    stage="GET settings",
                    payload=payload,
                )
            return self._normalise_settings(payload, node_type=node_type)

        self._log_non_htr_payload(
            node_type=node_type,
            dev_id=dev_id,
            addr=addr,
            stage="GET settings",
            payload=payload,
        )
        return payload

    async def set_node_settings(
        self,
        dev_id: str,
        node: NodeDescriptor,
        *,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
        units: str = "C",
    ) -> dict[str, Any]:
        """Write heater settings using the segmented endpoints."""

        node_type, addr = self._resolve_node_descriptor(node)
        if node_type == "htr":
            headers = await self._authed_headers()
            base = f"/api/v2/devs/{dev_id}/htr/{addr}"
            responses: dict[str, Any] = {}

            mode_value: str | None = None
            if mode is not None:
                mode_value = str(mode).lower()
                if mode_value == "heat":
                    mode_value = "manual"

            status_payload: dict[str, Any] = {}
            status_includes_mode = False
            if stemp is not None:
                try:
                    status_payload["stemp"] = self._ensure_temperature(stemp)
                except ValueError as err:
                    raise ValueError(f"Invalid stemp value: {stemp}") from err
                status_payload["units"] = self._ensure_units(units)
                if mode_value is not None:
                    status_payload["mode"] = mode_value
                    status_includes_mode = True
            elif (
                units is not None
                and mode is None
                and prog is None
                and ptemp is None
            ):
                status_payload["units"] = self._ensure_units(units)

            segment_plan: list[tuple[str, dict[str, Any]]] = []
            if status_payload:
                segment_plan.append(("status", status_payload))
            if mode_value is not None and not status_includes_mode:
                segment_plan.append(("mode", {"mode": mode_value}))
            if prog is not None:
                segment_plan.append(("prog", self._serialise_prog(prog)))
            if ptemp is not None:
                segment_plan.append(("prog_temps", self._serialise_prog_temps(ptemp)))

            for name, payload in segment_plan:
                responses[name] = await self._post_segmented(
                    f"{base}/{name}",
                    headers=headers,
                    payload=payload,
                    dev_id=dev_id,
                    addr=addr,
                    node_type=node_type,
                )
            return responses

        if node_type == "acm":
            return await self._set_acm_settings(
                dev_id,
                addr,
                mode=mode,
                stemp=stemp,
                prog=prog,
                ptemp=ptemp,
                units=units,
            )

        return await super().set_node_settings(
            dev_id,
            (node_type, addr),
            mode=mode,
            stemp=stemp,
            prog=prog,
            ptemp=ptemp,
            units=units,
        )

    async def get_node_samples(
        self,
        dev_id: str,
        node: NodeDescriptor,
        start: float,
        stop: float,
    ) -> list[dict[str, str | int]]:
        """Return samples converting epoch seconds to milliseconds for the API."""

        node_type, addr = self._resolve_node_descriptor(node)
        if node_type != "htr":
            return await super().get_node_samples(dev_id, (node_type, addr), start, stop)

        headers = await self._authed_headers()
        path = f"/api/v2/devs/{dev_id}/htr/{addr}/samples"
        params = {"start": int(start * 1000), "end": int(stop * 1000)}
        data = await self._request("GET", path, headers=headers, params=params)
        return self._extract_samples(data, timestamp_divisor=1000)

    def normalise_ws_nodes(self, nodes: dict[str, Any]) -> dict[str, Any]:
        """Normalise websocket nodes payloads for Ducaheat specifics."""

        if not isinstance(nodes, dict):
            return nodes

        normalised: dict[str, Any] = {}
        for node_type, sections in nodes.items():
            if not isinstance(sections, Mapping):
                normalised[node_type] = sections
                continue

            section_map: dict[str, Any] = {}
            for section, by_addr in sections.items():
                if section != "settings" or not isinstance(by_addr, Mapping):
                    section_map[section] = by_addr
                    continue

                addr_map: dict[str, Any] = {}
                for addr, payload in by_addr.items():
                    if not isinstance(payload, Mapping):
                        addr_map[addr] = payload
                        continue

                    addr_map[addr] = self._normalise_ws_settings(payload)

                section_map[section] = addr_map

            normalised[node_type] = section_map

        return normalised

    def _normalise_ws_settings(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Normalise half-hourly program data within websocket settings."""

        settings: dict[str, Any] = dict(payload)

        prog = settings.get("prog")
        normalised_prog = self._normalise_prog(prog)
        if normalised_prog is not None:
            settings["prog"] = normalised_prog

        self._merge_boost_metadata(settings, settings)

        return settings

    def _merge_boost_metadata(
        self,
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

        if "boost_end" in source:
            boost_end = source["boost_end"]
            _assign("boost_end", boost_end, allow_none=True)
            if isinstance(boost_end, Mapping):
                _assign("boost_end_day", boost_end.get("day"), prefer=True)
                _assign("boost_end_min", boost_end.get("minute"), prefer=True)

    def _normalise_settings(
        self, payload: Any, *, node_type: str = "htr"
    ) -> dict[str, Any]:
        """Flatten the vendor payload into HA-friendly heater settings."""
        if not isinstance(payload, dict):
            return {}

        status = payload.get("status")
        status_dict = status if isinstance(status, dict) else {}
        setup = payload.get("setup")
        setup_dict = setup if isinstance(setup, dict) else {}
        prog = payload.get("prog")
        prog_temps = payload.get("prog_temps")

        flattened: dict[str, Any] = {}

        mode = status_dict.get("mode")
        if isinstance(mode, str):
            flattened["mode"] = mode.lower()

        state = (
            status_dict.get("state")
            or status_dict.get("heating_state")
            or status_dict.get("output_state")
        )
        if isinstance(state, str):
            flattened["state"] = state

        units = status_dict.get("units")
        if isinstance(units, str):
            flattened["units"] = units.upper()

        stemp = status_dict.get("stemp")
        if stemp is None:
            stemp = (
                status_dict.get("set_temp")
                or status_dict.get("target")
                or status_dict.get("setpoint")
            )
        if stemp is not None:
            formatted = self._safe_temperature(stemp)
            if formatted is not None:
                flattened["stemp"] = formatted

        mtemp = (
            status_dict.get("mtemp")
            or status_dict.get("temp")
            or status_dict.get("ambient")
            or status_dict.get("room_temp")
        )
        if mtemp is not None:
            formatted = self._safe_temperature(mtemp)
            if formatted is not None:
                flattened["mtemp"] = formatted

        include_boost = node_type == "acm"
        for extra_key in (
            "boost_active",
            "boost_remaining",
            "boost_end",
            "lock",
            "lock_active",
            "max_power",
        ):
            if not include_boost and extra_key.startswith("boost"):
                continue
            if extra_key in status_dict:
                flattened[extra_key] = status_dict[extra_key]

        if include_boost:
            self._merge_boost_metadata(flattened, status_dict)

            extra = setup_dict.get("extra_options")
            if isinstance(extra, dict):
                if "boost_time" in extra:
                    flattened["boost_time"] = extra["boost_time"]
                if "boost_temp" in extra:
                    formatted = self._safe_temperature(extra["boost_temp"])
                    if formatted is not None:
                        flattened["boost_temp"] = formatted

                self._merge_boost_metadata(flattened, extra, prefer_existing=True)

            self._merge_boost_metadata(flattened, setup_dict, prefer_existing=True)

            if "boost_temp" not in flattened:
                boost_temp = status_dict.get("boost_temp")
                if boost_temp is not None:
                    formatted = self._safe_temperature(boost_temp)
                    if formatted is not None:
                        flattened["boost_temp"] = formatted

            if "boost_time" not in flattened:
                boost_time = status_dict.get("boost_time")
                if boost_time is not None:
                    flattened["boost_time"] = boost_time

        prog_list = self._normalise_prog(prog)
        if prog_list is not None:
            flattened["prog"] = prog_list

        ptemp_list = self._normalise_prog_temps(prog_temps)
        if ptemp_list is not None:
            flattened["ptemp"] = ptemp_list

        for key in ("addr", "name", "type", "brand", "model"):
            if key in payload:
                flattened[key] = payload[key]

        flattened["raw"] = deepcopy(payload)

        if node_type == "acm":
            capabilities = self._normalise_acm_capabilities(payload)
            if capabilities:
                flattened["capabilities"] = capabilities

        return flattened

    def _normalise_acm_capabilities(self, payload: Any) -> dict[str, Any]:
        """Merge accumulator capability dictionaries into a single mapping."""

        merged: dict[str, Any] = {}

        def _merge(target: dict[str, Any], source: dict[str, Any]) -> None:
            """Recursively merge capability dictionaries."""
            for key, value in source.items():
                if (
                    isinstance(value, dict)
                    and isinstance(target.get(key), dict)
                ):
                    _merge(target[key], value)
                else:
                    target[key] = value

        for container in (
            payload,
            payload.get("status") if isinstance(payload, dict) else None,
            payload.get("setup") if isinstance(payload, dict) else None,
        ):
            if isinstance(container, dict):
                candidate = container.get("capabilities")
                if isinstance(candidate, dict):
                    _merge(merged, candidate)

        return merged

    def _normalise_prog(self, data: Any) -> list[int] | None:
        """Convert vendor programme payloads into a 168-slot list."""
        if isinstance(data, list):
            try:
                return self._ensure_prog(data)
            except ValueError:
                return None

        if not isinstance(data, dict):
            return None

        days_section: dict[str, Any] | None = None
        if isinstance(data.get("days"), dict):
            days_section = data["days"]
        else:
            candidate = {k: v for k, v in data.items() if k in _DAY_ORDER}
            if candidate:
                days_section = candidate

        if days_section is None and isinstance(data.get("prog"), dict):
            days_section = data["prog"]

        if not isinstance(days_section, dict):
            return None

        def _coerce_slots(entry: Any) -> list[int] | None:
            """Convert a day's slots into a 24-value list."""

            candidate = entry
            if isinstance(candidate, dict):
                slots_candidate = candidate.get("slots")
                values_candidate = candidate.get("values")
                if isinstance(slots_candidate, list):
                    candidate = slots_candidate
                else:
                    candidate = values_candidate

            if candidate is None:
                return None

            if not isinstance(candidate, list):
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
        for idx, day in enumerate(_DAY_ORDER):
            entry = days_section.get(day)
            if entry is None:
                entry = days_section.get(str(idx))
            if entry is None:
                entry = days_section.get(idx)

            if entry is None:
                day_values = [0] * 24
            else:
                day_values = _coerce_slots(entry)
                if day_values is None:
                    return None
            values.extend(day_values)

        return values if len(values) == 168 else None

    def _normalise_prog_temps(self, data: Any) -> list[str] | None:
        """Convert preset temperature payloads into stringified list."""
        if not isinstance(data, dict):
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
            safe = self._safe_temperature(value)
            if safe is None:
                formatted.append(str(value))
            else:
                formatted.append(safe)
        return formatted

    async def _set_acm_settings(
        self,
        dev_id: str,
        addr: str,
        *,
        mode: str | None,
        stemp: float | None,
        prog: list[int] | None,
        ptemp: list[float] | None,
        units: str,
    ) -> dict[str, Any]:
        """Apply segmented writes for accumulator nodes."""

        headers = await self._authed_headers()
        base = f"/api/v2/devs/{dev_id}/acm/{addr}"
        responses: dict[str, Any] = {}

        mode_value: str | None = None
        if mode is not None:
            mode_value = str(mode).lower()

        status_payload: dict[str, str] = {}
        status_includes_mode = False
        if stemp is not None:
            try:
                status_payload["stemp"] = self._format_temp(stemp)
            except ValueError as err:
                raise ValueError(f"Invalid stemp value: {stemp}") from err
            status_payload["units"] = self._ensure_units(units)
            if mode_value is not None:
                status_payload["mode"] = mode_value
                status_includes_mode = True
        elif (
            units is not None
            and mode is None
            and prog is None
            and ptemp is None
        ):
            status_payload["units"] = self._ensure_units(units)

        segment_plan: list[tuple[str, dict[str, Any]]] = []
        if status_payload:
            segment_plan.append(("status", status_payload))
        if mode_value is not None and not status_includes_mode:
            segment_plan.append(("mode", {"mode": mode_value}))
        if prog is not None:
            if len(prog) != 168:
                raise ValueError(
                    f"acm weekly program must contain 168 slots; got {len(prog)}"
                )
            segment_plan.append(("prog", {"prog": self._ensure_prog(prog)}))
        if ptemp is not None:
            segment_plan.append(("prog_temps", {"ptemp": self._ensure_ptemp(ptemp)}))

        for name, payload in segment_plan:
            responses[name] = await self._post_acm_endpoint(
                f"{base}/{name}",
                headers,
                payload,
                dev_id=dev_id,
                addr=addr,
            )

        return responses

    async def _post_acm_endpoint(
        self,
        path: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        dev_id: str | None = None,
        addr: str | None = None,
    ) -> Any:
        """POST to an ACM endpoint, translating client errors."""

        try:
            return await self._post_segmented(
                path,
                headers=headers,
                payload=payload,
                dev_id=dev_id or "<unknown>",
                addr=addr or "<unknown>",
                node_type="acm",
            )
        except ClientResponseError as err:
            if 400 <= err.status < 500:
                message = getattr(err, "message", None)
                if not message and err.args:
                    message = str(err.args[0])
                raise DucaheatRequestError(
                    status=err.status,
                    path=path,
                    body=str(message or ""),
                ) from err
            raise

    async def set_acm_extra_options(
        self,
        dev_id: str,
        addr: str | int,
        *,
        boost_time: int | None = None,
        boost_temp: float | None = None,
    ) -> Any:
        """Write default boost configuration using segmented endpoints."""

        node_type, addr_str = self._resolve_node_descriptor(("acm", addr))
        headers = await self._authed_headers()
        payload = self._build_acm_extra_options_payload(boost_time, boost_temp)
        return await self._post_acm_endpoint(
            f"/api/v2/devs/{dev_id}/{node_type}/{addr_str}/setup",
            headers,
            payload,
            dev_id=dev_id,
            addr=addr_str,
        )

    async def set_acm_boost_state(
        self,
        dev_id: str,
        addr: str | int,
        *,
        boost: bool,
        boost_time: int | None = None,
    ) -> Any:
        """Toggle an accumulator boost session via segmented endpoints."""

        node_type, addr_str = self._resolve_node_descriptor(("acm", addr))
        headers = await self._authed_headers()
        payload = self._build_acm_boost_payload(boost, boost_time)
        return await self._post_acm_endpoint(
            f"/api/v2/devs/{dev_id}/{node_type}/{addr_str}/status",
            headers,
            payload,
            dev_id=dev_id,
            addr=addr_str,
        )

    def _format_temp(self, value: float | str) -> str:
        """Format temperatures using one decimal precision."""

        return self._ensure_temperature(value)

    def _ensure_units(self, value: str | None) -> str:
        """Validate and normalise temperature units."""

        if value is None:
            unit = "C"
        else:
            unit = str(value).strip().upper()
        if not unit:
            unit = "C"
        if unit not in {"C", "F"}:
            raise ValueError(f"Invalid units: {value}")
        return unit

    def _serialise_prog(self, prog: list[int]) -> dict[str, Any]:
        """Serialise the 168-slot programme back to API structure."""
        normalised = self._ensure_prog(prog)
        half_hour: dict[str, Any] = {}
        for idx in range(len(_DAY_ORDER)):
            start = idx * 24
            hourly = normalised[start : start + 24]
            slots: list[int] = []
            for value in hourly:
                slots.extend([value, value])
            half_hour[str(idx)] = slots
        return {"prog": half_hour}

    def _serialise_prog_temps(self, ptemp: list[float]) -> dict[str, str]:
        """Serialise preset temperatures into the API schema."""
        antifrost, eco, comfort = self._ensure_ptemp(ptemp)
        return {"antifrost": antifrost, "eco": eco, "comfort": comfort}

    def _safe_temperature(self, value: Any) -> str | None:
        """Defensively format inbound temperature values."""

        if value is None:
            return None
        try:
            return self._ensure_temperature(value)
        except ValueError:
            if isinstance(value, str):
                cleaned = value.strip()
                return cleaned or None
            return None


class DucaheatBackend(Backend):
    """Backend wiring for Ducaheat brand accounts."""

    def create_ws_client(
        self,
        hass: Any,
        entry_id: str,
        dev_id: str,
        coordinator: Any,
    ) -> WsClientProto:
        """Instantiate the unified websocket client for Ducaheat."""

        return DucaheatWSClient(
            hass,
            entry_id=entry_id,
            dev_id=dev_id,
            api_client=self.client,
            coordinator=coordinator,
            namespace=WS_NAMESPACE,
        )


__all__ = [
    "BRAND_DUCAHEAT",
    "DucaheatBackend",
    "DucaheatRESTClient",
    "DucaheatRequestError",
]
