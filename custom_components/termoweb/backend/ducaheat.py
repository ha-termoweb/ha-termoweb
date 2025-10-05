"""Ducaheat backend implementation and HTTP adapter."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import logging
from typing import Any

from aiohttp import ClientResponseError

from ..api import RESTClient
from ..const import BRAND_DUCAHEAT, WS_NAMESPACE
from ..nodes import NodeDescriptor
from ..ws_client import DucaheatWSClient
from .base import Backend, WsClientProto

_LOGGER = logging.getLogger(__name__)

_DAY_ORDER = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

class DucaheatRequestError(Exception):
    """Raised when the Ducaheat API returns a client error."""

    def __init__(self, *, status: int, path: str, body: str) -> None:
        """Initialise error metadata for logging and diagnostics."""

        super().__init__(f"Ducaheat request failed ({status}) for {path}: {body}")
        self.status = status
        self.path = path
        self.body = body


class DucaheatRESTClient(RESTClient):
    """HTTP adapter that speaks the segmented Ducaheat API."""

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

            mode_payload = None
            status_payload: dict[str, Any] = {}

            if mode is not None:
                mode_norm = str(mode).lower()
                if mode_norm == "heat":
                    mode_norm = "manual"
                mode_payload = {"mode": mode_norm}

            if stemp is not None:
                try:
                    status_payload["stemp"] = self._ensure_temperature(stemp)
                except ValueError as err:
                    raise ValueError(f"Invalid stemp value: {stemp}") from err
                if mode_payload is not None:
                    status_payload["mode"] = mode_payload["mode"]

            if status_payload:
                unit_str = units.upper() if isinstance(units, str) else "C"
                if unit_str not in {"C", "F"}:
                    raise ValueError(f"Invalid units: {units}")
                status_payload["units"] = unit_str
            else:
                status_payload = {}

            prog_payload = self._serialise_prog(prog) if prog is not None else None
            ptemp_payload = (
                self._serialise_prog_temps(ptemp) if ptemp is not None else None
            )

            requires_select = bool(status_payload or prog_payload or ptemp_payload)
            try:
                if requires_select:
                    await self._request(
                        "POST",
                        f"{base}/select",
                        headers=headers,
                        json={"select": True},
                        ignore_statuses=(404,),
                    )

                if status_payload:
                    responses["status"] = await self._request(
                        "POST",
                        f"{base}/status",
                        headers=headers,
                        json=status_payload,
                    )
                elif mode_payload is not None:
                    responses["mode"] = await self._request(
                        "POST",
                        f"{base}/mode",
                        headers=headers,
                        json=mode_payload,
                    )

                if prog_payload is not None:
                    responses["prog"] = await self._request(
                        "POST",
                        f"{base}/prog",
                        headers=headers,
                        json=prog_payload,
                    )

                if ptemp_payload is not None:
                    responses["prog_temps"] = await self._request(
                        "POST",
                        f"{base}/prog_temps",
                        headers=headers,
                        json=ptemp_payload,
                    )

            finally:
                if requires_select:
                    try:
                        await self._request(
                            "POST",
                            f"{base}/select",
                            headers=headers,
                            json={"select": False},
                            ignore_statuses=(404,),
                        )
                    except Exception as err:  # pragma: no cover - diagnostic only
                        _LOGGER.debug(
                            "Failed to release Ducaheat node %s/%s: %s",
                            dev_id,
                            addr,
                            err,
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

        return settings

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
            extra = setup_dict.get("extra_options")
            if isinstance(extra, dict):
                if "boost_time" in extra:
                    flattened["boost_time"] = extra["boost_time"]
                if "boost_temp" in extra:
                    formatted = self._safe_temperature(extra["boost_temp"])
                    if formatted is not None:
                        flattened["boost_temp"] = formatted

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

        mode_payload: dict[str, str] | None = None
        if mode is not None:
            mode_value = str(mode).lower()
            if mode_value == "heat":
                mode_value = "manual"
            mode_payload = {"mode": mode_value}

        status_payload: dict[str, str] = {}
        if stemp is not None:
            try:
                status_payload["stemp"] = self._format_temp(stemp)
            except ValueError as err:
                raise ValueError(f"Invalid stemp value: {stemp}") from err
            status_payload["units"] = self._ensure_units(units)
        elif (
            units is not None
            and mode is None
            and prog is None
            and ptemp is None
        ):
            status_payload["units"] = self._ensure_units(units)
        if status_payload and mode_payload is not None and stemp is not None:
            status_payload["mode"] = mode_payload["mode"]

        if mode_payload is not None:
            responses["mode"] = await self._post_acm_endpoint(
                f"{base}/mode", headers, mode_payload
            )

        if status_payload:
            responses["status"] = await self._post_acm_endpoint(
                f"{base}/status", headers, status_payload
            )

        if prog is not None:
            if len(prog) != 168:
                raise ValueError(
                    f"acm weekly program must contain 168 slots; got {len(prog)}"
                )
            responses["prog"] = await self._post_acm_endpoint(
                f"{base}/prog",
                headers,
                {"prog": self._ensure_prog(prog)},
            )

        if ptemp is not None:
            responses["prog_temps"] = await self._post_acm_endpoint(
                f"{base}/prog_temps",
                headers,
                {"ptemp": self._ensure_ptemp(ptemp)},
            )

        return responses

    async def _post_acm_endpoint(
        self,
        path: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> Any:
        """POST to an ACM endpoint, translating client errors."""

        try:
            return await self._request("POST", path, headers=dict(headers), json=dict(payload))
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

    def _format_temp(self, value: float | str) -> str:
        """Format temperatures using one decimal precision."""

        return self._ensure_temperature(value)

    def _ensure_units(self, value: str) -> str:
        """Validate and normalise accumulator units."""

        unit = str(value).upper()
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
