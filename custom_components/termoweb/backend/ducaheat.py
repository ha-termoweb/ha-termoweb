"""Ducaheat backend implementation and HTTP adapter."""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from ..api import RESTClient
from ..const import BRAND_DUCAHEAT
from ..ws_client_v2 import DucaheatWSClient
from .base import Backend, WsClientProto

_LOGGER = logging.getLogger(__name__)

_DAY_ORDER = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class DucaheatRESTClient(RESTClient):
    """HTTP adapter that speaks the segmented Ducaheat API."""

    async def get_htr_settings(self, dev_id: str, addr: str | int) -> dict[str, Any]:
        """Fetch and normalise heater settings for the Ducaheat API."""

        headers = await self._authed_headers()
        path = f"/api/v2/devs/{dev_id}/htr/{addr}"
        payload = await self._request("GET", path, headers=headers)
        return self._normalise_settings(payload)

    async def set_htr_settings(
        self,
        dev_id: str,
        addr: str | int,
        *,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
        units: str = "C",
    ) -> dict[str, Any]:
        """Write heater settings using the segmented endpoints."""

        headers = await self._authed_headers()
        base = f"/api/v2/devs/{dev_id}/htr/{addr}"
        responses: dict[str, Any] = {}

        mode_payload = None
        status_payload: dict[str, Any] = {}

        if mode is not None:
            mode_norm = str(mode).lower()
            if mode_norm == "heat":
                mode_norm = "manual"
            status_payload["mode"] = mode_norm
            mode_payload = {"mode": mode_norm}

        if stemp is not None:
            try:
                status_payload["stemp"] = self._ensure_temperature(stemp)
            except ValueError as err:
                raise ValueError(f"Invalid stemp value: {stemp}") from err

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

    async def get_htr_samples(
        self,
        dev_id: str,
        addr: str | int,
        start: float,
        stop: float,
    ) -> list[dict[str, str | int]]:
        """Return samples converting epoch seconds to milliseconds for the API."""

        headers = await self._authed_headers()
        path = f"/api/v2/devs/{dev_id}/htr/{addr}/samples"
        params = {"start": int(start * 1000), "end": int(stop * 1000)}
        data = await self._request("GET", path, headers=headers, params=params)
        return self._extract_samples(data, timestamp_divisor=1000)

    def _normalise_settings(self, payload: Any) -> dict[str, Any]:
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

        for extra_key in (
            "boost_active",
            "boost_remaining",
            "boost_end",
            "lock",
            "lock_active",
            "max_power",
        ):
            if extra_key in status_dict:
                flattened[extra_key] = status_dict[extra_key]

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
        return flattened

    def _normalise_prog(self, data: Any) -> list[int] | None:
        if isinstance(data, list):
            try:
                return self._ensure_prog(data)
            except ValueError:
                return None

        if not isinstance(data, dict):
            return None

        days_section = None
        if isinstance(data.get("days"), dict):
            days_section = data["days"]
        else:
            candidate = {k: v for k, v in data.items() if k in _DAY_ORDER}
            if candidate:
                days_section = candidate

        if not isinstance(days_section, dict):
            return None

        values: list[int] = []
        for day in _DAY_ORDER:
            entry = days_section.get(day)
            slots = None
            if isinstance(entry, dict):
                if isinstance(entry.get("slots"), list):
                    slots = entry["slots"]
                elif isinstance(entry.get("values"), list):
                    slots = entry["values"]
            elif isinstance(entry, list):
                slots = entry

            if slots is None:
                slots = [0] * 24
            try:
                day_values = [int(v) for v in slots[:24]]
            except (TypeError, ValueError):
                return None
            if len(day_values) < 24:
                day_values.extend([0] * (24 - len(day_values)))
            values.extend(day_values[:24])

        return values if len(values) == 168 else None

    def _normalise_prog_temps(self, data: Any) -> list[str] | None:
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

    def _serialise_prog(self, prog: list[int]) -> dict[str, Any]:
        normalised = self._ensure_prog(prog)
        days: dict[str, Any] = {}
        for idx, day in enumerate(_DAY_ORDER):
            start = idx * 24
            days[day] = {"slots": normalised[start : start + 24]}
        return {"days": days}

    def _serialise_prog_temps(self, ptemp: list[float]) -> dict[str, str]:
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
        """Instantiate the Socket.IO v2 websocket client."""

        return DucaheatWSClient(
            hass,
            entry_id=entry_id,
            dev_id=dev_id,
            api_client=self.client,
            coordinator=coordinator,
        )


__all__ = [
    "BRAND_DUCAHEAT",
    "DucaheatBackend",
    "DucaheatRESTClient",
]
