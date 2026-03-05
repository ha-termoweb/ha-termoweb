"""Ducaheat backend implementation and HTTP adapter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
import logging
import typing
from typing import Any

from aiohttp import ClientResponseError

from custom_components.termoweb.backend.base import (
    Backend,
    BoostContext,
    WsClientProto,
    fetch_normalised_hourly_samples,
)
from custom_components.termoweb.backend.ducaheat_ws import DucaheatWSClient
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.backend.sanitize import (
    build_acm_boost_payload,
    mask_identifier,
    redact_text,
)
from custom_components.termoweb.boost import (
    coerce_boost_bool,
    coerce_int,
    validate_boost_minutes,
)
from custom_components.termoweb.codecs.ducaheat_codec import decode_settings
from custom_components.termoweb.codecs.termoweb_codec import decode_samples
from custom_components.termoweb.const import (
    BRAND_DUCAHEAT,
    NODE_SAMPLES_PATH_FMT,
    WS_NAMESPACE,
)
from custom_components.termoweb.domain.commands import (
    BaseCommand,
    SetLock,
    SetMode,
    SetPresetTemps,
    SetPriority,
    SetProgram,
    SetSetpoint,
    SetUnits,
    StopBoost,
)
from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.inventory import Inventory, NodeDescriptor
from custom_components.termoweb.planner.ducaheat_planner import plan_command

_LOGGER = logging.getLogger(__name__)

_DAY_ORDER = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class DucaheatRequestError(Exception):
    """Raised when the Ducaheat API returns a client error."""

    def __init__(self, *, status: int, path: str, body: str) -> None:
        """Initialise error metadata for logging and diagnostics."""

        clean_body = redact_text(body)
        clean_path = redact_text(path)
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
        payload: Mapping[str, typing.Any],
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
        payload: Mapping[str, typing.Any] | None,
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
            redact_text(path),
            node_type,
            mask_identifier(dev_id),
            mask_identifier(addr),
            body_keys,
        )

    async def get_node_settings(
        self, dev_id: str, node: NodeDescriptor
    ) -> dict[str, Any]:
        """Fetch and normalise node settings for the Ducaheat API."""

        node_type, addr = self._resolve_node_descriptor(node)
        node_id = NodeId(NodeType(node_type), addr)
        headers = await self.authed_headers()
        if node_type == "thm":
            path = f"/api/v2/devs/{dev_id}/thm/{addr}/settings"
            payload = await self._request("GET", path, headers=headers)
            self._log_non_htr_payload(
                node_type=node_type,
                dev_id=dev_id,
                addr=addr,
                stage="GET settings",
                payload=payload,
            )
            return decode_settings(
                payload,
                node_type=node_id.node_type,
            )

        path = f"/api/v2/devs/{dev_id}/{node_type}/{addr}"
        payload = await self._request("GET", path, headers=headers)

        decoded_payload = decode_settings(
            payload,
            node_type=node_id.node_type,
        )
        if node_type != "htr":
            self._log_non_htr_payload(
                node_type=node_type,
                dev_id=dev_id,
                addr=addr,
                stage="GET settings",
                payload=decoded_payload,
            )
        return decoded_payload

    async def get_node_samples(
        self,
        dev_id: str,
        node: NodeDescriptor,
        start: float,
        end: float,
    ) -> list[dict[str, str | int]]:
        """Return heater samples with timestamps normalised to seconds.

        Non-heater nodes delegate to the base implementation. Heater payloads
        automatically detect millisecond timestamps to preserve second
        resolution for downstream consumers.
        """

        node_type, addr = self._resolve_node_descriptor(node)
        if node_type == "thm":
            _LOGGER.debug(
                "Skipping samples for thermostat node (dev=%s addr=%s)",
                mask_identifier(dev_id),
                mask_identifier(addr),
            )
            return []

        if node_type != "htr":
            return await super().get_node_samples(
                dev_id,
                (node_type, addr),
                start,
                end,
            )

        headers = await self.authed_headers()
        path = NODE_SAMPLES_PATH_FMT.format(
            dev_id=dev_id,
            node_type=node_type,
            addr=addr,
        )
        params = {
            "start": int(start),
            "end": int(end),
        }
        data = await self._request("GET", path, headers=headers, params=params)
        self._log_non_htr_payload(
            node_type=node_type,
            dev_id=dev_id,
            addr=addr,
            stage="GET samples",
            payload=data,
        )
        timestamp_divisor = 1.0
        sample_items: list[Any] | None = None
        if isinstance(data, dict) and isinstance(data.get("samples"), list):
            sample_items = data["samples"]
        elif isinstance(data, list):
            sample_items = data

        if sample_items:
            for item in sample_items:
                if not isinstance(item, Mapping):
                    continue
                raw_timestamp = item.get("t")
                if raw_timestamp is None:
                    raw_timestamp = item.get("timestamp")
                if (
                    isinstance(raw_timestamp, (int, float))
                    and raw_timestamp >= 1_000_000_000_000
                ):
                    timestamp_divisor = 1000.0
                    break

        return decode_samples(data, timestamp_divisor=timestamp_divisor, logger=_LOGGER)

    async def _execute_segmented_commands(
        self,
        dev_id: str,
        node_id: NodeId,
        commands: list[BaseCommand],
        *,
        units: str | None = None,
        use_acm_endpoint: bool = False,
    ) -> dict[str, Any]:
        """Apply one or more segmented commands and return merged segment responses."""

        if not commands:
            return {}

        write_calls: list[tuple[str, dict[str, Any]]] = []
        for command in commands:
            plan = plan_command(dev_id, node_id, command, units=units)
            write_call = plan[0]
            write_calls.append((write_call.path, write_call.json or {}))

        headers = await self.authed_headers()
        responses: dict[str, Any] = {}
        for path, payload in write_calls:
            if use_acm_endpoint:
                responses[path.rsplit("/", 1)[-1]] = await self._post_acm_endpoint(
                    path,
                    headers,
                    payload,
                    dev_id=dev_id,
                    addr=node_id.addr,
                )
                continue

            responses[path.rsplit("/", 1)[-1]] = await self._post_segmented(
                path,
                headers=headers,
                payload=payload,
                dev_id=dev_id,
                addr=node_id.addr,
                node_type=node_id.node_type.value,
            )

        return responses

    async def set_node_settings(  # noqa: C901
        self,
        dev_id: str,
        node: NodeDescriptor,
        *,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
        units: str = "C",
        boost_time: int | None = None,
        cancel_boost: bool = False,
    ) -> dict[str, Any]:
        """Write heater settings using the segmented endpoints."""

        node_type, addr = self._resolve_node_descriptor(node)
        node_id = NodeId(NodeType(node_type), addr)

        if node_type == "htr":
            commands: list[BaseCommand] = []
            mode_value: str | None = None
            if mode is not None:
                mode_value = str(mode).strip().lower()
                if mode_value == "heat":
                    mode_value = "manual"

            units_value = self._ensure_units(units)
            if stemp is not None:
                commands.append(SetSetpoint(stemp, mode=mode_value))
            elif (
                units_value is not None
                and mode_value is None
                and prog is None
                and ptemp is None
            ):
                commands.append(SetUnits(units_value))
            elif mode_value is not None:
                commands.append(SetMode(mode_value))

            if prog is not None:
                commands.append(SetProgram(self._ensure_prog(prog)))
            if ptemp is not None:
                commands.append(SetPresetTemps(self._ensure_ptemp(ptemp)))

            return await self._execute_segmented_commands(
                dev_id,
                node_id,
                commands,
                units=units_value,
            )

        if node_type == "thm":
            headers = await self.authed_headers()
            path = f"/api/v2/devs/{dev_id}/thm/{addr}/settings"
            payload: dict[str, Any] = {}

            if mode is not None:
                payload["mode"] = str(mode).strip().lower()
            if stemp is not None:
                try:
                    payload["stemp"] = self._ensure_temperature(stemp)
                except ValueError as err:
                    raise ValueError(f"Invalid stemp value: {stemp}") from err
                payload["units"] = self._ensure_units(units)
            if prog is not None:
                payload["prog"] = self._serialise_prog(prog)
            if ptemp is not None:
                payload["ptemp"] = self._serialise_prog_temps(ptemp)

            if not payload:
                return {}

            try:
                return await self._request(
                    "PATCH",
                    path,
                    headers=headers,
                    json=payload,
                )
            except ClientResponseError as err:
                if err.status not in {404, 405}:
                    raise
                return await self._request(
                    "POST",
                    path,
                    headers=headers,
                    json=payload,
                )

        if node_type == "acm":
            mode_value: str | None = None
            if mode is not None:
                mode_value = str(mode).strip().lower()

            units_value = self._ensure_units(units)
            commands: list[BaseCommand] = []
            boost_minutes: int | None = None
            if boost_time is not None and mode_value != "boost":
                raise ValueError("boost_time is only supported when mode is 'boost'")
            if mode_value == "boost" and stemp is None:
                boost_minutes = validate_boost_minutes(boost_time)

            if stemp is not None:
                formatted_stemp = self._ensure_temperature(stemp)
                commands.append(SetSetpoint(formatted_stemp, mode=mode_value))
            elif (
                units_value is not None
                and mode_value is None
                and prog is None
                and ptemp is None
            ):
                commands.append(SetUnits(units_value))

            if prog is not None:
                commands.append(SetProgram(self._ensure_prog(prog)))
            if ptemp is not None:
                commands.append(SetPresetTemps(self._ensure_ptemp(ptemp)))
            if mode_value is not None and stemp is None:
                commands.append(SetMode(mode_value, boost_time=boost_minutes))

            if cancel_boost:
                commands.append(StopBoost(boost_time=None, stemp=None, units=None))

            responses = await self._execute_segmented_commands(
                dev_id,
                node_id,
                commands,
                units=units_value,
                use_acm_endpoint=True,
            )

            if mode_value is not None or cancel_boost:
                minutes_param: int | None
                if mode_value == "boost":
                    minutes_param = boost_minutes
                else:
                    minutes_param = 0
                metadata: dict[str, Any] | None = None
                try:
                    metadata = await self._collect_boost_metadata(
                        dev_id,
                        addr,
                        boost_active=mode_value == "boost",
                        minutes=minutes_param,
                    )
                except Exception as err:  # noqa: BLE001 - defensive logging
                    _LOGGER.debug(
                        "Boost metadata collection failed dev=%s addr=%s: %s",
                        mask_identifier(dev_id),
                        mask_identifier(addr),
                        err,
                        exc_info=err,
                    )
                fallback = False
                boost_state: dict[str, Any] | None = None
                if isinstance(metadata, dict):
                    fallback = bool(metadata.pop("_fallback", False))
                    boost_state = metadata
                elif metadata is not None:
                    responses["boost_state"] = metadata

                if fallback and mode_value != "boost" and "status" not in responses:
                    boost_flag = bool((boost_state or {}).get("boost_active"))
                    responses["status_refresh"] = await self._post_acm_endpoint(
                        f"/api/v2/devs/{dev_id}/{node_type}/{addr}/boost",
                        await self.authed_headers(),
                        {"boost": boost_flag},
                        dev_id=dev_id,
                        addr=addr,
                    )
                if boost_state is not None:
                    responses["boost_state"] = boost_state

            return responses

        return await super().set_node_settings(
            dev_id,
            (node_type, addr),
            mode=mode,
            stemp=stemp,
            prog=prog,
            ptemp=ptemp,
            units=units,
            cancel_boost=cancel_boost,
        )

    async def set_node_lock(
        self,
        dev_id: str,
        node: NodeDescriptor,
        *,
        lock: bool,
    ) -> Any:
        """Toggle child lock via the segmented lock endpoint."""

        node_type, addr = self._resolve_node_descriptor(node)
        node_id = NodeId(NodeType(node_type), addr)
        return await self._execute_segmented_commands(
            dev_id,
            node_id,
            [SetLock(lock)],
            use_acm_endpoint=node_type == "acm",
        )

    async def set_node_priority(
        self,
        dev_id: str,
        node: NodeDescriptor,
        *,
        priority: int,
    ) -> Any:
        """Set the priority level via the segmented setup endpoint."""

        node_type, addr = self._resolve_node_descriptor(node)
        node_id = NodeId(NodeType(node_type), addr)
        return await self._execute_segmented_commands(
            dev_id,
            node_id,
            [SetPriority(priority)],
        )

    def normalise_ws_nodes(self, nodes: dict[str, Any]) -> dict[str, Any]:
        """Normalise websocket payloads and merge accumulator charge metadata."""

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

                    try:
                        node_id = NodeId(NodeType(node_type), str(addr))
                    except ValueError:
                        addr_map[addr] = payload
                        continue

                    addr_map[addr] = decode_settings(
                        payload,
                        node_type=node_id.node_type,
                    )

                section_map[section] = addr_map

            if (
                node_type == "acm"
                and isinstance(section_map.get("settings"), dict)
                and isinstance(section_map.get("status"), Mapping)
            ):
                settings_map = section_map["settings"]
                status_map = section_map["status"]
                for addr, status_payload in status_map.items():
                    if not isinstance(status_payload, Mapping):
                        continue
                    target_settings = settings_map.get(addr)
                    if not isinstance(target_settings, dict):
                        continue
                    self._merge_accumulator_charge_metadata(
                        target_settings, status_payload
                    )

            normalised[node_type] = section_map

        return normalised

    def _merge_accumulator_charge_metadata(
        self,
        target: dict[str, Any],
        source: Mapping[str, typing.Any] | None,
        *,
        prefer_existing: bool = False,
    ) -> None:
        """Copy accumulator charge metadata from ``source`` into ``target``."""

        if not isinstance(source, Mapping):
            return

        def _should_assign(key: str) -> bool:
            if not prefer_existing:
                return True
            if key not in target:
                return True
            return target[key] is None

        charging_value = coerce_boost_bool(source.get("charging"))
        if charging_value is not None and _should_assign("charging"):
            target["charging"] = charging_value

        for key in ("current_charge_per", "target_charge_per"):
            if not _should_assign(key):
                continue
            coerced = coerce_int(source.get(key))
            if coerced is None:
                continue
            target[key] = max(0, min(100, coerced))

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

    async def _collect_boost_metadata(
        self,
        dev_id: str,
        addr: str,
        *,
        boost_active: bool,
        minutes: int | None,
    ) -> dict[str, Any]:
        """Capture RTC metadata to derive boost end timing."""

        metadata: dict[str, Any] = {"boost_active": boost_active}
        rtc_payload: Mapping[str, typing.Any] | None = None
        try:
            rtc_payload = await self.get_rtc_time(dev_id)
        except Exception as err:  # noqa: BLE001 - defensive logging
            _LOGGER.debug(
                "RTC fetch failed after boost write dev=%s addr=%s: %s",
                mask_identifier(dev_id),
                mask_identifier(addr),
                err,
                exc_info=err,
            )
            metadata["_fallback"] = True
            if boost_active and minutes is not None:
                metadata["boost_minutes_delta"] = minutes
            else:
                metadata.setdefault("boost_minutes_delta", 0)
            metadata.setdefault("boost_end_day", None)
            metadata.setdefault("boost_end_min", None)
            metadata.setdefault("boost_end_timestamp", None)
            return metadata

        rtc_dt = self._rtc_payload_to_datetime(rtc_payload)
        if rtc_dt is None:
            _LOGGER.debug(
                "RTC payload invalid after boost write dev=%s addr=%s: %s",
                mask_identifier(dev_id),
                mask_identifier(addr),
                rtc_payload,
            )
            metadata["_fallback"] = True
            if boost_active and minutes is not None:
                metadata["boost_minutes_delta"] = minutes
            else:
                metadata.setdefault("boost_minutes_delta", 0)
            metadata.setdefault("boost_end_day", None)
            metadata.setdefault("boost_end_min", None)
            metadata.setdefault("boost_end_timestamp", None)
            return metadata

        if boost_active and minutes is not None:
            end_dt = rtc_dt + timedelta(minutes=minutes)
            day_of_year = end_dt.timetuple().tm_yday
            minute_of_day = end_dt.hour * 60 + end_dt.minute
            metadata.update(
                {
                    "boost_end_day": day_of_year,
                    "boost_end_min": minute_of_day,
                    "boost_minutes_delta": minutes,
                    "boost_end_timestamp": end_dt.isoformat(),
                }
            )
        else:
            metadata.update(
                {
                    "boost_end_day": None,
                    "boost_end_min": None,
                    "boost_minutes_delta": 0 if minutes is None else max(0, minutes),
                    "boost_end_timestamp": None,
                }
            )

        return metadata

    def _rtc_payload_to_datetime(
        self, payload: Mapping[str, typing.Any] | None
    ) -> datetime | None:
        """Convert an RTC payload into a ``datetime`` instance."""

        if not isinstance(payload, Mapping):
            return None
        try:
            year = int(payload.get("y"))
            month = int(payload.get("n"))
            day = int(payload.get("d"))
            hour = int(payload.get("h", 0))
            minute = int(payload.get("m", 0))
            second = int(payload.get("s", 0))
        except (TypeError, ValueError):
            return None
        try:
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            return None

    async def _post_acm_endpoint(
        self,
        path: str,
        headers: Mapping[str, str],
        payload: Mapping[str, typing.Any],
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

    async def _select_segmented_node(
        self,
        *,
        dev_id: str,
        node_type: str,
        addr: str,
        headers: Mapping[str, str],
        select: bool,
    ) -> Any:
        """Toggle node identification cues like flashing/backlight for the target node."""

        payload = {"select": bool(select)}
        path = f"/api/v2/devs/{dev_id}/{node_type}/{addr}/select"
        try:
            return await self._post_segmented(
                path,
                headers=headers,
                payload=payload,
                dev_id=dev_id,
                addr=addr,
                node_type=node_type,
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

    async def set_node_display_select(
        self,
        dev_id: str,
        node: NodeDescriptor,
        *,
        select: bool,
    ) -> Any:
        """Toggle display-identify cues for a Ducaheat node."""

        node_type, addr_str = self._resolve_node_descriptor(node)
        headers = await self.authed_headers()
        return await self._select_segmented_node(
            dev_id=dev_id,
            node_type=node_type,
            addr=addr_str,
            headers=headers,
            select=select,
        )

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
        headers = await self.authed_headers()
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
        stemp: float | None = None,
        units: str | None = None,
    ) -> Any:
        """Toggle an accumulator boost session via segmented endpoints."""

        node_type, addr_str = self._resolve_node_descriptor(("acm", addr))
        headers = await self.authed_headers()
        formatted_temp: str | None = None
        if stemp is not None:
            try:
                formatted_temp = self._ensure_temperature(stemp)
            except ValueError as err:
                raise ValueError(f"Invalid stemp value: {stemp!r}") from err

        unit_value: str | None = None
        if units is not None:
            unit_value = self._ensure_units(units)

        payload = build_acm_boost_payload(
            boost,
            boost_time,
            stemp=formatted_temp,
            units=unit_value,
        )
        if boost:
            minutes = validate_boost_minutes(boost_time)
            _LOGGER.info(
                "ACM boost start dev=%s addr=%s minutes=%s",
                mask_identifier(dev_id),
                mask_identifier(addr_str),
                minutes,
            )
        else:
            minutes = 0
            _LOGGER.info(
                "ACM boost cancel dev=%s addr=%s",
                mask_identifier(dev_id),
                mask_identifier(addr_str),
            )

        response = await self._post_acm_endpoint(
            f"/api/v2/devs/{dev_id}/{node_type}/{addr_str}/boost",
            headers,
            payload,
            dev_id=dev_id,
            addr=addr_str,
        )

        metadata = await self._collect_boost_metadata(
            dev_id,
            addr_str,
            boost_active=boost,
            minutes=minutes,
        )
        if isinstance(response, dict):
            if metadata:
                response.setdefault("boost_state", metadata)
            return response

        result: dict[str, Any] = {"response": response}
        if metadata:
            result["boost_state"] = metadata
        return result

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
        cold, night, day = self._ensure_ptemp(ptemp)
        return {"cold": cold, "night": night, "day": day}

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

    def _should_cancel_boost(self, context: BoostContext | None) -> bool:
        """Return True when accumulator updates should cancel boost."""

        if context is None:
            return False
        if context.active is not None:
            return bool(context.active)
        if context.mode is not None:
            return context.mode.strip().lower() == "boost"
        return False

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
        boost_context: BoostContext | None = None,
    ) -> Any:
        """Update node settings while applying Ducaheat boost heuristics."""

        node_type, _addr = self._resolve_node_descriptor(node)
        cancel_boost = node_type == "acm" and self._should_cancel_boost(boost_context)

        await self.client.set_node_settings(
            dev_id,
            node,
            mode=mode,
            stemp=stemp,
            prog=prog,
            ptemp=ptemp,
            units=units,
            cancel_boost=cancel_boost,
        )

    def create_ws_client(
        self,
        hass: Any,
        entry_id: str,
        dev_id: str,
        coordinator: Any,
        *,
        inventory: Inventory | None = None,
    ) -> WsClientProto:
        """Instantiate the unified websocket client for Ducaheat."""

        return DucaheatWSClient(
            hass,
            entry_id=entry_id,
            dev_id=dev_id,
            api_client=self.client,
            coordinator=coordinator,
            namespace=WS_NAMESPACE,
            inventory=inventory,
        )

    async def fetch_hourly_samples(
        self,
        dev_id: str,
        nodes: Iterable[tuple[str, str]],
        start_local: datetime,
        end_local: datetime,
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Return hourly samples for ``nodes`` using the segmented API."""

        return await fetch_normalised_hourly_samples(
            client=self.client,
            dev_id=dev_id,
            nodes=nodes,
            start_local=start_local,
            end_local=end_local,
            logger=_LOGGER,
            log_prefix="ducaheat",
        )


__all__ = [
    "BRAND_DUCAHEAT",
    "DucaheatBackend",
    "DucaheatRESTClient",
    "DucaheatRequestError",
]
