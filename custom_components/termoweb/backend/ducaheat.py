"""Ducaheat backend implementation and HTTP adapter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
import logging
import typing
from typing import Any

from aiohttp import ClientResponseError

from custom_components.termoweb.backend.base import (
    Backend,
    BackendCapabilities,
    BoostContext,
    WsClientProto,
    fetch_normalised_hourly_samples,
)
from custom_components.termoweb.backend.ducaheat_ws import DucaheatWSClient
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.backend.sanitize import mask_identifier, redact_text
from custom_components.termoweb.boost import validate_boost_minutes
from custom_components.termoweb.codecs.common import (
    format_temperature,
    validate_prog,
    validate_ptemp,
)
from custom_components.termoweb.codecs.ducaheat_codec import (
    decode_settings,
    encode_boost_command,
    encode_program_command,
    extract_prog_days,
)
from custom_components.termoweb.coerce import as_bool, as_percentage
from custom_components.termoweb.const import BRAND_DUCAHEAT, WS_NAMESPACE
from custom_components.termoweb.domain.commands import (
    BaseCommand,
    SetLock,
    SetMode,
    SetPresetTemps,
    SetPriority,
    SetProgram,
    SetSetpoint,
    SetUnits,
    StartBoost,
    StopBoost,
)
from custom_components.termoweb.domain.ids import NodeId, NodeType
from custom_components.termoweb.inventory import Inventory, NodeDescriptor
from custom_components.termoweb.planner.ducaheat_planner import plan_command

_LOGGER = logging.getLogger(__name__)


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
    ) -> Any:
        """Log segmented POST requests before delegating to ``_request``."""

        self._log_segmented_post(
            path=path,
            node_type=node_type,
            dev_id=dev_id,
            addr=addr,
            payload=payload,
        )
        return await self._request(
            "POST", path, headers=dict(headers), json=dict(payload)
        )

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
        """Return node samples (seconds timestamps); thermostats have none."""

        node_type, addr = self._resolve_node_descriptor(node)
        if node_type == "thm":
            _LOGGER.debug(
                "Skipping samples for thermostat node (dev=%s addr=%s)",
                mask_identifier(dev_id),
                mask_identifier(addr),
            )
            return []

        return await super().get_node_samples(
            dev_id,
            (node_type, addr),
            start,
            end,
        )

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

        headers = await self.authed_headers()
        current_prog: dict[str, list[int]] | None = None
        if any(isinstance(command, SetProgram) for command in commands):
            current_prog = await self._get_prog_days(dev_id, node_id, headers)

        write_calls: list[tuple[str, dict[str, Any]]] = []
        for command in commands:
            plan = plan_command(
                dev_id, node_id, command, units=units, current_prog=current_prog
            )
            write_call = plan[0]
            write_calls.append((write_call.path, write_call.json or {}))

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

    async def _get_prog_days(
        self, dev_id: str, node_id: NodeId, headers: Mapping[str, str]
    ) -> dict[str, list[int]]:
        """GET the node's weekly program so a write can echo its slot resolution."""

        node_type = node_id.node_type.value
        path = f"/api/v2/devs/{dev_id}/{node_type}/{node_id.addr}"
        if node_type == "thm":
            path = f"{path}/settings"
        payload = await self._request("GET", path, headers=dict(headers))
        section = payload.get("prog") if isinstance(payload, Mapping) else None
        return extract_prog_days(section)

    async def set_node_settings(  # noqa: C901
        self,
        dev_id: str,
        node: NodeDescriptor,
        *,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
        units: str | None = None,
        boost_time: int | None = None,
        cancel_boost: bool = False,
    ) -> dict[str, Any]:
        """Write heater settings; units are only written alone when passed explicitly."""

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
                units is not None
                and mode_value is None
                and prog is None
                and ptemp is None
            ):
                commands.append(SetUnits(units_value))
            elif mode_value is not None:
                commands.append(SetMode(mode_value))

            if prog is not None:
                commands.append(SetProgram(validate_prog(prog)))
            if ptemp is not None:
                commands.append(SetPresetTemps(validate_ptemp(ptemp)))

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
                    payload["stemp"] = format_temperature(stemp)
                except ValueError as err:
                    raise ValueError(f"Invalid stemp value: {stemp}") from err
                payload["units"] = self._ensure_units(units)
            if prog is not None:
                current_prog = await self._get_prog_days(dev_id, node_id, headers)
                payload["prog"] = encode_program_command(
                    SetProgram(validate_prog(prog)), current=current_prog
                )["prog"]
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
            if mode_value == "boost":
                boost_minutes = validate_boost_minutes(boost_time)

            if stemp is not None:
                formatted_stemp = format_temperature(stemp)
                commands.append(
                    SetSetpoint(
                        formatted_stemp, mode=mode_value, boost_time=boost_minutes
                    )
                )
            elif (
                units is not None
                and mode_value is None
                and prog is None
                and ptemp is None
            ):
                commands.append(SetUnits(units_value))

            if prog is not None:
                commands.append(SetProgram(validate_prog(prog)))
            if ptemp is not None:
                commands.append(SetPresetTemps(validate_ptemp(ptemp)))
            if mode_value is not None and stemp is None:
                commands.append(SetMode(mode_value, boost_time=boost_minutes))

            if cancel_boost:
                commands.append(StopBoost(boost_time=None, stemp=None, units=None))

            return await self._execute_segmented_commands(
                dev_id,
                node_id,
                commands,
                units=units_value,
                use_acm_endpoint=True,
            )

        return await super().set_node_settings(
            dev_id,
            (node_type, addr),
            mode=mode,
            stemp=stemp,
            prog=prog,
            ptemp=ptemp,
            units=self._ensure_units(units),
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
    ) -> None:
        """Copy accumulator charge metadata from ``source`` into ``target``."""

        if not isinstance(source, Mapping):
            return

        charging_value = as_bool(source.get("charging"))
        if charging_value is not None:
            target["charging"] = charging_value

        for key in ("current_charge_per", "target_charge_per"):
            coerced = as_percentage(source.get(key))
            if coerced is None:
                continue
            target[key] = coerced

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
                formatted_temp = format_temperature(stemp)
            except ValueError as err:
                raise ValueError(f"Invalid stemp value: {stemp!r}") from err

        unit_value: str | None = None
        if units is not None:
            unit_value = self._ensure_units(units)

        command_type = StartBoost if boost else StopBoost
        payload = encode_boost_command(
            command_type(boost_time=boost_time, stemp=formatted_temp, units=unit_value)
        )
        if boost:
            _LOGGER.info(
                "ACM boost start dev=%s addr=%s minutes=%s",
                mask_identifier(dev_id),
                mask_identifier(addr_str),
                validate_boost_minutes(boost_time),
            )
        else:
            _LOGGER.info(
                "ACM boost cancel dev=%s addr=%s",
                mask_identifier(dev_id),
                mask_identifier(addr_str),
            )

        return await self._post_acm_endpoint(
            f"/api/v2/devs/{dev_id}/{node_type}/{addr_str}/boost",
            headers,
            payload,
            dev_id=dev_id,
            addr=addr_str,
        )

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

    def _serialise_prog_temps(self, ptemp: list[float]) -> dict[str, str]:
        """Serialise preset temperatures into the API schema."""
        cold, night, day = validate_ptemp(ptemp)
        return {"cold": cold, "night": night, "day": day}


class DucaheatBackend(Backend):
    """Backend wiring for Ducaheat brand accounts."""

    capabilities = BackendCapabilities(
        lock=True,
        priority=True,
        energy_history=True,
        energy=True,
        geo_data=True,
        account_scope="ducaheat",  # Ducaheat and Tevolve accounts are one account
    )

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
        units: str | None = None,
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
