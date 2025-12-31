"""Domain runtime state objects."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .ids import NodeId, NodeType


def _copy_sequence(value: Any) -> list[Any] | None:
    """Return a shallow copy of a sequence when applicable."""

    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return None


def _copy_mapping(value: Any) -> dict[str, Any] | None:
    """Return a shallow copy of a mapping when applicable."""

    if isinstance(value, Mapping):
        return dict(value)
    return None


def _coerce_number(value: Any) -> float | int | None:
    """Return ``value`` as a number when possible."""

    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class HeaterState:
    """Runtime state for a heater node."""

    mode: str | None = None
    stemp: Any | None = None
    mtemp: Any | None = None
    temp: Any | None = None
    prog: list[Any] | None = None
    ptemp: list[Any] | None = None
    units: str | None = None
    state: str | None = None
    max_power: float | int | None = None
    batt_level: int | None = None
    status: dict[str, Any] | None = None
    capabilities: dict[str, Any] | None = None

    def to_legacy(self) -> dict[str, Any]:
        """Convert the state into the legacy coordinator payload shape."""

        payload: dict[str, Any] = {}
        if self.mode is not None:
            payload["mode"] = self.mode
        if self.stemp is not None:
            payload["stemp"] = self.stemp
        if self.mtemp is not None:
            payload["mtemp"] = self.mtemp
        if self.temp is not None:
            payload["temp"] = self.temp
        if self.prog is not None:
            payload["prog"] = list(self.prog)
        if self.ptemp is not None:
            payload["ptemp"] = list(self.ptemp)
        if self.units is not None:
            payload["units"] = self.units
        if self.state is not None:
            payload["state"] = self.state
        if self.max_power is not None:
            payload["max_power"] = self.max_power
        if self.batt_level is not None:
            payload["batt_level"] = self.batt_level
        if self.status is not None:
            payload["status"] = dict(self.status)
        if self.capabilities is not None:
            payload["capabilities"] = dict(self.capabilities)
        return payload


@dataclass(slots=True)
class AccumulatorState(HeaterState):
    """Runtime state for an accumulator node."""

    charge_level: float | int | None = None
    boost: bool | None = None
    charging: bool | None = None
    current_charge_per: int | float | None = None
    target_charge_per: int | float | None = None
    boost_active: bool | None = None
    boost_end: Any | None = None
    boost_remaining: float | int | None = None
    boost_time: int | float | None = None
    boost_temp: Any | None = None
    boost_end_day: int | None = None
    boost_end_min: int | None = None
    boost_end_datetime: Any | None = None
    boost_minutes_delta: int | None = None

    def to_legacy(self) -> dict[str, Any]:
        """Convert the accumulator state into the legacy payload shape."""

        payload = HeaterState.to_legacy(self)
        if self.charge_level is not None:
            payload["charge_level"] = self.charge_level
        if self.boost is not None:
            payload["boost"] = self.boost
        if self.charging is not None:
            payload["charging"] = self.charging
        if self.current_charge_per is not None:
            payload["current_charge_per"] = self.current_charge_per
        if self.target_charge_per is not None:
            payload["target_charge_per"] = self.target_charge_per
        if self.boost_active is not None:
            payload["boost_active"] = self.boost_active
        if self.boost_end is not None:
            payload["boost_end"] = _copy_mapping(self.boost_end) or self.boost_end
        if self.boost_remaining is not None:
            payload["boost_remaining"] = self.boost_remaining
        if self.boost_time is not None:
            payload["boost_time"] = self.boost_time
        if self.boost_temp is not None:
            payload["boost_temp"] = self.boost_temp
        if self.boost_end_day is not None:
            payload["boost_end_day"] = self.boost_end_day
        if self.boost_end_min is not None:
            payload["boost_end_min"] = self.boost_end_min
        if self.boost_end_datetime is not None:
            payload["boost_end_datetime"] = self.boost_end_datetime
        if self.boost_minutes_delta is not None:
            payload["boost_minutes_delta"] = self.boost_minutes_delta
        return payload


@dataclass(slots=True)
class ThermostatState(HeaterState):
    """Runtime state for a thermostat node."""


@dataclass(slots=True)
class PowerMonitorState:
    """Runtime state for a power monitor node."""

    status: dict[str, Any] | None = None
    capabilities: dict[str, Any] | None = None

    def to_legacy(self) -> dict[str, Any]:
        """Convert the power monitor state into the legacy payload shape."""

        payload: dict[str, Any] = {}
        if self.status is not None:
            payload["status"] = dict(self.status)
        if self.capabilities is not None:
            payload["capabilities"] = dict(self.capabilities)
        return payload


DomainState = HeaterState | AccumulatorState | ThermostatState | PowerMonitorState


@dataclass(slots=True)
class NodeDelta:
    """Base delta object for node updates."""

    node_id: NodeId

    @property
    def payload(self) -> Mapping[str, Any]:
        """Return the payload carried by the delta."""

        return {}


@dataclass(slots=True)
class NodeSettingsDelta(NodeDelta):
    """Settings delta for a node."""

    changes: Mapping[str, Any]

    @property
    def payload(self) -> Mapping[str, Any]:
        """Return the mapping of changed fields."""

        return self.changes


@dataclass(slots=True)
class NodeStatusDelta(NodeDelta):
    """Status delta for a node."""

    status: Mapping[str, Any]

    @property
    def payload(self) -> Mapping[str, Any]:
        """Return the status mapping payload."""

        return {"status": self.status}


@dataclass(slots=True)
class NodeSamplesDelta(NodeDelta):
    """Samples delta placeholder for future use."""

    samples: Mapping[str, Any]

    @property
    def payload(self) -> Mapping[str, Any]:
        """Return the samples mapping payload."""

        return {"samples": self.samples}


def _populate_heater_state(
    state: HeaterState,
    payload: Mapping[str, Any],
) -> HeaterState:
    """Populate base heater fields on ``state`` from ``payload``."""

    if "mode" in payload:
        raw_mode = payload.get("mode")
        if raw_mode is None:
            state.mode = None
        elif isinstance(raw_mode, str):
            state.mode = raw_mode
        else:
            state.mode = str(raw_mode)
    if "stemp" in payload:
        state.stemp = payload.get("stemp")
    if "mtemp" in payload:
        state.mtemp = payload.get("mtemp")
    if "temp" in payload:
        state.temp = payload.get("temp")
    if "prog" in payload:
        state.prog = _copy_sequence(payload.get("prog"))
    if "ptemp" in payload:
        state.ptemp = _copy_sequence(payload.get("ptemp"))
    if "units" in payload:
        raw_units = payload.get("units")
        if raw_units is None:
            state.units = None
        elif isinstance(raw_units, str):
            state.units = raw_units
        else:
            state.units = str(raw_units)
    if "state" in payload:
        raw_state = payload.get("state")
        if raw_state is None:
            state.state = None
        elif isinstance(raw_state, str):
            state.state = raw_state
        else:
            state.state = str(raw_state)
    if "max_power" in payload:
        state.max_power = _coerce_number(payload.get("max_power"))
    if "batt_level" in payload:
        try:
            state.batt_level = int(payload.get("batt_level"))
        except (TypeError, ValueError):
            state.batt_level = None
    if "status" in payload:
        state.status = _copy_mapping(payload.get("status"))
    if "capabilities" in payload:
        state.capabilities = _copy_mapping(payload.get("capabilities"))
    return state


def _build_heater_state(payload: Mapping[str, Any]) -> HeaterState:
    """Construct a heater state instance from ``payload``."""

    return _populate_heater_state(HeaterState(), payload)


def _build_accumulator_state(payload: Mapping[str, Any]) -> AccumulatorState:
    """Construct an accumulator state instance from ``payload``."""

    state = _populate_heater_state(AccumulatorState(), payload)
    if "charge_level" in payload:
        state.charge_level = _coerce_number(payload.get("charge_level"))
    if "boost" in payload:
        boost_value = payload.get("boost")
        if isinstance(boost_value, bool):
            state.boost = boost_value
        elif isinstance(boost_value, (int, float)):
            state.boost = bool(boost_value)
        else:
            state.boost = None
    if "charging" in payload:
        charging_value = payload.get("charging")
        if isinstance(charging_value, bool):
            state.charging = charging_value
        elif isinstance(charging_value, (int, float)):
            state.charging = bool(charging_value)
        else:
            state.charging = None
    if "current_charge_per" in payload:
        state.current_charge_per = _coerce_number(payload.get("current_charge_per"))
    if "target_charge_per" in payload:
        state.target_charge_per = _coerce_number(payload.get("target_charge_per"))
    if "boost_active" in payload:
        state.boost_active = payload.get("boost_active")
    if "boost_end" in payload:
        end_payload = payload.get("boost_end")
        state.boost_end = _copy_mapping(end_payload) or end_payload
    if "boost_remaining" in payload:
        state.boost_remaining = _coerce_number(payload.get("boost_remaining"))
    if "boost_time" in payload:
        state.boost_time = payload.get("boost_time")
    if "boost_temp" in payload:
        state.boost_temp = payload.get("boost_temp")
    if "boost_end_day" in payload:
        state.boost_end_day = payload.get("boost_end_day")
    if "boost_end_min" in payload:
        state.boost_end_min = payload.get("boost_end_min")
    if "boost_end_datetime" in payload:
        state.boost_end_datetime = payload.get("boost_end_datetime")
    if "boost_minutes_delta" in payload:
        state.boost_minutes_delta = payload.get("boost_minutes_delta")
    return state


def _build_thermostat_state(payload: Mapping[str, Any]) -> ThermostatState:
    """Construct a thermostat state instance from ``payload``."""

    return _populate_heater_state(ThermostatState(), payload)


def _build_power_monitor_state(payload: Mapping[str, Any]) -> PowerMonitorState:
    """Construct a power monitor state instance from ``payload``."""

    return PowerMonitorState(
        status=_copy_mapping(payload.get("status")),
        capabilities=_copy_mapping(payload.get("capabilities")),
    )


def _build_state(node_type: NodeType, payload: Mapping[str, Any]) -> DomainState:
    """Return a domain state object for ``node_type`` and ``payload``."""

    if node_type is NodeType.ACCUMULATOR:
        return _build_accumulator_state(payload)
    if node_type is NodeType.THERMOSTAT:
        return _build_thermostat_state(payload)
    if node_type is NodeType.POWER_MONITOR:
        return _build_power_monitor_state(payload)
    return _build_heater_state(payload)


def _merge_state(state: DomainState, payload: Mapping[str, Any]) -> DomainState:
    """Merge ``payload`` fields into ``state`` without altering the type."""

    if isinstance(state, AccumulatorState):
        return _build_accumulator_state({**state.to_legacy(), **payload})
    if isinstance(state, ThermostatState):
        return _build_thermostat_state({**state.to_legacy(), **payload})
    if isinstance(state, PowerMonitorState):
        return _build_power_monitor_state({**state.to_legacy(), **payload})
    return _build_heater_state({**state.to_legacy(), **payload})


class DomainStateStore:
    """Canonical state store keyed by immutable inventory identifiers."""

    def __init__(self, nodes: Iterable[NodeId]) -> None:
        """Initialise the store for a set of allowable nodes."""

        self._states: dict[NodeId, DomainState] = {}
        self._allowed: dict[NodeId, NodeId] = {}
        self._addresses_by_type: dict[NodeType, set[str]] = {}
        self.reset_nodes(nodes)

    def reset_nodes(self, nodes: Iterable[NodeId]) -> None:
        """Reset the allowed node list and prune any stale state."""

        allowed: dict[NodeId, NodeId] = {}
        addresses: dict[NodeType, set[str]] = {}
        for node in nodes:
            if not isinstance(node, NodeId):
                continue
            allowed[node] = node
            bucket = addresses.setdefault(node.node_type, set())
            bucket.add(node.addr)

        self._allowed = allowed
        self._addresses_by_type = addresses
        self._states = {
            node_id: state
            for node_id, state in self._states.items()
            if node_id in allowed
        }

    def _resolve_node_id(self, node_type: NodeType | str, addr: Any) -> NodeId | None:
        """Return a canonical NodeId when permitted by the inventory."""

        normalized_type = _normalize_node_type(node_type)
        if normalized_type is None:
            return None

        try:
            candidate = NodeId(normalized_type, addr)
        except ValueError:
            return None
        return self._allowed.get(candidate)

    def _apply_payload(
        self, node_id: NodeId, payload: Mapping[str, Any], *, replace: bool
    ) -> None:
        """Apply ``payload`` to ``node_id`` using replace semantics when requested."""

        if not isinstance(payload, Mapping):
            return

        if replace or node_id not in self._states:
            self._states[node_id] = _build_state(node_id.node_type, payload)
            return

        existing = self._states.get(node_id)
        if existing is None:
            self._states[node_id] = _build_state(node_id.node_type, payload)
            return

        self._states[node_id] = _merge_state(existing, payload)

    def apply_full_snapshot(
        self,
        node_type: NodeType | str,
        addr: Any,
        decoded_settings: Mapping[str, Any] | None,
    ) -> None:
        """Store a complete settings snapshot for ``(node_type, addr)``."""

        if not isinstance(decoded_settings, Mapping):
            return

        node_id = self._resolve_node_id(node_type, addr)
        if node_id is None:
            return

        self._apply_payload(node_id, decoded_settings, replace=True)

    def apply_patch(
        self,
        node_type: NodeType | str,
        addr: Any,
        delta: Mapping[str, Any] | None,
    ) -> None:
        """Merge partial updates for ``(node_type, addr)`` into the store."""

        if not isinstance(delta, Mapping):
            return

        node_id = self._resolve_node_id(node_type, addr)
        if node_id is None:
            return

        self._apply_payload(node_id, delta, replace=False)

    def apply_delta(self, delta: NodeDelta | None) -> None:
        """Apply a typed domain delta to the store."""

        if not isinstance(delta, NodeDelta):
            return

        node_id = self._allowed.get(delta.node_id)
        if node_id is None:
            return

        self._apply_payload(node_id, delta.payload, replace=False)

    def get_state(self, node_type: NodeType | str, addr: Any) -> DomainState | None:
        """Return the stored state for ``(node_type, addr)`` when known."""

        node_id = self._resolve_node_id(node_type, addr)
        if node_id is None:
            return None
        return self._states.get(node_id)

    @property
    def addresses_by_type(self) -> dict[str, tuple[str, ...]]:
        """Return known addresses grouped by node type."""

        return {
            node_type.value: tuple(sorted(addrs))
            for node_type, addrs in self._addresses_by_type.items()
        }

    @property
    def known_types(self) -> tuple[str, ...]:
        """Return the set of node types represented in the store."""

        return tuple(sorted(self.addresses_by_type))

    def iter_states(self) -> Iterator[tuple[NodeId, DomainState]]:
        """Yield stored ``(NodeId, DomainState)`` pairs."""

        yield from self._states.items()

    def legacy_view(self) -> dict[str, dict[str, Any]]:
        """Return a legacy settings mapping grouped by type and address."""

        legacy: dict[str, dict[str, Any]] = {}
        for node_type, addrs in self._addresses_by_type.items():
            bucket = legacy.setdefault(node_type.value, {})
            for addr in sorted(addrs):
                try:
                    node_id = NodeId(node_type, addr)
                except ValueError:
                    continue
                state = self._states.get(node_id)
                if state is None:
                    continue
                bucket[addr] = state.to_legacy()
        return legacy


def _normalize_node_type(node_type: NodeType | str) -> NodeType | None:
    """Return a canonical ``NodeType`` when possible."""

    if isinstance(node_type, NodeType):
        return node_type

    try:
        return NodeType(str(node_type))
    except ValueError:
        try:
            return NodeType(str(node_type).lower())
        except ValueError:
            return None


def build_state_from_payload(
    node_type: NodeType | str, payload: Mapping[str, Any]
) -> DomainState | None:
    """Return a domain state instance derived from a legacy payload."""

    normalized_type = _normalize_node_type(node_type)
    if normalized_type is None:
        return None
    if not isinstance(payload, Mapping):
        return None
    return _build_state(normalized_type, payload)
