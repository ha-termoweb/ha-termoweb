"""Domain runtime state objects."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, fields
import typing
from typing import Any

from .energy import EnergySnapshot
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


@dataclass(slots=True)
class AccumulatorState(HeaterState):
    """Runtime state for an accumulator node."""

    charge_level: float | int | None = None
    charging: bool | None = None
    current_charge_per: int | float | None = None
    target_charge_per: int | float | None = None
    boost_active: bool | None = None
    boost_remaining: float | int | None = None
    boost_time: int | float | None = None
    boost_temp: Any | None = None
    boost_end_day: int | None = None
    boost_end_min: int | None = None
    boost_end_datetime: Any | None = None
    boost_minutes_delta: int | None = None


@dataclass(slots=True)
class ThermostatState(HeaterState):
    """Runtime state for a thermostat node."""


@dataclass(slots=True)
class PowerMonitorState:
    """Runtime state for a power monitor node."""

    power: float | int | None = None
    voltage: float | int | None = None
    current: float | int | None = None
    energy: float | int | None = None


DomainState = HeaterState | AccumulatorState | ThermostatState | PowerMonitorState


@dataclass(slots=True)
class GatewayConnectionState:
    """Runtime connection state for a gateway."""

    status: str | None = None
    connected: bool = False
    last_event_at: float | None = None
    healthy_since: float | None = None
    healthy_minutes: float | None = None
    last_payload_at: float | None = None
    last_heartbeat_at: float | None = None
    payload_stale: bool | None = None
    payload_stale_after: float | None = None
    idle_restart_pending: bool | None = None


_SETTING_FIELD_NAMES: frozenset[str] = frozenset(
    field.name
    for cls in (HeaterState, AccumulatorState, ThermostatState, PowerMonitorState)
    for field in fields(cls)
)


def canonicalize_settings_payload(payload: Mapping[str, typing.Any]) -> dict[str, Any]:
    """Return canonical setting fields derived from ``payload``."""

    if not isinstance(payload, Mapping):
        return {}

    canonical: dict[str, Any] = {}

    def _merge(source: Mapping[str, typing.Any]) -> None:
        for key, value in source.items():
            if key not in _SETTING_FIELD_NAMES:
                continue
            cloned = _copy_mapping(value)
            if cloned is None:
                cloned = _copy_sequence(value) or value
            canonical.setdefault(key, cloned)

    _merge(payload)
    status = payload.get("status")
    if isinstance(status, Mapping):
        _merge(status)
    return canonical


@dataclass(slots=True)
class NodeDelta:
    """Base delta object for node updates."""

    node_id: NodeId

    @property
    def payload(self) -> Mapping[str, typing.Any]:
        """Return the payload carried by the delta."""

        return {}


@dataclass(slots=True)
class NodeSettingsDelta(NodeDelta):
    """Settings delta for a node."""

    changes: Mapping[str, typing.Any]

    @property
    def payload(self) -> Mapping[str, typing.Any]:
        """Return the mapping of changed fields."""

        return self.changes


@dataclass(slots=True)
class NodeStatusDelta(NodeDelta):
    """Status delta for a node."""

    status: Mapping[str, typing.Any]

    @property
    def payload(self) -> Mapping[str, typing.Any]:
        """Return the status mapping payload."""

        return canonicalize_settings_payload({"status": self.status})


@dataclass(slots=True)
class NodeSamplesDelta(NodeDelta):
    """Samples delta placeholder for future use."""

    samples: Mapping[str, typing.Any]

    @property
    def payload(self) -> Mapping[str, typing.Any]:
        """Return the samples mapping payload."""

        return {"samples": self.samples}


def _populate_heater_state(
    state: HeaterState,
    payload: Mapping[str, typing.Any],
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
    return state


def _build_heater_state(payload: Mapping[str, typing.Any]) -> HeaterState:
    """Construct a heater state instance from ``payload``."""

    return _populate_heater_state(HeaterState(), payload)


def _populate_accumulator_fields(
    state: AccumulatorState, payload: Mapping[str, typing.Any]
) -> AccumulatorState:
    """Populate accumulator-specific fields on ``state``."""

    _populate_heater_state(state, payload)
    if "charge_level" in payload:
        state.charge_level = _coerce_number(payload.get("charge_level"))
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


def _build_accumulator_state(payload: Mapping[str, typing.Any]) -> AccumulatorState:
    """Construct an accumulator state instance from ``payload``."""

    state = _populate_heater_state(AccumulatorState(), payload)
    return _populate_accumulator_fields(state, payload)


def _build_thermostat_state(payload: Mapping[str, typing.Any]) -> ThermostatState:
    """Construct a thermostat state instance from ``payload``."""

    return _populate_heater_state(ThermostatState(), payload)


def _build_power_monitor_state(payload: Mapping[str, typing.Any]) -> PowerMonitorState:
    """Construct a power monitor state instance from ``payload``."""

    state = PowerMonitorState()
    return _populate_power_monitor_state(state, payload)


def _populate_power_monitor_state(
    state: PowerMonitorState, payload: Mapping[str, typing.Any]
) -> PowerMonitorState:
    """Populate power monitor fields on ``state``."""

    if "power" in payload:
        state.power = _coerce_number(payload.get("power"))
    if "voltage" in payload:
        state.voltage = _coerce_number(payload.get("voltage"))
    if "current" in payload:
        state.current = _coerce_number(payload.get("current"))
    if "energy" in payload:
        state.energy = _coerce_number(payload.get("energy"))
    status = payload.get("status")
    if isinstance(status, Mapping):
        if state.power is None and "power" in status:
            state.power = _coerce_number(status.get("power"))
        if state.voltage is None and "voltage" in status:
            state.voltage = _coerce_number(status.get("voltage"))
        if state.current is None and "current" in status:
            state.current = _coerce_number(status.get("current"))
        if state.energy is None and "energy" in status:
            state.energy = _coerce_number(status.get("energy"))
    return state


def _build_state(node_type: NodeType, payload: Mapping[str, typing.Any]) -> DomainState:
    """Return a domain state object for ``node_type`` and ``payload``."""

    if node_type is NodeType.ACCUMULATOR:
        return _build_accumulator_state(payload)
    if node_type is NodeType.THERMOSTAT:
        return _build_thermostat_state(payload)
    if node_type is NodeType.POWER_MONITOR:
        return _build_power_monitor_state(payload)
    return _build_heater_state(payload)


def _merge_state(state: DomainState, payload: Mapping[str, typing.Any]) -> DomainState:
    """Merge ``payload`` fields into ``state`` without altering the type."""

    if isinstance(state, AccumulatorState):
        return _populate_accumulator_fields(state, payload)
    if isinstance(state, ThermostatState):
        return _populate_heater_state(state, payload)
    if isinstance(state, PowerMonitorState):
        return _populate_power_monitor_state(state, payload)
    return _populate_heater_state(state, payload)


class DomainStateStore:
    """Canonical state store keyed by immutable inventory identifiers."""

    def __init__(self, nodes: Iterable[NodeId]) -> None:
        """Initialise the store for a set of allowable nodes."""

        self._states: dict[NodeId, DomainState] = {}
        self._allowed: dict[NodeId, NodeId] = {}
        self._addresses_by_type: dict[NodeType, set[str]] = {}
        self._gateway_connection = GatewayConnectionState()
        self._energy_snapshot: EnergySnapshot | None = None
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
        if self._energy_snapshot is not None:
            self._energy_snapshot = self._prune_energy_snapshot(self._energy_snapshot)

    def _prune_energy_snapshot(self, snapshot: EnergySnapshot) -> EnergySnapshot:
        """Return ``snapshot`` with metrics restricted to allowed nodes."""

        allowed = self._allowed
        metrics = dict(snapshot.metrics)
        if not metrics:
            return snapshot
        filtered = {
            node_id: metrics_value
            for node_id, metrics_value in metrics.items()
            if node_id in allowed
        }
        if filtered == metrics:
            return snapshot
        return EnergySnapshot(
            dev_id=snapshot.dev_id,
            metrics=filtered,
            updated_at=snapshot.updated_at,
            ws_deadline=snapshot.ws_deadline,
        )

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

    def resolve_node_id(self, node_type: NodeType | str, addr: Any) -> NodeId | None:
        """Return the canonical NodeId when permitted by the inventory."""

        return self._resolve_node_id(node_type, addr)

    def _apply_payload(
        self, node_id: NodeId, payload: Mapping[str, typing.Any], *, replace: bool
    ) -> None:
        """Apply ``payload`` to ``node_id`` using replace semantics when requested."""

        if not isinstance(payload, Mapping):
            return

        normalized = canonicalize_settings_payload(payload)
        if not normalized:
            if replace:
                normalized = {}
            else:
                return

        if replace or node_id not in self._states:
            self._states[node_id] = _build_state(node_id.node_type, normalized)
            return

        existing = self._states.get(node_id)
        if existing is None:
            self._states[node_id] = _build_state(node_id.node_type, normalized)
            return

        self._states[node_id] = _merge_state(existing, normalized)

    def apply_full_snapshot(
        self,
        node_type: NodeType | str,
        addr: Any,
        decoded_settings: Mapping[str, typing.Any] | None,
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
        delta: Mapping[str, typing.Any] | None,
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

    def get_energy_snapshot(self) -> EnergySnapshot | None:
        """Return the stored energy snapshot when available."""

        return self._energy_snapshot

    def set_energy_snapshot(self, snapshot: EnergySnapshot) -> bool:
        """Store an energy snapshot and return ``True`` when it changed."""

        if not isinstance(snapshot, EnergySnapshot):
            return False
        pruned = self._prune_energy_snapshot(snapshot)
        if self._energy_snapshot == pruned:
            return False
        self._energy_snapshot = pruned
        return True

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

    def set_gateway_connection_state(self, state: GatewayConnectionState) -> None:
        """Store the latest gateway connection state."""

        if not isinstance(state, GatewayConnectionState):
            return
        self._gateway_connection = clone_gateway_connection_state(state)

    def get_gateway_connection_state(self) -> GatewayConnectionState:
        """Return a defensive copy of the gateway connection state."""

        return clone_gateway_connection_state(self._gateway_connection)

    def replace_state(
        self,
        node_type: NodeType | str,
        addr: Any,
        state: DomainState | None,
    ) -> None:
        """Replace the stored state for ``(node_type, addr)`` when valid."""

        if state is None:
            return

        node_id = self._resolve_node_id(node_type, addr)
        if node_id is None:
            msg = f"Unknown node for replace_state: type={node_type} addr={addr}"
            raise ValueError(msg)

        expected_type: type[DomainState]
        if node_id.node_type is NodeType.ACCUMULATOR:
            expected_type = AccumulatorState
        elif node_id.node_type is NodeType.THERMOSTAT:
            expected_type = ThermostatState
        elif node_id.node_type is NodeType.POWER_MONITOR:
            expected_type = PowerMonitorState
        else:
            expected_type = HeaterState

        if type(state) is not expected_type:
            msg = (
                f"State type {type(state).__name__} does not match "
                f"node_type={node_id.node_type.value}"
            )
            raise TypeError(msg)

        self._states[node_id] = state


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


def _copy_state_field_value(value: Any) -> Any:
    """Return a shallow copy for mappings and sequences."""

    cloned_mapping = _copy_mapping(value)
    if cloned_mapping is not None:
        return cloned_mapping
    cloned_sequence = _copy_sequence(value)
    if cloned_sequence is not None:
        return cloned_sequence
    return value


def state_to_dict(
    state: DomainState | None, *, include_none: bool = False
) -> dict[str, Any]:
    """Return a defensive mapping representation of ``state``."""

    if state is None:
        return {}

    filtered: dict[str, Any] = {}
    for field in fields(state):
        key = field.name
        value = getattr(state, key)
        if value is None and not include_none:
            continue
        filtered[key] = _copy_state_field_value(value)
    return filtered


def clone_state(state: DomainState | None) -> DomainState | None:
    """Return a detached copy of ``state`` when available."""

    if state is None:
        return None

    state_type = type(state)
    payload = {
        field.name: _copy_state_field_value(getattr(state, field.name))
        for field in fields(state_type)
    }
    return state_type(**payload)


def clone_gateway_connection_state(
    state: GatewayConnectionState | None,
) -> GatewayConnectionState:
    """Return a detached copy of ``state`` when available."""

    if state is None:
        return GatewayConnectionState()

    return GatewayConnectionState(
        status=state.status,
        connected=state.connected,
        last_event_at=state.last_event_at,
        healthy_since=state.healthy_since,
        healthy_minutes=state.healthy_minutes,
        last_payload_at=state.last_payload_at,
        last_heartbeat_at=state.last_heartbeat_at,
        payload_stale=state.payload_stale,
        payload_stale_after=state.payload_stale_after,
        idle_restart_pending=state.idle_restart_pending,
    )


def apply_payload_to_state(
    state: DomainState | None, payload: Mapping[str, typing.Any] | None
) -> DomainState | None:
    """Apply a mapping payload onto ``state`` when possible."""

    if state is None or not isinstance(payload, Mapping):
        return state

    canonical = canonicalize_settings_payload(payload)
    if isinstance(state, AccumulatorState):
        return _populate_accumulator_fields(state, canonical)
    if isinstance(state, ThermostatState):
        return _populate_heater_state(state, canonical)
    if isinstance(state, PowerMonitorState):
        return _populate_power_monitor_state(state, canonical)
    return _populate_heater_state(state, canonical)
