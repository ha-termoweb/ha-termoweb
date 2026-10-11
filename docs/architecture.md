# HA-TermoWeb Architecture Overview

This document is the authoritative reference for the **v2 architecture** of the
TermoWeb integration and describes the code as it is. Function-level detail is in
`docs/function_map.txt` (generated; see `AGENTS.md`). Radio details are in
`radio_backend.md` and `radio_protocol.md`.

## Architecture invariants

- **Latest Home Assistant only.** The integration targets the current HA
  stable release (`homeassistant>=2026.10.0`, Python 3.14). The minimum moves
  up with each HA release; there are no compatibility shims or fallback
  imports for older versions.
- **Single architecture.** There is one canonical implementation path; legacy
  or transitional code is not allowed.
- **Vendor payloads are fixed.** Vendor REST/WS schemas are stable external
  contracts. Wire models keep vendor field names intact; normalization happens
  after decoding into domain state.
- **Inventory is immutable.** A single inventory snapshot is captured during
  setup and never rebuilt or cached again elsewhere.
- **One canonical state pipeline.** Device updates flow through a single path:
  inventory snapshot → domain deltas → `DomainStateStore` → `DomainStateView`.
- **Vendor isolation.** TermoWeb vs Ducaheat differences exist only in
  backend/planner/codec modules. Entities and platforms remain vendor-agnostic:
  where a feature exists on only some backends (the lock platform, the
  installation power limit, heater priority, energy history import), callers
  ask `backend_capabilities(brand)` for the
  backend's declared `BackendCapabilities` instead of testing the brand.
- **Pydantic on the wire only.** Payload parsing/serialization uses Pydantic
  models; domain state is plain dataclasses or standard Python types.

## Canonical data pipeline

1. **Config entry setup** takes the brand selection, authenticates, and creates a
   single runtime container (`EntryRuntime`), stored as `entry.runtime_data`.
   Every listener and task it starts is tied to `entry.async_on_unload`, so a
   setup that fails part-way (or an unload) leaves nothing running.
2. **Inventory snapshot** (gateway + nodes) is retrieved once and stored in the
   runtime. Inventory never changes for the lifetime of the entry.
3. **Update sources** (REST polling, WebSocket push) produce **domain deltas**.
4. **DomainStateStore** applies deltas and holds the canonical in-memory state:
   per-node settings (including accumulator boost defaults `boost_time` /
   `boost_temp`), the gateway connection state, the installation power limit
   and the energy snapshot. `EntryRuntime` holds only runtime handles (clients,
   coordinators, tasks), never device state.
5. **DomainStateView** provides read-only access for entities. The view is
   owned by `StateCoordinator` (`coordinator.domain_view`).

What each store/view holds: per-node state (`HeaterState`, `AccumulatorState`,
`ThermostatState`, `PowerMonitorState`), the `GatewayConnectionState`, the
installation power limit (`apply_power_limit`) and the energy snapshot
(`apply_energy_snapshot`). Radio and WebSocket pushes enter through
`StateCoordinator.handle_ws_deltas`; entity-initiated writes patch the store
optimistically through `apply_entity_patch`.

Reality check on "entities only read through the view":

- Device **state** (node settings, gateway connection, power limit, energy
  metrics) is read only through `coordinator.domain_view`
  (`entity.py`, `sensor.py`, `number.py`, `binary_sensor.py`, `button.py`,
  `radio_pairing.py`). No entity or service reads raw REST/WS payloads or
  `coordinator.data`.
- Entities are **not** restricted to the view for everything else: they read
  the immutable `runtime.inventory` (node lists, names, sample targets), the
  coordinator's gateway metadata (`device_metadata`, `gateway_name`,
  `gateway_model`, `gateway_connected`), and the runtime's backend for writes.
  None of these rebuild or cache inventory.

## Vendor boundary

Vendor-specific logic is confined to:

- `codecs/` (wire parsing/serialization)
- `planner/` (command planning/validation)
- `backend/` (REST + WebSocket transport)

Entity platforms and shared domain logic must not reference vendor-specific
clients, payload shapes, or protocol details.

## Module map (authoritative)

This map defines responsibilities for each module family in the final design.

- `__init__.py` — service registration (`async_setup`), config entry
  setup/teardown, runtime construction, and platform forwarding.
- `runtime.py` — `EntryRuntime` definition, `require_runtime(...)` accessor, and
  runtime invariants (single instance per entry).
- `inventory.py` — the immutable `Inventory` (node lookups, address maps,
  sample targets, heater names).
- `nodes.py` — node models (`Node`, `HeaterNode`, `AccumulatorNode`,
  `PowerMonitorNode`, `ThermostatNode`), `build_node_inventory` (gateway
  payload to nodes) and the identifier normalisers. `normalize_node_type` is
  the string form (stripped, lowercased, any value).
- `domain/` — domain dataclasses, deltas, `DomainStateStore` (including energy
  snapshots and the gateway power limit), and `DomainStateView`. `domain/ids.py` owns `NodeType`, the
  node-type groups (`HEATER_NODE_TYPES`, `HEATING_NODE_TYPES`,
  `ENERGY_NODE_TYPES`) and `NodeType.coerce` (case-insensitive, `None` for
  unknown types); `NodeType(value)` is the strict, raising form.
- `coordinator/` — `state.py` holds `StateCoordinator` (REST settings polling,
  pending-write deferral, WebSocket delta merge, gateway device metadata);
  `energy.py` holds `EnergyStateCoordinator` (HH:05 REST sample polls,
  WebSocket sample leases, energy/power snapshots); `base.py` holds the shared
  `RaiseUpdateFailedCoordinator` base and logger wrapper.
- `backend/` — everything vendor- or transport-specific.
  - `base.py`: `Backend` (ABC), `BackendCapabilities`, `HttpClientProto`,
    `WsClientProto`. `factory.py`: `backend_capabilities(brand)`,
    `create_backend`, `create_entry_client`, `create_rest_client`,
    `create_radio_client`.
  - `termoweb.py` / `ducaheat.py` / `radio_backend.py` / `radio_monitor.py`:
    the four `Backend` subclasses (Tevolve uses `ducaheat.py`).
  - `rest_client.py`: the shared `RESTClient` (both cloud brands) with its
    `RequestLimiter`. `sanitize.py`: log redaction helpers.
  - `ws_client.py`, `ws_health.py`, `termoweb_ws.py`, `ducaheat_ws.py`,
    `ducaheat_engineio.py`, `radio_ws.py`: the WebSocket layer (see below).
  - `radio_client.py`, `radio_power.py`, `radio_ws.py` and `radio/`
    (`link`, `serial_link`, `protocol`, `dialect`, `discovery`, `pairing`,
    `survey`, `capture`): the local radio stack.
- `coerce.py` — the only lenient value coercers (`as_float`, `as_number`,
  `as_int`, `as_bool`, `as_percentage`); each returns `None` instead of raising
  and has one documented semantics (see the module docstring).
- `codecs/` — Pydantic payload models and conversion to/from domain types.
  `codecs/common.py` holds the shared wire helpers: `format_temperature`
  (strict), `safe_temperature` (inbound), `validate_units`, `validate_prog`
  and `validate_ptemp`. Vendor encoders (e.g. acm boost payloads) stay in
  their vendor codec.
  Files: `termoweb_codec`/`termoweb_models`, `ducaheat_codec`/`ducaheat_models`/
  `ducaheat_read_models`, `radio_codec`.
- `planner/` — vendor-specific write orchestration and validation rules.
  Only `ducaheat_planner.py` and `radio_planner.py` exist; TermoWeb needs no
  planner because its writes are single requests.
- Platform modules (`climate.py`, `sensor.py`, `binary_sensor.py`,
  `button.py`, `number.py`, `lock.py`) — vendor-agnostic entity
  implementations that read via `DomainStateView`; each exposes
  `async_setup_entry`.
- `entity.py` — shared entity base classes and helpers used by the platform
  modules (`HeaterNodeBase`, `HeaterPlatformDetails`, boost helpers, settings
  resolvers, backend write helpers).
  - Child-lock entities now use the `lock` platform; existing entity IDs may migrate
    from `switch.*_child_lock` to `lock.*_child_lock` based on registry behavior.
- `config_flow.py` — `TermoWebConfigFlow` (the class HA registers): the
  setup menu, cloud login, reconfigure dispatch and reauth.
  `config_flow_radio.py` holds the radio setup helpers (serial ports, probes,
  discovery, pairing, entry data) and `RadioConfigFlow`, the unregistered base
  class with the ESP32 gateway and nanoCUL steps. `options_flow.py` holds
  `TermoWebOptionsFlow` (radio rehoming).
- `services/` — one module per registered service, each with an
  `async_register_*` function called once from `async_setup`:
  `energy_history` (`import_energy_history`), `radio_pairing`, `radio_survey`,
  `radio_capture`. Services look up loaded entries with
  `runtime.loaded_runtimes(hass)`. Climate entity services are registered in
  `climate.py`.
- `energy.py` — the energy history import (see below). `throttle.py` — the
  process-wide samples limiter. `diagnostics.py` — the diagnostics download.
  `boost.py` — boost metadata parsing. `utils.py` — device-info builders and
  small shared helpers. `const.py` — domain, brand and option constants.
- `radio_pairing.py`, `radio_rehome.py`, `radio_survey.py` — the Home
  Assistant side of radio pairing, rehoming and surveys (settings snapshots,
  new nodes, saved reports); the radio protocol side is in `backend/radio/`.
  `radio_discovery.py` finds radio heaters after setup: it stores them in the
  entry data and reloads the entry, so the inventory is still built only at
  setup. `radio_autopair.py` pairs heaters that announce themselves while the
  entry's pairing window is open, then hands them to `radio_discovery`.

## WebSocket client layering

`Backend.create_ws_client` returns a `WsClientProto` (`start()` returns the
runner task, `stop()` is async). Setup stores it in `runtime.ws_clients` and
the task in `runtime.ws_tasks`.

- `ws_client.py` — `_WSStatusMixin` (status and `WsHealthTracker` updates,
  state bucket bookkeeping) and `_WSCommon(_WSStatusMixin)` (task lifecycle,
  backoff, the `ConnectionRateLimiter` for reconnects, payload-idle window, shared
  delta helpers `build_settings_delta`, `translate_path_update`,
  `forward_ws_sample_updates`).
- `termoweb_ws.py` — `TermoWebWSClient(_WSCommon)`: legacy **Socket.IO 0.9**
  (`GET /socket.io/1/` handshake, then `/socket.io/1/websocket/<sid>`).
- `ducaheat_engineio.py` — pure framing: **Engine.IO v3** (`EIO=3`) polling
  packets and Socket.IO event encode/decode. `ducaheat_ws.py` —
  `DucaheatWSClient(_WSCommon)`: stateful client (polling handshake, upgrade to
  websocket, subscriptions, pings, idle recovery).
- `radio_ws.py` — `RadioListener(_WSStatusMixin)` replaces the websocket
  client for the radio brands (it does not subclass `_WSCommon`).
- `ws_health.py` — `WsHealthTracker`. `__init__.py` watches the trackers: when
  every tracker is healthy with fresh payloads, REST polling is suspended
  (`runtime.poll_suspended`); it resumes when health is lost.

All clients emit domain deltas through `coordinator.handle_ws_deltas`.

## Backend capabilities

`BackendCapabilities` (frozen dataclass in `backend/base.py`) is declared as
the `capabilities` class attribute of each `Backend`; read it with
`backend_capabilities(brand)`. Nothing outside `backend/` tests the brand.

| Flag (default) | Gates | TermoWeb | Ducaheat/Tevolve | Radio | Radio monitor |
|---|---|---|---|---|---|
| `lock` (False) | `lock` platform (`_platforms_for_brand`) | no | yes | yes | no |
| `power_limit` (False) | installation power-limit number; limit fetch in `StateCoordinator` | yes | no | yes | no |
| `priority` (False) | heater priority number | yes | yes | yes | no |
| `energy_history` (False) | `import_energy_history` service | yes | yes | no | no |
| `energy` (False) | heater energy/power and installation total sensors | yes | yes | yes | no |
| `geo_data` (False) | installation location sensor | yes | yes | no | no |
| `frame_monitor` (False) | monitor-only platforms and the frames-heard sensor | no | no | no | yes |
| `local_radio` (False) | one entry per link, closed on unload, radio setup/reconfigure steps; skipped by cloud unique-ID migration | no | no | yes | yes |
| `options_flow` (False) | the options flow (`async_supports_options_flow`) | no | no | yes | no |
| `heater_discovery` (False) | background network search and heater finder (`radio_discovery.py`); found heaters are stored and the entry reloads | no | no | yes | no |
| `site_device` (True) | registration of the site device | yes | yes | yes | no |
| `web_portal` (True) | cloud web-portal link in device info (`utils.py`) | yes | yes | no | no |
| `account_scope` ("") | prefix of the cloud account unique ID | "" | `ducaheat` | "" | "" |

## Runtime flow diagram

```mermaid
flowchart LR
    User[[Home Assistant UI]]

    subgraph HA[Home Assistant · TermoWeb Integration]
        Setup[Config Entry Setup]
        Runtime[EntryRuntime]
        Inventory[Inventory Snapshot]
        Backend[Backend (REST + WS)]
        Deltas[Domain Deltas]
        Store[DomainStateStore]
        View[DomainStateView]
        Entities[Entity Platforms]
        Services[Services]
    end

    subgraph Cloud[TermoWeb / Ducaheat Cloud]
        REST[REST API]
        Socket[Socket.IO / Engine.IO]
    end

    subgraph LAN[Local radio · brand "radio"]
        RadioClient[RadioClient + RadioListener]
        Gateway[ESP32 radio gateway · TCP 2323]
        Heaters[Heaters · 869 MHz]
    end

    User --> Setup
    Setup --> Runtime
    Runtime --> Inventory
    Runtime --> Backend

    Backend --> REST
    Backend --> Socket
    REST --> Deltas
    Socket --> Deltas

    Backend -. radio brand .-> RadioClient
    RadioClient <--> Gateway
    Gateway <--> Heaters
    RadioClient --> Deltas

    Inventory --> Store
    Deltas --> Store
    Store --> View
    View --> Entities
    View --> Services
```

## Radio backend

The `radio` brand reaches heaters directly through a home-built ESP32 +
CC1101 gateway on the LAN instead of a cloud. It follows the same boundary:
`backend/radio/` holds the framing and payload protocol, `codecs/radio_codec.py`
maps radio records to the canonical settings dict, `planner/radio_planner.py`
orders writes, and `backend/radio_client.py` / `radio_ws.py` /
`radio_backend.py` implement `HttpClientProto`, `WsClientProto` and `Backend`.
`RadioListener` takes the websocket client's place: it answers heater traffic,
reads every heater periodically and pushes `NodeSettingsDelta`s through
`coordinator.handle_ws_deltas`, with the same health tracker that suspends
coordinator polling. Cloud-only features are switched off with
`BackendCapabilities` (`priority`, `energy_history`, `power_limit`, `lock`,
`energy` for the heater energy/power and installation total sensors,
`geo_data` for the installation location sensor). Other radio differences are
capabilities too: `local_radio` (the entry owns a gateway or serial link: one
entry per link, closed on unload, radio reconfigure steps), `options_flow`
(rehoming), `site_device` and `web_portal` (device
registry layout and links). `account_scope` prefixes cloud entry unique IDs,
so Ducaheat and Tevolve, which share one backend, share one account namespace
(`ducaheat:<account>`). The backend factory builds each entry's client
(`create_entry_client`), and `Backend.diagnostics` adds backend facts to the
diagnostics download.
See `radio_backend.md` for the capability matrix.

The `radio_monitor` brand is a listen-only radio entry without heaters:
`backend/radio_monitor.py` pairs a listen-only `RadioClient` (its link
refuses every transmit) with `RadioMonitor`, which counts frames instead of
performing station duties. Its `frame_monitor` capability limits the
platforms to the gateway-online binary sensor and the frames-heard sensor;
`site_device` is off, so its gateway has no site device.
See `radio_protocol.md` section 10.

## Device registry

Setup registers the site device (when the backend has `site_device`) and then
the gateway device, before the platforms load. Every other device names its
parent by registry id (`via_device_id`), never by the deprecated `via_device`
identifier: the gateway points at the site, and nodes and power monitors point
at the gateway. `utils.build_node_device_info` builds the node devices.

## Entity unique IDs

Every entity unique ID is built in `identifiers.py`, and nowhere else, with one
scheme: `termoweb:<dev_id>[:<device>]:<key>`. `<device>` is the device the
entity belongs to: nothing for the gateway, `site` for the site, or
`<node_type>:<node_key>` for a node. `<key>` is one snake_case token. Examples:
`termoweb:<dev>:online`, `termoweb:<dev>:site:energy_total`,
`termoweb:<dev>:acm:2:boost_end`.

The node key (`identifiers.node_key`) is the node's address, except for a
radio heater whose serial is known: then it is the serial, e.g.
`termoweb:<dev>:htr:X123456:climate`. Node devices use the same key:
`(termoweb, <dev_id>, <node_key>)`. A heater paired again at another address
therefore keeps its device and entities. Cloud nodes have no serial, so their
IDs do not change. When a radio heater's serial becomes known, setup moves its
address-keyed device identifier and entity unique IDs to the serial
(`radio_identity.async_rekey_node`, built on `identifiers.rekey_unique_id`);
taken targets are skipped with a WARNING. See
[radio backend: heater identity](radio_backend.md#heater-identity-and-names).

Config entry version 1.5 moves older IDs to this scheme with
`entity_registry.async_migrate_entries`. `identifiers.migrate_unique_id` holds
the table of old formats. Only the unique ID changes, so the entity ID,
history and user customisations stay. If an entity already has the new ID,
the old entity is left as it is and a WARNING is logged.

## Config entry versions

`TermoWebConfigFlow` creates entries at version 1.5. `async_migrate_entry`
applies the steps in order and refuses entries with a major version above 1:

| Minor | Change |
|---|---|
| 1.2 | Drop the legacy `poll_interval` from data and options |
| 1.3 | Drop the stored `supports_diagnostics` key |
| 1.4 | Entry unique ID: case-fold the username and scope it by backend (`ducaheat:<account>` for Ducaheat and Tevolve). If another entry already has that ID, the entry keeps its own and a WARNING is logged |
| 1.5 | Entity unique IDs move to the `identifiers.py` scheme (below) |

`async_remove_entry` deletes the energy import `Store`. Legacy import progress
kept in entry options is moved into that Store on the next import (only a
completed legacy import is trusted; other nodes re-import).

## Logging and redaction

- Levels: INFO for lifecycle (setup, import start/summary), DEBUG for REST/WS
  payloads and protocol traffic, WARNING for recoverable anomalies
  (e.g. rate-limit pauses, skipped migrations), ERROR for unrecoverable failures.
- Anything that can carry credentials or identifiers is passed through
  `backend/sanitize.py` before it is logged or put in an exception message:
  `redact_text` (bearer tokens, `token=` query values, e-mail addresses,
  `/devs/<id>` paths, "authorization"), `redact_token_fragment`,
  `mask_identifier`. The TermoWeb WS client logs URLs with `{token}`,
  `{dev_id}` and `{sid}` placeholders.
- Diagnostics use `async_redact_data` with `SENSITIVE_FIELDS`
  (`diagnostics.py`: tokens, password, username, `dev_id`, `device_id`,
  `serial`, `serial_id`, `client_secret`, `authorization`) and redact the
  installation geo data. Radio surveys and captures are redacted by
  `backend/radio` `redact` helpers (network ids, serial bytes, on-air hex)
  before they are saved for sharing.

## Operational constraints

- Entity services and writes fail loudly: invalid input or the wrong node type
  raises `ServiceValidationError`, a failed backend write raises
  `HomeAssistantError`. Only the debounced mode/setpoint write, which runs after
  the service call returned, logs its failure instead.
- REST requests must be rate-limited and treated as a fallback when WebSocket
  updates are unavailable. Each `RESTClient` owns one `RequestLimiter`: every
  request start (polling, writes, segmented writes, the TermoWeb RTC
  keepalive, token POSTs and the energy history import) is spaced at least
  0.5 s apart, so one config entry never exceeds 2 requests/s in total. A 429
  pauses all requests on that client for the `Retry-After` delay (seconds or
  HTTP date, capped at 15 min; 30 s when absent) and raises
  `BackendRateLimitError` with `retry_after` set.
- WebSocket first: once startup completes, REST polling is suspended while the
  WS trackers are healthy (see "WebSocket client layering"). Reconnects are
  paced by `ConnectionRateLimiter` and exponential backoff.
- The `import_energy_history` service is held to **2 queries per second** twice
  over: each sample request waits on the process-wide samples limiter
  (`throttle.py`, 0.5 s interval) and then passes through the client's
  `RequestLimiter` like every other request. One import runs at a time
  (`runtime.energy_import_lock`).
- Energy history import (`energy.py`) never deletes statistics (it overwrites
  rows in place by `start` and rewrites the sums that follow). It walks each
  node's window forward one day per request, upserts the hourly rows with
  `async_import_statistics`, waits for the recorder commit, then records the
  day as done in a `helpers.storage.Store` (`termoweb.energy_import.<entry_id>`),
  so an interrupted or failed import resumes where it stopped. Sums continue
  from the statistic before the window; afterwards later statistics are shifted
  with `async_adjust_statistics` so the series has no step at the seam.
- Inventory-driven assumptions (node list, addresses, and types) are immutable
  for the life of the entry; if hardware changes, the user must reload the
  integration.
- Node entities are available only while their coordinator's last update
  succeeded (`last_update_success`) **and** the node is in the inventory. The
  installation total energy is unknown unless every heater/accumulator reports
  a value, so a missing node never shows up as a drop in a `TOTAL_INCREASING`
  meter.
- Energy samples have one REST pipeline: `EnergyStateCoordinator` fetches the
  past hour for each node once at setup and then at HH:05 local time. The
  poll is skipped while WebSocket samples are fresh (lease plus margin), and
  WS samples never make REST polling more frequent. A 429 stops the per-node
  loop and skips the next 1, 2, 4… (max 8) hourly polls.

## Temperature units

Each node reports its units (`C` or `F`). Climate, temperature sensor and boost
temperature number entities use that unit as their native unit, and their
setpoint limits (5–30 °C, i.e. 41–86 °F) and clamps are in that unit. Home
Assistant converts to the user's unit system; writes carry the device's unit.

## Heater mode mapping (HA ↔ backend)

Implementation note for heater writes:

- HA **Auto** maps to backend `auto` when the active weekly program is followed
  with no user override.
- Changing target temperature while HA is still in **Auto** maps to backend
  `modified_auto` (temporary override). This override lasts until the next
  program slot boundary, then the backend resumes `auto` behavior.
- HA **Heat** (manual mode selected explicitly by the user) maps to backend
  `manual`.
- A backend `modified_auto` is reported as preset `temporary_override`. The
  preset is display-only (not in `preset_modes`); selecting preset `none`
  resumes `auto`.
- An unknown or missing backend mode is reported as an unknown HVAC mode
  (`None`), never as Heat. Accumulators list Heat only while the device reports
  `manual`, which the integration cannot set on them.
- `climate.turn_off` selects Off; `climate.turn_on` restores the mode used
  before the last turn off (Heat or Auto; accumulators always Auto), else Auto.

## Testing approach

- Tests run on **real Home Assistant** (`pytest-homeassistant-custom-component`,
  the `hass` fixture and `MockConfigEntry`). Home Assistant internals are never
  patched.
- Fakes exist **only at the backend boundary**: the `cloud` fixture in
  `tests/conftest.py` (a `FakeCloud` replacing the public `RESTClient` methods
  and the WebSocket client factory) and the doubles in `tests/fakes/`. Pure
  modules (codecs, domain, radio protocol) are tested without `hass`.
- `tests/` mirrors `custom_components/termoweb/`: `tests/backend/`,
  `tests/backend/radio/`, `tests/codecs/`, `tests/domain/`, `tests/planner/`,
  `tests/services/`, plus `tests/test_<module>.py` per top-level module. Add
  tests to the module's file; test modules never import each other.
- `filterwarnings = ["error"]`: warnings fail tests, so tests must close what
  they open. Order is random (`pytest-randomly`); re-run a failure with its
  printed `-p randomly --randomly-seed=N`.
- Coverage gate: 95% (`fail_under` in `pyproject.toml`, branch coverage).
  CI runs `timeout 120s uv run pytest --cov` (see `.github/workflows/`).
