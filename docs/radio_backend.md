# Radio backend: capabilities and gaps

The radio backend controls heaters directly through a home-built ESP32 +
CC1101 radio gateway on your local network (TCP port 2323). It does not use
the TermoWeb or Ducaheat cloud. It implements the same backend interfaces as
the cloud backends, so entities and the coordinator do not know which backend
they talk to.

Status: the backend code is complete and tested with a simulated gateway. It
is **not yet selectable** in the config flow; PR 4 wires up setup.

Legend:

- ✅ works on the reference heater (dialect B, node 06)
- 🟡 implemented, the protocol is known (dialect A), not yet verified on a
  dialect-B heater
- 🔬 partly known
- ❌ not available over radio; the "Radio behaviour" column says what happens

## Modules

| Module | Role |
|---|---|
| `backend/radio/` | Frame dialects, payload builders/decoders, the TCP `RadioLink` (PR 2). No Home Assistant code. |
| `backend/radio_client.py` | `RadioClient`: the `HttpClientProto` methods over one shared `RadioLink`. |
| `backend/radio_ws.py` | `RadioListener`: the push client (`WsClientProto`). Station duties, deltas, health. |
| `backend/radio_backend.py` | `RadioBackend`: capabilities, `create_ws_client`, `fetch_hourly_samples`. |
| `codecs/radio_codec.py` | Radio records ↔ the canonical settings dict; canonical commands → radio payloads. |
| `planner/radio_planner.py` | Settings write → ordered list of commands/payloads. |

The planner is small. A radio write is one frame per command, so the planner
only orders the commands (presets, program, then mode/setpoint) and validates
all of them before the first frame is sent. It exists to keep the same
codec/planner/backend split as Ducaheat.

The radio wire is raw bytes, not JSON, so there are no Pydantic models. The
decoders in `backend/radio/protocol.py` return frozen dataclasses.

## Capability flags

`backend_capabilities("radio")` returns:

| Flag | Radio | Cloud | Gates |
|---|---|---|---|
| `lock` | False | Ducaheat only | lock platform |
| `power_limit` | False | TermoWeb only | power-limit number, coordinator power-limit poll |
| `priority` | False | True | heater priority number entities |
| `energy_history` | False | True | `import_energy_history` service (logs an error and skips the entry) |

`priority` and `energy_history` are new in this PR. Both cloud backends
declare them True, so cloud behaviour does not change.

`lock` is False even though the client can send the `BA` toggle: the
dialect-B status record does not report the lock state, so a lock entity
could not show the real state.

## Canonical mapping (reads)

`get_node_settings` sends `B8` (status) and, at most once an hour, `B0`
(program). It returns the same canonical dict the TermoWeb cloud codec
produces. A key is present only when the radio record carries it. Nothing is
derived or guessed.

| Key | Source | Format | Status |
|---|---|---|---|
| `mode` | status mode byte | `01`→`auto`, `02`→`manual`, `03`→`modified_auto`, `04`→`off`; unknown codes omit the key | ✅ read |
| `ptemp` | status presets | `[anti_frost, eco, comfort]` = `[cold, night, day]`, strings with one decimal (`"16.5"`), like the cloud | ✅ read |
| `prog` | `B1` program reply | 168 ints, **Monday 00:00 first**, values 0/1/2 | ✅ read (24-slot), 🟡 (48-slot) |
| `units` | fixed | `"C"`; the radio carries every temperature in half degrees Celsius | ✅ |
| `mtemp` | full status record (E6/E4/E5/E3) | one-decimal string | 🟡 dialect A; ❌ the dialect-B short record has no room temperature |
| `stemp` | full status record | one-decimal string | 🟡 dialect A; ❌ dialect-B short record |
| `state` | full record flag `01` | `"on"` / `"off"` | 🟡 dialect A; ❌ dialect B |
| `lock` | full record flag `02` | bool | 🟡 dialect A; ❌ dialect B |
| `max_power` | full record power, or the `BE` power request (bytes 3–4, deciwatts) | float watts | ✅ (from `BE`) |

Mode: radio `03` is a temporary override that ends at the next program slot.
That is exactly the cloud's `modified_auto`, which `climate.py` shows as HVAC
mode Auto with preset `temporary_override`.

Program order: the radio wire is Sunday-first; the integration (cloud codec,
`climate.py` slot lookup `weekday() * 24 + hour`) is Monday-first.
`ProgramRecord.hourly_monday_first` does the rotation. A 48-slot program
whose two half hours differ in some hour has no faithful hourly form, so
`prog` is omitted (and the read is not repeated within the hour).

Missing keys: the coordinator stores each poll with replace semantics, so the
client keeps the last program it read and the last `BE` power value and puts
them into every reply. A missing key therefore means "this heater never
reported it", not "lost since the last poll".

A heater that does not answer, or a gateway that cannot be reached, makes
`get_node_settings` return None. The coordinator then skips that heater and
keeps its last state. Gateway health is shown by the listener (below).

## Canonical mapping (writes)

`set_node_settings` turns its arguments into commands, validates all of them,
then sends one frame per command. Each frame must be acked by the heater
(link layer) and answered `<opcode+1> 55`. No ack, no answer, `56`
(rejected) or any other answer raises `RadioCommandError`; entities log it.

| Argument | Radio payload | Status |
|---|---|---|
| `ptemp` | `B6 af eco comfort` (half degrees, strictly increasing, 7–35 °C) | 🟡 |
| `prog` | `B2` + 84 bytes, 48 slots a day, Sunday first | 🟡 |
| `mode="manual"` + `stemp` | `B4 02 <half-degrees>` (manual setpoint) | 🟡 |
| `mode="modified_auto"` + `stemp` | `B4 03 <half-degrees>` (temporary override) | 🟡 |
| `stemp` alone | `B4 02 <half-degrees>` (manual setpoint) | 🟡 |
| `mode` alone: `auto` / `manual` (`heat`) / `off` | `B4 01` / `B4 02` / `B4 04` | 🟡 |
| `mode="modified_auto"` without `stemp`; `auto`/`off` with `stemp` | ValueError before sending | — |
| `units` other than `"C"` | ValueError before sending | — |
| `boost_time`, `cancel_boost` | `RadioUnsupportedError` | ❌ |

After a program write the cached program is dropped, so the next read
re-reads it.

## Client methods (HttpClientProto)

| Method | Radio behaviour | Status |
|---|---|---|
| `list_devices()` | One entry: `dev_id` = gateway MAC from the `# Q` line (lowercase hex, no colons), name "Radio gateway", model "ESP32 + CC1101 radio gateway (dialect X)", `fw_version` from the `# Q` line. No MAC → `RadioError`. | 🟡 needs firmware that prints `mac=` |
| `get_nodes(dev_id)` | `{"nodes": [...]}` from the node list stored in the config entry | ✅ |
| `get_node_settings` | see "Canonical mapping (reads)" | ✅ / 🟡 |
| `set_node_settings` | see "Canonical mapping (writes)" | 🟡 |
| `set_node_lock` | `BA 01` / `BA 00`, verdict `BB 55`. No entity uses it (capability `lock` is False). | 🟡 |
| `set_acm_boost_state` | heater: `D2 01` / `D2 00`, verdict `D3 55`; the heater uses its own boost time and temperature. Accumulator: `RadioUnsupportedError`. No current entity calls it for heaters. | 🟡 heater, ❌ accumulator |
| `set_node_display_select` | `RadioUnsupportedError`. The flash opcode `5E 01` is documented for dialect A but has no builder in `backend/radio/protocol.py` yet. | ❌ (follow-up) |
| `set_node_priority` | `RadioUnsupportedError`; entities not created (`priority` False) | ❌ |
| `get_power_limit` | None ("no data"); never polled (`power_limit` False) | ❌ |
| `set_power_limit` | `RadioUnsupportedError`; entity not created | ❌ |
| `set_acm_extra_options` | `RadioUnsupportedError` (accumulator boost defaults) | ❌ |
| `get_node_samples` | `[]` ("no data"): heaters keep no history. The energy coordinator then logs "no samples" and moves on. | ❌ |
| `get_geo_data` | None ("no data"); setup treats it as optional | ❌ |
| `get_rtc_time` | Home Assistant local time as `{"y","n","d","h","m","s"}`, the cloud shape. The radio station is the heaters' clock master. | ✅ |

`RadioUnsupportedError` messages always read "… is not supported over the
radio gateway". Every caller of these methods already catches `Exception`
(entity writes, button press) or is gated by a capability flag.

`RadioBackend.fetch_hourly_samples` returns `{}`: there is no energy history
to import, so the hourly poller stores nothing.

## Push traffic (RadioListener)

The listener subscribes to the link and plays the part of a real TermoWeb
gateway. The gateway firmware sends the link-layer acks itself.

| Heater frame | Station reply | Integration effect | Status |
|---|---|---|---|
| registration `50` | EB `51` clock sync, dialect suffix (`dialect.eb_clock_suffix`: dialect B has no trailing `03`; the 9-byte form is rejected `53 56`) | — | ✅ |
| power request `BE ..` | `BF 01` (grant) | `max_power` delta; value kept for later reads | ✅ |
| report `56 B9 ..` | `57 55` | status delta (full record) | 🟡 |
| program report `56 B1 ..` | `57 55` | `prog` delta | 🟡 |
| route probe (tag `06`), acks, unknown frames, frames from nodes not in the inventory | none | ignored | ✅ |

Every `REFRESH_INTERVAL_S` (120 s; heaters need a clock sync at least every
150 s) the listener, for each heater in the inventory:

1. sends the EB `52` keepalive clock sync and expects `53 55` (counts as a
   heartbeat);
2. calls `get_node_settings` (B8, plus B0 when due) and pushes the result as a
   delta. This covers heaters that never push reports, such as the dialect-B
   reference heater.

Station replies run as background tasks; a heater that misses one simply
repeats its frame. All transmits share the link's single send lock.

## Health and polling

The listener uses the same `WsHealthTracker` and status signal as the cloud
websocket clients (`_WSStatusMixin`):

- gateway connected → status `connected`; first pushed payload → `healthy`;
- gateway connection drops → `disconnected` at once (from the link's
  disconnect callback), `coordinator.update_gateway_connection(connected=False)`;
- reconnect with backoff 5, 10, 30, 120, 300 s; the backoff resets after a
  successful connect;
- payload window: 3 × refresh interval (360 s).

Coordinator polling: **off while the listener is healthy and fresh, on as a
fallback otherwise.** This is the existing `__init__` logic: a healthy tracker
with a fresh payload suspends coordinator polling; a stale or unhealthy one
resumes it at the normal interval (30 min). The listener already reads every
heater every 120 s, so polling in parallel would only double the radio
traffic on a shared 869 MHz channel with a duty-cycle limit. The fallback
poll uses the same gateway, so it mainly retries while the listener
reconnects; it does no harm.

## Cloud features and what radio users see

| Cloud feature | Radio | Notes |
|---|---|---|
| Climate entity: mode, target temperature, presets, schedule | 🟡 | Writes proven in dialect A only. On dialect-B heaters target and current temperature stay unknown (short status record). |
| Climate HVAC action (heating / idle) | 🟡 / ❌ | Needs the full status record. |
| Temporary override (`modified_auto`) | 🟡 | `B4 03`. |
| Child lock entity | ❌ | Lock state not readable on dialect B; capability off. |
| Heater boost | 🟡 (client only) | No heater boost entity exists today. |
| Accumulator boost, boost defaults | ❌ | Unknown on radio. |
| Display flash button | ❌ | Button exists and shows an error; needs a `5E 01` builder. |
| Heater priority numbers | ❌ | Not created. |
| Installation power limit | ❌ | Not created, not polled. |
| Energy and power sensors | ❌ | Created but stay unknown: no energy counter is known for dialect B (`BC` returns the power record). Gating them needs a follow-up. |
| Energy history import service | ❌ | Logs "not supported by this backend" for radio entries. |
| Hourly samples poller | ❌ | Gets `{}`. |
| Gateway connectivity binary sensor | ✅ | From the listener's health tracker. |
| Websocket debug probe service | ❌ | Not applicable. |
| Geo data / device location | ❌ | None. |
| Gateway RTC | ✅ | Local time. |

## Known gaps

- Room temperature and setpoint on dialect-B heaters: no record seen so far
  carries them (see `ops/esp32/README.md`).
- Every write is unverified on a dialect-B heater.
- Display flash (`5E 01`): add a builder to `protocol.py`, then implement it.
- Energy counter: dialect A has `BC` → `BD` + u32 Wh; dialect B unknown. Not
  used by this backend yet.
- Energy/power sensor entities are still created for radio entries.

## What PR 4 must do

- Config flow: ask for host and port, then work out the dialect and the
  network id from the heater's own traffic. A dialect-B network id belongs to
  one installation (there is no default), and registration and power-request
  frames carry it in bytes 1-2. Then scan for heaters, and store the dialect,
  the network id and the node list as `[{"type": "htr", "addr": "<decimal radio id>", "name": ...}]`.
  Store addresses as plain decimal strings without leading zeros (`"6"`, not
  `"06"`): the listener matches heater frames by `str(src)`.
- Setup: build the client with `create_radio_client(host, port, dialect, nodes, network_id)`
  and the backend with `create_backend(brand=BRAND_RADIO, client=...)` instead
  of `create_rest_client`; skip username/password.
- `list_devices` connects to the gateway and raises `RadioLinkError` when it
  cannot, or `RadioError` when the gateway reports no MAC. Map both to
  `ConfigEntryNotReady` (the current handler only catches
  `TimeoutError`/`ClientError`/`BackendRateLimitError`).
- Unload: `RadioListener.stop()` closes the gateway connection.
- Do not add `"radio"` to `BRAND_LABELS` without a separate flow branch; the
  cloud login form would offer it.
- Manifest: no new requirements (the radio package uses only asyncio).
