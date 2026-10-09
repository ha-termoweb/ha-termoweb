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
| `lock` | True (keypad lock) | Ducaheat only | lock platform |
| `power_limit` | True (local power manager) | TermoWeb only | power-limit number, coordinator power-limit poll |
| `priority` | True (local power manager) | True | heater priority number entities |
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
| `mtemp` | full status record (E6/E4/E5/E3); dialect B: power record byte 1 (tenths of a degree) | one-decimal string | 🟡 dialect A; ✅ dialect B |
| `stemp` | full status record; dialect B: power record byte 3, the heater's active setpoint (half degrees) | one-decimal string | 🟡 dialect A; ✅ dialect B |
| `state` | full record flag `01`; dialect B: power record byte 7 (`BC` → `BD`, read when the status lacks it, and 90 s after each `BF 01` grant) | `"on"` / `"off"` | 🟡 dialect A; ✅ dialect B |
| `lock` | full record flag `02`; dialect B: the last lock state written (the short record has none) | bool | 🟡 dialect A; 🟡 dialect B |
| `max_power` | full record power, or the dialect-A `BE hi lo` power request (deciwatts) | float watts | 🟡 dialect A; ❌ dialect B (its power record carries no power) |

Mode: radio `03` is a temporary override that ends at the next program slot.
That is exactly the cloud's `modified_auto`, which `climate.py` shows as HVAC
mode Auto with preset `temporary_override`.

Program order: the radio wire is Sunday-first; the integration (cloud codec,
`climate.py` slot lookup `weekday() * 24 + hour`) is Monday-first.
`ProgramRecord.hourly_monday_first` does the rotation. A 48-slot program
whose two half hours differ in some hour has no faithful hourly form, so
`prog` is omitted (and the read is not repeated within the hour).

Target temperature on dialect-B heaters: the short status record has no
setpoint, but the power record does. Byte 3 is the heater's active setpoint:
the manual setpoint in manual mode (its own value, not a preset), the current
program slot's preset in program mode, and the override target in temporary
override. A target is written as a sixth `B6` byte with mode 02 (manual) or 03
(temporary override, which the heater ends at the next program change); the
presets are re-sent unchanged. Verified: byte 3 followed every write, and an
override to 24 °C in a night slot made the heater heat at full duty within
30 s. `B4 02 <setpoint>` is ignored by these heaters.

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

| Argument | Dialect A payload | Dialect B payload |
|---|---|---|
| `ptemp` | `B6 af eco comfort` (half degrees, strictly increasing, 7–35 °C) 🟡 | `B6 af eco comfort mode`, mode re-sent from a fresh status read ✅ |
| `prog` | `B2` + 84 bytes, 48 slots a day, Sunday first 🟡 | `B2` + 42 bytes, 24 slots a day, Sunday first ✅ |
| `mode` alone: `auto` / `manual` (`heat`) / `off` | `B4 01` / `B4 02` / `B4 04` 🟡 | `B6 af eco comfort mode`, presets re-sent from a fresh status read ✅ (`B4` is acked but ignored) |
| `ptemp` + `mode` together | two writes 🟡 | one `B6` ✅ |
| `mode="manual"` (or `heat`) + `stemp`, or `stemp` alone | `B4 02 <half-degrees>` (manual setpoint) 🟡 | `B6 af eco comfort 02 <stemp>`: manual setpoint, presets unchanged ✅ |
| `mode="modified_auto"` + `stemp` | `B4 03 <half-degrees>` (temporary override) 🟡 | `B6 af eco comfort 03 <stemp>`: temporary override, ended by the heater at the next program change ✅ |
| `mode="modified_auto"` without `stemp`; `auto`/`off` with `stemp` | ValueError before sending | ValueError before sending |
| `units` other than `"C"` | ValueError before sending | ValueError before sending |
| `boost_time`, `cancel_boost` | `RadioUnsupportedError` ❌ | `RadioUnsupportedError` ❌ |

A dialect-B preset or mode write first reads the status (`B8`), because `B6`
always carries all three presets and the mode. Every value is validated before
that read, so an invalid write sends nothing. A preset-only write keeps the
current mode, including a temporary override (`03`).

The dialect-B rows were verified on the reference heater by writing back its
current values, and by switching it to off and back to manual: the status
record followed each `B6`, and `B4 04` left the mode unchanged.

After a program write the cached program is dropped, so the next read
re-reads it.

## Client methods (HttpClientProto)

| Method | Radio behaviour | Status |
|---|---|---|
| `list_devices()` | One entry: `dev_id` = gateway MAC from the `# Q` line (lowercase hex, no colons), name "Radio gateway", model "ESP32 + CC1101 radio gateway (dialect X)", `fw_version` from the `# Q` line. No MAC → `RadioError`. | 🟡 needs firmware that prints `mac=` |
| `get_nodes(dev_id)` | `{"nodes": [...]}` from the node list stored in the config entry | ✅ |
| `get_node_settings` | see "Canonical mapping (reads)" | ✅ / 🟡 |
| `set_node_settings` | see "Canonical mapping (writes)" | 🟡 |
| `set_node_lock` | `BA 01` / `BA 00`; dialect A requires `BB 55`, dialect B only acks. The written state is remembered for records without a lock flag. | 🟡 |
| `set_acm_boost_state` | heater: `D2 01` / `D2 00`, verdict `D3 55`; the heater uses its own boost time and temperature. Accumulator: `RadioUnsupportedError`. No current entity calls it for heaters. | 🟡 heater, ❌ accumulator |
| `set_node_display_select` | `select=True`: `5E 01`; dialect A must answer `5F 55`, dialect B only acks. `select=False`: no-op. | 🟡 |
| `set_node_priority` | `RadioUnsupportedError`; entities not created (`priority` False) | ❌ |
| `get_power_limit` | None ("no data"); never polled (`power_limit` False) | ❌ |
| `set_power_limit` | `RadioUnsupportedError`; entity not created | ❌ |
| `set_acm_extra_options` | `RadioUnsupportedError` (accumulator boost defaults) | ❌ |
| `get_node_samples` | One sample: the estimated Wh counter (see Local power manager); `[]` while the heater's power is unknown. No history. | 🟡 |
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
| power request `BE ..` | `BF 01` (an acknowledgement; `BF 00` does not stop heating), then a `BC` read 90 s later and the power-limit check | dialect A: `max_power` delta (kept for later reads); dialect B: `state` delta from the power record read 90 s later | ✅ |
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
| Climate entity: mode, target temperature, presets, schedule | 🟡 | Dialect B: mode, presets and schedule writes ✅; target temperature writes ❌ and target/current temperature stay unknown (short status record). Dialect A: 🟡. |
| Climate HVAC action (heating / idle) | 🟡 / ✅ | Dialect A: full status record flags. Dialect B: power record byte 7, verified against a house meter. |
| Temporary override (`modified_auto`) | 🟡 / ✅ | `B4 03` in dialect A; `B6 .. 03 <setpoint>` in dialect B (ended by the heater at the next program change). |
| Child lock entity | 🟡 | `BA 01`/`BA 00`. A dialect-B heater acks it without a reply and does not report the lock, so the entity shows the last state written (unknown until first used). The keypad effect on dialect B is not yet confirmed. |
| Heater boost | 🟡 (client only) | No heater boost entity exists today. |
| Accumulator boost, boost defaults | ❌ | Unknown on radio. |
| Display flash button | 🟡 | `5E 01`: dialect A answers `5F 55`; a dialect-B heater acks it (no reply record). The visible flash on dialect B is not yet confirmed. |
| Heater priority numbers | 🟡 | Local power manager (see below). Higher numbers win. |
| Installation power limit | 🟡 | Local power manager (see below). Needs each heater's power: reported by dialect-A heaters, entered in the options for dialect B. |
| Energy and power sensors | 🟡 | Estimated: rated power × duty while the heating flag is set, integrated between power records. Needs each heater's power (options). Starts at 0 when Home Assistant starts. |
| Energy history import service | ❌ | Logs "not supported by this backend" for radio entries. |
| Hourly samples poller | ❌ | Gets `{}`. |
| Gateway connectivity binary sensor | ✅ | From the listener's health tracker. |
| Websocket debug probe service | ❌ | Not applicable. |
| Geo data / device location | ❌ | Not applicable; the location sensor is not created (`geo_data` capability off). |
| Gateway RTC | ✅ | Local time. |

## Local power manager

The cloud gateway keeps an installation under its power limit. On the radio,
a heater's `BE` power request is only acknowledged by `BF`: a dialect-B heater
heats whether the answer is `BF 01` or `BF 00` (checked against a house meter:
eleven 1.5 kW heating pulses after eleven `BF 00` answers). The station always
answers `BF 01`. The lever that does stop heating is the mode, so the local
power manager (`backend/radio_power.py`) sheds load:

- After every power-record read and every refresh, it adds up the power of the
  heaters reporting heating (power record byte 7, or the status flags).
- Over the limit, the heating heaters with the lowest priority are switched off
  (`B6 .. 04`, or `B4 04` in dialect A) until the rest fit. Their previous mode
  is stored in the entry options, so a restart does not lose it.
- When there is room again, switched-off heaters get their previous mode back,
  highest priority first, while each one's power fits. A temporary override
  comes back as program mode. Removing the limit restores every heater.
- A heater whose power is unknown is never switched off. Changing a switched-off
  heater's mode or target in Home Assistant hands it back to the user.

Higher priority numbers win. While switched off, a heater shows as Off in Home
Assistant.

Energy: the heaters keep no energy counter on the radio, so the client
estimates one (`EnergyEstimator`). Between two power records a heater draws
its power × duty (byte 6 of the record, ≈ the measured pulse ratio) while its
heating flag is set, and nothing otherwise; a gap longer than 15 minutes is
cut off. The counter feeds the normal energy and power sensors through
`get_node_samples`, and the listener pushes it to the energy sensors after
every power-record read (the energy coordinator itself polls only hourly). It
restarts at 0 when Home Assistant restarts, which the
energy sensors treat as a meter reset.

Settings live in the entry options (`radio_power`): the limit and priorities
are set through the usual number entities; each heater's power in watts is set
in the integration's options (**Configure**). Dialect-A heaters report their
own power in `BE`, which is used when no power is entered.

## Known gaps

- Keypad lock on dialect B: acked; whether the keypad locks is not yet confirmed.
- Boost, runback and EASY on dialect B: `D2` answers `D3 00 00`, `D6` answers `D7 19`, `D4` is ack only; no effect seen. Their meaning is unknown.
- Display flash on dialect-B heaters: the command is acked, the visible flash is not yet confirmed.
- Energy counter: dialect A has `BC` → `BD` + u32 Wh (not used yet; the
  estimate is used for both dialects); dialect B has none, so energy is
  estimated.

## Setup (config flow)

The first step is a menu: **cloud** (the existing brand + login form) or
**radio**.

The radio form asks for host, port (2323), an optional dialect (`auto`, `A`,
`B`) and an optional network id (4 hex digits).

1. `probe_gateway` connects with auto-ack off (nothing is transmitted) and
   reads the gateway MAC from the `# Q` line. The MAC becomes the `dev_id`
   and the entry's unique id (`radio:<mac>`). No MAC → error
   `no_gateway_mac`.
2. `radio_discover` is a progress step running `discover_radio`:
   - with a dialect and a known network id (given, or dialect A's fixed
     `1B30`), it skips listening;
   - otherwise `discovery.discover_network` alternates 10 s listening windows
     over dialects B then A, up to 6 minutes. A heater that wants heat sends
     a power-request spell about every 5 minutes: 5 frames over about 27 s,
     never more than 8 s apart. A 10 s window that overlaps a spell therefore
     catches a frame, and with ~21 s per A+B cycle every spell overlaps a
     window of each dialect. An associated heater that does not want heat
     may stay silent. Every CRC-valid data frame
     gives the network id (bytes 1-2) and its sender. Dialect-B network ids
     belong to one installation, so there is no default to fall back on;
   - `discovery.probe_heaters` sends `B8` to addresses 2-32 plus every
     sender heard. Addresses that answer with a status record are the
     heaters.
3. The entry stores `brand: radio`, `host`, `port`, `dialect`, `network_id`
   (hex) and `nodes` as `[{"type": "htr", "addr": "6", "name": "Heater 6"}]`.
   Addresses are plain decimal strings: the listener matches frames by
   `str(src)`.

Errors: `cannot_connect_radio`, `no_gateway_mac`, `no_traffic`,
`no_heaters`, `invalid_network_id`.

Reconfigure changes host and port. With **Scan for heaters again** it
re-runs the scan with the stored dialect and network id, and replaces the
node list.

Setup builds the client with `create_radio_client(host, port, dialect, nodes,
network_id)`. `RadioLinkError` and `RadioError` from `list_devices` raise
`ConfigEntryNotReady`, so Home Assistant retries while the gateway is
offline. Unload stops the listener, which closes the connection.

