# Radio backend: capabilities and gaps

The radio backend controls heaters directly through a home-built ESP32 +
CC1101 radio gateway on your local network (TCP port 2323). It does not use
the TermoWeb or Ducaheat cloud. It implements the same backend interfaces as
the cloud backends, so entities and the coordinator do not know which backend
they talk to.

Status: the backend code is complete and tested with a simulated gateway. It
is **not yet selectable** in the config flow; PR 4 wires up setup.

Legend:

- ✅ verified on a real dialect-B heater through the ESP32 gateway
- 🟡 implemented, the protocol is known (dialect A), not yet verified on a
  dialect-B heater
- 🔬 partly known
- ❌ not available over radio; the "Radio behaviour" column says what happens
- ➖ not applicable to a local radio gateway

## Modules

| Module | Role |
|---|---|
| `backend/radio/` | Frame dialects, payload builders/decoders, the TCP `RadioLink` (PR 2). No Home Assistant code. |
| `backend/radio_client.py` | `RadioClient`: the `HttpClientProto` methods over one shared `RadioLink`. |
| `backend/radio_ws.py` | `RadioListener`: the push client (`WsClientProto`). Station duties, deltas, health. |
| `backend/radio_backend.py` | `RadioBackend`: capabilities, `create_ws_client`, `fetch_hourly_samples`. |
| `codecs/radio_codec.py` | Radio records ↔ the canonical settings dict; canonical commands → radio payloads. |
| `planner/radio_planner.py` | Settings write → ordered list of commands/payloads. |
| `radio_identity.py` | Reads missing heater serials at setup; moves registry entries from address to serial IDs. |

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
| `local_radio` | True | False | one entry per gateway/stick, link closed on unload, radio reconfigure steps |
| `options_flow` | True | False | options flow (rehoming) |
| `web_portal` | False | True | `configuration_url` on devices |

`priority` and `energy_history` are new in this PR. Both cloud backends
declare them True, so cloud behaviour does not change.

`lock` is True: the client sends the `BA` toggle. The dialect-B status
record does not report the lock state, so the lock entity shows the last
state written (unknown until first used).

## Canonical mapping (reads)

`get_node_settings` sends `B8` (status) and, at most once an hour, `B0`
(program). While a heater's power is unknown it also sends `5A` (identity)
once, for the rating (see Local power manager). It returns the same canonical dict the TermoWeb cloud codec
produces. A key is present only when the radio record carries it. Nothing is
derived or guessed.

| Key | Source | Format | Status |
|---|---|---|---|
| `mode` | status mode byte | `01`→`auto`, `02`→`manual`, `03`→`modified_auto`, `04`→`off`; unknown codes omit the key | ✅ read |
| `ptemp` | status presets | `[anti_frost, eco, comfort]` = `[cold, night, day]`, strings with one decimal (`"16.5"`), like the cloud | ✅ read |
| `prog` | `B1` program reply | 168 ints, **Monday 00:00 first**, values 0/1/2 | ✅ read (24-slot), 🟡 (48-slot) |
| `units` | fixed | `"C"`; the radio carries every temperature in half degrees Celsius | ✅ |
| `mtemp` | full status record (E6/E4/E5/E3); dialect B: power record byte 1, the heater's own probe (tenths of a degree; reads about 1 °C above a nearby room sensor) | one-decimal string | 🟡 dialect A; ✅ dialect B |
| `stemp` | full status record; dialect B: power record byte 3, the heater's active setpoint (half degrees) | one-decimal string | 🟡 dialect A; ✅ dialect B |
| `state` | full record flag `01`; dialect B: power record byte 7 (`BC` → `BD`, read when the status lacks it, and 90 s after each `BF 01` grant) | `"on"` / `"off"` | 🟡 dialect A; ✅ dialect B |
| `lock` | full record flag `02`; dialect B: the last lock state written (the short record has none) | bool | 🟡 dialect A; 🟡 dialect B |
| `max_power` | full record power, or the dialect-A `BE hi lo` power request (deciwatts); dialect B: the rating in the `5B 55` identity reply (byte 10 × 100 W) | float watts | 🟡 dialect A; 🟡 dialect B (one 1500 W heater checked) |

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
to import.

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

Every `REFRESH_INTERVAL_S` (120 s; a dialect-A heater shows its lost-link
indication about 12 minutes after its last clock sync, see
[research: keepalive](research/sessions.md#keepalive-and-the-link-indication)) the listener, for each heater in the inventory:

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
- a gateway that loses power leaves a half-open socket that the operating
  system may only notice after about 15 minutes, so the link drops the
  connection itself after 3 transmit confirmations in a row go missing (about
  9 s), which starts the reconnect above;
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
| Climate entity: mode, target temperature, presets, schedule | 🟡 / ✅ | Dialect B, verified on a real heater: off / manual / program modes, the three presets, the weekly schedule, the target temperature (read and written as the heater's own setpoint) and the current room temperature. Dialect A: 🟡 (proven in the research, see [research/messages.md](research/messages.md); not tested with this integration). |
| Climate HVAC action (heating / idle) | 🟡 / ✅ | Dialect A: full status record flags. Dialect B: power record byte 7, verified against a house meter. |
| Temporary override (`modified_auto`) | 🟡 / ✅ | `B4 03` in dialect A; `B6 .. 03 <setpoint>` in dialect B (ended by the heater at the next program change). |
| Child lock entity | 🟡 | `BA 01`/`BA 00`. A dialect-B heater acks it without a reply and does not report the lock, so the entity shows the last state written (unknown until first used). The keypad effect on dialect B is not yet confirmed. |
| Heater boost | 🟡 (client only) | No heater boost entity exists today. |
| Accumulator boost, boost defaults | ❌ | Unknown on radio; no accumulator was available to test. |
| Display flash button | 🟡 | `5E 01`: dialect A answers `5F 55`; a dialect-B heater acks it (no reply record). The visible flash on dialect B is not yet confirmed. |
| Heater priority numbers | ✅ | Local power manager (see below). Higher numbers win. |
| Installation power limit | ✅ | Local power manager (see below), verified with a house meter. Needs each heater's power: reported by dialect-A heaters, read from the identity record for dialect B. |
| Energy and power sensors | ✅ (estimated) | Within ~4 % of a house meter over a heating episode. Estimated: rated power × duty while the heating flag is set, integrated between power records. Needs each heater's power (read from the heater). Starts at 0 when Home Assistant starts. |
| Energy history import service | ➖ | Not applicable: radio heaters keep no energy history to import. Logs "not supported by this backend". |
| Hourly samples poller | ➖ | Not needed: the listener pushes the energy estimate live; the poller gets `{}`. |
| Gateway connectivity binary sensor | ✅ | From the listener's health tracker. |
| Geo data / device location | ➖ | Not applicable; the location sensor is not created (`geo_data` capability off). |
| Gateway RTC | ✅ | Local time. |

## Local power manager

The cloud gateway keeps an installation under its power limit. On the radio,
a heater's `BE` power request is only acknowledged by `BF`: a dialect-B heater
heats whether the answer is `BF 01` or `BF 00` (checked against a house meter:
eleven 1.5 kW heating pulses after eleven `BF 00` answers). In dialect A the
vendor gateway's firmware treats `BF 00` as a refusal with a back-off, but
only `BF 01` was ever seen on air
([research](research/messages.md#power-request-be-and-verdict-bf)). The station always
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

Energy: dialect-B heaters keep no energy counter on the radio. Dialect-A
heaters do (`BC` → `BD` + u32 Wh, the heater's own cumulative counter, see
[research](research/messages.md#energy-counter-bc)), but the client does not
read it yet. For both, the client estimates one (`EnergyEstimator`). Between two power records a heater draws
its power × duty (byte 6 of the record, ≈ the measured pulse ratio) while its
heating flag is set, and nothing otherwise; a gap longer than 15 minutes is
cut off. The counter feeds the normal energy and power sensors through
`get_node_samples`, and the listener pushes it to the energy sensors after
every power-record read (the energy coordinator itself polls only hourly). It
restarts at 0 when Home Assistant restarts, which the
energy sensors treat as a meter reset.

Settings live in the entry options (`radio_power`): the limit and priorities
are set through the usual number entities. Each heater's power is read from
the heater, so there is nothing to enter:

- dialect A reports its measured full-load power in `BE hi lo` power requests
  (deciwatts); this wins when known;
- otherwise the client reads the heater's identity (`5A`) once and uses its
  rating: byte 10 of the `5B 55` reply × 100 W (`0F` = 1500 W). Ratings
  outside 100-4000 W are ignored. An unanswered `5A` is retried at most
  hourly; a decoded reply is never read again until Home Assistant restarts.

Older versions had a manual power field in the options; the 1.6 entry
migration removes stored values.

## Pairing and factory reset

| Service | What it does |
|---|---|
| `termoweb.radio_factory_reset` | Saves a heater's settings, then factory-resets it (dialect B only; dialect A raises `RadioUnsupportedError`). Destructive: the heater also leaves the network. |

There is one pairing path: the pairing window below. A user pairs by putting
a heater into pairing mode within 30 minutes of adding the gateway, or after
pressing **Pair new heaters**. A reset heater is paired the same way and gets
its saved settings back when its serial is recognised. The network move
(`radio_rehome`) runs the same pairing exchange internally. Pairing runs in
the client (`async_pair`, then `async_restore`) and holds the exchange lock,
so heater commands wait. Protocol details:
[radio protocol §7-8](research/protocol.md#7-pairing).

### Automatic pairing (pairing window)

`radio_autopair` pairs heaters by themselves while the entry's pairing window
is open. Neither the config flow nor the options flow has a pairing step,
and there is no pairing service: the window and the button replace them.

- **Window.** `pairing_window_end(hass, entry)` is the later of
  `entry.created_at + 30 min` (`FIRST_WINDOW`) and the end stored in
  `hass.data[PAIRING_WINDOWS][entry_id]`, or None once both have passed. The
  **Pair new heaters** button (`button.<gateway>_pair_new_heaters`, entity
  category config, unique id `<dev_id>:pair_heaters`) calls
  `async_open_pairing_window` (now + 10 min, `BUTTON_WINDOW`), which sends
  `signal_radio_pairing_window(entry_id)`. The binary sensor **Pairing open**
  (`pairing_open`) is on while the window is open; its `ends_at` attribute is
  the end (ISO), and it schedules its own state change for that moment (the
  timer is cancelled when the entity is removed). Both entities exist only
  when `pairing_supported(runtime)`: the `heater_discovery` capability and a
  client that may transmit. The window lives in `hass.data`, so it survives
  the reload that adds a heater: several heaters pair in one window. An
  entry that is still learning its network (no heaters yet) opens a
  30-minute window at every setup. Removing the entry drops its window.
- **Running entry.** `RadioListener` classifies every frame first
  (`pairing.classify_announcement`, any network id, any swept destination).
  A direct announcement is sent as `signal_radio_pairing_announcement`;
  relayed dialect-A copies are dropped. Listening is passive: no lock is
  taken. `HeaterPairer.async_announced` ignores the announcement (DEBUG log)
  when the window is closed, while a pairing run holds the entry's
  `radio_pairing_lock` (the network move holds it for its whole run), while its own attempt runs, or for 30 s
  (`RETRY_PAUSE_S`) after an attempt that paired nothing. Otherwise it runs
  one `client.async_pair(20 s, max_heaters=1)` (`PAIR_ATTEMPT_S`): the
  exchange lock is held for that attempt only, far below the 150 s keepalive
  limit (#1040); a dialect-B sweep reaches this station within a few
  seconds. New heaters get the lowest id not used by a stored node nor
  assigned since setup (`reserved_ids`). A dialect-A announcement carries the
  identity, so a known serial is paired back to its stored address
  (`wanted_id`); a known dialect-B heater gets a new id and its node moves
  (`merge_heaters`). After the pairing, `client.async_restore` sends the
  clock; a known heater also gets its saved settings
  (`saved_snapshot`, else the last state Home Assistant saw), which are then
  dropped. The heater goes to `HeaterFinder.async_paired`: it joins the
  10 s found-heater batch (identity already read), is stored, the entry
  reloads once and the "Found TermoWeb heater" notification is shown.
- **Network-learning entry.** It runs on the site network, so pairing works
  the same way. The first pairing stores the client's dialect and network as
  the entry's own, which ends network learning.

## Heater identity and names

A heater's radio address (`addr`) follows the pairing order, so it can change
when the heater is paired again. Its serial does not. Each stored node keeps
both: `addr` addresses radio frames; `serial` names and identifies it.

- **Serial token** (`protocol.serial_token`, `IdentityRecord.serial`): the
  ASCII serial at the end of a dialect-B `5B 55` identity reply. A dialect-A
  heater (`5B 77` E0 reply, or its pairing announcement) has no printable
  serial: its token is the last 8 hex digits (lowercase) of the 12-byte
  identity, e.g. `1a2b3c4d`. An ASCII serial that is not 4-32 characters of
  `A-Z a-z 0-9 . _ -` with at least one digit also falls back to the hex form.
- **Default name**: `Heater <serial>` (`Thermostat <serial>` for a
  thermostat), set by `radio_pairing.radio_node`/`with_serial`. A node keeps a
  name the user gave it; only the address default `Heater <addr>` is replaced.
  The device name in Home Assistant is the user's to change.
- **Where the serial comes from**: pairing (`PairedHeater.serial`, from the
  `5A` reply or the dialect-A announcement), the network move (the identities
  it reads anyway), background discovery (it adds only heaters that answer
  `5A`), and setup. A rescan only probes `B8`, so rescanned heaters get their
  serial at the next setup.
- **Setup** (`radio_identity.async_learn_serials`): before the inventory is
  built, every stored node without a serial gets one `5A` read (10 s limit).
  Learned serials are saved in the entry data. A heater that does not answer
  keeps its address identity for this run; an INFO line says so, and the next
  setup tries again.
- **IDs**: device identifiers and entity unique IDs use the node key
  (`identifiers.node_key`): the serial when known, else the address. See
  [architecture: entity unique IDs](architecture.md#entity-unique-ids).
- **Registry move** (`radio_identity.async_rekey_node`): at every setup, each
  node with a serial has its address-keyed device and entities (if any) moved
  to the serial in the device and entity registries. Entity IDs, history and
  customisations stay. A target that is already taken is skipped with a
  WARNING.
- **Pairing again**: `add_nodes` matches a paired heater by serial. A known
  serial at a new address updates that node's `addr`; name and serial stay, so
  its devices and entities stay the same. The network move does the same and
  moves the registry entries of heaters that had no serial yet.
- **Radio address sensor**: each radio heater has a diagnostic sensor
  `Radio address` (`radio_address`, disabled by default) whose value is the
  current address. Its unique ID uses the node key, so it survives pairing
  again while its value changes. The serial is the device's `serial_number`.
- **Privacy**: serials identify a device. Logs at INFO and above mask them
  (`mask_identifier`); diagnostics redact the `serial` field and the serial
  inside node names.

## Known gaps

- Keypad lock on dialect B: acked; whether the keypad locks is not yet confirmed.
- Boost, runback and EASY on dialect B: `D2` answers `D3 00 00`, `D6` answers `D7 19`, `D4` is ack only; no effect seen. Their meaning is unknown.
- Display flash on dialect-B heaters: the command is acked, the visible flash is not yet confirmed.
- Energy counter: dialect A has `BC` → `BD` + u32 Wh (not used yet; the
  estimate is used for both dialects); dialect B has none, so energy is
  estimated.

## Setup (config flow)

The first step is a menu: **cloud** (the existing brand + login form),
**radio** (the ESP32 gateway over TCP) or **nanocul** (a nanoCUL USB stick).

The ESP32 gateway is normally found by zeroconf instead. Its ESPHome image
(`esphome/termoweb-radio.yaml`) advertises `_termoweb-radio._tcp` on port
2323 with TXT records `mac` (WiFi MAC, lowercase hex: the `dev_id`) and
`version` (radio firmware). The manifest's `zeroconf` key starts
`async_step_zeroconf`:

- IPv6-only advertisements abort (`not_ipv4_address`).
- The unique id `radio:<mac>` comes from the TXT `mac`, so nothing is
  opened before the user confirms. Without `mac` the gateway is probed.
- A configured gateway at a new address: `_abort_if_unique_id_configured`
  updates `host`/`port` and reloads the entry; a listen-only entry of the
  same gateway (`radio_monitor:<mac>`) is moved too.
- Otherwise `zeroconf_confirm` asks "Set it up?", probes the gateway on
  submit (a different MAC re-keys the flow) and continues with the
  `radio_method` menu below.

No form asks for a network id: discovery learns it from heater traffic,
pairing derives the site network (`async_site_network_id`), and existing
entries keep the stored `network_id`. Probing a gateway or stick that another
entry uses is allowed: Home Assistant takes the link over (the ESP32 drops
its old client; a serial port has one reader).

nanoCUL sticks speak the same line protocol over a serial port at 115200 baud
(`backend/radio/serial_link.py`: an `open_connection` for `RadioLink` built
on pyserial-asyncio-fast; any pyserial URL works, including `socket://`). The
`nanocul` step lists the host's serial ports (by-id paths) and offers a manual
path. The form pre-selects the most likely port without opening any: a port
whose by-id path, description, manufacturer or product names CUL (`nanoCUL`,
`CUL868`, `culfw`), else the only USB serial port that no TermoWeb entry
uses, else none. There is no USB discovery matcher: the supported nanoCUL is
an ATmega328P Arduino Nano behind a generic FTDI/CH340 USB-serial bridge,
whose ids also match Zigbee sticks and other serial devices. Stock
`termoweb_rx` firmware differs from the ESP32 port in two ways
that the flow and the link detect from the `Q` line:

- no `dialect=` field: the firmware only speaks dialect A. Connecting in
  dialect B raises `UnsupportedDialectError`. **Add the gateway** with such
  a stick aborts (`nanocul_firmware_too_old`: update the firmware); listen
  only still works. Reconfiguring an entry that detected dialect B onto such
  a stick aborts the same way.
- no `mac=` field: the device id comes from the USB serial number of the
  chosen port (else a hash of the path) and is stored as `radio_device_id`;
  `list_devices` falls back to it.

Entry data for a stick: `radio_type: nanocul`, `device` (path or URL),
`radio_device_id`, plus the dialect, network id and nodes below. ESP32
entries store `radio_type: esp32`; entries without the key are ESP32
entries. Reconfigure changes the port and can re-scan.

The manual radio form asks only for the host. The port (2323) sits in a
collapsed **Advanced settings** section (`advanced`); the nanoCUL forms ask
only for the port. There is no dialect choice: it is always detected.

1. `probe_gateway` connects with auto-ack off (nothing is transmitted) and
   reads the gateway MAC from the `# Q` line. The MAC becomes the `dev_id`
   and the entry's unique id (`radio:<mac>`). No MAC → error
   `no_gateway_mac`. A menu (`radio_method`) then offers **Add the gateway**
   (`radio_discover`, below; it finds known heaters and pairs new ones by
   itself, see [Automatic pairing](#automatic-pairing-pairing-window)) or
   **listen only**
   (`radio_monitor`): an entry with `brand: radio_monitor`, no heaters and no
   network id, which never transmits and records traffic with the
   `radio_capture` service
   ([radio protocol §10](research/protocol.md#10-listening-only-and-frame-captures)).
   It has its own unique id, `radio_monitor:<dev_id>`; only one entry can
   use the gateway's port at a time (the newest connection wins). A listen-only entry cannot be
   reconfigured (`monitor_reconfigure`): delete it and add it again.
2. `radio_discover` ("Add the gateway") creates the
   entry at once, with no progress step and no heaters
   (`radio_new_entry_data`): `brand: radio`, the connection, `nodes: []`,
   and no dialect: `network_id` is missing, which means "network still to
   be learned".
   Heaters are found after setup, see
   [Background heater discovery](#background-heater-discovery).
3. Once known, the entry stores `dialect`, `network_id` (hex) and `nodes` as
   `[{"type": "htr", "addr": "6", "name": "Heater 6"}]`. Addresses are plain
   decimal strings: the listener matches frames by `str(src)`. Once known, a
   node also stores its `serial` and is named `Heater <serial>` (see
   [Heater identity and names](#heater-identity-and-names)).

Errors: `cannot_connect_radio`, `no_gateway_mac`.

Reconfigure changes host and port. With **Scan for heaters again** it
re-runs the scan (`radio_rescan`, a progress step running `discover_radio`
with the stored dialect and network id, so it never listens) and replaces
the node list (error `no_heaters` when no heater answers). An entry that is
still learning its network has nothing to scan: the address is saved and
the entry reloads.

Setup builds the client with `create_radio_client(host, port, dialect, nodes,
network_id)`. `RadioLinkError` and `RadioError` from `list_devices` raise
`ConfigEntryNotReady`, so Home Assistant retries while the gateway is
offline. Unload stops the listener, which closes the connection.

### Background heater discovery

`radio_discovery` finds heaters after setup; setup never waits for them.
It runs only for backends with the `heater_discovery` capability (the radio
gateway backend, not listen-only entries).

1. **Network learning** (`network_pending`: no `network_id` in the entry).
   The entry still sets up fully (client, listener, gateway device, the
   pairing button and sensor): `async_learning_network` gives the client the
   installation's site network (`async_site_network_id` of the gateway id in
   the entry's unique id) and the chosen dialect, else B. A nanoCUL with
   stock firmware (no runtime dialects, checked with `probe_nanocul`) can
   only speak dialect A, whose network is fixed: `A`/`1B30` is stored at
   once and the entry sets up normally. Heaters can be paired into the site
   network meanwhile (see
   [Automatic pairing](#automatic-pairing-pairing-window)); the first pairing
   stores the site network as the entry's network, which ends learning.
   `NetworkLearner` learns the network of heaters that are already paired
   elsewhere (a stock gateway, another Home Assistant):
   - The listener sends every valid data frame from another network (not an
     ack, not from `FF`) as `signal_radio_network_heard(entry_id)`. A network
     heard twice (`NETWORK_FRAMES`) is adopted. Nets `0000` and `FFFF` are
     ignored: a dialect-B pairing sweep is sent on `0000`. A heater that wants
     heat sends a power-request spell (5 frames) about every 5 minutes; an
     idle heater may stay silent, so pairing it is the way in.
   - Every 10 minutes (`NETWORK_RETRY_S`) a 120 s raw survey runs through the
     client (gateway firmware 3.7-esp32+; without it, surveys stop). It
     decodes any dialect: a known dialect's first network id (not `0000`) is
     adopted (dialect A: `1B30`). The survey holds the exchange lock, which
     is safe: a learning entry has no heaters to keep alive (#1040). An
     unknown dialect saves the report file, shows a persistent notification
     ("TermoWeb heaters not recognised") with the file and the issue link,
     and stops; a reload listens again.

   An adopted network is stored (`dialect`, `network_id`) and the entry
   reloads into a normal setup with no heaters. A learning entry does not
   scan addresses on its site network.
2. **Scan** (network known, no heaters). `HeaterFinder.async_scan` asks every
   address 2-32 for its `5A` identity through the running client, 0.2 s
   apart (`SCAN_GAP_S`). Each read holds the exchange lock for one exchange
   only.
3. **Heard heaters** (network known). `RadioListener` drops frames from
   other networks and frames not sent to its station. A valid data frame
   from an address that is not in the inventory is sent as the dispatcher
   signal `signal_radio_unknown_heater(entry_id)`. `HeaterFinder` collects
   the addresses for 10 s (`FOUND_DEBOUNCE_S`), so heaters heard together
   cause one reload, then reads each one's `5A` identity (10 s limit). An
   address that does not answer is asked again at most every 10 minutes
   (`IDENTIFY_RETRY_S`).
4. **Store.** Only heaters that answer the identity read are added, so each
   one has its serial. `radio_pairing.merge_heaters` matches by serial
   first: a known serial at a new address moves that node (as pairing does),
   so its devices and entities stay. An address that is already stored is
   not added again. Changed nodes are saved with `async_update_entry` and
   the entry reloads (`async_schedule_reload`): the inventory is only built
   at setup. The identity read also notes the heater's rated power. Each new
   heater gets a persistent notification: "Found TermoWeb heater
   `<serial>`". Logs mask the serial. After a store the finder stops until
   the reload.

Unload cancels the search and the scan (entry background tasks) and the
pending debounce timer.

