# TermoWeb 869 MHz radio protocol

Technical reference for `custom_components/termoweb/backend/radio/`. This
package talks to TermoWeb / Sun Ray RF heaters directly over radio, through
an ESP32-S3 + CC1101 radio gateway. It does not use the TermoWeb cloud.

The radio backend (`docs/radio_backend.md`) uses this package.

## 1. Radio

| Item | Value |
|---|---|
| Carrier | 869.525 MHz |
| Modulation | 2-FSK, 50.8 kHz deviation |
| Data rate | 9.6 kbps |

## 2. Frame layout

Logical bytes. "Logical" means after descrambling in dialect A. In dialect B
the on-air bytes are the logical bytes.

| Offset | Size | Field |
|---|---|---|
| 0 | 1 | length byte (rule depends on the dialect) |
| 1 | 2 | network id |
| 3 | 1 | src: this hop's sender |
| 4 | 1 | dst: this hop's receiver |
| 5 | 1 | flags (`80` = ack, or "fire and forget" on a data frame) |
| 6 | 5 | path: originator, then hops, padded (`00` from a station, `01` from a heater) |
| 11 | 1 | tag (`00` commands, `03` dialect-B pairing announcement, `04` id assignment, `06` route probe) |
| 12 | n | payload |
| 12+n | 2 | CRC, high byte first, over bytes `0 .. total-3` |

An **ack** is 8 bytes: `len net(2) acker sender 80 crc(2)`. It has no path,
tag or payload.

## 3. Dialects

The layout above is the same in both dialects. Only the framing differs.

| | Dialect A | Dialect B |
|---|---|---|
| Sync word | `2DE5` | `2DD4` |
| Byte 0 | total length − 3 | total length |
| Whitening | PN9, whole frame from byte 0 | none |
| CRC | CRC-16/CCITT, poly `1021`, init `1D0F`, xorout `FFFF` | CRC-16/MODBUS, reflected poly `A001`, init `FFFF`, no xorout |
| Network id | `1B 30` on stock gateways; a network Home Assistant pairs uses its own id (section 7) | per installation, learned from traffic (no default) |
| Ack | logical `05 net acker sender 80 crc` (scrambled) | `08 net acker sender 80 crc` |
| EB clock sync | 9 bytes, ends `03` | 8 bytes, no `03` |
| Gateway `Y` mode | `0` | `1` |

### PN9 whitening (dialect A)

`x^9 + x^5 + 1`, seed all ones, one byte per eight shifts, bytes assembled
MSB first. The keystream starts `FF 87 B8 59 B7 A1 CC 24`. It is the
bit-reverse of the table in TI DN509. Because the first keystream byte is
`FF`, the total length is `(air[0] XOR FF) + 3`.

### Dialect B network id

Each dialect-B installation has its own two-byte network id. There is no
shared default, so the code never assumes one: `build_frame`, `build_ack` and
`RadioLink` raise `ValueError` for dialect B unless a network id is passed.
Learn the id from the heater's own traffic: every frame carries it in logical
bytes 1-2, for example the registration and power-request frames a heater
sends on its own. The examples below use the synthetic id `12 34`.

### Test vectors (dialect B, heater 06, synthetic network id `12 34`)

| Frame | Bytes |
|---|---|
| Registration, 06 → 01 | `0F 12 34 06 01 00 06 01 01 01 01 00 50 72 22` |
| Route probe, 06 → 02 | `0E 12 34 06 02 00 06 02 01 01 01 06 F6 91` |
| Ack, station 01 for 06 | `08 12 34 01 06 80 A0 E4` |

### Test vectors (dialect A)

| Frame | Bytes |
|---|---|
| Setpoint 25.5 C, 01 → 04 | `F1 9C 88 58 B3 A1 CD 20 57 5E 4B 9C BA EB D9 F6 E5` |
| Ack, 04 for 01 | `FA 9C 88 5D B6 21 FB D1` |

## 4. Opcodes

Status column:

- **A** — proven on air in dialect A (ha-termoweb-local captures).
- **B** — observed on air in dialect B (heater 06, 2026-10-09).
- **?** — not yet tried or not understood in that dialect.

A reply's first payload byte is the request opcode + 1. A two-byte
`<op+1> 55` means accepted, `<op+1> 56` means rejected.

| Payload | Direction | Meaning | A | B |
|---|---|---|---|---|
| `B4 01` / `B4 02` / `B4 04` | station → heater | mode auto / manual / off, reply `B5 55`. Dialect B acks it but ignores it: no reply, mode unchanged | A | ignored |
| `B4 02 <half-deg>` | station → heater | manual setpoint, 7–35 C | A | ? |
| `B4 03 <half-deg>` | station → heater | temporary override setpoint | A | ? |
| `B6 <af> <eco> <comfort>` | station → heater | preset temperatures, strictly increasing, reply `B7 55` | A | rejected |
| `B6 <af> <eco> <comfort> <mode> [<setpoint>]` | station → heater | presets plus mode (01 program / 02 manual / 03 temporary override / 04 off), and for 02/03 the target setpoint in half degrees; dialect B's only mode and setpoint write, reply `B7 55`. Mode 03 ends at the next program change. | ? | B |
| `B2` + 84 bytes | station → heater | weekly program write, 48 slots/day, Sunday first, reply `B3 55` | A | rejected |
| `B2` + 42 bytes | station → heater | weekly program write, 24 slots/day, same layout as the `B1` read, reply `B3 55` | ? | B |
| `D2` / `D4` / `D6` / `BA` `01|00` | station → heater | boost / runback / EASY / keypad lock toggles, reply `<op+1> 55` | A | ? |
| `C4` + 8 bytes | station → heater | advanced setup record, reply `C5 55/56` | A | ? |
| `5E 01` | station → heater | flash display, reply `5F 55` (dialect B: ack only) | A | B (acked) |
| `57 55` | station → heater | confirm a `56 ..` report | A | ? |
| `51|52 YY MM DD DOW HH MM SS [03]` | station → heater | clock sync (`51` while registering), DOW 0 = Sunday, reply `53 55` | A (9 bytes) | B (8 bytes; the 9-byte form gets `53 56`) |
| `BF 01` | station → heater | grant a `BE` power request | A | B (the repeats stop) |
| `B8` → `B9 ..` | request / reply | status: E6 (14 bytes), E4 (16, with boost tail) | A | B (short 5 bytes: `B9 af eco comfort mode`) |
| `B0` → `B1 ..` | request / reply | program: 85 bytes (48 slots/day) or 43 bytes (24 slots/day) | A | B (43 bytes) |
| `BC` → `BD ..` | request / reply | dialect A: `BD` + u32 Wh energy counter | A | B: 9-byte power record, **not** energy (see below) |
| `5A` → `5B ..` | request / reply | identity: `5B 77 01 01 01 <id>` + 12-byte tail + `0A 1A` | A | B (`5B 55` + 16 bytes ending in the ASCII serial) |
| `DA` → `DB ..` | request / reply | advanced record, 17 bytes | A | ? |
| `C2` → `C3 55` | request / reply | meaning unknown | A | B (ack only) |
| `D0` → `EC`-class reply | request / reply | capability, meaning unknown | A | B (ack only) |
| `C6` → `C7 ..` | request / reply | meaning unknown | A | B (`C7 55`) |
| `C8 <p> <v>` | station → heater | write parameter, reply `C9 55`. `C8 01 D0` is the factory reset (section 8); no other value is used | ? | B |
| `50` | heater → station | registration; the station answers with a `51` clock sync | A | B |
| `56 B9 ..` | heater → station | status report (E5, 15 bytes; E3, 17 bytes with boost tail) | A | ? |
| `56 DB ..` | heater → station | advanced record report (E2) | A | ? |
| `56 B1` + 84 bytes | heater → station | program report (9E class) | A | ? |
| `BE ..` | heater → station | power request: the element wants to switch on | A (`BE hi lo`, 3 bytes) | B (9 bytes) |
| empty, tag `06` | heater → any | route probe | A | B |
| `55`, tag `03`, from `FF` | heater → any | dialect-B pairing announcement (section 7) | – | B |
| `77` + 12-byte identity, from `FF` | heater → any | dialect-A pairing announcement (section 7) | A | – |
| `<id>`, tag `04`, to `FF` | station → heater | id assignment (section 7) | A | B |

### Status record fields (E6 offsets, payload without the `56` marker)

`B9 af eco comfort mode room_hi room_lo setpoint power_hi power_lo duty flags b12 b13 [boost_hi boost_lo]`

- Presets and setpoint: half degrees. Room: tenths of a degree, big-endian.
- Power: measured full-load power, deciwatts, big-endian.
- Flags: `01` heating, `02` locked, `04` presence, `08` window open,
  `10` true radiant, `20` boost, `40` EASY, `80` runback.
- Boost tail: high 4 bits = day (0 = Sunday), low 12 bits = minute of day.
- Mode: `01` auto, `02` manual, `03` override, `04` off.

The dialect-B short form stops after `mode`. The decoder returns None for
every field it does not carry.

### Dialect-B power record (`BE ..` and the `BD ..` reply to `BC`)

`<op> <room> 00 <setpoint> <v lo> <v hi> <duty> <heating> 00`, for example
`BE DC 00 2C 78 3A 00 00 00` (idle) and `BD DB 00 2C E4 3A 0C 01 00`
(heating).

The fields were checked against a whole-house energy meter during a heat test:

- Byte 7: `01` while the element heats, else `00`; the heater heats in duty
  pulses while this flag is set. Heating follows the active setpoint against
  the room temperature: answering `BE` with `BF 01` or `BF 00` made no
  visible difference.
- Bytes 4–5, little-endian ÷ 64: mains voltage in volts (`78 3A` = 233.9 V).
  It tracked the meter's voltage, including the dips during heating pulses.
- Byte 6: duty in percent (`0C` while heating ~11 % of the time, `64` =
  100 % right after a large setpoint step).
- Byte 1: the heater's own room-temperature probe, in tenths of a degree
  (`DB` = 21.9 °C): the value the heater regulates on. Overnight, with the
  heater idle, it followed the room's fall in step with a reference sensor
  placed next to the probe, reading 1.0–1.4 °C higher (5 readings, 00:00–05:00).
- Byte 3 (byte 2 is `00`): the active setpoint in half degrees: the manual
  setpoint in manual mode, the slot preset in program mode, the override
  target in override (`2C` = 22.0 °C manual, `25` = 18.5 °C night slot).
- The record carries no power value. An earlier reading of bytes 3–4 as
  deciwatts (`2C F0` ≈ 1150 W) was a coincidence of the 22.0 °C setpoint.

### Program record

Four 2-bit slot codes per byte, MSB first, day-major, **Sunday first** on the
wire. Codes: `0` cold (anti-frost), `1` night (eco), `2` day (comfort);
`3` is never seen and decodes to None. Slots per day = record length × 4 / 7:
24 for the 43-byte reply, 48 for the 85-byte one. The package API takes and
offers Monday-first weeks and rotates at the boundary.

## 5. Gateway line protocol (TCP port 2323)

One client at a time. Each new connection resets the gateway session
(station id, auto-ack, dialect, network id). Lines end with `\r\n`.

### Gateway → client

| Line | Meaning |
|---|---|
| `# termoweb_rx <ver> freq=.. rate=9.6k sync=.. mode=dynamic tx=pa..` | banner, first line of every connection (also the `V` reply) |
| `# Q termoweb_rx <ver> freq=.. pa=.. sync=.. mode=dynamic autoack=on|off id=<hex> [mac=..]` | `Q` reply |
| `RX <micros> <rssi_dbm> <lqi> <crc_ok_bit> <HEX>` | received frame, raw on-air bytes; the client ignores the CRC bit and decodes with its own dialect |
| `TX <micros> <n> <HEX>` | a `T` frame went on air |
| `TXERR empty or bad hex` | the `T` line arrived garbled; the client resends once |
| `TXERR <other>` | transmit failed; the client raises |
| `ACK <micros> <HEX>` | the firmware sent an auto-ack |
| `RAWB <n> <rssi_dbm>` / `RAW <n> <runs…>` / `RAWE <n> <count>` | survey burst `n`: start, run lengths (`+µs` high, `-µs` low, demodulator output), end |
| `# ...` | any other status line |

### Client → gateway

| Command | Meaning | Reply |
|---|---|---|
| `T<hex>` | transmit raw on-air bytes (up to 255) | `TX` or `TXERR` |
| `I<hex2>` | set this station's id (default `01`) | none |
| `A0` / `A1` | firmware auto-ack off / on | none |
| `Q` | status | `# Q ...` |
| `V` | banner | `# termoweb_rx ...` |
| `X` | toggle raw mode (appends 2 status bytes to RX lines) | `# raw=on|off` |
| `D` | chip state dump | `# marcstate=..` |
| `F<kHz>` | retune the carrier | `# freq=..` or `# F? expected kHz` |
| `R<seconds>` | raw survey: capture every RF burst without sync, length or CRC rules, then return to normal reception | `# survey on secs=N`, survey lines, `# survey off` |
| **`Y<0|1>`** (new) | select the dialect framing: sync word, length rule and ack format. `0` = dialect A, `1` = dialect B | `# ...` |
| **`N<hex4>`** (new) | network id the firmware writes into its auto-acks, e.g. `N1234` | `# ...` |

`Y` and `N` are new. The firmware must implement them before this package can
be used. Older firmware ignores unknown letters but parses the following
characters as new commands, so `N` followed by a network id that contains
`A` or `D` would be misread there (`A` sets auto-ack, `D` dumps chip state).

### Connect sequence used by `RadioLink.connect()`

1. Open TCP, wait for the `# termoweb_rx` banner.
2. Send `I<id>`, `A1`, `Y<dialect mode>`, `N<network id>`, then `Q`.
3. Parse the `# Q` line into `GatewayInfo`. A sync word that does not match
   the dialect is logged as an error.

### Transmit rules

- All transmits are serialised with one lock.
- At least 40 ms between command lines.
- `send_frame` waits for `TX`, then for the heater's ack (flags `80`, from the
  destination, to this station). No ack within 160 ms: resend, up to 3 times.
- `TXERR empty or bad hex`: wait 50 ms and resend the same line once.

## 6. Surveying an unknown dialect

Two dialects are known, and there may be more. A heater that speaks another
one stays invisible in packet mode: the radio drops every frame whose sync
word does not match. The gateway's survey (`R<seconds>`) switches the CC1101
to asynchronous serial mode and reports each burst above its carrier
threshold as demodulator run lengths, which assume no sync word, framing or
bit rate. `RadioLink.survey(seconds)` collects them as `RawBurst`s; a burst
whose `RAWE` line was lost is still kept.

`survey.analyse(bursts)` then, for every burst:

1. merges runs of the same level and folds glitches shorter than 30 µs into
   their neighbours;
2. finds candidate bit timings: stretches of single-bit runs with steady pair
   sums (a few bad pairs, from flipped bits or glitches, are tolerated) give
   the bit period (median half pair sum) and the demodulator's high/low
   stretch. When no stretch is clean, 104 µs ± 10 % and the 4.8, 19.2 and
   38.4 kbps periods are tried as well;
3. for each timing, folds glitches shorter than 0.35 bit, refines the period
   over the whole burst (total time / total bits: 2 % off is enough to miscount
   a run of eleven equal bits), and samples the runs with a phase-tracking bit
   clock that is pulled half-way to mid-bit at every transition, so one
   jittered edge does not shift the bits after it. Idle gaps split segments;
4. slides each known dialect's sync word over every bit offset, in both
   polarities, accepting up to 2 bit errors, and decodes with
   `dialect.decode`: one CRC-valid frame is enough for a known dialect;
5. otherwise takes the longest bit-level preamble (at least 16 bits; a lone
   error is skipped when 8 alternating bits follow it) and searches the
   framing after it: frame start 8–40 bits after the preamble's end (the 16
   bits before it are the sync word), polarity, whitening (none or PN9), the
   length rule (total = byte 0 + k, k = −4…+4), 14 CRC-16 parameter sets (the
   two known ones plus the usual CCITT, ARC, MODBUS, USB, MAXIM, CMS, DNP…
   variants), the CRC covering byte 0 or byte 1 onwards, and big- or
   little-endian CRC bytes;
6. counts, per framing, the bursts whose CRC validates. A single match can be
   chance (about 16 000 trials per burst), so a framing becomes a candidate
   only when at least two bursts agree on it, sync word included.

On synthetic dialect-B bursts with ±15 % edge jitter, 3 glitches and 3 % bit
errors in the preamble and sync, 38 of 40 decode. A bit error inside the
frame breaks its CRC, so with errors everywhere only the frames without one
can decode (about 30 % at 1 %).

`analyse(bursts, keep_runs=True)` keeps every burst's raw runs in the report
(`raw_runs`) so a survey can be replayed after a decoder change.

The `SurveyReport` verdict is `known` (with the network ids heard), `candidate`
(with a ready-to-paste `Dialect(...)` suggestion and what the codec would
still need, such as a little-endian CRC), `undecodable` (the bits of each
burst are kept for a human) or `silent`.

Before a report is shared, `survey.redact(report)` zeroes the network id
bytes (1–2) of every decoded frame and cuts identity replies (`5B`) after
their opcode, and drops kept raw runs. Undecodable bursts keep their raw bits,
which may still contain the network id: `contains_raw_bits` in the report
says so.

### From Home Assistant

- `RadioLink.survey` needs ESP32 firmware 3.7-esp32 or newer
  (`link.supports_survey(GatewayInfo)`); stock nanoCUL firmware has no `R`.
- Config flow: when `discover_network` hears nothing, `survey_sighting` runs a
  120 s survey (`discovery.survey_network`, auto-ack off, nothing
  transmitted) and analyses it in the executor. `known` continues the setup
  with the dialect and network id it heard; `silent` (or no survey firmware)
  gives `no_traffic`; anything else saves a redacted report as
  `<config>/termoweb_radio_survey_setup_<UTC>.json` and shows
  `unknown_dialect` with the file name and the issue link.
- Service `termoweb.radio_survey` (`entry_id`, `seconds` 10-600, default 120)
  runs `RadioClient.async_survey` under the exchange lock, saves
  `termoweb_radio_survey_<entry_id>_<UTC>.json` and returns a summary
  (`verdict`, `dialect`, `bursts`, `confidence`, `suggestion`, `file`). The
  summary is kept as `runtime.last_radio_survey`.
- Diagnostics of a radio entry have a `radio` section: radio type, dialect,
  connection, gateway firmware/frequency/sync/dialect/auto-ack/station id,
  survey support and the last survey summary. MAC and network id are left
  out.
- Report file: `format`, `created`, `integration_version`, `radio_type`,
  `configured_dialect`, `gateway` (firmware, freq, sync; no MAC),
  `survey_seconds`, and `report` (`redact(report).as_dict()`).


## 7. Pairing

A heater joins a network when its owner puts it into pairing mode on its
panel. It then announces itself from the broadcast id `FF`, sweeping the
destination across the id range. The station answers with one **id
assignment**: tag `04`, payload the new id, from station `01` to `FF`, path
`01 FF 00 00 00`, on the station's network id. The heater adopts the id and
the network id.

| | Dialect A | Dialect B |
|---|---|---|
| Announcement | payload `77` + 12-byte identity (13 bytes), tag `00` | payload `55`, tag `03`, net `00 00`, path `FF <dst> 01 01 01` |
| Sweep | about 200 ms per destination | about one frame a second (`01, 04, 07, 08, 0A, 0B, 0E, …`) |
| Identity | in the announcement | none: heaters are paired one at a time |
| After the assignment | (ha-termoweb-local captures) registration `50` | link ack from `FF` on the new net, an empty tag-`84` frame, then `50` registrations from the new id; `B8` reads answer at once |
| Status | documented by ha-termoweb-local, not yet tried here | proven 2026-10-10 |

Assignment test vector (dialect B, id 06, synthetic network id `12 34`):
`0F 12 34 01 FF 00 01 FF 00 00 00 04 06 D0 57`.

`pairing.pair_heaters(link, ...)` runs on a connected `RadioLink` and uses the
link's dialect and network id:

1. Every announcement in the link's dialect is queued with the time it was
   heard.
2. The id is `wanted_id` when given (a heater re-paired to its old address),
   else, in dialect A, the id this identity got earlier in the run, else the
   lowest id in `02..41` that no stored node uses. No free id raises
   `NoFreeAddressError`, which carries the heaters paired before.
3. The assignment goes out with the normal link retries. Without an ack the
   next announcement is answered again.
4. After an ack, announcements heard during the next 5 s are ignored: they
   are the same sweep. The new id must then answer a `B8` status read (3
   tries of 1 s). A `5A` identity read follows; its tail (and, in dialect B,
   the ASCII serial) is kept when the heater answers.
5. Dialect A: one assignment per identity per 200 ms. An identity that is
   already paired in this run is not answered again.

**Relayed dialect-A announcements are ignored.** Already-paired heaters
forward copies of an announcement under their own source id (the link sender
differs from the path's originator). The relay sits on its own network, so an
assignment sent through it on the station's network id would be dropped.
A heater out of the gateway's direct range is paired by moving the gateway
(or the heater) closer.

`pair_heaters` stops after `max_heaters` heaters (one with `wanted_id`), or
when the window ends. With `idle_stop_s`, each announcement keeps the window
open that long (up to `total_s`), and after a pairing the run ends
`idle_stop_s` after the last announcement.

`pairing.pair_new_network(host, port, network_id)` is for a new installation
of unknown dialect: it opens one link per dialect in turn (15 s slices, 5
minutes in all, stop 60 s after the last pairing) and returns the first
dialect that pairs a heater. A dialect the gateway firmware cannot speak is
dropped.

### Site network id

Stock gateways have a per-installation network id. Home Assistant derives its
own: `pairing.site_network_id(seed)` takes the first two bytes of the SHA-256
of the seed that are not `00 00`, `FF FF` or `1B 30` (the stock dialect-A
id). The config flow seeds it with the Home Assistant instance id
(`homeassistant.helpers.instance_id`) plus the gateway id, `<instance>:<gateway
dev_id>`, so two gateways in one installation get two networks. The id is
stored in the config entry (`network_id`); heaters paired later join the
entry's network. A network id typed into the setup form wins over the
derived one.

### Notes

- The firmware does not filter received frames by network id; `N<net>` only
  sets the id in its auto-acks. Announcements on net `00 00` therefore reach
  the client.
- Right after pairing, a dialect-B heater took an `EB 51` clock sync but did
  not answer `53`; the restore below sends the clock with an ack only.

### From Home Assistant

- Config flow: after the gateway (or nanoCUL) step, a menu offers "Find
  heaters that are already paired" (discovery, section 6) or "Pair new
  heaters". Pairing runs `config_flow.pair_radio` →
  `pairing.pair_new_network` on the site network id, in the chosen dialect
  (`auto`: B and A in turn; a stock nanoCUL: A only), and creates the entry
  with the paired heaters. No heater paired: error `no_heaters_paired`.
- Options flow of a radio entry: a menu offers the settings form or "Pair a
  new heater". Pairing uses the running client (`async_pair`, 5 minutes,
  stops 60 s after the last pairing), adds the new heaters to the entry's
  nodes and reloads the entry.
- `RadioClient.async_pair(window_s, wanted_id=None, max_heaters=None)` pairs
  under the exchange lock: heater commands wait until it ends. New heaters
  get ids no stored node uses.
- Service `termoweb.radio_pair` (`entry_id`, optional `heater`, `restore`
  default true, `timeout` 30-600 s, default 300) pairs one heater:
  - with `heater` (a stored node): the heater gets that id back, then its
    settings are restored: the clock (ack only), then `B6 af eco comfort
    mode [setpoint]` (`B7 55`) and the `B2` program (`B3 55`), through the
    normal settings write. The settings come from the snapshot saved by
    `radio_factory_reset`, else from Home Assistant's current state. A
    temporary override is restored as program mode. The heater is then
    refreshed.
  - without `heater`: a new heater gets the lowest free id, is added to the
    entry's node list, and the entry reloads (the inventory is fixed while
    the entry runs).
  - The response is `{"heater", "added", "restored"}`.

## 8. Factory reset

Dialect B: `C8 01 D0` → `C9 55`. Within 3 s the heater goes silent. All
settings, the clock, the program and the radio pairing are wiped (mode off,
presets 5.0 / 17.0 / 19.0 °C, default program). It stays silent until its
owner puts it into pairing mode; the pairing above brings it back. Proven
twice on 2026-10-10, each time followed by a pairing and a restore whose
read-back matched.

`C8` with two argument bytes writes a parameter. Other values are accepted
and not yet understood; the integration sends no other value.

Dialect A: no reset is known. `protocol.factory_reset(DIALECT_A)` raises
`ValueError`; `RadioClient.async_factory_reset` raises
`RadioUnsupportedError`.

Service `termoweb.radio_factory_reset` (`entry_id`, `heater`) first saves
the heater's restorable settings (mode, manual setpoint, presets, program)
from Home Assistant's state into the entry options (`radio_restore`), then
sends the reset. The saved settings survive a restart and are used (and
removed) by the next successful `radio_pair` restore of that heater.
