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
shared default, so the code never assumes one: `build_frame` and
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
| `5E 01` | station → heater | flash display, reply `5F 55` (dialect B: ack only; what a real gateway sends for "identify" in dialect B is not yet captured, see section 10) | A | B (acked) |
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
| `C8 <p> <v>` | station → heater | write parameter: exactly two argument bytes, reply `C9 55` (`C9 56` for any other length). `C8 01 D0` is the factory reset (section 8); no other value is used | ? | B |
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
| Sweep | about 200 ms per destination | 120-160 ms per destination, for example `18, 19, 1C, 1D, 1E, 02 … 15, 01` |
| Answered frame | any direct copy (src and dst are not checked) | only the frame swept to this station (dst `01`): the heater acks an assignment only right after it |
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
3. Airtime is kept low. Dialect B answers only the sweep frame addressed to
   this station; frames swept to other ids get no transmission. Each
   assignment is sent once and waits 200 ms for the ack, with no link
   retries: a retry would land after the heater moved on. An announcement
   older than 100 ms is not answered (its dwell is over), and at most one
   assignment goes out per second. Without an ack, the next pass of the
   sweep is answered again. (Answering every sweep frame with three retries
   put 34 un-acked transmissions on air in 10 s during a live test.)
4. After an ack, announcements heard during the next 5 s are ignored: they
   are the same sweep. The new id must then answer a `B8` status read (3
   tries of 1 s). A `5A` identity read follows; its tail (and, in dialect B,
   the ASCII serial) is kept when the heater answers.
5. Dialect A: an identity that is already paired in this run is not answered
   again.

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
  The integration ignores acks, replies and unsolicited frames from any other
  network id, because neighbouring installations reuse the same short ids.
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
    normal settings write. A reset sets the manual target to 19.0 °C, so a
    heater that was not in manual mode first gets its last manual target
    with `B6 af eco comfort 02 <target>` (for example `B6 21 25 2A 02 28`
    = 20.0 °C; the `BD` setpoint byte reads it back), then its real mode.
    The client remembers the manual target from every read or write in
    manual mode; the snapshot keeps it as `manual_stemp`. The settings come from the snapshot saved by
    `radio_factory_reset`, else from Home Assistant's current state. A
    temporary override is restored as program mode. The heater is then
    refreshed.
  - without `heater`: a new heater gets the lowest free id, is added to the
    entry's node list, and the entry reloads (the inventory is fixed while
    the entry runs).
  - The response is `{"heater", "added", "restored"}`.

## 8. Factory reset

Dialect B: `C8 01 D0` → `C9 55`. Within 3 s the heater goes silent. All
settings, the clock, the program and the radio pairing are wiped: mode byte
`04` (off), presets 5.0 / 17.0 / 19.0 °C, manual target 19.0 °C, default
program. It stays silent until its owner puts it into pairing mode; the
pairing above brings it back. On 2026-10-10 the reset → pair → restore
sequence (assignment, clock sync without a `53` reply, `B6`, `B2`) ran four
times; each restore finished about 6 s after the pairing press and read back
correctly.

`C8 <param> <value>` writes a parameter: it takes exactly two argument bytes
(`C9 56` for any other length). `C8 01 D1` and `C8 01 D2` also reset the
heater. Other parameters are accepted and not yet understood. The
integration sends only `C8 01 D0` and exposes no other parameter write.

Dialect A: no reset is known. `protocol.factory_reset(DIALECT_A)` raises
`ValueError`; `RadioClient.async_factory_reset` raises
`RadioUnsupportedError`.

Service `termoweb.radio_factory_reset` (`entry_id`, `heater`) first saves
the heater's restorable settings (mode, manual setpoint, presets, program)
from Home Assistant's state into the entry options (`radio_restore`), then
sends the reset. The saved settings survive a restart and are used (and
removed) by the next successful `radio_pair` restore of that heater.

## 9. Moving heaters to the site network ("re-home")

An entry set up with "Find heaters that are already paired" keeps the
network id of the gateway the heaters came with. `radio_rehome.async_rehome`
moves every heater onto the site network id (section 7), in this order:

1. **Identity check, nothing changes yet.** Every stored heater must answer a
   `5A` identity read. A silent heater stops the move: the user moves the
   gateway closer and tries again.
2. **Save and reset.** For each heater: save the restorable settings (as
   `radio_factory_reset` does, including the manual target) under its old
   number in `radio_restore`, then send `C8 01 D0`. A failed reset stops the
   move; the network is not changed, and heaters reset before it are paired
   back later with `radio_pair` and their old number.
3. **Switch the station.** `RadioClient.async_set_network_id` sends `N<net>`
   on the running link (`RadioLink.set_network_id`) and keeps the id for
   reconnects. The site id is seeded with the Home Assistant instance id and
   the gateway's `dev_id`, as in the config flow.
4. **Pair.** `async_pair` with `max_heaters` = the number of heaters, 5 min,
   stop 60 s after the last pairing. New ids start at 2 and skip the old
   numbers, so a heater that is not paired keeps its node without a clash.
5. **Map new to old.** One heater: direct. More: the `5A` identity read at
   pairing (read again when it was missing) is matched against step 1.
6. **Restore** each matched heater from its saved settings (clock, `B6` with
   presets, mode and setpoint, the manual target first when the mode is not
   manual, then `B2`). The saved settings are dropped after a restore, or
   moved to the new number when the restore fails.
7. **Update the entry and reload**: `network_id` = the site id; matched
   heaters replace their old node under the new number and keep the user's
   name; paired heaters that were not recognised are added as `Heater <id>`;
   heaters that were not paired keep their old node and saved settings, so
   `radio_pair` with that number pairs and restores them later.

Once all heaters are reset the entry always moves to the site network, also
when nothing was paired in time: the result lists those heaters as
`waiting` and the summary tells the user to pair each one with `radio_pair`
and its number. The entry never stays on a network without saying so.

Dialect A: no reset over the radio is known, so the move is refused with a
message: reset dialect-A heaters on their panel, then set the integration up
again with "Pair new heaters".

From Home Assistant: the options flow item "Move heaters to this
installation's own network" (explanation, progress, then a summary), or the
service `termoweb.radio_rehome` (`entry_id`, `timeout` 30-600 s, default
300). The service returns `moved` (`from`, `to`, `restored`),
`unidentified`, `waiting` and a plain-English `summary`.

## 10. Listening only and frame captures

A listen-only ("monitor") entry records traffic next to a real TermoWeb
gateway, which is station `01` on that network. It must never transmit:
an ack or clock sync from a second station would disturb the installation.
It is used to learn unknown commands, first of all the dialect-B form of the
display flash ("identify"; `5E 01` → `5F 55` is proven in dialect A only).

### No-transmit guarantee

- `RadioLink(listen_only=True)` writes only `I<id>`, `A0`, `Y<0|1>`,
  `N<net>`, `Q`, `V` and `R<seconds>`. Any other command, above all `T`
  (transmit) and `A1` (auto-ack on), raises `TransmitBlockedError` before
  anything reaches the gateway. `listen_only` with `auto_ack=True` is a
  `ValueError`.
- `RadioClient(listen_only=True)` raises `TransmitBlockedError` in every
  command path (`_exchange`, `async_send`, `async_pair`) before a frame is
  built, and builds its link with `listen_only=True, auto_ack=False`.
- The monitor station uses id `FE` (no heater sends to it) and network id
  `0000` (never transmitted). The stock nanoCUL firmware ignores `Y` and `N`
  but would read the hex digits after them as commands; `Y0` and `N0000`
  contain no command letter.
- The push client (`RadioMonitor`) has none of the station duties: no clock
  syncs, `BF 01` grants, `57 55` confirmations or status polling.

### Monitor runtime

- Entry data: `brand: radio_monitor`, plus `radio_type` and the ESP32
  `host`/`port` or the nanoCUL `device`/`radio_device_id`. No dialect,
  network id or nodes. Title `Radio monitor (<host or serial path>)`. Unique
  id `radio_monitor:<dev_id>`, so a normal `radio:<dev_id>` entry for the
  same gateway can be added later. Only one entry can hold the TCP or serial
  port at a time.
- Backend `RadioMonitorBackend` (capabilities: `frame_monitor`, `local_radio`;
  no `site_device`, no `web_portal`). The
  device is named **Radio monitor**. Platforms: `binary_sensor` (gateway
  online) and `sensor` (**Frames heard**: frames since start, attribute
  `last_frame` (ISO UTC), pushed on the `termoweb_<entry>_radio_frames`
  signal). Home Assistant names it `sensor.radio_monitor_frames_heard`.
- ESP32 firmware (a `dialect=` field in `Q`): the monitor switches the
  dialect with `RadioLink.set_dialect` every 10 s, A then B
  (`MONITOR_DIALECTS`). Stock nanoCUL firmware (`termoweb_rx`, no `Y`) stays
  on dialect A.

### Capture service

`termoweb.radio_capture` fields:

| Field | Type | Default |
|---|---|---|
| `entry_id` | config entry, required | – |
| `seconds` | int 10–1800 | 120 |
| `redact` | bool | false |
| `note` | free text, stored in the file | none |

It works on monitor entries and, passively, on normal radio entries: it
holds the exchange lock like `radio_survey`, so heater commands (and the
listener's keepalive clock syncs) wait until it ends. Keep captures on a
normal entry short: heaters want a clock sync every 150 s.

`RadioClient.async_capture` adds a frame listener and a line listener
(`RadioLink.add_line_listener`: every line that is not a decoded frame,
except raw survey lines) for the window and returns a `FrameCapture`.

The file is `<config>/termoweb_radio_capture_<YYYYmmddTHHMMSSZ>.json`:

| Key | Content |
|---|---|
| `version` | 1 |
| `started`, `ended` | ISO UTC with milliseconds |
| `gateway` | the firmware's `# Q` line; `mac=` is always masked, `net=` too with `redact` |
| `dialects_listened` | `["A", "B"]` on a listen-only ESP32 entry, else the one dialect (`["A"]` on a stock nanoCUL) |
| `note` | the `note` field, or null |
| `redacted` | bool |
| `summary` | `frames` (data + acks), `acks`, `networks`, `nodes`, `opcodes` (`{"5E": {"count": 3, "name": "flash display (identify)"}}`; key `tag NN` for pairing tags and empty frames; name null when unknown) |
| `frames` | one item per decoded frame, below |
| `raw` | `{t, line}` for every firmware line that is not a decoded frame (`#` status, `TX`, `ACK`) and for frames no known dialect decodes, written as `RX <micros> <rssi> <lqi> 0 <HEX>` |

Each `frames` item: `t` (ISO UTC, ms, when Home Assistant read the line),
`kind` (`data` or `ack`), `rssi`, `lqi`, `micros`, `dialect` (a frame that
fails in the link's dialect, because it switched while the frame was on air,
is tried in every known dialect first), `net`, `src`, `dst`, `flags`,
`path`, `tag`, `payload` (null for acks), `op` (histogram key), `name`
(from `protocol.OPCODE_NAMES` / `TAG_NAMES`, or null) and `air` (raw on-air
bytes). All bytes are upper-case hex strings without spaces.

The service returns `{file, frames, networks, nodes, opcodes}` (the summary
values) and keeps it as `runtime.last_radio_capture`; diagnostics show it
without network ids.

`redact=true` (`capture.redact`) replaces network ids with `NET1`, `NET2`…
in order of appearance, masks identity and serial bytes (`5B` replies after
the form byte, dialect-A `77` announcements after the marker) with `XX`,
drops `air`, and removes network ids and hex runs from `raw` lines and the
`gateway` line. Every other payload byte stays. Without `redact` the file
contains the network id, so testers share it privately.

The tester guide is [Help find the identify command](radio_identify_capture.md).
