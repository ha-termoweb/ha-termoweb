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
| 11 | 1 | tag (`00` commands, `04` id assignment, `06` route probe) |
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
| Network id | `1B 30` (fixed) | per installation, learned from traffic (no default) |
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
| `50` | heater → station | registration; the station answers with a `51` clock sync | A | B |
| `56 B9 ..` | heater → station | status report (E5, 15 bytes; E3, 17 bytes with boost tail) | A | ? |
| `56 DB ..` | heater → station | advanced record report (E2) | A | ? |
| `56 B1` + 84 bytes | heater → station | program report (9E class) | A | ? |
| `BE ..` | heater → station | power request: the element wants to switch on | A (`BE hi lo`, 3 bytes) | B (9 bytes) |
| empty, tag `06` | heater → any | route probe | A | B |

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
- Byte 1: room temperature in tenths of a degree (`DB` = 21.9 °C). Checked
  against a reference sensor placed next to the heater's probe.
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
