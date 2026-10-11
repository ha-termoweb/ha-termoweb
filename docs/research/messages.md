# Radio protocol: dialect-A payloads in detail

Deep reference for the opcode table in [protocol.md §4](protocol.md#4-opcodes).
Everything here was proven on dialect-A heaters (ATC Sun Ray RF) by transmit
experiments, panel reads or the vendor gateway's firmware, unless marked
"desk only". Example frames use the synthetic network id `12 34`, station
`01` and heater `04`; they were built with `backend/radio/dialect.build_frame`.
The class names (`E5`, `EF`…) are the first on-air byte, see
[physical.md](physical.md#length-byte-and-the-old-type-byte).

## Reply and report rules

- **Reply = opcode + 1.** `B4`→`B5`, `B0`→`B1`, `B8`→`B9`, `BC`→`BD`,
  `5A`→`5B`, `D0`→`D1`, `C6`→`C7`, `CA`→`CB`, `5E`→`5F`, `52`→`53`. A
  two-byte reply `<op+1> 55` means accepted, `<op+1> 56` rejected. The
  heater answers 20–90 ms after its ack.
- **Accepted is not applied.** `B5 55` follows every processed `B4`
  command, including an out-of-range setpoint the heater then ignores. Read
  the status back to know.
- **Reports start with `56`.** A heater payload starting with `56` is a
  report; the second byte names the record (`B9` status, `B1` program, `C7`
  diagnostic, `DB` advanced). The station confirms each one with `57 55`
  (the vendor gateway would send `57 56` for a report it cannot parse).
  Replies to requests (`B9` in E6, `B1` in C9/9F, `BD`) carry no `56` and get
  no confirmation.
- **Replies often come twice**, the second copy about 200 ms after the first
  (status, preset, `CC` and `B5` replies). Use the first.

## Status record

Report (E5): `56 B9` + 13 bytes, or + 15 with the boost tail (E3). Reply to
`B8` (E6): the same without `56`, or E4 with the tail.

| Byte (E5 payload) | Field |
|---|---|
| 0 | `56` report marker |
| 1 | `B9` record type |
| 2, 3, 4 | anti-frost, eco, comfort presets (half °C). Factory: 7.0 / 18.0 / 21.0 |
| 5 | mode: `01` auto, `02` manual, `03` override, `04` off |
| 6-7 | room temperature, tenths °C, big-endian, after the panel's offset |
| 8 | setpoint in force (half °C): manual setpoint, override target, or the slot's preset in auto |
| 9-10 | measured full-load power, deciwatts, big-endian |
| 11 | duty, percent |
| 12 | flags |
| 13 | `pcb_temp`, whole °C |
| 14 | error code |
| 15-16 | boost end: high 4 bits day (0 = Sunday), low 12 bits minute of day |

- **Power** is measured, not a nameplate value: a few percent above the
  rating at mains above 230 V, latched while idle, updated while the element
  runs, and drifting down slightly as the element warms. Power drawn = power ×
  duty. The `BE` power request carries the same value.
- **Duty** is capped by the panel's Low Surface Temperature setting (75, 50
  or 23 %).
- **Flags**: `01` heating, `02` keypad locked, `04` presence, `08` window
  open, `10` true radiant, `20` boost, `40` EASY, `80` runback. They combine
  freely. Duty > 0 if and only if `01` is set, in every record checked.
  Turning Runback on at the panel sets `80` and `20` together.
- **Boost tail**: present only while a boost runs (flag `20`). A boost heats
  with the mode reading `04` (off); count `01` or `20` as heating.
- **Under Runback**, byte 4 carries the runback setpoint / Max Temp instead
  of the comfort preset.
- **After a Runback on/off cycle**, an off heater can report its anti-frost
  value as the setpoint after any panel menu visit; the heater still holds
  the real manual setpoint. Do not write the reported value back.
- **`pcb_temp`** rises under load and decays over tens of minutes. Observed
  25–64 °C.
- **Reported state can be wrong.** After a panel boost was cancelled, one
  heater ran its element for an hour while every record said mode off, duty
  0, flags clear; `pcb_temp` climbed 29 → 57 °C and the room 24 → 30 °C. A
  power cycle at the wall stopped it. A rising `pcb_temp` (or energy
  counter) with duty 0 is the only sign.

Cadence: a heater reports about every 300 s when idle and about every 104 s
while heating, plus within seconds of any change (command, panel, schedule
slot). See [sessions.md](sessions.md#timing).

Example, heater 04 reporting manual 22.0 °C, room 22.0 °C, 750.0 W, duty
50 %, heating, board 33 °C:

```
on air    E5 95 8C 5D B6 A1 C8 25 56 5F 4A 9C 58 50 E4 74 00 BC B4 C7 9A AD 11 C3 E7 BB E3 EF 1F
logical   1A 12 34 04 01 00 04 01 01 01 01 00 56 B9 0E 24 2A 02 00 DC 2C 1D 4C 32 01 21 00 AA E2
```

## Setpoint and mode (`B4`)

- `B4 01` auto, `B4 02` manual, `B4 04` off: reply `B5 55`.
- `B4 02 <half °C>` manual setpoint, `B4 03 <half °C>` temporary override.
  A setpoint alone switches an off heater to manual.
- Range: 7.0–35.0 °C after a factory reset. One heater refused everything
  above 26.0 °C until it was factory reset; no panel setting explained it.
  Out-of-range values are acked, answered `B5 55` and ignored. While Runback
  is on, the heater clamps setpoints to its Max Temp.
- The override ends by itself at the heater's next program change or at
  midnight, with nothing sent.
- Leaving auto for off keeps the slot's preset as the reported setpoint until
  a manual setpoint is written.
- `B4 01 <sp>` and `B4 04 <sp>` were never tried on air.

## Presets (`B6`)

`B6 <anti-frost> <eco> <comfort>` (half °C, 18-byte frame) → `B7 55`.
The heater's panel allows anti-frost 7.0–17.5, eco 7.5–20.5 and comfort
18.5–35.0 °C. The vendor's cloud returns an error for preset writes on this
heater type, so on these heaters the radio can do more than the cloud.

```
on air    F0 95 8C 58 B3 A1 CD 20 57 5E 4B 9C B8 E7 CE 7A 75 60
logical   0F 12 34 01 04 00 01 04 00 00 00 00 B6 0E 24 2A 5F DE
```

## Weekly program (`B2`, `B0`)

- Write: `B2` + 84 bytes (99-byte frame, 48 half-hour slots a day) or `B2` +
  42 bytes (24 hourly slots), Sunday first → `B3 55` about 12 ms after the
  ack.
- In the 84-byte form each byte is two hours, one nibble per hour, high
  nibble first; each nibble is two 2-bit half-hour slots (`0` anti-frost, `1`
  eco, `2` comfort), and the two halves are stored independently.
- Read: `B0` → `B1` + 84 bytes (class 9F) or `B1` + 42 bytes (class C9),
  whichever length was written last.
- The heater pushes `56 B1` + 84 bytes (class 9E) at registration and when
  someone leaves the PROGRAM screen on its panel.

## Energy counter (`BC`)

`BC` → `BD` + u32 big-endian Wh: the heater's own cumulative energy counter,
live at 1 Wh resolution. It is only ever a reply, never pushed, and needs no
confirmation. The vendor gateway reads it from every heater at the top of
each hour and its cloud "samples" history is that counter passed straight
through. A factory reset sets it to 0.

```
on air    EF 95 8C 5D B6 A1 C8 25 56 5F 4A 9C B3 E9 EA 60 13 F8 9A
logical   10 12 34 04 01 00 04 01 01 01 01 00 BD 00 00 30 39 46 2E
```

(`00 00 30 39` = 12 345 Wh.)

## Power request (`BE`) and verdict (`BF`)

The heater sends `BE <power hi> <power lo>` (its measured full-load power,
deciwatts) when its element wants to switch on, and also after panel menu
changes. The vendor gateway's power manager answers `BF 01` (granted) or
`BF 00` (refused) within about 50 ms. On a refusal it waits before the heater
may ask again: 2.5 s below 1200 W, rising to 33.5 s at 2600 W and above. Only
`BF 01` was ever seen on air; the heater heats whether or not anything
answers. An unanswered `BE` coincides with the LINK indication when the
heater has had no clock sync for about 12 minutes
([sessions.md](sessions.md#keepalive-and-the-link-indication)).

```
on air    F1 95 8C 5D B6 A1 C8 25 56 5F 4A 9C B0 F4 A6 D7 FB
logical   0E 12 34 04 01 00 04 01 01 01 01 00 BE 1D 4C 87 D1
```

## Toggles

| Payload | Reply | Effect |
|---|---|---|
| `D2 01` / `D2 00` | `D3 55` | boost start / cancel. Only bit 0 counts (`D2 02` = `D2 01`). Duration (60 min default) and temperature come from the heater's own settings; the setpoint reads the higher of the current one and the boost temperature. |
| `D4 01` / `D4 00` | `D5 55` | Runback Config on / off. On forces manual at the anti-frost setpoint; off does not restore the previous setpoint. |
| `D6 01` / `D6 00` | `D7 55` | EASY mode: the panel keeps only plus and minus and the heater is forced to manual; off returns to the previous mode. |
| `BA 01` / `BA 00` | `BB 55` | keypad lock, flag `02` |
| `5E 01` | `5F 55` | flash the display. `5E 00` is accepted and does nothing visible. |

## Advanced setup (`C4`, `DA`, `E2`)

Write: `C4 <control_mode> <units> <offset> <away_mode> <away_offset>
<modified_auto_span> <window_mode> <true_radiant>` → `C5 55`.

| Field | Values |
|---|---|
| `control_mode` | `0`–`3` hysteresis 0.25 / 0.35 / 0.50 / 0.75 °C, `4` PID (panel default) |
| `units` | `1` shows °F on the panel. Display only: radio values stay Celsius. |
| `offset` | signed tenths of a degree: reported = measured − offset (wire `F6` = −1.0 raises the reported value by 1.0). Applied within seconds, with a fresh report. |
| `away_mode`, `away_offset`, `modified_auto_span` | only ever sent as 0 (desk only) |
| `window_mode` | Open Window detection on/off (does not set flag `08` by itself) |
| `true_radiant` | accepted; this panel has no such menu |

A ninth byte (`max_stemp_limit`) is rejected `C5 56`. There is no way to read
most fields back, so a write must carry every field; the research used the
panel defaults (`04 00 00 00 00 00 00 00` plus the wanted change).

Read: `DA` → `DB` record (17 bytes; the E2 report without `56`). The heater
also pushes it (`56 DB`, class E2) at registration and after panel settings
changes. Known fields (index from `DB`): eco preset (2), comfort preset (5),
Low Surface Temperature (14: `00` off, `01` 75 %, `02` 50 %, `03` 23 %), and
probably the boost temperature (15). Under Runback the slot after eco carries
Max Temp. The rest is constant and unexplained.

Panel-only settings with no known radio write: Self Adaptive, Low Surface
Temperature, boost time and temperature, Runback Max Temp and setback,
display brightness, beep, password, the panel's own RESET.

## Factory block (`CC`), desk only

From the vendor gateway's firmware: `CC <temp_compensation_enabled>
<temp_compensation_factor> <tag> <duty_limit/10 << 4 | true_radiant_available>
<operating_mode> [<power_factor hi> <power_factor lo>]`. Three 6-byte forms were sent
and each was refused `CD 56` with nothing changed; the 8-byte form was not
tried. Treat it as unusable.

## Identity (`5A`) and capability (`D0`)

- `5A` → `5B 77 01 01 01 04` + 12-byte identity + `0A 1A` (class E0). The
  `04` is constant on every heater, not the node id. The 12 bytes are the
  same ones the heater sends in its pairing announcement, and they survive a
  factory reset. Some identity bytes differ between power ratings.
- `D0` → `D1 00 05 01 01 00 27 10` (class EC), identical on every heater
  (`27 10` = 10000 is not a rating).
- `C6` → `C7 04 00 00 00 14 00 00 00` (class EB); the heater also pushes it
  as `56 C7 ..` (class EA) at registration. Meaning unknown.
- `C0` → `C1 55`, `C2` → `C3 55`: meaning unknown, probably not recoverable
  from the gateway firmware.

## Clock (`51`/`52`)

`5x YY MM DD DOW HH MM SS 03` (class EB): `51` while the heater registers,
`52` otherwise; two-digit year; DOW 0 = Sunday; local 24-hour time → `53 55`.
After a factory reset the heater's clock reads 00:00 until it gets one.

```
on air    EB 95 8C 58 B3 A1 CD 20 57 5E 4B 9C 5C F3 E0 5B 2A B2 96 23 B5 D5 00
logical   14 12 34 01 04 00 01 04 00 00 00 00 52 1A 0A 0B 00 0C 22 38 03 65 5D
```

(Sunday 2026-10-11 12:34:56.)

### The 157-second "association" frames

The vendor gateway also sends each heater a `52 …` payload ending `01`
instead of `03`, every 157.6–157.8 s, with calendar-invalid contents. Most
bytes are constant per heater model; one byte is the node id − 1; two bytes
change between gateway power-ups for an unknown reason. A station that never
sends them works: heaters ran for weeks without. What they do keep needing is
a clock sync now and then
([sessions.md](sessions.md#keepalive-and-the-link-indication)).
