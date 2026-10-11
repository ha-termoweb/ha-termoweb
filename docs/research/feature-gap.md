# Cloud features and their radio paths (dialect A)

What the research found for each cloud feature of a TermoWeb heater: which
radio payload does the same thing, and how sure we are. "Proven" means sent
on air and confirmed by a status read or at the heater's panel. Payload
details are in [messages.md](messages.md). What this integration implements
today is in [radio_backend.md](../radio_backend.md).

| Feature | Radio | Status |
|---|---|---|
| Mode off / manual / auto | `B4 04` / `B4 02` / `B4 01` → `B5 55` | proven |
| Manual setpoint | `B4 02 <half °C>` | proven, 7.0–35.0 °C (a heater can hold a lower limit) |
| Temporary override | `B4 03 <half °C>` | proven; the heater ends it at the next slot or midnight |
| Presets | read: status record; write: `B6 af eco comfort` → `B7 55` | proven (the cloud refuses this write for this heater type) |
| Weekly program | `B2` + 42 or 84 bytes → `B3 55`; read `B0` | proven, 24 or 48 slots a day |
| Room temperature | status record, tenths °C | proven |
| Heating now | status flags `01`, plus `20` (boost) | proven, with the blind spot in [messages.md](messages.md#status-record) |
| Power | measured full-load power × duty | proven |
| Energy | `BC` → `BD` u32 Wh (cumulative, no history) | proven |
| Flash display | `5E 01` → `5F 55` | proven |
| Keypad lock | `BA 01/00` → `BB 55`, flag `02` | proven, panel-confirmed (the cloud's lock call fails before reaching the radio) |
| Boost start / cancel | `D2 01/00` → `D3 55` | proven; time and temperature are the heater's own settings |
| EASY mode | `D6 01/00` → `D7 55` | proven, no cloud equivalent |
| Runback | `D4 01/00` → `D5 55` | proven, no cloud equivalent |
| Control type, units, temperature offset, open window, true radiant | `C4` + 8 bytes → `C5 55` | proven; only the offset can be read back |
| Away mode, away offset, modified-auto span | `C4` bytes 4–6 | desk only |
| Maximum setpoint limit | `C4` ninth byte | rejected (`C5 56`) |
| Low Surface Temperature | read: `DB` record; write: `CC` | read proven; every `CC` write refused |
| Power grant | `BF 01` / `BF 00` in reply to `BE` | `01` seen; `00` only from the gateway firmware |
| Clock | `52 …` → `53 55` | write proven; it cannot be read back |
| Status, identity, advanced record | `B8`, `5A`, `DA` | proven |
| Pairing | announcement / assignment (tag `04`) / adoption (tag `84`) | proven, four runs |

## No radio path found

- Main switch / standby, display brightness and backlight time, eco display,
  beep, password, the panel's RESET, Self Adaptive.
- Boost duration and temperature, Runback Max Temp and setback settings.
- Daylight saving: the station sends local time, so there is nothing to set.
- Heater priority and the installation power limit: these live in the
  vendor gateway (its power manager), not in the heater.
- Energy history: the heater only keeps a running total.
- Accumulators (`acm`) and power monitors (`pmo`): no hardware was tested.

## Open questions

- Why a heater obeys only originator `01` (a constant or a peer table), and
  what two stations sharing `01` do.
- The id the vendor gateway gives a never-seen identity, on air.
- Pairing and enrolment through a relaying heater.
- What two bytes of the 157 s association frames track.
- Why `CC` is refused; what `CE` does with other values; what `C0`/`C2`
  mean.
- What starts a heater's route-probe sweep.
- The unexplained bytes of the advanced record.
- Whether the cloud's own program array is Monday or Sunday first (the wire
  is Sunday first).
- The cause of one heater's 26.0 °C setpoint limit, and of the heater that
  heated while reporting off.
- The heaters' preamble threshold below 2 bytes.
