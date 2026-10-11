# Radio protocol: timing, station duties and dialect-A pairing

Deep reference for [protocol.md](protocol.md). Measured on dialect-A
equipment unless stated. Example frames use the synthetic network id `12 34`,
station `01`, heater `04` and the synthetic identity `01 02 … 0C`; they were
built with `backend/radio/dialect.build_frame`.

## Timing

| Behaviour | Value |
|---|---|
| Ack after a frame | 0–3 ms from the vendor gateway or a heater; about 16 ms from a CC1101 station's firmware auto-ack |
| Link retry when no ack comes | 3 more transmissions, about 160–190 ms apart, byte-identical |
| `<op+1> 55` reply after the ack | 20–40 ms for a setpoint, 70–90 ms for a mode-only `B4`, about 45–50 ms for toggles and `DA` |
| `B8` status reply | 48–56 ms after the request; often a second identical copy about 208 ms later |
| Program write | ack about 21 ms after the 99-byte frame ends, `B3 55` about 12 ms later |
| Status report after a command | 2.5–5 s; after a slot boundary 3–5 s; after a panel change within seconds |
| Report period | about 300 s idle, about 104 s while heating |
| Unconfirmed report | repeated 5 times, 4.6–8.1 s apart, every cycle, until confirmed |
| Registration after power-on | the heater speaks first, about 30 s after power-on |
| Vendor gateway re-probing an absent heater | roughly 1, 1, 4, then 2 minutes apart |
| Vendor cloud → radio | 620–650 ms from the API call to the frame on air |
| LINK indication with no clock sync | about 12 minutes |
| Scan probe (`B8` to an id) | answers in 118–164 ms; give up after about 230 ms. A full 02–41 scan takes about 12 s. |

**Report delivery in practice.** A heater's timers are exact (300.0 ± 1.4 s),
but a station in a house received a report in only 80–90 % of 300 s windows,
and each heater had silences of 7 to 47 minutes a few times a day. The
silences were independent between heaters and worst on the weakest links, so
they are radio loss, not heater behaviour. A station should poll (`B8`) as
well as listen, and should not alarm on one missed report.

The idle reports come as two 300 s trains per heater whose phases drift
apart, so the gap between reports varies from seconds to 300 s.

## Station duties

What a heater needs from the station (proven with the vendor gateway
unplugged and a CC1101 station as `01`):

1. **Ack** every frame addressed to the station, within the heater's retry
   window. Only firmware is fast enough.
2. **Confirm** every report (`56 …`) with `57 55`. The link ack alone does
   not stop the heater repeating the report every 4.6–8.1 s. `57 55` sent
   alone, with no report to confirm, does nothing (it is not a poll).
3. **Answer registration**: when a heater sends `50` (power-on, or straight
   after pairing a heater that had no pairing record), send the `51` clock
   sync within about 50 ms of the ack, then confirm each record it pushes
   (status, program `56 B1`, diagnostic `56 C7`, advanced `56 DB`). The
   vendor gateway's query burst is not needed.
4. **Send a clock sync now and then** (below).
5. Optionally answer `BE` with `BF 01`, and poll with `B8` for fresh values.

Example registration answer (`51`, Sunday 2026-10-11 12:34:56):

```
on air    EB 95 8C 58 B3 A1 CD 20 57 5E 4B 9C 5F F3 E0 5B 2A B2 96 23 B5 F8 44
logical   14 12 34 01 04 00 01 04 00 00 00 00 51 1A 0A 0B 00 0C 22 38 03 48 19
```

### Keepalive and the LINK indication

A heater's panel shows the lost-gateway indication (the Link icon goes, the
antenna flashes red) about 12 minutes after the last clock sync it received;
at that moment it also sends a `BE` that nobody answers. One clock sync clears
it within seconds. The vendor gateway sends clock syncs on elapsed-time
timers. The research station sent one every 150 s and the indication stayed
off; this integration sends one every 120 s. A two-minute station outage does
not trigger it.

The vendor gateway separately marks a heater stale after 350 minutes without
hearing it.

### The vendor gateway's start-up

When it powers up, the vendor gateway sends each paired heater, in turn: the
157 s "association" frame ([messages.md](messages.md#the-157-second-association-frames)),
a `B0` program read, `CA 00` (answered `CB 55`), then `C2`, `5A`, `D0`, `C6`
(and sometimes `B8`); then confirms the reports that follow and sends each
heater the wall-clock `52` sync. It runs the same query burst on every
157 s cycle and after enrolling a heater. None of this is required by the
heaters.

Turning a vendor gateway back on next to a Home Assistant station re-applies
the cloud's settings to every heater, and what two stations with id `01` do
to each other was never tested.

## Pairing in dialect A

User side: on the vendor gateway, hold Link for 2 s (orange LED flashes once a
second; the window stays open 1 minute, plus 1 minute per device added); on
the heater, hold OK for 3 s in OFF, AUTO or MANUAL. The heater shows "PAIRED
WITH GATEWAY" with a steady Link icon on success. A heater with no pairing
record shows neither Link nor antenna.

### On the air

1. **Announcement.** The heater sends its 13-byte identity (`77` + 12 bytes)
   from id `FF`, on network `00 00`, tag `03`, path `FF 01 01 01 01`:

   ```
   on air    E7 87 B8 A6 B6 A1 33 25 56 5F 4A 9F 79 E8 E8 53 2E BB B2 1C BE B9 57 FA EA C4 37
   logical   18 00 00 FF 01 00 FF 01 01 01 01 03 77 01 02 03 04 05 06 07 08 09 0A 0B 0C 5E D4
   ```

   It sweeps the destination: three frames to `01` about 2 s apart, then
   `02, 03, …` up to `21`, about 200 ms per step, two passes in about 22 s.
   The sweep is started by the OK press only, not by a lost gateway.
   Already-paired heaters relay copies to the station under their own id
   (network rewritten to theirs, their id in path byte 7).

2. **Assignment.** The station answers, inside that step's 200 ms dwell, from
   `01` to `FF` on its own network, path `01 FF 00 00 00`, **tag `04`**,
   payload = the new id. The vendor gateway answers about 62 ms after the
   announcement.

   ```
   on air    F3 95 8C 58 48 A1 CD DB 57 5E 4B 98 0A 4A 7D
   logical   0C 12 34 01 FF 00 01 FF 00 00 00 04 04 A3 97
   ```

   With tag `00` the heater acks the assignment and ignores it (77 tries, 0
   adoptions). The tag is a constant, not an echo of the id: a heater adopted
   id `06`, which it had never held, with tag `04`.

3. **Adoption.** 50–150 ms after the announcement the heater sends an empty
   frame from its new id with tag `84`, and the sweep stops:

   ```
   on air    F4 95 8C 5D B6 A1 C8 25 56 5F 4A 18 67 1E
   logical   0B 12 34 04 01 00 04 01 01 01 01 84 69 F7
   ```

4. **Registration** follows about 0.5 s later only for a heater that had no
   pairing record (new or factory-reset): `50`, then its status, program
   (all anti-frost after a reset), diagnostic and advanced records. A heater
   re-paired to a station it already belonged to just resumes reporting.

5. **Enrolment.** About 60 s later the vendor gateway sends the clock sync
   and the query burst (`C2 B0 5A D0 C6`). For about that minute it acks the
   heater's reports without confirming them. A replacement station may enrol
   on its own timing or skip it for a heater it already knows.

### Id choice

The vendor gateway keeps a 32-entry table of identities: a known identity
gets its old id back, a new one the lowest never-used entry (then the lowest
released one). Ids run `02`–`21`, which is also where the sweep stops. The
"new identity" branch was never seen on air (every test used a heater the
station already knew).

### Things that go wrong

- An unpaired (or factory-reset) heater is silent: no reports, no
  registration on power-up, no answer to scans. Only pairing finds it. A
  scan finds only heaters already paired to the network and station id it
  uses.
- Only a station with id `01` can answer a relayed announcement (the path
  reversal looks for `01`), and even then the enrolment that follows has no
  route through the relay. Move the heater or the station instead.
- A heater moved to a new id and then sent a missed rollback can drop both
  ids and answer neither until it is paired again.
- Only a fraction of answers land inside the dwell when the station's own
  ack-wait or a busy radio delays the transmit; answer once per sweep step,
  quickly, and let the next pass retry.
- Deleting a heater from the station's records does not unpair it: its next
  report or a scan brings it back.
