# Radio protocol: header, addressing and filtering

Deep reference for [radio_protocol.md](protocol.md) §2. Proven on
dialect-A equipment by transmit experiments and the vendor gateway's
firmware; dialect B uses the same header layout. All examples use the
synthetic network id `12 34`, station `01` and heater `04`, and were built
with `backend/radio/dialect.build_frame`.

## Header fields

| Byte | Field | Notes |
|---|---|---|
| 0 | length | dialect rule ([physical.md](physical.md#length-byte-and-the-old-type-byte)) |
| 1-2 | network id | the station's network. `00 00` only on a pairing announcement from a heater that has no network yet. |
| 3 | src | the device that transmitted **this hop**. A relay rewrites it. |
| 4 | dst | the **next hop**. The vendor gateway copies it from byte 7. |
| 5 | flags | `00` data, `80` ack. On a data frame, `80` means "no ack wanted": the heater does not ack it but still processes it. |
| 6 | path[0], originator | the node the frame comes from, end to end. The vendor gateway's application layer treats this byte, not byte 3, as the sender. A frame the gateway writes always has `01` here. |
| 7 | path[1] | the first hop towards the final destination |
| 8-10 | path[2..4] | further hops. Padding: `00` from a station, `01` from a heater. |
| 11 | tag | set per call site by the routing layer, see below |
| 12.. | payload | first byte = opcode |
| last 2 | CRC | |

Bytes 3-4 are per hop; bytes 6-10 are one end-to-end routing path (originator
then up to four hops), carried unchanged by relays. On a network where every
heater hears the station directly, byte 6 equals byte 3, byte 7 equals byte 4
and bytes 8-10 are padding, which is why early notes called them "source and
destination repeated". That is not a rule: relayed frames break it.

### Tags (byte 11)

| Tag | Used on |
|---|---|
| `00` | every command, reply and report |
| `03` | a pairing announcement (dialect A and B) |
| `04` | the id assignment that answers it |
| `06` | a route probe or advertisement from a heater |
| `07` | the vendor gateway's answer to a route probe |
| `84` | the empty probe a heater sends right after adopting a new id |
| `86` | the vendor gateway's reply after learning a route |

A wrong tag is not caught by the link layer: a command with tag `01`, or an
id assignment with tag `00`, is acked and then ignored.

### The path from a heater

A heater fills unused hop slots with `01`, so a heater's frame to the station
reads `<id> 01 01 01 01`. The vendor gateway accepts a frame only if its own
id (`01`) appears in bytes 7-10 and every byte after it is `00` or `01`.

A station that sends a command with the heater-side padding (`01 01 01` in
bytes 8-10) does not get the command executed: the heater re-transmits the
payload towards the gateway as if relaying it. `00 00 01` is processed
normally.

### Relayed frames

Heaters relay. Captured examples:

- A heater forwarding a station command one hop: bytes 3-4 rewritten
  (`src` = the relay, `dst` = the next hop), bytes 6-10 unchanged.
- A heater relaying another heater's route probe to the station: `src` = the
  relay, path = `<origin> <relay> 01 01 01`, tag `06`.
- A heater relaying a pairing announcement: network id rewritten from
  `00 00` to its own network, `src` = the relay, byte 7 = the relay's id; byte
  6 (`FF`), bytes 8-10 and the tag (`03`) unchanged.

The vendor gateway keeps a next-hop table learned from tag-`06` route probes
("to reach node X, go via node Y"). This integration does not route: it
ignores relayed frames and talks only to heaters in direct range.

## Who a heater accepts commands from

A heater acts on commands only when the originator (byte 6) is `01`. Tested
with 8 source ids (`01 02 03 04 05 0E 40 FF`) × 2 setpoints × 3 repeats: only
`01` was accepted, including when the heater's own id was used. The
integration's station id is therefore always `01`. (An early test that
changed only byte 3 looked like "any sender is accepted"; byte 6 was still
`01`.) Whether the heater checks byte 6 itself or a field that equals it on
one-hop frames is not proven.

Because the heater pads its path with `01`, the vendor gateway finds its own
position in a relayed announcement by looking for `01`. A station with any
other id cannot reverse such a path, which is a second reason to be `01`.

## What is dropped, acked or processed

Proven by transmit against a dialect-A heater:

| Change to a valid command | Ack | Reply / effect |
|---|---|---|
| none | yes | `<op+1> 55`, applied |
| CRC corrupted | no | none |
| length byte ±1 with a stale CRC | no | none |
| length +1, padded, CRC recomputed | yes | `B5 56` |
| network id off by one | no | none (the vendor gateway's radio also filters in hardware) |
| destination is an unused id | no | none |
| bytes 6-7 `00 00`, or a path that does not include the heater | yes | none |
| bytes 8-10 `01 01 01` | yes | the heater relays the payload instead |
| tag `01` | yes | none |
| flags `80` | **no** | `B5 55`, applied |
| unknown mode byte (`B4 05`) | yes | `B5 55` |

So the ack only says "the link layer took it". A station must wait for the
heater's `<op+1> 55` reply (or a fresh status record) to know a command was
processed, which is what `RadioLink.send_frame` callers in this integration
do.

## No security

There is no encryption, no authentication, no session key, no sequence
counter and no nonce. Retries and repeated commands are byte-identical, and
an old frame replayed later is accepted. Anyone in radio range who knows the
network id can command a heater.

## Worked frames (dialect A, synthetic network `12 34`)

Setpoint 25.5 °C, station 01 → heater 04 (`B4 02 33`):

```
on air    F1 95 8C 58 B3 A1 CD 20 57 5E 4B 9C BA EB D9 63 42
logical   0E 12 34 01 04 00 01 04 00 00 00 00 B4 02 33 33 68
```

Ack from heater 04 to station 01:

```
on air    FA 95 8C 5D B6 21 99 5C
logical   05 12 34 04 01 80 55 78
```

Report confirmation `57 55`, station 01 → heater 04:

```
on air    F2 95 8C 58 B3 A1 CD 20 57 5E 4B 9C 59 BC B4 4D
logical   0D 12 34 01 04 00 01 04 00 00 00 00 57 55 5E 1D
```

More examples are in [messages.md](messages.md) and
[sessions.md](sessions.md).
