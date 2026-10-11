# Radio protocol: physical layer, whitening and CRC

Deep reference for [radio_protocol.md](protocol.md) §1–3. Most of
this page was measured on dialect-A equipment (the TermoWeb gateway with its
AX5043 radio and Sun Ray RF heaters); [radio_research.md](history.md)
explains how. Dialect B shares the carrier, bit rate and modulation; its
framing differences are in the main page.

## Radio parameters

| Parameter | Value |
|---|---|
| Carrier | 869.525 MHz nominal (the gateway's case label; the gateway's AX5043 is programmed for exactly this). In the 869.40–869.65 MHz SRD sub-band: 500 mW ERP, 10 % duty cycle. Vendors call it "868 MHz". |
| Modulation | 2-FSK, NRZ. No Manchester, no NRZI, no inversion. |
| Bit rate | 9.6 kbps (the gateway's AX5043 is set to 9599.7 bps). |
| Bit order | MSB first. |
| Deviation | about 50 kHz from a CC1101 station (`DEVIATN 0x50` = 50.8 kHz); about 61 kHz from the heaters (the gateway's AX5043 is set to 60 kHz). |
| Preamble | `AA…` alternation. The gateway sends about 30 bits; a CC1101 station sends 8 bytes by default. |
| Sync | dialect A `2DE5`, dialect B `2DD4` (16 bits, see below). |
| Longest frame | 99 bytes on air (the dialect-A 48-slot program write). |
| Ack window | the receiver acks 0–3 ms after a frame ends (vendor gateway and heaters). A CC1101 station's firmware auto-ack lands about 16 ms after the frame. |

### Measured on air

An RTL-SDR Blog V4 calibrated to about 0 ppm measured one station (a
nanoCUL CC1101 stick) and three heaters of two power ratings:

| | Station (CC1101) | Heaters |
|---|---|---|
| Carrier | 869.500 MHz, −24.8 kHz (−28.6 ppm) from nominal | 869.516 MHz, about −8.9 kHz (−10.1 to −10.5 ppm), all three within 0.4 ppm |
| Peak deviation | 50.85 kHz (σ 14 Hz) | 61.3 kHz (σ 16–54 Hz), all three within about 110 Hz |
| Symbol rate | 9595.6 baud | 9599.1–9599.7 baud |
| 99 % occupied bandwidth | about 148–154 kHz | about 189–192 kHz |

- The link is asymmetric: heaters deviate about 20 % wider than the station.
  Carrier, deviation and rate are the same on both heater models, so they
  are a fleet property, not a per-model one.
- The two ends sit about 16 kHz apart and the link does not mind. Both
  occupied spans stay inside the sub-band, about 20 kHz from its lower edge.
- Measure at 1.024 Msps or more: at 250 ksps the RTL-SDR's IF filter clips
  the signal (SNR and bandwidth both drop).

### Preamble length and transmit power

- A CC1101 transmitter worked at +10 dBm (`PATABLE 0xC0`) with an 8-byte
  preamble. One early test at 0 dBm with a 4-byte preamble got no ack; it
  changed power and preamble together, so it proves nothing about either.
- With the power held at +10 dBm, the preamble was shortened step by step
  down to 2 bytes (the CC1101's minimum, `MDMCFG1 NUM_PREAMBLE`): every step
  got every reply, with no timing change, from a heater at about −74 dBm and
  one at about −45 dBm. The heaters' own lock threshold was not reached.

### Station RSSI in a house

Over 24 hours, a nanoCUL in one house heard its three heaters at a median of
−68, −68.5 and −44 dBm (10th percentile −72, −72.5, −49). The weakest
readings were around −80 to −86 dBm and still decoded. Noise frames with
nonsense headers arrive at −98 to −100 dBm and fail the CRC.

## Preamble and sync

On air the sync is 17 bits: `0001 0110 1111 0010 1`. Read as a 16-bit window
starting one bit early it is `16F2`; every byte decoded at that alignment is
shifted by one bit. The frame-aligned 16-bit sync is `2DE5`, immediately
followed by byte 0 of the frame. Program a receiver with `SYNC1/SYNC0 = 2D E5`
and an ordinary `AA` preamble.

The gateway's AX5043 agrees: its match pattern, read bit-reversed per byte
(the AX5043's on-air order for that register), is `E5 2D`.

## Whitening (PN9)

Dialect A whitens every frame from byte 0 with the standard PN9 data
whitener: polynomial x⁹ + x⁵ + 1, 9-bit register seeded all ones (`1FF`),
reseeded for every frame, one output byte per eight shifts. Scrambling and
descrambling are the same XOR.

```
function keystream(n):              # PN9, seed all ones, bytes MSB first
    reg = 0x1FF
    out = []
    repeat n times:
        byte = 0
        repeat 8 times:
            bit = reg AND 1
            reg = (reg >> 1) OR ((bit XOR ((reg >> 5) AND 1)) << 8)
            byte = (byte << 1) OR bit
        out.append(byte)
    return out
```

First 32 keystream bytes:

```
FF 87 B8 59 B7 A1 CC 24 57 5E 4B 9C 0E E9 EA 50 2A BE B4 1B B6 B0 5D F1 E6 9A E3 45 FD 2C 53 18
```

**Bit order matters.** This link puts the first bit shifted out into bit 7 of
the keystream byte. TI's Design Note DN509 (SWRA322) tabulates the same
generator the other way round (`FF E1 1D 9A ED 85 33 24 …`): the two are
bit-reverses of each other, byte for byte, over the whole 511-byte period.
Consequences, all checked:

- A CC1101 with `PKTCTRL0.WHITE_DATA` set does not produce these bytes. Keep
  hardware whitening off and whiten in software.
- rtl_433 and Universal Radio Hacker de-whiten in DN509's order, so they do
  not decode this link unmodified.

The vendor gateway whitens in its ESP32 firmware, because its AX5043 radio
has no PN9 whitener (only a G3RUH-style self-synchronising scrambler, which
this link does not use).

PN9 is line coding, not security: the sequence is public, fixed and keyed by
nothing.

## Length byte and the old "type byte"

Byte 0 is the logical length (dialect A: total − 3). Because the first
keystream byte is `FF`, the first on-air byte is the length XOR `FF`, which
early notes read as a "frame type". Those names are still handy, and
`capture` and survey output use the length to tell frame classes apart:

| On-air byte 0 | Logical length | Total bytes | Usual content |
|---|---|---|---|
| `FA` | `05` | 8 | ack |
| `F4` | `0B` | 14 | header only: route probe (tag `06`) or post-adoption probe (tag `84`) |
| `F3` | `0C` | 15 | one-byte request (`B8`, `B0`, `BC`, `5A`, `C0`, `C2`, `C6`, `D0`, `DA`), heater registration `50`, id assignment |
| `F2` | `0D` | 16 | two-byte command or reply (`B4 xx`, `57 55`, `<op+1> 55`, toggles, `BF xx`) |
| `F1` | `0E` | 17 | setpoint `B4 02/03 xx`; from a heater, power request `BE hi lo` |
| `F0` | `0F` | 18 | preset write `B6 af eco comfort` |
| `EF` | `10` | 19 | energy counter reply `BD` + 4 bytes |
| `EC` | `13` | 22 | `D1 ..` reply to `D0` |
| `EB` | `14` | 23 | clock sync `5x …` (9-byte payload), `C7 ..` reply to `C6` |
| `EA` | `15` | 24 | `56 C7 ..` diagnostic push |
| `E7` | `18` | 27 | pairing announcement |
| `E6` | `19` | 28 | status reply `B9 ..` (14 bytes) |
| `E5` | `1A` | 29 | status report `56 B9 ..` (15 bytes) |
| `E4` | `1B` | 30 | status reply with boost tail (16 bytes) |
| `E3` | `1C` | 31 | status report with boost tail (17 bytes), or the `DB` record reply to `DA` |
| `E2` | `1D` | 32 | `56 DB ..` advanced record report |
| `E0` | `1F` | 34 | identity reply `5B 77 ..` |
| `C9` | `36` | 57 | program reply `B1` + 42 bytes (24 slots a day) |
| `9F` | `60` | 99 | program write `B2` + 84 bytes, or program reply `B1` + 84 bytes |
| `9E` | `61` | 100 | program report `56 B1` + 84 bytes |

The same opcode can arrive in two classes (`B1` in `C9` and `9F`), so a
receiver must use the payload length, not only the first payload byte.

Classes `F5` and `FD` in old notes were misframed captures taken at the wrong
centre frequency, not real frames.

## CRC (dialect A)

CRC-16/CCITT: polynomial `1021`, MSB first, init `1D0F`, final XOR `FFFF`,
over the **logical** (descrambled) bytes 0 … total−3, sent big-endian.

```
function crc16(bytes):
    crc = 0x1D0F
    for b in bytes:
        crc = crc XOR (b << 8)
        repeat 8 times:
            crc = ((crc << 1) XOR 0x1021) if crc AND 0x8000 else (crc << 1)
            crc = crc AND 0xFFFF
    return crc XOR 0xFFFF
```

Computing a CRC over the whitened bytes instead gives a different constant
for every frame length. That is why early notes listed five "per-class CRC
seeds"; there is one CRC and one keystream.

The vendor gateway computes the CRC in software; its AX5043 hardware CRC is
off. A CC1101 cannot check it either: the CC1101's CRC only supports init
`FFFF`, so `PKTCTRL0.CRC_EN` must stay off and the host checks the CRC.

The research corpus (4010 CRC-valid frames from 18 classes) was rebuilt from
each frame's parsed fields with this keystream and CRC with zero mismatches.

## Building a CC1101 receiver

The register values below are the working set from the research firmware
(26 MHz crystal). The firmware in this repository uses the same approach.

| Register | Value | Why |
|---|---|---|
| `FREQ2/1/0` | `21 71 7A` | 869.525 MHz (`21 71 A0` = 869.540 MHz also works) |
| `MDMCFG4` | `68` | RX filter about 270 kHz, `DRATE_E` 8 |
| `MDMCFG3` | `83` | 9596 bps. `93` (9993 bps) duplicates a bit every 24–28 bits |
| `MDMCFG2` | `02` | 2-FSK, no Manchester, 16/16 sync bits |
| `MDMCFG1` | `42` | 8-byte TX preamble (2 bytes also works) |
| `SYNC1/0` | `2D E5` (dialect A) / `2D D4` (dialect B) | |
| `DEVIATN` | `50` | 50.8 kHz |
| `PKTCTRL0` | CRC off, whitening off | see above |
| `PKTCTRL1` | `04` | append RSSI and LQI/CRC status bytes |
| `FIFOTHR`, `TEST2/1/0` | `47`, `81 35 09` | required by the datasheet for a filter of 325 kHz or less |
| `PATABLE` | `C0` | +10 dBm at 868 MHz (`50` = 0 dBm) |

Other registers use the common 2-FSK baseline of the SmartRC/ELECHOUSE
CC1101 driver. RSSI in dBm is the status byte / 2 − 74 (signed status byte).

**Finding the end of a frame.** The CC1101 cannot end a variable-length frame
on signal quality: carrier sense dips on ordinary 2-FSK tone changes, the
RXBYTES count never stalls in infinite-length mode (the demodulator clocks
noise), and an RSSI drop still truncates frames. What works: start in
infinite-length mode with a FIFO threshold interrupt, read byte 0 as soon as
it arrives, compute the total length from the dialect's length rule
(dialect A: `(byte0 XOR FF) + 3`), then switch to fixed-length mode with
`PKTLEN` = the remaining length. Reject totals under 8 or over 255.

**Other CC1101 lessons.**

- Transmitting more than 64 bytes: pre-fill 63 bytes (the chip reads
  `TXBYTES` as 0 when the FIFO is exactly full), then top up whenever it
  drops below 32.
- Hardware sync detection has no error tolerance and fires on noise; drop
  weak frames and rely on the software CRC.
- With dynamic framing, the LQI status byte reads 0 on every frame (RSSI is
  fine). Reading the LQI register directly or turning on the RX CRC engine
  did not change it. Ignore LQI; the CRC is the quality gate.
- Auto-ack in firmware: the ack must leave within the heater's window (the
  heater retries about 160–190 ms later), which a host on a serial or TCP
  link cannot guarantee.

## The vendor gateway's radio (AX5043)

For anyone building on an AX5043, these values were read from the vendor
gateway's firmware (16.000 MHz crystal):

| Register | Value | Meaning |
|---|---|---|
| `FREQA` | `36586667` | 869.525001 MHz |
| `TXRATE` | `002752` | 9599.7 bps |
| `MODULATION` | `08` | 2-FSK |
| `FSKDEV` | `00 F5 C3` | 60.0 kHz (matches the heaters' measured 61.3 kHz) |
| `MATCH1PAT` / `MATCH1LEN` | `A7B4` / `8F` | the sync word, bit-reversed per byte; 16-bit match |
| `FRAMING` | `06` | hardware CRC off |
| `PKTMAXLEN` | `C1` | 193 bytes |
| `PKTADDR` / `PKTADDRMASK` | the whitened network id / `FF FF` | the radio drops frames for other networks in hardware |

While its pairing (discovery) window is open, the gateway sets the address
mask to `00 00` and its software also accepts network id `00 00`, which is
how it hears a heater that has no network yet
([sessions.md](sessions.md#pairing-in-dialect-a)).
