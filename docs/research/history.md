# How the radio protocol was reverse-engineered

This page tells contributors where the radio knowledge in this repository
comes from: the hardware, the tools, the order in which the unknowns fell, and
the mistakes worth not repeating. The protocol itself is in
[protocol.md](protocol.md) and the deeper reference pages it links
to.

## Credit

The protocol was reverse-engineered by **Andre Faria
([@andremmfaria](https://github.com/andremmfaria))** in September 2026, on his
own gateway and heaters, working with Claude Code sessions.
Everything below summarises his research notes, which stay private because
they contain captures from a real home. Mistakes in this summary are ours.

## Ground rules the research followed

These rules still apply to anyone who extends the protocol work:

- **Your own equipment only.** Every capture came from the researcher's own
  gateway and heaters. A neighbour's frame is someone else's personal data:
  never publish it.
- **Publish facts, not identifiers.** Network ids, heater identities, serials,
  gateway ids, MAC addresses and room names identify a home. Docs and tests
  use synthetic values (network `12 34`, identity `01 02 03 …`, serial
  `X123456`) and frames rebuilt with the integration's codec. See
  [privacy.md](privacy.md).
- **Facts from firmware analysis are fine; the vendor's code is not.** Register
  values, CRC parameters, frame layouts and field meanings learned by reading
  the gateway firmware are facts. Decompiled listings, disassembly and hex
  dumps of the vendor's firmware are the vendor's copyright and must not be
  published.
- **Transmit on purpose, once.** The link sits in the 869.40–869.65 MHz
  sub-band (500 mW ERP, 10 % duty cycle). Every experiment sent single frames
  on demand, never a loop. Commands that can wipe a heater ([protocol.md, "Dangerous commands"](protocol.md#dangerous-commands))
  were only sent with the owner present.
- **A claim needs a proof.** A statement went into the specification only
  after a transmit experiment, a whole-corpus check, or a firmware code path
  supported it. Everything else stayed an open question.

## The equipment

| Item | Role |
|---|---|
| ATC Sun Ray RF oil-filled radiators of two power ratings | Targets. The same platform as other brands on the TermoWeb cloud. |
| Tevolve-branded TermoWeb gateway | The vendor's station: ESP32-D0WDQ6, 4 MB SPI flash, LAN8720A Ethernet PHY, ON Semiconductor (Axsem) AX5043 sub-GHz radio in QFN-28, PCB trace antenna, CP2102N USB-UART behind the micro-USB. Case label: 869.525 MHz. |
| nanoCUL868 USB stick (Arduino Nano clone, FT232R, TI CC1101) | Bench receiver, later the transmitter. Started on stock culfw, then reflashed with custom firmware. |
| Flipper Zero | First raw captures (FM476 preset). |
| RTL-SDR Blog V4 (R828D tuner, 1 ppm TCXO) | Calibrated carrier, deviation and bit-rate measurements. |
| Home Assistant with the cloud integration | Drove the heaters through the cloud with logged timestamps, so each capture could be matched to a known command. |

The supply chain, for orientation: Casple S.A. (Burgos, Spain) runs the
TermoWeb platform and cloud; ATC Electrical and Mechanical makes the
Sun Ray RF radiators; Ducasa sells the same platform as Ducaheat / Tevolve.
"Helki" is a different white-label platform, but the gateway firmware is from
the Helki firmware line (ESP-IDF). The vendor documents name only "868 MHz"
and "RF"; nothing public described the chip, modulation or framing.

## Timeline

All dates are 2026.

| Date | Result |
|---|---|
| 09-05 | Recon: vendors, the cloud data model (nodes, modes, presets, the 168-slot week), the sub-GHz toolchain, and prior art. rtl_433 has no decoder for this family; the closest precedent is the Honeywell RAMSES-II work (`ramses_rf` + `evofw3`). |
| 09-05 | The bench stick identified as a nanoCUL868 on culfw 1.67. Stock culfw cannot receive this link (below), so it became an RSSI scanner instead. |
| 09-05 | Carrier found by an RSSI sweep: two 2-FSK lobes around 869.54 MHz, inside the 869.4–869.65 MHz sub-band, not at "868". An unplug test and the bench geometry told heater from gateway without decoding a byte: heaters report about every 300 s, the gateway retries in triplets about 175 ms apart. |
| 09-05 | Flipper Zero raw capture: NRZ (pulse widths cluster at 1–4 bit units of about 104 µs), a 30-bit alternating preamble, a fixed non-standard 16-bit sync. A custom decoder recovered 34 of 34 packets. |
| 09-05 | CRC hunt, partial: solving for the CRC initial value by linear algebra found CRC-16/CCITT with init `1D0F` and final XOR `FFFF`, and showed the frame starts one bit later than first assumed. Only one frame class verified. |
| 09-05 | First custom CC1101 receiver firmware; a duplicated bit every 24–28 bits showed the bit rate is 9.6 kbps, not 10 kbps. |
| 09-05 | Gateway teardown, and a full flash dump over the USB port (no flash encryption, no secure boot). |
| 09-06 | First local command: a setpoint frame from the nanoCUL was acked and showed up in Home Assistant. Mode frames matched the gateway's byte for byte. |
| 09-06 | Ghidra on the gateway firmware: the AX5043 register table, the software CRC and a "scrambler" reseeded every frame. Descrambling plus one CRC verified every frame class: the "five per-class CRC seeds" were one CRC computed over whitened bytes. Header bytes 6–10 turned out to be a mesh routing path. |
| 09-06 | Proof programme (one claim per batch, raw capture per run): filtering rules, query opcodes and their replies, the setpoint range, report confirmation (`57 55`), cold start, and a full pairing capture. The integration ran on the stick for the first time. |
| 09-09 | Pairing from Home Assistant solved (header byte 11 must be `04` on the id assignment). Two factory resets showed the heater identity is permanent and an unpaired heater is silent. Presets, measured power and the energy counter decoded. |
| 09-10 | The "bespoke scrambler" identified as standard PN9 with the bytes assembled MSB first. Source-id rule ({01} only), preamble floor and duty byte settled; corpus rebuild at 4010 frames, 18 classes, zero mismatches. |
| 09-11 | RTL-SDR measurements: carrier offsets, deviation per side, occupied bandwidth. |
| 09-12 | Sunday-first week, the heater running its own schedule, the 48-slot resolution, override end, preset write `B6`, the setpoint ceiling, and the boost incident (below). |
| 09-13 | Boost, EASY, runback, keypad lock and the `C4` advanced record proven on air and at the panel. The LINK indication timeout found and fixed with a clock keepalive. |
| 09-15/16 | Low Surface Temperature, the `DB` record, `CC` refused, a relayed pairing announcement captured, the LQI investigation closed. |
| 10-09 | A second protocol family ("dialect B") found on another installation; see [protocol.md §3](protocol.md#3-dialects). |

## Tools and what each was good for

### nanoCUL on stock culfw: an RSSI meter, nothing more

- culfw's report mask is off by default, so a passive read of a fresh stick
  returns nothing at any baud rate. `X21` turns reports on.
- Every culfw receive mode loads a compiled-in CC1101 table (HomeMatic, MAX,
  LaCrosse, SlowRF). An unknown framing never shows up in any of them.
- `W` writes the AVR EEPROM, not live CC1101 registers, and the decode modes
  reload their tables, so culfw cannot be retuned to an unknown protocol.
  Reflashing was the only path.
- What did work: SlowRF reloads its registers from EEPROM each time it is
  entered, and `C34` polls RSSI about 63 times a second. Together they make a
  crude retunable RSSI meter (noise floor −97 to −102 dBm). Log level changes,
  not raw levels: the AGC decays over about 400 ms and fakes long bursts.
- culfw's transmit credit assumes the 868.0–868.6 MHz sub-band and its 1 %
  duty cycle; this link is in the 10 % sub-band.

### Flipper Zero: triage

- RX works across 779–928 MHz; the region lock only gates TX.
- The Frequency Analyzer scans a fixed list, so it cannot find a carrier.
- Read RAW records mark/space durations (about 60 µs resolution), not IQ. The
  FM476 preset at 869.55 MHz demodulated this link cleanly enough to recover
  the bit timing, the coding, the preamble and the sync.

### RTL-SDR: calibrated measurements

- A genuine RTL-SDR Blog V4 (USB-C "V4C") was used. The V4 is end of life
  (its R828D tuner is discontinued); the vendor recommends the V3 or V4L, and
  counterfeit "V4" listings exist. A genuine unit reports `RTLSDRBlog` /
  `Blog V4` in `rtl_eeprom` and an R828D tuner in `rtl_test`.
- The R828D needs librtlsdr 2.0.x or the rtl-sdr-blog fork. Blacklist the
  kernel DVB drivers.
- Method: passive receive only. A burst recorder squelched on the noise
  floor wrote one IQ file per burst; each burst was demodulated, CRC-checked
  and attributed to a node by its source id; carrier and deviation came from
  an instantaneous-frequency histogram cross-checked against the spectral
  peak pair. 1.024 Msps: at 250 ksps the tuner's IF filter clips the signal.
- Results are in [physical.md](physical.md#measured-on-air).

### Analysis software surveyed

URH, inspectrum, rtl_433 (`-A` pulse analysis), CRC RevEng and similar tools.
None of them de-whitens this link out of the box: they use TI's DN509 byte
order for PN9, and this link uses the bit-reversed order
([physical.md](physical.md#whitening-pn9)).

### Custom receiver firmware (`termoweb_rx`)

The nanoCUL was reflashed with a small avr-gcc firmware, the ancestor of the
nanoCUL and ESP32 firmware in this repository. Its history explains several
design choices in the line protocol:

| Version | Change | Why |
|---|---|---|
| first | fixed 40-byte packets with RSSI and LQI appended | first byte-exact receiver; the first rate (9.99 kbps) duplicated bits |
| 3.1 | fixed 64-byte reads, frames split on the host; `T<hex>` transmit | the CC1101 cannot end a variable frame by signal quality: carrier sense dips on ordinary 2-FSK tone changes, an RXBYTES stall never fires in infinite-length mode, and an RSSI drop still truncated frames |
| 3.2 | dynamic length: start in infinite-length mode, read byte 0, switch to fixed length `(byte0 XOR FF) + 3`; firmware auto-ack (`A1`, `I<id>`) | the heater's ack window is about 16 ms; the host is too slow |
| 3.2.1 | enable interrupts only after the CC1101 is configured | GDO2's power-on toggle fired an interrupt mid-SPI and hung the boot |
| 3.3 | chunked TX past the 64-byte FIFO (pre-fill 63, refill below 32) | the 99-byte program write; the CC1101 reads `TXBYTES` as 0 when the FIFO is exactly full |
| 3.4 | interrupt-driven UART receive (128-byte ring) | host commands were lost while a long RX line was printed |
| 3.5 | re-arm the receiver after a frame lands while a `T` command has interrupts masked, plus a 250 ms liveness guard (`# rxreset`) | receive went silent (TX still worked) about every 12 minutes under real traffic |
| 3.6, 3.7 | LQI experiments, rolled back | LQI reads 0 on every frame since dynamic framing; reading the register directly or turning on the RX CRC engine did not change it. The host CRC is the real quality gate, so this costs nothing. |

Other lessons from the firmware work:

- Hardware sync detection has no error tolerance and fires on noise; the host
  drops weak frames (below about −95 dBm) and relies on the CRC.
- The CC1101's hardware CRC cannot be used (it only supports init `FFFF`), and
  its hardware whitener uses the wrong bit order for this link. Both stay off.
- Tool-paced tests did not show the UART overrun; the real client did within
  a minute. Test with the real client's timing.

## How the big unknowns fell

- **Bit rate.** 104 µs per bit was first rounded to 10 kbps. A CC1101 at
  9.99 kbps duplicated one bit every 24–28 bits; 9.6 kbps (CC1101
  `MDMCFG3 0x83`) is byte exact, and the gateway's AX5043 `TXRATE` gives
  9599.7 bps. Snap a measured bit period to the nearest standard rate.
- **Sync alignment.** Read as 16 bits the sync looks like `16F2`; the frame
  really starts one bit later, so the byte-aligned sync is `2DE5`
  ([physical.md](physical.md#preamble-and-sync)).
- **CRC.** Treating the CRC initial value as an unknown and solving for it
  (a CRC is linear in its init) found CRC-16/CCITT with init `1D0F`. The
  per-class "seeds" seen at first were one CRC computed over whitened bytes:
  a fixed keystream folded into a CRC at five fixed frame lengths looks like
  five seeds.
- **Whitening.** The gateway firmware whitens in software (the AX5043 has no
  PN9 whitener, only a G3RUH-style scrambler, which was ruled out on air). An
  exhaustive search over all 511 PN9 seeds and both bit orders found exactly
  one match: PN9 seeded all ones, bytes assembled MSB first.
- **Length byte.** On air the first byte looked like a "type byte" (`F1`,
  `F2`, `E5`…). It is the logical length XOR `FF`, because the keystream
  starts with `FF`.
- **Addressing.** Powering the heaters on one at a time showed fixed ids, not
  a slot-plus-base scheme. The firmware showed bytes 3–4 are per-hop link
  addresses and bytes 6–10 an end-to-end routing path.
- **"Poll" was a confirmation.** With the gateway unplugged, heaters repeated
  each report five times; sending `57 55` after each report stopped the
  repeats. A station needs both the link ack and this confirmation.
- **Pairing.** Answers that were byte-identical to the gateway's except for
  header byte 11 (`00` instead of `04`) were acked and ignored. The test had
  compared output with the project's own rebuild instead of the gateway's
  captured frame.
- **Firmware code paths.** Nine caller-walk passes in Ghidra missed the
  command builders because they hand frames to the sender through a shared
  memory slot, not a call. A whole-image scan for scalar operands found them.
- **Panel experiments.** Changing one setting at the heater's keypad while
  logging decoded the presets, the advanced settings and the Low Surface
  Temperature index. Factory resets were used deliberately (with the owner
  present) to learn what survives one.

## Mistakes worth not repeating

- **PN9 was "ruled out" for four days.** The first CRC search tried PN9 but
  hard-coded the CRC init to `FFFF`; the second solved the init but no longer
  tried PN9. Record the search space, not just the conclusion.
- **A slow reader looked like a slow heater.** The 63–65 ms "ack delay" was
  the time firmware 3.1 took to clock in its fixed 64-byte read; the real ack
  comes about 16 ms after the frame.
- **A spoofed-source test that spoofed the wrong byte.** Changing byte 3 alone
  looked like "the heater accepts any sender"; byte 6, the originator, was
  still `01`. With byte 6 spoofed, only `01` is accepted.
- **Two data points make a pattern.** Early address arithmetic and a setpoint
  "seed" both fitted two samples and were wrong. Build deliberately varied
  corpora; the corpus rebuild (every CRC-valid frame rebuilt from its parsed
  fields, zero mismatches) became the proof standard.
- **Phantom frame classes.** `F5` and `FD` stood in the notes for days; they
  were misframed captures taken at the wrong centre frequency.
- **"Monday first"** came from cloud documentation; the wire is Sunday first.
- **The setpoint ceiling** looked like a protocol limit; it was a heater
  setting that a factory reset cleared.
- **An observer effect.** A diagnostic tool that turned on auto-ack cured the
  fault it was meant to observe. Use a passive tap.
- **Operations.** A scratch folder on a wipeable path was lost on a reboot;
  `needrestart` restarted `ser2net` and ended an acceptance run.

## Safety findings

- **The reported state can be wrong.** After a panel Boost was cancelled, one
  heater kept its element on for about an hour while every report said mode
  off, duty 0 and not heating. Its board temperature (`pcb_temp`) climbed
  from 29 to 57 °C; a power cycle at the wall stopped it. Do not treat
  duty 0 or a clear heating flag as proof that the element is off.
- **Some writes wipe a heater.** See
  [protocol.md §8](protocol.md#8-factory-reset).
- **There is no security on the link.** No encryption, no authentication, no
  replay protection, no sequence counter. Anyone in radio range can command
  a heater. This integration does not change that.

## What was left out of the public docs

- Raw captures, photos and logs from the researcher's home.
- Decompiled listings and firmware addresses (the facts they established are
  in the protocol pages).
- Vendor manuals and data sheets (link to the vendor's own copies:
  [Sun Ray RF manual](https://atc.ie/wp-content/uploads/Manual-atc-Sun-Ray-RF_v07.pdf),
  [gateway manual](https://www.manualslib.com/manual/2183395/Atc-Sunray-Rf.html)).
- Cloud credentials.

## Contributing new findings

1. Capture on your own equipment. The integration's capture service
   ([radio_identify_capture.md](../radio_identify_capture.md)) records frames
   with a timestamp next to what you did.
2. Change one thing at a time, and write down what you changed and when.
3. Before you share a capture, use the redacted export, or replace network ids,
   identities and serials yourself ([privacy.md](privacy.md)).
4. In docs and tests, rebuild example frames with
   `custom_components/termoweb/backend/radio/dialect.build_frame` and the
   synthetic values above, so CRCs and whitening stay correct.
