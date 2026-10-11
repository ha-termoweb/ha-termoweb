# Radio firmware

Two firmwares let Home Assistant talk to TermoWeb / Sun Ray heaters over
their own 869 MHz radio link, both with a CC1101 radio chip:

| | nanoCUL stick | ESP32 radio gateway |
|---|---|---|
| Hardware | Arduino Nano (ATmega328P, 16 MHz, 32 KB flash, 2 KB RAM) + CC1101 (nanoCUL868) | ESP32 / ESP32-S3 + CC1101 module |
| Source | `firmware/nanocul/` + `firmware/radio_core/` | `esphome/termoweb_radio.h` + `esphome/termoweb-radio.yaml` |
| Link to Home Assistant | USB serial, 115200 8N1 | TCP port 2323 over WiFi |
| Version | `3.7-nanocul` | `3.7-esp32` |
| User guide | [docs/radio_nanocul.md](../docs/radio_nanocul.md) | [docs/radio_gateway.md](../docs/radio_gateway.md) |

Both speak the same line protocol (`termoweb_rx`) with the same commands
and the same replies, so the integration's `RadioLink` drives either one.
The protocol reference is [Line protocol](#line-protocol) below.

## Versions

`<protocol>-<port>`: the number is the line protocol and feature level,
which both ports share; the suffix names the hardware. 3.7 is the first
level with runtime dialects (`Y`, `N`), `F` and the raw survey (`R`). The
integration treats a firmware as dialect-capable when its `Q` line has a
`dialect=` field, and as survey-capable from 3.7 with an `-esp32` or
`-nanocul` suffix. Older nanoCUL firmware (termoweb_rx 3.5/3.6 from the
retired ha-termoweb-local project) has no suffix and speaks dialect A only.

## The shared core: `radio_core/`

Portable C99 (no stdio, no heap, no floating point; strings in flash on
AVR) with everything that does not depend on the hardware:

- `radio_core.c`: the two dialects (length rule, PN9, CRC-16/CCITT-1D0F and
  CRC-16/MODBUS, CRC verdict, the 8-byte link ack), the command parser (an
  incremental state machine fed one byte at a time, so an argument such as
  `NAFDA` is never read as commands), every reply line, and the receive
  state machine (RX FIFO drain, dynamic frame length, packet handler with
  auto-ack, stuck-receiver guard).
- `cc1101.c`: the CC1101 register table and driver (RX arm, fixed-length
  switch, transmit with TX FIFO refill up to 255 bytes, survey mode).
- `rc_hal.h`: the few functions a port provides: one output byte, a
  microsecond clock, a busy-wait, register-level SPI, GDO masking, a short
  critical section, and the survey's edge capture.

The core is written so `esphome/termoweb_radio.h` can include it later
(C linkage, register-level SPI HAL, `mac=` support). The ESP32 firmware does
not use it yet: it runs on the owner's gateway, and switching it to the core
is a separate change after a flash test. Until then, the core follows the
ESP32 file function for function, and the host tests check its replies
against the ESP32's formats.

## nanoCUL: build and flash

```
cd firmware/nanocul
make                          # termoweb_rx.hex, prints flash/RAM use
make flash PORT=/dev/ttyUSB0  # BAUD=57600 for the classic bootloader
make signature PORT=...       # find the bootloader baud rate
```

Needs `gcc-avr`, `avr-libc`, `binutils-avr` and `avrdude`. `PA=0x50`
builds a 0 dBm stick (default `0xC0`, +10 dBm). Every GitHub Release has a
ready-made `termoweb_rx-<version>.hex` (the Firmware workflow attaches it),
and every CI run keeps one as an artifact.

Wiring (a-culfw nanoCUL board): MOSI D11, MISO D12, SCK D13, CS D10,
GDO0 D3 (INT1), GDO2 D2 (INT0).

Size, ATmega328P (avr-gcc, 3.7-nanocul): 9322 bytes of flash (of the 30 KB
left by the bootloader) and 1113 bytes of static RAM (two 257-byte packet
buffers, the 255-byte `T` buffer, the 256-byte serial input ring), leaving
about 900 bytes of the 2 KB for the stack. CI fails the build above 30720 bytes of flash or
1536 bytes of static RAM.

## No radio chip

At every reset (on the nanoCUL: every port open) the core reads the
CC1101's PARTNUM and VERSION status registers (`cc1101_present()`). A
CC1101 reports 00/14 (older silicon 00/04); an empty socket reads the MISO
idle level, all 0x00 or all 0xFF (the nanoCUL port enables MISO's pull-up,
so it reads 0xFF rather than noise). VERSION 0x00 or 0xFF, or PARTNUM 0xFF,
means no radio; any other id, a clone's included, counts as one. Without a
radio:

- the banner ends ` radio=missing`, followed by `# error: no CC1101 radio found`;
- `Q` ends ` radio=missing` (with a radio: ` radio=ok`), and the integration
  stops the setup with "its radio module is not connected";
- `T` replies `TXERR no radio`, `R` replies `# error: no CC1101 radio found`;
- the chip is never configured, GDO edges stay masked, nothing is received.
  `Y`, `F`, `N`, `I`, `A` still answer, so the host's handshake completes.

## Differences between the ports

Same commands, replies and behaviour, except where the hardware differs:

| | nanoCUL | ESP32 |
|---|---|---|
| `mac=` in banner and `Q` | left out: the Nano has no unique id; the integration names the stick by its USB serial number | WiFi MAC |
| Reset to defaults | on every port open (DTR resets the Nano) | on every new TCP connection |
| Raw survey per burst | up to 256 runs (48 before the RSSI trigger, 208 after): preamble, sync word and about the first 20 bytes | up to 3000 runs (10 ms before, 100 ms after) |
| Survey run resolution | 0.5 µs, runs of 32 ms or more report 32767 | 1 µs |
| Command input buffer | 256 bytes | 1024 bytes |
| Boot self-test | prints `# self-test FAILED ...` only when it fails | logs to the ESPHome log |

## Tests

`tests/firmware/` compiles `radio_core` with the host `gcc` into a shared
library and drives it from pytest through a fake CC1101
(`tests/firmware/host_hal.c`). The frames and acks are checked byte for byte
against the integration's codec (`backend/radio/dialect.py`), the replies
against the ESP32's formats. The tests skip when `gcc` is missing.

## Needs testing on hardware

- Flash on an Arduino Nano + CC1101; banner within 2 s of opening the port.
- Without the CC1101 (unplugged module): `radio=missing` in the banner and `Q`.
- Receive and auto-ack in dialect A and in dialect B (`Y1`, `N<net>`).
- Long frames: a 99-byte program write sent (`T`) and received.
- `F` retune and back; `D` counters.
- Raw survey (`R60`): bursts reported, the integration recognises the
  dialect, and the main loop keeps up with the edge interrupts on noise.

## Line protocol

Every line the gateway sends ends with `\r\n`. Send one command per line,
ending with `\n`.

| Command | Reply | Meaning |
|---|---|---|
| (connect) or `V` | `# termoweb_rx 3.7-esp32 freq=869.525 rate=9.6k sync=2DE5 mode=dynamic tx=paC0 mac=AA:BB:CC:DD:EE:FF` | Banner. Sent first on every new connection. |
| `Q` | `# Q termoweb_rx 3.7-esp32 freq=869.525 pa=C0 sync=2DE5 mode=dynamic autoack=off id=01 dialect=A net=1234 mac=AA:BB:CC:DD:EE:FF` | Status. `mac` is the WiFi MAC: a stable device id. The nanoCUL firmware (3.7-nanocul) appends `radio=ok`, or `radio=missing` when no CC1101 answered at the last reset; the integration then stops the setup. |
| `Y0` / `Y1` | `# dialect=A sync=2DE5` / `# dialect=B sync=2DD4` | Select the on-air dialect (below). Error: `# Y? expected 0 or 1`. |
| `N<hex4>`, e.g. `N1234` | `# net=1234` | Network id used in auto-acks. Error: `# N? expected 4 hex digits`. |
| `I<hex2>`, e.g. `I01` | `# id=01` | Our station id, used by auto-ack. |
| `A0` / `A1` | `# autoack=off` / `# autoack=on` | Auto-ack off/on. |
| `T<hex>` | `TX <micros> <n> <hex>` | Transmit up to 255 bytes exactly as given. Errors: `TXERR empty or bad hex`, `TXERR underflow`, `TXERR marcstate=<hex>`, `TXERR no radio` (no CC1101; nanoCUL). |
| `F<kHz>`, e.g. `F869525` | `# freq=869.525 word=21717A` | Retune (779000 to 928000). Error: `# F? expected kHz`. |
| `X` | `# raw=on` / `# raw=off` | Append the 2 raw radio status bytes to RX lines. |
| `D` | `# marcstate=.. pktstatus=.. rxbytes=.. syncs=<n> rxreset=<n>` | Radio chip state dump. |
| `R<secs>`, e.g. `R120` | `# survey on secs=120 floor=-104.0 thr=-96.0`, then per burst `RAWB <n> <rssi>`, `RAW <n> +<us> -<us> ...`, `RAWE <n> <runs>`; finally `# survey off bursts=<n>` | Raw survey for unknown dialects (1–600 s, `R0` ends it early). The radio switches to raw bit mode, so it hears any 2-FSK signal at this frequency and bit rate range, whatever its sync word and framing. Each burst louder than the noise floor + 8 dB is sent as run lengths of the demodulated bits (sign = level, value = µs). Packet reception, auto-ack and transmit (`TXERR survey active`) are off until the survey ends. |

Unprompted lines:

- `RX <micros> <rssi_dbm> <lqi> <crc_ok> <hex>`: a received frame, bytes as on
  air. `crc_ok` is 1 when the frame's CRC is valid in the current dialect.
- `ACK <micros> <hex>`: an auto-ack the gateway sent, bytes as on air.
- `# rxreset n=<count>`, `# uart overrun`: receiver recovery, and command
  input lost because it arrived too fast.

Dialects (`Y`):

| | Dialect A (`Y0`, default) | Dialect B (`Y1`) |
|---|---|---|
| Sync word | 2DE5 | 2DD4 |
| Byte 0 | total length minus 3 | total length |
| Scrambling | PN9 keystream (FF 87 B8 59 ...) | none |
| CRC | CRC-16/CCITT, init 0x1D0F, final XOR 0xFFFF, over the descrambled bytes | CRC-16/MODBUS (0xA001 reflected, init 0xFFFF) |
| Auto-ack on air | `05 <net> <id> <src> 80 <crc>`, scrambled | `08 <net> <id> <src> 80 <crc>` |

In both, the CRC covers bytes 0 to total-3 and is sent high byte first.
Auto-ack fires only for a frame with a valid CRC, addressed to our station id,
with flags 00.

Every new connection resets the gateway to its defaults: dialect A, network
0000 (none until `N` sets it), station id 01, auto-ack off,
869.525 MHz. Send `Y`, `N`, `I` and `A1`
again after each connect.
