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
The protocol reference is the table in
[docs/radio_gateway.md](../docs/radio_gateway.md#for-developers-the-line-protocol).

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

Expected size, ATmega328P: about 7-9 KB of the 30 KB flash left by the
bootloader, and about 1.1 KB of static RAM (two 257-byte packet buffers,
the 255-byte `T` buffer, the 256-byte serial input ring), leaving the rest
of the 2 KB for the stack. CI fails the build above 30720 bytes of flash or
1536 bytes of static RAM.

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
- Receive and auto-ack in dialect A and in dialect B (`Y1`, `N<net>`).
- Long frames: a 99-byte program write sent (`T`) and received.
- `F` retune and back; `D` counters.
- Raw survey (`R60`): bursts reported, the integration recognises the
  dialect, and the main loop keeps up with the edge interrupts on noise.
