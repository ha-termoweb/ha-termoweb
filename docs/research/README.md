# Radio research

This directory holds everything known about the 869 MHz radio link between
TermoWeb / Sun Ray RF heaters and their gateway: how it was found, the
protocol reference, and the open questions. It is for developers. Users who
want to set up a radio gateway start at
[radio_gateway.md](../radio_gateway.md) or [radio_nanocul.md](../radio_nanocul.md).

## Who did the research

The protocol was reverse-engineered by **Andre Faria
([@andremmfaria](https://github.com/andremmfaria))** in September 2026, on
his own gateway and heaters. His work became
ha-termoweb-local, the
first Home Assistant integration to control these heaters over the radio,
and its nanoCUL firmware. This integration's radio backend and firmware grew
out of that project. The pages here summarise his research notes, which stay
private because they contain captures from a real home. Mistakes in the
summary are ours.

## Two protocol families

The radio exists in two variants that share the header layout but differ in
framing. In these pages they are called **dialect A** (sync `2DE5`, PN9
whitening, CRC-16/CCITT; the variant the original research studied) and
**dialect B** (sync `2DD4`, no whitening, CRC-16/MODBUS; found on another
installation in October 2026). The code uses the same names
(`backend/radio/dialect.py`).

## Pages

| Page | Contents |
|---|---|
| [protocol.md](protocol.md) | The overview reference used by `custom_components/termoweb/backend/radio/`: frame layout, dialects, opcode table, the gateway line protocol, survey, pairing, factory reset, re-homing, listen-only captures. Start here. |
| [physical.md](physical.md) | Carrier, deviation and bit rate measured on air; sync; PN9 whitening and its bit order; the CRC; frame classes by length; CC1101 and AX5043 settings. |
| [frames.md](frames.md) | Header fields, routing path and tags, relays, which sender a heater obeys, what it drops, acks or processes, and the lack of security. |
| [messages.md](messages.md) | Every known dialect-A payload in detail: status record, setpoint and mode, presets, program, energy counter, power request, toggles, advanced settings, identity, clock. |
| [sessions.md](sessions.md) | Timing, what a station must do (ack, confirm, registration, keepalive), the vendor gateway's start-up, and dialect-A pairing on the air. |
| [feature-gap.md](feature-gap.md) | Each cloud feature and its radio path, features with none, open questions. |
| [history.md](history.md) | How the protocol was reverse-engineered: equipment, tools, timeline, the mistakes, safety findings. |
| [privacy.md](privacy.md) | Rules for captures, examples and tests, and the CI leak scan. |

## How it relates to the code

| Code | Docs |
|---|---|
| `backend/radio/dialect.py` (framing, PN9, CRCs, `build_frame`) | [protocol.md §2-3](protocol.md#2-frame-layout), [physical.md](physical.md), [frames.md](frames.md) |
| `backend/radio/protocol.py` (payload builders and decoders) | [protocol.md §4](protocol.md#4-opcodes), [messages.md](messages.md) |
| `backend/radio/link.py`, `esphome/`, `firmware/` (the gateway line protocol) | [protocol.md §5](protocol.md#5-gateway-line-protocol-tcp-port-2323), [physical.md](physical.md#building-a-cc1101-receiver) |
| `backend/radio/survey.py`, `capture.py` | [protocol.md §6 and §10](protocol.md#6-surveying-an-unknown-dialect) |
| `backend/radio/pairing.py`, `radio_*.py` | [protocol.md §7-9](protocol.md#7-pairing), [sessions.md](sessions.md#pairing-in-dialect-a) |
| `backend/radio_ws.py` (station duties) | [sessions.md](sessions.md#station-duties) |
| `backend/radio_client.py`, `radio_backend.py` | [radio_backend.md](../radio_backend.md) |

Every example frame in these pages uses synthetic values (network `12 34`,
identity `01 02 … 0C`, serial `X123456`) and was built with the codec, so the
whitening and CRCs are real.
