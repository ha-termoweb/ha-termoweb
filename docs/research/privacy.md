# Privacy rules for radio research

This repository is public. Radio captures, logs and test vectors easily carry
values that identify one home. Keep them out of commits, issues and pull
requests.

## What identifies an installation

| Value | Where it shows up |
|---|---|
| Network id (2 bytes, both dialects) | Bytes 1-2 of every frame, `net=` in the gateway's `Q` line, config entries, raw captures. Treat every network id seen in a real capture as private, including dialect A. |
| Heater identity (12 bytes) and serial | Pairing announcements (`77 …`), identity replies (`5B …`), the device registry. A heater keeps its identity for life, across factory resets. |
| Gateway MAC, cloud `dev_id`, gateway serial | The `mac=` field, the cloud integration's attributes, the gateway's label. |
| USB serial of a nanoCUL | `/dev/serial/by-id/…` paths, ser2net configs. |
| Hostnames, IP addresses, home directory paths | URLs, configs, logs, screenshots. |
| Room names, schedules, energy totals, measured power | Heater names, entity ids, status records. |

## Rules for docs and tests

- Use synthetic values: network `12 34`, identity `01 02 03 … 0C`, serial
  `X123456`, documentation addresses `192.0.2.x`.
- Never paste captured bytes. Rebuild example frames with
  `custom_components/termoweb/backend/radio/dialect.build_frame` from
  synthetic fields, so the whitening and CRC are right.
- Facts are fine: opcodes, field layouts, timings, CRC and whitening
  parameters, behaviours.
- Facts learned from the vendor's firmware are fine; decompiled code,
  disassembly and firmware hex dumps are not.
- When you share a capture file in an issue, use the capture service's
  `redact` option, or send the unredacted file privately.

## CI leak scan

The **Leak scan** workflow (`.github/workflows/leak-scan.yml`) fails a build
when a tracked file contains a string from the `LEAK_PATTERNS` repository
secret. The list lives in a secret because a committed list would publish
exactly the values it protects. On a match the log shows only file and line
numbers, never the matched text.

Maintainers set it up once:

1. Write the values to a file outside any repository, one per line: network
   ids (both `1234` and `12 34` forms), heater identities (24 hex digits, and
   the spaced form), serials, the gateway MAC and `dev_id`, hostnames, your
   usernames. Avoid short or common strings: they match ordinary code.
2. Check the current tree is clean: `git grep -nIiF -f <file>`. No output
   means the scan will start green.
3. Store it: `gh secret set LEAK_PATTERNS -R ha-termoweb/ha-termoweb < <file>`
   (or Settings → Secrets and variables → Actions), then delete the file.
   Keep the source list in a password manager to extend it later.

Limits: the scan checks the files of the commit being built, not the git
history; it matches exact strings only; pull requests from forks get no
secrets, so the scan skips there and runs once the change reaches `main`.
