# Ducaheat (Ducasa) Cloud API — Tevolve v2 (Unofficial)

This document summarizes the Ducaheat backend as implemented by the mobile app (v1.40.1). It diverges from the consolidated **TermoWeb** contract: Ducaheat uses **segmented endpoints** per function and a **different Socket.IO path**.

**Base host:** `https://api-tevolve.termoweb.net`  
**Frontend:** `https://ducaheat.termoweb.net`  
**Auth client (Basic):** `5c49dce977510351506c42db:tevolve` → Base64 `NWM0OWRjZTk3NzUxMDM1MTUwNmM0MmRiOnRldm9sdmU=`

---

## Auth

**POST** `/client/token`  (HTTP Basic + form)

Form fields:
```
grant_type=password&username=<email>&password=<password>
```

Response:
```json
{ "access_token":"…", "token_type":"Bearer", "expires_in":3600, "refresh_token":"…" }
```

Use the `access_token` as `Authorization: Bearer …` for all API calls.

---

## Heater node model (read)

**GET** `/api/v2/devs/{dev_id}/htr/{addr}`

Returns a consolidated object with nested sections such as `status`, `setup`, `prog`, `prog_temps`. Keys are model‑dependent; the app treats unknown keys leniently.

---

## Writes are segmented

Unlike TermoWeb’s `/settings` write, Ducaheat splits writes across multiple endpoints:

### 1) Status (mode / setpoint / units / boost)

**POST** `/api/v2/devs/{dev_id}/htr/{addr}/status`

Body examples:
```json
{ "mode": "manual", "stemp": "22.0", "units": "C" }
```
```json
{ "boost": true }
```

Returns `201 {}`.

> Temperatures are strings with one decimal (e.g., `"22.0"`). `units` is `"C"` or `"F"`.

### 2) Mode only

**POST** `/api/v2/devs/{dev_id}/htr/{addr}/mode`  
Body:
```json
{ "mode": "manual" }
```

### 3) Weekly program

**POST** `/api/v2/devs/{dev_id}/htr/{addr}/prog`  
Payload mirrors the app’s weekly schedule object (structure varies by model). Use the object echoed by the GET call as a template.

### 4) Program temperatures

**POST** `/api/v2/devs/{dev_id}/htr/{addr}/prog_temps`
```json
{ "comfort": "21.0", "eco": "18.0", "antifrost": "7.0" }
```

### 5) Advanced setup / extra options

**POST** `/api/v2/devs/{dev_id}/htr/{addr}/setup`
```json
{ "extra_options": { "boost_time": 60, "boost_temp": "22.0" } }
```

### 6) Lock (child lock)

**POST** `/api/v2/devs/{dev_id}/htr/{addr}/lock`
```json
{ "lock": true }
```

### 7) Select (ownership hint for writes)

**POST** `/api/v2/devs/{dev_id}/htr/{addr}/select`
```json
{ "select": true }
```
Some write operations may require `select: true` before they take effect; de‑select with `select: false` afterwards.

---

## Samples (history / telemetry)

**GET** `/api/v2/devs/{dev_id}/htr/{addr}/samples?start=<ms>&end=<ms>`

`start` and `end` are **epoch milliseconds**. The response shape varies by device and firmware; treat as opaque JSON until stabilized by capture.

---

## Boost / Runback behavior

- Timed override that runs at a boost setpoint for a fixed duration, then reverts.  
- Activation paths:
  - Immediate: `POST …/status { "boost": true }`
  - Defaults: set via `POST …/setup { "extra_options": { "boost_time": 60, "boost_temp": "22.0" } }`
- Related read keys often present under `setup.extra_options` and/or `status`:
  - `boost_time`, `boost_temp`, and `boost_active`.

---

## WebSocket (Socket.IO)

**Path:** `/api/v2/socket_io?token=<access_token>`

The app listens for at least these events:
- `dev_handshake` — initial device list / permissions
- `dev_data` — node payloads (status/setup/prog updates)
- `update` — incremental changes

This differs from TermoWeb’s legacy `/socket.io/1/…` path. Use Socket.IO v0.9 semantics as the server reports.

---

## Example cURL flow

```bash
# 1) Token
TOK=$(curl -sS -u '5c49dce977510351506c42db:tevolve'   -d 'grant_type=password&username=EMAIL&password=PASS'   https://api-tevolve.termoweb.net/client/token | jq -r .access_token)

# 2) Read heater 2
curl -sS -H "Authorization: Bearer $TOK"   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/htr/2

# 3) (Optional) Select before write
curl -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json"   -d '{"select": true}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/htr/2/select

# 4) Set manual 22.0°C
curl -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json"   -d '{"mode":"manual","stemp":"22.0","units":"C"}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/htr/2/status

# 5) Start Boost now
curl -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json"   -d '{"boost": true}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/htr/2/status

# 6) Configure default Boost time
curl -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json"   -d '{"extra_options":{"boost_time":60,"boost_temp":"22.0"}}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/htr/2/setup

# 7) De-select after write
curl -sS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json"   -d '{"select": false}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV_ID/htr/2/select
```

---

## Notes and caveats

- Treat unknown keys as forward‑compatible; the app is tolerant of additional fields.
- Program payload (`/prog`) structure varies by model/firmware. Capture the GET shape first and write it back unchanged after edits.
- Some deployments may accept `/htr/{addr}/boost` as a synonym for boosting, but `/status` with `{ "boost": true }` is consistently present in the app bundle.
