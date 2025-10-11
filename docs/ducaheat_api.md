# Ducaheat (Ducasa) Cloud API — Tevolve v2 (Unofficial, updated)
**Updated:** 2025-10-11

This revision preserves correct details from the existing docs and adds/clarifies items that were **observed in the attached protocol dump** (`ducaheat_htr_dump.jsonl`). It removes unverified claims.

---

## Base & Auth

- **Host:** `https://api-tevolve.termoweb.net`
- **Auth:** OAuth2 password grant at **POST** `/client/token` using the Ducaheat app client (HTTP Basic). Response contains `access_token` for `Authorization: Bearer ...`.

---

## Key findings from the capture (ground truth)

- **Selection gate is in use for heaters (`htr`)**: app issues `POST .../htr/{addr}/select` with `{"select": true}` **before** writes and `{"select": false}` after.
- **Segmented writes are used on heaters:** `POST .../htr/{addr}/status` and `POST .../htr/{addr}/prog` are present.
- **Preset temperatures are written via `/status`**: payloads include `comf_temp, eco_temp, ice_temp` along with `units` when updating presets.
- **Weekly program resolution in this dump is 24 slots per day**, days `0, 1, 2, 3, 4, 5, 6` (example bodies are hourly with 24 integers per day).
- **Ancillary reads in use:** `GET .../mgr/nodes` (node inventory) and `GET .../mgr/rtc/time` (gateway RTC).
- **Power‑monitor telemetry:** `GET .../pmo/{addr}/samples?start=<sec>&end=<sec>` observed with **epoch seconds**.

> The dump does **not** include `/prog_temps` calls, `/acm` samples, or `/htr` samples. Those behaviours remain as previously documented but are unverified in this dataset.

---

## Snapshot (`dev_data`) and node model
The app consumes a gateway snapshot over Socket.IO (`dev_data`) and REST endpoints for details. Snapshot objects include
fleet/system sections (`htr_system`, `pmo_system`), gateway `geoData/geo_data`, `away_status`, `discovery`, `connected`, and an array of `nodes` with core identity and capabilities.

Each thermal node (`htr` or `acm`) has a consolidated REST read at:

**GET** `/api/v2/devs/{dev_id}/{type}/{addr}`

with sections like `status`, `setup`, `prog`, `prog_temps` (presence may vary by model/firmware). Echo the shape you read back to the server when writing program structures.

---

## Writes — segmented endpoints

> **Always claim selection immediately before a write and release afterward.**

### Change live status
**POST** `/api/v2/devs/{dev_id}/{type}/{addr}/status`

Used for mode/setpoint/units. In the capture it is also used on `htr` to update **preset temperatures**:

Examples observed:
```json
{"mode":"off"}
{"mode":"manual","stemp":"20.5","units":"C"}
{"ice_temp":"5.0","eco_temp":"17.5","comf_temp":"20.5","units":"C"}
```

Formatting rules:
- Temperatures are **strings with exactly one decimal**, e.g. `"22.0"`.
- Units are **uppercase** `"C"` or `"F"`.
- Returns `201 {}` on success.

### Change mode only
**POST** `/api/v2/devs/{dev_id}/{type}/{addr}/mode`

`{"mode":"auto"|"manual"|"off"}`

### Weekly program
**POST** `/api/v2/devs/{dev_id}/{type}/{addr}/prog`

Send the **full** program object echoed from GET. In this dump, `htr` days `"0"..."6"` each carry **24** integers (hourly). Example (truncated):

```json
{"prog":{"0":[2,2,2,2,...,2],"1":[...],"2":[...]}}
```

> Older notes refer to 30‑minute resolution. This capture shows **hourly (24‑slot)** payloads for the tested heater.

### Program preset temperatures
**POST** `/api/v2/devs/{dev_id}/{type}/{addr}/prog_temps`

Not observed in this dump. The app used `/status` with keys `ice_temp`, `eco_temp`, `comf_temp`. Keep this endpoint for compatibility where devices expose it; prefer the capture‑proven `/status` shape for `htr` preset updates.

### Advanced setup & lock
**POST** `/api/v2/devs/{dev_id}/{type}/{addr}/setup` — defaults like `extra_options.boost_time` / `boost_temp` (strings for temps).  
**POST** `/api/v2/devs/{dev_id}/{type}/{addr}/lock` — `{"lock": true|false}`.

### Selection (required gate)
**POST** `/api/v2/devs/{dev_id}/{type}/{addr}/select`
- Claim: `{"select": true}` → `201 {}`
- Release: `{"select": false}` → `201 {}`

Apply `select → write → deselect` around **every** write.

### Boost (accumulator)
**POST** `/api/v2/devs/{dev_id}/acm/{addr}/boost`

Start: `{"boost": true, "boost_time": 60..600, "stemp": "##.#", "units": "C|F"}`  
Stop: `{"boost": false}`

Not present in this heater‑focused dump; documented here for completeness.

---

## Samples (history/telemetry)

- **Power monitor (`pmo`)**: `GET .../pmo/{addr}/samples?start=<sec>&end=<sec>` — **epoch seconds** observed.  
- **Heater/Accumulator (`htr`/`acm`)**: Not present in this dump; prior docs treat thermal nodes as **epoch milliseconds**.

---

## WebSocket (Engine.IO v3 + Socket.IO v2)

- **Endpoint:** `/socket.io?token=<access_token>&dev_id=<dev_id>`
- **Namespace:** `/api/v2/socket_io`
- **Flow (as implemented by the app and integration):** polling open → join namespace (polling) → drain → WebSocket upgrade → `2probe/3probe/5` → namespace ack → request `["dev_data"]` → subscribe to `"/{type}/{addr}/{status|samples}"`.
- **Keepalives:** reply Engine.IO `2` with `3`; reply Socket.IO pings; answer app `"ping"` with `"pong"` events.

---

## Other endpoints seen in the dump

- `GET /api/v2/devs/{dev_id}/mgr/nodes` — node inventory
- `GET /api/v2/devs/{dev_id}/mgr/rtc/time` — gateway RTC
- `GET /api/v2/grouped_devs` — grouped device metadata
- `GET /api/v2/user/preferences`, `GET /api/v2/users/{user_id}/privacy` — user settings
- `GET /client-data/15/*` and `GET /assets/maxPowerConfig.json` — static app resources
- `GET /api/v2/devs/{dev_id}/pmo/{addr}/power` — instantaneous power (shape varies)

---

## cURL examples (reflecting capture)

```bash
# Token
TOK=$(curl -sS -u '5c49dce977510351506c42db:tevolve'   -d 'grant_type=password&username=EMAIL&password=PASS'   https://api-tevolve.termoweb.net/client/token | jq -r .access_token)

# Select → write → deselect (heater)
curl -sS -H "Authorization: Bearer $TOK" -H 'Content-Type: application/json'   -d '{"select": true}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV/htr/$ADDR/select

curl -sS -H "Authorization: Bearer $TOK" -H 'Content-Type: application/json'   -d '{"mode":"manual","stemp":"21.0","units":"C"}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV/htr/$ADDR/status

curl -sS -H "Authorization: Bearer $TOK" -H 'Content-Type: application/json'   -d '{"select": false}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV/htr/$ADDR/select

# Program (24 slots per day in this capture)
curl -sS -H "Authorization: Bearer $TOK" -H 'Content-Type: application/json'   -d '{"prog":{"0":[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]}}'   https://api-tevolve.termoweb.net/api/v2/devs/$DEV/htr/$ADDR/prog

# PMO samples (epoch seconds)
curl -sS -H "Authorization: Bearer $TOK"   "https://api-tevolve.termoweb.net/api/v2/devs/$DEV/pmo/$PMO_ADDR/samples?start=1759269600&end=1761955200"
```

---

## Validation invariants

- Temperatures **must** be strings with **one decimal**; **units uppercase**.
- Claim selection before **every** write; release after, including on failure.
- For weekly programs, **echo the GET shape** and write the whole object.
