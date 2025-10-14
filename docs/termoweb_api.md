# TermoWeb Cloud API — REST & WebSocket (capture‑verified, 2025‑08‑13)

**Tevolve app family**: TermoWeb, Ducaheat, and (future) Tevolve white-label the same API shape but talk to separate backend deployments with isolated user databases.

**Backends & OAuth clients**

| App (brand) | Base host | Client ID | Client secret | Notes |
| --- | --- | --- | --- | --- |
| TermoWeb | `https://control.termoweb.net` | `52172dc84f63d6c759000005` | `bxv4Z3xUSe` | Legacy/primary deployment. |
| Ducaheat | `https://api-tevolve.termoweb.net` | `5c49dce977510351506c42db` | `tevolve` | Uses identical endpoints with brand-specific assets. |
| Tevolve | _TODO_ | _TODO_ | _TODO_ | Placeholder for the third Android app. |

For both implemented apps, Basic client credentials are sent as `Authorization: Basic <base64(client_id:client_secret)>`:

- TermoWeb: `NTIxNzJkYzg0ZjYzZDZjNzU5MDAwMDA1OmJ4djRaM3hVU2U=`
- Ducaheat: `NWM0OWRjZTk3NzUxMDM1MTUwNmM0MmRiOnRldm9sdmU=`

**Auth**: Bearer token obtained via **POST `/client/token`** (basic client credentials + password grant).
**Content type**: JSON for all REST endpoints (request/response).

> This document reflects endpoints and behaviours seen in live traffic from the official Android app on 2025‑08‑13. The vendor’s app traffic is treated as authoritative.


## Auth

### POST `/client/token`
- **Headers**
  - `Authorization: Basic <base64(client_id:client_secret)>`
  - `Content-Type: application/x-www-form-urlencoded`
- **Body**: `grant_type=password&username=<email>&password=<pwd>`
- **200 Response**
  ```json
  {
    "access_token": "<opaque>",
    "token_type": "Bearer",
    "expires_in": 604800,
    "scope": "/user:W /devs/*:W"
  }
  ```

Use `Authorization: Bearer <access_token>` for the endpoints below.


## Devices

### GET `/api/v2/devs/`
Returns devices you own and invitations.
```json
{ "devs": [ { "dev_id": "abcdef...", "name": "Hub", "fw_version": "1.34", "serial_id": "1234" } ], "invited_to": [] }
```

### GET `/api/v2/devs/{dev_id}/mgr/nodes`
Returns discovered nodes (heaters, etc.).
```json
{ "nodes": [ { "type":"htr","addr":2,"name":"Living room ","installed":true,"lost":false,"hw_version":"1.5","fw_version":"1.13"} ] }
```

### GET `/api/v2/devs/{dev_id}/mgr/rtc/time`
Hub RTC snapshot.
```json
{ "y":2025,"n":7,"d":13,"h":15,"m":1,"s":31,"w":3 }
```

### GET `/api/v2/devs/{dev_id}/geo_data`
Location & timezone for the hub.
```json
{ "country":"FR","state":"Paris","city":"Paris","tz_code":"Europe/Paris","zip":"4321" }
```

### GET `/api/v2/devs/{dev_id}/storage`
Misc persistent data.
```json
{ "location_data": {}, "RGPD_time": "Mon Dec 23 2024 11:56:54 UTC", "version": 2 }
```

### GET `/api/v2/devs/{dev_id}/htr_system/power_limit`
System power limit (string).
```json
{ "power_limit": "0" }
```


## Heaters (htr)

### GET `/api/v2/devs/{dev_id}/htr/{addr}/settings`
Representative response:
```
{
  "name":"Guest bedroom ",
  "priority":0,
  "prog":[ /* 168 ints (Mon 00→Sun 23), values 0/1/2 = cold/night/day */ ],
  "units":"C",
  "ptemp":["10.0","16.0","21.0"],       // [cold, night, day]
  "mtemp":"25.7",                       // ambient
  "stemp":"10.0",                       // setpoint (manual)
  "mode":"off",                         // "auto"|"manual"|"off"
  "max_power":"974",
  "state":"off",
  "true_radiant_active":false,
  "window_state_active":false,
  "sync_status":"ok"
}
```

**Type quirks**
- Many numeric fields are returned as strings (e.g. `"stemp"`, `"max_power"`). Preserve one decimal place when writing temperatures.

### POST `/api/v2/devs/{dev_id}/htr/{addr}/settings`
Partial updates; server merges and replies **201** with `{}`.
Common bodies:
```json
{ "mode":"manual","units":"C","stemp":"11.5" }
{ "mode":"auto" }
{ "units":"C","prog":[ /* 168 values in {0,1,2} */ ] }
```

### GET `/api/v2/devs/{dev_id}/htr/{addr}/advanced_setup`
```json
{ "control_mode":4, "units":"C", "true_radiant_enabled":false, "window_mode_enabled":false, "sync_status":"ok" }
```


## Samples (history)

### GET `/api/v2/devs/{dev_id}/{node_type}/{addr}/samples?start=<unix>&end=<unix>`
- `node_type`: observed `htr`
- Response:
```json
{
  "samples": [
    { "t": 1755039600, "temp": "25.3", "counter": "2635097.00" }
  ]
}
```
- `counter` is a cumulative **Wh** total (monotonic). kWh delta between samples = `(Δcounter)/1000`.

Observed cadence for `htr`: ~3600 s.

### Home Assistant energy integration

- Each heater’s `counter` value maps to a Home Assistant **Energy** sensor, and an aggregate sensor sums all heater counters for whole-home tracking.
- Add these sensors in **Settings → Dashboards → Energy** so they appear in the Energy Dashboard.
- The integration polls this endpoint hourly. When the cloud emits optional `htr/samples` WebSocket events, sensors update as those pushes arrive.
- Call the `termoweb.import_energy_history` service to backfill past samples into Home Assistant’s statistics database. The importer now extends through the current minute and reconciles with any previously recorded totals.



## Real‑time (legacy Socket.IO 0.9)

### Handshake
```
GET /socket.io/1/?token=<Bearer>&dev_id=<dev_id>&t=<ms>
→ "<sid>:60:60:websocket,xhr-polling"
```
- **Headers**: include `Origin: https://localhost` alongside the mobile app UA.

### WebSocket URL
```
wss://control.termoweb.net/socket.io/1/websocket/<sid>?token=...&dev_id=...
```
- **Headers**: reuse `Origin: https://localhost` when opening the websocket.

### Namespace & heartbeats
- Join: `1::/api/v2/socket_io`
- Heartbeat: server sends `2::`, reply `2::` about every 25–30 s.

### Events (typical)
- Request snapshot: `5::/api/v2/socket_io:{"name":"dev_data","args":[]}`
- Batched deltas pushed as: `5::/api/v2/socket_io:{"name":"data","args":[ [ { "path":"...", "body":{...} } ] ]}`

Observed paths in pushes: `/htr/<addr>/settings`, `/htr/<addr>/advanced_setup`, `/mgr/nodes`, `/geo_data`, `/htr_system/power_limit`.

---

### Notes / gotchas
- Mobile apps send `User-Agent: TermoWeb/...` together with `X-Requested-With: com.casple.termoweb.v2` on REST and websocket calls; mirror these headers to avoid WAF oddities.
- Always send temperatures as strings with one decimal (e.g. `"20.0"`), otherwise some backends return `400`.
- POST `/htr/*/settings` returns **201** `{}` and echoes via WebSocket shortly after; do a timed fallback poll if needed.
- Schedules: 168‑element `prog` array with values `{0,1,2}` mapping to presets `[cold, night, day]` in `ptemp`.
