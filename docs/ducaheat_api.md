# Ducaheat (Ducasa) Cloud API — Tevolve v2 (Unofficial, accumulators supported)

This document folds in the latest user-provided captures for **accumulators (type `acm`)** and **power meter (type `pmo`)**, and aligns the path patterns with TermoWeb while preserving Ducaheat’s segmented writes.

Base host: `https://api-tevolve.termoweb.net`
Auth client (Basic): `5c49dce977510351506c42db:tevolve` (Base64: `NWM0OWRjZTk3NzUxMDM1MTUwNmM0MmRiOnRldm9sdmU=`)
Socket namespace: `/api/v2/socket_io` (Socket.IO 0.9 semantics)

## 1) Discover devices and nodes

- GET `/api/v2/devs/` → find your `dev_id` (hub). Example (obfuscated):
{ "devs": [ { "dev_id": "1234", "name": "HUB-A", "product_id":"1234", "fw_version":"1.36.0", "serial_id":"1.2.3.4" } ], "invited_to": [] }

- GET `/api/v2/devs/{dev_id}/mgr/nodes` → list nodes (rooms/devices). Example (obfuscated):
{
  "nodes": [
    { "name":"Bedroom",  "addr":2, "type":"acm", "installed":true, "lost":false, "uid":"ABCD1..." },
    { "name":"Entrance", "addr":3, "type":"acm", "installed":true, "lost":false, "uid":"ABCD2..." },
    { "name":"Dining",   "addr":4, "type":"acm", "installed":true, "lost":false, "uid":"ABCD3..." },
    { "name":"Meter",    "addr":5, "type":"pmo", "installed":true, "lost": true, "uid":"ABCD4..." }
  ]
}

## 2) Generalized node path

Use `{type}` in paths:

- READ consolidated node (heater or accumulator):
  - GET `/api/v2/devs/{dev_id}/{type}/{addr}` with `type ∈ {htr, acm}`
- Power meter root:
  - GET `/api/v2/devs/{dev_id}/pmo/{addr}` → `{ "power_limit": "6900" }`

Historical samples (shape varies by model):
- GET `/api/v2/devs/{dev_id}/{type}/{addr}/samples?start=<ms>&end=<ms>`

## 3) Example `acm` (accumulator) response (obfuscated)

Includes `status` (heating / ventilation / charging, current_charge_per, target_charge_per, boost, temperatures as strings), `setup` (charging_conf with slot_1/slot_2 and active_days), and a weekly `prog` keyed by `"0"`…"6" with 48 half-hour slots in many firmwares. Preserve the shape you read before writing back.

## 4) Power meter (`pmo`) minimal read

- GET `/api/v2/devs/{dev_id}/pmo/{addr}` → `{ "power_limit": "6900" }`
- Samples (if implemented for pmo): `/api/v2/devs/{dev_id}/pmo/{addr}/samples?start=<ms>&end=<ms>`

## 5) Writes (segmented) for `htr` and `acm`

- POST `/api/v2/devs/{dev_id}/{type}/{addr}/status`
  - `{ "mode": "manual", "stemp": "22.0", "units": "C" }`
  - `{ "boost": true }`
- POST `/api/v2/devs/{dev_id}/{type}/{addr}/mode` → `{ "mode": "off" }`
- POST `/api/v2/devs/{dev_id}/{type}/{addr}/prog` → weekly schedule object
- POST `/api/v2/devs/{dev_id}/{type}/{addr}/prog_temps` → `{ "comfort":"21.0","eco":"18.0","antifrost":"7.0" }`
- POST `/api/v2/devs/{dev_id}/{type}/{addr}/setup` → `{ "extra_options": { "boost_time": 60, "boost_temp": "22.0" } }`
- POST `/api/v2/devs/{dev_id}/{type}/{addr}/lock` → `{ "lock": true }`
- POST `/api/v2/devs/{dev_id}/{type}/{addr}/select` → `{ "select": true }`

## 6) WebSocket

- Connect to `/api/v2/socket_io?token=<Bearer>` (Socket.IO 0.9). Typical events: `dev_handshake`, `dev_data`, `update`.
- Push paths observed: `/mgr/nodes`, `/{type}/{addr}/status`, `/{type}/{addr}/setup`, `/{type}/{addr}/prog`.


## 7) cURL sanity flow

1. Token:
   `curl -u '5c49dce977510351506c42db:tevolve' -d 'grant_type=password&username=EMAIL&password=PASS' https://api-tevolve.termoweb.net/client/token`

2. List devices:
   `curl -H "Authorization: Bearer $TOK" https://api-tevolve.termoweb.net/api/v2/devs/`

3. Nodes:
   `curl -H "Authorization: Bearer $TOK" https://api-tevolve.termoweb.net/api/v2/devs/$DEV/mgr/nodes`

4. Read an accumulator:
   `curl -H "Authorization: Bearer $TOK" https://api-tevolve.termoweb.net/api/v2/devs/$DEV/acm/2`

5. Start Boost:
   `curl -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" -d '{"boost":true}' https://api-tevolve.termoweb.net/api/v2/devs/$DEV/acm/2/status`

6. Power meter:
   `curl -H "Authorization: Bearer $TOK" https://api-tevolve.termoweb.net/api/v2/devs/$DEV/pmo/5`
