# HA-TermoWeb Architecture Overview

This document defines the **target end state** for the TermoWeb integration.
It is authoritative: implementation work must align to this design and remove
any legacy or transitional paths that conflict with it.

## Design principles

- **Latest Home Assistant only.** The integration targets the current HA
  baseline (`homeassistant>=2025.1.0`). There are no compatibility shims or
  fallback imports for older versions.
- **Docs-first contract.** Documentation describes the final architecture; code
  changes must move the runtime toward this end state.
- **Vendor payloads are fixed.** Vendor REST/WS schemas are stable external
  contracts. Wire models keep vendor field names intact; normalization happens
  after decoding into domain state.
- **Inventory is immutable.** A single inventory snapshot is captured during
  setup and never rebuilt or cached again elsewhere.
- **One canonical state pipeline.** Device updates flow through a single path:
  inventory snapshot → domain deltas → `DomainStateStore` → `DomainStateView`.
- **Vendor isolation.** TermoWeb vs Ducaheat differences exist only in
  backend/planner/codec modules. Entities and platforms remain vendor-agnostic.
- **Pydantic on the wire only.** Payload parsing/serialization uses Pydantic
  models; domain state is plain dataclasses or standard Python types.

## Canonical data pipeline

1. **Config entry setup** takes the brand selection, authenticates, and creates a
   single runtime container (`EntryRuntime`).
2. **Inventory snapshot** (gateway + nodes) is retrieved once and stored in the
   runtime. Inventory never changes for the lifetime of the entry.
3. **Update sources** (REST polling, WebSocket push) produce **domain deltas**.
4. **DomainStateStore** applies deltas and holds the canonical in-memory state.
5. **DomainStateView** provides read-only access for entities.

Entities and services never read raw REST/WS payloads. They **only** read through
`DomainStateView` and must not reconstruct or cache inventory data.

## Vendor boundary

Vendor-specific logic is confined to:

- `codecs/` (wire parsing/serialization)
- `planner/` (command planning/validation)
- `backend/` (REST + WebSocket transport)

Entity platforms and shared domain logic must not reference vendor-specific
clients, payload shapes, or protocol details.

## Module map (authoritative)

This map defines responsibilities for each module family in the final design.

- `__init__.py` — config entry setup/teardown, runtime construction, and platform
  forwarding.
- `runtime.py` — `EntryRuntime` definition, `require_runtime(...)` accessor, and
  runtime invariants (single instance per entry).
- `inventory.py` — immutable inventory models and lookup helpers.
- `domain/` — domain dataclasses, deltas, `DomainStateStore`, and
  `DomainStateView`.
- `backend/` — vendor-specific REST/WS clients (including the REST client
  implementation), protocol details, and brand selection.
- `codecs/` — Pydantic payload models and conversion to/from domain types.
- `planner/` — vendor-specific write orchestration and validation rules.
- `entities/` — vendor-agnostic entity implementations (climate, sensor,
  binary_sensor, button, number, etc.) that read via `DomainStateView`.
- `services/` — Home Assistant services and rate-limited import flows that rely
  on the runtime and domain store.

## Runtime flow diagram

```mermaid
flowchart LR
    User[[Home Assistant UI]]

    subgraph HA[Home Assistant · TermoWeb Integration]
        Setup[Config Entry Setup]
        Runtime[EntryRuntime]
        Inventory[Inventory Snapshot]
        Backend[Backend (REST + WS)]
        Deltas[Domain Deltas]
        Store[DomainStateStore]
        View[DomainStateView]
        Entities[Entity Platforms]
        Services[Services]
    end

    subgraph Cloud[TermoWeb / Ducaheat Cloud]
        REST[REST API]
        Socket[Socket.IO / Engine.IO]
    end

    User --> Setup
    Setup --> Runtime
    Runtime --> Inventory
    Runtime --> Backend

    Backend --> REST
    Backend --> Socket
    REST --> Deltas
    Socket --> Deltas

    Inventory --> Store
    Deltas --> Store
    Store --> View
    View --> Entities
    View --> Services
```

## Operational constraints

- REST requests must be rate-limited and treated as a fallback when WebSocket
  updates are unavailable.
- The `import_energy_history` service must throttle to **2 queries per second**.
- Inventory-driven assumptions (node list, addresses, and types) are immutable
  for the life of the entry; if hardware changes, the user must reload the
  integration.
