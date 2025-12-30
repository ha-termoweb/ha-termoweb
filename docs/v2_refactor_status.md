# TermoWeb v2 refactor status

## Invariants and staged plan
- Inventory is immutable for each config entry: one gateway `dev_id` and a fixed set of nodes identified by `(type, addr)` with static capabilities.
- Inventory is fetched once at setup; do not re-fetch or build auxiliary node caches outside the domain inventory.
- All interactions go through the cloud APIs (REST + WebSocket); no direct hardware access.
- Vendor differences live in codecs and, for Ducaheat, a planner; domain models are vendor-neutral dataclasses.
- Pydantic v2 models are only for wire parsing/encoding. Domain state must not store Pydantic models long-term.
- The refactor progresses via a strangler pattern: new domain/codecs coexist with legacy code until fully migrated.

## Progress log
- **v2.0.0-pre2**: Added accumulator domain scaffolding (state + commands) building on heater model; no runtime behavior changes. Testing: targeted unit tests for domain ids/inventory/state.
- **v2.0.0-pre1**: Added domain and codec scaffolding; no runtime behavior changes. Testing: targeted unit tests for domain ids/inventory.
