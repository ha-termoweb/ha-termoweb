# Design Contract (Clean-Slate v2)

This contract defines the non-negotiable design rules for the TermoWeb
integration. It is the source of truth for all refactor work.

## What must always be true

1. **Latest Home Assistant only.** The integration supports the current HA
   baseline (`homeassistant>=2025.1.0`) with no compatibility shims or fallback
   imports.
2. **No legacy paths.** There is exactly one architecture; transitional or
   strangler code is not allowed.
3. **Immutable inventory.** The gateway and node inventory is captured once at
   setup and never rebuilt or cached again outside the runtime.
4. **Single state pipeline.** All updates flow through:
   inventory snapshot → domain deltas → `DomainStateStore` → `DomainStateView`.
5. **Vendor isolation.** Brand differences exist only in backend, planner, and
   codec modules. Entities are vendor-agnostic.
6. **Pydantic on the wire only.** Domain state uses dataclasses/standard Python
   types; Pydantic is reserved for payload parsing/serialization.

## Guardrails for implementation

- Do not add or keep compatibility imports for older HA releases.
- Do not allow entities to read raw REST/WS payloads.
- Do not rebuild inventory outside of setup.
- Do not mix vendor-specific logic into entity or platform modules.
- Prefer deletion and mechanical moves over rewrites.

## Documentation expectations

Docs are authoritative. Any behavior change must update documentation first,
then align implementation to match.
