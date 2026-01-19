# TermoWeb v2 refactor status

## Status

The v2 refactor is a **clean-slate** effort for internal architecture. There is
no legacy path and no strangler architecture inside the integration.
Documentation defines the target end state and the code must converge toward
it. Vendor payload schemas are stable external contracts and are not subject to
legacy removal.

## Authoritative references

- `docs/architecture.md` — target architecture and module map
- `docs/design_contract.md` — non-negotiable rules
- `docs/legacy_removal.md` — deletion checklist for legacy artifacts

## What "done" means

The refactor is complete when:

- The codebase matches the design contract in full.
- Only the canonical state pipeline exists.
- Inventory is immutable and created once at setup.
- Entity platforms are vendor-agnostic.
- No compatibility shims or legacy scaffolding remain.
- Wire models preserve vendor payload fields exactly.
