# TermoWeb v2 refactor status

## Status

The v2 refactor is a **clean-slate** effort. There is no legacy path and no
strangler architecture. Documentation defines the target end state and the code
must converge toward it.

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
