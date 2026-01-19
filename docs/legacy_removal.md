# Legacy Removal Checklist

Use this checklist to verify that **no legacy or transitional code** remains
inside the integration. Vendor REST/WS payload schemas are stable external
contracts and are not subject to "legacy removal."

## Disallowed artifacts (must not exist)

- Compatibility imports or `try/except ImportError` blocks for older Home
  Assistant versions.
- Any "strangler" or "legacy" adapters that shadow new behavior.
- Wire model field renames or deletions that diverge from vendor payloads.
- Entity reads that pull directly from REST/WS payload dictionaries.
- Inventory rebuilds outside config entry setup.
- Vendor-specific client checks inside entity or platform modules.
- Alternate state caches outside `DomainStateStore`.
- Deprecated modules kept "just in case" or for backward compatibility.

## Required confirmations before release

- [ ] Inventory is captured once and stored only in the runtime.
- [ ] All entity reads use `DomainStateView`.
- [ ] All WS/REST updates are translated to domain deltas.
- [ ] Vendor-specific logic is isolated to backend/planner/codec modules.
- [ ] No legacy or compatibility shims remain.
- [ ] Documentation matches the implementation.
