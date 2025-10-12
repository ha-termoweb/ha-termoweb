# Codex Tasks: Replace Node Caches with Inventory

The following tasks align the integration with the requirement that the immutable
`Inventory` object is the sole source of truth for gateway metadata. Each task is
scoped to a single Python module and removes code that recreates node lists or
stores redundant node caches. Update or remove tests tied to the deleted logic so
coverage remains at 100% (`timeout 30s pytest --cov=custom_components.termoweb --cov-report=term-missing`).

## Task: `custom_components/termoweb/coordinator.py`
* **Problem spots**
  * `StateCoordinator.update_nodes` rebuilds an `Inventory` from raw REST `nodes`
    payloads instead of requiring the shared container (`build_node_inventory`
    usage at lines 555-569). 【F:custom_components/termoweb/coordinator.py†L555-L569】
  * `EnergyStateCoordinator._coerce_inventory` manufactures temporary
    inventories from ad-hoc mappings or iterables, again bypassing the shared
    topology (`build_node_inventory` calls at lines 841-853). 【F:custom_components/termoweb/coordinator.py†L841-L853】
* **Required changes**
  * Delete the rebuild paths and enforce passing a valid `Inventory` instance to
    both coordinators.
  * Simplify callers/tests to supply the authoritative inventory instead of raw
    node payloads.
* **Tests**: `timeout 30s pytest --cov=custom_components.termoweb --cov-report=term-missing`

## Task: `custom_components/termoweb/backend/termoweb_ws.py`
* **Problem spots**
  * `_apply_nodes_payload` deep-copies websocket payloads into `_nodes_raw`,
    effectively maintaining a parallel nodes cache (lines 665-726). 【F:custom_components/termoweb/backend/termoweb_ws.py†L665-L726】
  * `_nodes_raw` is initialised and mutated solely to persist these duplicates,
    including in legacy update handlers (lines 1172, 1824-1840). 【F:custom_components/termoweb/backend/termoweb_ws.py†L1172-L1175】【F:custom_components/termoweb/backend/termoweb_ws.py†L1824-L1840】
* **Required changes**
  * Remove `_nodes_raw` and the associated merge logic; rely on the shared
    `Inventory` metadata when dispatching updates.
  * Adjust websocket tests that assert cache contents to reflect the simplified
    flow.
* **Tests**: `timeout 30s pytest --cov=custom_components.termoweb --cov-report=term-missing`

## Task: `custom_components/termoweb/backend/ducaheat_ws.py`
* **Problem spots**
  * The Socket.IO client mirrors nodes into `_nodes_raw` snapshots on every
    `dev_data` and `update` message (lines 549-603). 【F:custom_components/termoweb/backend/ducaheat_ws.py†L549-L603】
  * Additional helpers `_coerce_dev_data_nodes` and `_dispatch_nodes` reshape raw
    payloads into bespoke structures prior to dispatch (lines 672-735,
    993-1036). 【F:custom_components/termoweb/backend/ducaheat_ws.py†L672-L735】【F:custom_components/termoweb/backend/ducaheat_ws.py†L993-L1036】
* **Required changes**
  * Delete the redundant caches and coercion helpers; dispatch websocket updates
    using only the immutable `Inventory` data.
  * Update tests that expected `_nodes_raw` mutations or list coercion side
    effects.
* **Tests**: `timeout 30s pytest --cov=custom_components.termoweb --cov-report=term-missing`

## Task: `custom_components/termoweb/backend/ws_client.py`
* **Problem spots**
  * `_ensure_type_bucket` materialises `nodes_by_type`, `settings`, and
    `nodes_by_type` caches in the Home Assistant record, re-implementing
    inventory-derived structures (lines 594-664). 【F:custom_components/termoweb/backend/ws_client.py†L594-L664】
* **Required changes**
  * Remove the bucket-building logic and fetch node metadata directly from
    `Inventory` within websocket dispatch paths.
  * Adjust unit tests to stop asserting on the temporary caches.
* **Tests**: `timeout 30s pytest --cov=custom_components.termoweb --cov-report=term-missing`

## Task: `custom_components/termoweb/boost.py`
* **Problem spots**
  * `iter_inventory_heater_metadata` builds a nested `lookup` of nodes by type
    and address even though `Inventory` already exposes `nodes_by_type` and
    name resolution helpers (lines 199-235). 【F:custom_components/termoweb/boost.py†L199-L235】
* **Required changes**
  * Replace the manual lookup with direct calls to `Inventory` helpers and drop
    the redundant container.
  * Update tests consuming the lookup artefacts.
* **Tests**: `timeout 30s pytest --cov=custom_components.termoweb --cov-report=term-missing`

## Task: `custom_components/termoweb/button.py`
* **Problem spots**
  * `_iter_accumulator_contexts` constructs a `node_lookup` map duplicating the
    inventory’s accumulator nodes (lines 108-117). 【F:custom_components/termoweb/button.py†L108-L117】
* **Required changes**
  * Fetch accumulator nodes directly from `Inventory` helpers and remove the
    bespoke dictionary.
  * Update button tests that rely on the lookup side effects.
* **Tests**: `timeout 30s pytest --cov=custom_components.termoweb --cov-report=term-missing`

## Task: `custom_components/termoweb/climate.py`
* **Problem spots**
  * Climate setup builds `node_lookup` structures keyed by `(type, addr)` despite
    the shared inventory already providing canonical heater metadata (lines
    73-112). 【F:custom_components/termoweb/climate.py†L73-L112】
* **Required changes**
  * Source heater entities directly from `Inventory` iterators instead of
    maintaining parallel lookups.
  * Refresh tests that expect the intermediate dictionaries.
* **Tests**: `timeout 30s pytest --cov=custom_components.termoweb --cov-report=term-missing`

## Task: `custom_components/termoweb/binary_sensor.py`
* **Problem spots**
  * `_iter_boostable_inventory_nodes` duplicates node-to-address mapping work
    that `Inventory` already handles (lines 302-324). 【F:custom_components/termoweb/binary_sensor.py†L302-L324】
* **Required changes**
  * Use the inventory’s heater metadata directly to enumerate boostable nodes.
  * Update binary sensor tests tied to the manual mapping.
* **Tests**: `timeout 30s pytest --cov=custom_components.termoweb --cov-report=term-missing`

## Task: `custom_components/termoweb/inventory.py`
* **Problem spots**
  * `resolve_record_inventory` provides numerous fallbacks that rebuild
    inventories from snapshots, raw nodes, or stored lists even though the
    authoritative container should already exist (lines 600-709).
    【F:custom_components/termoweb/inventory.py†L600-L709】
* **Required changes**
  * Remove the fallback construction paths and require callers to keep a valid
    `Inventory` reference.
  * Delete or refactor tests covering the deprecated fallback behaviour.
* **Tests**: `timeout 30s pytest --cov=custom_components.termoweb --cov-report=term-missing`

## Task: `custom_components/termoweb/energy.py`
* **Problem spots**
  * `async_import_energy_history` accepts raw `nodes` overrides and attempts to
    resolve inventory data when it should rely on the cached container (lines
    355-383). 【F:custom_components/termoweb/energy.py†L355-L383】
* **Required changes**
  * Require an `Inventory` argument (or reuse the stored instance) and remove
    fallback resolution via raw nodes.
  * Update energy import tests that supply alternate node inputs.
* **Tests**: `timeout 30s pytest --cov=custom_components.termoweb --cov-report=term-missing`
