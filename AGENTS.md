# AGENTS.md

## Purpose
This repository provides a Home Assistant integration for the TermoWeb cloud platform, which manages heaters, thermostats, and power monitors for residential and commercial users across multiple branded instances (TermoWeb, Ducaheat, Tevolve). The integration lets Home Assistant users control their devices through TermoWeb instead of the mobile apps.

## Key Concepts
* **Platform instances:** TermoWeb and Ducaheat run on separate backend URLs with slightly different APIs. The Tevolve-branded app also uses the Ducaheat backend.
* **Device:** A single gateway (identified by a hexadecimal `dev_id`) that bridges a user's site to the TermoWeb backend.
* **Node:** An endpoint attached to the gateway. Supported types include heaters (`htr`), accumulators (`acm`), power monitors (`pmo`), and thermostats (`thm`). Each node exposes an integer `addr` and a `type` code.
* **Physical deployment assumptions:** Gateways and their nodes are fixed hardware that gets installed during construction. Treat this inventory as immutable for the lifetime of the integration: runtime code should not anticipate gateways or nodes being added, removed, or rebranded. Only the initial setup needs to interrogate the backend for device metadata. If hardware is physically replaced, the user can reload the integration to refresh the details.
* The Inventory of nodes, node types, node addresses and dev_id of the gateway are instantiated ONCE at startup and are IMMUTABLE and ALWAYS available. Use Inventory and methods to find that information, do not make additional lists or caches of nodes.

## Documentation Map
* `termoweb_openapi.yaml`, `termoweb_api.md` — REST and WebSocket reference for the TermoWeb backend.
* `ducaheat_openapi.yaml`, `ducaheat_api.md` — API documentation for the Ducaheat backend.
* `docs/architecture.md` — Integration architecture and Python class hierarchy.

## Usage and Fair Access
* Treat the hosted TermoWeb backend as a shared resource; avoid abusive traffic patterns.
* Prefer WebSocket updates once startup completes. Use REST only as a fallback while WebSocket connectivity is unavailable, and apply rate limiting to every REST call.
* Throttle the `import_energy_history` service to a maximum of **2 queries per second**. Importing a year of hourly data for multiple nodes can produce tens of thousands of records, so exceeding this rate risks destabilizing the backend. Each user should normally run this import only once.

## Audience Expectations
End users are non-technical Home Assistant operators. Documentation must be task-oriented, step-by-step, and written in clear, plain English to support readers for whom English may be a second language.

## Development Standards
* Follow the Python version declared in `pyproject.toml` and add type hints to all new code.
* USe uv as the package and environment manager
* Provide a concise one-line docstring for every function.
* Apply the **minimal viable change** for each task; avoid touching unrelated code.
* Adhere to DRY principles and practice defensive programming—anticipate invalid input, communication failures, and other error conditions, and handle them gracefully.
* Log major function entry/exit points at `INFO`, protocol interactions at `DEBUG`, and errors at `ERROR`.
* Format and lint all changes with `ruff` before committing.

## Testing Requirements
* Execute `timeout 30s pytest --cov=custom_components.termoweb --cov-report=term-missing` (or the equivalent platform-specific command) and ensure the suite finishes within 30 seconds with 100% coverage before committing.
* Capture partial logs whenever the timed run aborts; treat timeouts as failures requiring investigation. During debugging, you may run targeted, no-coverage subsets, but rerun the full timed command before completion.
* If tests approach the 30-second limit, suspect an asynchronous wait issue and stop the run rather than letting it hang.
* Write meaningful tests that exercise edge cases, error handling, and invalid inputs, with particular focus on component interfaces.

## Documentation Responsibilities
Document every new feature or behavior change and keep existing documentation in sync.
Add docstrings to all functions
Save all functions and docstrings in docs/function_map.txt

## Pull Request Expectations
* Keep each PR focused on a single feature or test with the minimal supporting changes.
* Summarize the modifications and describe how they were tested.
* Include documentation updates whenever behavior changes.
