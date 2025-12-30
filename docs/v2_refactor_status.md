# TermoWeb v2 refactor status

## Invariants and staged plan
- Inventory is immutable for each config entry: one gateway `dev_id` and a fixed set of nodes identified by `(type, addr)` with static capabilities.
- Inventory is fetched once at setup; do not re-fetch or build auxiliary node caches outside the domain inventory.
- All interactions go through the cloud APIs (REST + WebSocket); no direct hardware access.
- Vendor differences live in codecs and, for Ducaheat, a planner; domain models are vendor-neutral dataclasses.
- Pydantic v2 models are only for wire parsing/encoding. Domain state must not store Pydantic models long-term.
- The refactor progresses via a strangler pattern: new domain/codecs coexist with legacy code until fully migrated.

## Progress log
- **v2.0.0-pre4**: Routed TermoWeb write paths through domain commands and codec encoders, including new Pydantic request models and payload helpers for settings, extra options, and boost toggles. Added targeted unit tests for write payload formatting and validation. Testing: `ruff check custom_components/termoweb/codecs/termoweb_codec.py custom_components/termoweb/codecs/termoweb_models.py custom_components/termoweb/domain/commands.py`; `pytest -q tests/test_termoweb_codec_writes.py tests/test_api.py::test_set_node_settings_includes_prog_and_ptemp tests/test_api.py::test_set_node_settings_invalid_units tests/test_api.py::test_set_node_settings_invalid_program tests/test_api.py::test_set_node_settings_invalid_temperatures tests/test_api.py::test_set_acm_extra_options_forwards_payload tests/test_api.py::test_set_acm_boost_state_formats_payload tests/test_api.py::test_set_acm_boost_state_rejects_invalid_units tests/test_api.py::test_set_acm_boost_state_rejects_invalid_stemp`.
- **v2.0.0-pre3**: Added Pydantic v2 models and codecs for TermoWeb node settings and samples; wired REST client read paths through codecs without changing outward behaviors. Testing: targeted codec + REST client sample/settings tests.
- **v2.0.0-pre2**: Added TermoWeb REST codec and Pydantic v2 models for device and node inventory responses; wired REST client through codecs without changing return shapes. Testing: unit tests for codec payload normalization and API paths.
- **v2.0.0-pre1**: Added domain and codec scaffolding; no runtime behavior changes. Testing: targeted unit tests for domain ids/inventory.
