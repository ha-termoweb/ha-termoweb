# Changelog

## Unreleased

### Requirements
- Home Assistant **2026.10.0** or newer is required (HACS enforces it).

### Added
- Re-authentication: when your password changes, Home Assistant asks for the new password (Settings → Repairs or the integration card) and reloads the integration.
- **Reconfigure** now reloads the integration.
- The same account cannot be added twice. Ducaheat and Tevolve count as the same account. Email case does not matter.
- Climate `turn_on` and `turn_off` are supported.
- Energy history import options: `max_history_retrieval` (1 to 3650 days) and `reset_progress`.

### Changed
- Entity unique IDs move to one scheme on upgrade. Entity IDs, history and customisations are kept. If an old unavailable duplicate remains (for example the 2.3.0 "Installation info" sensor), delete the unavailable one.
- Energy history import was rewritten: correct sums, resumable, a re-import overwrites in place, limited to 2 requests per second. Run it once.
- All cloud (REST) requests are limited to 2 per second. The integration prefers the WebSocket for updates.
- Entities show **unavailable** when the cloud cannot be reached.
- Gateway ids are masked in logs at INFO level and above.

### Fixed
- Devices that use °F show correct temperatures.
- Boost works for 60 to 600 minutes. The boost buttons work. The boost duration and boost temperature numbers write to the device.

### Removed
- The `ws_debug_probe` action and the **debug** option. Use Home Assistant's `logger:` settings for `custom_components.termoweb` (see DEBUG.md).
- The `python-socketio` dependency.
