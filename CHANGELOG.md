# Changelog

## Unreleased

### Changed
- The integration is now listed as **local push** instead of cloud: with a nanoCUL stick or the ESP32 radio gateway it works without internet access. Cloud accounts work as before.
- Setup menu order: cloud account, then the **nanoCUL USB stick** (easiest, no soldering), then the ESP32 radio gateway (advanced, needs soldering).
- Radio gateway: each heater's power is now read from the heater itself. Dialect-B heaters report their rated power in their identity record; dialect-A heaters report their measured power. The power limit and the estimated energy and power sensors now work without any setup.
- Radio gateway: heaters are named and identified by serial number (for example **Heater X123456**) instead of their radio number, which depends on the pairing order. A heater paired again at another number keeps its device, entities and history. On the first start each heater is asked for its serial; its device and entities move to the serial by themselves. Entity IDs, history and customisations are kept; entity IDs such as `climate.heater_2` are not renamed. Serials are hidden in diagnostics and logs. The serial is shown in the device info, and a new **Radio address** diagnostic sensor (disabled by default) shows the heater's current radio number.

### Removed
- Radio gateway: the manual heater power field in the integration options (**Configure** → **Settings**). Stored values are removed on upgrade.

## 2.4.0b1 (pre-release, 2026-10-11)

Beta for testing. HACS shows it only when "Show beta versions" is enabled for this repository.

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
