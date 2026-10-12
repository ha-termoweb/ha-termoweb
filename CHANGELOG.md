# Changelog

## 2.4.0 (2026-10-12)

The big change: you can now control your heaters **locally, without the cloud and without the vendor gateway**, through a small 868 MHz radio. Use a nanoCUL USB stick (easiest, no soldering) or a home-built ESP32 radio gateway. Cloud accounts (TermoWeb, Ducaheat, Tevolve) work as before.

### Before you upgrade
- Home Assistant **2026.10.0** or newer is required (HACS enforces it).
- Entity unique IDs move to one scheme on upgrade. Entity IDs, history and customisations are kept. If an old, unavailable duplicate remains (for example the 2.3.0 "Installation info" sensor), delete the unavailable one.
- The **Start accumulator boost** and **Cancel accumulator boost** actions are now called **Start boost** and **Cancel boost**. Update automations and scripts that use them.
- The `ws_debug_probe` action and the **debug** option are removed. Use Home Assistant's `logger:` settings for `custom_components.termoweb` (see DEBUG.md).

### Added: local radio control
- **Two kinds of radio.** A **nanoCUL USB stick** with this project's firmware (the ready-made file is attached to each release), or an **ESP32 radio gateway** built with ESPHome. See [Radio stick: use a nanoCUL USB stick](docs/radio_nanocul.md) and [Radio gateway: build your own with an ESP32](docs/radio_gateway.md).
- **Easy setup.** Home Assistant finds an ESP32 gateway on your network by itself (**Settings → Devices & Services → Discovered**). For a nanoCUL stick, the form pre-selects the most likely USB port. You do not type a network id or a radio type: they are found by themselves.
- **Heaters are added by themselves.** Setup finishes at once. Heaters that already work with another gateway are found in the background, usually within a few minutes, and you get a notification for each one.
- **Pairing.** Put a heater into pairing mode within 30 minutes of adding the gateway, or press **Pair new heaters** on the gateway device first (pairing then stays open for 10 minutes). The **Pairing open** sensor shows when pairing is open. Outside that time, heaters in pairing mode are ignored, so a neighbour's heater stays out.
- **Heaters keep their identity.** Heaters are named and identified by serial number (for example **Heater X123456**), not by their radio number. A heater paired again keeps its device, entities and history, and gets its saved settings back.
- **Control.** Mode, target temperature, the three preset temperatures and the weekly schedule. Power and estimated energy work without setup; heaters that keep their own energy meter report it, read every 10 minutes.
- **Heater features, where the heater supports them.** **Boost** (the **Boost** preset, or the **Start boost** and **Cancel boost** actions); **Runback** and **EASY mode** switches; and advanced settings under the device's **Configuration**: **Temperature offset**, **Control mode**, **Panel temperature unit**, **Open window detection** and **True radiant**. The heater cannot report the advanced settings, so they show **Unknown** until you set them in Home Assistant.
- **Move heaters to your own network.** **Configure** on a radio entry moves heaters from an old vendor gateway's radio network to this installation's own network.
- **Tools.** **Radio survey** and **Radio capture** actions record what the radio hears, and **Download diagnostics** includes the last 200 radio messages. Serial numbers and network ids are hidden. A **Radio address** diagnostic sensor (disabled by default) shows each heater's current radio number.
- **Listen only.** A listen-only entry shows what the radio hears without controlling anything.
- The integration is now listed as **local push**.

### Added: cloud accounts
- Re-authentication: when your password changes, Home Assistant asks for the new one and reloads the integration.
- **Reconfigure** reloads the integration.
- The same account cannot be added twice. Ducaheat and Tevolve count as the same account. Email case does not matter.
- Climate **turn on** and **turn off** are supported.
- Energy history import options: `max_history_retrieval` (1 to 3650 days) and `reset_progress`.
- Translations for 25 languages cover all of the integration's text.

### Added: the schedule card installs itself
- The **TermoWeb Schedule** card (weekly schedule and preset temperatures) now comes with the integration. You no longer copy a file or add a dashboard resource: edit a dashboard, click **+ Add card** and search for `termoweb`. It works for cloud and radio heaters. See [Schedule card: edit the weekly schedule](docs/schedule_card.md).
- If you installed the card by hand before, the integration replaces the old dashboard resource by itself. You can delete the `www/termoweb` folder.
- If you manage dashboard resources in YAML, add `/termoweb/termoweb_schedule_card.js` as a module resource yourself.

### Changed
- Heaters without boost show no preset choice. To end a temporary temperature change while in Auto, select **Auto** again.
- Energy history import was rewritten: correct sums, resumable, a re-import overwrites in place, limited to 2 requests per second. Run it once.
- All cloud requests are limited to 2 per second. The integration prefers the WebSocket for updates.
- Entities show **unavailable** when the cloud cannot be reached.
- Gateway ids are masked in logs at INFO level and above.
- Docs: the README and the radio guides are rewritten for local-first use, with shopping lists and step-by-step instructions.

### Fixed
- Devices that use °F show correct temperatures.
- Accumulator boost works for 60 to 600 minutes. The boost buttons work. The boost duration and boost temperature numbers write to the device.
- Setting a temperature together with a mode now respects that mode.
- The schedule card shows the selected heater in its list when it first opens.

### Removed
- The `ws_debug_probe` action and the **debug** option.
- The `python-socketio` dependency.
- The manual card install guide and the `install_termoweb_card.sh` script. The card installs itself.

### Known limitations
- Heater control pauses while a radio survey, capture or pairing runs.
- The nanoCUL firmware is new; please report any problems.
