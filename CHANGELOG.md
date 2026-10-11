# Changelog

## Unreleased

### Added
- nanoCUL stick: new firmware **3.7-nanocul** in this repository (`firmware/nanocul`). The stick now works with the same heaters as the ESP32 gateway, and has the same commands: radio survey, frequency setting and network id. Download the ready-made file from the release page, or build it yourself; see [Radio stick: use a nanoCUL USB stick](docs/radio_nanocul.md), step 2.
- The radio survey (**TermoWeb: Radio survey**) also works with a nanoCUL stick that has the new firmware.
- Radio: heaters of radio type A keep their own energy meter. The **Energy** sensor now shows that meter, read every 10 minutes, instead of an estimate. The first reading after this update is the heater's total so far, so the Energy Dashboard can show one large step that day. Heaters of radio type B have no meter and keep the estimate.
- Radio: heaters of radio type A can **boost**. Choose the **Boost** preset on the heater, or use the **Start boost** and **Cancel boost** actions. The heater boosts for its own boost time, so leave the duration out. The heater shows when the boost ends.
- Radio: heaters of radio type A get two switches, **Runback** and **EASY mode**. When you turn Runback off, the heater gets back the mode and temperature it had before.
- Radio: heater advanced settings under the device's **Configuration**: **Temperature offset**, **Control mode**, **Panel temperature unit**, **Open window detection** and **True radiant**. The heater cannot report them, so they show **Unknown** until you set them in Home Assistant. When you set one, the others are sent too (unknown ones as the factory defaults). Heaters that do not support these settings show them as unavailable.

### Changed
- The **Start accumulator boost** and **Cancel accumulator boost** actions are now called **Start boost** and **Cancel boost**, because they also work for radio heaters.
- Radio setup: the radio type is always detected by itself. The **Radio dialect** choice is gone from the ESP32 gateway and nanoCUL forms (the nanoCUL forms now ask only for the port). A nanoCUL with old firmware that cannot detect it stops the setup with "The firmware on this nanoCUL stick is too old": put the new firmware on the stick, then try again. Listen only still works with old firmware.
- Radio, dialect A: Home Assistant no longer has a built-in dialect-A network id. A new dialect-A gateway learns its heaters' network from their radio traffic, the same way as dialect B, or uses its own network when you pair heaters. Gateways you already set up keep the network they have. A nanoCUL stick with the old firmware (3.5) cannot change its network: update it to the new firmware.
- ESP32 gateway and nanoCUL firmware: after a restart the network id is empty until Home Assistant sets it (Home Assistant always sets it when it connects).
- Docs: the README and the radio guides are rewritten for local-first use: why a local radio is cheaper and more private, a comparison of the nanoCUL stick and the ESP32 gateway, shopping lists, and step-by-step firmware flashing (in the browser first, then a desktop program, then the command line).

### Fixed
- nanoCUL stick (new firmware): the stick now notices when its radio module (CC1101) is missing or not wired. It says so when it starts, refuses to send, and the setup stops with "The stick answers, but its radio module is not connected. Check the wiring." Before, it acted as if it had sent.
- nanoCUL stick (new firmware): the stick acks only frames with a valid checksum, and letters inside a command's hex value (for example `NAFDA`) are no longer run as separate commands.

## 2.4.0b6 (pre-release, 2026-10-11)

Beta for testing. HACS shows it only when "Show beta versions" is enabled for this repository.

### Added
- Radio: **Download diagnostics** now includes the last 200 radio messages the gateway heard and sent, with counts per kind (own network, other networks, pairing announcements, acks, sent). Heater serial numbers and network ids are hidden. This helps us find out why a heater does not pair.

### Removed
- Radio: the separate ways to pair heaters are gone, because heaters now pair by themselves. The setup menu offers **Add the gateway** and **Listen only**; the **Pair new heaters** menu choice and the **TermoWeb: Radio pair** action were removed. To pair a heater, put it into pairing mode within 30 minutes of adding the gateway, or press **Pair new heaters** on the gateway device first. A heater you reset gets its saved settings back when it is recognised.

## 2.4.0b5 (pre-release, 2026-10-11)

Beta for testing. HACS shows it only when "Show beta versions" is enabled for this repository.

### Added
- Radio: heaters pair by themselves. Put the heater into pairing mode within 30 minutes of adding the gateway, or press **Pair new heaters** on the gateway device first (this opens pairing for 10 minutes). The heater is paired, gets its clock (and, for a heater you had before, its settings) back, and appears under **Devices** with a "Found TermoWeb heater" notification. You can pair several heaters, one after the other.
- Radio: the gateway device has a **Pairing open** sensor. It is on while new heaters are paired, and shows when pairing closes.
- Radio: a new gateway that has not found its heaters yet now sets up at once (gateway device, **Pair new heaters**, **Pairing open**). It runs on your installation's own radio network, so a heater can be paired even when it is idle or belongs to another gateway or Home Assistant. Heaters that already work with another gateway are still found in the background.

### Changed
- Radio: outside the pairing time, a heater in pairing mode is ignored. This keeps a neighbour's heater out of your installation.
- Radio: the options no longer have **Pair a new heater**; use the **Pair new heaters** button instead. **Configure** now opens **Move heaters to this installation's own network** directly.

### Fixed
- Radio: while a new gateway is still looking for your heaters, a heater in pairing mode is no longer mistaken for your heaters' radio network.

## 2.4.0b4 (pre-release, 2026-10-11)

Beta for testing. HACS shows it only when "Show beta versions" is enabled for this repository.

### Changed
- Radio setup (ESP32 gateway and nanoCUL) no longer waits up to 6 minutes for your heaters. Choose **Add the gateway and find my heaters**: setup finishes at once. The integration then listens in the background and adds each heater by itself, usually within a few minutes. You get a notification for each heater it finds.
- Radio: a heater that is paired to your heaters' network but not set up yet is added by itself as soon as it talks. A heater paired again at a new radio number keeps its device and entities.

## 2.4.0b3 (pre-release, 2026-10-11)

Beta for testing. HACS shows it only when "Show beta versions" is enabled for this repository.

### Added
- ESP32 radio gateway: Home Assistant finds the gateway by itself. After you flash the updated `esphome/termoweb-radio.yaml`, the gateway shows up under **Settings → Devices & Services → Discovered**. Click **Add**, confirm, and choose how to find your heaters. You type nothing. When the gateway gets a new IP address, the existing entry follows it by itself.
- nanoCUL stick: the setup form pre-selects the most likely USB port (a port named after a CUL, or the only free USB serial port).

### Changed
- Radio setup forms no longer ask for a network id. Setup learns it from the heaters' radio traffic, and pairing creates one for your installation. Existing entries keep their stored network id.
- The manual ESP32 gateway form asks only for the IP address. Port (2323) and radio dialect moved to a collapsed **Advanced settings** section, as did the nanoCUL dialect.
- Radio setup no longer refuses a gateway or stick that another entry uses. Home Assistant takes the connection over; the other entry loses it.
- Radio: the site device is now called "Home" (a name you set yourself is kept).
- Radio: devices show the real maker. Heaters and the site say "TermoWeb", the ESP32 gateway says "ESPHome", and the nanoCUL stick says "nanoCUL". The gateway is named "Radio gateway" (ESP32) or "nanoCUL stick". Cloud accounts are unchanged.

## 2.4.0b2 (pre-release, 2026-10-11)

Beta for testing. HACS shows it only when "Show beta versions" is enabled for this repository.

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
