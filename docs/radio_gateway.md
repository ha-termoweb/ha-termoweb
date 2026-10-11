# Radio gateway: build your own with an ESP32

This guide shows you how to build a small radio gateway for TermoWeb family
electric heaters (for example ATC Sun Ray). The gateway is an **ESP32 board**
with a **CC1101 868 MHz radio module**.

- The gateway talks to your heaters over their own radio link.
- Home Assistant talks to the gateway over your Wi-Fi network.
- You do **not** need the TermoWeb cloud, an account, or the vendor's Wi-Fi
  gateway (at least about €130). The parts cost about €20 to €40.

This is the **advanced** option. You connect wires and usually solder a few
points. If you want no soldering at all, use a
[nanoCUL USB stick](radio_nanocul.md) instead. Both work with the same
heaters. The ESP32 gateway is a good choice when your Home Assistant computer
is far from the heaters: the gateway only needs Wi-Fi and a USB power supply.

You need about one to two hours:

1. Buy the parts.
2. Prepare and solder.
3. Connect the wires.
4. Put our firmware on the ESP32. **Everyone must do this once.** No shop
   sells the board with our firmware on it.
5. Connect the gateway to your Wi-Fi and to Home Assistant.
6. Connect the TermoWeb integration.
7. Pair your heaters.

## Safety notice

This software switches mains-powered electric heaters in people's homes.

- A heater can report one state and do another. Check the heater itself
  after you change something, especially while you set things up.
- Keep every heater's own thermal cut-outs, panel controls and isolation in
  place. This gateway is not a safety device. Never rely on it to turn a
  heater off.
- Do not leave heaters unattended because of this software alone.
- The software is provided as is, without warranty of any kind.

## Radio notice

The gateway **transmits**. It is made for the European 869.4 to 869.65 MHz
short-range-device band only. Do not use it where that band is not allowed.

| Setting | Value |
|---|---|
| Frequency | 869.525 MHz |
| Modulation | 2-FSK, about 50.8 kHz deviation, 9.6 kbps |
| Output power | about +10 dBm (about 10 mW) |
| Band limits | 500 mW e.r.p., at most 10 percent duty cycle |

In normal use the gateway transmits for well under 1 percent of the time.
Across the EU these limits are band 54 of Commission Decision 2006/771/EC.
CEPT countries use the same band under ERC/REC 70-03. Check your own
national rules. **You are the operator of the transmitter** and are
responsible for using it within these limits.

## Step 1: buy the parts

### Bill of materials

| Part | What to look for |
|---|---|
| **ESP32-S3 board** (recommended) | **ESP32-S3-DevKitC-1**. The **N16R8** version (16 MB flash) and the **N8R8** or **N8** version (8 MB flash) all work. Buy a board with the **pin headers already soldered** if you can. |
| or: a plain **ESP32 board** (cheaper) | "ESP32 DevKitC", "ESP32 NodeMCU", "ESP32 WROOM-32" with 4 MB flash. It uses other pins: see step 3. |
| **CC1101 radio module, 868 MHz** | The common small green or blue module with **8 pins in two rows of 4** (2.54 mm pitch) and a spring antenna. It must say **868 MHz**. See "868 or 433?" below. |
| **Antenna, 868 MHz** | Usually comes with the module (a small spring, or an SMA antenna). |
| **8 female-to-female jumper wires** | "Dupont" wires, 10 or 20 cm. Only if both boards have pin headers. |
| **USB cable** | Fits your ESP32 board (most ESP32-S3 boards: USB-C). Must carry data, not only power. |
| **5 V USB power supply** | Any phone charger with 1 A or more. You probably have one. |
| Optional: **plastic case** | Plastic only. A metal case blocks the radio. |
| Optional: **perfboard** | For a soldered, permanent build instead of jumper wires. |

### 868 or 433? Check this carefully

Sellers offer the CC1101 module in **433 MHz** and **868 MHz**. They look
almost the same. A 433 MHz module hears the heaters badly or not at all.

- The listing title must say **868 MHz** (or 868MHz).
- On a variant button, choose **868 MHz**.
- The 868 MHz spring antenna is **shorter** than the 433 MHz one.
- The module must have a **26 MHz crystal**. Almost all modules do.

Ebyte modules: the **E07-868MS10** is a small **surface-mount** module with
tiny solder pads (1.27 mm). It works, but it is hard to solder by hand. For a
first build, take the common module with 2 rows of 4 pins. Ebyte's
**E07-M1101D** module with pins is sold in a **433 MHz** version: check the
frequency before you buy it.

### Unlabelled CC1101 boards

Some cheap modules (for example the GERUI "CC1101 868MHz" module, model
1101-2) are a bare green board with **half-round solder pads** on two edges.
There are **no pin names** and no instructions.

How to read such a board (example: GERUI 1101-2):

- **Antenna edge:** 3 pads, next to the "CC1101 868MHz Module" text and the
  tiny antenna parts. The order is **GND, ANT, GND**. The **middle** pad is
  the antenna. Both outer pads are ground.
- **Opposite edge:** a row of **8 pads**. These are the signal and power
  pads.
- At one end of the 8-pad row there is a **large capacitor** and a small black
  filter part. The two pads at this end are **pad 1 (GND)** and
  **pad 2 (VCC)**.
- The other end of the row is near the silver **26 MHz crystal**. That is
  **pad 8 (GDO2)**.

Most of these boards use this pad order, counted from the capacitor end:

| Pad | Signal | Connect to (ESP32-S3) | Connect to (plain ESP32) |
|---|---|---|---|
| 1 | GND | GND | GND |
| 2 | VCC, **3.3 V only** | 3V3 | 3V3 |
| 3 | GDO0 | GPIO9 | GPIO4 |
| 4 | CSN | GPIO10 | GPIO5 |
| 5 | SCK | GPIO12 | GPIO18 |
| 6 | MOSI | GPIO11 | GPIO23 |
| 7 | MISO (also GDO1) | GPIO14 | GPIO19 |
| 8 | GDO2 | GPIO13 | GPIO16 |
| ANT (middle pad on the antenna edge) | antenna | spring or wire antenna | spring or wire antenna |

ESPHome's own CC1101 guide uses the same numbers for GDO0 (pin 3) and GDO2
(pin 8).

**Check pads 1 and 2 before you connect power.** Use a multimeter in
**continuity mode** (the mode that beeps). The board must not be connected to
anything.

1. Put one probe on an **outer pad of the antenna edge** (a ground pad).
2. Touch the two pads next to the large capacitor, one after the other, with
   the other probe.
3. The pad that **beeps** is **GND** (pad 1). The other one, which does
   **not** beep, is **VCC** (pad 2).
4. Count the other pads from pad 1: pad 3 is next to VCC, pad 8 is at the
   crystal end.

If **neither** pad beeps, or **both** beep, stop. Do not guess. Ask the seller
for a pinout, or use a module with printed pin names.

> **Warning.** Never connect 5 V to this module. If you swap VCC and GND, the
> module is destroyed at once. Check pads 1 and 2 twice before you plug in
> the USB cable.

These boards have no pins. You must solder a wire (or a pin header) to each
of the 8 pads, and the antenna to the middle pad of the antenna edge. Do not
solder anything to the two ground pads beside the antenna pad.

### Where to buy

These shops sell the hardware only. You put our firmware on the ESP32 in
step 4.

Prices are approximate, with VAT, without shipping. Listings change: check the
listing before you buy.

**ESP32 board**

| Shop | Country | Product | Price |
|---|---|---|---|
| [Reichelt](https://www.reichelt.de/de/de/shop/produkt/entwicklungsboard_esp32-s3-wroom-1-n8r8-411139) | Germany | Espressif ESP32-S3-DevKitC-1-N8R8 | about €17.50 (checked 2026-10-11) |
| [BerryBase](https://www.berrybase.de/detail/01a0a08fef6771fab6d4b97a830c7c92) | Germany | Espressif ESP32-S3-DevKitC-1-N8R8 | about €21 (checked 2026-10-11) |
| [Botland](https://botland.de/esp32-module/26547-esp32-s3-devkitc-1-n8r8-wifi-bluetooth-entwicklungsboard-mit-esp32-s3-wroom-1-chip.html) | Poland (ships EU) | Espressif ESP32-S3-DevKitC-1-N8R8 | about €22 (checked 2026-10-11) |
| [DigiKey](https://www.digikey.de/en/products/detail/espressif-systems/ESP32-S3-DEVKITC-1-N8R8/15295894) | EU delivery | Espressif ESP32-S3-DevKitC-1-N8R8 | about €16 with VAT; out of stock (checked 2026-10-11) |
| [Amazon.nl](https://www.amazon.nl/dp/B0CXT1Z2WD) | Netherlands | DollaTek ESP32-S3-DevKitC-1, N16R8. Pin headers **not soldered** ("solderen"). | about €12 (checked 2026-10-11) |
| [Amazon.nl](https://www.amazon.nl/dp/B0DBHHDZ88) | Netherlands | ESP32-S3 N16R8 development board | about €25 (checked 2026-10-11) |
| [Amazon.de](https://www.amazon.de/dp/B0CXT1Z2WD) | Germany | Same DollaTek N16R8 listing | price not shown to us; check the listing |
| [BerryBase](https://www.berrybase.de/esp32-nodemcu-development-board) | Germany | ESP32 NodeMCU Development Board (plain ESP32, ESP-WROOM-32, 4 MB) | about €7 (checked 2026-10-11) |

**CC1101 868 MHz radio module**

| Shop | Country | Product | Price |
|---|---|---|---|
| [roboter-bausatz.de](https://www.roboter-bausatz.de/p/cc1101-868mhz-rf-funkmodul-transceiver-fuer-arduino-iot) | Germany | CC1101 868MHz RF Funkmodul Transceiver, with antenna | about €9.70 (checked 2026-10-11) |
| [nanoCUL.de](https://www.nanocul.de/selfsite.php?aktion=shop&info=CC1101%20Funkmodul%20868MHz) | Germany | CC1101 Funkmodul 868MHz | about €6 (checked 2026-10-11) |
| [Domoticx](https://domoticx.net/webshop/rf-zender-868mhz-module-10mw-18-36v-met-antenne-cc1101) | Netherlands | RF zender 868MHz module 10mW, with antenna, CC1101 | about €4 (checked 2026-10-11) |
| [Amazon.de](https://www.amazon.de/dp/B0CHJSHPR2) | Germany | TECNOULAB CC1101 868 MHz with antenna, pack of 2 (also on Amazon.fr, .it, .es, .nl) | about €18 for 2 (checked 2026-10-11) |
| [Amazon.nl](https://www.amazon.nl/dp/B07RH67FWV) | Netherlands | 2 pcs CC1101 868MHz wireless module with antenna | about €17 for 2 (checked 2026-10-11) |
| [Amazon.de](https://www.amazon.de/dp/B0FB3K3CQP) | Germany | GERUI 2pcs CC1101 868MHz Micro Power UHF transceiver, spring antenna. **Pads without labels and without pins**: see "Unlabelled CC1101 boards" above. Needs soldering. | about €7 for 2 (checked 2026-10-11) |
| [Amazon.nl](https://www.amazon.nl/dp/B0FB3K3CQP) | Netherlands | Same GERUI listing (also on Amazon.es, .fr, .it, currently unavailable there) | about €9 for 2 (checked 2026-10-11) |
| [eBay.de "CC1101 868MHz"](https://www.ebay.de/b/Cc1101-868MHz/bn_7005638801) | Germany | Category page with many offers. Many ship from China (slower). | about €2 to €10 |

**Extras**

| Shop | Country | Product | Price |
|---|---|---|---|
| [BerryBase](https://www.berrybase.de/detail/019234a38fbc73d195bb2fec42ba2c3f) | Germany | 40-pin jumper / Dupont cable, female-female | from about €1.60 (checked 2026-10-11) |
| [nanoCUL.de](https://www.nanocul.de/selfsite.php?aktion=shop&info=SMA%20Antenne%20868MHz) | Germany | SMA antenna 868 MHz (only for a module with an SMA socket) | about €6 (checked 2026-10-11) |

### What it costs in total

| Build | Parts | About |
|---|---|---|
| Budget | plain ESP32 (€7) + CC1101 868 (€4 to €10) + jumper wires (€2) | **€15 to €25** with shipping |
| Recommended | ESP32-S3-DevKitC-1 (€12 to €22) + CC1101 868 (€4 to €10) + jumper wires (€2) | **€25 to €40** with shipping |
| Add if needed | USB power supply, case | €5 to €10 each |

For comparison: the vendor's Wi-Fi gateway costs at least about €130.

## Step 2: prepare and solder

You need a soldering iron, solder, and about 15 minutes. If you have never
soldered, ask a friend or a local repair café or makerspace.

1. **Antenna.** Most CC1101 modules come with a loose spring antenna. Solder
   one end into the hole marked **ANT** on the module. Keep it straight, and
   do not cut it. A module with an SMA socket needs no soldering: screw on the
   868 MHz antenna.
2. **Pin headers.** If the ESP32 board or the radio module came with loose
   pin headers, solder them in. The short side of the pins goes through the
   board. Solder each pin on the other side.
3. **Check.** Look at every solder joint. No solder may connect two pins.

For a permanent build, you can solder the 8 wires directly between the two
boards, or put both boards on a piece of perfboard.

## Step 3: connect the wires

Unplug the ESP32 from USB before you connect wires.

> **Warning: the CC1101 runs on 3.3 V only.** Connect VCC to the **3V3** pin,
> never to **5V** (also called VIN or VBUS). 5 V destroys the module.

Connect each CC1101 pin to the ESP32 pin in the same row:

| CC1101 pin | Other names on the module | ESP32-S3 pin | Plain ESP32 pin | What it does |
|---|---|---|---|---|
| VCC | 3V3 | 3V3 | 3V3 | power, **3.3 V only** |
| GND | | GND | GND | ground |
| SCK | SCLK, CLK | GPIO12 | GPIO18 | SPI clock |
| MOSI | SI | GPIO11 | GPIO23 | SPI data to the radio |
| MISO | SO | GPIO14 | GPIO19 | SPI data from the radio |
| CSN | CS, SS | GPIO10 | GPIO5 | chip select |
| GDO0 | | GPIO9 | GPIO4 | "frame finished" signal |
| GDO2 | | GPIO13 | GPIO16 | "data waiting" signal |

On many boards, the pin labels show only the number: GPIO12 is printed as
**12** (or **IO12**).

**Plain ESP32 warning:** on a plain ESP32, GPIO 6, 7, 8, 9, 10 and 11 are
connected to the board's flash memory. Do not connect the radio to them. If
you do, the board does not start. Use the "Plain ESP32 pin" column. (The
ESP32-S3 pins are safe on an S3.)

Check every wire again before you plug in the USB cable.

**Where to put the gateway:** in a central place in your home, away from metal
and away from the Wi-Fi router. The heaters' signals are weak. A good position
helps more than anything else.

## Step 4: put the firmware on the ESP32

You do this once. Choose **one** of the three ways below. Way A is the
easiest.

For ways A and B, first download the firmware file:

1. Open the [releases page of this project](https://github.com/ha-termoweb/ha-termoweb/releases).
2. Open the newest release and click **Assets** to see the files.
3. Download the file for your board:
   - **ESP32-S3 board** (N16R8, N8R8 or N8): `termoweb-radio-esp32s3.factory.bin`
   - **Plain ESP32 board**: `termoweb-radio-esp32.factory.bin`

Connect the ESP32 with the USB cable to your computer. If the board has two
USB ports, use the one marked **UART** (or **COM**).

**Windows:** if the computer does not find the board, install the driver for
its USB chip (CP2102 or CH340).

### Way A (recommended): flash in your web browser

You need **Google Chrome** or **Microsoft Edge** on a computer. Other browsers
(Firefox, Safari) and phones cannot do this.

1. Open **ESPHome Web**: <https://web.esphome.io>. This is the free web
   installer of the ESPHome project. Nothing is installed.
2. Click **Connect**. The browser shows a list of ports. Choose your board
   (often "USB JTAG/serial", "CP2102" or "USB Serial"). Click **Connect** in
   that list.
3. Click **Install**. Choose the `.factory.bin` file you downloaded. Click
   **Install** again.
4. Wait until it says the installation is complete. This takes 1 to 2
   minutes. Do not unplug the board.
5. Then set up Wi-Fi: click the **three dots** (⋮) and choose **Configure
   Wi-Fi**. Choose your Wi-Fi network, type the password and click
   **Connect**. Go to step 5.

If **Connect** finds no port, or the install does not start: hold the
**BOOT** button on the board, press and release **RESET** (or **EN**), then
release **BOOT**. Try again.

### Way B: flash with a desktop program (Windows)

Espressif, the maker of the ESP32, has a free Windows program: the **Flash
Download Tool**.

1. Download it from
   <https://www.espressif.com/en/support/download/other-tools> (look for
   **Flash Download Tools**). Unpack the zip and start the `.exe` file.
2. Set **ChipType** to **ESP32-S3** (or **ESP32** for a plain board),
   **WorkMode** to **Develop**, and **LoadMode** to **UART**. Click **OK**.
3. In the first row, click **...** and choose the `.factory.bin` file. Type
   `0x0` in the address box after the **@**. Tick the box at the start of
   that row.
4. Choose your board's port in **COM** (see **Device Manager → Ports (COM &
   LPT)**). Set **BAUD** to **460800**.
5. Click **START**. Wait until it says **FINISH**.
6. Unplug the board and plug it in again. Then set up Wi-Fi (step 5).

If it stays at "waiting for power-on sync": hold the **BOOT** button on the
board, press and release **RESET** (or **EN**), then release **BOOT**.

### Way C: build and flash with ESPHome (for experienced users)

Use this way if you want to change the firmware, for example other pins.
You build the firmware yourself from the configuration file in this
repository.

You need the **ESPHome Device Builder** add-on in Home Assistant
(**Settings → Add-ons → Add-on store → ESPHome Device Builder**).

1. Download two files from the
   [`esphome/` folder of this repository](https://github.com/ha-termoweb/ha-termoweb/tree/main/esphome):
   `termoweb-radio.yaml` and `termoweb_radio.h`.
2. Open the ESPHome add-on. Click **Secrets** (top right). Add these three
   lines, with your own values, and click **Save**:

   ```yaml
   wifi_ssid: "your WiFi name"
   wifi_password: "your WiFi password"
   api_encryption_key: "paste a key here"
   ```

   To get a key, open <https://esphome.io/components/api/> and copy the random
   key shown on that page. Each refresh of the page shows a new one. Keep the
   key: Home Assistant may ask for it in step 5.
3. Copy `termoweb_radio.h` into the ESPHome configuration folder
   (`/config/esphome/` on your Home Assistant). You can use the **File
   editor** or **Samba share** add-on for this. The file must sit directly in
   that folder, next to the YAML files, not in a subfolder.
4. In the ESPHome add-on, click **+ New device**, then **Continue**. Give it
   the name `termoweb-radio`, and choose **ESP32-S3** (or **ESP32** for a
   plain board). Skip the install step it offers.
5. Click **Edit** on the new device. Delete everything in the editor. Paste
   the whole content of `termoweb-radio.yaml`. Click **Save**.
6. Change the board settings if needed, in the `substitutions:` section at
   the top:
   - **ESP32-S3 with N16R8:** change nothing.
   - **ESP32-S3 with 8 MB flash (N8R8, N8):** change `flash_size: 16MB` to
     `flash_size: 8MB`.
   - **Plain ESP32:** change the values as the comment in the file explains:
     `board: esp32dev`, `variant: esp32`, `flash_size: 4MB`, and the six pins
     (`sck_pin: "18"`, `mosi_pin: "23"`, `miso_pin: "19"`, `csn_pin: "5"`,
     `gdo0_pin: "4"`, `gdo2_pin: "16"`).

   Click **Save**.
7. Connect the ESP32 with the USB cable to the computer where you use Home
   Assistant in the browser.
8. Click **Install**, then **Plug into this computer**, and follow the
   instructions. This needs the **Chrome** or **Edge** browser. The first
   build takes several minutes. Later updates can use **Wirelessly**.

   If **Plug into this computer** does not work: click **Install → Manual
   download** and save the file. Then flash it with way A or way B above.

The firmware gets your Wi-Fi from the secrets, so you can skip the Wi-Fi part
of step 5.

Other command-line tools work too: `esphome run termoweb-radio.yaml` with the
[ESPHome command line](https://esphome.io/guides/cli/), or
`esptool -p <port> write-flash 0x0 <file>.factory.bin` with
[esptool](https://docs.espressif.com/projects/esptool/en/latest/esp32s3/).

## Step 5: connect the gateway to Wi-Fi and Home Assistant

**Wi-Fi.** After way A or B, the gateway does not know your Wi-Fi yet. Set it
up in one of these two ways:

- In Chrome or Edge, open <https://web.esphome.io>, click **Connect**, choose
  the board, then click the **three dots** (⋮) → **Configure Wi-Fi**.
- Or, about one minute after power-on, the gateway opens its own Wi-Fi
  network called **termoweb-radio**. Connect your phone to it. A page opens
  (if not, open `http://192.168.4.1` in the phone's browser). Choose your
  Wi-Fi network, type the password and click **Save**.

**Check that the radio works (optional).** In ESPHome Web, click **Logs**
(or, with way C, click **Logs** in the ESPHome add-on). You should see these
lines:

```
self-test passed
CC1101 found: partnum=00 version=14
listening on TCP port 2323
```

If you see `unexpected CC1101 id` or `MISO stayed high`, check the wiring
(step 3) and that VCC goes to 3V3.

**Home Assistant.** Home Assistant finds the new device by itself. Go to
**Settings → Devices & Services**, and accept the new **ESPHome** device.
With way C, enter the API key from step 4 if asked.

You do not need to note the gateway's IP address. The TermoWeb integration
finds the gateway by itself, and follows it when the address changes.

The device page also shows:

- **Home Assistant link**: on while the TermoWeb integration is connected.
- **Frames received**: heater radio frames with a valid checksum since the
  last restart. If it stays at 0 for a long time, the gateway does not hear
  your heaters: move it closer, and check that the module is the 868 MHz
  version.
- **Restart**: restarts the gateway.

## Step 6: connect the TermoWeb integration

You need the TermoWeb integration installed first. See the
[README](../README.md#install-simple-step-by-step).

1. In Home Assistant, go to **Settings → Devices & Services**. Under
   **Discovered** you see **TermoWeb** with **Radio gateway (<address>)**.
   It can take a minute after the gateway starts.
2. Click **Add** (or **Configure**). Home Assistant asks "Found a TermoWeb
   radio gateway at ... Set it up?". Press **Submit**. You do not type
   anything. You do not need a network id.
3. Choose **Add the gateway**. Setup finishes at once.
   - Your heaters already work with a TermoWeb gateway or app: go to
     "Find paired heaters" below.
   - Your heaters are new, or you reset them: go to "Add new heaters"
     below.

   (**Listen only** is for tests: it records radio traffic next to a
   TermoWeb gateway that still runs, and never transmits.)

The power of each heater (its **rated power**) is read from the heater by
itself. You do not type it.

### If the gateway is not under Discovered

Some networks block discovery. If the gateway is not under **Discovered**
after a few minutes, add it by hand:

1. Open the gateway's device page (from step 5). The **IP address** sensor
   shows its address, for example `192.168.1.50`.
2. Go to **Settings → Devices & Services → Add Integration** and search for
   **TermoWeb**.
3. Choose **Local radio gateway: ESP32 + CC1101 (advanced, needs soldering)**.
4. Enter the IP address and press **Submit**. Leave **Advanced settings**
   as they are. Then continue with step 6.3 above.
5. Give the gateway a fixed address in your router (often called "DHCP
   reservation" or "static lease"). A gateway added by hand is not followed
   when its address changes.

## Step 7: pair your heaters

### Find paired heaters

1. Setup finishes at once. You do not wait on this screen.
2. Turn one heater's temperature up, so that it starts heating. Heaters talk
   when they start heating.
3. The integration listens to your heaters in the background to learn their
   radio network. Then it checks every heater it can reach. This usually
   takes a few minutes.
4. Each heater it finds is added by itself, and you get a notification:
   "Found TermoWeb heater ...". The heater then appears under **Devices**.
5. A heater that was quiet is added later, as soon as it talks. You do not
   need to do anything.

### Add new heaters

New heaters pair by themselves. During the first 30 minutes you do not
start anything in Home Assistant.

1. Put the heater into pairing mode **within 30 minutes of adding the
   gateway**. The heater's manual tells you which buttons to press.
   Later, first press **Pair new heaters** on the gateway device
   (**Settings → Devices & Services → TermoWeb →** your gateway). Pairing
   is then open for 10 minutes.
2. Wait about 10 seconds. The heater is paired and gets its clock.
3. To add more heaters, put the next heater into pairing mode. Do one heater
   at a time. Each press of **Pair new heaters** gives you 10 more minutes.
4. After a few seconds you get a notification: "Found TermoWeb heater ...".
   The heater appears under **Devices**.

The gateway device shows **Pairing open**. It is on while heaters are
paired. Its attribute **Ends at** shows when pairing closes. When pairing is
closed, a heater in pairing mode is ignored. This keeps a neighbour's heater
out of your installation.

A heater that you had before keeps its device, entities and name when you
pair it again, and it gets its saved settings back.

If the gateway has not found any heater yet, restarting Home Assistant (or
reloading the integration) opens pairing for 30 minutes again.

A gateway that has not found any heater yet gives your heaters their own
radio network. It does not use the network of an old TermoWeb gateway. If a
heater is not paired, move the gateway closer to the heater, press **Pair
new heaters** and try again.

A heater that is already paired to your heaters' network is added by itself
when it talks. If you removed a heater, open the integration, choose
**Reconfigure**, and tick **Scan for heaters again**.

## Troubleshooting

| What you see | What to do |
|---|---|
| Logs: `unexpected CC1101 id` or `MISO stayed high` | Check the wiring (step 3). Check that VCC goes to 3V3. Check the solder joints. |
| The board does not start after you connect the radio (plain ESP32) | You used GPIO 6 to 11. Use the "Plain ESP32 pin" column. |
| The board restarts again and again after the first install | You used the file for the wrong board. Flash the file for your board again (step 4). With way C: set `flash_size` to match your board (step 4, way C, point 6) and install again. |
| **Frames received** stays at 0 | Move the gateway closer to a heater. Turn a heater's temperature up. Check that the module and antenna are the **868 MHz** version. Check that the antenna is soldered on. |
| The gateway is not under **Discovered** | Wait a few minutes. Or add it by hand (step 6). |
| A heater often shows **unavailable** | The signal is weak. Move the gateway to a more central place, away from metal and the Wi-Fi router. |
| The browser does not find the board | Use Chrome or Edge. Use a USB data cable, not a charge-only cable. Try the other USB port on the board. Some boards need a driver for their USB chip (CP2102 or CH340). Hold **BOOT** while you press **RESET**. |

### If no heater is found

If no heater shows up after 15 minutes:

- Turn one heater's temperature up so that it starts heating. A heater that
  wants heat talks every few minutes.
- Move the gateway closer to a heater (a few metres is best).
- Or pair the heater with this gateway: press **Pair new heaters** on the
  gateway device and put the heater into pairing mode (see "Add new
  heaters"). This also works for a heater that is idle, or that belongs to
  another gateway or another Home Assistant.

The integration keeps listening by itself. Every 10 minutes it also records
the radio signals for 2 minutes. It then finds your heaters even if their
signal is weak. If it hears signals it cannot read, you get the notification
"TermoWeb heaters not recognised", and it stops listening. This has two
possible causes:

- The signal is too weak. Move the gateway closer to a heater. Then open the
  integration, click the **three dots** (⋮) and choose **Reload** to listen
  again.
- Your heaters talk in a way that the integration does not know yet. You can
  help to add support for them:
  1. The notification shows the name of a report file, for example
     `/config/termoweb_radio_survey_setup_20261001T120000Z.json`. The file is
     in your Home Assistant configuration folder.
  2. Download the file. For example, use the **File editor** or **Samba**
     add-on.
  3. Open a new issue at
     <https://github.com/ha-termoweb/ha-termoweb/issues/new>. Write your
     heater brand and model, and attach the file.

The report does not contain your gateway's MAC address. The network ids of
the decoded radio messages are hidden. Signals that could not be decoded are
included as raw bits.

### Record a radio report later

After setup you can record a new report at any time. Use this when a new
heater does not show up, or when a developer asks for a report.

1. Go to **Developer tools → Actions**.
2. Choose **TermoWeb: Radio survey**.
3. Choose your radio gateway, and set **Duration** (120 seconds is a good
   start). Turn a heater's temperature up so it talks while you wait.
4. Press **Perform action**. Heater control pauses while the survey runs.
5. The response shows the result and the name of the report file. The
   integration's diagnostics download also shows the last result.

The survey needs gateway firmware 3.7-esp32 or newer (step 4), or nanoCUL
firmware 3.7-nanocul or newer. The ESP32 records more of each signal than
the nanoCUL.

## More tasks

### Move your heaters to this installation's own network

If the gateway found heaters that already worked with a TermoWeb gateway, they
still use the radio network of their old TermoWeb gateway. You can move them
to a network of their own. Some older heaters cannot be reset over the radio,
so they cannot be moved this way.

1. Open the integration and choose **Configure**. The screen **Move heaters
   to this installation's own network** opens.
2. Read the steps on the screen and press **Submit**.
3. Wait about 10 seconds. Home Assistant saves the settings of each heater
   and resets it.
4. Put one heater into pairing mode. Wait about 10 seconds, then the next
   heater. Do one heater at a time. You have 5 minutes.
5. Read the result. It tells you the new number of each heater, for example
   "Heater 6 is now heater 2". The settings are restored. The entity ids
   change, so check your automations and dashboards.

If a heater was not paired in time, the result says so. Pair it later: press
**Pair new heaters** on the gateway device and put the heater into pairing
mode. It gets its saved settings back when Home Assistant recognises it.

You can also start the move with the action **TermoWeb: Radio move heaters
to own network**.

### Reset a heater to factory settings

**Warning: this deletes all settings on the heater.** The heater forgets its
mode, preset temperatures, weekly program, clock and its radio pairing. It
stops talking to Home Assistant until you pair it again. Home Assistant saves
the settings first, so the pairing can give them back.

1. Go to **Developer tools → Actions**.
2. Choose **TermoWeb: Radio factory reset**.
3. Choose your radio gateway and enter the heater number.
4. Press **Perform action**.
5. Pair the heater again: press **Pair new heaters** on the gateway device
   and put the heater into pairing mode (see "Add new heaters" above). It
   keeps its device and gets its saved settings back.

Some older heaters cannot be reset over the radio. The action then stops
with an error. Reset such a heater by hand: the heater's manual tells you
how.

### Good to know

Only one program can use the gateway at a time. If a second program
connects, the first one is disconnected.

What works over the radio, and what does not yet, is listed in
[the radio backend reference](radio_backend.md).

## For developers: the line protocol

The gateway serves the `termoweb_rx` line protocol on TCP port 2323. The
nanoCUL stick speaks the same protocol over USB (`firmware/nanocul`, version
3.7-nanocul). The command reference, and a comparison of the two firmwares, is in
[`firmware/README.md`](../firmware/README.md#line-protocol).

The gateway announces itself over mDNS as `_termoweb-radio._tcp` on port 2323,
with TXT records `version` (radio firmware) and `mac` (Wi-Fi MAC). Home
Assistant uses this for discovery.

Technical details of the heaters' radio are in [the radio protocol reference](radio_protocol.md).
