# Radio gateway: build your own with an ESP32

This guide shows you how to build a small radio gateway for Sun Ray / Termoweb
electric heaters. The gateway is an ESP32 board with a CC1101 radio module.
It talks to your heaters over their own radio link. Home Assistant talks to
the gateway over your WiFi network. You do not need the Termoweb cloud or the
Termoweb Smart App Gateway.

You need about one hour. You do not need to solder if you buy modules with
pin headers and use jumper wires.

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

| Part | Notes |
|---|---|
| ESP32-S3 board, for example an ESP32-S3-DevKitC-1 | A plain ESP32 board (ESP32-DevKitC, "ESP32 WROOM") also works. See step 2. |
| CC1101 radio module, **868 MHz version** | Buy the 868 MHz version. Sellers also sell a 433 MHz version that looks the same. A 433 MHz module hears the heaters badly or not at all. Check the listing and the antenna: an 868 MHz antenna is shorter. |
| 8 female-to-female jumper wires | Only if your boards have pin headers. |
| USB cable | To power the ESP32 and to flash it the first time. |
| 5 V USB power supply | For daily use. |

The CC1101 module must have a **26 MHz crystal**. Almost all modules do.

## Step 2: connect the wires

Unplug the ESP32 from USB before you connect wires.

The CC1101 runs on **3.3 V only**. Connect VCC to the 3V3 pin, never to 5V.
5 V destroys the module.

| CC1101 pin | Other names on the module | ESP32-S3 pin | Plain ESP32 pin |
|---|---|---|---|
| VCC | 3V3 | 3V3 | 3V3 |
| GND | | GND | GND |
| SCK | SCLK, CLK | GPIO12 | GPIO18 |
| MOSI | SI | GPIO11 | GPIO23 |
| MISO | SO | GPIO14 | GPIO19 |
| CSN | CS, SS | GPIO10 | GPIO5 |
| GDO0 | | GPIO9 | GPIO4 |
| GDO2 | | GPIO13 | GPIO16 |

**Plain ESP32 warning:** on a plain ESP32, GPIO 6, 7, 8, 9, 10 and 11 are
connected to the board's flash memory. Do not connect the radio to them. If
you do, the board does not start. Use the "Plain ESP32 pin" column.

Put the gateway in a central place in your home, away from metal and away
from the WiFi router. The heaters' signals are weak; a good position helps
more than anything else.

## Step 3: install the firmware with the ESPHome add-on

You need the **ESPHome Device Builder** add-on in Home Assistant
(Settings > Add-ons > Add-on store > ESPHome Device Builder).

1. Download two files from the `esphome/` folder of this repository:
   `termoweb-radio.yaml` and `termoweb_radio.h`.
2. Open the ESPHome add-on. Click **Secrets** (top right). Add these three
   lines, with your own values, and click **Save**:

   ```yaml
   wifi_ssid: "your WiFi name"
   wifi_password: "your WiFi password"
   api_encryption_key: "paste a key here"
   ```

   To get a key, open https://esphome.io/components/api/ and copy the random
   key shown on that page. Each refresh of the page shows a new one.
3. Copy `termoweb_radio.h` into the ESPHome configuration folder
   (`/config/esphome/` on your Home Assistant). You can use the **File
   editor** or **Samba share** add-on for this. The file must sit directly in
   that folder, next to the YAML files, not in a subfolder.
4. In the ESPHome add-on, click **+ New device**, then **Continue**, give it
   the name `termoweb-radio`, and choose **ESP32-S3** (or **ESP32** for a plain
   board). Skip the install step it offers.
5. Click **Edit** on the new device. Delete everything in the editor. Paste
   the whole content of `termoweb-radio.yaml`. Click **Save**.
6. **Plain ESP32 only:** in the `substitutions:` section at the top, change the
   values as the comment in the file explains (board, variant, flash size and
   the six pins).
7. Connect the ESP32 to the computer that shows Home Assistant in its browser,
   with the USB cable. Click **Install**, then **Plug into this computer**,
   and follow the instructions. The first build takes several minutes.
   Later updates can use **Wirelessly**.
8. When the install has finished, click **Logs**. You should see these lines:

   ```
   self-test passed
   CC1101 found: partnum=00 version=14
   listening on TCP port 2323
   ```

   If you see `unexpected CC1101 id` or `MISO stayed high`, check the wiring
   (step 2) and that VCC goes to 3V3.

## Step 4: find the gateway's IP address

Home Assistant finds the new device by itself. Go to **Settings > Devices &
services**, accept the new **ESPHome** device, and enter the API key from
step 3 if asked. Then open the device page. The **IP address** sensor shows
the address, for example `192.168.1.57`.

Give the gateway a fixed address in your router (often called "DHCP
reservation" or "static lease"). Otherwise the address can change after a
power cut, and the integration loses the gateway.

The device page also shows:

- **Home Assistant link**: on while the Termoweb integration is connected.
- **Frames received**: heater radio frames with a valid checksum since the
  last restart. If it stays at 0 for a long time, the gateway does not hear
  your heaters: move it closer, and check that the module is the 868 MHz
  version.
- **Restart**: restarts the gateway.

## Step 5: connect the integration

1. In Home Assistant, go to **Settings → Devices & Services → Add Integration**
   and search for **TermoWeb**.
2. Choose **Local radio gateway (ESP32 + CC1101)**.
3. Enter the gateway's IP address from step 4. Keep the port at **2323**.
4. Leave **Radio dialect** on **auto** and **Network id** empty.
5. Press **Submit** and wait. The integration listens to your heaters to learn
   their radio network. This takes up to 6 minutes. Then it checks every
   heater it can reach. Heaters talk when they start heating, so turn one
   heater's temperature up before you press **Submit**.
6. When it finishes, your heaters appear under **Devices**.

If you see "No heater radio traffic was heard":

- Turn one heater's temperature up so that it starts heating. A heater that
  wants heat talks every few minutes.
- Move the gateway closer to a heater (a few metres is best) and try again.

If setup still hears nothing, it records the radio signals for 2 more
minutes. This takes extra time. It finds your heaters if their signal is
weak. If it hears signals it cannot read, you see "Radio bursts were heard,
but none could be decoded". This has two possible causes:

- The signal is too weak. Move the gateway closer to a heater and try again.
- Your heaters use a radio "dialect" that the integration does not know yet.
  You can help to add it:
  1. The error message shows the name of a report file, for example
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

The survey needs gateway firmware 3.7-esp32 or newer (step 3). The nanoCUL
stick cannot record surveys.

### Pair a heater

Home Assistant can pair a heater with the gateway itself. You need this for
a new heater, or for a heater that you reset (below).

1. Go to **Developer tools → Actions**.
2. Choose **TermoWeb: Radio pair**.
3. Choose your radio gateway.
   - For a **new heater**, leave **Heater number** empty.
   - For a heater that you **reset**, enter its old number. For "Heater 6",
     enter 6.
4. Press **Perform action**. You have 5 minutes for the next step.
5. On the heater, start pairing mode. The heater's manual tells you which
   buttons to press. Put **only one** heater into pairing mode at a time.
6. When the action finishes, the response shows the heater number.
   - A new heater appears under **Devices** after the integration reloads.
   - A heater with its old number gets its settings back: mode, preset
     temperatures and weekly program.

Heater control pauses while pairing runs. If the action says that no heater
was paired, move the gateway closer to the heater and try again.

### Reset a heater to factory settings

**Warning: this deletes all settings on the heater.** The heater forgets its
mode, preset temperatures, weekly program, clock and its radio pairing. It
stops talking to Home Assistant until you pair it again. Home Assistant saves
the settings first, so the pairing can give them back.

1. Go to **Developer tools → Actions**.
2. Choose **TermoWeb: Radio factory reset**.
3. Choose your radio gateway and enter the heater number.
4. Press **Perform action**.
5. Pair the heater again with **TermoWeb: Radio pair** and the same heater
   number (see above).

Only heaters that use radio dialect B can be reset. The integration's
diagnostics download shows the dialect.

If you added or removed a heater later, open the integration, choose
**Reconfigure**, and tick **Scan for heaters again**.

Only one program can use the gateway at a time. If a second program
connects, the first one is disconnected.

What works over the radio, and what does not yet, is listed in
[the radio backend reference](radio_backend.md).

## For developers: the line protocol

The gateway serves a line protocol on TCP port 2323. It is the protocol of
`termoweb_rx` 3.5, the nanoCUL868 firmware in ha-termoweb-local, with a few
additions. The full reference is the comment at the top of
`esphome/termoweb_radio.h`.

The firmware is a **transparent radio**. RX lines carry the on-air bytes
exactly as received, and `T` transmits the given bytes exactly as given. The
host's codec builds, scrambles, checks and decodes frames. The firmware only
uses the dialect for three time-critical jobs: finding the end of a frame
from its length byte, checking its CRC, and sending the 8-byte link ack
within a few milliseconds.

Every line the gateway sends ends with `\r\n`. Send one command per line,
ending with `\n`.

| Command | Reply | Meaning |
|---|---|---|
| (connect) or `V` | `# termoweb_rx 3.7-esp32 freq=869.525 rate=9.6k sync=2DE5 mode=dynamic tx=paC0 mac=AA:BB:CC:DD:EE:FF` | Banner. Sent first on every new connection. |
| `Q` | `# Q termoweb_rx 3.7-esp32 freq=869.525 pa=C0 sync=2DE5 mode=dynamic autoack=off id=01 dialect=A net=1B30 mac=AA:BB:CC:DD:EE:FF` | Status. `mac` is the WiFi MAC: a stable device id. |
| `Y0` / `Y1` | `# dialect=A sync=2DE5` / `# dialect=B sync=2DD4` | Select the on-air dialect (below). Error: `# Y? expected 0 or 1`. |
| `N<hex4>`, e.g. `N1234` | `# net=1234` | Network id used in auto-acks. Error: `# N? expected 4 hex digits`. |
| `I<hex2>`, e.g. `I01` | `# id=01` | Our station id, used by auto-ack. |
| `A0` / `A1` | `# autoack=off` / `# autoack=on` | Auto-ack off/on. |
| `T<hex>` | `TX <micros> <n> <hex>` | Transmit up to 255 bytes exactly as given. Errors: `TXERR empty or bad hex`, `TXERR underflow`, `TXERR marcstate=<hex>`. |
| `F<kHz>`, e.g. `F869525` | `# freq=869.525 word=21717A` | Retune (779000 to 928000). Error: `# F? expected kHz`. |
| `X` | `# raw=on` / `# raw=off` | Append the 2 raw radio status bytes to RX lines. |
| `D` | `# marcstate=.. pktstatus=.. rxbytes=.. syncs=<n> rxreset=<n>` | Radio chip state dump. |
| `R<secs>`, e.g. `R120` | `# survey on secs=120 floor=-104.0 thr=-96.0`, then per burst `RAWB <n> <rssi>`, `RAW <n> +<us> -<us> ...`, `RAWE <n> <runs>`; finally `# survey off bursts=<n>` | Raw survey for unknown dialects (1–600 s, `R0` ends it early). The radio switches to raw bit mode, so it hears any 2-FSK signal at this frequency and bit rate range, whatever its sync word and framing. Each burst louder than the noise floor + 8 dB is sent as run lengths of the demodulated bits (sign = level, value = µs). Packet reception, auto-ack and transmit (`TXERR survey active`) are off until the survey ends. |

Unprompted lines:

- `RX <micros> <rssi_dbm> <lqi> <crc_ok> <hex>`: a received frame, bytes as on
  air. `crc_ok` is 1 when the frame's CRC is valid in the current dialect.
- `ACK <micros> <hex>`: an auto-ack the gateway sent, bytes as on air.
- `# rxreset n=<count>`, `# uart overrun`: receiver recovery, and command
  input lost because it arrived too fast.

Dialects (`Y`):

| | Dialect A (`Y0`, default) | Dialect B (`Y1`) |
|---|---|---|
| Sync word | 2DE5 | 2DD4 |
| Byte 0 | total length minus 3 | total length |
| Scrambling | PN9 keystream (FF 87 B8 59 ...) | none |
| CRC | CRC-16/CCITT, init 0x1D0F, final XOR 0xFFFF, over the descrambled bytes | CRC-16/MODBUS (0xA001 reflected, init 0xFFFF) |
| Auto-ack on air | `05 <net> <id> <src> 80 <crc>`, scrambled | `08 <net> <id> <src> 80 <crc>` |

In both, the CRC covers bytes 0 to total-3 and is sent high byte first.
Auto-ack fires only for a frame with a valid CRC, addressed to our station id,
with flags 00.

Every new connection resets the gateway to its defaults: dialect A, network
1B30, station id 01, auto-ack off, 869.525 MHz. Send `Y`, `N`, `I` and `A1`
again after each connect.
