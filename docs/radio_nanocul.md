# Radio stick: use a nanoCUL USB stick

A nanoCUL is a small USB stick: an Arduino Nano with a CC1101 radio for
868 MHz. With the `termoweb_rx` firmware it talks to your heaters directly,
like the [ESP32 radio gateway](radio_gateway.md), but it plugs into a USB
port of the computer that runs Home Assistant.

## Before you start: which heaters work

The nanoCUL firmware from this repository (version 3.7-nanocul or newer)
speaks both radio "dialects" that TermoWeb heaters use, A and B, like the
ESP32 radio gateway. The setup finds out which one your heaters use.

An older stick firmware (termoweb_rx 3.5 or 3.6, from the old
ha-termoweb-local project) only speaks dialect A. Put the new firmware on the
stick (step 1) to use dialect-B heaters.

## Step 1: put the termoweb_rx firmware on the stick

The firmware is in the `firmware/nanocul` folder of this repository. You can
download a ready-made file or build it yourself. Both ways need `avrdude`,
the tool that writes the firmware to the stick (on Debian, Ubuntu or a
Raspberry Pi: `sudo apt install avrdude`).

**Download (easiest)**

1. Open the [releases page](https://github.com/ha-termoweb/ha-termoweb/releases)
   and download `termoweb_rx-3.7-nanocul.hex` (or the newest
   `termoweb_rx-*.hex`) from the latest release.
2. Plug the nanoCUL into the computer.
3. Run, with your stick's port instead of `/dev/ttyUSB0`:

   ```
   avrdude -c arduino -p atmega328p -P /dev/ttyUSB0 -b 115200 -D -U flash:w:termoweb_rx-3.7-nanocul.hex:i
   ```

   If avrdude says "not in sync", your stick has the older bootloader: use
   `-b 57600` instead of `-b 115200`.
4. Wait until avrdude says the flash was verified.

**Build it yourself**

You also need `avr-gcc` (`sudo apt install gcc-avr avr-libc binutils-avr`).

1. Download this repository and open a terminal in `firmware/nanocul`.
2. Run `make flash PORT=/dev/ttyUSB0`. Use your stick's port if it is
   different. Add `BAUD=57600` if avrdude says "not in sync"
   (`make signature PORT=/dev/ttyUSB0` tells you which one your stick needs).
3. Wait until avrdude says the flash was verified.

The firmware transmits at +10 dBm, the level the heaters need to hear the
stick. `firmware/README.md` explains the firmware for developers.

## Step 2: plug the stick into Home Assistant

1. Plug the nanoCUL into a USB port of the Home Assistant computer.
2. Put the stick where it has a clear path to your heaters. A short USB
   extension cable helps to move it away from the computer and metal.

## Step 3: connect the integration

1. In Home Assistant, go to **Settings → Devices & Services → Add Integration**
   and search for **TermoWeb**.
2. Choose **Local radio: nanoCUL USB stick (CC1101 868 MHz)**.
3. Your stick is usually already selected in the **USB port** list. If
   another port is selected, or none, choose your stick: its name often
   contains "CUL", "FTDI", "CH340" or "USB Serial". If it is not in the
   list, choose **Enter the port path or URL myself** and type the path,
   for example `/dev/serial/by-id/usb-...`.
4. Leave **Advanced settings** as they are, and press **Submit**.
5. Choose **Add the gateway**. Setup finishes at once.
   - Heaters that already work with a TermoWeb gateway or app: turn one
     heater's temperature up, so that it starts heating. The integration
     listens in the background and adds each heater by itself, usually
     within a few minutes. You get a notification for each heater it finds.
   - New or reset heaters: within 30 minutes, put one heater at a time
     into pairing mode (see the heater's manual). Wait about 10 seconds
     between heaters.
6. Your heaters appear under **Devices**. A heater that was quiet is added
   later, as soon as it talks.

To add a new heater, put it into pairing mode within 30 minutes of adding the
stick, or press **Pair new heaters** on the stick's device first. Pairing is
then open for 10 minutes. The heater is added by itself, and you get a
notification. See [Add new heaters](radio_gateway.md#add-new-heaters).

If you see "Cannot open the nanoCUL stick": check the cable, the port and the
firmware (step 1).

If you see "This nanoCUL firmware only knows radio dialect A": your heaters
use dialect B and the stick has an old firmware. Put the new firmware on the
stick (step 1).

To change the port later, open the integration and choose **Reconfigure**.

A stick shared over the network (for example with `ser2net`) also works: type
its address as `socket://<address>:<port>` in step 3.

## Listen only: record radio traffic for a test

If your heaters still use a TermoWeb gateway, the stick can record what the
gateway and the heaters send, without transmitting anything. We use these
recordings to learn unknown commands.

1. In step 3, after you press **Submit**, choose **Listen only (record radio
   traffic, never transmit)**.
2. The entry is called **Radio monitor** and has no heaters. It shows
   **Gateway online** and **Frames heard**.
3. Record with the **TermoWeb: Radio capture** action. The guide
   [Help find the identify command](radio_identify_capture.md) explains every
   step.

The stick never transmits in this mode, not even link-layer acks.

Only one entry can use the stick at a time. When you set up the stick to
control your heaters, the listen-only entry loses the stick. Delete the
listen-only entry first.
