# Radio stick: use a nanoCUL USB stick

This guide shows you how to control your heaters with a **nanoCUL USB stick**.
You do not need the TermoWeb cloud, an account, or the vendor's Wi-Fi gateway.
You do not need to solder.

You need about 30 minutes:

1. Buy the stick.
2. Put our firmware on the stick. **Everyone must do this once.** No shop
   sells the stick with our firmware on it.
3. Plug the stick into the Home Assistant computer.
4. Add the stick in Home Assistant.
5. Pair your heaters.

## What is a nanoCUL?

A nanoCUL is a small USB stick. Inside are two parts:

- an **Arduino Nano** board (chip: ATmega328P), and
- a **CC1101 radio** for **868 MHz**.

You put our `termoweb_rx` firmware on the stick. Then it talks to your heaters
directly over their own radio, like the
[ESP32 radio gateway](radio_gateway.md). The stick plugs into a USB port of the
computer that runs Home Assistant.

What you get:

- **Private.** Nothing leaves your home. No cloud, no account.
- **Cheap.** A stick costs about €25 to €60. The vendor's Wi-Fi gateway costs
  at least about €130.
- **Works without internet.** Your heaters work in Home Assistant even when
  your internet is down.

The stick and the ESP32 radio gateway work with the same heaters. You do not
need to know how your heaters talk: the setup finds out by itself.

## Safety and radio notice

- This software switches mains-powered electric heaters. A heater can report
  one state and do another. Check the heater itself after you change
  something, especially during setup.
- Keep every heater's own thermal cut-outs and controls. The stick is not a
  safety device. Never rely on it to turn a heater off.
- The stick **transmits** on 869.525 MHz, in the European 869.4 to 869.65 MHz
  band (about 10 mW). Use it only where this band is allowed. **You are the
  operator of the transmitter.** See the
  [radio notice in the ESP32 guide](radio_gateway.md#radio-notice) for the
  details.

## Step 1: buy the stick

### What to buy

Check these four points in the listing before you buy:

| Check | Why |
|---|---|
| **868 MHz** (not 433 MHz) | Many sellers offer the same stick in 433 MHz. A 433 MHz stick cannot hear your heaters. On Amazon, the 433/868 choice is often a **variant button**: choose **868 MHz**. |
| **Arduino Nano / ATmega328P** inside | Our firmware runs only on this chip. The listing often says "Nano V3.0 ATmega328P". |
| **CC1101** radio | Not RFM12B, RFM69 or another radio. |
| **An antenna** | A small bent antenna works at short range. An SMA antenna (straight or with a magnetic foot) reaches further. |

The USB chip does not matter. FTDI and CH340 both work. A stick with a
"level shifter" protects the radio chip a little better, but both kinds work.

**Do not buy** these. They look similar, but use a different chip and our
firmware does not run on them:

- Busware **CUL**, **CUL V3**, **CUNO** (chip ATmega32U4)
- Busware's own small "nanoCUL" (chip ATmega32U2)
- **JeeLink** sticks (different radio)

### Where to buy

These shops sell the hardware only. The sticks come with other firmware
(usually "culfw"). You replace it with our firmware in step 2.

Prices are approximate, with VAT, without shipping. Listings change: check the
listing before you buy.

| Shop | Country | Product | Price |
|---|---|---|---|
| [nanoCUL.de](https://www.nanocul.de/selfsite.php?aktion=shop&info=nanoCUL868) | Germany | nanoCUL 868MHz (Nano V3 ATmega328P, CH340) | about €23 (checked 2026-10-11) |
| [nanoCUL.de](https://www.nanocul.de/selfsite.php?aktion=shop&info=nanoCUL868LS) | Germany | nanoCUL 868MHz mit Levelshifter (Nano V3 ATmega328P, FTDI, level shifter) | about €28 (checked 2026-10-11) |
| [nanoCUL.de](https://www.nanocul.de/selfsite.php?aktion=shop&info=SMA%20Antenne%20868MHz) | Germany | SMA antenna 868 MHz (optional, for more range) | about €6 (checked 2026-10-11) |
| [Amazon.nl](https://www.amazon.nl/dp/B09PKDW9P4) | Netherlands | nanoCUL USB stick FTDI CC1101 868 MHz, bent antenna (seller: smart-home-komponente). Choose the **868 MHz** variant. | about €57 (checked 2026-10-11) |
| [Amazon.de](https://www.amazon.de/dp/B09PKDW9P4) | Germany | Same listing as Amazon.nl. Choose the **868 MHz** variant. | price not shown to us; check the listing |
| [eBay.de](https://www.ebay.de/itm/372221622516) | Germany | nanoCUL USB Stick FTDI CC1101 868MHz, bent antenna (smart-home-komponente) | about €48 (search result, checked 2026-10-11) |
| [eBay.de](https://www.ebay.de/itm/372267253563) | Germany | nanoCUL USB Stick FTDI CC1101 868MHz, SMA magnetic-foot antenna | check the listing |
| [eBay.de "CUL 868MHz"](https://www.ebay.de/b/Cul-868MHz/44996/bn_7005328986) | Germany | Category page with more offers. Check 868 MHz and ATmega328P. | varies |

## Step 2: put the termoweb_rx firmware on the stick

You do this once. It takes about 10 minutes. You can use any computer
(Windows, macOS or Linux). It does not need to be the Home Assistant
computer.

First, download the firmware file:

1. Open the [releases page of this project](https://github.com/ha-termoweb/ha-termoweb/releases).
2. Find the newest release that has the file `termoweb_rx-3.7-nanocul.hex`
   (or a newer `termoweb_rx-*-nanocul.hex`). Click **Assets** to see the files.
3. Click the file to download it.

Then plug the stick into a USB port of the computer, and choose **one** of
the three ways below. Way A is the easiest.

**Windows:** if the computer does not find the stick, install the driver for
its USB chip. Sticks with a CH340 chip need the CH340 driver. FTDI sticks
usually work without a new driver.

### Way A (recommended): flash in your web browser

You need **Google Chrome** or **Microsoft Edge** on a computer. Other browsers
(Firefox, Safari) and phones cannot do this.

1. Open the **Web AVR Programmer**:
   <https://geo-tp.github.io/ESP32-Bit-Pirate/web-tools/avr-programmer/>.
   This is a free tool from another open-source project. It runs AVRDUDE
   inside your browser. Nothing is installed.
2. In **Programmer**, choose **Arduino bootloader**.
3. In **Baudrate**, keep **115200**.
4. Click **Connect**. The browser shows a list of ports. Choose your stick
   (often "USB Serial", "CH340" or "FT232R"). Click **Connect** in that list.
5. Open the **Flash** tab. Click **Click to choose a Flash image** and choose
   the `termoweb_rx-...-nanocul.hex` file.
6. Click **Write Flash**. Wait until the output says the flash was written
   and verified.

If it says the programmer is "not responding" or "not in sync": your stick
has the older bootloader. Click **Disconnect**, set **Baudrate** to **57600**,
and do steps 4 to 6 again.

### Way B: flash with a desktop program (Windows)

**AVRDUDESS** is a free Windows program with buttons for AVRDUDE.

1. Download `AVRDUDESS-<version>-setup.exe` (or the portable zip) from
   <https://github.com/ZakKemble/AVRDUDESS/releases/latest> and install it.
2. Open AVRDUDESS.
3. In **Programmer (-c)**, choose the entry that starts with **Arduino for
   bootloader** (avrdude's name: `arduino`).
4. In **Port (-P)**, choose your stick's port, for example `COM3`. (Open
   **Device Manager → Ports (COM & LPT)** if you do not know it.)
5. In **Baud rate (-b)**, type **115200**.
6. In **MCU (-p)**, choose **ATmega328P**. (Or click **Detect**.)
7. In **Flash**, choose the `termoweb_rx-...-nanocul.hex` file and select
   **Write**.
8. Click **Program!**. Wait until the output says the flash was verified.

If it says "not responding" or "not in sync", set **Baud rate (-b)** to
**57600** and click **Program!** again.

### Way C: flash on the command line (for experienced users)

**avrdude** is the command-line tool behind ways A and B.

1. Install avrdude:
   - **Windows:** download the latest zip (for example
     `avrdude-v8.3-windows-x64.zip`) from
     <https://github.com/avrdudes/avrdude/releases> and unpack it. Open a
     command window in that folder.
   - **macOS:** with Homebrew, run `brew install avrdude`.
   - **Linux / Raspberry Pi:** run `sudo apt install avrdude`.
2. Find the stick's port:
   - **Windows:** **Device Manager → Ports (COM & LPT)**, for example `COM3`.
   - **macOS:** run `ls /dev/cu.usb*`.
   - **Linux:** run `ls /dev/ttyUSB*`. Usually `/dev/ttyUSB0`.
3. Run this command. Change `COM3` to your port, and the file name to your
   file:

   ```
   avrdude -c arduino -p m328p -P COM3 -b 115200 -D -U flash:w:termoweb_rx-3.7-nanocul.hex:i
   ```

4. Wait until avrdude says the flash was **verified**.

If avrdude says `programmer is not responding` or `not in sync`: your stick has
the older bootloader. Run the same command with `-b 57600` instead of
`-b 115200`.

To only test the connection, without writing anything, run
`avrdude -c arduino -p m328p -P COM3 -b 115200`. It should say
`device signature = 0x1e950f`.

**Build the firmware yourself.** The source is in the
[`firmware/nanocul`](../firmware/nanocul) folder of this repository. On Linux
you need `gcc-avr`, `avr-libc`, `binutils-avr` and `avrdude`
(`sudo apt install gcc-avr avr-libc binutils-avr avrdude`). In
`firmware/nanocul`, run:

```
make
make flash PORT=/dev/ttyUSB0
```

Use your stick's port. Add `BAUD=57600` if avrdude says "not in sync"
(`make signature PORT=/dev/ttyUSB0` tells you which one your stick needs).
The firmware transmits at +10 dBm, the level the heaters need to hear the
stick. [`firmware/README.md`](../firmware/README.md) explains the firmware for
developers.

## Step 3: plug the stick into Home Assistant

1. Plug the nanoCUL into a USB port of the Home Assistant computer.
2. Put the stick where it has a clear path to your heaters. A short USB
   extension cable helps: it moves the stick away from the computer and from
   metal.

## Step 4: add the stick in Home Assistant

You need the TermoWeb integration installed first. See the
[README](../README.md#install-simple-step-by-step).

1. Go to **Settings → Devices & Services → Add Integration** and search for
   **TermoWeb**.
2. Choose **Local radio: nanoCUL USB stick (easiest, no soldering)**.
3. Your stick is usually **already selected** in the **USB port** list. Press
   **Submit**.
   - If another port is selected, or none: choose your stick. Its name often
     contains "CUL", "FTDI", "CH340" or "USB Serial".
   - If it is not in the list: choose **Enter the port path or URL myself**
     and type the path, for example `/dev/serial/by-id/usb-...`.
   - The setup finds out by itself how your heaters talk.
4. Choose **Add the gateway**. Setup finishes at once. Then go to step 5.

## Step 5: pair your heaters

### Heaters that already work with a TermoWeb gateway or app

1. Turn one heater's temperature up, so that it starts heating. Heaters talk
   when they start heating.
2. The integration listens in the background. It adds each heater by itself,
   usually within a few minutes.
3. You get a notification for each heater it finds. The heater then appears
   under **Devices**.
4. A heater that was quiet is added later, as soon as it talks. You do not
   need to do anything.

### New or reset heaters

New heaters pair by themselves. You do not start anything in Home Assistant
during the first 30 minutes.

1. Put one heater into pairing mode **within 30 minutes of adding the stick**.
   The heater's manual tells you which buttons to press.
2. Wait about 10 seconds. Then do the next heater. Do one heater at a time.
3. After a few seconds you get a notification: "Found TermoWeb heater ...".
   The heater appears under **Devices**.

Later, to add another heater: open **Settings → Devices & Services →
TermoWeb →** your stick, and press **Pair new heaters**. Pairing is then open
for 10 minutes. See also
[Add new heaters](radio_gateway.md#add-new-heaters).

The power of each heater (its **rated power**) is read from the heater by
itself. You do not type it.

## Change the port later

Open the integration and choose **Reconfigure**.

## Use a stick on another computer

A stick shared over the network (for example with `ser2net`) also works. In
step 4, choose **Enter the port path or URL myself** and type
`socket://<address>:<port>`.

## Listen only: record radio traffic for a test

If your heaters still use a TermoWeb gateway, the stick can record what the
gateway and the heaters send, without transmitting anything. We use these
recordings to learn unknown commands.

1. In step 4, after you press **Submit**, choose **Listen only (record radio
   traffic, never transmit)**.
2. The entry is called **Radio monitor** and has no heaters. It shows
   **Gateway online** and **Frames heard**.
3. Record with the **TermoWeb: Radio capture** action (**Developer tools →
   Actions**). It saves a file in your Home Assistant configuration folder.

The stick never transmits in this mode, not even link-layer acks.

Only one entry can use the stick at a time. When you set up the stick to
control your heaters, the listen-only entry loses the stick. Delete the
listen-only entry first.

## Troubleshooting

| What you see | What to do |
|---|---|
| "Cannot open the nanoCUL stick" | Check the cable and the port. Check that the firmware is `termoweb_rx` (step 2), not culfw. Only one program can use the stick: stop any other program that uses it. |
| "The stick answers, but its radio module is not connected" | The Arduino works, but its radio board (CC1101) does not answer. Check that the radio board is plugged in or soldered on, and check its wires (the pins are in [firmware/README.md](../firmware/README.md)). Then unplug the stick, plug it in again and try again. |
| "The firmware on this nanoCUL stick is too old" | Put the newest firmware on the stick (step 2). |
| The stick is not in the **USB port** list | Unplug it and plug it in again. Wait 10 seconds and open the form again. Or type the path yourself (step 4). |
| No heater is found | Turn a heater's temperature up so that it starts heating. Move the stick closer to a heater (a USB extension cable helps). Check that the stick is the **868 MHz** version. Or pair the heater again (step 5). |
| A heater is found, then often shows **unavailable** | The signal is weak. Move the stick away from the computer and metal, or use an SMA antenna. |
| Flashing: "programmer is not responding" or "not in sync" | Use baud rate **57600** instead of 115200 (step 2). Check the port. Close other programs that use the port. |
| Flashing: the browser does not show the stick | Use Chrome or Edge. Install the USB driver (CH340) on Windows. Try another USB cable: some cables only charge. |
| Flashing: `device signature = 0x1e9587` or another value | The stick does not have an ATmega328P. Our firmware does not run on it. |

Still stuck? Download diagnostics (**Settings → Devices & Services → TermoWeb →
⋮ → Download diagnostics**) and open an issue at
<https://github.com/ha-termoweb/ha-termoweb/issues>.

## Put the old firmware back

This project does not ship a culfw file. To use the stick for FHEM or other
systems again, build culfw from its source and flash it the same way as in
step 2.

What works over the radio, and what does not yet, is listed in
[the radio backend reference](radio_backend.md). Technical details of the
heaters' radio are in [the radio protocol reference](radio_protocol.md).
