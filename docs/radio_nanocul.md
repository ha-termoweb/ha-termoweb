# Radio stick: use a nanoCUL USB stick

A nanoCUL is a small USB stick: an Arduino Nano with a CC1101 radio for
868 MHz. With the `termoweb_rx` firmware it talks to your heaters directly,
like the [ESP32 radio gateway](radio_gateway.md), but it plugs into a USB
port of the computer that runs Home Assistant.

## Before you start: which heaters work

Your heaters use one of two radio "dialects". The stock `termoweb_rx`
firmware for the nanoCUL only speaks **dialect A**.

- If the setup finds a **dialect B** network, the stick cannot talk to those
  heaters yet. Use the ESP32 radio gateway instead, or wait for a nanoCUL
  firmware with dialect support.
- The setup tells you which dialect it hears.

## Step 1: put the termoweb_rx firmware on the stick

You need a computer with `avr-gcc` and `avrdude` (for example a Linux PC or
a Raspberry Pi), and the
[ha-termoweb-local](https://github.com/ha-termoweb/ha-termoweb-local)
repository.

1. Plug the nanoCUL into that computer.
2. Open a terminal in `firmware/termoweb_rx`.
3. Run `make flash PORT=/dev/ttyUSB0`. Use your stick's port if it is
   different.
4. Wait until avrdude says the flash was verified.

## Step 2: plug the stick into Home Assistant

1. Plug the nanoCUL into a USB port of the Home Assistant computer.
2. Put the stick where it has a clear path to your heaters. A short USB
   extension cable helps to move it away from the computer and metal.

## Step 3: connect the integration

1. In Home Assistant, go to **Settings → Devices & Services → Add Integration**
   and search for **TermoWeb**.
2. Choose **Local radio: nanoCUL USB stick (CC1101 868 MHz)**.
3. Choose your stick in the **USB port** list. If it is not in the list,
   choose **Enter the port path or URL myself** and type the path, for
   example `/dev/serial/by-id/usb-...`.
4. Leave **Radio dialect** on **auto** and **Network id** empty.
5. Turn one heater's temperature up, so that it starts heating, then press
   **Submit** and wait. This takes up to 6 minutes.
6. When it finishes, your heaters appear under **Devices**.

If you see "Cannot open the nanoCUL stick": check the cable, the port and the
firmware (step 1).

If you see "This nanoCUL firmware only knows radio dialect A": your heaters
use dialect B. See "Before you start".

To change the port later, open the integration and choose **Reconfigure**.

A stick shared over the network (for example with `ser2net`) also works: type
its address as `socket://<address>:<port>` in step 3.
