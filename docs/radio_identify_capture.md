# Help find the identify command

The TermoWeb app has an **identify** button. When you press it, the display
of one heater flashes, so you can see which heater is which.

We know the radio command for this on older heaters (radio "dialect A"). We do
not know it yet for newer heaters (radio "dialect B"). You can help: record
what your TermoWeb gateway sends when you press **identify** in the app, and
send us the file.

It takes about 15 minutes. In this mode the integration **only listens**. It
never transmits, so it does not disturb your TermoWeb gateway or your heaters.

## What you need

- Home Assistant with HACS.
- Your TermoWeb gateway, still working, and the TermoWeb app on your phone.
- One radio receiver near your heaters:
  - a **nanoCUL USB stick** with the `termoweb_rx` firmware
    ([how to prepare it](radio_nanocul.md)), plugged into the Home Assistant
    computer, or shared over the network (for example with `ser2net`); or
  - the **ESP32 radio gateway** ([build guide](radio_gateway.md)).

The stock nanoCUL firmware hears only dialect A. The ESP32 gateway hears both
dialects. Both are useful: if you have a nanoCUL, please send a capture too.

## Step 1: install the integration

1. Install **TermoWeb** from HACS (see the [README](../README.md)).
2. The listen-only mode is new. If the newest release has no **Radio
   capture** action yet, open TermoWeb in HACS, choose **⋮ → Redownload** and
   pick the version **main**.
3. Restart Home Assistant.

## Step 2: set up "Listen only"

1. Go to **Settings → Devices & services → Add integration** and search for
   **TermoWeb**.
2. Choose your receiver:
   - **Local radio: nanoCUL USB stick (CC1101 868 MHz)**. Choose the stick in
     the **USB port** list. For a stick shared over the network, choose
     **Enter the port path or URL myself** and type
     `socket://<address>:<port>`.
   - **Local radio gateway: ESP32 + CC1101 on your network**. Type the
     gateway's IP address. Leave the port at 2323.
3. Leave **Radio dialect** on **auto** and **Network id** empty. Press
   **Submit**.
4. Choose **Listen only (record radio traffic, never transmit)**.
5. A new device **Radio gateway** appears. It has two entities:
   - **Gateway online**: it should show *Connected*.
   - **Radio frames heard**: this number goes up when the receiver hears
     radio traffic. Heaters and the gateway talk every few minutes. If the
     number stays at 0 for 10 minutes, move the receiver closer to a heater.

Only one TermoWeb entry can use the stick or the ESP32 gateway at a time. If
you later want to control heaters with the radio, first delete the listen-only
entry, then add the integration again with the normal setup.

## Step 3: record while you press identify

Read all of this step first. You need your phone and a clock.

1. In Home Assistant, go to **Developer tools → Actions**.
2. Choose the action **TermoWeb: Radio capture**.
3. **Config entry ID**: choose your listen-only entry. **Duration**: 120
   seconds. Leave **Hide private data** off (see step 4).
4. Press **Perform action**. The recording starts now and runs for 2 minutes.
5. In the TermoWeb app, press **identify** on one heater **3 times**, about
   **20 seconds** apart. Each time, write down the time, with seconds if
   possible (for example 14:03:25). Also write down which heater you chose.
6. Wait until the 2 minutes are over. Home Assistant then shows the result,
   for example `frames: 42` and the name of a file.

If `frames` is 0, the receiver heard nothing. Check **Gateway online**, move
the receiver closer, and try again.

## Step 4: send us the file

The file is in your Home Assistant configuration folder. Its name starts with
`termoweb_radio_capture_`. You can download it with the **File editor** or
**Studio Code Server** add-on, or with Samba.

Send us, **privately** (not in a public GitHub issue):

- the file;
- the 3 times you wrote down, and your time zone;
- the heater model, if you know it.

**Privacy.** The file contains your radio network id. This id is not a
password, but please share the file only with us. If you must share it in
public, run the capture again with **Hide private data** turned on: network
ids and heater serial numbers are then replaced, and everything else stays.

Thank you for your help!
