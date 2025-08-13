# TermoWeb heaters for Home Assistant

Control your **TermoWeb** electric heaters in **Home Assistant** — from the HA app, automations, scenes, and voice assistants.

[![Open in HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=ha-termoweb&repository=ha-termoweb&category=integration)
[![Open your Home Assistant instance and start setting up the integration.](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=termoweb)

> You must install the integration first (via HACS or manual copy) before the “Add integration” button will work.

---

## Who is this for?

For someone who runs Home Assistant and already uses the **TermoWeb** mobile app or signs in at **control.termoweb.net** (or **control2.termoweb.net**) to manage their electric heaters. They want to see and control those heaters in Home Assistant, use automations (e.g., bedtime setback), and enable voice control. The manufacturer’s app doesn’t integrate with HA — this add-on provides the missing link.

---

## Brands commonly using the TermoWeb app

These product lines are documented to work with the **TermoWeb** portal/app:

- **S&P — Soler & Palau**: “**Termoweb**” kits and **EMI-TECH Termoweb** radiators.  
- **Ecotermi / Linea Plus**: **Serie Termoweb** radiators.  
- **EHC — Electric Heating Company**: **eco SAVE** Smart Gateway kits that register on the TermoWeb portal.
- **ATC (UK/Ireland)**: **Sun Ray Wifi** radiators with Wifi gateway.


> If a brand isn’t listed but the user signs in at **control.termoweb.net** (or **control2.termoweb.net**) with an app called **TermoWeb**, this integration should work.

_Not supported:_ brands using different apps/backends (for example “Ducaheat/Ducasa”’s own “Termoweb” app, which is a separate system).

---

## What you can do in Home Assistant

- Turn heaters **On/Off** and set **target temperature**.
- Choose **Auto** or **Manual** (as “presets” in HA).
- See room temperature and heating state.
- Use **automations**, **scenes**, and **voice assistants** (via HA’s Google/Alexa integrations).

---

## What you’ll need

- A working TermoWeb setup (gateway connected to the router, heaters paired).
- The **TermoWeb account email & password** (the same used in the mobile app / web).
- Home Assistant (Core, OS, or Container) with internet access.

---

## Install (simple, step-by-step)

### Option A — HACS (recommended)

1) Open **HACS → Integrations** in Home Assistant.  
2) Click **⋮** (top-right) → **Custom repositories** → **Add**.  
3) Paste: `https://github.com/ha-termoweb/ha-termoweb` and choose **Integration**.  
   Or click:  
   [![Open in HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=ha-termoweb&repository=ha-termoweb&category=integration)  
4) Search for **TermoWeb** in HACS and **Install**.  
5) **Restart Home Assistant** when prompted.

### Option B — Manual

1) Download this repository.  
2) Copy the folder **`custom_components/termoweb`** to **`<config>/custom_components/termoweb`** on the HA system.  
3) **Restart Home Assistant**.

---

## Set up the integration
ha-termoweb/ha-termoweb
1) In Home Assistant go to **Settings → Devices & Services → Add Integration** and search **TermoWeb**,  
   or click:  
   [![Open your Home Assistant instance and start setting up the integration.](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=termoweb)
2) Enter the **TermoWeb login** (email & password).  
3) **Portal address**:  
   - If the website used is **`https://control.termoweb.net`**, enter that.  
   - Some systems use **`https://control2.termoweb.net`** — use whichever site is used today.  
4) Complete the wizard. Heaters will appear under **Devices**; add them to dashboards or use them in automations.

---

## Tips
- **Voice control:** Expose heater entities via Home Assistant’s Google or Alexa integrations.
- **Automations idea:** Lower temperature when nobody’s home; switch to **Off** if a window sensor is open for 10+ minutes.

## Energy monitoring & history
- Each heater provides an **Energy** sensor in kWh and the integration adds a **Total Energy** sensor aggregating all heaters.
- Add these sensors in **Settings → Dashboards → Energy** to include them in Home Assistant’s Energy Dashboard.
- Energy counters are fetched hourly from TermoWeb.
- Use the `termoweb.import_energy_history` service (Developer Tools → Services) to backfill past consumption after installing the integration.
- No extra configuration is required beyond selecting the sensors in the Energy Dashboard.

---

## Troubleshooting

- **Login fails:** First confirm credentials at the TermoWeb website (control.termoweb.net / control2.termoweb.net).  
- **No devices found:** Check the **gateway** is powered and online (LEDs), and that the manufacturer app shows heaters online.  
- **Need help?** Open a GitHub issue with brand/model and a brief description. **Never share passwords or private info.**

---

## Privacy & Security

- Credentials stay in Home Assistant.  
- Access tokens are **redacted** from logs.  
- This project is **not affiliated** with S&P, ATC, Ecotermi/Linea Plus, EHC, or TermoWeb.

---

## Search keywords

*Home Assistant TermoWeb, TermoWeb heaters Home Assistant, ATC radiators, S&P TermoWeb Home Assistant, Soler & Palau Termoweb, Ecotermi Termoweb, Linea Plus Termoweb, Electric Heating Company eco SAVE Home Assistant, eco SAVE Smart Gateway Home Assistant*

