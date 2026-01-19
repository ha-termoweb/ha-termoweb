# INSTALL – TermoWeb Schedule Editor Card

This guide shows how to install and use the optional **Lovelace schedule editor** (`custom:termoweb-schedule-card`) **using only the Home Assistant web UI**. A CLI method is provided at the end for advanced users.

---

## Prerequisites

- The **TermoWeb integration** is installed and working.
- You can see one or more **climate** entities (your heaters) in Home Assistant.
- Each heater shows `prog` and `ptemp` attributes under **Developer Tools → States** (select your climate entity).

> The schedule card is optional. The integration works without it.

---

## A) UI‑only install (no CLI)

### 1) Install a file editor add‑on (one-time)

Choose either add‑on (both work from the HA web UI):

- **File editor** (simple)
- **Studio Code Server** (VS Code in the browser)

Steps (same for both):
1. Open **Settings → Add-ons → Add-on Store**.
2. Search for **File editor** (or **Studio Code Server**) → **Install**.
3. **Start** the add-on.
4. Optional: enable **Show in sidebar** for quick access.

### 2) Create the folder and place the card file

You need this final path in your HA config:

```
<config>/www/termoweb/termoweb_schedule_card.js
```

Using the add‑on you installed:

1. Open **File editor** (or **Studio Code Server**).
2. Navigate to your HA **config** directory (usually `/config`).
3. If the `www` folder doesn’t exist, **create it**.
4. Inside `www`, **create a folder** named `termoweb`.
5. Put the card file **`termoweb_schedule_card.js`** inside the `termoweb` folder.

**Where do I get the card file?**  
Either:
- If the card is already bundled with your build, copy it from
  `custom_components/termoweb/assets/termoweb_schedule_card.js` to `www/termoweb/`.


### 3) Add a Lovelace resource

From the HA web UI:
1. **Settings → Dashboards**.
2. Click the **⋮** menu (top‑right) → **Resources** → **Add Resource**.
3. **URL**: `/local/termoweb/termoweb_schedule_card.js`  
   **Type**: `JavaScript Module`

Hard‑reload your browser (Ctrl/Cmd + Shift + R).

### 4) Add the card to a dashboard view

Edit your dashboard → **Add card** → **Manual** and use:

```yaml
type: custom:termoweb-schedule-card
entity: climate.YOUR_HEATER_ENTITY
```

Replace `climate.YOUR_HEATER_ENTITY` with your climate entity id.

---

## B) CLI method (optional)

If you prefer SSH/CLI (e.g., via the **SSH & Web Terminal** add‑on), run this helper script to copy the card from the integration directory to the public `www` path. The script **does not** add the Lovelace resource (that part must still be done in the UI).

1. Download the script:
   - [Download install_termoweb_card.sh](sandbox:/mnt/data/install_termoweb_card.sh)

2. Copy it to your HA **/config** directory (e.g., with the File editor add‑on or `scp`).

3. SSH into HA and run:
   ```bash
   cd /config
   chmod +x install_termoweb_card.sh
   ./install_termoweb_card.sh
   ```

If the card file is present at either:
- `/config/custom_components/termoweb/www/termoweb_schedule_card.js`, or
- `/config/custom_components/termoweb/termoweb_schedule_card.js`

…the script will copy it into `/config/www/termoweb/` and print the final path.

Then, add the Lovelace resource in the UI (see step A‑3).

---

## Using the card

- **Heater dropdown** – select which TermoWeb climate entity to edit when multiple heaters are available.
- **Day-copy controls** – choose a source and target day (or all days) to duplicate schedules in one click.
- **Paint mode buttons** – pick Cold, Night, or Day then drag across the grid to set hours.
- The card still uses manual **Refresh** and an edit-**freeze** window; click Refresh or wait for the freeze to expire to re-sync with Home Assistant.

---

## Verify it works

1. Open **Developer Tools → States**, pick your climate entity, and confirm it has `prog` and `ptemp` attributes.
2. Open your dashboard and the **TermoWeb Schedule** card.
3. Paint a small change (e.g., Monday 00‑02 to “day”) and **Save**.
4. Within ~2–3 seconds you should see `last_updated` bump and the `prog` array reflect your change. The manufacturer’s mobile app should also show the new schedule shortly after.

---

## Troubleshooting

- **Card doesn’t render**  
  Ensure the resource URL is `/local/termoweb/termoweb_schedule_card.js` and **Type** is `JavaScript Module`. Hard‑reload the browser.

- **“Action … not found” when saving**  
  The card calls `termoweb.set_schedule` (entity service). Confirm it appears in **Developer Tools → Actions**. If missing, restart Home Assistant.

- **Save succeeds but nothing updates**  
  Make sure your `prog` has exactly 168 integers in `{0,1,2}`. Check the log for `custom_components.termoweb.climate` and `custom_components.termoweb.backend.rest_client` lines.

- **File paths**  
  The public resource path is `/config/www/termoweb/termoweb_schedule_card.js`. In the browser this is `/local/termoweb/termoweb_schedule_card.js`.
