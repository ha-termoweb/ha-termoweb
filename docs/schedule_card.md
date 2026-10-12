# Schedule card: edit the weekly schedule

The integration includes a dashboard card. With it you can:

- set the **Cold**, **Night** and **Day** preset temperatures of a heater,
- choose, for each hour of the week, which preset the heater uses,
- copy a day to other days, or copy a whole schedule to another heater.

It works with cloud heaters (TermoWeb, Ducaheat, Tevolve) and with local radio heaters. The heater follows the schedule when its mode is **Auto**.

![The schedule card](programming-card-preview.png)

You do not need to install anything. The card comes with the integration and is updated with it.

## Add the card to a dashboard

1. Open the dashboard where you want the card.
2. Click the **pencil** icon at the top right to edit the dashboard. If Home Assistant asks whether to take control of the dashboard, click **Take control**.
3. Click **+ Add card**.
4. Open the **By card** tab and type `termoweb` in the search box.
5. Click **TermoWeb Schedule**. A preview of the card opens.
6. Click **Save**, then click **Done**.

The card shows your first heater. Use the list at the top of the card to choose another heater.

## Use the card

- **Heater list** (top right): choose the heater to edit.
- **Cold**, **Night**, **Day**: the preset temperatures. Change a value and click **Save Presets**.
- **Set hourly slot to**: click **Cold**, **Night** or **Day**. Then click an hour in the grid, or hold the mouse button and drag over several hours. The colours show which preset each hour uses.
- **Copy From** / **Copy To**: copy the hours of one day to another day, or to all days. Click **Copy**.
- **Copy everything to heater**: choose another heater and click **Copy to Entity**. The card switches to that heater and shows the copied schedule and presets. Nothing is sent yet: click **Save** to send them.
- **Save**: send your changes to the heater. **● unsaved** shows that you have changes that are not sent yet.
- **Revert**: throw away your changes.
- **Refresh**: show the heater's current schedule again.

After **Save**, the card shows **Saved**. If it shows **Save failed**, the message after it says why.

## Card options

You can set these options in the card's code editor (the card has no visual editor):

```yaml
type: custom:termoweb-schedule-card
entity: climate.living_room   # optional: the heater to show first
include_pattern: "^climate\\.bedroom"   # optional: only list heaters whose entity ID matches
```

## Updating from an older version

Before version 2.4.0 you copied the card file into the `www/termoweb` folder and added a dashboard resource by hand. You do not need these any more:

- The integration replaces the old resource by itself. Your cards keep working.
- You can delete the folder `www/termoweb` (for example with the **File editor** app). This is optional.
- Reload the page in your browser once after the update.

## If you manage dashboard resources in YAML

Most people can skip this. If your `configuration.yaml` sets `resource_mode: yaml` under `lovelace:`, the integration cannot add the resource for you. Add it yourself and restart Home Assistant:

```yaml
lovelace:
  resource_mode: yaml
  resources:
    - url: /termoweb/termoweb_schedule_card.js
      type: module
```

## Troubleshooting

- **"Custom element doesn't exist: termoweb-schedule-card"**: reload the page in your browser. On a computer, press **Ctrl + F5** (**Cmd + Shift + R** on a Mac). In the Home Assistant app, pull the page down. Also check that the TermoWeb integration is loaded under **Settings → Devices & Services**.
- **The heater list is empty**: the card lists the climate entities of this integration. Check that your heaters show in **Settings → Devices & Services → TermoWeb**. If you set `include_pattern`, check that it matches your entity IDs.
- **"Save failed"**: the message says why. For a cloud heater, check that Home Assistant can reach the cloud. For a radio heater, check that the radio stick or gateway is connected.
