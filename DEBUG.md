# Collecting Debug Logs for `custom_components.termoweb` (UI-only)

This guide shows a non-technical end‑user how to capture **startup and runtime** debug logs and diagnostics for the **Termoweb** custom integration using only the Home Assistant UI (no add‑ons, no CLI).

> Assumptions: Home Assistant is running; the integration is installed; debug logging can be enabled per‑integration in your HA version.

---

## Quick Steps

1. **Enable debug logging for Termoweb**
   - Go to **Settings → Devices & Services**.
   - Open the **Termoweb** integration card.
   - Click the **⋮** menu (top‑right) → **Enable debug logging**.

2. **Reload the integration to capture startup logs**
   - On the same Termoweb card, **⋮ → Reload**.
   - If you don’t see a Reload item, use the fallback service below.

3. **Reproduce the issue**
   - Operate the device or perform the action that triggers the problem right after the reload (so startup + runtime paths are logged).

4. **Download the logs**
   - Return to the Termoweb card **⋮ → Disable debug logging**.
   - Your browser will download a log file (keep it as is).

5. **Download diagnostics (JSON)**
   - On the **Termoweb** integration card, **⋮ → Download diagnostics**.
   - Save the downloaded `.json` file. This is often requested together with the log file.

6. **(Optional) Inspect logs in the UI**
   - Go to **Settings → System → Logs → Load full logs**.
   - Use the search box to filter for `custom_components.termoweb`.

---

## What to Send
- The **downloaded log file** after you disabled debug logging.
- The **Termoweb diagnostics JSON** file.
- Optionally, screenshots of any visible errors in **Settings → System → Logs** filtered by `custom_components.termoweb`.

> Tip: The TermoWeb integration makes an effort to redact sensitive and identifying info such as the gateeway ID and session tokens. However, many other components in HA do not, so you should review the log before posting it to avoid revealing information that is private.

---

## Energy history import service

The integration exposes the `termoweb.import_energy_history` service to backfill hourly energy statistics. The service supports the following fields:

- `max_history_retrieval` *(int, 1-3650, default 7)* — how many days back to import. If a node was already imported for fewer days, only the older days are fetched.
- `reset_progress` *(bool, default false)* — forget the saved progress and import the whole period again. Existing statistics are overwritten in place, never deleted.

### Internal rules

- Requests go through the shared samples limiter: at most **2 queries per second**, one request per node per day of history.
- A sample's counter is the cumulative total at `t`, so consumption between the samples at T and T+1h is stored in the statistic that starts at T.
- Each day's statistics are written and committed by the recorder *before* the progress is saved (in `.storage/termoweb.energy_import.<entry_id>`). If a request fails (login, rate limit, network), the import stops with an error and keeps the progress of the last written day; run the service again to resume.
- Sums continue from the statistic before the import window. After a node finishes, every later statistic (hourly and 5-minute) is shifted by the difference, so the Energy dashboard shows no jump.
- Counter drops of **0.2 kWh** or more are counted as device resets; the drop is not subtracted from the sum.
- Only one import runs at a time. A second call while one is running fails with "already running".
- A per-node summary is logged at INFO level and the latest run is exposed in diagnostics (`energy_import.last_run`).
- Older versions kept progress in the entry options (`energy_history_progress`, `energy_history_imported`, `max_history_retrieved`). The first import moves a completed import's progress to storage and removes those options; an unfinished old import starts again.

---

## Websocket health telemetry (diagnostics)

The **TermoWeb diagnostics JSON** now surfaces extra websocket fields to help support identify “connected but quiet” issues for Ducaheat devices:

- `subscribe_attempts_total`, `subscribe_success_total`, `subscribe_fail_total`, `last_subscribe_success_at` — show whether subscriptions were installed and when they last succeeded.
- `recovery_attempts_total`, `last_recovery_at` — count idle recovery attempts and the timestamp of the most recent recovery.
- `last_update_event_at` — the most recent device update (dev_data/update) received from the cloud.
- `parse_errors_total` — number of malformed websocket frames that were skipped.
