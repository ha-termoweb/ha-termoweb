// TermoWeb Schedule Card — rebuilt with edit-freeze + manual refresh
// Features:
// - 7×24 schedule painter (0=cold, 1=night, 2=day)
// - Preset temperature editors (ptemp: [cold, night, day])
// - Writes via entity services on the integration domain:
//     termoweb.set_schedule
//     termoweb.set_preset_temperatures
// - Local edit freeze: while the user is editing or after Save, the card will
//   NOT hydrate from HA state until the user clicks Refresh, or until a timed
//   window elapses and the incoming state matches the last-sent payload.
// - Colors: Cold = Cyan (#00BCD4), Day = Orange (#FB8C00), Night = Dark Blue (#0D47A1)
// - Indexing: Monday-based; index = day*24 + hour
//
// v1.1.0

(() => {
  const DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const HOUR_LABELS = Array.from({ length: 24 }, (_, h) => `${String(h).padStart(2, "0")}:00`);

  // Requested palette
  const COLORS = {
    0: "var(--termoweb-cold-color, #00BCD4)",  // Cold -> Cyan
    1: "var(--termoweb-night-color, #0D47A1)", // Night -> Dark Blue
    2: "var(--termoweb-day-color, #FB8C00)",   // Day -> Orange
    border: "var(--divider-color, rgba(255,255,255,0.12))",
    cellBg: "var(--card-background-color, #1f1f1f)",
    label: "var(--secondary-text-color, #9e9e9e)",
    text: "var(--primary-text-color, #e0e0e0)",
    subtext: "var(--secondary-text-color, #a0a0a0)",
  };

  // Card picker registration
  window.customCards = window.customCards || [];
  window.customCards.push({
    type: "termoweb-schedule-card",
    name: "TermoWeb Schedule",
    description: "Edit the weekly schedule and presets of a TermoWeb heater",
    preview: false,
  });

  const nowMs = () => Date.now();
  const deepEqArray = (a, b) => {
    if (a === b) return true;
    if (!Array.isArray(a) || !Array.isArray(b)) return false;
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  };

  class TermoWebScheduleCard extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: "open" });
      this._hass = null;
      this._config = null;
      this._stateObj = null;

      // Local working copies
      this._progLocal = null;        // int[168]
      this._ptempLocal = [null, null, null]; // [cold, night, day]

      // Dirty flags (user edited)
      this._dirtyProg = false;
      this._dirtyPresets = false;

      // Freeze window: ignore hass updates while editing / just after save
      this._freezeUntil = 0; // epoch ms; 0 means not frozen
      this._freezeWindowMs = 15000; // 15s after save

      // Last-sent payloads to detect echo
      this._lastSent = { prog: null, ptemp: null };

      // painting
      this._dragging = false;
      this._paintValue = null;
      this._selectedMode = 0;
      this._boundMouseUp = () => this._onMouseUp();

      // Available TermoWeb heater entities
      this._entities = [];
    }

    setConfig(config) {
      if (!config || !config.entity) {
        throw new Error("termoweb-schedule-card: 'entity' is required");
      }
      this._config = config;
      this._render();
    }

    set hass(hass) {
      this._hass = hass;
      if (!this._config) return;

      // Collect available TermoWeb heater entities
      this._entities = Object.entries(hass.states)
        .filter(([eid, st]) => {
          if (!eid.startsWith("climate.")) return false;
          const a = st?.attributes || {};
          return Array.isArray(a.prog) && a.prog.length === 168;
        })
        .map(([eid, st]) => ({
          id: eid,
          name: st.attributes?.friendly_name || st.attributes?.name || eid,
        }));

      const st = hass.states[this._config.entity];
      this._stateObj = st || null;

      const canHydrateNow = this._canHydrateFromState();
      const attrs = st?.attributes || {};

      if (canHydrateNow) {
        // Prog
        if (Array.isArray(attrs.prog) && attrs.prog.length === 168) {
          // If we were waiting for echo and it matches last sent, unfreeze.
          if (this._lastSent.prog && deepEqArray(attrs.prog, this._lastSent.prog)) {
            this._freezeUntil = 0;
          }
          this._progLocal = attrs.prog.slice();
        }
        // Presets
        if (Array.isArray(attrs.ptemp) && attrs.ptemp.length === 3) {
          if (this._lastSent.ptemp && deepEqArray(attrs.ptemp, this._lastSent.ptemp)) {
            this._freezeUntil = 0;
          }
          this._ptempLocal = attrs.ptemp.slice();
        }
      }
      // Re-render regardless (for header / unit changes)
      this._render();
    }

    _canHydrateFromState() {
      // Only hydrate when:
      // - No local copy yet (first load)
      // - Not currently dirty
      // - Not within freeze window
      const now = nowMs();
      const inFreeze = now < this._freezeUntil;
      const hasLocal = Array.isArray(this._progLocal) && this._progLocal.length === 168;
      if (!hasLocal) return true;
      if (this._dirtyProg || this._dirtyPresets) return false;
      if (inFreeze) return false;
      return true;
    }

    getCardSize() { return 16; }

    // ---------- helpers ----------
    _units() {
      const u = this._stateObj?.attributes?.units;
      return (u === "F" || u === "C") ? u : "C";
    }
    _idx(day, hour) { return day * 24 + hour; }
    _copyDay(fromDay, toDay) {
      if (!this._progLocal) return;
      const copyTo = (d) => {
        for (let h = 0; h < 24; h++) {
          const src = this._idx(fromDay, h);
          const dst = this._idx(d, h);
          this._progLocal[dst] = this._progLocal[src];
        }
      };
      if (toDay === "All") {
        for (let d = 0; d < 7; d++) {
          if (d !== fromDay) copyTo(d);
        }
      } else if (Number.isInteger(toDay) && toDay >= 0 && toDay < 7) {
        copyTo(toDay);
      }
    }
    _toast(msg) {
      const el = document.createElement("div");
      el.textContent = msg;
      el.style.cssText =
        "position:fixed;left:50%;bottom:16px;transform:translateX(-50%);background:rgba(0,0,0,0.75);color:#fff;padding:8px 12px;border-radius:6px;z-index:9999;font-size:12px;";
      document.body.appendChild(el);
      setTimeout(() => el.remove(), 1800);
    }

    // ---------- schedule interaction ----------
    _onCellClick(day, hour) {
      if (!this._progLocal) return;
      const i = this._idx(day, hour);
      this._progLocal[i] = this._selectedMode;
      this._dirtyProg = true;
      this._renderGridOnly();
    }
    _onMouseDown(day, hour) {
      if (!this._progLocal) return;
      this._dragging = true;
      const i = this._idx(day, hour);
      const next = this._selectedMode;
      this._paintValue = next;
      this._progLocal[i] = next;
      this._dirtyProg = true;
      window.addEventListener("mouseup", this._boundMouseUp, { once: true });
      this._renderGridOnly();
    }
    _onMouseOver(day, hour) {
      if (!this._dragging || this._paintValue == null || !this._progLocal) return;
      const i = this._idx(day, hour);
      if (this._progLocal[i] !== this._paintValue) {
        this._progLocal[i] = this._paintValue;
        this._dirtyProg = true;
        this._colorCell(day, hour, this._paintValue);
      }
    }
    _onMouseUp() { this._dragging = false; this._paintValue = null; }

    _revert() {
      // Force re-hydrate from current HA state
      const st = this._hass?.states?.[this._config.entity];
      const attrs = st?.attributes || {};
      if (Array.isArray(attrs.prog) && attrs.prog.length === 168) {
        this._progLocal = attrs.prog.slice();
      }
      if (Array.isArray(attrs.ptemp) && attrs.ptemp.length === 3) {
        this._ptempLocal = attrs.ptemp.slice();
      }
      this._dirtyProg = false;
      this._dirtyPresets = false;
      this._freezeUntil = 0;
      this._lastSent = { prog: null, ptemp: null };
      this._render();
    }

    _refreshFromState() {
      // Manual refresh, ignoring freeze; useful if user wants to sync now
      const st = self._hass?.states?.[this._config.entity];
      const attrs = st?.attributes || {};
      if (Array.isArray(attrs.prog) && attrs.prog.length === 168) {
        this._progLocal = attrs.prog.slice();
      }
      if (Array.isArray(attrs.ptemp) && attrs.ptemp.length === 3) {
        this._ptempLocal = attrs.ptemp.slice();
      }
      this._dirtyProg = false;
      this._dirtyPresets = false;
      this._freezeUntil = 0;
      this._render();
    }

    // ---------- preset editing ----------
    _parseInputNum(id) {
      const el = this.shadowRoot.getElementById(id);
      if (!el) return null;
      const n = Number(el.value);
      return Number.isFinite(n) ? n : null;
    }

    async _savePresets() {
      if (!this._hass || !this._config) return;
      const cold = this._parseInputNum("tw_p_cold");
      const night = this._parseInputNum("tw_p_night");
      const day = this._parseInputNum("tw_p_day");
      if (cold == null || night == null || day == null) {
        this._toast("Enter valid numbers for Cold / Night / Day");
        return;
      }
      const payload = [cold, night, day];
      try {
        await this._hass.callService("termoweb", "set_preset_temperatures", {
          entity_id: this._config.entity,
          ptemp: payload.slice(),
        });
        this._ptempLocal = payload.slice();
        this._dirtyPresets = false;
        this._lastSent.ptemp = payload.slice();
        this._freezeUntil = nowMs() + this._freezeWindowMs;
        this._toast("Preset temperatures sent (waiting for device to update)");
      } catch (e) {
        this._toast("Failed to save presets");
        console.error("TermoWeb card: set_preset_temperatures error:", e);
      }
    }

    // ---------- save schedule ----------
    async _saveSchedule() {
      if (!this._hass || !this._config || !this._progLocal) return;

      if (!Array.isArray(this._progLocal) || this._progLocal.length !== 168) {
        this._toast("Invalid program (expected 168 values)");
        return;
      }
      for (const v of this._progLocal) {
        if (v !== 0 && v !== 1 && v !== 2) {
          this._toast("Program has invalid values (allowed: 0,1,2)");
          return;
        }
      }

      const body = this._progLocal.slice();
      try {
        await this._hass.callService("termoweb", "set_schedule", {
          entity_id: this._config.entity,
          prog: body,
        });
        this._dirtyProg = false;
        this._lastSent.prog = body.slice();
        this._freezeUntil = nowMs() + this._freezeWindowMs;
        this._toast("Schedule sent (waiting for device to update)");
      } catch (e) {
        this._toast("Failed to save schedule");
        console.error("TermoWeb card: set_schedule error:", e);
      }
    }

    // ---------- render ----------
    _render() {
      const root = this.shadowRoot;
      if (!root) return;

      const title =
        (this._stateObj?.attributes?.friendly_name || this._stateObj?.attributes?.name) ||
        this._config?.entity || "TermoWeb schedule";

      const hasProg = Array.isArray(this._progLocal) && this._progLocal.length === 168;
      const units = this._units();
      const stepAttr = units === "F" ? "1" : "0.5";
      const [cold, night, day] = this._ptempLocal ?? [null, null, null];

      const dirtyBadge = (this._dirtyProg || this._dirtyPresets) ?
        `<span class="dirty">● unsaved</span>` : ``;

      const frozen = nowMs() < this._freezeUntil;

      const entityOptions = (this._entities || [])
        .map((e) => `<option value="${e.id}" ${e.id === this._config?.entity ? "selected" : ""}>${e.name}</option>`)
        .join("\n");

      const dayOptions = DAY_NAMES.map((d, i) => `<option value="${i}">${d}</option>`).join("\n");

      root.innerHTML = `
        <style>
          :host { display:block; }
          .card { padding: 12px; color: ${COLORS.text}; }
          .header { display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;font-weight:600; }
          .sub { color: ${COLORS.subtext}; font-size: 12px; display:flex; align-items:center; gap:8px; }
          .dirty { color: var(--warning-color, #ffa000); font-size: 11px; }
          .grid { display: grid; grid-template-columns: 56px repeat(7, 1fr); gap: 6px; margin-top: 8px; }
          .hour { color: ${COLORS.label}; font-size: 12px; text-align: right; padding: 4px 6px; }
          .dayhdr { color: ${COLORS.label}; font-size: 12px; text-align: center; padding: 4px 0 8px 0; }
          .cell { background: ${COLORS.cellBg}; border: 1px solid ${COLORS.border}; height: 20px; border-radius: 6px; cursor: pointer; transition: filter .06s; }
          .cell:hover { filter: brightness(1.08); }
          .legend { display:flex;gap:12px;align-items:center;flex-wrap:wrap;color:${COLORS.label}; font-size: 12px; }
          .legend .swatch { display:inline-block;width:14px;height:14px;border-radius:4px;border:1px solid ${COLORS.border};vertical-align:-2px;margin-right:6px; }
          .row { display:flex; gap:10px; align-items:center; margin-top:10px; flex-wrap: wrap; color:${COLORS.label}; }
          .modeToggle { display:flex; gap:8px; margin-top:10px; }
          .modeToggle button { flex:1; padding:8px 0; border-radius:8px; border:2px solid ${COLORS.border}; color:${COLORS.text}; cursor:pointer; font-weight:600; opacity:0.6; }
          .modeToggle button.active { opacity:1; border-color:${COLORS.text}; }
          input[type="number"] {
            width: 72px;
            border-radius: 8px;
            border: 1px solid ${COLORS.border};
            background: var(--secondary-background-color, #2b2b2b);
            color: ${COLORS.text};
            padding: 5px 8px;
          }
          select {
            border-radius: 6px;
            border: 1px solid ${COLORS.border};
            background: var(--secondary-background-color, #2b2b2b);
            color: ${COLORS.text};
            padding: 3px 4px;
          }
          .footer { display:flex;justify-content:flex-end;gap:8px;margin-top:10px; flex-wrap: wrap; }
          button { background: var(--secondary-background-color, #2b2b2b); color: ${COLORS.text}; border: 1px solid ${COLORS.border}; border-radius: 8px; padding: 6px 10px; font-size: 12px; cursor: pointer; }
          button:hover { filter: brightness(1.1); }
          .warn { color: var(--error-color, #ef5350); }
          .chip { padding:2px 6px; border:1px solid ${COLORS.border}; border-radius: 10px; font-size:11px; }
        </style>

        <ha-card class="card">
          <div class="header">
            <div>${title}</div>
            <div class="sub">
              <select id="entitySelect">${entityOptions}</select>
              ${dirtyBadge}
              ${frozen ? `<span class="chip">waiting for device update…</span>` : ``}
              <button id="refreshBtn" title="Refresh from current state">Refresh</button>
            </div>
          </div>

          <!-- Legend -->
          <div class="legend">
            <span><span class="swatch" style="background:${COLORS[0]}"></span>Cold</span>
            <span><span class="swatch" style="background:${COLORS[2]}"></span>Day</span>
            <span><span class="swatch" style="background:${COLORS[1]}"></span>Night</span>
            <span>Units: ${units}</span>
          </div>

          <!-- Preset editors -->
          <div class="row">
            <label>Cold <input id="tw_p_cold" type="number" step="${stepAttr}" value="${cold ?? ""}"></label>
            <label>Night <input id="tw_p_night" type="number" step="${stepAttr}" value="${night ?? ""}"></label>
            <label>Day <input id="tw_p_day" type="number" step="${stepAttr}" value="${day ?? ""}"></label>
            <button id="savePresetsBtn">Save Presets</button>
          </div>

          <div class="modeToggle">
            <button id="modeCold" class="${this._selectedMode === 0 ? 'active' : ''}" style="background:${COLORS[0]}">Cold</button>
            <button id="modeNight" class="${this._selectedMode === 1 ? 'active' : ''}" style="background:${COLORS[1]}">Night</button>
            <button id="modeDay" class="${this._selectedMode === 2 ? 'active' : ''}" style="background:${COLORS[2]}">Day</button>
          </div>

          ${!hasProg ? `<div class="warn" style="margin-top:8px;">This entity has no valid 'prog' (expected 168 ints).</div>` : ""}

          ${this._renderGridShell()}

          <div class="row">
            <label>Copy From <select id="copyFromSel">${dayOptions}</select></label>
            <label>Copy To <select id="copyToSel"><option value="All">All</option>${dayOptions}</select></label>
            <button id="copyBtn">Copy</button>
          </div>

          <div class="footer">
            <button id="revertBtn">Revert</button>
            <button id="saveBtn">Save</button>
          </div>
        </ha-card>
      `;

      // Bind Refresh
      root.getElementById("refreshBtn")?.addEventListener("click", () => this._refreshFromState());

      // Bind entity selector
      root.getElementById("entitySelect")?.addEventListener("change", (ev) => {
        const newEntity = ev.target.value;
        if (newEntity && newEntity !== this._config.entity) {
          this._config.entity = newEntity;
          this._stateObj = this._hass?.states?.[newEntity] || null;
          this._revert();
        }
      });

      // Bind preset inputs to update local state and set dirty flag
      root.getElementById("tw_p_cold")?.addEventListener("input", () => {
        this._ptempLocal[0] = this._parseInputNum("tw_p_cold");
        this._dirtyPresets = true;
      });
      root.getElementById("tw_p_night")?.addEventListener("input", () => {
        this._ptempLocal[1] = this._parseInputNum("tw_p_night");
        this._dirtyPresets = true;
      });
      root.getElementById("tw_p_day")?.addEventListener("input", () => {
        this._ptempLocal[2] = this._parseInputNum("tw_p_day");
        this._dirtyPresets = true;
      });

      // Bind preset save
      root.getElementById("savePresetsBtn")?.addEventListener("click", () => this._savePresets());

      // Bind mode buttons
      root.getElementById("modeCold")?.addEventListener("click", () => { this._selectedMode = 0; this._render(); });
      root.getElementById("modeNight")?.addEventListener("click", () => { this._selectedMode = 1; this._render(); });
      root.getElementById("modeDay")?.addEventListener("click", () => { this._selectedMode = 2; this._render(); });

      // Bind schedule buttons
      root.getElementById("revertBtn")?.addEventListener("click", () => this._revert());
      root.getElementById("saveBtn")?.addEventListener("click", () => this._saveSchedule());

      root.getElementById("copyBtn")?.addEventListener("click", () => {
        const fromEl = root.getElementById("copyFromSel");
        const toEl = root.getElementById("copyToSel");
        if (!fromEl || !toEl) return;
        const from = Number(fromEl.value);
        const toVal = toEl.value;
        const to = toVal === "All" ? "All" : Number(toVal);
        this._copyDay(from, to);
        this._dirtyProg = true;
        this._renderGridOnly();
      });

      // Paint cells
      this._renderGridOnly();
    }

    _renderGridShell() {
      // header row
      let headerRow = `<div></div>`;
      for (let d = 0; d < 7; d++) headerRow += `<div class="dayhdr">${DAY_NAMES[d]}</div>`;

      // body rows (24 hours × 7 days)
      let rows = "";
      for (let h = 0; h < 24; h++) {
        rows += `<div class="hour">${HOUR_LABELS[h]}</div>`;
        for (let d = 0; d < 7; d++) {
          rows += `<div class="cell" data-d="${d}" data-h="${h}"></div>`;
        }
      }
      return `<div class="grid">${headerRow}${rows}</div>`;
    }

    _renderGridOnly() {
      const root = this.shadowRoot;
      if (!root) return;
      const cells = root.querySelectorAll(".cell");
      if (!cells || !this._progLocal || this._progLocal.length !== 168) return;

      cells.forEach((cell) => {
        const d = Number(cell.getAttribute("data-d"));
        const h = Number(cell.getAttribute("data-h"));
        const idx = this._idx(d, h);
        const v = Number(this._progLocal[idx] ?? 0);

        cell.style.background = COLORS[v in COLORS ? v : 0];

        if (!cell._twBound) {
          cell._twBound = true;
          cell.addEventListener("click", () => this._onCellClick(d, h));
          cell.addEventListener("mousedown", () => this._onMouseDown(d, h));
          cell.addEventListener("mouseover", () => this._onMouseOver(d, h));
        }
      });
    }

    _colorCell(day, hour, v) {
      const root = this.shadowRoot;
      const el = root && root.querySelector(`.cell[data-d="${day}"][data-h="${hour}"]`);
      if (el) el.style.background = COLORS[v in COLORS ? v : 0];
    }

    static getConfigElement() { return null; }
  }

  customElements.define("termoweb-schedule-card", TermoWebScheduleCard);
})();