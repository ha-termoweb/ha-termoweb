// TermoWeb Schedule Card — rebuilt with edit-freeze + manual refresh
// Features:
// - 7×24 schedule painter (0=cold, 1=night, 2=day)
// - Preset temperature editors (ptemp: [cold, night, day])
// - Writes via entity services on the integration domain:
//     termoweb.set_schedule
//     termoweb.set_preset_temperatures
// - Local edit freeze: while the user is editing or after Save, the card will
//   NOT hydrate from HA state until the user clicks Refresh, or until a timed
//   window elapses and the incoming state matches the pending echo payload.
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
      this._pendingEcho = { prog: null, ptemp: null };

      // Track which preset input is actively being edited (focus index: 0=cold,1=night,2=day)
      this._editingPresetIdx = -1;
      this._presetSelection = null;
      this._pendingPresetFocusRestore = false;
      this._presetFocusRelease = false;
      this._restoreFocusRaf = null;
      this._boundWindowPointerDown = (ev) => this._handleWindowPointerDown(ev);
      this._pointerDownAttached = false;

      // painting
      this._dragging = false;
      this._paintValue = null;
      this._selectedMode = 0;
      this._activePointerId = null;
      this._windowPointerTracking = false;
      this._boundWindowPointerUp = (ev) => this._handleWindowPointerFinish(ev);

      // Available TermoWeb heater entities
      this._entities = [];
      this._entityOptionsKey = "";

      // Track copy selectors
      this._copyFrom = 0;
      this._copyTo = "All";
      this._entity = null; // currently selected entity

      this._hasRendered = false;

      this._els = {
        title: null,
        entitySelect: null,
        dirtyBadge: null,
        freezeBadge: null,
        unitsLabel: null,
        presetInputs: { cold: null, night: null, day: null },
        modeButtons: { cold: null, night: null, day: null },
        refreshBtn: null,
        copyFromSel: null,
        copyToSel: null,
        copyBtn: null,
        revertBtn: null,
        saveBtn: null,
        progWarn: null,
      };
      this._gridCells = null;

    }

    connectedCallback() {
      if (super.connectedCallback) super.connectedCallback();
      if (!this._pointerDownAttached) {
        window.addEventListener("pointerdown", this._boundWindowPointerDown, true);
        this._pointerDownAttached = true;
      }
    }

    disconnectedCallback() {
      if (this._pointerDownAttached) {
        window.removeEventListener("pointerdown", this._boundWindowPointerDown, true);
        this._pointerDownAttached = false;
      }
      if (super.disconnectedCallback) super.disconnectedCallback();
    }

    setConfig(config) {
      if (!config) {
        throw new Error("termoweb-schedule-card: invalid configuration");
      }
      this._config = config;
      this._entity = config.entity || null;
      this._render();
    }

    set hass(hass) {
      this._hass = hass;
      if (!this._config) return;

      const prevEntity = this._entity;

      // Collect available TermoWeb heater entities
      const entities = Object.entries(hass.states)
        .filter(([eid, st]) => {
          if (!eid.startsWith("climate.")) return false;
          const a = st?.attributes || {};
          return Array.isArray(a.prog) && a.prog.length === 168;
        })
        .map(([eid, st]) => ({
          id: eid,
          name: st.attributes?.friendly_name || st.attributes?.name || eid,
        }))
        .sort((a, b) => {
          const nameCmp = (a.name || "").localeCompare(b.name || "");
          if (nameCmp !== 0) return nameCmp;
          return a.id.localeCompare(b.id);
        });

      this._entities = entities;

      const prevEntityOptionsKey = this._entityOptionsKey;
      const nextEntityOptionsKey = JSON.stringify(entities.map((e) => `${e.id}|${e.name}`));
      this._entityOptionsKey = nextEntityOptionsKey;

      if (!this._entity && this._entities.length > 0) {
        this._entity = this._entities[0].id;
        if (this._config) this._config.entity = this._entity;
      }

      const st = this._entity ? hass.states[this._entity] : undefined;
      this._stateObj = st || null;

      const attrs = st?.attributes || {};
      const now = nowMs();

      let waitingForEcho = false;
      if (this._pendingEcho.prog) {
        if (Array.isArray(attrs.prog) && attrs.prog.length === 168 && deepEqArray(attrs.prog, this._pendingEcho.prog)) {
          this._pendingEcho.prog = null;
        } else {
          waitingForEcho = true;
        }
      }
      if (this._pendingEcho.ptemp) {
        if (Array.isArray(attrs.ptemp) && attrs.ptemp.length === 3 && deepEqArray(attrs.ptemp, this._pendingEcho.ptemp)) {
          this._pendingEcho.ptemp = null;
        } else {
          waitingForEcho = true;
        }
      }

      if (!waitingForEcho) {
        this._freezeUntil = 0;
      }

      const freezeActive = waitingForEcho || (now < this._freezeUntil);
      const canHydrateNow = this._canHydrateFromState({ freezeActive });

      const progValid = Array.isArray(attrs.prog) && attrs.prog.length === 168;
      const ptempValid = Array.isArray(attrs.ptemp) && attrs.ptemp.length === 3;

      let hydrated = false;
      const discardProg = () => {
        this._progLocal = null;
        this._dirtyProg = false;
        this._pendingEcho.prog = null;
        this._freezeUntil = 0;
        hydrated = true;
      };
      const discardPtemp = () => {
        this._ptempLocal = [null, null, null];
        this._dirtyPresets = false;
        this._pendingEcho.ptemp = null;
        this._freezeUntil = 0;
        hydrated = true;
      };

      if (canHydrateNow) {
        if (progValid) {
          if (!Array.isArray(this._progLocal) || !deepEqArray(this._progLocal, attrs.prog)) {
            this._progLocal = attrs.prog.slice();
            hydrated = true;
          }
        } else {
          discardProg();
        }
        if (ptempValid) {
          if (!Array.isArray(this._ptempLocal) || !deepEqArray(this._ptempLocal, attrs.ptemp)) {
            this._ptempLocal = attrs.ptemp.slice();
            hydrated = true;
          }
        } else {
          discardPtemp();
        }
      } else {
        if (!progValid) discardProg();
        if (!ptempValid) discardPtemp();
      }

      const entityChanged = prevEntity !== this._entity;
      const entityOptionsChanged = prevEntityOptionsKey !== nextEntityOptionsKey;
      if (!this._hasRendered || entityChanged || entityOptionsChanged || hydrated) {
        this._render({ forceFull: !this._hasRendered || entityChanged });
      } else {
        this._updateStatusIndicators();
        this._restorePresetFocusIfNeeded();
      }
    }

    _canHydrateFromState({ freezeActive } = {}) {
      // Only hydrate when:
      // - No local copy yet (first load)
      // - Not currently dirty
      // - Not within freeze window / awaiting echo
      const now = nowMs();
      const inFreeze = freezeActive != null ? freezeActive : (now < this._freezeUntil);
      const hasProgLocal = Array.isArray(this._progLocal) && this._progLocal.length === 168;
      const hasPresetLocal = Array.isArray(this._ptempLocal) && this._ptempLocal.length === 3;
      const hasAnyLocal = hasProgLocal || hasPresetLocal;
      if (!hasAnyLocal) return true;
      if (this._editingPresetIdx !== -1) return false;
      if (this._dirtyProg || this._dirtyPresets) return false;
      if (inFreeze) return false;
      return true;
    }

    _isFrozen() {
      if (this._pendingEcho.prog || this._pendingEcho.ptemp) return true;
      return nowMs() < this._freezeUntil;
    }

    _updateStatusIndicators() {
      if (!this._hasRendered) return;
      const dirty = this._dirtyProg || this._dirtyPresets;
      const dirtyEl = this._els.dirtyBadge;
      if (dirtyEl) dirtyEl.hidden = !dirty;

      const waiting = this._isFrozen();
      const waitEl = this._els.freezeBadge;
      if (waitEl) waitEl.hidden = !waiting;
    }

    _syncEntityOptions() {
      const select = this._els.entitySelect;
      if (!select) return;
      const optionsKey = this._entityOptionsKey || "";
      if (select._twOptionsKey !== optionsKey) {
        select.innerHTML = (this._entities || [])
          .map((e) => `<option value="${e.id}">${e.name}</option>`)
          .join("");
        select._twOptionsKey = optionsKey;
      }
      if (this._entity && select.value !== this._entity) {
        select.value = this._entity;
      }
    }

    _updateModeButtons() {
      const buttons = this._els.modeButtons;
      if (!buttons) return;
      if (buttons.cold) buttons.cold.classList.toggle("active", this._selectedMode === 0);
      if (buttons.night) buttons.night.classList.toggle("active", this._selectedMode === 1);
      if (buttons.day) buttons.day.classList.toggle("active", this._selectedMode === 2);
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
      this._updateStatusIndicators();
    }
    _handleWindowPointerFinish(ev) {
      this._onPointerUp(ev);
    }
    _onPointerDown(ev, day, hour) {
      if (!this._progLocal) return;
      if (this._activePointerId != null && this._activePointerId !== ev.pointerId) return;
      this._dragging = true;
      this._activePointerId = ev.pointerId;
      const i = this._idx(day, hour);
      const next = this._selectedMode;
      this._paintValue = next;
      this._progLocal[i] = next;
      this._dirtyProg = true;
      const target = ev.currentTarget;
      if (target?.setPointerCapture) {
        try {
          target.setPointerCapture(ev.pointerId);
          if (target?.releasePointerCapture) target.releasePointerCapture(ev.pointerId);
        } catch (err) { /* no-op */ }
      }
      if (!this._windowPointerTracking) {
        window.addEventListener("pointerup", this._boundWindowPointerUp, true);
        window.addEventListener("pointercancel", this._boundWindowPointerUp, true);
        this._windowPointerTracking = true;
      }
      this._renderGridOnly();
      this._updateStatusIndicators();
      ev.preventDefault();
    }
    _onPointerEnter(ev, day, hour) {
      if (!this._dragging || this._paintValue == null || !this._progLocal) return;
      if (ev.pointerId !== this._activePointerId) return;
      const i = this._idx(day, hour);
      if (this._progLocal[i] !== this._paintValue) {
        this._progLocal[i] = this._paintValue;
        this._dirtyProg = true;
        this._colorCell(day, hour, this._paintValue);
        this._updateStatusIndicators();
      }
    }
    _onPointerUp(ev) {
      if (this._activePointerId !== ev.pointerId) return;
      const target = ev.currentTarget;
      if (target?.hasPointerCapture && target.hasPointerCapture(ev.pointerId)) {
        target.releasePointerCapture(ev.pointerId);
      } else if (target?.releasePointerCapture) {
        try { target.releasePointerCapture(ev.pointerId); } catch (err) { /* no-op */ }
      }
      this._dragging = false;
      this._paintValue = null;
      this._activePointerId = null;
      if (this._windowPointerTracking) {
        window.removeEventListener("pointerup", this._boundWindowPointerUp, true);
        window.removeEventListener("pointercancel", this._boundWindowPointerUp, true);
        this._windowPointerTracking = false;
      }
    }

    _revert() {
      // Force re-hydrate from current HA state
      const st = this._hass?.states?.[this._entity];
      const attrs = st?.attributes || {};
      if (Array.isArray(attrs.prog) && attrs.prog.length === 168) {
        this._progLocal = attrs.prog.slice();
      } else {
        this._progLocal = null;
      }
      if (Array.isArray(attrs.ptemp) && attrs.ptemp.length === 3) {
        this._ptempLocal = attrs.ptemp.slice();
      } else {
        this._ptempLocal = [null, null, null];
      }
      this._dirtyProg = false;
      this._dirtyPresets = false;
      this._editingPresetIdx = -1;
      this._pendingPresetFocusRestore = false;
      this._presetFocusRelease = false;
      this._presetSelection = null;
      this._freezeUntil = 0;
      this._pendingEcho = { prog: null, ptemp: null };
      this._render();
    }

    _refreshFromState() {
      // Manual refresh, ignoring freeze; useful if user wants to sync now
      const st = this._hass?.states?.[this._entity];
      const attrs = st?.attributes || {};
      if (Array.isArray(attrs.prog) && attrs.prog.length === 168) {
        this._progLocal = attrs.prog.slice();
      } else {
        this._progLocal = null;
      }
      if (Array.isArray(attrs.ptemp) && attrs.ptemp.length === 3) {
        this._ptempLocal = attrs.ptemp.slice();
      } else {
        this._ptempLocal = [null, null, null];
      }
      this._dirtyProg = false;
      this._dirtyPresets = false;
      this._editingPresetIdx = -1;
      this._pendingPresetFocusRestore = false;
      this._presetFocusRelease = false;
      this._presetSelection = null;
      this._freezeUntil = 0;
      this._pendingEcho = { prog: null, ptemp: null };
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
          entity_id: this._entity,
          ptemp: payload.slice(),
        });
        this._ptempLocal = payload.slice();
        this._dirtyPresets = false;
        this._editingPresetIdx = -1;
        this._pendingPresetFocusRestore = false;
        this._presetFocusRelease = false;
        this._presetSelection = null;
        this._pendingEcho.ptemp = payload.slice();
        this._freezeUntil = nowMs() + this._freezeWindowMs;
        this._toast("Preset temperatures sent (waiting for device to update)");
        this._updateStatusIndicators();
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
          entity_id: this._entity,
          prog: body,
        });
        this._dirtyProg = false;
        this._pendingEcho.prog = body.slice();
        this._freezeUntil = nowMs() + this._freezeWindowMs;
        this._toast("Schedule sent (waiting for device to update)");
        this._updateStatusIndicators();
      } catch (e) {
        this._toast("Failed to save schedule");
        console.error("TermoWeb card: set_schedule error:", e);
      }
    }

    // ---------- render ----------
    _render({ forceFull = false } = {}) {
      const root = this.shadowRoot;
      if (!root) return;

      const title =
        (this._stateObj?.attributes?.friendly_name || this._stateObj?.attributes?.name) ||
        this._entity || "TermoWeb schedule";

      const hasProg = Array.isArray(this._progLocal) && this._progLocal.length === 168;
      const units = this._units();
      const stepAttr = units === "F" ? "1" : "0.5";
      const [cold, night, day] = this._ptempLocal ?? [null, null, null];

      if (!this._hasRendered || forceFull) {
        if (forceFull) {
          this._editingPresetIdx = -1;
        }
        const copyOptions = DAY_NAMES.map((d, i) => `<option value="${i}">${d}</option>`).join("");
        const gridShell = this._renderGridShell();

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
              <div id="tw_title"></div>
              <div class="sub">
                <select id="entitySelect"></select>
                <span id="tw_dirtyBadge" class="dirty" hidden>● unsaved</span>
                <span id="tw_freezeBadge" class="chip" hidden>waiting for device update…</span>
                <button id="refreshBtn" title="Refresh from current state">Refresh</button>
              </div>
            </div>

            <div class="legend">
              <span><span class="swatch" style="background:${COLORS[0]}"></span>Cold</span>
              <span><span class="swatch" style="background:${COLORS[2]}"></span>Day</span>
              <span><span class="swatch" style="background:${COLORS[1]}"></span>Night</span>
              <span id="tw_units"></span>
            </div>

            <div class="row">
              <label>Cold <input id="tw_p_cold" type="number"></label>
              <label>Night <input id="tw_p_night" type="number"></label>
              <label>Day <input id="tw_p_day" type="number"></label>
              <button id="savePresetsBtn">Save Presets</button>
            </div>

            <div class="modeToggle">
              <button id="modeCold" style="background:${COLORS[0]}">Cold</button>
              <button id="modeNight" style="background:${COLORS[1]}">Night</button>
              <button id="modeDay" style="background:${COLORS[2]}">Day</button>
            </div>

            <div id="tw_prog_warn" class="warn" style="margin-top:8px;" hidden>This entity has no valid 'prog' (expected 168 ints).</div>

            ${gridShell}

            <div class="row">
              <label>Copy From <select id="copyFromSel">${copyOptions}</select></label>
              <label>Copy To <select id="copyToSel"><option value="All">All</option>${copyOptions}</select></label>
              <button id="copyBtn">Copy</button>
            </div>

            <div class="footer">
              <button id="revertBtn">Revert</button>
              <button id="saveBtn">Save</button>
            </div>
          </ha-card>
        `;

        this._els = {
          title: root.getElementById("tw_title"),
          entitySelect: root.getElementById("entitySelect"),
          dirtyBadge: root.getElementById("tw_dirtyBadge"),
          freezeBadge: root.getElementById("tw_freezeBadge"),
          unitsLabel: root.getElementById("tw_units"),
          presetInputs: {
            cold: root.getElementById("tw_p_cold"),
            night: root.getElementById("tw_p_night"),
            day: root.getElementById("tw_p_day"),
          },
          modeButtons: {
            cold: root.getElementById("modeCold"),
            night: root.getElementById("modeNight"),
            day: root.getElementById("modeDay"),
          },
          refreshBtn: root.getElementById("refreshBtn"),
          copyFromSel: root.getElementById("copyFromSel"),
          copyToSel: root.getElementById("copyToSel"),
          copyBtn: root.getElementById("copyBtn"),
          revertBtn: root.getElementById("revertBtn"),
          saveBtn: root.getElementById("saveBtn"),
          progWarn: root.getElementById("tw_prog_warn"),
        };

        this._gridCells = Array.from(root.querySelectorAll(".cell"));

        this._els.refreshBtn?.addEventListener("click", () => this._refreshFromState());

        this._els.entitySelect?.addEventListener("change", (ev) => {
          const newEntity = ev.target.value;
          if (newEntity && newEntity !== this._entity) {
            this._entity = newEntity;
            this._config.entity = newEntity;
            this._stateObj = this._hass?.states?.[newEntity] || null;
            this._revert();
          }
        });

        this._bindPresetInput(this._els.presetInputs.cold, 0);
        this._bindPresetInput(this._els.presetInputs.night, 1);
        this._bindPresetInput(this._els.presetInputs.day, 2);

        root.getElementById("savePresetsBtn")?.addEventListener("click", () => this._savePresets());

        this._els.modeButtons.cold?.addEventListener("click", () => { this._selectedMode = 0; this._render(); });
        this._els.modeButtons.night?.addEventListener("click", () => { this._selectedMode = 1; this._render(); });
        this._els.modeButtons.day?.addEventListener("click", () => { this._selectedMode = 2; this._render(); });

        this._els.revertBtn?.addEventListener("click", () => this._revert());
        this._els.saveBtn?.addEventListener("click", () => this._saveSchedule());

        this._els.copyFromSel?.addEventListener("change", (ev) => {
          this._copyFrom = Number(ev.target.value);
        });
        this._els.copyToSel?.addEventListener("change", (ev) => {
          const v = ev.target.value;
          this._copyTo = v === "All" ? "All" : Number(v);
        });

        this._els.copyBtn?.addEventListener("click", () => {
          this._copyDay(this._copyFrom, this._copyTo);
          this._dirtyProg = true;
          this._renderGridOnly();
          this._updateStatusIndicators();
        });

        this._hasRendered = true;
      }

      if (this._els.title) this._els.title.textContent = title;

      this._syncEntityOptions();

      if (this._els.unitsLabel) this._els.unitsLabel.textContent = `Units: ${units}`;

      const presetInputs = this._els.presetInputs;
      if (presetInputs) {
        const presetValues = [cold, night, day];
        [presetInputs.cold, presetInputs.night, presetInputs.day].forEach((inputEl, idx) => {
          if (!inputEl) return;
          if (inputEl.step !== stepAttr) inputEl.step = stepAttr;
          const target = presetValues[idx];
          if (this._editingPresetIdx === idx) return;
          const valueStr = (target === null || target === undefined) ? "" : String(target);
          if (inputEl.value !== valueStr) inputEl.value = valueStr;
        });
      }

      if (this._els.copyFromSel) this._els.copyFromSel.value = String(this._copyFrom);
      if (this._els.copyToSel) this._els.copyToSel.value = this._copyTo === "All" ? "All" : String(this._copyTo);

      if (this._els.progWarn) this._els.progWarn.hidden = hasProg;

      this._updateModeButtons();
      this._renderGridOnly();
      this._updateStatusIndicators();
      this._restorePresetFocusIfNeeded();
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
      if (!Array.isArray(this._gridCells) || !this._progLocal || this._progLocal.length !== 168) return;

      this._gridCells.forEach((cell) => {
        const d = Number(cell.getAttribute("data-d"));
        const h = Number(cell.getAttribute("data-h"));
        const idx = this._idx(d, h);
        const v = Number(this._progLocal[idx] ?? 0);

        cell.style.background = COLORS[v in COLORS ? v : 0];

        if (!cell._twBound) {
          cell._twBound = true;
          cell.addEventListener("click", () => this._onCellClick(d, h));
          cell.addEventListener("pointerdown", (ev) => this._onPointerDown(ev, d, h));
          cell.addEventListener("pointerenter", (ev) => this._onPointerEnter(ev, d, h));
          cell.addEventListener("pointerup", (ev) => this._onPointerUp(ev));
          cell.addEventListener("pointercancel", (ev) => this._onPointerUp(ev));
        }
      });
    }

    _colorCell(day, hour, v) {
      const root = this.shadowRoot;
      const el = root && root.querySelector(`.cell[data-d="${day}"][data-h="${hour}"]`);
      if (el) el.style.background = COLORS[v in COLORS ? v : 0];
    }

    _presetInputsArray() {
      const inputs = this._els?.presetInputs;
      if (!inputs) return [];
      return [inputs.cold, inputs.night, inputs.day];
    }

    _presetInputByIndex(idx) {
      const arr = this._presetInputsArray();
      return Number.isInteger(idx) && idx >= 0 && idx < arr.length ? arr[idx] : null;
    }

    _bindPresetInput(inputEl, idx) {
      if (!inputEl || inputEl._twPresetBound) return;
      inputEl._twPresetBound = true;

      const updateLocal = () => {
        const raw = inputEl.value;
        const n = Number(raw);
        this._ptempLocal[idx] = Number.isFinite(n) ? n : null;
      };

      const captureSelection = () => {
        this._capturePresetSelection(idx);
      };

      inputEl.addEventListener("input", () => {
        updateLocal();
        this._dirtyPresets = true;
        this._pendingPresetFocusRestore = false;
        captureSelection();
        this._updateStatusIndicators();
      });

      inputEl.addEventListener("focus", () => {
        this._handlePresetFocus(idx);
        captureSelection();
      });

      inputEl.addEventListener("blur", () => {
        captureSelection();
        this._handlePresetBlur(idx);
      });

      inputEl.addEventListener("click", captureSelection);
      inputEl.addEventListener("keyup", captureSelection);
      inputEl.addEventListener("keydown", (ev) => {
        if (ev.key === "Tab") {
          this._presetFocusRelease = true;
        }
      });
    }

    _capturePresetSelection(idx) {
      if (this._editingPresetIdx !== idx) return;
      const inputEl = this._presetInputByIndex(idx);
      if (!inputEl) return;
      let start = null;
      let end = null;
      let dir = "none";
      try {
        start = inputEl.selectionStart;
        end = inputEl.selectionEnd;
        dir = inputEl.selectionDirection || "none";
      } catch (e) {
        // selection metadata unavailable
      }
      if (start == null || end == null) {
        const len = inputEl.value != null ? String(inputEl.value).length : 0;
        start = len;
        end = len;
      }
      this._presetSelection = { start, end, dir };
    }

    _handlePresetFocus(idx) {
      this._editingPresetIdx = idx;
      this._presetFocusRelease = false;
      this._pendingPresetFocusRestore = false;
    }

    _handlePresetBlur(idx) {
      if (this._presetFocusRelease) {
        if (this._editingPresetIdx === idx) {
          this._editingPresetIdx = -1;
        }
        this._presetFocusRelease = false;
        this._pendingPresetFocusRestore = false;
        return;
      }
      if (this._editingPresetIdx === idx) {
        this._pendingPresetFocusRestore = true;
        this._schedulePresetFocusRestore();
      }
    }

    _schedulePresetFocusRestore() {
      if (!this._pendingPresetFocusRestore) return;
      if (this._restoreFocusRaf != null) return;
      const raf = window?.requestAnimationFrame || ((cb) => setTimeout(cb, 16));
      this._restoreFocusRaf = raf(() => {
        this._restoreFocusRaf = null;
        this._restorePresetFocusIfNeeded();
      });
    }

    _restorePresetFocusIfNeeded() {
      const idx = this._editingPresetIdx;
      if (!Number.isInteger(idx) || idx < 0) {
        this._pendingPresetFocusRestore = false;
        return;
      }
      if (this._presetFocusRelease) return;
      const inputEl = this._presetInputByIndex(idx);
      if (!inputEl) {
        this._pendingPresetFocusRestore = false;
        return;
      }
      const active = this.shadowRoot?.activeElement;
      if (active === inputEl) {
        this._pendingPresetFocusRestore = false;
        return;
      }
      if (!this._pendingPresetFocusRestore && active && active !== inputEl) {
        return;
      }
      this._pendingPresetFocusRestore = false;
      try {
        inputEl.focus({ preventScroll: true });
      } catch (e) {
        try { inputEl.focus(); } catch (_) { /* ignore */ }
      }
      const sel = this._presetSelection;
      if (sel && sel.start != null && sel.end != null && inputEl.setSelectionRange) {
        try {
          inputEl.setSelectionRange(sel.start, sel.end, sel.dir || "none");
        } catch (e) {
          // ignore browsers that disallow selection updates
        }
        this._presetSelection = { start: sel.start, end: sel.end, dir: sel.dir || "none" };
      }
    }

    _handleWindowPointerDown(ev) {
      if (!Number.isInteger(this._editingPresetIdx) || this._editingPresetIdx < 0) return;
      const current = this._presetInputByIndex(this._editingPresetIdx);
      if (!current) return;
      const path = typeof ev.composedPath === "function" ? ev.composedPath() : [];
      if (path.includes(current)) {
        this._presetFocusRelease = false;
      } else {
        this._presetFocusRelease = true;
      }
    }

    static getConfigElement() { return null; }
  }

  customElements.define("termoweb-schedule-card", TermoWebScheduleCard);
})();