
/* TermoWeb Schedule Card — Vanilla focus-safe build (Copy-to-Entity)
 * - Clear label before mode buttons: "Set hourly slot to:"
 * - Larger mode buttons
 * - Legend chips removed; only Units label
 * - NEW: Copy everything (prog + presets) to another heater entity, switch selection, DO NOT save.
 *   UI is below the day-copy section.
 * - Services:
 *     termoweb.set_schedule
 *     termoweb.set_preset_temperatures
 */
(function () {
  const DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const MODE = { COLD: 0, NIGHT: 1, DAY: 2 };
  const COLORS = { 0: "#00BCD4", 1: "#0D47A1", 2: "#FB8C00" };
  function clamp(n, min, max) { return Math.max(min, Math.min(max, n)); }
  function hourIdx(day, hour) { return day * 24 + hour; }

  class TermoWebScheduleCard extends HTMLElement {
    constructor() {
      super();
      this._hass = null; this._config = null;
      this._entity = null; this._entities = []; this._entitiesKey = "";
      this._stateObj = null;
      this._progLocal = null;
      this._ptempLocal = [null, null, null];
      this._presetInvalid = [false, false, false];
      this._presetFeedback = "";
      this._dirtyProg = false; this._dirtyPresets = false;
      this._savingProg = false; this._savingPresets = false;
      this._selectedMode = MODE.COLD;
      this._copyFrom = 0; this._copyTo = "All";
      this._units = "C";
      this._isDragging = false; this._isInteracting = false;
      this._copyEntityTarget = null; // selected target in "Copy to heater"
      this._statusWarn = "";
      this._els = {}; this._built = false;

      this._onDocPointerUp = () => this._stopDrag();
      this._onFocusIn = () => { this._isInteracting = true; };
      this._onFocusOut = () => setTimeout(() => {
        const root = this; this._isInteracting = root.contains(document.activeElement);
      }, 0);
    }
    connectedCallback() {
      document.addEventListener("pointerup", this._onDocPointerUp, true);
      this.addEventListener("focusin", this._onFocusIn, true);
      this.addEventListener("focusout", this._onFocusOut, true);
    }
    disconnectedCallback() {
      document.removeEventListener("pointerup", this._onDocPointerUp, true);
      this.removeEventListener("focusin", this._onFocusIn, true);
      this.removeEventListener("focusout", this._onFocusOut, true);
    }

    setConfig(config) { this._config = { ...config }; if (config.entity) this._entity = config.entity; if (!this._built) this._build(); }
    static getConfigElement() { return null; }
    static getStubConfig() { return {}; }
    getCardSize() { return 16; }

    set hass(hass) {
      this._hass = hass; if (!hass || !this._config) return;
      if (!this._built) this._build();

      const entities = Object.entries(hass.states)
        .filter(([eid, st]) => this._isTermoWebCandidate(eid, st))
        .map(([eid, st]) => ({ id: eid, name: st.attributes.friendly_name || st.attributes.name || eid }))
        .sort((a,b) => a.name.localeCompare(b.name));
      if (this._config.entity && hass.states[this._config.entity] && !entities.find((e) => e.id === this._config.entity)) {
        const st = hass.states[this._config.entity];
        entities.unshift({
          id: this._config.entity,
          name: st.attributes.friendly_name || st.attributes.name || this._config.entity,
        });
      }
      const entKey = JSON.stringify(entities.map(e => e.id + "|" + e.name));
      if (entKey !== this._entitiesKey) {
        this._entitiesKey = entKey; this._entities = entities;
        this._syncEntityOptions();  // now updates both entitySelect and copyEntitySel when not focused
        // Set default copy target to first other entity if not set
        if (!this._copyEntityTarget || !this._entities.find(e => e.id === this._copyEntityTarget)) {
          const alt = this._entities.find(e => e.id !== this._entity);
          this._copyEntityTarget = alt ? alt.id : (this._entities[0]?.id || null);
        }
        this._syncCopyEntitySelect();
      }
      if (!this._entity && this._entities.length) this._entity = this._entities[0].id;

      const st = this._entity ? hass.states[this._entity] : undefined; this._stateObj = st;
      const u = st?.attributes?.units; this._units = (u === "F" || u === "C") ? u : "C";

      if (!this._isInteracting && !this._dirtyProg) {
        const p = st?.attributes?.prog; this._progLocal = (Array.isArray(p) && p.length === 168) ? p.slice() : null;
        this._renderGridColors();
      }
      if (!this._isInteracting && !this._dirtyPresets) {
        const t = st?.attributes?.ptemp; this._ptempLocal = this._normalizePresetTriplet(t);
        this._presetInvalid = [false, false, false];
        this._presetFeedback = "";
        this._syncPresetInputs();
        this._syncPresetValidationUI();
      }

      this._syncHeader(); this._syncCopySelectors(); this._syncWarn(); this._syncModeButtons();
      this._syncCopyEntitySelect();
    }

    _matchesIncludePattern(entityId) {
      const pattern = this._config?.include_pattern;
      if (!pattern) return true;
      try {
        return new RegExp(pattern).test(entityId);
      } catch (err) {
        console.warn("termoweb-schedule-card: invalid include_pattern regex", pattern, err);
        return true;
      }
    }

    _hasTermoWebMarkers(attrs) {
      if (!attrs || typeof attrs !== "object") return false;
      const hasIdentity = typeof attrs.dev_id === "string" && attrs.dev_id.length > 0 && attrs.addr != null;
      const hasScheduleShape = Array.isArray(attrs.ptemp) && attrs.ptemp.length === 3;
      return hasIdentity && hasScheduleShape;
    }

    _isTermoWebCandidate(entityId, stateObj) {
      if (!entityId?.startsWith("climate.")) return false;
      if (!this._matchesIncludePattern(entityId)) return false;
      const attrs = stateObj?.attributes;
      if (this._hasTermoWebMarkers(attrs)) return true;
      return Array.isArray(attrs?.prog) && attrs.prog.length === 168;
    }

    _build() {
      this._built = true;
      this.innerHTML = "";
      const card = document.createElement("ha-card"); card.className = "tw-card";
      const style = document.createElement("style"); style.textContent = `
        .card{padding:12px;color:var(--primary-text-color)}
        .header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;font-weight:600}
        .sub{color:var(--secondary-text-color);font-size:12px;display:flex;align-items:center;gap:8px}
        .dirty{color:var(--warning-color,#ffa000);font-size:11px}
        .grid{display:grid;grid-template-columns:56px repeat(7,1fr);gap:6px;margin-top:8px}
        .hour{color:var(--secondary-text-color);font-size:12px;text-align:right;padding:4px 6px}
        .dayhdr{color:var(--secondary-text-color);font-size:12px;text-align:center;padding:4px 0 8px 0}
        .cell{background:var(--card-background-color);border:1px solid var(--divider-color);height:20px;border-radius:6px;cursor:pointer;transition:filter .06s}
        .cell:hover{filter:brightness(1.08)}
        .legend{display:flex;gap:12px;align-items:center;flex-wrap:wrap;color:var(--secondary-text-color);font-size:12px}
        .unitsChip{font-size:12px;color:var(--secondary-text-color)}
        .row{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap;color:var(--secondary-text-color)}
        .modeToggle{display:flex;gap:10px;margin-top:10px;align-items:center;flex-wrap:wrap}
        .modeLabel{color:var(--secondary-text-color);font-size:13px;margin-right:4px}
        .modeToggle button{border:none;color:#fff;padding:10px 14px;border-radius:12px;cursor:pointer;opacity:.95;font-weight:600;font-size:14px;min-width:70px}
        .modeToggle button.active{filter:brightness(.95);box-shadow:inset 0 0 0 2px #ffffff33;opacity:1}
        .preset-feedback{font-size:12px;color:var(--error-color,#db4437);min-height:16px;margin-top:4px}
        .input-invalid{border-color:var(--error-color,#db4437)!important}
        .warn{color:var(--warning-color);font-size:12px;margin-top:8px}
        .top-right{display:flex;gap:8px;align-items:center}
        .entity-select{min-width:160px}
        .wide-btn{min-width:96px}
        select,input[type=number]{background:var(--card-background-color);color:var(--primary-text-color);border:1px solid var(--divider-color);border-radius:6px;padding:4px 8px}
        button{background:var(--primary-color);color:#fff;border:none;border-radius:6px;padding:6px 10px;cursor:pointer}
        .ghost{opacity:.6;pointer-events:none}
        .btn-muted{opacity:.75}
      `;
      card.appendChild(style);
      const wrap = document.createElement("div"); wrap.className = "card"; card.appendChild(wrap);
      const header = document.createElement("div"); header.className = "header"; wrap.appendChild(header);
      const titleEl = document.createElement("div"); header.appendChild(titleEl);
      const right = document.createElement("div"); right.className = "top-right"; header.appendChild(right);
      const entitySelect = document.createElement("select"); entitySelect.className = "entity-select";
      entitySelect.addEventListener("change", () => { const v = entitySelect.value; if (v && v !== this._entity) { this._entity = v; this._dirtyProg = false; this._dirtyPresets = false; this.hass = this._hass; }});
      right.appendChild(entitySelect);
      const dirtyBadge = document.createElement("span"); dirtyBadge.className = "dirty"; dirtyBadge.textContent = "● unsaved"; right.appendChild(dirtyBadge);
      const refreshBtn = document.createElement("button"); refreshBtn.className = "wide-btn"; refreshBtn.textContent = "Refresh"; refreshBtn.addEventListener("click",()=>this._refresh()); right.appendChild(refreshBtn);

      // Legend (now only Units)
      const legend = document.createElement("div"); legend.className = "legend";
      legend.innerHTML = `<span id="tw_units" class="unitsChip">Units: –</span>`;
      wrap.appendChild(legend);

      // Presets row
      const presetRow = document.createElement("div"); presetRow.className="row";
      presetRow.innerHTML = `<label>Cold <input id="tw_p_cold" type="number"></label>
        <label>Night <input id="tw_p_night" type="number"></label>
        <label>Day <input id="tw_p_day" type="number"></label>`;
      const savePresetsBtn = document.createElement("button"); savePresetsBtn.textContent="Save Presets"; savePresetsBtn.addEventListener("click",()=>void this._savePresets()); presetRow.appendChild(savePresetsBtn); wrap.appendChild(presetRow);
      const presetFeedback = document.createElement("div"); presetFeedback.className="preset-feedback"; wrap.appendChild(presetFeedback);

      // Mode toggle with label
      const modeRow = document.createElement("div"); modeRow.className="modeToggle";
      const modeLabel = document.createElement("span"); modeLabel.className = "modeLabel"; modeLabel.textContent = "Set hourly slot to:";
      const modeCold = document.createElement("button"); modeCold.textContent="Cold"; modeCold.style.background=COLORS[0]; modeCold.addEventListener("click",()=>this._setMode(MODE.COLD));
      const modeNight = document.createElement("button"); modeNight.textContent="Night"; modeNight.style.background=COLORS[1]; modeNight.addEventListener("click",()=>this._setMode(MODE.NIGHT));
      const modeDay = document.createElement("button"); modeDay.textContent="Day"; modeDay.style.background=COLORS[2]; modeDay.addEventListener("click",()=>this._setMode(MODE.DAY));
      modeRow.append(modeLabel, modeCold, modeNight, modeDay); wrap.appendChild(modeRow);

      const warn = document.createElement("div"); warn.className="warn"; wrap.appendChild(warn);

      // Grid
      const grid = document.createElement("div"); grid.className="grid";
      grid.appendChild(document.createElement("div"));
      for (let d=0; d<7; d++){ const dh=document.createElement("div"); dh.className="dayhdr"; dh.textContent=DAY_NAMES[d]; grid.appendChild(dh); }
      const cells = [];
      for (let h=0; h<24; h++){ const hl=document.createElement("div"); hl.className="hour"; hl.textContent=String(h).padStart(2,"0")+":00"; grid.appendChild(hl);
        for (let d=0; d<7; d++){ const cell=document.createElement("div"); cell.className="cell"; cell.dataset.day=String(d); cell.dataset.hour=String(h);
          cell.addEventListener("pointerdown",(ev)=>{ev.preventDefault();this._startDrag();this._paintCell(d,h);});
          cell.addEventListener("pointerenter",()=>{if(this._isDragging)this._paintCell(d,h);});
          grid.appendChild(cell); cells.push(cell); }}
      grid.addEventListener("pointerleave",()=>this._stopDrag()); wrap.appendChild(grid);

      // Copy day row
      const copyRow = document.createElement("div"); copyRow.className="row";
      copyRow.innerHTML = `<label>Copy From <select id="copyFromSel">${DAY_NAMES.map((d,i)=>`<option value="${i}">${d}</option>`).join("")}</select></label>
        <label>Copy To <select id="copyToSel"><option value="All">All</option>${DAY_NAMES.map((d,i)=>`<option value="${i}">${d}</option>`).join("")}</select></label>`;
      const copyBtn=document.createElement("button"); copyBtn.textContent="Copy"; copyBtn.addEventListener("click",()=>this._copyDays()); copyRow.appendChild(copyBtn); wrap.appendChild(copyRow);

      // Copy EVERYTHING to another heater
      const copyAllRow = document.createElement("div"); copyAllRow.className="row";
      copyAllRow.innerHTML = `<label>Copy everything to heater
          <select id="copyEntitySel"></select>
        </label>`;
      const copyAllBtn = document.createElement("button"); copyAllBtn.textContent = "Copy to Entity";
      copyAllBtn.addEventListener("click", () => this._copyToEntity());
      copyAllRow.appendChild(copyAllBtn);
      wrap.appendChild(copyAllRow);

      // Footer
      const footer=document.createElement("div"); footer.className="row";
      const revertBtn=document.createElement("button"); revertBtn.textContent="Revert"; revertBtn.addEventListener("click",()=>this._revert());
      const saveBtn=document.createElement("button"); saveBtn.textContent="Save"; saveBtn.addEventListener("click",()=>void this._save()); footer.append(revertBtn,saveBtn); wrap.appendChild(footer);

      // Refs
      this._els = {
        card, titleEl, entitySelect, dirtyBadge, refreshBtn,
        unitsLabel: legend.querySelector("#tw_units"),
        presetCold: presetRow.querySelector("#tw_p_cold"),
        presetNight: presetRow.querySelector("#tw_p_night"),
        presetDay: presetRow.querySelector("#tw_p_day"),
        savePresetsBtn,
        presetFeedback,
        modeCold, modeNight, modeDay, warn, grid, cells,
        copyFromSel: copyRow.querySelector("#copyFromSel"),
        copyToSel: copyRow.querySelector("#copyToSel"),
        copyEntitySel: copyAllRow.querySelector("#copyEntitySel"),
        revertBtn,
        saveBtn,
      };

      // Preset input handlers
      const presetHandler = (idx) => (ev)=>{
        const val=ev.target.value;
        const num=Number(val);
        if (val===""||val==null||!Number.isFinite(num)) {
          this._presetInvalid[idx]=true;
          this._presetFeedback="Please enter finite numbers for all presets before saving.";
        } else {
          const step=(this._units==="F")?1:0.5;
          const r=Math.round(num/step)*step;
          this._ptempLocal[idx]=r;
          this._presetInvalid[idx]=false;
          if (!this._presetInvalid.some(Boolean)) this._presetFeedback="";
        }
        this._dirtyPresets=true;
        this._syncHeader();
        this._syncPresetValidationUI();
      };
      this._els.presetCold.addEventListener("input",presetHandler(0));
      this._els.presetNight.addEventListener("input",presetHandler(1));
      this._els.presetDay.addEventListener("input",presetHandler(2));

      // Copy selectors
      this._els.copyFromSel.addEventListener("change",()=>{ this._copyFrom=Number(this._els.copyFromSel.value); });
      this._els.copyToSel.addEventListener("change",()=>{ const v=this._els.copyToSel.value; this._copyTo=(v==="All")?"All":Number(v); });
      this._els.copyEntitySel.addEventListener("change",()=>{
        const v = this._els.copyEntitySel.value;
        this._copyEntityTarget = v || null;
      });

      this.appendChild(card);

      // Initial sync
      this._syncHeader(); this._syncModeButtons(); this._syncWarn(); this._renderGridColors();
      this._syncPresetInputs(); this._syncPresetValidationUI(); this._syncCopySelectors(); this._syncEntityOptions(); this._syncCopyEntitySelect();
      this._syncSaveButtons();
    }

    // Sync helpers
    _syncHeader(){ const t=(this._stateObj?.attributes?.friendly_name||this._stateObj?.attributes?.name||this._entity||"TermoWeb schedule");
      this._els.titleEl.textContent=t; this._els.dirtyBadge.hidden=!(this._dirtyProg||this._dirtyPresets); this._els.unitsLabel.textContent=`Units: ${this._units}`;
      this._syncActionButtons();
    }

    _syncActionButtons(){
      const isDirty = this._dirtyProg || this._dirtyPresets;
      if (!this._els.revertBtn) return;
      this._els.revertBtn.disabled = !isDirty;
      this._els.revertBtn.classList.toggle("btn-muted", !isDirty);
    }

    _syncSaveButtons(){
      const isSavingAny = this._savingProg || this._savingPresets;
      if (this._els.saveBtn) {
        this._els.saveBtn.disabled = isSavingAny;
        this._els.saveBtn.textContent = this._savingProg ? "Saving…" : "Save";
      }
      if (this._els.savePresetsBtn) {
        this._els.savePresetsBtn.disabled = isSavingAny;
        this._els.savePresetsBtn.textContent = this._savingPresets ? "Saving…" : "Save Presets";
      }
    }

    _reconcileSelectOptions(select, entities, currentValue) {
      if (!select) return;
      if (select === document.activeElement) return;
      const wanted = entities.map(e => e.id);
      const existing = Array.from(select.options).map(o => o.value);
      // Remove extras
      for (let i = select.options.length - 1; i >= 0; i--) {
        const v = select.options[i].value;
        if (!wanted.includes(v)) select.remove(i);
      }
      // Add / update labels in order
      let pos = 0;
      for (const e of entities) {
        let opt = Array.from(select.options).find(o => o.value === e.id);
        if (!opt) {
          opt = document.createElement("option");
          opt.value = e.id;
          select.add(opt, pos);
        }
        if (opt.textContent !== e.name) opt.textContent = e.name;
        pos++;
      }
      if (currentValue != null && select.value !== currentValue) select.value = currentValue;
    }

    _syncEntityOptions(){
      this._reconcileSelectOptions(this._els.entitySelect, this._entities, this._entity || "");
    }

    _syncCopyEntitySelect() {
      const currentTarget = this._copyEntityTarget;
      // prefer first "other" entity if target missing
      const target = (currentTarget && this._entities.find(e => e.id === currentTarget))
        ? currentTarget
        : (this._entities.find(e => e.id !== this._entity)?.id || this._entities[0]?.id || "");
      this._copyEntityTarget = target || null;
      this._reconcileSelectOptions(this._els.copyEntitySel, this._entities, this._copyEntityTarget || "");
    }

    _syncPresetInputs(){ const [c,n,d]=this._ptempLocal??[null,null,null]; const step=(this._units==="F")?1:0.5;
      const apply=(el,v)=>{ el.step=String(step); if(el!==document.activeElement) el.value=(v==null?"":String(v)); };
      apply(this._els.presetCold,c); apply(this._els.presetNight,n); apply(this._els.presetDay,d);
    }
    _syncPresetValidationUI(){
      const inputs=[this._els.presetCold,this._els.presetNight,this._els.presetDay];
      for(let i=0;i<inputs.length;i++){
        const el=inputs[i];
        const invalid=Boolean(this._presetInvalid[i]);
        el?.classList.toggle("input-invalid",invalid);
        if(el && typeof el.setAttribute === "function") el.setAttribute("aria-invalid", invalid ? "true" : "false");
      }
      if(this._els.presetFeedback) this._els.presetFeedback.textContent=this._presetFeedback;
    }
    _syncModeButtons(){ const s=this._selectedMode; this._els.modeCold.classList.toggle("active",s===MODE.COLD);
      this._els.modeNight.classList.toggle("active",s===MODE.NIGHT); this._els.modeDay.classList.toggle("active",s===MODE.DAY); }
    _syncWarn(){ const hasEntities=Array.isArray(this._entities)&&this._entities.length>0;
      const has=Array.isArray(this._progLocal)&&this._progLocal.length===168;
      let baseWarn="";
      if(!hasEntities) {
        baseWarn="No valid TermoWeb climate entities found. Set 'entity' explicitly or adjust 'include_pattern'.";
      } else if (!has) {
        baseWarn="This entity has no valid 'prog' (expected 168 ints).";
      }
      this._els.warn.textContent=[baseWarn,this._statusWarn].filter(Boolean).join(" ");
      this._els.grid.classList.toggle("ghost",!has); }
    _syncCopySelectors(){ const fromEl=this._els.copyFromSel, toEl=this._els.copyToSel;
      if(fromEl && fromEl!==document.activeElement) fromEl.value=String(this._copyFrom);
      if(toEl && toEl!==document.activeElement) toEl.value=(this._copyTo==="All")?"All":String(this._copyTo);
    }
    _renderGridColors(){ const prog=this._progLocal, cells=this._els.cells; if(!cells||!Array.isArray(prog)||prog.length!==168) return;
      for(let h=0;h<24;h++){ for(let d=0; d<7; d++){ const k=hourIdx(d,h); const m=clamp(Number(prog[k]??0),0,2); const cell=cells[h*7+d]; if(cell) cell.style.background=COLORS[m]; }}}

    // Actions
    _setMode(m){ this._selectedMode=m; this._syncModeButtons(); }
    _startDrag(){ this._isDragging=true; } _stopDrag(){ this._isDragging=false; }
    _paintCell(day,hour){ if(!Array.isArray(this._progLocal)||this._progLocal.length!==168) return;
      const k=hourIdx(day,hour); if(this._progLocal[k]!==this._selectedMode){ this._progLocal=this._progLocal.slice(); this._progLocal[k]=this._selectedMode; this._dirtyProg=true;
        const cell=this._els.cells[hour*7+day]; if(cell) cell.style.background=COLORS[this._selectedMode]; this._syncHeader(); }}

    _copyDays(){ if(!Array.isArray(this._progLocal)||this._progLocal.length!==168) return;
      const from=clamp(Number(this._copyFrom),0,6); const to=this._copyTo; const next=this._progLocal.slice(); const src=next.slice(from*24,from*24+24);
      if(to==="All"){ for(let d=0; d<7; d++) next.splice(d*24,24,...src); } else { const d=clamp(Number(to),0,6); next.splice(d*24,24,...src); }
      this._progLocal=next; this._dirtyProg=true; this._renderGridColors(); this._syncHeader(); }

    _normalizePresetTriplet(raw){ if(!Array.isArray(raw)||raw.length!==3) return [null,null,null];
      return raw.map((value)=>{ const n=Number(value); return Number.isFinite(n)?n:null; }); }

    _copyToEntity() {
      if (!this._hass) return;
      const target = this._copyEntityTarget;
      if (!target || !this._hass.states[target]) return;

      const srcProg = Array.isArray(this._progLocal) && this._progLocal.length === 168 ? this._progLocal.slice() : null;
      const srcPresets = this._normalizePresetTriplet(this._ptempLocal);

      // Switch selected entity (do NOT trigger change handler)
      this._entity = target;
      if (this._els.entitySelect && this._els.entitySelect !== document.activeElement) this._els.entitySelect.value = target;

      // Reset stateObj/units to target's context
      this._stateObj = this._hass.states[target];
      const u = this._stateObj?.attributes?.units; this._units = (u === "F" || u === "C") ? u : "C";

      // Apply copies locally and mark dirty; DO NOT save
      if (srcProg && srcProg.length === 168) {
        this._progLocal = srcProg.slice();
        this._dirtyProg = true;
      } else {
        this._progLocal = null;
        this._dirtyProg = false;
      }
      if (srcPresets && srcPresets.length === 3) {
        this._ptempLocal = srcPresets.slice();
        this._presetInvalid = [false, false, false];
        this._presetFeedback = "";
        this._dirtyPresets = true;
      }

      // Paint UI
      this._renderGridColors();
      this._syncPresetInputs();
      this._syncPresetValidationUI();
      this._syncHeader();
      this._syncWarn();
      // keep copyEntitySel selection as-is; user might repeat
    }

    _revert(){ const st=this._stateObj; const prog=st?.attributes?.prog; const ptemp=st?.attributes?.ptemp;
      this._progLocal=(Array.isArray(prog)&&prog.length===168)?prog.slice():null; this._ptempLocal=this._normalizePresetTriplet(ptemp);
      this._presetInvalid=[false,false,false]; this._presetFeedback=""; this._dirtyProg=false; this._dirtyPresets=false; this._renderGridColors(); this._syncPresetInputs(); this._syncPresetValidationUI(); this._syncHeader(); this._syncWarn(); }
    _refresh(){ this._isInteracting=false; this._dirtyProg=false; this._dirtyPresets=false; this._presetInvalid=[false,false,false]; this._presetFeedback=""; this.hass=this._hass; }
    async _save(){ if(!this._entity||!this._hass) return;
      if(this._savingProg||this._savingPresets) return;
      this._savingProg=true;
      this._statusWarn="Saving changes...";
      this._syncSaveButtons();
      this._syncWarn();
      try {
        if(Array.isArray(this._progLocal)&&this._progLocal.length===168&&this._dirtyProg){
          await this._hass.callService("termoweb","set_schedule",{ entity_id:this._entity, prog:this._progLocal }); this._dirtyProg=false; }
        if(this._dirtyPresets) await this._savePresets(true);
      } catch (err) {
        this._statusWarn=`Save failed: ${err?.message || err || "Unknown error."}`;
      } finally {
        if (this._statusWarn === "Saving changes...") this._statusWarn = "";
        this._savingProg=false;
        this._syncSaveButtons();
        this._syncHeader();
        this._syncWarn();
      }
    }
    async _savePresets(fromSave=false){ if(!this._entity||!this._hass) return; const [c,n,d]=this._ptempLocal??[null,null,null];
      if(this._savingPresets || (this._savingProg && !fromSave)) return;
      if([c,n,d].some(v=>!Number.isFinite(v)) || this._presetInvalid.some(Boolean)) {
        this._dirtyPresets=true;
        this._presetFeedback="Please correct preset values before saving.";
        this._statusWarn="Preset save blocked: all preset temperatures must be finite numbers.";
        this._syncPresetValidationUI();
        this._syncHeader();
        this._syncWarn();
        return;
      }
      this._savingPresets=true;
      this._statusWarn="Saving changes...";
      this._syncSaveButtons();
      this._syncWarn();
      try {
        await this._hass.callService("termoweb","set_preset_temperatures",{ entity_id:this._entity, ptemp:[c,n,d] });
        this._dirtyPresets=false;
        this._presetFeedback="";
      } catch (err) {
        this._statusWarn=`Save presets failed: ${err?.message || err || "Unknown error."}`;
      } finally {
        if (this._statusWarn === "Saving changes...") this._statusWarn = "";
        this._savingPresets=false;
        this._syncSaveButtons();
        this._syncPresetValidationUI();
        this._syncHeader();
        this._syncWarn();
      }
    }
  }

  if(!customElements.get("termoweb-schedule-card")) customElements.define("termoweb-schedule-card", TermoWebScheduleCard);
  window.customCards = window.customCards || []; window.customCards.push({ type:"termoweb-schedule-card", name:"TermoWeb Schedule Card (Vanilla)", description:"Focus-safe scheduler for TermoWeb heaters." });
})();
