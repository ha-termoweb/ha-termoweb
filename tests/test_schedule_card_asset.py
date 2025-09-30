"""Tests for the TermoWeb schedule card asset."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_schedule_card_clears_cache_when_prog_missing() -> None:
    """Ensure cached schedule data is dropped when prog disappears."""

    if shutil.which("node") is None:
        pytest.skip("Node.js runtime is required for schedule card tests")

    repo_root = Path(__file__).resolve().parents[1]
    card_path = repo_root / "custom_components" / "termoweb" / "assets" / "termoweb_schedule_card.js"
    card_path_str = json.dumps(str(card_path))

    script_lines = [
        "const fs = require('fs');",
        "const vm = require('vm');",
        f"const cardPath = {card_path_str};",
        "",
        "global.window = global;",
        "window.customCards = [];",
        "window.addEventListener = () => {};",
        "window.removeEventListener = () => {};",
        "window.requestAnimationFrame = (cb) => setTimeout(cb, 16);",
        "",
        "global.customElements = {",
        "  _registry: new Map(),",
        "  define(name, ctor) { this._registry.set(name, ctor); },",
        "  get(name) { return this._registry.get(name); },",
        "};",
        "",
        "class ShadowRoot {",
        "  constructor() {",
        "    this.innerHTML = '';",
        "    this.activeElement = null;",
        "  }",
        "  getElementById() { return null; }",
        "  querySelector() { return null; }",
        "  querySelectorAll() { return []; }",
        "}",
        "",
        "global.HTMLElement = class {",
        "  attachShadow() {",
        "    this.shadowRoot = new ShadowRoot();",
        "    return this.shadowRoot;",
        "  }",
        "};",
        "",
        "global.document = {",
        "  createElement() {",
        "    return {",
        "      style: {},",
        "      remove() {},",
        "      set textContent(value) { this._text = value; },",
        "      get textContent() { return this._text; },",
        "    };",
        "  },",
        "  body: { appendChild() {}, removeChild() {} },",
        "};",
        "",
        "vm.runInThisContext(fs.readFileSync(cardPath, 'utf8'), { filename: cardPath });",
        "",
        "const Card = customElements.get('termoweb-schedule-card');",
        "const card = new Card();",
        "card._renderCalls = 0;",
        "card._config = { entity: 'climate.test' };",
        "card._entity = 'climate.test';",
        "card._els = { progWarn: { hidden: true } };",
        "",
        "card._render = function() {",
        "  this._renderCalls += 1;",
        "  const hasProg = Array.isArray(this._progLocal) && this._progLocal.length === 168;",
        "  if (!this._els) this._els = {};",
        "  if (!this._els.progWarn) this._els.progWarn = { hidden: true };",
        "  this._els.progWarn.hidden = hasProg;",
        "  this._hasRendered = true;",
        "};",
        "card._renderGridOnly = function() {};",
        "card._updateStatusIndicators = function() {};",
        "card._restorePresetFocusIfNeeded = function() {};",
        "card._syncEntityOptions = function() {};",
        "card._updateModeButtons = function() {};",
        "",
        "const validState = {",
        "  states: {",
        "    'climate.test': {",
        "      attributes: {",
        "        prog: Array(168).fill(1),",
        "        ptemp: [10, 15, 20],",
        "      },",
        "    },",
        "  },",
        "};",
        "card.hass = validState;",
        "const renderAfterValid = card._renderCalls;",
        "const afterValid = {",
        "  hasProg: Array.isArray(card._progLocal) && card._progLocal.length === 168,",
        "  warnHidden: card._els.progWarn.hidden,",
        "  renderCalls: renderAfterValid,",
        "};",
        "",
        "card._dirtyProg = true;",
        "card._freezeUntil = Date.now() + 5000;",
        "card._pendingEcho.prog = Array(168).fill(1);",
        "",
        "const invalidState = {",
        "  states: {",
        "    'climate.test': {",
        "      attributes: {",
        "        ptemp: [10, 15, 20],",
        "      },",
        "    },",
        "  },",
        "};",
        "",
        "card.hass = invalidState;",
        "",
        "const afterInvalid = {",
        "  progIsNull: card._progLocal === null,",
        "  dirtyProg: card._dirtyProg,",
        "  freeze: card._freezeUntil,",
        "  pendingProg: card._pendingEcho.prog,",
        "  warnHidden: card._els.progWarn.hidden,",
        "  renderCalls: card._renderCalls,",
        "};",
        "",
        "const result = {",
        "  afterValid,",
        "  afterInvalid,",
        "};",
        "",
        "console.log(JSON.stringify(result));",
    ]

    script = "\n".join(script_lines)

    proc = subprocess.run(
        ["node", "-e", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    stdout = proc.stdout.strip()
    assert stdout, "No output from Node test harness"
    data = json.loads(stdout)

    assert data["afterValid"]["hasProg"] is True
    assert data["afterValid"]["warnHidden"] is True
    assert data["afterInvalid"]["progIsNull"] is True
    assert data["afterInvalid"]["dirtyProg"] is False
    assert data["afterInvalid"]["freeze"] == 0
    assert data["afterInvalid"]["pendingProg"] is None
    assert data["afterInvalid"]["warnHidden"] is False
    assert data["afterInvalid"]["renderCalls"] > data["afterValid"]["renderCalls"]
