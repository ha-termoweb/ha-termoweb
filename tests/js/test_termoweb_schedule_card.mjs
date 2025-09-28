import fs from "fs";
import path from "path";
import vm from "vm";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const readCardSource = () => {
  const repoRoot = path.resolve(
    __dirname,
    "..",
    "..",
    "custom_components",
    "termoweb",
    "assets",
  );
  const cardPath = path.join(repoRoot, "termoweb_schedule_card.js");
  return fs.readFileSync(cardPath, "utf8");
};

class FakeShadowRoot {
  constructor() {
    this.innerHTML = "";
  }

  getElementById() {
    return null;
  }

  querySelector() {
    return null;
  }

  querySelectorAll() {
    return [];
  }

  appendChild() {}
}

class FakeHTMLElement {
  constructor() {
    this.shadowRoot = null;
  }

  attachShadow() {
    const root = new FakeShadowRoot();
    this.shadowRoot = root;
    return root;
  }

  addEventListener() {}

  removeEventListener() {}

  connectedCallback() {}

  disconnectedCallback() {}
}

const windowStub = {
  customCards: [],
  _listeners: new Map(),
  addEventListener(type, handler, options) {
    const entry = {
      handler,
      options: options || {},
      invoke: (...args) => {
        handler(...args);
        if (entry.options?.once) {
          windowStub._listeners.delete(type);
        }
      },
    };
    this._listeners.set(type, entry);
  },
  removeEventListener(type) {
    this._listeners.delete(type);
  },
  dispatchEvent(type, ...args) {
    const entry = this._listeners.get(type);
    if (entry) {
      entry.invoke(...args);
    }
  },
};

const documentStub = {
  body: {
    appendChild() {},
  },
  createElement() {
    return {
      style: {},
      textContent: "",
      remove() {},
    };
  },
};

const customElementsRegistry = new Map();

const customElementsStub = {
  define(name, clazz) {
    customElementsRegistry.set(name, clazz);
  },
  get(name) {
    return customElementsRegistry.get(name);
  },
};

const context = {
  window: windowStub,
  document: documentStub,
  customElements: customElementsStub,
  HTMLElement: FakeHTMLElement,
  console,
  setTimeout,
  clearTimeout,
};

context.globalThis = context;
context.self = context;

const script = new vm.Script(readCardSource(), {
  filename: "termoweb_schedule_card.js",
});

vm.createContext(context);
script.runInContext(context);

const cardClass = customElementsRegistry.get("termoweb-schedule-card");
if (!cardClass) {
  throw new Error("Card class was not registered");
}

const assert = (condition, message) => {
  if (!condition) {
    throw new Error(message);
  }
};

const card = new cardClass();

// ---------- _copyDay tests ----------
card._progLocal = Array.from({ length: 168 }, (_, i) => i % 3);
card._dirtyProg = false;
card._copyDay(1, 3);
for (let h = 0; h < 24; h++) {
  const srcIdx = card._idx(1, h);
  const dstIdx = card._idx(3, h);
  assert(
    card._progLocal[srcIdx] === card._progLocal[dstIdx],
    "copyDay should copy selected day values",
  );
}

card._progLocal = Array.from({ length: 168 }, (_, i) => (i < 24 ? 1 : 0));
card._copyDay(0, "All");
for (let day = 1; day < 7; day++) {
  for (let h = 0; h < 24; h++) {
    const srcIdx = card._idx(0, h);
    const dstIdx = card._idx(day, h);
    assert(
      card._progLocal[srcIdx] === card._progLocal[dstIdx],
      "copyDay should copy to all days when All is selected",
    );
  }
}

// ---------- _canHydrateFromState tests ----------
card._progLocal = null;
card._ptempLocal = [null, null, null];
card._dirtyProg = false;
card._dirtyPresets = false;
card._editingPresetIdx = -1;
card._freezeUntil = 0;
assert(
  card._canHydrateFromState({ freezeActive: false }) === true,
  "Should hydrate when no local data",
);

card._progLocal = new Array(168).fill(0);
card._dirtyProg = true;
assert(
  card._canHydrateFromState({ freezeActive: false }) === false,
  "Dirty program prevents hydrate",
);
card._dirtyProg = false;
card._dirtyPresets = true;
assert(
  card._canHydrateFromState({ freezeActive: false }) === false,
  "Dirty presets prevent hydrate",
);
card._dirtyPresets = false;
card._editingPresetIdx = 1;
assert(
  card._canHydrateFromState({ freezeActive: false }) === false,
  "Editing preset prevents hydrate",
);
card._editingPresetIdx = -1;
assert(
  card._canHydrateFromState({ freezeActive: true }) === false,
  "Freeze prevents hydrate",
);
assert(
  card._canHydrateFromState({ freezeActive: false }) === true,
  "Clean state hydrates",
);

// ---------- Pointer drag helper tests ----------
card._progLocal = new Array(168).fill(0);
card._selectedMode = 2;
card._dirtyProg = false;
let renderCalls = 0;
let statusCalls = 0;
let colorCalls = 0;
card._renderGridOnly = () => {
  renderCalls += 1;
};
card._updateStatusIndicators = () => {
  statusCalls += 1;
};
card._colorCell = () => {
  colorCalls += 1;
};

const cellStub = {
  _captured: false,
  setPointerCaptureCalls: [],
  releasePointerCaptureCalls: [],
  setPointerCapture(id) {
    this._captured = true;
    this.setPointerCaptureCalls.push(id);
  },
  releasePointerCapture(id) {
    this._captured = false;
    this.releasePointerCaptureCalls.push(id);
  },
  hasPointerCapture() {
    return this._captured;
  },
};

const makePointerEvent = (overrides = {}) => ({
  pointerId: 101,
  currentTarget: cellStub,
  preventDefault() {},
  ...overrides,
});

card._onPointerDown(makePointerEvent(), 0, 0);
assert(card._dragging === true, "Pointer down should start dragging");
assert(card._paintValue === 2, "Pointer down should set paint value");
assert(card._progLocal[0] === 2, "Pointer down should paint initial cell");
assert(card._dirtyProg === true, "Pointer down marks program dirty");
assert(renderCalls === 1, "Pointer down should render grid");
assert(statusCalls === 1, "Pointer down should update status");
assert(cellStub.setPointerCaptureCalls.length === 1, "Pointer down captures pointer once");
assert(cellStub.releasePointerCaptureCalls.length === 1, "Pointer capture released immediately to allow enters");
assert(windowStub._listeners.has("pointerup"), "Pointer down should register global pointerup listener");
assert(windowStub._listeners.has("pointercancel"), "Pointer down should register global pointercancel listener");

card._onPointerEnter(makePointerEvent({ pointerId: 101 }), 0, 1);
assert(card._progLocal[1] === 2, "Pointer enter should paint new cells while dragging");
assert(colorCalls === 1, "Pointer enter should color cell when value changes");
assert(statusCalls === 2, "Pointer enter should update status when painting");

card._onPointerEnter(makePointerEvent({ pointerId: 101 }), 0, 1);
assert(colorCalls === 1, "Pointer enter should not repaint identical values");

const pointerUpEvent = makePointerEvent({ pointerId: 101, currentTarget: windowStub });
windowStub.dispatchEvent("pointerup", pointerUpEvent);
assert(card._dragging === false, "Pointer up should stop dragging");
assert(card._paintValue === null, "Pointer up clears paint value");
assert(card._activePointerId === null, "Pointer up clears active pointer id");
assert(!windowStub._listeners.has("pointerup"), "Pointer up listener should clear after firing");
assert(!windowStub._listeners.has("pointercancel"), "Pointer cancel listener should clear after pointer finishes");
assert(card._windowPointerTracking === false, "Pointer tracking flag should reset after finish");

console.log("All schedule card JS checks passed");
