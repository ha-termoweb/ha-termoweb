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
    this.childNodes = [];
  }

  attachShadow() {
    const root = new FakeShadowRoot();
    this.shadowRoot = root;
    return root;
  }

  appendChild(node) {
    if (node) {
      this.childNodes.push(node);
      node.parentNode = this;
    }
    return node;
  }

  removeChild(node) {
    this.childNodes = this.childNodes.filter((child) => child !== node);
  }

  addEventListener() {}

  removeEventListener() {}

  connectedCallback() {}

  disconnectedCallback() {}
}

class FakeDomElement extends FakeHTMLElement {
  constructor(tagName) {
    super();
    this.tagName = tagName ? tagName.toUpperCase() : "";
    this.childNodes = [];
    this.parentNode = null;
    this.dataset = {};
    this.style = {};
    this.textContent = "";
    this.className = "";
    this.classList = {
      toggle() {},
      add() {},
      remove() {},
    };
    this.value = "";
    this.hidden = false;
    this._innerHTML = "";
    this._idMap = new Map();
    this._isSelect = this.tagName === "SELECT";
    this.options = [];
  }

  set innerHTML(markup) {
    this._innerHTML = String(markup);
    this._idMap.clear();
    const regex = /id="([^"]+)"/g;
    let match;
    while ((match = regex.exec(this._innerHTML)) !== null) {
      const el = new FakeDomElement("div");
      el.id = match[1];
      this._idMap.set(`#${el.id}`, el);
    }
  }

  get innerHTML() {
    return this._innerHTML;
  }

  appendChild(node) {
    if (node) {
      this.childNodes.push(node);
      node.parentNode = this;
      if (this._isSelect && !this.options.includes(node)) {
        this.options.push(node);
      }
    }
    return node;
  }

  append(...nodes) {
    nodes.forEach((node) => this.appendChild(node));
  }

  add(option, index) {
    if (!this._isSelect || !option) {
      return;
    }
    if (index == null || index >= this.options.length) {
      this.options.push(option);
    } else {
      this.options.splice(index, 0, option);
    }
    option.parentNode = this;
  }

  remove(index) {
    if (this._isSelect && typeof index === "number") {
      if (index >= 0 && index < this.options.length) {
        const [removed] = this.options.splice(index, 1);
        if (removed) {
          removed.parentNode = null;
        }
      }
      return;
    }
    if (!this.parentNode) {
      return;
    }
    this.parentNode.childNodes = this.parentNode.childNodes.filter(
      (node) => node !== this,
    );
    this.parentNode = null;
  }

  querySelector(selector) {
    return this._idMap.get(selector) || null;
  }

  querySelectorAll() {
    return [];
  }

  addEventListener() {}

  removeEventListener() {}

  contains(target) {
    if (target === this) {
      return true;
    }
    return this.childNodes.some((child) =>
      typeof child.contains === "function" && child.contains(target),
    );
  }
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
  body: new FakeDomElement("body"),
  activeElement: null,
  createElement(tagName) {
    return new FakeDomElement(tagName);
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
card.setConfig({ entity: "climate.test" });

const makeProg = (value) => Array.from({ length: 168 }, () => value);

const hass = {
  states: {
    "climate.test": {
      attributes: {
        friendly_name: "Alpha Heater",
        prog: makeProg(0),
        ptemp: [10, 15, 20],
        units: "C",
      },
    },
    "climate.other": {
      attributes: {
        friendly_name: "Beta Heater",
        prog: makeProg(1),
        ptemp: [5, 10, 15],
        units: "C",
      },
    },
  },
  callServiceCalls: [],
  callService(domain, service, data) {
    this.callServiceCalls.push({ domain, service, data });
  },
};

card.hass = hass;

assert(Array.isArray(card._progLocal), "Card should hydrate program array");
assert(card._progLocal.length === 168, "Program array should have 168 slots");
assert(Array.isArray(card._ptempLocal), "Preset temperatures should hydrate");
assert(card._entities.length === 2, "Two climate entities should be detected");

// ---------- _copyDays behaviour ----------
card._progLocal = Array.from({ length: 168 }, (_, idx) => (idx % 3));
card._copyFrom = 1;
card._copyTo = 3;
card._dirtyProg = false;
card._copyDays();
for (let hour = 0; hour < 24; hour++) {
  const srcIndex = 1 * 24 + hour;
  const dstIndex = 3 * 24 + hour;
  assert(
    card._progLocal[srcIndex] === card._progLocal[dstIndex],
    "Copying a single day should mirror values",
  );
}
assert(card._dirtyProg === true, "Copying days marks program dirty");

card._copyFrom = 0;
card._copyTo = "All";
card._dirtyProg = false;
card._copyDays();
for (let day = 1; day < 7; day++) {
  for (let hour = 0; hour < 24; hour++) {
    const base = hour;
    const compare = day * 24 + hour;
    assert(
      card._progLocal[base] === card._progLocal[compare],
      "Copying to All should duplicate the source day",
    );
  }
}

// ---------- Painting cells ----------
card._selectedMode = 2;
card._paintCell(0, 0);
assert(card._progLocal[0] === 2, "Painting a cell should set selected mode");
const paintedCell = card._els.cells[0];
assert(
  paintedCell.style.background === "#FB8C00",
  "Painting should apply the mode colour",
);

// ---------- Copying to another entity ----------
const clonedProg = Array.from({ length: 168 }, (_, idx) => (idx % 2));
card._progLocal = clonedProg.slice();
card._ptempLocal = [11, 22, 33];
card._dirtyProg = false;
card._dirtyPresets = false;
card._copyEntityTarget = "climate.other";
card._copyToEntity();

assert(card._entity === "climate.other", "Copy to entity should retarget entity");
assert(card._dirtyProg === true, "Copy to entity marks program dirty");
assert(card._dirtyPresets === true, "Copy to entity marks presets dirty");
assert(
  JSON.stringify(card._progLocal) === JSON.stringify(clonedProg),
  "Program data should be cloned to target",
);
assert(
  JSON.stringify(card._ptempLocal) === JSON.stringify([11, 22, 33]),
  "Preset temperatures should be cloned to target",
);
assert(
  card._els.entitySelect.value === "climate.other",
  "Entity select should track the new target",
);
assert(card._els.warn.textContent === "", "Valid program keeps warning hidden");

console.log("All schedule card JS checks passed");
