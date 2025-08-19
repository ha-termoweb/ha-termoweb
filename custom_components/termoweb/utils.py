from __future__ import annotations

import math
from typing import Any


def extract_heater_addrs(nodes: dict[str, Any] | None) -> list[str]:
    """Return heater addresses from a nodes payload.

    The function expects a mapping like the one returned by the
    `/mgr/nodes` endpoint and extracts the `addr` of all entries
    whose `type` is `htr` (case-insensitive).
    """
    addrs: dict[str, None] = {}
    if isinstance(nodes, dict):
        node_list = nodes.get("nodes")
        if isinstance(node_list, list):
            for n in node_list:
                if isinstance(n, dict) and (n.get("type") or "").lower() == "htr":
                    addr = str(n.get("addr"))
                    addrs.setdefault(addr)
    return list(addrs)


def float_or_none(value: Any) -> float | None:
    """Return value as ``float`` if possible, else ``None``.

    Converts integers, floats, and numeric strings to ``float`` while safely
    handling ``None`` and non-numeric inputs.
    """
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            num = float(value)
        else:
            string_val = str(value).strip()
            if not string_val:
                return None
            num = float(string_val)
        return num if math.isfinite(num) else None
    except Exception:
        return None
