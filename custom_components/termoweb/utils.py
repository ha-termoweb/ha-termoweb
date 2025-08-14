from __future__ import annotations

from typing import Any


def extract_heater_addrs(nodes: dict[str, Any] | None) -> list[str]:
    """Return heater addresses from a nodes payload.

    The function expects a mapping like the one returned by the
    `/mgr/nodes` endpoint and extracts the `addr` of all entries
    whose `type` is `htr` (case-insensitive).
    """
    addrs: list[str] = []
    if isinstance(nodes, dict):
        node_list = nodes.get("nodes")
        if isinstance(node_list, list):
            for n in node_list:
                if isinstance(n, dict) and (n.get("type") or "").lower() == "htr":
                    addrs.append(str(n.get("addr")))
    return addrs
