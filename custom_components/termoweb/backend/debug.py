"""Backend debug helpers."""

from __future__ import annotations

from collections.abc import Mapping

from custom_components.termoweb.const import uses_ducaheat_backend
from custom_components.termoweb.inventory import (
    normalize_node_addr,
    normalize_node_type,
)


def build_unknown_node_probe_requests(
    brand: str,
    dev_id: str,
    node_type: str,
    addr: str,
) -> tuple[tuple[str, Mapping[str, str] | None], ...]:
    """Return common REST endpoints to query for an unknown node type."""

    dev_id_str = str(dev_id).strip()
    normalized_type = normalize_node_type(
        node_type,
        use_default_when_falsey=True,
    )
    normalized_addr = normalize_node_addr(
        addr,
        use_default_when_falsey=True,
    )
    if not dev_id_str or not normalized_type:
        return ()

    base_path = f"/api/v2/devs/{dev_id_str}/{normalized_type}"
    node_path = base_path if not normalized_addr else f"{base_path}/{normalized_addr}"

    requests: list[tuple[str, Mapping[str, str] | None]] = []
    seen_paths: set[str] = set()

    def _append(path: str, params: Mapping[str, str] | None = None) -> None:
        if path in seen_paths:
            return
        seen_paths.add(path)
        requests.append((path, params))

    if uses_ducaheat_backend(brand) and normalized_addr:
        _append(base_path)

    _append(node_path)
    _append(f"{node_path}/settings")
    if normalized_addr:
        _append(f"{node_path}/samples", {"start": "0", "end": "0"})
    return tuple(requests)
