"""Tests for Ducaheat legacy section updates."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from homeassistant.core import HomeAssistant

from custom_components.termoweb.backend import ducaheat_ws


class DummyREST:
    """Provide the minimal REST interface required by the websocket client."""

    def __init__(self) -> None:
        self._session = SimpleNamespace()
        self._headers = {"Authorization": "Bearer rest-token"}

    async def authed_headers(self) -> dict[str, str]:
        """Return cached REST headers with an access token."""

        return self._headers


def _make_client(coordinator: Any) -> ducaheat_ws.DucaheatWSClient:
    """Instantiate a websocket client for legacy section tests."""

    hass = HomeAssistant()
    session = SimpleNamespace(closed=False)
    return ducaheat_ws.DucaheatWSClient(
        hass,
        entry_id="entry",
        dev_id="device",
        api_client=DummyREST(),
        coordinator=coordinator,
        session=session,
    )


def test_update_legacy_section_normalises_accumulator_settings() -> None:
    """Accumulator settings updates should normalise addresses and call helpers."""

    apply_helper = MagicMock()
    now_helper = MagicMock(return_value=1234.5)
    coordinator = SimpleNamespace(
        _apply_accumulator_boost_metadata=apply_helper,
        _device_now_estimate=now_helper,
    )

    client = _make_client(coordinator)
    client._nodes_raw = {}

    body = {"mode": "auto", "boost": {"remaining": 3600}}
    dev_map: dict[str, Any] = {"settings": {}}

    updated = client._update_legacy_section(
        node_type="acm",
        addr=" 02 ",
        section="settings",
        body=body,
        dev_map=dev_map,
    )

    assert updated is True
    apply_helper.assert_called_once()
    helper_args = apply_helper.call_args
    assert helper_args.args[0] == {"mode": "auto", "boost": {"remaining": 3600}}
    assert helper_args.kwargs == {"now": 1234.5}
    now_helper.assert_called_once_with()

    settings_bucket = dev_map["settings"].get("acm")
    assert isinstance(settings_bucket, dict)
    stored_payload = settings_bucket.get("02")
    assert stored_payload == {"mode": "auto", "boost": {"remaining": 3600}}
    assert stored_payload is not body

    raw_payload = client._nodes_raw.get("acm", {}).get("settings", {}).get(" 02 ")
    assert raw_payload == body
    assert raw_payload is not body


def test_update_legacy_section_rejects_non_mapping_section_map() -> None:
    """Non-mapping section buckets should be ignored without side effects."""

    coordinator = SimpleNamespace(
        _apply_accumulator_boost_metadata=MagicMock(),
        _device_now_estimate=MagicMock(return_value=0.0),
    )

    client = _make_client(coordinator)
    existing_raw = {"acm": {"settings": {" 02 ": {"mode": "manual"}}}}
    client._nodes_raw = deepcopy(existing_raw)

    dev_map: dict[str, Any] = {"settings": {"acm": ["invalid"]}}
    dev_map_snapshot = deepcopy(dev_map)

    updated = client._update_legacy_section(
        node_type="acm",
        addr=" 02 ",
        section="settings",
        body={"mode": "auto"},
        dev_map=dev_map,
    )

    assert updated is False
    assert dev_map == dev_map_snapshot
    assert client._nodes_raw == existing_raw
    coordinator._apply_accumulator_boost_metadata.assert_not_called()
    coordinator._device_now_estimate.assert_not_called()

