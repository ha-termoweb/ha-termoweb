from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient


def _make_rest_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    side_effects: dict[str, Exception] | None = None,
) -> tuple[DucaheatRESTClient, list[dict[str, Any]]]:
    """Create a Ducaheat REST client that records segmented POST calls."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
    calls: list[dict[str, Any]] = []
    effects = dict(side_effects or {})

    async def fake_headers() -> dict[str, str]:
        """Return static authentication headers for tests."""

        return {"Authorization": "Bearer token"}

    async def fake_post_segmented(
        path: str,
        *,
        headers: dict[str, str],
        payload: Mapping[str, Any],
        dev_id: str,
        addr: str,
        node_type: str,
        ignore_statuses: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        """Record POST calls instead of sending them to the backend."""

        record = {
            "path": path,
            "payload": dict(payload),
            "dev_id": dev_id,
            "addr": addr,
            "node_type": node_type,
            "ignore_statuses": tuple(ignore_statuses or ()),
            "headers": dict(headers),
        }
        calls.append(record)
        effect = effects.get(path)
        if effect is not None:
            raise effect
        return {"ok": True}

    monkeypatch.setattr(client, "_authed_headers", fake_headers)
    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)
    return client, calls


def _assert_mode_only(
    _: DucaheatRESTClient, calls: list[dict[str, Any]], __: dict[str, Any]
) -> None:
    """Validate the call sequence for a mode-only update."""

    assert [call["path"] for call in calls] == [
        "/api/v2/devs/dev/htr/1/mode",
    ]
    assert calls[0]["payload"] == {"mode": "manual"}
    assert calls[0]["ignore_statuses"] == ()


def _assert_stemp_only(
    _: DucaheatRESTClient, calls: list[dict[str, Any]], kwargs: dict[str, Any]
) -> None:
    """Validate the call sequence for a temperature-only update."""

    assert [call["path"] for call in calls] == [
        "/api/v2/devs/dev/htr/1/status",
    ]
    assert calls[0]["payload"] == {
        "stemp": f"{float(kwargs['stemp']):.1f}",
        "units": "C",
    }


def _assert_mode_and_stemp(
    _: DucaheatRESTClient, calls: list[dict[str, Any]], kwargs: dict[str, Any]
) -> None:
    """Validate the call sequence when mode and stemp change together."""

    assert [call["path"] for call in calls] == [
        "/api/v2/devs/dev/htr/1/status",
    ]
    assert calls[0]["payload"] == {
        "stemp": f"{float(kwargs['stemp']):.1f}",
        "units": "C",
        "mode": "manual" if str(kwargs["mode"]).lower() == "heat" else str(kwargs["mode"]).lower(),
    }


def _assert_stemp_with_units(
    _: DucaheatRESTClient, calls: list[dict[str, Any]], kwargs: dict[str, Any]
) -> None:
    """Validate the call sequence when temperature and units update."""

    assert [call["path"] for call in calls] == [
        "/api/v2/devs/dev/htr/1/status",
    ]
    units_value = str(kwargs["units"]).strip().upper()
    if not units_value:
        units_value = "C"
    assert calls[0]["payload"] == {
        "stemp": f"{float(kwargs['stemp']):.1f}",
        "units": units_value,
    }


def _assert_units_only(
    _: DucaheatRESTClient, calls: list[dict[str, Any]], kwargs: dict[str, Any]
) -> None:
    """Validate the call sequence when only the units value changes."""

    assert [call["path"] for call in calls] == [
        "/api/v2/devs/dev/htr/1/status",
    ]
    units_value = str(kwargs["units"]).strip().upper()
    if not units_value:
        units_value = "C"
    assert calls[0]["payload"] == {
        "units": units_value,
    }


def _assert_prog_only(
    client: DucaheatRESTClient, calls: list[dict[str, Any]], kwargs: dict[str, Any]
) -> None:
    """Validate the call sequence when only the weekly programme updates."""

    assert [call["path"] for call in calls] == [
        "/api/v2/devs/dev/htr/1/prog",
    ]
    assert calls[0]["payload"] == client._serialise_prog(kwargs["prog"])


@pytest.mark.parametrize(
    ("kwargs", "validator"),
    [
        ({"mode": "heat"}, _assert_mode_only),
        ({"stemp": 21.0}, _assert_stemp_only),
        ({"mode": "auto", "stemp": 19.5}, _assert_mode_and_stemp),
        ({"stemp": 18.5, "units": "f"}, _assert_stemp_with_units),
        ({"stemp": 17.5, "units": None}, _assert_stemp_only),
        ({"units": "f"}, _assert_units_only),
        ({"units": "   "}, _assert_units_only),
        ({"prog": [0, 1, 2] * 56}, _assert_prog_only),
    ],
)
def test_ducaheat_heater_segment_plan(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], validator
) -> None:
    """Verify segmented heater writes use the minimal endpoint set."""

    async def _run() -> None:
        """Execute the asynchronous body for the heater write test."""

        client, calls = _make_rest_client(monkeypatch)
        await client.set_node_settings("dev", ("htr", "1"), **kwargs)
        validator(client, calls, kwargs)

    asyncio.run(_run())


def test_ducaheat_acm_ptemp_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate ACM segmented writes for preset temperatures."""

    async def _run() -> None:
        """Execute the asynchronous body for the ACM write test."""

        client, calls = _make_rest_client(monkeypatch)
        await client.set_node_settings(
            "dev", ("acm", "3"), ptemp=[18.0, 20.0, 22.0]
        )
        assert [call["path"] for call in calls] == [
            "/api/v2/devs/dev/acm/3/prog_temps",
        ]
        assert calls[0]["payload"] == {
            "ptemp": ["18.0", "20.0", "22.0"],
        }

    asyncio.run(_run())


def test_ducaheat_post_segmented_ignore_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure ignore_statuses are forwarded by _post_segmented."""

    async def _run() -> None:
        """Execute the asynchronous body for ignore_status coverage."""

        client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
        captured: dict[str, Any] = {}

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            """Capture the underlying request invocation."""

            captured["method"] = method
            captured["path"] = path
            captured.update(kwargs)
            return {"ok": True}

        monkeypatch.setattr(client, "_request", fake_request)

        result = await client._post_segmented(
            "/api/test",
            headers={"Authorization": "Bearer token"},
            payload={"value": 1},
            dev_id="dev",
            addr="1",
            node_type="htr",
            ignore_statuses=(404,),
        )

        assert result == {"ok": True}
        assert captured["ignore_statuses"] == (404,)

    asyncio.run(_run())
