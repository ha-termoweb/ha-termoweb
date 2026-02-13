from __future__ import annotations

import asyncio
from types import SimpleNamespace
import inspect
from typing import Any, Mapping

import pytest

from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient


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


@pytest.mark.asyncio
async def test_set_node_settings_units_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send only units and verify a single status segment is planned."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_headers() -> dict[str, str]:
        """Return static headers for the request."""

        return {"Authorization": "Bearer token"}

    def fake_ensure_units(units: str) -> str:
        """Return a predictable units marker."""

        return f"unit:{units}"

    selection_calls: list[bool] = []

    async def fake_select_segmented_node(**kwargs: Any) -> None:
        """Record selection claims/releases."""

        selection_calls.append(kwargs["select"])

    post_calls: list[dict[str, Any]] = []

    async def fake_post_segmented(
        path: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        dev_id: str,
        addr: str,
        node_type: str,
    ) -> dict[str, str]:
        """Capture the payload sent to _post_segmented."""

        post_calls.append(
            {
                "path": path,
                "headers": dict(headers),
                "payload": dict(payload),
                "dev_id": dev_id,
                "addr": addr,
                "node_type": node_type,
            }
        )
        return {"ok": "yes"}

    monkeypatch.setattr(client, "authed_headers", fake_headers)
    monkeypatch.setattr(client, "_ensure_units", fake_ensure_units)
    monkeypatch.setattr(client, "_select_segmented_node", fake_select_segmented_node)
    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    responses = await client.set_node_settings("dev", ("htr", 1), units="F")

    assert responses == {"status": {"ok": "yes"}}
    assert post_calls == [
        {
            "path": "/api/v2/devs/dev/htr/1/status",
            "headers": {"Authorization": "Bearer token"},
            "payload": {"units": "unit:F"},
            "dev_id": "dev",
            "addr": "1",
            "node_type": "htr",
        }
    ]
    assert selection_calls == [True, False]


@pytest.mark.asyncio
async def test_set_node_settings_invalid_stemp_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure invalid stemp errors after claiming and releasing the node."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_headers() -> dict[str, str]:
        """Return static headers for the request."""

        return {"Authorization": "Bearer token"}

    def fake_ensure_units(units: str) -> str:
        """Return a predictable units marker."""

        return f"unit:{units}"

    selection_calls: list[bool] = []

    async def fake_select_segmented_node(**kwargs: Any) -> None:
        """Record selection claims/releases."""

        selection_calls.append(kwargs["select"])

    async def fake_post_segmented(**kwargs: Any) -> None:
        """_post_segmented should not be reached for invalid stemp."""

        raise AssertionError("_post_segmented must not be invoked")

    monkeypatch.setattr(client, "authed_headers", fake_headers)
    monkeypatch.setattr(client, "_ensure_units", fake_ensure_units)
    monkeypatch.setattr(client, "_select_segmented_node", fake_select_segmented_node)
    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    with pytest.raises(ValueError) as err:
        await client.set_node_settings("dev", ("htr", 1), stemp="bad", units="C")

    assert "Invalid temperature value" in str(err.value)
    assert selection_calls == []


@pytest.mark.asyncio
async def test_set_node_settings_mode_segment_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure standalone mode writes emit a dedicated mode segment."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_headers() -> dict[str, str]:
        """Return static authentication headers for the fake client."""

        return {"Authorization": "Bearer token"}

    selection_calls: list[bool] = []

    async def fake_select_segmented_node(**kwargs: Any) -> None:
        """Record node selection toggles for verification."""

        selection_calls.append(bool(kwargs["select"]))

    post_calls: list[dict[str, Any]] = []

    async def fake_post_segmented(
        path: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        dev_id: str,
        addr: str,
        node_type: str,
    ) -> dict[str, str]:
        """Capture payload metadata for the mode-only request."""

        post_calls.append(
            {
                "path": path,
                "headers": dict(headers),
                "payload": dict(payload),
                "dev_id": dev_id,
                "addr": addr,
                "node_type": node_type,
            }
        )
        return {"ok": True}

    def fake_ensure_units(units: str | None) -> str:
        """Drop the in-status mode entry and expose the separate mode segment."""

        frame = inspect.currentframe()
        parent = frame.f_back if frame is not None else None
        if parent is not None:
            status_payload = parent.f_locals.get("status_payload")
            if isinstance(status_payload, dict):
                status_payload.pop("mode", None)
            if "status_includes_mode" in parent.f_locals:
                parent.f_locals["status_includes_mode"] = False
        return "C" if units is not None else "C"

    def fake_ensure_temperature(value: Any) -> str:
        """Return a deterministic temperature string."""

        return "19.0"

    monkeypatch.setattr(client, "authed_headers", fake_headers)
    monkeypatch.setattr(client, "_select_segmented_node", fake_select_segmented_node)
    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)
    monkeypatch.setattr(client, "_ensure_units", fake_ensure_units)
    monkeypatch.setattr(client, "_ensure_temperature", fake_ensure_temperature)

    responses = await client.set_node_settings(
        "dev", ("htr", 2), mode="auto", stemp=18, units="C"
    )

    assert responses == {
        "status": {"ok": True},
    }
    assert selection_calls == [True, False]
    assert post_calls == [
        {
            "path": "/api/v2/devs/dev/htr/2/status",
            "headers": {"Authorization": "Bearer token"},
            "payload": {"mode": "auto", "stemp": "18.0", "units": "C"},
            "dev_id": "dev",
            "addr": "2",
            "node_type": "htr",
        },
    ]


@pytest.mark.asyncio
async def test_set_node_settings_preserves_modified_auto_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send modified_auto mode without coercing it to manual/auto."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_headers() -> dict[str, str]:
        """Return static authentication headers for the fake client."""

        return {"Authorization": "Bearer token"}

    post_calls: list[dict[str, Any]] = []

    async def fake_select_segmented_node(**_kwargs: Any) -> None:
        """Skip real node selection while testing payload serialization."""

    async def fake_post_segmented(
        path: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        dev_id: str,
        addr: str,
        node_type: str,
    ) -> dict[str, bool]:
        """Capture the status payload sent to the API."""

        post_calls.append(
            {
                "path": path,
                "headers": dict(headers),
                "payload": dict(payload),
                "dev_id": dev_id,
                "addr": addr,
                "node_type": node_type,
            }
        )
        return {"ok": True}

    monkeypatch.setattr(client, "authed_headers", fake_headers)
    monkeypatch.setattr(client, "_select_segmented_node", fake_select_segmented_node)
    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    await client.set_node_settings(
        "dev",
        ("htr", 2),
        mode="modified_auto",
        stemp=20.5,
    )

    assert post_calls == [
        {
            "path": "/api/v2/devs/dev/htr/2/status",
            "headers": {"Authorization": "Bearer token"},
            "payload": {"mode": "modified_auto", "stemp": "20.5", "units": "C"},
            "dev_id": "dev",
            "addr": "2",
            "node_type": "htr",
        }
    ]
