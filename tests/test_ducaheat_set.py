from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

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
