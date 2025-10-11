import asyncio
import logging
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientResponseError

from custom_components.termoweb.api import RESTClient
from custom_components.termoweb.backend.ducaheat import (
    DucaheatBackend,
    DucaheatRESTClient,
)
from custom_components.termoweb.backend.sanitize import (
    build_acm_boost_payload,
    mask_identifier,
    redact_text,
    redact_token_fragment,
    validate_boost_minutes,
)
from custom_components.termoweb.const import WS_NAMESPACE
from custom_components.termoweb.backend.ducaheat_ws import DucaheatWSClient


class DummyClient:
    def __init__(self) -> None:
        self._session = SimpleNamespace()

    async def list_devices(self) -> list[dict[str, object]]:
        return []

    async def get_nodes(self, dev_id: str) -> dict[str, object]:
        return {"dev_id": dev_id}

    async def get_node_settings(
        self, dev_id: str, node: tuple[str, str | int]
    ) -> dict[str, object]:
        node_type, addr = node
        return {"dev_id": dev_id, "addr": addr, "type": node_type}

    async def set_node_settings(
        self,
        dev_id: str,
        node: tuple[str, str | int],
        *,
        mode: str | None = None,
        stemp: float | None = None,
        prog: list[int] | None = None,
        ptemp: list[float] | None = None,
        units: str = "C",
    ) -> dict[str, object]:
        return {}

    async def get_node_samples(
        self,
        dev_id: str,
        node: tuple[str, str | int],
        start: float,
        stop: float,
    ) -> list[dict[str, object]]:
        return []

    async def _authed_headers(self) -> dict[str, str]:  # pragma: no cover - stub
        return {"Authorization": "Bearer token"}


@pytest.fixture
def ducaheat_rest_client(monkeypatch: pytest.MonkeyPatch) -> DucaheatRESTClient:
    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_headers() -> dict[str, str]:
        return {"Authorization": "Bearer token"}

    monkeypatch.setattr(client, "_authed_headers", fake_headers)
    return client


@pytest.mark.asyncio
async def test_ducaheat_backend_creates_ws_client() -> None:
    backend = DucaheatBackend(brand="ducaheat", client=DummyClient())
    hass = SimpleNamespace(loop=asyncio.get_running_loop(), data={})
    inventory = object()
    ws_client = backend.create_ws_client(
        hass,
        entry_id="entry",
        dev_id="dev",
        coordinator=object(),
        inventory=inventory,
    )
    assert isinstance(ws_client, DucaheatWSClient)
    assert ws_client.dev_id == "dev"
    assert ws_client.entry_id == "entry"
    assert ws_client._namespace == WS_NAMESPACE
    assert getattr(ws_client, "_inventory", None) is inventory


def test_dummy_client_get_node_settings_accepts_acm() -> None:
    client = DummyClient()

    async def _run() -> dict[str, object]:
        return await client.get_node_settings("dev", ("acm", "3"))

    data = asyncio.run(_run())
    assert data["type"] == "acm"
    assert data["addr"] == "3"


@pytest.mark.asyncio
async def test_ducaheat_rest_client_fetches_generic_node(
    ducaheat_rest_client: DucaheatRESTClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    async def fake_request(method: str, path: str, **kwargs: object):
        seen["method"] = method
        seen["path"] = path
        seen["kwargs"] = kwargs
        return {"status": {"power": 0}}

    monkeypatch.setattr(ducaheat_rest_client, "_request", fake_request)

    result = await ducaheat_rest_client.get_node_settings("dev", ("pmo", "9"))
    assert result == {"status": {"power": 0}}
    assert seen["method"] == "GET"
    assert seen["path"] == "/api/v2/devs/dev/pmo/9"
    assert seen["kwargs"] == {"headers": {"Authorization": "Bearer token"}}


@pytest.mark.asyncio
async def test_ducaheat_rest_client_normalises_acm(
    ducaheat_rest_client: DucaheatRESTClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    async def fake_request(method: str, path: str, **kwargs: object):
        seen["method"] = method
        seen["path"] = path
        seen["kwargs"] = kwargs
        return {"status": {"mode": "AUTO"}}

    monkeypatch.setattr(ducaheat_rest_client, "_request", fake_request)

    def fake_normalise(self, payload, *, node_type: str = "htr"):
        seen["node_type"] = node_type
        seen["payload"] = payload
        return {"normalized": True}

    monkeypatch.setattr(DucaheatRESTClient, "_normalise_settings", fake_normalise)

    result = await ducaheat_rest_client.get_node_settings("dev", ("acm", "2"))
    assert result == {"normalized": True}
    assert seen["node_type"] == "acm"
    assert seen["payload"] == {"status": {"mode": "AUTO"}}
    assert seen["method"] == "GET"
    assert seen["path"] == "/api/v2/devs/dev/acm/2"
    assert seen["kwargs"] == {"headers": {"Authorization": "Bearer token"}}


@pytest.mark.asyncio
async def test_ducaheat_rest_set_node_settings_routes_non_special(
    ducaheat_rest_client: DucaheatRESTClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    async def fake_super(self, dev_id: str, node: tuple[str, str], **kwargs):
        captured["args"] = (dev_id, node, kwargs)
        return {"ok": True}

    monkeypatch.setattr(RESTClient, "set_node_settings", fake_super)

    result = await ducaheat_rest_client.set_node_settings(
        "dev",
        ("pmo", "4"),
        mode="auto",
        stemp=20.5,
    )

    assert result == {"ok": True}
    assert captured["args"] == (
        "dev",
        ("pmo", "4"),
        {
            "mode": "auto",
            "stemp": 20.5,
            "prog": None,
            "ptemp": None,
            "units": "C",
            "cancel_boost": False,
        },
    )


@pytest.mark.asyncio
async def test_ducaheat_rest_set_node_settings_acm_mode_heat(
    ducaheat_rest_client: DucaheatRESTClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_headers() -> dict[str, str]:
        return {"Authorization": "Bearer"}

    calls: list[tuple[str, str, dict[str, object]]] = []

    async def fake_request(method: str, path: str, **kwargs: object):
        calls.append((method, path, kwargs))
        return {"status": "ok"}

    monkeypatch.setattr(ducaheat_rest_client, "_authed_headers", fake_headers)
    monkeypatch.setattr(ducaheat_rest_client, "_request", fake_request)

    rtc_mock = AsyncMock(return_value={"y": 2024, "n": 1, "d": 1, "h": 0, "m": 0, "s": 0})
    monkeypatch.setattr(ducaheat_rest_client, "get_rtc_time", rtc_mock)

    await ducaheat_rest_client.set_node_settings(
        "dev", ("acm", "3"), mode="boost", boost_time=30
    )

    assert calls == [
        (
            "POST",
            "/api/v2/devs/dev/acm/3/mode",
            {
                "headers": {"Authorization": "Bearer"},
                "json": {"mode": "boost", "boost_time": 30},
            },
        )
    ]
    rtc_mock.assert_awaited_once_with("dev")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"stemp": "bad", "units": "C"},
        {"stemp": 21.0, "units": "kelvin"},
    ],
)
async def test_ducaheat_rest_set_node_settings_acm_invalid_inputs(
    ducaheat_rest_client: DucaheatRESTClient,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
) -> None:
    async def fake_headers() -> dict[str, str]:
        return {}

    monkeypatch.setattr(ducaheat_rest_client, "_authed_headers", fake_headers)

    with pytest.raises(ValueError):
        await ducaheat_rest_client.set_node_settings("dev", ("acm", "3"), **kwargs)


@pytest.mark.asyncio
async def test_ducaheat_post_acm_endpoint_rethrows_server_error(
    ducaheat_rest_client: DucaheatRESTClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_request(method: str, path: str, **kwargs: object):
        raise ClientResponseError(
            request_info=None,
            history=(),
            status=500,
            message="server error",
        )

    monkeypatch.setattr(ducaheat_rest_client, "_request", fake_request)

    with pytest.raises(ClientResponseError):
        await ducaheat_rest_client._post_acm_endpoint(
            "/api/v2/devs/dev/acm/3/boost", {}, {"mode": "boost"}
        )


@pytest.mark.asyncio
async def test_ducaheat_set_acm_boost_state_claims_select(
    ducaheat_rest_client: DucaheatRESTClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    async def fake_post_segmented(
        self,
        path: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        dev_id: str,
        addr: str,
        node_type: str,
        ignore_statuses: Iterable[int] | None = None,
    ) -> dict[str, object]:
        calls.append((path, dict(payload)))
        if path.endswith("/boost"):
            return {"boost": True}
        return {"select": payload.get("select")}

    monkeypatch.setattr(
        DucaheatRESTClient, "_post_segmented", fake_post_segmented, raising=False
    )

    rtc_mock = AsyncMock(
        return_value={"y": 2024, "n": 1, "d": 1, "h": 0, "m": 0, "s": 0}
    )
    monkeypatch.setattr(ducaheat_rest_client, "get_rtc_time", rtc_mock)

    result = await ducaheat_rest_client.set_acm_boost_state(
        "dev", "3", boost=True, boost_time=15
    )

    assert result["boost"] is True
    assert calls == [
        ("/api/v2/devs/dev/acm/3/select", {"select": True}),
        ("/api/v2/devs/dev/acm/3/boost", {"boost": True, "boost_time": 15}),
        ("/api/v2/devs/dev/acm/3/select", {"select": False}),
    ]
    rtc_mock.assert_awaited_once_with("dev")


@pytest.mark.asyncio
async def test_ducaheat_rest_get_node_samples_forwards_non_htr(
    ducaheat_rest_client: DucaheatRESTClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    async def fake_super(
        self, dev_id: str, node: tuple[str, str], start: float, stop: float
    ):
        captured["args"] = (dev_id, node, start, stop)
        return [{"t": 1}]

    monkeypatch.setattr(RESTClient, "get_node_samples", fake_super)

    result = await ducaheat_rest_client.get_node_samples("dev", ("acm", "7"), 1.0, 2.0)
    assert result == [{"t": 1}]
    assert captured["args"] == ("dev", ("acm", "7"), 1.0, 2.0)


def test_ducaheat_rest_normalise_ws_nodes_prog() -> None:
    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    nodes = {
        "acm": {
            "settings": {
                "02": {
                    "prog": {
                        "prog": {
                            str(day): [day % 3] * 48 for day in range(7)
                        }
                    },
                    "mode": "auto",
                }
            },
            "status": {"02": {"temp": 21}},
        }
    }

    result = client.normalise_ws_nodes(nodes)
    settings = result["acm"]["settings"]["02"]
    assert len(settings["prog"]) == 168
    assert settings["prog"][24:48] == [1] * 24
    # Original payload should remain unchanged
    assert len(nodes["acm"]["settings"]["02"]["prog"]["prog"]["1"]) == 48


def test_ducaheat_rest_normalise_ws_nodes_passthrough() -> None:
    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    assert client.normalise_ws_nodes(["bad"]) == ["bad"]

    nodes = {"htr": [1, 2, 3]}
    assert client.normalise_ws_nodes(nodes)["htr"] == [1, 2, 3]

    nodes_with_scalar = {"htr": {"settings": {"01": 5}}}
    normalised = client.normalise_ws_nodes(nodes_with_scalar)
    assert normalised["htr"]["settings"]["01"] == 5


def test_sanitize_helpers_mask_sensitive_tokens() -> None:
    assert redact_text("") == ""
    sample = "Bearer abc token=secret user@example.com"
    redacted = redact_text(sample)
    assert "secret" not in redacted
    assert "user@example.com" not in redacted
    assert "Bearer ***" in redacted
    assert "token=***" in redacted
    assert "***@***" in redacted

    assert redact_token_fragment("   ") == ""
    assert redact_token_fragment("abcd") == "***"
    assert redact_token_fragment("abcdefgh") == "ab***gh"

    assert mask_identifier("   ") == ""
    assert mask_identifier("abcd") == "***"
    assert mask_identifier("abcdefgh") == "ab...gh"
    assert mask_identifier("abcdefghijklmnop") == "abcdef...mnop"

    class _Blank:
        def __bool__(self) -> bool:
            return True

        def __str__(self) -> str:
            return ""

    assert redact_text(_Blank()) == ""
    assert redact_token_fragment(None) == ""
    assert mask_identifier(None) == ""


def test_validate_boost_minutes_and_payload() -> None:
    assert validate_boost_minutes(None) is None
    assert validate_boost_minutes(15) == 15
    assert build_acm_boost_payload(True, None) == {"boost": True}
    assert build_acm_boost_payload(False, 30) == {"boost": False, "boost_time": 30}

    with pytest.raises(ValueError):
        validate_boost_minutes(0)
    with pytest.raises(ValueError):
        validate_boost_minutes("bad")
    with pytest.raises(ValueError):
        build_acm_boost_payload(True, 0)


def test_ducaheat_log_segmented_post_noop_when_not_debug(
    ducaheat_rest_client: DucaheatRESTClient, caplog: pytest.LogCaptureFixture
) -> None:
    logger_name = "custom_components.termoweb.backend.ducaheat"
    caplog.set_level(logging.INFO, logger=logger_name)
    caplog.clear()

    ducaheat_rest_client._log_segmented_post(
        path="https://example.invalid/path?token=abc",
        node_type="acm",
        dev_id="device@example.com",
        addr="03",
        payload={"mode": "auto"},
    )

    assert caplog.records == []

    caplog.set_level(logging.DEBUG, logger=logger_name)
    caplog.clear()

    ducaheat_rest_client._log_segmented_post(
        path="https://example.invalid/path?token=abc",
        node_type="acm",
        dev_id="device@example.com",
        addr="03",
        payload={"mode": "auto"},
    )

    assert "body_keys=('mode',)" in caplog.text


def test_ducaheat_log_segmented_post_handles_non_mapping(
    ducaheat_rest_client: DucaheatRESTClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(
        logging.DEBUG, logger="custom_components.termoweb.backend.ducaheat"
    )
    ducaheat_rest_client._log_segmented_post(
        path="https://example.invalid/path?token=abc",
        node_type="acm",
        dev_id="device@example.com",
        addr="03",
        payload=["unexpected"],
    )
    assert "<non-mapping>" in caplog.text
    assert "token=***" in caplog.text
    assert "device....com" in caplog.text
    caplog.clear()
    ducaheat_rest_client._log_segmented_post(
        path="https://example.invalid/path?token=abc",
        node_type="acm",
        dev_id="device@example.com",
        addr="03",
        payload={"mode": "auto"},
    )
    assert "('mode',)" in caplog.text
    caplog.clear()
    ducaheat_rest_client._log_segmented_post(
        path="https://example.invalid/path?token=abc",
        node_type="acm",
        dev_id="device@example.com",
        addr="03",
        payload=None,
    )
    assert "body_keys=()" in caplog.text
