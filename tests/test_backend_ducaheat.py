import asyncio
import logging
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientResponseError

from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.backend.base import BoostContext
from custom_components.termoweb.backend.ducaheat import (
    DucaheatBackend,
    DucaheatRESTClient,
)
from custom_components.termoweb.backend.sanitize import (
    build_acm_boost_payload,
    mask_identifier,
    redact_text,
    redact_token_fragment,
)
from custom_components.termoweb.boost import validate_boost_minutes
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
        cancel_boost: bool = False,
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

    async def authed_headers(self) -> dict[str, str]:  # pragma: no cover - stub
        return {"Authorization": "Bearer token"}


@pytest.fixture
def ducaheat_rest_client(monkeypatch: pytest.MonkeyPatch) -> DucaheatRESTClient:
    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_headers() -> dict[str, str]:
        return {"Authorization": "Bearer token"}

    monkeypatch.setattr(client, "authed_headers", fake_headers)
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("context", "expected_cancel"),
    [
        (BoostContext(active=True), True),
        (BoostContext(active=False), False),
        (BoostContext(active=None, mode="boost"), True),
        (BoostContext(active=None, mode="auto"), False),
        (None, False),
    ],
)
async def test_ducaheat_backend_cancel_boost_heuristic(
    context: BoostContext | None,
    expected_cancel: bool,
) -> None:
    """Ensure Ducaheat backend maps boost hints into cancel_boost flags."""

    client = AsyncMock()
    backend = DucaheatBackend(brand="ducaheat", client=client)

    await backend.set_node_settings(
        "dev-2",
        ("acm", "9"),
        mode="auto",
        stemp=20.0,
        units="C",
        boost_context=context,
    )

    client.set_node_settings.assert_awaited_once_with(
        "dev-2",
        ("acm", "9"),
        mode="auto",
        stemp=20.0,
        prog=None,
        ptemp=None,
        units="C",
        cancel_boost=expected_cancel,
    )


@pytest.mark.asyncio
async def test_ducaheat_backend_skips_cancel_boost_for_non_acm() -> None:
    """Ensure non-accumulator writes never request boost cancellation."""

    client = AsyncMock()
    backend = DucaheatBackend(brand="ducaheat", client=client)

    await backend.set_node_settings(
        "dev-3",
        ("htr", "1"),
        mode="auto",
        stemp=19.0,
        units="F",
        boost_context=BoostContext(active=True),
    )

    client.set_node_settings.assert_awaited_once_with(
        "dev-3",
        ("htr", "1"),
        mode="auto",
        stemp=19.0,
        prog=None,
        ptemp=None,
        units="F",
        cancel_boost=False,
    )


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
    assert result == {"power": 0}
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

    result = await ducaheat_rest_client.get_node_settings("dev", ("acm", "2"))
    assert result == {"mode": "auto"}
    assert seen["path"] == "/api/v2/devs/dev/acm/2"
    assert seen["method"] == "GET"
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
async def test_ducaheat_rest_set_htr_mode_uses_status_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure heater mode changes are sent via the /status segment."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_headers() -> dict[str, str]:
        return {"Authorization": "Bearer token"}

    monkeypatch.setattr(client, "authed_headers", fake_headers)

    calls: list[tuple[str, Mapping[str, Any], str]] = []

    async def fake_post_segmented(
        path: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        dev_id: str,
        addr: str,
        node_type: str,
        ignore_statuses: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        calls.append((path, dict(payload), node_type))
        return {}

    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    result = await client.set_node_settings("dev", ("htr", "2"), mode="auto")

    assert result == {"status": {}}

    assert calls[0][0].endswith("/select")
    assert calls[0][1] == {"select": True}
    assert calls[0][2] == "htr"

    status_calls = [call for call in calls if call[0].endswith("/status")]
    assert status_calls == [("/api/v2/devs/dev/htr/2/status", {"mode": "auto"}, "htr")]

    assert all(not path.endswith("/mode") for path, _, _ in calls)

    assert calls[-1][0].endswith("/select")
    assert calls[-1][1] == {"select": False}
    assert calls[-1][2] == "htr"


@pytest.mark.asyncio
async def test_ducaheat_rest_set_htr_mode_preserves_modified_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure modified_auto mode is posted without being coerced."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    monkeypatch.setattr(
        client,
        "authed_headers",
        AsyncMock(return_value={"Authorization": "token"}),
    )

    posted_payloads: list[dict[str, Any]] = []

    async def fake_post_segmented(
        path: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        dev_id: str,
        addr: str,
        node_type: str,
        ignore_statuses: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        if path.endswith("/status"):
            posted_payloads.append(dict(payload))
        return {}

    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    await client.set_node_settings("dev", ("htr", "2"), mode=" modified_auto ")

    assert posted_payloads == [{"mode": "modified_auto"}]


@pytest.mark.asyncio
async def test_ducaheat_rest_set_htr_full_segment_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure heater updates emit status, prog, and prog_temps segments."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    monkeypatch.setattr(
        client,
        "authed_headers",
        AsyncMock(return_value={"Authorization": "token"}),
    )

    select_calls: list[bool] = []

    async def fake_select(**kwargs: Any) -> None:
        select_calls.append(bool(kwargs.get("select")))

    monkeypatch.setattr(
        client, "_select_segmented_node", AsyncMock(side_effect=fake_select)
    )

    async def fake_post_segmented(
        path: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        dev_id: str,
        addr: str,
        node_type: str,
        ignore_statuses: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        return {"segment": path.rsplit("/", 1)[-1], "payload": dict(payload)}

    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    weekly_prog = [1] * 168
    preset_temps = [10.0, 15.0, 20.0]

    responses = await client.set_node_settings(
        "dev",
        ("htr", "1"),
        mode="heat",
        stemp=21,
        prog=weekly_prog,
        ptemp=preset_temps,
        units=" f ",
    )

    assert set(responses) == {"status", "prog", "prog_temps"}
    assert responses["status"]["payload"] == {
        "mode": "manual",
        "stemp": "21.0",
        "units": "F",
    }
    prog_payload = responses["prog"]["payload"]["prog"]
    assert set(prog_payload) == {str(idx) for idx in range(7)}
    assert all(len(slots) == 48 for slots in prog_payload.values())
    assert all(set(slots) == {1} for slots in prog_payload.values())
    assert responses["prog_temps"]["payload"] == {
        "cold": "10.0",
        "night": "15.0",
        "day": "20.0",
        "units": "F",
    }

    assert select_calls == [True, False]


@pytest.mark.asyncio
async def test_ducaheat_rest_set_htr_units_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure unit-only updates send a single status segment and release."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    monkeypatch.setattr(
        client,
        "authed_headers",
        AsyncMock(return_value={"Authorization": "token"}),
    )

    select_calls: list[bool] = []

    async def fake_select(**kwargs: Any) -> None:
        select_calls.append(bool(kwargs.get("select")))

    monkeypatch.setattr(
        client, "_select_segmented_node", AsyncMock(side_effect=fake_select)
    )

    payloads: dict[str, Mapping[str, Any]] = {}

    async def fake_post_segmented(
        path: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        dev_id: str,
        addr: str,
        node_type: str,
        ignore_statuses: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        payloads[path] = dict(payload)
        return {"segment": path.rsplit("/", 1)[-1], "payload": dict(payload)}

    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    responses = await client.set_node_settings("dev", ("htr", "9"), units="F")

    assert responses == {"status": {"segment": "status", "payload": {"units": "F"}}}
    assert payloads == {"/api/v2/devs/dev/htr/9/status": {"units": "F"}}
    assert select_calls == [True, False]


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

    monkeypatch.setattr(ducaheat_rest_client, "authed_headers", fake_headers)

    with pytest.raises(ValueError):
        await ducaheat_rest_client.set_node_settings("dev", ("acm", "3"), **kwargs)


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
                    "prog": {"prog": {str(day): [day % 3] * 48 for day in range(7)}},
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


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (60, 60),
        ("120", 120),
        (300.0, 300),
    ],
)
def test_validate_boost_minutes_accepts_valid_inputs(
    value: int | str | float | None, expected: int | None
) -> None:
    assert validate_boost_minutes(value) == expected


@pytest.mark.parametrize(
    "value",
    [0, 59, 61, 90, 601, "bad", 75.0],
)
def test_validate_boost_minutes_rejects_invalid_inputs(value: object) -> None:
    with pytest.raises(ValueError):
        validate_boost_minutes(value)  # type: ignore[arg-type]


def test_build_acm_boost_payload_normalises_optional_fields() -> None:
    payload = build_acm_boost_payload(
        True,
        "180",
        stemp=" 21.5 ",
        units="f",
    )

    assert payload == {
        "boost": True,
        "boost_time": 180,
        "stemp": "21.5",
        "units": "F",
    }


def test_build_acm_boost_payload_rejects_empty_stemp() -> None:
    baseline = build_acm_boost_payload(True, 120, stemp="20", units="C")

    with pytest.raises(ValueError):
        build_acm_boost_payload(False, 60, stemp="   ")

    assert baseline == {
        "boost": True,
        "boost_time": 120,
        "stemp": "20",
        "units": "C",
    }


def test_build_acm_boost_payload_rejects_invalid_units() -> None:
    baseline = build_acm_boost_payload(True, 60)

    with pytest.raises(ValueError):
        build_acm_boost_payload(True, 120, stemp="21", units="kelvin")

    assert baseline == {"boost": True, "boost_time": 60}


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
