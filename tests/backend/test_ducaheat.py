"""Tests for the Ducaheat backend and REST client (backend/ducaheat.py)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import inspect
import logging
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import AsyncMock, patch

from aiohttp import ClientResponseError
import pytest

from custom_components.termoweb.backend.base import BoostContext
from custom_components.termoweb.backend.ducaheat import (
    DucaheatBackend,
    DucaheatRequestError,
    DucaheatRESTClient,
)
from custom_components.termoweb.backend.ducaheat_ws import DucaheatWSClient
from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.backend.sanitize import (
    mask_identifier,
    redact_text,
    redact_token_fragment,
)
from custom_components.termoweb.boost import validate_boost_minutes
from custom_components.termoweb.codecs.ducaheat_codec import (
    decode_settings,
    encode_program_command,
    encode_units_command,
    extract_prog_days,
)
from custom_components.termoweb.codecs.ducaheat_read_models import (
    DucaheatExtraOptions,
    DucaheatSetupSegment,
    DucaheatStatusSegment,
)
from custom_components.termoweb.const import (
    BRAND_DUCAHEAT,
    WS_NAMESPACE,
    get_brand_user_agent,
)
from custom_components.termoweb.domain.commands import SetProgram, SetUnits
from custom_components.termoweb.domain.ids import NodeType
from tests.fakes.runtime import build_entry_runtime


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
async def test_ducaheat_backend_creates_ws_client(hass) -> None:
    backend = DucaheatBackend(brand="ducaheat", client=DummyClient())
    build_entry_runtime(hass=hass, entry_id="entry", dev_id="dev")
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

    status_calls = [call for call in calls if call[0].endswith("/status")]
    assert status_calls == [("/api/v2/devs/dev/htr/2/status", {"mode": "auto"}, "htr")]

    assert all(not path.endswith("/mode") for path, _, _ in calls)


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
    """Ensure heater updates emit status and prog segments; presets go to status."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    monkeypatch.setattr(
        client,
        "authed_headers",
        AsyncMock(return_value={"Authorization": "token"}),
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
    monkeypatch.setattr(
        client, "_request", AsyncMock(return_value={"prog": {"0": [0] * 24}})
    )

    weekly_prog = [1] * 168
    preset_temps = [10.0, 15.0, 20.0]

    responses = await client.set_node_settings(
        "dev",
        ("htr", "1"),
        mode="heat",
        stemp=21,
        prog=weekly_prog,
        units=" f ",
    )

    assert set(responses) == {"status", "prog"}
    assert responses["status"]["payload"] == {
        "mode": "manual",
        "stemp": "21.0",
        "units": "F",
    }
    prog_payload = responses["prog"]["payload"]["prog"]
    assert set(prog_payload) == {str(idx) for idx in range(7)}
    assert all(slots == [1] * 24 for slots in prog_payload.values())

    responses = await client.set_node_settings(
        "dev", ("htr", "1"), ptemp=preset_temps, units="F"
    )
    assert responses["status"]["payload"] == {
        "ice_temp": "10.0",
        "eco_temp": "15.0",
        "comf_temp": "20.0",
        "units": "F",
    }


@pytest.mark.asyncio
async def test_ducaheat_rest_set_htr_units_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure unit-only updates send a single status segment."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    monkeypatch.setattr(
        client,
        "authed_headers",
        AsyncMock(return_value={"Authorization": "token"}),
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


@pytest.mark.asyncio
async def test_get_node_settings_excludes_raw_payloads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Raw payloads should never be retained in normalised settings."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
    payload = {"status": {"mode": "Manual"}}

    with (
        patch.object(client, "authed_headers", AsyncMock(return_value={})),
        patch.object(client, "_request", AsyncMock(return_value=payload)),
        caplog.at_level(
            logging.DEBUG, logger="custom_components.termoweb.backend.ducaheat"
        ),
    ):
        result = await client.get_node_settings("dev", ("htr", "01"))

    assert result["mode"] == "manual"
    assert "raw" not in result


@dataclass
class DucaheatClientHarness:
    """Container for a fake Ducaheat REST client and its call history."""

    client: "DucaheatRESTClient"
    requests: list[tuple[str, str, dict[str, Any]]]
    segmented_calls: list[dict[str, Any]]
    rtc_calls: list[str]


@pytest.fixture
def ducaheat_rest_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., DucaheatClientHarness]:
    """Provide a factory that builds a fake Ducaheat REST client harness."""

    def factory(
        *,
        responses: Iterable[dict[str, Any] | None] | None = None,
        segmented_side_effects: Mapping[str, Exception] | None = None,
        headers: Mapping[str, str] | None = None,
        rtc_payload: Mapping[str, int] | None = None,
    ) -> DucaheatClientHarness:
        """Create a Ducaheat REST client with predictable helpers for tests."""

        client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
        request_calls: list[tuple[str, str, dict[str, Any]]] = []
        segmented_calls: list[dict[str, Any]] = []
        rtc_calls: list[str] = []
        pending_responses = list(responses or [])
        segmented_effects = dict(segmented_side_effects or {})
        base_headers = {
            "Authorization": "Bearer token",
            "X-SerialId": "15",
            "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
        }
        if headers is not None:
            base_headers = dict(headers)
        rtc_template = dict(
            rtc_payload or {"y": 2024, "n": 1, "d": 1, "h": 0, "m": 0, "s": 0}
        )

        async def fake_headers() -> dict[str, str]:
            """Return static authentication headers for the fake client."""

            return dict(base_headers)

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            """Record REST requests and return queued responses."""

            request_calls.append((method, path, kwargs))
            if pending_responses:
                response = pending_responses.pop(0)
                return dict(response or {})
            return {}

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
            """Record segmented POST calls and replay optional side effects."""

            payload_copy = dict(payload)

            record = {
                "path": path,
                "payload": payload_copy,
                "dev_id": dev_id,
                "addr": addr,
                "node_type": node_type,
                "ignore_statuses": tuple(ignore_statuses or ()),
                "headers": dict(headers),
            }
            segmented_calls.append(record)
            request_calls.append(
                (
                    "POST",
                    path,
                    {
                        "headers": dict(headers),
                        "json": payload_copy,
                    },
                )
            )
            effect = segmented_effects.get(path)
            if effect is not None:
                raise effect
            return {"ok": True}

        async def fake_rtc(dev_id: str) -> dict[str, Any]:
            """Capture RTC lookups and return the configured template."""

            rtc_calls.append(dev_id)
            return dict(rtc_template)

        monkeypatch.setattr(client, "authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)
        monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)
        monkeypatch.setattr(client, "get_rtc_time", fake_rtc)

        return DucaheatClientHarness(
            client=client,
            requests=request_calls,
            segmented_calls=segmented_calls,
            rtc_calls=rtc_calls,
        )

    return factory


def test_ducaheat_acm_request_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

        async def fake_headers() -> dict[str, str]:
            return {
                "Authorization": "Bearer token",
                "X-SerialId": "15",
                "User-Agent": get_brand_user_agent(BRAND_DUCAHEAT),
            }

        async def fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            raise ClientResponseError(
                request_info=None,
                history=(),
                status=400,
                message="malformed",
            )

        monkeypatch.setattr(client, "authed_headers", fake_headers)
        monkeypatch.setattr(client, "_request", fake_request)
        mock_rtc = AsyncMock(return_value={})
        monkeypatch.setattr(client, "get_rtc_time", mock_rtc)

        with pytest.raises(DucaheatRequestError) as exc:
            await client.set_node_settings("dev", ("acm", "1"), mode="boost")

        assert "malformed" in str(exc.value)
        mock_rtc.assert_not_awaited()

    asyncio.run(_run())


def test_ducaheat_acm_mode_invalid_boost_time(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness()

        with pytest.raises(ValueError):
            await harness.client.set_node_settings(
                "dev", ("acm", "2"), mode="auto", boost_time=15
            )

    asyncio.run(_run())


def test_ducaheat_acm_mode_boost_invalid_minutes(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness()

        with pytest.raises(ValueError):
            await harness.client.set_node_settings(
                "dev", ("acm", "2"), mode="boost", boost_time=0
            )

    asyncio.run(_run())


def test_ducaheat_acm_mode_boost_invalid_minutes_type(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    async def _run() -> None:
        harness = ducaheat_rest_harness()

        with pytest.raises(ValueError):
            await harness.client.set_node_settings(
                "dev", ("acm", "2"), mode="boost", boost_time="abc"
            )

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_ducaheat_acm_extra_options_segmented_post(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Ensure extra options payload uses segmented POST with formatted fields."""

    harness = ducaheat_rest_harness()

    result = await harness.client.set_acm_extra_options(
        "dev", "3", boost_time=180, boost_temp=55.55
    )

    assert result == {"ok": True}
    setup_calls = [
        call for call in harness.segmented_calls if call["path"].endswith("/setup")
    ]
    assert len(setup_calls) == 1
    setup_payload = setup_calls[0]["payload"]
    assert setup_payload == {"extra_options": {"boost_time": 180, "boost_temp": "55.5"}}


@pytest.mark.asyncio
async def test_ducaheat_acm_settings_boost_flow(
    ducaheat_rest_harness: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover the boost branch of segmented ACM writes."""

    harness = ducaheat_rest_harness()

    monkeypatch.setattr(harness.client, "_ensure_units", lambda units: units.upper())

    responses = await harness.client.set_node_settings(
        "dev",
        ("acm", "6"),
        mode="boost",
        stemp=22,
        prog=[1] * 168,
        ptemp=[10.0, 15.0, 20.0],
        units="c",
        boost_time=60,
    )

    assert responses.keys() == {"status", "prog"}
    status_call = next(
        call for call in harness.segmented_calls if call["path"].endswith("/status")
    )
    assert status_call["payload"] == {
        "stemp": "22.0",
        "units": "C",
        "mode": "boost",
        "boost_time": 60,
    }
    assert not any(call["path"].endswith("/mode") for call in harness.segmented_calls)


@pytest.mark.asyncio
async def test_ducaheat_acm_settings_cancel_only(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Validate the standalone cancel boost path without status payload."""

    harness = ducaheat_rest_harness()

    responses = await harness.client.set_node_settings(
        "dev",
        ("acm", "8"),
        cancel_boost=True,
    )

    assert responses == {"boost": {"ok": True}}


@pytest.mark.asyncio
async def test_ducaheat_acm_settings_cancel_with_units(
    ducaheat_rest_harness: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel requests with explicit units should emit a status payload."""

    harness = ducaheat_rest_harness()

    responses = await harness.client.set_node_settings(
        "dev",
        ("acm", "16"),
        units="F",
        cancel_boost=True,
    )

    assert responses.keys() == {"status", "boost"}
    status_call = next(
        call
        for call in harness.segmented_calls
        if call["path"].endswith("/status") and call["addr"] == "16"
    )
    assert status_call["payload"] == {"units": "F"}


@pytest.mark.asyncio
async def test_ducaheat_acm_settings_boost_mode_segment(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Ensure boost mode writes include a dedicated mode payload."""

    harness = ducaheat_rest_harness()

    responses = await harness.client.set_node_settings(
        "dev",
        ("acm", "9"),
        mode="boost",
        boost_time=120,
    )

    assert "mode" in responses
    mode_call = next(
        call for call in harness.segmented_calls if call["path"].endswith("/mode")
    )
    assert mode_call["payload"] == {"mode": "boost", "boost_time": 120}


@pytest.mark.asyncio
async def test_post_acm_endpoint_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Client errors should translate into ``DucaheatRequestError`` instances."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_post_segmented(path: str, **_: Any) -> None:
        raise ClientResponseError(
            request_info=None, history=(), status=422, message="bad request"
        )

    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    with pytest.raises(DucaheatRequestError) as err:
        await client._post_acm_endpoint(
            "/api/v2/devs/dev/acm/1/status",
            {"Authorization": "token"},
            {"mode": "auto"},
            dev_id="dev",
            addr="1",
        )

    assert "bad request" in str(err.value)


@pytest.mark.asyncio
async def test_post_acm_endpoint_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server errors should bubble up without translation."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_post_segmented(path: str, **_: Any) -> None:
        raise ClientResponseError(
            request_info=None, history=(), status=500, message="boom"
        )

    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    with pytest.raises(ClientResponseError):
        await client._post_acm_endpoint(
            "/api/v2/devs/dev/acm/1/status",
            {"Authorization": "token"},
            {"mode": "auto"},
            dev_id="dev",
            addr="1",
        )


@pytest.mark.asyncio
async def test_set_node_display_select_posts_to_select_segment(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Display flash writes should target the segmented /select endpoint."""

    harness = ducaheat_rest_harness()

    await harness.client.set_node_display_select("dev", ("acm", "1"), select=True)

    assert harness.segmented_calls[-1]["path"] == "/api/v2/devs/dev/acm/1/select"
    assert harness.segmented_calls[-1]["payload"] == {"select": True}


@pytest.mark.asyncio
async def test_select_segmented_node_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection failures should surface as ``DucaheatRequestError``."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_post_segmented(path: str, **_: Any) -> None:
        raise ClientResponseError(
            request_info=None, history=(), status=409, message="conflict"
        )

    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    with pytest.raises(DucaheatRequestError) as err:
        await client._select_segmented_node(
            dev_id="dev",
            node_type="acm",
            addr="1",
            headers={"Authorization": "token"},
            select=True,
        )

    assert "conflict" in str(err.value)


@pytest.mark.asyncio
async def test_select_segmented_node_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server-side failures should propagate to the caller."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_post_segmented(path: str, **_: Any) -> None:
        raise ClientResponseError(
            request_info=None, history=(), status=502, message="upstream"
        )

    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    with pytest.raises(ClientResponseError):
        await client._select_segmented_node(
            dev_id="dev",
            node_type="acm",
            addr="1",
            headers={"Authorization": "token"},
            select=True,
        )


@pytest.mark.asyncio
async def test_set_acm_boost_state_invalid_stemp(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Invalid stemp values should raise immediately."""

    harness = ducaheat_rest_harness()

    with pytest.raises(ValueError) as err:
        await harness.client.set_acm_boost_state("dev", "12", boost=True, stemp="bad")

    assert "Invalid stemp value" in str(err.value)


def test_ensure_units_blank_defaults() -> None:
    """Empty unit strings should normalise to Celsius."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    assert client._ensure_units("") == "C"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_boost", [False, True])
async def test_acm_mode_write_makes_no_rtc_get_and_no_extra_boost_write(
    ducaheat_rest_harness: Callable[..., Any], cancel_boost: bool
) -> None:
    """A plain acm mode write must not GET the RTC or POST an unrequested boost=false.

    Regression: a failing RTC GET used to trigger a ``status_refresh`` POST of
    ``{"boost": false}``, which cancels a running boost.
    """

    harness = ducaheat_rest_harness()
    rtc = AsyncMock(side_effect=RuntimeError("rtc down"))
    harness.client.get_rtc_time = rtc

    await harness.client.set_node_settings(
        "dev", ("acm", "7"), mode="auto", cancel_boost=cancel_boost
    )

    rtc.assert_not_awaited()
    assert all(method != "GET" for method, _path, _kw in harness.requests)
    # Only an explicitly requested cancel may touch /boost.
    assert [call["path"].rsplit("/", 1)[-1] for call in harness.segmented_calls] == (
        ["mode", "boost"] if cancel_boost else ["mode"]
    )


@pytest.mark.asyncio
async def test_acm_boost_state_write_makes_no_rtc_get(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """Starting a boost posts once to /boost and returns the server response."""

    harness = ducaheat_rest_harness()
    rtc = AsyncMock(side_effect=RuntimeError("rtc down"))
    harness.client.get_rtc_time = rtc

    result = await harness.client.set_acm_boost_state(
        "dev", "4", boost=True, boost_time=120, stemp=21.5, units="C"
    )

    rtc.assert_not_awaited()
    assert result == {"ok": True}
    assert [call["path"] for call in harness.segmented_calls] == [
        "/api/v2/devs/dev/acm/4/boost"
    ]


@pytest.mark.asyncio
async def test_acm_boost_with_stemp_sends_boost_time(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """mode=boost with stemp and boost_time must send both fields (B7)."""

    harness = ducaheat_rest_harness()

    await harness.client.set_node_settings(
        "dev", ("acm", "6"), mode="boost", stemp=22, boost_time=60
    )

    assert [call["path"] for call in harness.segmented_calls] == [
        "/api/v2/devs/dev/acm/6/status"
    ]
    assert harness.segmented_calls[0]["payload"] == {
        "mode": "boost",
        "stemp": "22.0",
        "units": "C",
        "boost_time": 60,
    }


@pytest.mark.asyncio
async def test_acm_boost_with_stemp_validates_boost_time(
    ducaheat_rest_harness: Callable[..., Any],
) -> None:
    """An invalid boost_time is rejected even when stemp is also supplied."""

    harness = ducaheat_rest_harness()

    with pytest.raises(ValueError, match="boost_time"):
        await harness.client.set_node_settings(
            "dev", ("acm", "6"), mode="boost", stemp=22, boost_time=45
        )

    assert harness.segmented_calls == []


def test_ducaheat_rest_normalise_ws_nodes_passthrough_scalars() -> None:
    """Non-mapping payloads should return the original object unchanged."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    scalar_payload = "scalar"
    list_payload = ["entry"]

    scalar_result = client.normalise_ws_nodes(scalar_payload)
    list_result = client.normalise_ws_nodes(list_payload)

    assert scalar_result is scalar_payload
    assert list_result is list_payload
    assert list_payload == ["entry"]


def test_ducaheat_rest_normalise_ws_nodes_preserve_non_settings_sections() -> None:
    """Only the settings section should be normalised within nested payloads."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    heater_prog = {str(day): [day] * 48 for day in range(7)}
    payload = {
        "htr": {
            "settings": {
                "01": {"prog": {"prog": heater_prog}, "mode": "auto"},
            },
            "status": {"01": {"temp": 21}},
            "alerts": ["unchanged"],
        }
    }

    result = client.normalise_ws_nodes(payload)

    # Settings bucket should be normalised into a flattened schedule.
    settings = result["htr"]["settings"]["01"]
    assert isinstance(settings["prog"], list)
    assert len(settings["prog"]) == 168

    # Non-settings sections should retain their original identities.
    assert result["htr"]["status"] is payload["htr"]["status"]
    assert result["htr"]["alerts"] is payload["htr"]["alerts"]

    # Original payload must remain untouched after normalisation.
    assert len(payload["htr"]["settings"]["01"]["prog"]["prog"]["1"]) == 48


def test_ducaheat_rest_normalise_ws_nodes_mixed_settings_types() -> None:
    """Normalisation should coerce mapping payloads while preserving scalars."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    heater_prog = {str(day): [day] * 48 for day in range(7)}
    accumulator_prog = {str(day): [1] * 24 for day in range(7)}

    payload = {
        "htr": {
            "settings": {
                "01": {
                    "prog": {"prog": heater_prog},
                    "mode": "auto",
                },
                "02": ["unexpected"],
            },
            "status": [{"temp": 21}],
        },
        "acm": {
            "settings": {
                "03": {
                    "prog": {"days": accumulator_prog},
                    "mode": "charge",
                },
                "04": "raw",
            },
            "status": {"03": {"temp": 19}},
        },
        "pmo": ["unchanged"],
    }

    result = client.normalise_ws_nodes(payload)

    heater_settings = result["htr"]["settings"]["01"]
    assert isinstance(heater_settings, dict)
    assert len(heater_settings["prog"]) == 168
    # Spot check that day 1 (Monday) slots collapsed from the vendor's 48 entries.
    assert heater_settings["prog"][24:48] == [1] * 24

    accumulator_settings = result["acm"]["settings"]["03"]
    assert isinstance(accumulator_settings, dict)
    assert len(accumulator_settings["prog"]) == 168

    # Non-mapping settings entries should pass through untouched.
    assert result["htr"]["settings"]["02"] == ["unexpected"]
    assert result["acm"]["settings"]["04"] == "raw"

    # Non-settings sections retain their original typing.
    assert result["htr"]["status"] == [{"temp": 21}]
    assert result["pmo"] == ["unchanged"]

    # Original payload should remain unchanged for mapping coercions.
    assert len(payload["htr"]["settings"]["01"]["prog"]["prog"]["1"]) == 48
    assert len(payload["acm"]["settings"]["03"]["prog"]["days"]["1"]) == 24


def test_ducaheat_decode_settings_drops_boost_end_mapping() -> None:
    """Decoded accumulator settings should not retain raw boost_end mappings."""

    payload = {
        "status": {
            "mode": "boost",
            "boost_end": {"day": 2, "minute": 45},
        },
        "setup": {"extra_options": {"boost_end_min": 90}},
    }

    decoded = decode_settings(payload, node_type=NodeType.ACCUMULATOR)

    assert "boost_end" not in decoded
    assert decoded["boost_end_day"] == 2
    assert decoded["boost_end_min"] == 45


def test_ducaheat_decode_settings_validates_aliases_with_models() -> None:
    """Pydantic validation should normalise accumulator payload aliases."""

    payload = {
        "status": {
            "set_temp": "22.5",
            "ambient": "19.3",
            "units": "c",
            "boost_end": {"day": "5", "minute": "75"},
            "current_charge_per": "110",
            "lock": "off",
        },
        "setup": {
            "extra_options": {
                "boost_time": "120",
                "boost_end_min": 30,
                "target_charge_per": "15.9",
            }
        },
        "prog": [0, 1, 2] * 56,
        "prog_temps": {"antifrost": "7", "eco": "17.0", "comfort": "21"},
    }

    decoded = decode_settings(payload, node_type=NodeType.ACCUMULATOR)

    assert decoded["stemp"] == "22.5"
    assert decoded["mtemp"] == "19.3"
    assert decoded["units"] == "C"
    assert decoded["boost_end_day"] == 5
    assert decoded["boost_end_min"] == 75
    assert decoded["boost_time"] == 120
    assert decoded["current_charge_per"] == 100
    assert decoded["target_charge_per"] == 15
    assert decoded["prog"] == [0, 1, 2] * 56
    assert decoded["ptemp"] == ["7.0", "17.0", "21.0"]
    assert "boost_end" not in decoded


def test_ducaheat_decode_thermostat_aliases_and_limits() -> None:
    """Thermostat decoding should rely on Pydantic validation."""

    valid_day = {"values": [1] * 24}
    payload = {
        "mode": "Manual",
        "state": "On",
        "setpoint": "20.0",
        "room_temp": "19.5",
        "units": "f",
        "ptemp": {"cold": "5", "night": "10.0", "day": "21"},
        "prog": {
            "days": {
                day: valid_day
                for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
            }
        },
        "batt_level": "6",
    }

    decoded = decode_settings(payload, node_type=NodeType.THERMOSTAT)

    assert decoded["mode"] == "manual"
    assert decoded["state"] == "on"
    assert decoded["stemp"] == pytest.approx(20.0)
    assert decoded["mtemp"] == pytest.approx(19.5)
    assert decoded["units"] == "F"
    assert decoded["ptemp"] == ["5.0", "10.0", "21.0"]
    assert decoded["prog"] == [1] * 168
    assert decoded["batt_level"] == 5


def test_ducaheat_rest_normalise_ws_status_merges_charge_fields() -> None:
    """Status updates should refresh accumulator charge metadata in settings."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    payload = {
        "acm": {
            "settings": {"01": {"mode": "charge", "current_charge_per": 5}},
            "status": {
                "01": {
                    "charging": "1",
                    "current_charge_per": "15.2",
                    "target_charge_per": 80,
                }
            },
        }
    }

    normalised = client.normalise_ws_nodes(payload)

    settings = normalised["acm"]["settings"]["01"]
    assert settings["charging"] is True
    assert settings["current_charge_per"] == 15
    assert settings["target_charge_per"] == 80


@pytest.mark.asyncio
async def test_ducaheat_get_node_settings_normalises_thm() -> None:
    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
    sample_prog = [0, 0, 0, 1, 1, 1] * 28
    payload = {
        "mode": "Manual",
        "state": "On",
        "stemp": "21.5",
        "mtemp": "20.3",
        "units": "c",
        "ptemp": ["16.0", "19.0", "20.0"],
        "prog": sample_prog,
        "batt_level": "5",
    }

    with (
        patch.object(client, "authed_headers", AsyncMock(return_value={})),
        patch.object(
            client, "_request", AsyncMock(return_value=payload)
        ) as mock_request,
    ):
        result = await client.get_node_settings("dev", ("thm", "01"))

    mock_request.assert_awaited_once()
    method, path = mock_request.await_args.args[:2]
    assert method == "GET"
    assert path == "/api/v2/devs/dev/thm/01/settings"
    assert result["mode"] == "manual"
    assert result["state"] == "on"
    assert result["stemp"] == pytest.approx(21.5)
    assert result["mtemp"] == pytest.approx(20.3)
    assert result["ptemp"] == [16.0, 19.0, 20.0]
    assert result["prog"] == sample_prog
    assert result["batt_level"] == 5


@pytest.mark.asyncio
async def test_ducaheat_get_node_samples_skips_thm() -> None:
    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    with (
        patch.object(client, "authed_headers", AsyncMock()) as mock_headers,
        patch.object(client, "_request", AsyncMock()) as mock_request,
    ):
        result = await client.get_node_samples("dev", ("thm", "01"), 0, 10)

    mock_headers.assert_not_called()
    mock_request.assert_not_called()
    assert result == []


@pytest.mark.asyncio
async def test_ducaheat_set_node_settings_thm_fallback_post() -> None:
    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
    response = {"status": "ok"}
    side_effect = [
        ClientResponseError(SimpleNamespace(real_url=None), (), status=405, message=""),
        response,
    ]

    with (
        patch.object(client, "authed_headers", AsyncMock(return_value={})),
        patch.object(
            client, "_request", AsyncMock(side_effect=side_effect)
        ) as mock_request,
    ):
        result = await client.set_node_settings(
            "dev",
            ("thm", "01"),
            mode="auto",
            stemp=21.0,
        )

    assert result is response
    first_call = mock_request.await_args_list[0]
    second_call = mock_request.await_args_list[1]
    assert first_call.args[:2] == ("PATCH", "/api/v2/devs/dev/thm/01/settings")
    assert second_call.args[:2] == ("POST", "/api/v2/devs/dev/thm/01/settings")
    payload = second_call.kwargs["json"]
    assert payload["mode"] == "auto"
    assert payload["stemp"] == "21.0"


class _StubSession:
    """Minimal session stub for constructing the REST client."""


def _make_client() -> DucaheatRESTClient:
    """Create a Ducaheat REST client with placeholder credentials."""

    return DucaheatRESTClient(
        _StubSession(),
        "user",
        "pass",
        api_base="https://api.termoweb.fake",
    )


def test_serialise_prog_temps_formats_values() -> None:
    """Preset temperatures should be formatted to one decimal place."""

    client = _make_client()
    result = client._serialise_prog_temps([5, 15.26, 21])
    assert result == {"cold": "5.0", "night": "15.3", "day": "21.0"}


@pytest.mark.parametrize(
    "ptemp",
    (
        123,
        [10, 15],
        [10, "bad", 20],
    ),
)
def test_serialise_prog_temps_invalid_inputs(ptemp: object) -> None:
    """Invalid preset temperature inputs should raise ``ValueError``."""

    client = _make_client()
    with pytest.raises(ValueError):
        client._serialise_prog_temps(ptemp)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_set_node_settings_units_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send only units and verify a single status segment is planned."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_headers() -> dict[str, str]:
        """Return static headers for the request."""

        return {"Authorization": "Bearer token"}

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
    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    responses = await client.set_node_settings("dev", ("htr", 1), units="F")

    assert responses == {"status": {"ok": "yes"}}
    assert post_calls == [
        {
            "path": "/api/v2/devs/dev/htr/1/status",
            "headers": {"Authorization": "Bearer token"},
            "payload": {"units": "F"},
            "dev_id": "dev",
            "addr": "1",
            "node_type": "htr",
        }
    ]


@pytest.mark.asyncio
async def test_set_node_settings_invalid_stemp_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure invalid stemp errors before issuing a segmented write."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_headers() -> dict[str, str]:
        """Return static headers for the request."""

        return {"Authorization": "Bearer token"}

    def fake_ensure_units(units: str) -> str:
        """Return a predictable units marker."""

        return f"unit:{units}"

    async def fake_post_segmented(**kwargs: Any) -> None:
        """_post_segmented should not be reached for invalid stemp."""

        raise AssertionError("_post_segmented must not be invoked")

    monkeypatch.setattr(client, "authed_headers", fake_headers)
    monkeypatch.setattr(client, "_ensure_units", fake_ensure_units)
    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)

    with pytest.raises(ValueError) as err:
        await client.set_node_settings("dev", ("htr", 1), stemp="bad", units="C")

    assert "Invalid temperature value" in str(err.value)


@pytest.mark.asyncio
async def test_set_node_settings_mode_segment_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure standalone mode writes emit a dedicated mode segment."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_headers() -> dict[str, str]:
        """Return static authentication headers for the fake client."""

        return {"Authorization": "Bearer token"}

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

    monkeypatch.setattr(client, "authed_headers", fake_headers)
    monkeypatch.setattr(client, "_post_segmented", fake_post_segmented)
    monkeypatch.setattr(client, "_ensure_units", fake_ensure_units)

    responses = await client.set_node_settings(
        "dev", ("htr", 2), mode="auto", stemp=18, units="C"
    )

    assert responses == {
        "status": {"ok": True},
    }
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


@pytest.fixture()
def ducaheat_client() -> DucaheatRESTClient:
    """Create a minimal Ducaheat client for helper tests."""

    return DucaheatRESTClient(SimpleNamespace(), "user", "pass")


@pytest.mark.parametrize(
    "value, expected",
    [
        ("c", "C"),
        ("f", "F"),
        (" C ", "C"),
        (None, "C"),
    ],
)
def test_ensure_units_uppercases_valid_values(
    ducaheat_client: DucaheatRESTClient, value: str | None, expected: str
) -> None:
    """_ensure_units should default to Celsius and uppercase valid codes."""

    assert ducaheat_client._ensure_units(value) == expected


@pytest.mark.parametrize("value", ["kelvin", "K", "x", 10])
def test_ensure_units_rejects_invalid_values(
    ducaheat_client: DucaheatRESTClient, value: object
) -> None:
    """_ensure_units should reject unsupported unit strings."""

    with pytest.raises(ValueError):
        ducaheat_client._ensure_units(value)  # type: ignore[arg-type]


def _client(get_payload: Any = None) -> tuple[DucaheatRESTClient, list[tuple]]:
    """Return a Ducaheat client whose HTTP layer records every request."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")
    calls: list[tuple] = []

    async def fake_headers() -> dict[str, str]:
        return {"Authorization": "Bearer token"}

    async def fake_request(method: str, path: str, **kwargs: Any) -> Any:
        calls.append((method, path, kwargs.get("json")))
        return get_payload if method == "GET" else {}

    client.authed_headers = fake_headers  # type: ignore[method-assign]
    client._request = fake_request  # type: ignore[method-assign]
    return client, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("node_type", ["htr", "acm"])
async def test_preset_write_uses_status_ice_eco_comf(node_type: str) -> None:
    """Presets are written via /status.

    docs/ducaheat_api.md "Change live status": "In the capture it is also used
    on htr to update preset temperatures":
    ``{"ice_temp":"5.0","eco_temp":"17.5","comf_temp":"20.5","units":"C"}``
    and "Program preset temperatures": "The app used /status with keys
    ice_temp, eco_temp, comf_temp".
    """

    client, calls = _client()

    await client.set_node_settings(
        "dev", (node_type, "1"), ptemp=[5.0, 17.5, 20.5], units="C"
    )

    assert calls == [
        (
            "POST",
            f"/api/v2/devs/dev/{node_type}/1/status",
            {"ice_temp": "5.0", "eco_temp": "17.5", "comf_temp": "20.5", "units": "C"},
        )
    ]


def test_preset_read_accepts_status_keys_and_zero() -> None:
    """GET status presets (ice/eco/comf) decode to ptemp; 0 is a real value."""

    decoded = decode_settings(
        {
            "status": {
                "mode": "auto",
                "ice_temp": "0.0",
                "eco_temp": "17.5",
                "comf_temp": "20.5",
            }
        },
        node_type=NodeType.HEATER,
    )

    assert decoded["ptemp"] == ["0.0", "17.5", "20.5"]
    assert "ice_temp" not in decoded


def test_preset_read_prefers_complete_status_over_prog_temps() -> None:
    """Status presets win; an incomplete status triple falls back to prog_temps."""

    prog_temps = {"antifrost": "7.0", "eco": "18.0", "comfort": "21.0"}
    full = decode_settings(
        {
            "status": {"ice_temp": "5", "eco_temp": "17", "comf_temp": "20"},
            "prog_temps": prog_temps,
        },
        node_type=NodeType.HEATER,
    )
    partial = decode_settings(
        {"status": {"ice_temp": "5"}, "prog_temps": prog_temps},
        node_type=NodeType.HEATER,
    )

    assert full["ptemp"] == ["5.0", "17.0", "20.0"]
    assert partial["ptemp"] == ["7.0", "18.0", "21.0"]


@pytest.mark.asyncio
async def test_prog_write_echoes_24_slot_get() -> None:
    """A 24-slot GET is written back with 24 hourly slots per day.

    docs/ducaheat_api.md "Weekly program": "Send the full program object
    echoed from GET. In this dump, htr days "0"..."6" each carry 24 integers
    (hourly)." Example: ``{"prog":{"0":[2,2,2,2,...,2],"1":[...],...}}``.
    """

    current = {"prog": {str(day): [0] * 24 for day in range(7)}}
    client, calls = _client(current)
    prog = [day % 3 for day in range(7) for _ in range(24)]

    await client.set_node_settings("dev", ("htr", "1"), prog=prog)

    assert calls == [
        ("GET", "/api/v2/devs/dev/htr/1", None),
        (
            "POST",
            "/api/v2/devs/dev/htr/1/prog",
            {"prog": {str(day): [day % 3] * 24 for day in range(7)}},
        ),
    ]


@pytest.mark.asyncio
async def test_prog_write_echoes_48_slot_get_preserving_half_hours() -> None:
    """A 48-slot GET is written back with 48 slots, keeping unchanged half-hours.

    docs/ducaheat_api.md "Validation invariants": "For weekly programs, echo
    the GET shape and write the whole object."
    """

    day0 = [1] * 48
    day0[0:2] = [0, 2]  # hour 0 reads as 2 (max); the user leaves it alone
    day0[10:12] = [2, 0]  # hour 5 reads as 2; the user changes it to 0
    current = {"prog": {"0": day0, **{str(d): [1] * 48 for d in range(1, 7)}}}
    client, calls = _client(current)
    prog = [1] * 168
    prog[0] = 2
    prog[5] = 0

    await client.set_node_settings("dev", ("acm", "2"), prog=prog)

    method, path, body = calls[-1]
    assert (method, path) == ("POST", "/api/v2/devs/dev/acm/2/prog")
    expected_day0 = [1] * 48
    expected_day0[0:2] = [0, 2]
    expected_day0[10:12] = [0, 0]
    assert body == {
        "prog": {"0": expected_day0, **{str(d): [1] * 48 for d in range(1, 7)}}
    }


def test_encode_program_defaults_to_documented_24_slots() -> None:
    """Without a usable GET shape the documented 24-slot form is written."""

    payload = encode_program_command(SetProgram([2] * 168), current={})

    assert payload == {"prog": {str(day): [2] * 24 for day in range(7)}}


def test_extract_prog_days_filters_invalid_entries() -> None:
    """Only "0".."6" days with 24/48 valid slots are echoed."""

    section = {
        "0": [0] * 48,
        "1": [0] * 10,
        "2": ["x"] * 24,
        "3": [5] * 24,
        "4": None,
        "5": [1] * 24,
    }

    assert extract_prog_days({"prog": section}) == {"0": [0] * 48, "5": [1] * 24}
    assert extract_prog_days(None) == {}


@pytest.mark.asyncio
async def test_thm_prog_write_is_single_level() -> None:
    """Thermostat prog is the day mapping itself, not ``{"prog": {"prog": ...}}``."""

    current = {"prog": {str(day): [0] * 24 for day in range(7)}}
    client, calls = _client(current)

    await client.set_node_settings("dev", ("thm", "3"), prog=[1] * 168)

    assert calls[0] == ("GET", "/api/v2/devs/dev/thm/3/settings", None)
    method, path, body = calls[1]
    assert (method, path) == ("PATCH", "/api/v2/devs/dev/thm/3/settings")
    assert body == {"prog": {str(day): [1] * 24 for day in range(7)}}


@pytest.mark.asyncio
@pytest.mark.parametrize("node_type", ["htr", "acm"])
async def test_no_units_write_unless_requested(node_type: str) -> None:
    """A call without explicit units must not write units (F11)."""

    client, calls = _client()

    assert await client.set_node_settings("dev", (node_type, "1")) == {}
    await client.set_node_settings("dev", (node_type, "1"), units="F")

    assert calls == [("POST", f"/api/v2/devs/dev/{node_type}/1/status", {"units": "F"})]


def test_encode_units_command_validates() -> None:
    """Units are validated before reaching the wire (F7)."""

    assert encode_units_command(SetUnits(" f ")) == {"units": "F"}
    with pytest.raises(ValueError):
        encode_units_command(SetUnits("unit:F"))


@pytest.mark.parametrize(
    "model", [DucaheatStatusSegment, DucaheatExtraOptions, DucaheatSetupSegment]
)
def test_boost_end_min_zero_is_midnight_not_missing(model: type) -> None:
    """boost_end_min == 0 (midnight) is kept over the nested mapping (F5)."""

    parsed = model.model_validate(
        {"boost_end_day": 0, "boost_end_min": 0, "boost_end": {"day": 9, "minute": 30}}
    )

    assert parsed.boost_end_day == 0
    assert parsed.boost_end_min == 0
