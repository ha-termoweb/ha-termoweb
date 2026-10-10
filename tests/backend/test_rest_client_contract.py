"""Wire contract of the TermoWeb REST client through a scripted aiohttp session.

Each test drives the public client API and asserts on the HTTP request that
reaches the session or on the decoded result handed to the integration.
Payload shapes follow docs/termoweb_api.md.
"""

from __future__ import annotations

from typing import Any

import pytest

from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.codecs.termoweb_codec import (
    build_settings_payload,
    decode_samples,
)
from custom_components.termoweb.const import API_BASE
from custom_components.termoweb.domain.commands import SetLock
from tests.fakes.rest import FakeSession, MockResponse

DEV = "0123456789abcdef"
BASE = f"{API_BASE}/api/v2/devs/{DEV}"

# docs/termoweb_api.md, "GET /api/v2/devs/{dev_id}/htr/{addr}/settings".
DOCUMENTED_HTR_SETTINGS = {
    "name": "Guest bedroom ",
    "priority": 0,
    "prog": [0] * 168,
    "units": "C",
    "ptemp": ["10.0", "16.0", "21.0"],
    "mtemp": "25.7",
    "stemp": "10.0",
    "mode": "off",
    "max_power": "974",
    "state": "off",
    "true_radiant_active": False,
    "window_state_active": False,
    "sync_status": "ok",
}


def _client() -> tuple[RESTClient, FakeSession]:
    """Return a TermoWeb client whose token request succeeds."""
    session = FakeSession()
    session.queue_post(MockResponse(200, {"access_token": "tok", "expires_in": 3600}))
    return RESTClient(session, "user@example.com", "secret"), session


def _sent(session: FakeSession) -> list[tuple[str, str, Any]]:
    """Return ``(method, url, json)`` for every request the client sent."""
    return [(m, url, kw.get("json")) for m, url, kw in session.request_calls]


async def test_documented_heater_settings_decode_to_canonical_fields() -> None:
    """The documented heater settings read keeps the fields the domain uses."""
    client, session = _client()
    session.queue_request(
        MockResponse(
            200,
            DOCUMENTED_HTR_SETTINGS,
            headers={"Content-Type": "application/json"},
            text_data="{}",
        )
    )

    settings = await client.get_node_settings(DEV, ("htr", "2"))

    assert _sent(session) == [("GET", f"{BASE}/htr/2/settings", None)]
    assert settings["mode"] == "off"
    assert settings["stemp"] == "10.0"
    assert settings["mtemp"] == "25.7"
    assert settings["ptemp"] == ["10.0", "16.0", "21.0"]
    assert settings["priority"] == 0
    assert settings["max_power"] == "974"


async def test_settings_read_with_empty_body_yields_no_settings() -> None:
    """A 200 with an empty body decodes to no settings instead of failing."""
    client, session = _client()
    session.queue_request(MockResponse(200, None, text_data=""))

    assert await client.get_node_settings(DEV, ("htr", "2")) == {}


async def test_malformed_json_body_is_returned_as_text() -> None:
    """A JSON-typed body that does not parse is returned as raw text."""
    client, session = _client()
    session.queue_request(
        MockResponse(
            200,
            None,
            headers={"Content-Type": "application/json"},
            text_data="OK",
            json_exc=ValueError("not json"),
        )
    )

    assert await client.set_power_limit(DEV, power_limit=3000) == "OK"


async def test_priority_write_posts_to_settings() -> None:
    """Heater priority is written as an integer on the settings endpoint."""
    client, session = _client()
    session.queue_request(MockResponse(201, {}))

    await client.set_node_priority(DEV, ("htr", "2"), priority=7)

    assert _sent(session) == [("POST", f"{BASE}/htr/2/settings", {"priority": 7})]


async def test_boost_stop_posts_boost_false_only() -> None:
    """Stopping an accumulator boost sends ``{"boost": false}`` and nothing else."""
    client, session = _client()
    session.queue_request(MockResponse(201, {}))

    await client.set_acm_boost_state(DEV, "3", boost=False)

    assert _sent(session) == [("POST", f"{BASE}/acm/3/boost", {"boost": False})]


def test_settings_payload_rejects_non_settings_commands() -> None:
    """Commands with their own endpoint cannot be folded into a settings write."""
    with pytest.raises(TypeError, match="SetLock"):
        build_settings_payload([SetLock(True)])


def test_sample_with_counter_mapping_but_no_reading_is_dropped() -> None:
    """A counter mapping without a reading is skipped, not stringified."""
    samples = decode_samples(
        [
            {"t": 1000, "counter": {"min": 1, "max": 2}},
            {"t": 2000, "counter": "5.0"},
        ]
    )

    assert samples == [{"t": 2000, "counter": "5.0"}]
