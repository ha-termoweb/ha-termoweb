"""Wire contract of the Ducaheat REST client through a scripted aiohttp session.

Each test drives the public client API and asserts on the HTTP request that
reaches the session (method, path, JSON body) or on the error surfaced to the
caller. Payload shapes follow docs/ducaheat_api.md.
"""

from __future__ import annotations

from typing import Any

from aiohttp import ClientResponseError
import pytest

from custom_components.termoweb.backend.base import BoostContext
from custom_components.termoweb.backend.ducaheat import (
    DucaheatBackend,
    DucaheatRequestError,
    DucaheatRESTClient,
)
from custom_components.termoweb.backend.rest_client import DUCAHEAT_API_BASE
from tests.fakes.rest import FakeSession, MockResponse

DEV = "0123456789abcdef"
BASE = f"{DUCAHEAT_API_BASE}/api/v2/devs/{DEV}"


def _client() -> tuple[DucaheatRESTClient, FakeSession]:
    """Return a Ducaheat client whose token request succeeds."""
    session = FakeSession()
    session.queue_post(MockResponse(200, {"access_token": "tok", "expires_in": 3600}))
    client = DucaheatRESTClient(
        session, "user@example.com", "secret", api_base=DUCAHEAT_API_BASE
    )
    return client, session


def _sent(session: FakeSession) -> list[tuple[str, str, Any]]:
    """Return ``(method, url, json)`` for every request the client sent."""
    return [(m, url, kw.get("json")) for m, url, kw in session.request_calls]


async def test_priority_write_posts_to_setup_segment() -> None:
    """The heater priority number is written via the /setup segment."""
    client, session = _client()
    session.queue_request(MockResponse(201, {}))

    await client.set_node_priority(DEV, ("htr", "2"), priority=7)

    assert _sent(session) == [("POST", f"{BASE}/htr/2/setup", {"priority": 7})]


async def test_boost_stop_posts_boost_false_only() -> None:
    """Stopping an accumulator boost sends the documented ``{"boost": false}``."""
    client, session = _client()
    session.queue_request(MockResponse(201, {}))

    await client.set_acm_boost_state(DEV, "3", boost=False)

    assert _sent(session) == [("POST", f"{BASE}/acm/3/boost", {"boost": False})]


async def test_acm_client_error_with_empty_body_has_empty_error_body() -> None:
    """A 4xx with no body must not leak request metadata into the error body."""
    client, session = _client()
    session.queue_request(MockResponse(400, None, text_data=""))

    with pytest.raises(DucaheatRequestError) as err:
        await client.set_acm_boost_state(DEV, "3", boost=False)

    assert err.value.status == 400
    assert err.value.body == ""


async def test_select_client_error_with_empty_body_has_empty_error_body() -> None:
    """The /select identify helper reports an empty 4xx body as empty."""
    client, session = _client()
    session.queue_request(MockResponse(404, None, text_data=""))

    with pytest.raises(DucaheatRequestError) as err:
        await client.set_node_display_select(DEV, ("htr", "2"), select=True)

    assert err.value.status == 404
    assert err.value.body == ""


async def test_thermostat_preset_write_patches_settings() -> None:
    """Thermostat presets go to /thm/{addr}/settings as cold/night/day strings."""
    client, session = _client()
    session.queue_request(MockResponse(200, {}))

    await client.set_node_settings(DEV, ("thm", "4"), ptemp=[7, 16.5, 21])

    assert _sent(session) == [
        (
            "PATCH",
            f"{BASE}/thm/4/settings",
            {"ptemp": {"cold": "7.0", "night": "16.5", "day": "21.0"}},
        )
    ]


async def test_thermostat_write_without_fields_sends_nothing() -> None:
    """A thermostat write carrying no writable field makes no HTTP request."""
    client, session = _client()

    assert await client.set_node_settings(DEV, ("thm", "4"), units="C") == {}
    assert session.request_calls == []


async def test_thermostat_invalid_setpoint_is_rejected_before_sending() -> None:
    """An unparseable thermostat setpoint raises and nothing is sent."""
    client, session = _client()

    with pytest.raises(ValueError, match="Invalid stemp"):
        await client.set_node_settings(DEV, ("thm", "4"), stemp="warm")  # type: ignore[arg-type]

    assert session.request_calls == []


async def test_thermostat_server_error_is_not_retried_as_post() -> None:
    """Only 404/405 fall back to POST; other errors propagate unchanged."""
    client, session = _client()
    session.queue_request(MockResponse(500, None, text_data="boom"))

    with pytest.raises(ClientResponseError) as err:
        await client.set_node_settings(DEV, ("thm", "4"), stemp=20)

    assert err.value.status == 500
    assert [call[0] for call in _sent(session)] == ["PATCH"]


async def test_acm_write_with_unknown_boost_state_does_not_cancel_boost() -> None:
    """Without boost hints an accumulator setpoint write does not stop boost."""
    client, session = _client()
    session.queue_request(MockResponse(201, {}))
    backend = DucaheatBackend(brand="ducaheat", client=client)

    await backend.set_node_settings(
        DEV, ("acm", "3"), stemp=19, boost_context=BoostContext()
    )

    assert [url for _m, url, _j in _sent(session)] == [f"{BASE}/acm/3/status"]


def test_ws_settings_for_unsupported_node_type_pass_through() -> None:
    """Settings for a node type the domain does not model are left as received."""
    client, _session = _client()
    raw = {"status": {"mode": "auto"}}

    nodes = client.normalise_ws_nodes({"xyz": {"settings": {"1": raw}}})

    assert nodes == {"xyz": {"settings": {"1": raw}}}


def test_ws_status_for_accumulator_without_settings_is_ignored() -> None:
    """Charge metadata only merges into accumulators that carry settings."""
    client, _session = _client()

    nodes = client.normalise_ws_nodes(
        {
            "acm": {
                "settings": {"1": {"status": {"mode": "auto"}}},
                "status": {
                    "1": {"charging": True, "current_charge_per": 40},
                    "2": {"charging": True, "current_charge_per": 80},
                },
            }
        }
    )

    settings = nodes["acm"]["settings"]
    assert set(settings) == {"1"}
    assert settings["1"]["charging"] is True
    assert settings["1"]["current_charge_per"] == 40
