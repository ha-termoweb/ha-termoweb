"""The best-effort gateway location lookup of the TermoWeb REST client."""

from __future__ import annotations

from aiohttp import ClientError
import pytest

from custom_components.termoweb.backend.rest_client import RESTClient
from custom_components.termoweb.domain.state import GeoData
from tests_ha.fakes.rest import FakeSession, MockResponse

JSON = {"Content-Type": "application/json"}


def _client(*responses: object) -> tuple[RESTClient, FakeSession]:
    """Return a client whose session answers the token POST and ``responses``."""
    session = FakeSession()
    session.queue_post(
        MockResponse(200, {"access_token": "tok", "expires_in": 3600}, headers=JSON)
    )
    session.queue_request(*responses)
    return RESTClient(session, "user", "pass"), session


async def test_geo_data_is_read_from_the_gateway_endpoint() -> None:
    """A JSON body becomes GeoData; the request names the gateway."""
    body = {
        "country": "Testland",
        "state": "North",
        "city": "Sampleton",
        "tz_code": "Europe/X",
        "zip": "12345",
    }
    client, session = _client(MockResponse(200, body, headers=JSON))

    result = await client.get_geo_data("0123456789abcdef")

    assert result == GeoData(**body)
    [(method, url, _kwargs)] = session.request_calls
    assert method == "GET"
    assert url.endswith("/api/v2/devs/0123456789abcdef/geo_data")


@pytest.mark.parametrize(
    "response",
    [
        MockResponse(404, {}, headers=JSON, text_data="not found"),
        MockResponse(200, ["unexpected"], headers=JSON),
        ClientError("network down"),
    ],
    ids=["404", "not-a-mapping", "network-error"],
)
async def test_geo_data_is_none_when_unavailable(response: object) -> None:
    """A missing endpoint, odd body or network error gives None, never an error."""
    client, _session = _client(response)

    assert await client.get_geo_data("0123456789abcdef") is None
