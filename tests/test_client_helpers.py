import asyncio
from typing import Any

import pytest

from conftest import _install_stubs

_install_stubs()

from custom_components.termoweb import client as client_helpers
from homeassistant.core import HomeAssistant


def test_create_rest_client_uses_termoweb_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    hass = HomeAssistant()
    session = object()
    calls: list[tuple[Any, ...]] = []

    class DummyClient:
        def __init__(
            self,
            sess: Any,
            username: str,
            password: str,
            *,
            api_base: str,
            basic_auth_b64: str,
        ) -> None:
            calls.append((sess, username, password, api_base, basic_auth_b64))

    monkeypatch.setattr(
        client_helpers.aiohttp_client,
        "async_get_clientsession",
        lambda hass_arg: session if hass_arg is hass else object(),
    )
    monkeypatch.setattr(client_helpers, "RESTClient", DummyClient)
    monkeypatch.setattr(client_helpers, "DucaheatRESTClient", object())

    client_helpers.create_rest_client(hass, "user@example.com", "pw", "termoweb")

    assert calls == [
        (
            session,
            "user@example.com",
            "pw",
            client_helpers.get_brand_api_base("termoweb"),
            client_helpers.get_brand_basic_auth("termoweb"),
        )
    ]


def test_create_rest_client_uses_ducaheat_client(monkeypatch: pytest.MonkeyPatch) -> None:
    hass = HomeAssistant()
    session = object()
    calls: list[tuple[Any, ...]] = []

    class DummyDucaheat:
        def __init__(
            self,
            sess: Any,
            username: str,
            password: str,
            *,
            api_base: str,
            basic_auth_b64: str,
        ) -> None:
            calls.append((sess, username, password, api_base, basic_auth_b64))

    monkeypatch.setattr(
        client_helpers.aiohttp_client,
        "async_get_clientsession",
        lambda hass_arg: session if hass_arg is hass else object(),
    )
    monkeypatch.setattr(client_helpers, "DucaheatRESTClient", DummyDucaheat)
    monkeypatch.setattr(client_helpers, "RESTClient", object())

    client_helpers.create_rest_client(
        hass, "user@example.com", "pw", client_helpers.BRAND_DUCAHEAT
    )

    assert calls == [
        (
            session,
            "user@example.com",
            "pw",
            client_helpers.get_brand_api_base(client_helpers.BRAND_DUCAHEAT),
            client_helpers.get_brand_basic_auth(client_helpers.BRAND_DUCAHEAT),
        )
    ]


@pytest.mark.parametrize(
    "exc",
    [
        client_helpers.BackendAuthError("bad"),
        TimeoutError("timeout"),
        client_helpers.BackendRateLimitError("slow"),
    ],
)
def test_async_list_devices_with_logging_propagates(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    class DummyClient:
        async def list_devices(self) -> None:
            raise exc

    client = DummyClient()
    monkeypatch.setattr(client_helpers._LOGGER, "info", lambda *args, **kwargs: None)

    async def _run() -> None:
        await client_helpers.async_list_devices_with_logging(client)  # type: ignore[arg-type]

    with pytest.raises(type(exc)):
        asyncio.run(_run())
