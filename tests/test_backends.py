import types
from typing import Any

import pytest

from conftest import _install_stubs

_install_stubs()

import custom_components.termoweb.backends as backends  # noqa: E402
from custom_components.termoweb.const import (  # noqa: E402
    BRAND_DUCAHEAT,
    BRAND_TERMOWEB,
)
def test_backend_factory_returns_expected_clients() -> None:
    hass = types.SimpleNamespace(
        loop=types.SimpleNamespace(create_task=lambda coro, name=None: coro),
        data={},
    )
    coordinator = types.SimpleNamespace()
    api_client = types.SimpleNamespace(_session=types.SimpleNamespace())

    class LegacyStub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    class EngineStub(LegacyStub):
        pass

    termoweb_backend_cls = backends.get_backend_for_brand(BRAND_TERMOWEB)
    termoweb_backend = termoweb_backend_cls(
        hass,
        entry_id="entry",
        api_client=api_client,
        coordinator=coordinator,
        ws_client_factory=LegacyStub,
    )
    legacy_client = termoweb_backend.create_ws_client("dev")
    assert isinstance(legacy_client, LegacyStub)

    duca_backend_cls = backends.get_backend_for_brand(BRAND_DUCAHEAT)
    duca_backend = duca_backend_cls(
        hass,
        entry_id="entry",
        api_client=api_client,
        coordinator=coordinator,
        ws_client_factory=EngineStub,
    )
    v2_client = duca_backend.create_ws_client("dev")
    assert isinstance(v2_client, EngineStub)

    default_backend_cls = backends.get_backend_for_brand("unknown")
    assert default_backend_cls is backends.TermowebBackend


def test_base_backend_requires_create_override() -> None:
    hass = types.SimpleNamespace(loop=None, data={})
    backend = backends.BaseBackend(
        hass,
        entry_id="entry",
        api_client=object(),
        coordinator=object(),
        ws_client_factory=lambda *args, **kwargs: None,
    )

    with pytest.raises(NotImplementedError):
        backend.create_ws_client("dev")
