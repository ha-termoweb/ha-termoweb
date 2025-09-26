import types

from conftest import _install_stubs

_install_stubs()

import custom_components.termoweb.backends as backends  # noqa: E402
from custom_components.termoweb.const import (  # noqa: E402
    BRAND_DUCAHEAT,
    BRAND_TERMOWEB,
)
from custom_components.termoweb.ws_client_legacy import (  # noqa: E402
    TermoWebWSLegacyClient,
)
from custom_components.termoweb.ws_client_v2 import TermoWebWSV2Client  # noqa: E402


def test_backend_factory_returns_expected_clients() -> None:
    hass = types.SimpleNamespace(
        loop=types.SimpleNamespace(create_task=lambda coro, name=None: coro),
        data={},
    )
    coordinator = types.SimpleNamespace()
    api_client = types.SimpleNamespace(_session=types.SimpleNamespace())

    termoweb_backend_cls = backends.get_backend_for_brand(BRAND_TERMOWEB)
    termoweb_backend = termoweb_backend_cls(
        hass,
        entry_id="entry",
        api_client=api_client,
        coordinator=coordinator,
        ws_client_factory=TermoWebWSLegacyClient,
    )
    legacy_client = termoweb_backend.create_ws_client("dev")
    assert isinstance(legacy_client, TermoWebWSLegacyClient)

    duca_backend_cls = backends.get_backend_for_brand(BRAND_DUCAHEAT)
    duca_backend = duca_backend_cls(
        hass,
        entry_id="entry",
        api_client=api_client,
        coordinator=coordinator,
        ws_client_factory=TermoWebWSV2Client,
    )
    v2_client = duca_backend.create_ws_client("dev")
    assert isinstance(v2_client, TermoWebWSV2Client)

    default_backend_cls = backends.get_backend_for_brand("unknown")
    assert default_backend_cls is backends.TermowebBackend
