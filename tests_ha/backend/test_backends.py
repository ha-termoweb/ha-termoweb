from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from homeassistant.util import dt as dt_util
import pytest

from custom_components.termoweb.backend import Backend  # noqa: E402
from custom_components.termoweb.backend.ducaheat import DucaheatBackend  # noqa: E402
from custom_components.termoweb.backend.termoweb import TermoWebBackend  # noqa: E402
from custom_components.termoweb.const import BRAND_DUCAHEAT  # noqa: E402


class DummyHttpClient:
    """Minimal HTTP client stub exposing a session attribute."""

    def __init__(self) -> None:
        self._session = SimpleNamespace()


def test_backend_requires_create_override() -> None:
    """Attempting to instantiate a backend without a websocket factory fails."""

    class InvalidBackend(Backend):
        pass

    client = DummyHttpClient()
    with pytest.raises(TypeError):
        InvalidBackend(brand="termoweb", client=client)


@pytest.mark.asyncio
async def test_termoweb_backend_fetch_hourly_samples_normalises() -> None:
    """TermoWeb hourly fetch returns UTC timestamps and energy in Wh."""

    client = SimpleNamespace()
    client.get_node_samples = AsyncMock(
        return_value=[{"t": 1_700_000_000, "counter": 1_800.0, "power": 600.0}]
    )
    backend = TermoWebBackend(brand="termoweb", client=client)
    tz = dt_util.get_time_zone("Europe/Paris")
    start_local = datetime(2023, 3, 27, 9, 0, tzinfo=tz)
    end_local = start_local + timedelta(hours=1)

    result = await backend.fetch_hourly_samples(
        "dev",
        [("htr", "A")],
        start_local,
        end_local,
    )

    call = client.get_node_samples.await_args
    assert call.args[0] == "dev"
    assert call.args[1] == ("htr", "A")
    assert call.args[2] == pytest.approx(
        start_local.astimezone(timezone.utc).timestamp()
    )
    assert call.args[3] == pytest.approx(end_local.astimezone(timezone.utc).timestamp())

    bucket = result[("htr", "A")]
    assert len(bucket) == 1
    sample = bucket[0]
    assert sample["energy_wh"] == pytest.approx(1_800.0)
    assert sample["power_w"] == pytest.approx(600.0)
    assert sample["ts"].tzinfo is timezone.utc


@pytest.mark.asyncio
async def test_ducaheat_backend_fetch_hourly_samples_normalises() -> None:
    """Ducaheat hourly fetch delegates to the shared normalisation helper."""

    client = SimpleNamespace()
    client.get_node_samples = AsyncMock(
        return_value=[{"t": 1_700_000_000, "counter": 7_200_000.0}]
    )
    backend = DucaheatBackend(brand=BRAND_DUCAHEAT, client=client)
    tz = dt_util.get_time_zone("Europe/Paris")
    start_local = datetime(2023, 3, 27, 9, 0, tzinfo=tz)
    end_local = start_local + timedelta(hours=1)

    result = await backend.fetch_hourly_samples(
        "dev",
        [("pmo", "M")],
        start_local,
        end_local,
    )

    call = client.get_node_samples.await_args
    assert call.args[1] == ("pmo", "M")
    bucket = result[("pmo", "M")]
    assert bucket[0]["energy_wh"] == pytest.approx(2_000.0)
    assert bucket[0]["ts"].tzinfo is timezone.utc
