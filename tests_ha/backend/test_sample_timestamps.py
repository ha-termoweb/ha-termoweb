"""Millisecond sample timestamps and per-record/per-node fault isolation."""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.termoweb.backend import base as base_module
from custom_components.termoweb.backend.base import (
    fetch_normalised_hourly_samples,
    normalise_sample_records,
)
from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient
from custom_components.termoweb.codecs.termoweb_codec import decode_samples

_SECONDS = 1_700_000_000
_MILLIS = _SECONDS * 1000


@pytest.mark.asyncio
@pytest.mark.parametrize("node_type", ["htr", "acm", "pmo"])
async def test_ducaheat_samples_detect_milliseconds_for_every_node_type(
    node_type: str,
) -> None:
    """Thermal nodes report ms timestamps (docs "Samples"); all types are normalised."""

    client = DucaheatRESTClient(SimpleNamespace(), "user", "pass")

    async def fake_headers() -> dict[str, str]:
        return {"Authorization": "Bearer token"}

    async def fake_request(method: str, path: str, **_: Any) -> dict[str, Any]:
        return {"samples": [{"t": _MILLIS, "counter": "10"}]}

    client.authed_headers = fake_headers  # type: ignore[method-assign]
    client._request = fake_request  # type: ignore[method-assign]

    samples = await client.get_node_samples("dev", (node_type, "1"), 0, 1)

    assert samples == [{"t": _SECONDS, "counter": "10"}]


def test_decode_samples_normalises_mixed_seconds_and_milliseconds() -> None:
    """Each record is scaled on its own; seconds stay seconds."""

    decoded = decode_samples(
        [{"t": _SECONDS, "counter": 1}, {"t": _MILLIS + 3_600_000, "counter": 2}]
    )

    assert [item["t"] for item in decoded] == [_SECONDS, _SECONDS + 3600]


def test_normalise_sample_records_skips_out_of_range_timestamp(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One out-of-range timestamp is skipped with a DEBUG log, not raised."""

    with caplog.at_level(logging.DEBUG, logger=base_module.__name__):
        samples = normalise_sample_records(
            "htr",
            [{"t": 1e20, "counter": "5"}, {"t": _SECONDS, "counter": "7"}],
        )

    assert samples == [
        {"ts": datetime.fromtimestamp(_SECONDS, tz=UTC), "energy_wh": 7.0}
    ]
    assert any(
        rec.levelno == logging.DEBUG and "timestamp" in rec.getMessage()
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_fetch_normalised_hourly_samples_isolates_bad_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normalisation failure for one node does not abort the batch."""

    class Client:
        async def get_node_samples(self, dev_id, node, start, end):
            return [{"t": _SECONDS, "counter": "1000"}]

    real = normalise_sample_records

    def flaky(node_type: str, records: Any) -> list[dict[str, Any]]:
        if node_type == "acm":
            raise ValueError("boom")
        return real(node_type, records)

    monkeypatch.setattr(base_module, "normalise_sample_records", flaky)

    result = await fetch_normalised_hourly_samples(
        client=Client(),
        dev_id="dev",
        nodes=[("acm", "1"), ("htr", "2")],
        start_local=datetime(2023, 11, 14, tzinfo=UTC),
        end_local=datetime(2023, 11, 15, tzinfo=UTC),
    )

    assert list(result) == [("htr", "2")]
