"""Tests for heater sample subscription frame generation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from custom_components.termoweb.backend import termoweb_ws as module


def _make_client() -> module.TermoWebWSClient:
    """Return an uninitialised ``TermoWebWSClient`` for targeted testing."""

    client = object.__new__(module.TermoWebWSClient)
    client._namespace = module.WS_NAMESPACE  # type: ignore[attr-defined]
    return client


@pytest.mark.asyncio
async def test_subscribe_htr_samples_filters_blank_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure only valid targets yield subscribe frames."""

    client = _make_client()
    valid_pairs = [("htr", "001"), ("acm", "002")]
    monkeypatch.setattr(
        client,
        "_heater_sample_subscription_targets",
        lambda: [
            valid_pairs[0],
            ("", ""),
            valid_pairs[1],
            ("htr", ""),
            (None, None),
            ("aux", None),
        ],
    )
    sent: list[str] = []

    async def record_send(payload: str) -> None:
        sent.append(payload)

    monkeypatch.setattr(client, "_send_text", record_send)
    await client._subscribe_htr_samples()

    expected_payloads = [
        {
            "name": "subscribe",
            "args": [f"/{node_type}/{addr}/samples"],
        }
        for node_type, addr in valid_pairs
    ]
    expected = [
        f"5::{client._namespace}:{json.dumps(payload, separators=(',', ':'))}"
        for payload in expected_payloads
    ]

    assert sent == expected


@pytest.mark.asyncio
async def test_subscribe_htr_samples_skips_empty_target_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm no frames are sent when there are no subscription targets."""

    client = _make_client()
    monkeypatch.setattr(
        client, "_heater_sample_subscription_targets", lambda: []
    )
    send_text = AsyncMock()
    monkeypatch.setattr(client, "_send_text", send_text)

    await client._subscribe_htr_samples()

    send_text.assert_not_awaited()
