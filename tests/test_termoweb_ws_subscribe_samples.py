"""Tests for heater sample subscription frame generation."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from custom_components.termoweb.backend import termoweb_ws as module


def _make_client() -> module.TermoWebWSClient:
    """Return an uninitialised ``TermoWebWSClient`` for targeted testing."""

    client = object.__new__(module.TermoWebWSClient)
    client._namespace = module.WS_NAMESPACE  # type: ignore[attr-defined]
    return client


@pytest.mark.asyncio
async def test_subscribe_htr_samples_skips_empty_target_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm no frames are sent when there are no subscription targets."""

    client = _make_client()
    monkeypatch.setattr(client, "_heater_sample_subscription_targets", lambda: [])
    send_text = AsyncMock()
    monkeypatch.setattr(client, "_send_text", send_text)

    await client._subscribe_htr_samples()

    send_text.assert_not_awaited()
