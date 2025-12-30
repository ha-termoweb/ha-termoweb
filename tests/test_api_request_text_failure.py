"""Tests for RESTClient error handling when response text fails."""

from __future__ import annotations

import asyncio
import logging

import aiohttp
import pytest

import custom_components.termoweb.api as api
from tests.test_api import FakeSession, MockResponse


def test_request_text_failure_logs_placeholder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ClientResponseError should use placeholder message when text() fails."""

    caplog.set_level(logging.DEBUG, logger=api.__name__)

    async def _run() -> None:
        session = FakeSession()
        response = MockResponse(
            500,
            {"detail": "boom"},
            headers={"Content-Type": "application/json"},
            text_exc=lambda: RuntimeError("text decode failed"),
        )
        session.queue_request(response)

        client = api.RESTClient(session, "user@example.com", "secret")

        with pytest.raises(aiohttp.ClientResponseError) as excinfo:
            await client._request("GET", "/api/v2/fail")

        err = excinfo.value
        message_attr = getattr(err, "message", None)
        if message_attr is not None:
            assert message_attr == "<no body>"
        else:
            assert "<no body>" in str(err)
        assert response.text_calls == 1

    asyncio.run(_run())

    error_logs = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR and record.name == api.__name__
    ]
    assert error_logs, "Expected error log records for failed request"
    assert any("<no body>" in record.getMessage() for record in error_logs)
