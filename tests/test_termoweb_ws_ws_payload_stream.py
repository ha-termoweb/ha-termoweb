"""Tests for websocket payload stream decoding and error handling."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable, Iterator, List

import pytest
from aiohttp import WSMsgType

from custom_components.termoweb.backend.termoweb_ws import TermoWebWSClient
from custom_components.termoweb.backend.ws_client import WSStats


class StubWebSocket:
    """Provide an async iterator returning predefined websocket messages."""

    def __init__(
        self,
        messages: Iterable[SimpleNamespace],
        *,
        exception: BaseException | None = None,
    ) -> None:
        self._messages: List[SimpleNamespace] = list(messages)
        self._exception = exception

    def __aiter__(self) -> "StubWebSocket":
        self._iter: Iterator[SimpleNamespace] = iter(self._messages)
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._iter)
        except StopIteration as err:
            raise StopAsyncIteration from err

    def exception(self) -> BaseException | None:
        """Return the stored websocket exception."""

        return self._exception


@pytest.mark.asyncio
async def test_ws_payload_stream_decodes_text_and_binary() -> None:
    """TEXT and decodable BINARY frames should be yielded and counted."""

    client = TermoWebWSClient.__new__(TermoWebWSClient)
    client._stats = WSStats()  # type: ignore[attr-defined]

    messages = [
        SimpleNamespace(type=WSMsgType.TEXT, data="alpha"),
        SimpleNamespace(type=WSMsgType.BINARY, data=b"bravo"),
        SimpleNamespace(type=WSMsgType.BINARY, data=b"\xff\xfe"),
        SimpleNamespace(type=WSMsgType.ERROR, data=None),
        SimpleNamespace(type=WSMsgType.CLOSE, data=None),
    ]
    websocket = StubWebSocket(messages, exception=RuntimeError("boom"))

    payloads: list[str] = []
    with pytest.raises(RuntimeError, match="^ctx error: boom$"):
        async for payload in client._ws_payload_stream(  # type: ignore[attr-defined]
            websocket,
            context="ctx",
        ):
            payloads.append(payload)

    assert payloads == ["alpha", "bravo"]
    assert client._stats.frames_total == 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_ws_payload_stream_raises_on_close() -> None:
    """CLOSE frames should raise a runtime error with the provided context."""

    client = TermoWebWSClient.__new__(TermoWebWSClient)
    client._stats = WSStats()  # type: ignore[attr-defined]

    messages = [
        SimpleNamespace(type=WSMsgType.TEXT, data="ignored"),
        SimpleNamespace(type=WSMsgType.CLOSE, data=None),
    ]
    websocket = StubWebSocket(messages)

    with pytest.raises(RuntimeError, match="^alt closed$"):
        async for _ in client._ws_payload_stream(  # type: ignore[attr-defined]
            websocket,
            context="alt",
        ):
            pass
