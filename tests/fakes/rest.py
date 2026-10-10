"""Minimal ``aiohttp.ClientSession`` double for REST client tests.

Only ``request`` and ``post`` exist, with the real keyword-argument shapes the
REST clients use. Responses are queued per method; each call records its
arguments so tests can assert on the wire request.
"""

from __future__ import annotations

from collections.abc import Callable
import copy
from typing import Any, Self

import aiohttp
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL


class MockResponse:
    """Queued stand-in for ``aiohttp.ClientResponse`` used as a context manager."""

    def __init__(
        self,
        status: int,
        json_data: Any,
        *,
        headers: dict[str, str] | None = None,
        text_data: str | Callable[[], str] | None = "",
        text_exc: Exception | Callable[[], Exception] | None = None,
        json_exc: Exception | Callable[[], Exception] | None = None,
    ) -> None:
        """Store the canned status, headers and body for this response."""
        self.status = status
        self._json = json_data
        self._text = text_data
        self._text_exc = text_exc
        self._json_exc = json_exc
        self.headers = CIMultiDictProxy(CIMultiDict(headers or {}))
        self.request_info: aiohttp.RequestInfo | None = None
        self.history: tuple[Any, ...] = ()
        self.text_calls = 0
        self.json_calls = 0

    async def __aenter__(self) -> Self:
        """Return the response itself, like ``aiohttp``'s request context."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Release nothing; there is no connection."""

    async def text(self, encoding: str | None = None, errors: str = "strict") -> str:
        """Return the canned text body, or raise the configured error."""
        self.text_calls += 1
        if self._text_exc is not None:
            raise self._text_exc() if callable(self._text_exc) else self._text_exc
        value = self._text() if callable(self._text) else self._text
        return "" if value is None else value

    async def json(
        self,
        *,
        encoding: str | None = None,
        loads: Callable[[str], Any] | None = None,
        content_type: str | None = "application/json",
    ) -> Any:
        """Return the canned JSON body, or raise the configured error."""
        self.json_calls += 1
        if self._json_exc is not None:
            raise self._json_exc() if callable(self._json_exc) else self._json_exc
        return self._json() if callable(self._json) else self._json


class LatchedResponse:
    """A queued value that is returned for every call instead of being popped."""

    def __init__(self, value: Any) -> None:
        """Remember the value to return on every call."""
        self._value = value

    def get(self) -> Any:
        """Return the latched value."""
        return self._value


def _request_info(method: str, url: str) -> aiohttp.RequestInfo:
    """Return a real ``aiohttp.RequestInfo`` for ``method`` and ``url``."""
    return aiohttp.RequestInfo(URL(url), method, CIMultiDictProxy(CIMultiDict()))


class FakeSession:
    """Scripted ``aiohttp.ClientSession`` exposing ``request`` and ``post``."""

    closed = False

    def __init__(self) -> None:
        """Start with empty response queues and call logs."""
        self._request_queue: list[Any] = []
        self._post_queue: list[Any] = []
        self.request_calls: list[tuple[str, str, dict[str, Any]]] = []
        self.post_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def queue_request(self, *responses: Any) -> None:
        """Queue responses (or exceptions, or callables) for ``request``."""
        self._request_queue.extend(responses)

    def queue_post(self, *responses: Any) -> None:
        """Queue responses (or exceptions, or callables) for ``post``."""
        self._post_queue.extend(responses)

    def clear_calls(self) -> None:
        """Forget the recorded calls."""
        self.request_calls.clear()
        self.post_calls.clear()

    @staticmethod
    def _resolve(queue: list[Any], label: str) -> Any:
        """Pop (or peek, for a latched value) the next scripted result."""
        if not queue:
            raise AssertionError(f"Unexpected {label} call with no queued response")
        item = queue[0]
        result = item.get() if isinstance(item, LatchedResponse) else queue.pop(0)
        return result() if callable(result) else result

    @staticmethod
    def _finish(result: Any, method: str, url: str) -> Any:
        """Raise a scripted exception or attach request info to a response."""
        if isinstance(result, BaseException):
            raise result
        if isinstance(result, MockResponse) and result.request_info is None:
            result.request_info = _request_info(method, url)
        return result

    def request(self, method: str, str_or_url: str, **kwargs: Any) -> Any:
        """Record a request and return the next queued response."""
        self.request_calls.append((method, str_or_url, copy.deepcopy(kwargs)))
        result = self._resolve(self._request_queue, "request")
        return self._finish(result, method, str_or_url)

    def post(self, url: str, *, data: Any = None, **kwargs: Any) -> Any:
        """Record a POST and return the next queued response."""
        if data is not None:
            kwargs["data"] = data
        self.post_calls.append((url, (), copy.deepcopy(kwargs)))
        result = self._resolve(self._post_queue, "post")
        return self._finish(result, "POST", url)
