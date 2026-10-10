"""Per-client REST rate limiting, 429 handling and token reuse."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import logging

import pytest

import custom_components.termoweb.backend.rest_client as api
from custom_components.termoweb.backend.ducaheat import DucaheatRESTClient
from tests.test_api import FakeSession, MockResponse

JSON = {"Content-Type": "application/json"}


class FakeClock:
    """Monotonic clock whose sleep advances time instantly."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay
        await asyncio.sleep(0)


class StampedSession(FakeSession):
    """FakeSession that records the fake time at which each call starts."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__()
        self.clock = clock
        self.starts: list[tuple[str, float]] = []

    def request(self, method, url, *args, **kwargs):
        self.starts.append((f"{method} {url}", self.clock.now))
        return super().request(method, url, *args, **kwargs)

    def post(self, url, *args, **kwargs):
        self.starts.append((f"POST {url}", self.clock.now))
        return super().post(url, *args, **kwargs)


def _token(value: str) -> MockResponse:
    return MockResponse(200, {"access_token": value, "expires_in": 3600}, headers=JSON)


def _ok(body: object | None = None) -> MockResponse:
    return MockResponse(200, body or {}, headers=JSON, text_data="{}")


def _client(
    monkeypatch: pytest.MonkeyPatch, cls: type[api.RESTClient] = api.RESTClient, **kw
) -> tuple[api.RESTClient, StampedSession, FakeClock]:
    """Build a client whose limiter uses the production spacing and a fake clock."""
    monkeypatch.undo()  # restore the real REST_MIN_INTERVAL_S (conftest zeroes it)
    clock = FakeClock()
    session = StampedSession(clock)
    client = cls(session, "user@example.com", "secret", **kw)
    client._limiter = api.RequestLimiter(clock=clock, sleep=clock.sleep)
    return client, session, clock


def _gaps(starts: list[tuple[str, float]]) -> list[float]:
    times = [t for _, t in starts]
    return [b - a for a, b in zip(times, times[1:], strict=False)]


def test_concurrent_requests_are_spaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """N concurrent calls (token POST included) start >= 0.5 s apart."""
    client, session, _ = _client(monkeypatch)
    session.queue_post(_token("T1"))
    session.queue_request(*[_ok({"y": 2020}) for _ in range(6)])

    async def _run() -> None:
        await asyncio.gather(*(client.get_rtc_time("dev") for _ in range(6)))

    asyncio.run(_run())

    assert len(session.starts) == 7  # 1 token POST + 6 GETs
    assert session.starts[0][0].startswith("POST ")
    assert all(gap >= 0.5 for gap in _gaps(session.starts))
    assert api.REST_MIN_INTERVAL_S == 0.5


def test_429_retry_after_blocks_following_requests(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A 429 with Retry-After holds every later request on the client."""
    client, session, clock = _client(monkeypatch)
    session.queue_post(_token("T1"))
    session.queue_request(
        MockResponse(429, {}, headers={**JSON, "Retry-After": "7"}, text_data="{}"),
        _ok({"y": 2020}),
    )

    async def _run() -> None:
        headers = await client.authed_headers()
        with pytest.raises(api.BackendRateLimitError) as excinfo:
            await client._request("GET", "/a", headers=dict(headers))
        assert excinfo.value.retry_after == 7.0
        limited_at = clock.now
        await client.get_rtc_time("dev")
        assert session.starts[-1][1] >= limited_at + 7.0

    with caplog.at_level(logging.DEBUG, logger=api.__name__):
        asyncio.run(_run())

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_429_without_retry_after_uses_default_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare 429 still backs off by the default pause."""
    client, session, clock = _client(monkeypatch)
    session.queue_post(_token("T1"))
    session.queue_request(
        MockResponse(429, {}, headers=JSON, text_data="{}"), _ok({"y": 1})
    )

    async def _run() -> None:
        headers = await client.authed_headers()
        with pytest.raises(api.BackendRateLimitError) as excinfo:
            await client._request("GET", "/a", headers=dict(headers))
        assert excinfo.value.retry_after == api.RATE_LIMIT_DEFAULT_PAUSE_S
        limited_at = clock.now
        await client.get_rtc_time("dev")
        assert session.starts[-1][1] >= limited_at + api.RATE_LIMIT_DEFAULT_PAUSE_S

    asyncio.run(_run())


def test_token_429_pauses_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 from the token endpoint pauses the client and carries retry_after."""
    client, session, clock = _client(monkeypatch)
    session.queue_post(
        MockResponse(429, {}, headers={**JSON, "Retry-After": "12"}, text_data=""),
        _token("T1"),
    )

    async def _run() -> None:
        with pytest.raises(api.BackendRateLimitError) as excinfo:
            await client._ensure_token()
        assert excinfo.value.retry_after == 12.0
        limited_at = clock.now
        assert await client._ensure_token() == "T1"
        assert session.starts[-1][1] >= limited_at + 12.0

    asyncio.run(_run())


def test_segmented_write_after_401_fetches_one_token(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Four stale-token requests after server-side expiry cost one token POST."""
    client, session, _ = _client(
        monkeypatch, DucaheatRESTClient, api_base=api.DUCAHEAT_API_BASE
    )
    session.queue_post(_token("T1"), _token("T2"), _token("T3"), _token("T4"))

    def _unauth() -> MockResponse:
        return MockResponse(401, {"e": 1}, headers=JSON, text_data='{"e":1}')

    # The server rejects T1 for the prog GET (slot-resolution echo) and for
    # each segment (status setpoint, prog, status presets).
    session.queue_request(
        _unauth(), _ok(), _unauth(), _ok(), _unauth(), _ok(), _unauth(), _ok()
    )

    async def _run() -> None:
        await client.set_node_settings(
            "dev", ("htr", "1"), stemp=21, prog=[0] * 168, ptemp=[5, 17, 21]
        )

    with caplog.at_level(logging.DEBUG, logger=api.__name__):
        asyncio.run(_run())

    assert len(session.post_calls) == 2  # initial token + exactly one refresh
    assert len(session.request_calls) == 8
    retried = [c for c in session.request_calls[1::2]]
    assert all(c[2]["headers"]["Authorization"] == "Bearer T2" for c in retried)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert all(gap >= 0.5 for gap in _gaps(session.starts))


def test_401_after_reauth_logs_one_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A 401 that survives re-auth raises BackendAuthError with one ERROR line."""
    session = FakeSession()
    session.queue_post(_token("T1"), _token("T2"))
    session.queue_request(
        MockResponse(401, {}, headers=JSON, text_data="{}"),
        MockResponse(401, {}, headers=JSON, text_data="{}"),
    )
    client = api.RESTClient(session, "user@example.com", "secret")

    async def _run() -> None:
        headers = await client.authed_headers()
        with pytest.raises(api.BackendAuthError):
            await client._request("GET", "/a", headers=dict(headers))

    with caplog.at_level(logging.DEBUG, logger=api.__name__):
        asyncio.run(_run())

    assert [r.levelno for r in caplog.records if r.levelno >= logging.WARNING] == [
        logging.ERROR
    ]


def test_ignored_status_and_client_error_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ignored statuses log at DEBUG; a non-ignored 4xx logs one ERROR."""
    session = FakeSession()
    session.queue_request(
        MockResponse(404, {}, headers=JSON, text_data="nope"),
        MockResponse(404, {}, headers=JSON, text_data="nope"),
    )
    client = api.RESTClient(session, "user@example.com", "secret")

    async def _run() -> None:
        assert await client._request("GET", "/a", ignore_statuses=(404,)) is None
        with pytest.raises(api.aiohttp.ClientResponseError):
            await client._request("GET", "/a")

    with caplog.at_level(logging.DEBUG, logger=api.__name__):
        asyncio.run(_run())

    levels = [r.levelno for r in caplog.records if r.levelno >= logging.WARNING]
    assert levels == [logging.ERROR]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("  ", None),
        ("soon", None),
        ("nan", None),
        ("5", 5.0),
        (" 2.5 ", 2.5),
        ("-3", 0.0),
        ("999999", api.RATE_LIMIT_MAX_PAUSE_S),
        ("Sat, 10 Oct 2026 12:00:30 GMT", 30.0),
        ("Sat, 10 Oct 2026 12:00:30 -0000", 30.0),
        ("Sat, 10 Oct 2026 11:00:00 GMT", 0.0),
    ],
)
def test_parse_retry_after(value: str | None, expected: float | None) -> None:
    """Retry-After accepts delta seconds or an HTTP date and is clamped."""
    now = datetime(2026, 10, 10, 12, 0, 0, tzinfo=UTC)
    assert api.parse_retry_after(value, now=now) == expected


def test_parse_retry_after_defaults_to_wall_clock() -> None:
    """Without ``now`` an HTTP date in the past yields zero."""
    assert api.parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0


def test_limiter_pause_never_shortens() -> None:
    """A shorter pause does not cut an existing longer one."""
    clock = FakeClock()
    limiter = api.RequestLimiter(0.5, clock=clock, sleep=clock.sleep)
    limiter.pause(10)
    limiter.pause(2)

    asyncio.run(limiter.acquire())

    assert clock.sleeps == [10.0]
