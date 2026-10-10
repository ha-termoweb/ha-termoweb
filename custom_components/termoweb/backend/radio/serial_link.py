"""Serial transport for USB radio sticks (nanoCUL with the termoweb_rx firmware).

The nanoCUL speaks the same line protocol as the ESP32 gateway, over a USB
serial port at 115200 8N1 instead of TCP. ``serial_opener`` returns an
``open_connection`` callable for :class:`RadioLink`, so the link works
unchanged. Any pyserial URL works: a device path such as
``/dev/serial/by-id/usb-...``, or ``socket://host:port`` / ``rfc2217://``
for a stick served over the network. pyserial-asyncio-fast opens the port in
an executor and ``socket://`` with the loop's own connect, so neither blocks
the event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import hashlib

BAUDRATE = 115200

StreamPair = tuple[asyncio.StreamReader, asyncio.StreamWriter]
SerialOpen = Callable[..., Awaitable[StreamPair]]


def _default_open() -> SerialOpen:
    """Return pyserial-asyncio-fast's stream opener (imported on first use)."""
    import serial_asyncio_fast  # noqa: PLC0415 - optional, only for USB sticks

    return serial_asyncio_fast.open_serial_connection


def serial_opener(url: str) -> Callable[[str, int], Awaitable[StreamPair]]:
    """Return an ``open_connection(host, port)`` that opens serial ``url`` instead."""

    async def _open(_host: str, _port: int) -> StreamPair:
        """Open the serial port; RadioLink's host and port are only labels."""
        # The first import of pyserial reads many files: keep it off the loop.
        opener = await asyncio.get_running_loop().run_in_executor(None, _default_open)
        return await opener(url=url, baudrate=BAUDRATE)

    return _open


def serial_device_id(device: str, usb_serial: str | None = None) -> str:
    """Return a stable device id: the USB serial number, else a hash of the path."""
    if usb_serial:
        cleaned = "".join(ch for ch in usb_serial.lower() if ch.isalnum())
        if cleaned:
            return f"nanocul-{cleaned}"
    digest = hashlib.sha256(device.encode()).hexdigest()[:12]
    return f"nanocul-{digest}"


__all__ = ["BAUDRATE", "serial_device_id", "serial_opener"]
