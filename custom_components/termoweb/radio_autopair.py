"""Pair radio heaters by themselves while the entry's pairing window is open.

A heater in pairing mode announces itself (``docs/research/protocol.md``, section
7). The window is open for 30 minutes after a radio entry is created, and for
10 minutes after the "Pair new heaters" button is pressed. Inside the window
the integration answers the announcement; outside it, announcements are
ignored, so a neighbour's heater in pairing mode is never taken. The window
end lives in ``hass.data``, so it survives the reload that adds a heater.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
import logging
import time
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import (
    async_dispatcher_connect,
    async_dispatcher_send,
)
from homeassistant.util import dt as dt_util
from homeassistant.util.hass_dict import HassKey

from .backend.radio import protocol
from .backend.radio.link import RadioLinkError
from .backend.radio.pairing import Announcement, PairedHeater, PairingError
from .backend.radio_client import RadioError
from .backend.sanitize import mask_identifier
from .const import (
    CONF_DIALECT,
    CONF_NETWORK_ID,
    CONF_NODES,
    DOMAIN,
    signal_radio_pairing_announcement,
    signal_radio_pairing_window,
)
from .radio_pairing import (
    heater_snapshot,
    saved_snapshot,
    store_snapshot,
    with_manual_target,
)

if TYPE_CHECKING:
    from .radio_discovery import HeaterFinder
    from .runtime import EntryRuntime

_LOGGER = logging.getLogger(__name__)

FIRST_WINDOW = timedelta(minutes=30)  # after the entry is created
BUTTON_WINDOW = timedelta(minutes=10)  # after "Pair new heaters" is pressed
PAIR_ATTEMPT_S = 20.0  # one exchange: a sweep reaches this station within seconds
RETRY_PAUSE_S = 30.0  # after an attempt that paired nothing
PAIRING_WINDOWS: HassKey[dict[str, datetime]] = HassKey(f"{DOMAIN}_pairing_windows")


def pairing_supported(runtime: EntryRuntime) -> bool:
    """Return True for a running radio entry that finds heaters and may transmit."""

    return runtime.backend.capabilities.heater_discovery and not getattr(
        runtime.client, "listen_only", False
    )


def pairing_window_end(hass: HomeAssistant, entry: ConfigEntry) -> datetime | None:
    """Return when the entry's pairing window closes, or None while it is closed."""

    end = entry.created_at + FIRST_WINDOW
    opened = hass.data.get(PAIRING_WINDOWS, {}).get(entry.entry_id)
    if opened is not None:
        end = max(end, opened)
    return end if dt_util.utcnow() < end else None


@callback
def async_open_pairing_window(
    hass: HomeAssistant, entry: ConfigEntry, duration: timedelta = BUTTON_WINDOW
) -> datetime:
    """Open (or extend) the entry's pairing window for ``duration``; return its end."""

    end = dt_util.utcnow() + duration
    hass.data.setdefault(PAIRING_WINDOWS, {})[entry.entry_id] = end
    _LOGGER.info("Pairing window open until %s", end.isoformat())
    async_dispatcher_send(hass, signal_radio_pairing_window(entry.entry_id))
    return pairing_window_end(hass, entry) or end


@callback
def async_forget_pairing_window(hass: HomeAssistant, entry_id: str) -> None:
    """Drop a removed entry's pairing window."""

    hass.data.get(PAIRING_WINDOWS, {}).pop(entry_id, None)


def _stored_addr(entry: ConfigEntry, serial: str | None) -> int | None:
    """Return the stored radio id of the heater with ``serial``, if it is known."""

    if serial is None:
        return None
    for node in entry.data.get(CONF_NODES, []):
        if node.get("serial") == serial:
            try:
                return int(node["addr"])
            except KeyError, TypeError, ValueError:
                return None
    return None


class HeaterPairer:
    """Answer pairing announcements while the window is open; one heater at a time."""

    def __init__(
        self,
        hass: HomeAssistant,
        runtime: EntryRuntime,
        finder: HeaterFinder,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Store collaborators; nothing transmits until a heater announces itself."""

        self.hass = hass
        self._runtime = runtime
        self.entry: ConfigEntry = runtime.config_entry
        self._client: Any = runtime.client
        self._finder = finder
        self._clock = clock
        self._busy = False
        self._stopped = False
        self._retry_at: float | None = None
        self._assigned: set[int] = set()  # ids handed out since setup

    @callback
    def async_announced(self, announcement: Announcement) -> None:
        """Pair the announcing heater when the window is open; else ignore it."""

        if self._stopped or self._busy or self._runtime.radio_pairing_lock.locked():
            return
        if pairing_window_end(self.hass, self.entry) is None:
            _LOGGER.debug(
                "Ignoring a pairing announcement: the pairing window is closed"
            )
            return
        if self._retry_at is not None and self._clock() < self._retry_at:
            return
        self._busy = True
        self.entry.async_create_background_task(
            self.hass,
            self._async_pair(announcement),
            f"{DOMAIN}-pair-heater-{self.entry.entry_id}",
        )

    async def _async_pair(self, announcement: Announcement) -> None:
        """Pair one heater, give its settings back and hand it to the finder."""

        try:
            heater = await self._async_pair_one(announcement)
        finally:
            self._busy = False
        if heater is None:
            self._retry_at = self._clock() + RETRY_PAUSE_S
            return
        self._assigned.add(heater.node_id)
        if CONF_NETWORK_ID not in self.entry.data:
            # A learning entry runs on the site network: it is now the heaters'.
            self.hass.config_entries.async_update_entry(
                self.entry,
                data={
                    **self.entry.data,
                    CONF_DIALECT: self._client.dialect.name,
                    CONF_NETWORK_ID: self._client.link.network_id.hex().upper(),
                },
            )
        await self._async_restore(heater)
        self._finder.async_paired(heater)

    async def _async_pair_one(self, announcement: Announcement) -> PairedHeater | None:
        """Run one short pairing exchange; return the paired heater or None."""

        known = None
        if announcement.identity is not None:  # dialect A: the serial is known now
            known = _stored_addr(
                self.entry, protocol.serial_token(announcement.identity)
            )
        _LOGGER.info("A heater is in pairing mode; pairing it")
        try:
            async with self._runtime.radio_pairing_lock:
                paired = await self._client.async_pair(
                    PAIR_ATTEMPT_S,
                    wanted_id=known,
                    max_heaters=1,
                    reserved_ids=self._assigned,
                )
        except (PairingError, RadioError, RadioLinkError) as err:
            _LOGGER.error("Pairing the heater failed: %s", err)
            return None
        if not paired:
            _LOGGER.info("The heater in pairing mode did not take an address")
            return None
        return paired[0]

    async def _async_restore(self, heater: PairedHeater) -> None:
        """Send the clock; a known heater also gets its saved settings back."""

        entry, client = self.entry, self._client
        old = _stored_addr(entry, heater.serial)
        snapshot = None
        if old is not None:
            snapshot = saved_snapshot(entry, old) or with_manual_target(
                heater_snapshot(self._runtime, old), client, old
            )
        _LOGGER.info(
            "Paired heater %s as radio id %d",
            mask_identifier(heater.serial),
            heater.node_id,
        )
        try:
            await client.async_restore(heater.node_id, **(snapshot or {}))
        except (RadioError, RadioLinkError, ValueError) as err:
            _LOGGER.error("Restoring heater %s failed: %s", heater.node_id, err)
            if snapshot is not None and old != heater.node_id:
                store_snapshot(self.hass, entry, heater.node_id, snapshot)
                store_snapshot(self.hass, entry, old, None)
            return
        if old is not None:
            store_snapshot(self.hass, entry, old, None)

    @callback
    def async_stop(self) -> None:
        """Stop on unload: later announcements are ignored."""

        self._stopped = True


@callback
def async_start_auto_pairing(
    hass: HomeAssistant, runtime: EntryRuntime, finder: HeaterFinder
) -> HeaterPairer:
    """Answer the listener's pairing announcements while the window is open."""

    entry = runtime.config_entry
    pairer = HeaterPairer(hass, runtime, finder)
    entry.async_on_unload(
        async_dispatcher_connect(
            hass,
            signal_radio_pairing_announcement(entry.entry_id),
            pairer.async_announced,
        )
    )
    entry.async_on_unload(pairer.async_stop)
    return pairer


__all__ = [
    "BUTTON_WINDOW",
    "FIRST_WINDOW",
    "PAIRING_WINDOWS",
    "HeaterPairer",
    "async_forget_pairing_window",
    "async_open_pairing_window",
    "async_start_auto_pairing",
    "pairing_supported",
    "pairing_window_end",
]
