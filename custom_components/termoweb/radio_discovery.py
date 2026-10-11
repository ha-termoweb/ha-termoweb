"""Find radio heaters in the background and add them to the entry by themselves.

Setup never waits for heaters. A new radio entry that does not know its heater
network yet listens for it in the background (listening windows, then one
bounded raw survey per round) and stores the network once heard. An entry that
knows its network scans the usual addresses while it has no heaters, and adds
every heater it hears on its own network that is not set up yet. Found heaters
are stored in the entry data and the entry reloads once per batch: the
inventory stays fixed for the lifetime of a setup. See ``docs/radio_backend.md``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
import functools
import logging
import time
from typing import Any

from homeassistant.components import persistent_notification
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.event import async_call_later

from .backend import backend_capabilities
from .backend.radio.discovery import (
    PROBE_GAP_S,
    SCAN_ADDRESSES,
    SURVEY_S,
    PairingAnnounced,
)
from .backend.radio.link import RadioLinkError
from .backend.radio.pairing import PairedHeater
from .backend.radio_client import RadioError
from .backend.sanitize import mask_identifier
from .config_flow_radio import (
    DIALECT_AUTO,
    RadioSetupError,
    UnknownDialectError,
    find_network,
    radio_address,
    radio_link_factory,
)
from .const import (
    CONF_BRAND,
    CONF_DIALECT,
    CONF_NETWORK_ID,
    CONF_NODES,
    CONF_RADIO_TYPE,
    DEFAULT_BRAND,
    DOMAIN,
    signal_radio_unknown_heater,
)
from .radio_autopair import async_pair_new_network, pairing_window_end
from .radio_pairing import async_notify_added, merge_heaters
from .radio_survey import ISSUE_URL, async_analyse, async_save_report, report_payload
from .utils import async_get_integration_version

_LOGGER = logging.getLogger(__name__)

NETWORK_RETRY_S = 600.0  # pause between rounds of listening for the network
FOUND_DEBOUNCE_S = 10.0  # heaters heard together are added with one reload
IDENTIFY_RETRY_S = 600.0  # ask an address that did not answer again after this
IDENTITY_TIMEOUT_S = 10.0  # one identity read
SCAN_GAP_S = PROBE_GAP_S


def network_pending(entry: ConfigEntry) -> bool:
    """Return True for a radio entry that has not learned its heater network yet."""

    brand = entry.data.get(CONF_BRAND, DEFAULT_BRAND)
    return (
        backend_capabilities(brand).heater_discovery
        and CONF_NETWORK_ID not in entry.data
    )


@callback
def async_start_network_search(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Listen for the entry's heater network in the background (stops on unload)."""

    _LOGGER.info("Listening for the heaters' radio network in the background")
    entry.async_create_background_task(
        hass,
        async_search_network(hass, entry),
        f"{DOMAIN}-network-search-{entry.entry_id}",
    )


async def async_search_network(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Listen until the heater network is heard; store it and reload the entry.

    A heater announcing pairing while the pairing window is open is paired
    into the site network instead (radio_autopair.async_pair_new_network).
    """

    data = dict(entry.data)
    host, port = radio_address(data)
    while True:
        try:
            sighting = await find_network(
                host,
                port,
                data.get(CONF_DIALECT, DIALECT_AUTO),
                None,
                link_factory=radio_link_factory(data),
                analyse_survey=functools.partial(async_analyse, hass),
                pairing_open=lambda: pairing_window_end(hass, entry) is not None,
            )
        except PairingAnnounced as heard:
            if await async_pair_new_network(hass, entry, heard.dialect):
                return
            continue  # listen again at once: the heater may still be announcing
        except UnknownDialectError as err:
            await _async_report_unknown_dialect(hass, entry, err)
            return
        except (RadioSetupError, RadioLinkError) as err:
            _LOGGER.info(
                "No heater network heard yet (%s); listening again in %d minutes",
                err,
                NETWORK_RETRY_S // 60,
            )
        except Exception:
            _LOGGER.exception("Listening for the heater network failed")
        else:
            break
        await asyncio.sleep(NETWORK_RETRY_S)
    _LOGGER.info(
        "Heard the heaters' radio network (dialect %s); reloading to find heaters",
        sighting.dialect.name,
    )
    hass.config_entries.async_update_entry(
        entry,
        data={
            **entry.data,
            CONF_DIALECT: sighting.dialect.name,
            CONF_NETWORK_ID: sighting.network_id.hex().upper(),
        },
    )
    hass.config_entries.async_schedule_reload(entry.entry_id)


async def _async_report_unknown_dialect(
    hass: HomeAssistant, entry: ConfigEntry, err: UnknownDialectError
) -> None:
    """Save the survey report and tell the user where it is; listening stops."""

    payload = report_payload(
        err.report,
        seconds=SURVEY_S,
        gateway=None,
        radio_type=entry.data.get(CONF_RADIO_TYPE),
        dialect=entry.data.get(CONF_DIALECT, DIALECT_AUTO),
        version=await async_get_integration_version(hass),
    )
    path = await async_save_report(hass, "setup", payload)
    _LOGGER.error("Radio bursts were heard, but no known dialect decoded them")
    persistent_notification.async_create(
        hass,
        "Radio bursts were heard, but none could be decoded. A weak signal can "
        "cause this: move the radio closer to a heater, then reload the "
        "integration to listen again. If it happens again, your heaters may use "
        "a radio dialect this integration does not know yet. Please open an "
        f"issue at {ISSUE_URL} and attach the report file "
        f"{path or '(the report could not be saved, see the log)'}.",
        title="TermoWeb heaters not recognised",
        notification_id=f"{DOMAIN}_{entry.entry_id}_unknown_dialect",
    )


class HeaterFinder:
    """Add heaters heard on the entry's own network: identify, store, reload once."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        client: Any,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Store collaborators; nothing runs until a heater is heard or scanned."""

        self.hass = hass
        self.entry = entry
        self._client = client
        self._clock = clock
        self._heard: set[int] = set()
        self._paired: dict[int, PairedHeater] = {}  # paired here, awaiting the batch
        self._asked: dict[int, float] = {}  # address -> last identity read
        self._unsub_timer: CALLBACK_TYPE | None = None
        self._done = False  # heaters were stored; the entry reloads

    @callback
    def async_heard(self, addr: int) -> None:
        """Note a frame from an unknown heater; identify it after a short wait."""

        if self._done or addr in self._heard or not 0 < addr < 0xFF:
            return
        asked = self._asked.get(addr)
        if asked is not None and self._clock() - asked < IDENTIFY_RETRY_S:
            return
        _LOGGER.info("Heard heater %s, which is not set up yet", addr)
        self._heard.add(addr)
        self._schedule()

    @callback
    def async_paired(self, heater: PairedHeater) -> None:
        """Add a heater this station just paired with the next batch of found heaters."""

        if self._done:
            return
        if heater.serial is None:  # its identity read failed: ask again
            self.async_heard(heater.node_id)
            return
        self._paired[heater.node_id] = heater
        self._schedule()

    @callback
    def _schedule(self) -> None:
        """Start the debounce timer unless it already runs."""

        if self._unsub_timer is None:
            self._unsub_timer = async_call_later(
                self.hass, FOUND_DEBOUNCE_S, self._async_identify_heard
            )

    @callback
    def _async_identify_heard(self, _now: Any) -> None:
        """Identify every heater heard during the debounce window."""

        self._unsub_timer = None
        paired, self._paired = list(self._paired.values()), {}
        addrs = sorted(self._heard - {heater.node_id for heater in paired})
        self._heard = set()
        self.entry.async_create_background_task(
            self.hass,
            self.async_add(addrs, paired=paired),
            f"{DOMAIN}-add-heaters-{self.entry.entry_id}",
        )

    async def async_scan(self) -> None:
        """Ask every usual heater address for its identity (an entry without heaters)."""

        _LOGGER.info("No heaters set up yet: scanning for heaters")
        await self.async_add(SCAN_ADDRESSES, gap_s=SCAN_GAP_S)

    async def async_add(
        self,
        addrs: Iterable[int],
        *,
        gap_s: float = 0.0,
        paired: Iterable[PairedHeater] = (),
    ) -> None:
        """Store ``paired`` and the heaters that answer an identity read; reload."""

        found: list[PairedHeater] = list(paired)
        for addr in addrs:
            if self._done:
                return
            heater = await self._async_identify(addr)
            if heater is not None:
                found.append(heater)
            if gap_s:
                await asyncio.sleep(gap_s)
        self._store(found)

    async def _async_identify(self, addr: int) -> PairedHeater | None:
        """Return the heater at ``addr`` with its serial, or None if it did not answer."""

        self._asked[addr] = self._clock()
        try:
            async with asyncio.timeout(IDENTITY_TIMEOUT_S):
                record = await self._client.async_read_identity(addr)
        except (TimeoutError, RadioError, RadioLinkError) as err:
            _LOGGER.debug("Heater %s did not report its identity: %s", addr, err)
            return None
        return PairedHeater(addr, None, identity=record.tail, serial=record.serial)

    def _store(self, found: list[PairedHeater]) -> None:
        """Merge found heaters into the entry's nodes; reload and notify if any changed."""

        if self._done or not found:
            return
        nodes, added, moved = merge_heaters(self.entry.data.get(CONF_NODES, []), found)
        if not added and not moved:
            return
        self._done = True
        self.hass.config_entries.async_update_entry(
            self.entry, data={**self.entry.data, CONF_NODES: nodes}
        )
        _LOGGER.info(
            "Found %d new heater(s) (%s) and %d moved; reloading the entry",
            len(added),
            ", ".join(mask_identifier(node.get("serial")) for node in added),
            moved,
        )
        self.hass.config_entries.async_schedule_reload(self.entry.entry_id)
        async_notify_added(self.hass, self.entry, added)

    @callback
    def async_stop(self) -> None:
        """Stop on unload: drop a pending identification."""

        self._done = True
        if self._unsub_timer is not None:
            self._unsub_timer()
            self._unsub_timer = None


@callback
def async_start_heater_finder(
    hass: HomeAssistant, entry: ConfigEntry, client: Any, has_heaters: bool
) -> HeaterFinder:
    """Add unknown heaters the listener hears; scan once when there are none."""

    finder = HeaterFinder(hass, entry, client)
    entry.async_on_unload(
        async_dispatcher_connect(
            hass, signal_radio_unknown_heater(entry.entry_id), finder.async_heard
        )
    )
    entry.async_on_unload(finder.async_stop)
    if not has_heaters:
        entry.async_create_background_task(
            hass, finder.async_scan(), f"{DOMAIN}-scan-heaters-{entry.entry_id}"
        )
    return finder


__all__ = [
    "FOUND_DEBOUNCE_S",
    "HeaterFinder",
    "async_search_network",
    "async_start_heater_finder",
    "async_start_network_search",
    "network_pending",
]
