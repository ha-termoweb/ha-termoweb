"""Find radio heaters in the background and add them to the entry by themselves.

Setup never waits for heaters. A new radio entry that does not know its heater
network yet runs on the installation's own network (heaters can be paired into
it) and learns the existing network in the background: from frames the
listener hears, and from a raw survey every 10 minutes. An entry that knows its
network scans the usual addresses while it has no heaters, and adds every
heater it hears on its own network that is not set up yet. Found heaters are
stored in the entry data and the entry reloads once per batch: the inventory
stays fixed for the lifetime of a setup. See ``docs/radio_backend.md``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
import logging
import time
from typing import Any

from homeassistant.components import persistent_notification
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.event import async_call_later

from .backend import backend_capabilities
from .backend.radio.dialect import DIALECT_A, DIALECTS
from .backend.radio.discovery import PROBE_GAP_S, SCAN_ADDRESSES, SURVEY_S
from .backend.radio.link import RadioLinkError
from .backend.radio.pairing import PairedHeater
from .backend.radio.survey import SurveyReport
from .backend.radio_client import RadioError
from .backend.sanitize import mask_identifier
from .config_flow_radio import DIALECT_AUTO, probe_nanocul
from .const import (
    CONF_BRAND,
    CONF_DEVICE,
    CONF_DIALECT,
    CONF_NETWORK_ID,
    CONF_NODES,
    CONF_RADIO_TYPE,
    DEFAULT_BRAND,
    DOMAIN,
    RADIO_TYPE_NANOCUL,
    signal_radio_network_heard,
    signal_radio_unknown_heater,
)
from .radio_pairing import async_notify_added, async_site_network_id, merge_heaters
from .radio_survey import ISSUE_URL, async_analyse, async_save_report, report_payload
from .utils import async_get_integration_version

_LOGGER = logging.getLogger(__name__)

NETWORK_RETRY_S = 600.0  # pause between raw surveys while learning the network
NETWORK_FRAMES = 2  # frames from one network before it is adopted
LEARNING_DIALECT = "B"  # what a learning entry speaks unless a dialect was chosen
IGNORED_NETWORKS = frozenset({b"\x00\x00", b"\xff\xff"})  # pairing sweeps, noise
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


async def async_learning_network(
    hass: HomeAssistant, entry: ConfigEntry
) -> tuple[str, bytes] | None:
    """Return (dialect, site network) for an entry still learning its network.

    Such an entry runs on the installation's own network, so heaters can be
    paired into it while the heaters' existing network is still being
    learned. None when the network is known. A nanoCUL with stock firmware
    only speaks dialect A, whose network is fixed: that is stored at once.
    """

    if not network_pending(entry):
        return None
    dialect = entry.data.get(CONF_DIALECT)
    if dialect is None and entry.data.get(CONF_RADIO_TYPE) == RADIO_TYPE_NANOCUL:
        _dev_id, capable = await probe_nanocul(entry.data[CONF_DEVICE])
        if not capable:
            _LOGGER.info("This nanoCUL firmware speaks dialect A only")
            _store_network(hass, entry, DIALECT_A.name, DIALECT_A.network_id)
            return None
    network_id = await async_site_network_id(hass, gateway_id(entry))
    return dialect or LEARNING_DIALECT, network_id


def gateway_id(entry: ConfigEntry) -> str:
    """Return the gateway id from a radio entry's unique id (``radio:<dev_id>``)."""

    return str(entry.unique_id or "").partition(":")[2]


def _store_network(
    hass: HomeAssistant, entry: ConfigEntry, dialect: str, network_id: bytes
) -> None:
    """Store the heaters' dialect and network id in the entry data."""

    hass.config_entries.async_update_entry(
        entry,
        data={
            **entry.data,
            CONF_DIALECT: dialect,
            CONF_NETWORK_ID: network_id.hex().upper(),
        },
    )


class NetworkLearner:
    """Learn the heaters' network while the entry runs on the site network.

    Heaters already paired to another network (a stock gateway, another Home
    Assistant) are heard by the running listener (``signal_radio_network_heard``)
    or found by a raw survey every 10 minutes. The network is then stored and
    the entry reloads. A heater paired into the site network ends learning too.
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, client: Any) -> None:
        """Store collaborators; nothing runs until frames arrive or a survey is due."""

        self.hass = hass
        self.entry = entry
        self._client = client
        self._heard: dict[bytes, int] = {}
        self._unsub_timer: CALLBACK_TYPE | None = None
        self._done = False

    @callback
    def async_heard(self, network_id: bytes) -> None:
        """Count a frame from another network; adopt it once it repeats."""

        if self._done or network_id in IGNORED_NETWORKS:
            return
        count = self._heard[network_id] = self._heard.get(network_id, 0) + 1
        if count >= NETWORK_FRAMES:
            self._adopt(self._client.dialect.name, network_id)

    @callback
    def async_schedule_survey(self) -> None:
        """Run the next raw survey in 10 minutes."""

        if not self._done:
            self._unsub_timer = async_call_later(
                self.hass, NETWORK_RETRY_S, self._async_survey_due
            )

    @callback
    def _async_survey_due(self, _now: Any) -> None:
        """Start the due survey as an entry background task."""

        self._unsub_timer = None
        self.entry.async_create_background_task(
            self.hass,
            self.async_survey(),
            f"{DOMAIN}-network-survey-{self.entry.entry_id}",
        )

    async def async_survey(self) -> None:
        """Survey the air once; store a network in a known dialect, else go on."""

        if self._done or not network_pending(self.entry):
            return
        try:
            bursts = await self._client.async_survey(SURVEY_S)
        except RadioLinkError as err:
            _LOGGER.info("Radio survey failed (%s); trying again later", err)
            self.async_schedule_survey()
            return
        except RadioError as err:
            _LOGGER.info("No raw survey: %s", err)
            return
        report = await async_analyse(self.hass, bursts)
        _LOGGER.info("Radio survey verdict: %s", report.verdict)
        if report.verdict == "silent":
            self.async_schedule_survey()
            return
        if report.verdict != "known" or report.dialect not in DIALECTS:
            self._done = True
            await _async_report_unknown_dialect(self.hass, self.entry, report)
            return
        dialect = DIALECTS[report.dialect]
        heard = [bytes.fromhex(net) for net in report.network_ids]
        if dialect.network_id is not None:
            heard.append(dialect.network_id)
        network_id = next((net for net in heard if net not in IGNORED_NETWORKS), None)
        if network_id is None:
            self.async_schedule_survey()
            return
        self._adopt(dialect.name, network_id)

    def _adopt(self, dialect: str, network_id: bytes) -> None:
        """Store the heard network and reload the entry (once)."""

        self.async_stop()
        if not network_pending(self.entry):
            return  # a heater was paired into the site network meanwhile
        _LOGGER.info(
            "Heard the heaters' radio network (dialect %s); reloading to find heaters",
            dialect,
        )
        _store_network(self.hass, self.entry, dialect, network_id)
        self.hass.config_entries.async_schedule_reload(self.entry.entry_id)

    @callback
    def async_stop(self) -> None:
        """Stop learning: drop a pending survey."""

        self._done = True
        if self._unsub_timer is not None:
            self._unsub_timer()
            self._unsub_timer = None


@callback
def async_start_network_learning(
    hass: HomeAssistant, entry: ConfigEntry, client: Any
) -> NetworkLearner:
    """Learn the heaters' network in the background (stops on unload)."""

    _LOGGER.info("Listening for the heaters' radio network in the background")
    learner = NetworkLearner(hass, entry, client)
    entry.async_on_unload(
        async_dispatcher_connect(
            hass, signal_radio_network_heard(entry.entry_id), learner.async_heard
        )
    )
    entry.async_on_unload(learner.async_stop)
    learner.async_schedule_survey()
    return learner


async def _async_report_unknown_dialect(
    hass: HomeAssistant, entry: ConfigEntry, report: SurveyReport
) -> None:
    """Save the survey report and tell the user where it is; learning stops."""

    payload = report_payload(
        report,
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
    "NetworkLearner",
    "async_learning_network",
    "async_start_heater_finder",
    "async_start_network_learning",
    "gateway_id",
    "network_pending",
]
