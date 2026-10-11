"""Options flow for radio gateway entries: pairing and rehoming."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntry
import voluptuous as vol

from .backend.radio import RadioLinkError
from .backend.radio.pairing import IDLE_STOP_S, PAIR_WINDOW_S, PairingError
from .const import CONF_NODES
from .radio_pairing import add_nodes
from .radio_rehome import RehomeError, async_rehome, rehome_summary
from .runtime import require_runtime

_LOGGER = logging.getLogger(__name__)


class TermoWebOptionsFlow(config_entries.OptionsFlow):
    """Options flow for radio gateway entries: pairing and rehoming."""

    def __init__(self, entry: ConfigEntry) -> None:
        """Store the entry being configured."""
        self.entry = entry
        self._pair_task: asyncio.Task[Any] | None = None
        self._pair_error: str | None = None
        self._paired: list[int] = []
        self._rehome_task: asyncio.Task[Any] | None = None
        self._rehome_summary = ""
        self._rehome_summary_shown = False

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        """Let the user choose pairing or rehoming."""
        return self.async_show_menu(
            step_id="init", menu_options=["pair_heaters", "rehome"]
        )

    async def async_step_pair_heaters(self, user_input: dict[str, Any] | None = None):
        """Explain pairing mode; start pairing into this entry's network on submit."""
        if user_input is None:
            errors = {"base": self._pair_error} if self._pair_error else {}
            return self.async_show_form(
                step_id="pair_heaters", data_schema=vol.Schema({}), errors=errors
            )
        return await self.async_step_pair_run()

    async def async_step_pair_run(self, user_input: dict[str, Any] | None = None):
        """Pair heaters with the running gateway connection while showing progress."""
        if self._pair_task is None:
            try:
                client = require_runtime(self.hass, self.entry.entry_id).client
            except LookupError:
                return self.async_abort(reason="not_loaded")
            self._pair_task = self.hass.async_create_task(
                client.async_pair(PAIR_WINDOW_S, idle_stop_s=IDLE_STOP_S)
            )
        if not self._pair_task.done():
            return self.async_show_progress(
                step_id="pair_run",
                progress_action="pair_heaters",
                progress_task=self._pair_task,
            )
        task, self._pair_task = self._pair_task, None
        self._pair_error = None
        try:
            paired = task.result()
        except PairingError as err:
            paired = err.paired
            if not paired:
                self._pair_error = "no_free_address"
        except RadioLinkError:
            paired, self._pair_error = [], "cannot_connect_radio"
        except Exception:
            _LOGGER.exception("Unexpected error while pairing radio heaters")
            paired, self._pair_error = [], "unknown"
        self._paired = [heater.node_id for heater in paired]
        if not self._paired and self._pair_error is None:
            self._pair_error = "no_heaters_paired"
        return self.async_show_progress_done(next_step_id="pair_done")

    async def async_step_pair_done(self, user_input: dict[str, Any] | None = None):
        """Add the paired heaters and reload, or show the pairing form with the error."""
        if not self._paired:
            return await self.async_step_pair_heaters()
        add_nodes(self.hass, self.entry, self._paired)
        return self.async_create_entry(title="", data=dict(self.entry.options))

    async def async_step_rehome(self, user_input: dict[str, Any] | None = None):
        """Explain the network move; start it when the user submits."""
        if user_input is None:
            heaters = ", ".join(
                str(node.get("name") or node.get("addr"))
                for node in self.entry.data.get(CONF_NODES, [])
            )
            return self.async_show_form(
                step_id="rehome",
                data_schema=vol.Schema({}),
                description_placeholders={"heaters": heaters},
            )
        return await self.async_step_rehome_run()

    async def async_step_rehome_run(self, user_input: dict[str, Any] | None = None):
        """Reset, re-pair and restore the heaters while showing progress."""
        if self._rehome_task is None:
            try:
                runtime = require_runtime(self.hass, self.entry.entry_id)
            except LookupError:
                return self.async_abort(reason="not_loaded")
            self._rehome_task = self.hass.async_create_task(
                async_rehome(self.hass, runtime, PAIR_WINDOW_S)
            )
        if not self._rehome_task.done():
            return self.async_show_progress(
                step_id="rehome_run",
                progress_action="rehome",
                progress_task=self._rehome_task,
            )
        task, self._rehome_task = self._rehome_task, None
        try:
            self._rehome_summary = rehome_summary(task.result())
        except RehomeError as err:
            self._rehome_summary = str(err)
        except Exception:
            _LOGGER.exception("Unexpected error while moving the radio heaters")
            self._rehome_summary = (
                "Unexpected error. See the Home Assistant log. Heaters that were "
                "reset keep their saved settings for the Radio pair action."
            )
        return self.async_show_progress_done(next_step_id="rehome_done")

    async def async_step_rehome_done(self, user_input: dict[str, Any] | None = None):
        """Show what the move did; close the options when the user confirms."""
        # A move refused before its first await finishes eagerly, and the flow
        # manager then re-sends the rehome form's input here: show the summary.
        if user_input is None or not self._rehome_summary_shown:
            self._rehome_summary_shown = True
            return self.async_show_form(
                step_id="rehome_done",
                data_schema=vol.Schema({}),
                description_placeholders={"summary": self._rehome_summary},
            )
        return self.async_create_entry(title="", data=dict(self.entry.options))
