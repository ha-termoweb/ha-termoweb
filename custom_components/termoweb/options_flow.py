"""Options flow for radio gateway entries: moving heaters to the site network."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntry
import voluptuous as vol

from .backend.radio.pairing import PAIR_WINDOW_S
from .const import CONF_NODES
from .radio_rehome import RehomeError, async_rehome, rehome_summary
from .runtime import require_runtime

_LOGGER = logging.getLogger(__name__)


class TermoWebOptionsFlow(config_entries.OptionsFlow):
    """Options flow for radio gateway entries: moving heaters to the site network."""

    def __init__(self, entry: ConfigEntry) -> None:
        """Store the entry being configured."""
        self.entry = entry
        self._rehome_task: asyncio.Task[Any] | None = None
        self._rehome_summary = ""
        self._rehome_summary_shown = False

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        """Offer the network move; heaters are paired with the gateway's button."""
        return await self.async_step_rehome(user_input)

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
                'reset keep their saved settings: pair them with "Pair new heaters".'
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
