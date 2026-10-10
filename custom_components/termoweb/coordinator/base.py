"""Shared coordinator base class and logger helper."""

from __future__ import annotations

from typing import Any, TypeVar

from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

_DataT = TypeVar("_DataT")


class RaiseUpdateFailedCoordinator(DataUpdateCoordinator[_DataT]):
    """Coordinator that propagates ``UpdateFailed`` to manual refresh callers."""

    async def async_refresh(self) -> None:
        """Refresh data and raise ``UpdateFailed`` when polling fails."""

        await super().async_refresh()
        exc = getattr(self, "last_exception", None)
        if not self.last_update_success and isinstance(exc, UpdateFailed):
            raise exc


def wrap_logger(logger: Any) -> Any:
    """Return a logger proxy that suppresses coordinator manual-update noise."""

    suppressed_message = "Manually updated %s data"

    class _LoggerProxy:
        """Proxy that adds ``isEnabledFor`` support and filters known noise."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def debug(self, msg: Any, *args: Any, **kwargs: Any) -> Any:
            if msg == suppressed_message:
                return None
            return self._inner.debug(msg, *args, **kwargs)

    return _LoggerProxy(logger)
