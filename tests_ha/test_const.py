"""Tests for the TermoWeb constants module."""

from __future__ import annotations

from custom_components.termoweb import const


def test_uses_ducaheat_backend_aliases() -> None:
    """Brands mapped to Ducaheat share backend selection."""

    assert const.uses_ducaheat_backend(const.BRAND_DUCAHEAT) is True
    assert const.uses_ducaheat_backend(const.BRAND_TEVOLVE) is True
    assert const.uses_ducaheat_backend(const.BRAND_TERMOWEB) is False
