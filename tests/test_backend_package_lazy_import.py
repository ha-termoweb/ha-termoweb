"""Tests for lazy imports in the backend package."""
from __future__ import annotations

import importlib
from contextlib import contextmanager
from types import ModuleType
from typing import Iterator

import pytest

_MISSING = object()


def _get_backend_module() -> ModuleType:
    """Import and return the backend package."""
    return importlib.import_module("custom_components.termoweb.backend")


@contextmanager
def manage_cached_attribute(module: ModuleType, name: str) -> Iterator[None]:
    """Temporarily remove a cached attribute and restore its prior value."""
    original = module.__dict__.get(name, _MISSING)
    module.__dict__.pop(name, None)
    try:
        yield
    finally:
        if original is _MISSING:
            module.__dict__.pop(name, None)
        else:
            module.__dict__[name] = original


@pytest.fixture(name="backend_module")
def fixture_backend_module() -> ModuleType:
    """Return the backend package for lazy import testing."""
    return _get_backend_module()


def test_lazy_getattr_caches_ducaheat_backend(backend_module: ModuleType) -> None:
    """Ensure DucaheatBackend lookups populate and reuse the cache."""
    with manage_cached_attribute(backend_module, "DucaheatBackend"):
        first = getattr(backend_module, "DucaheatBackend")
        second = getattr(backend_module, "DucaheatBackend")

        assert first is second
        assert backend_module.__dict__.get("DucaheatBackend") is first


def test_lazy_getattr_caches_ducaheat_rest_client(backend_module: ModuleType) -> None:
    """Ensure DucaheatRESTClient lookups populate and reuse the cache."""
    with manage_cached_attribute(backend_module, "DucaheatRESTClient"):
        first = getattr(backend_module, "DucaheatRESTClient")
        second = getattr(backend_module, "DucaheatRESTClient")

        assert first is second
        assert backend_module.__dict__.get("DucaheatRESTClient") is first


def test_lazy_getattr_caches_termoweb_backend(backend_module: ModuleType) -> None:
    """Ensure TermoWebBackend lookups populate and reuse the cache."""
    with manage_cached_attribute(backend_module, "TermoWebBackend"):
        first = getattr(backend_module, "TermoWebBackend")
        second = getattr(backend_module, "TermoWebBackend")

        assert first is second
        assert backend_module.__dict__.get("TermoWebBackend") is first


def test_lazy_getattr_raises_for_unknown_symbol(backend_module: ModuleType) -> None:
    """Ensure unknown attributes raise AttributeError with the expected message."""
    with pytest.raises(AttributeError) as exc:
        getattr(backend_module, "MissingSymbol")

    message = str(exc.value)
    assert "has no attribute" in message
    assert "MissingSymbol" in message
