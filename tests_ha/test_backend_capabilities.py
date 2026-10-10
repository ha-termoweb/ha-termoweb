"""Vendor isolation: shared code asks the backend's capabilities, not the brand."""

from __future__ import annotations

import inspect
import re
from types import ModuleType

import pytest

from custom_components import termoweb
from custom_components.termoweb import config_flow, const, diagnostics, utils
from custom_components.termoweb.identifiers import build_cloud_unique_id

# Brand names or vendor client types that only backend/ may branch on.
_VENDOR_BRANCH = re.compile(
    r"BRAND_RADIO|RADIO_BRANDS|brand(\)|\])? [!=]=|isinstance\([^)]*RadioClient"
)


@pytest.mark.parametrize(
    "source",
    [
        termoweb,
        utils,
        diagnostics,
        config_flow.TermoWebConfigFlow.async_supports_options_flow,
        config_flow.TermoWebConfigFlow.async_step_reconfigure,
        config_flow._radio_in_use,  # noqa: SLF001
        const.get_brand_label,
        build_cloud_unique_id,
    ],
    ids=lambda source: (
        getattr(source, "__qualname__", None) or getattr(source, "__name__", "?")
    ),
)
def test_shared_code_does_not_branch_on_brand(source: ModuleType) -> None:
    """Radio and brand differences come from BackendCapabilities or the factory."""
    assert not _VENDOR_BRANCH.findall(inspect.getsource(source))


@pytest.mark.parametrize(
    ("brand", "unique_id"),
    [
        (const.BRAND_TERMOWEB, "user@example.com"),
        (const.BRAND_DUCAHEAT, "ducaheat:user@example.com"),
        (const.BRAND_TEVOLVE, "ducaheat:user@example.com"),
    ],
)
def test_cloud_unique_id_is_scoped_by_backend(brand: str, unique_id: str) -> None:
    """Brands that share a backend share an account namespace."""
    assert build_cloud_unique_id(brand, " User@Example.COM ") == unique_id
