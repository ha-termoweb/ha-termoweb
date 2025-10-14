from __future__ import annotations

import types
from typing import Any

import pytest

import custom_components.termoweb.inventory as inventory_module
from custom_components.termoweb.const import DOMAIN
import custom_components.termoweb.identifiers as identifiers_module
from custom_components.termoweb.identifiers import build_heater_energy_unique_id
from custom_components.termoweb.inventory import (
    HEATER_NODE_TYPES,
    addresses_by_node_type,
    build_node_inventory,
    normalize_heater_addresses,
    normalize_node_addr,
    normalize_node_type,
    parse_heater_energy_unique_id,
)
from custom_components.termoweb.utils import (
    _entry_gateway_record,
    async_get_integration_version,
    build_gateway_device_info,
    coerce_power_watts,
    extract_power_timestamp,
    extract_power_watts,
    float_or_none,
    format_timestamp_iso,
)


def test_addresses_by_node_type_skips_invalid_entries() -> None:
    nodes = [
        types.SimpleNamespace(type=" ", addr="skip"),
        types.SimpleNamespace(type="acm", addr=""),
        types.SimpleNamespace(type="acm", addr="B"),
        types.SimpleNamespace(type="acm", addr="B"),
    ]

    mapping, unknown = addresses_by_node_type(nodes, known_types=["htr"])
    assert mapping == {"acm": ["B"]}
    assert unknown == {"acm"}


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("abc", None),
        ("123", 123.0),
        (5, 5.0),
        ("   ", None),
        (float("nan"), None),
        (float("inf"), None),
    ],
)
def test_float_or_none(value, expected) -> None:
    assert float_or_none(value) == expected


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_float_or_none_non_finite_strings(value) -> None:
    assert float_or_none(value) is None


@pytest.mark.parametrize(
    "value,default,use_default_when_falsey,expected",
    [
        (" HTR ", "htr", False, "htr"),
        ("AcM", "htr", False, "acm"),
        (None, "htr", True, "htr"),
        ("  ", "htr", False, "htr"),
        (None, "", False, "none"),
    ],
)
def test_normalize_node_type_cases(
    value: Any, default: str, use_default_when_falsey: bool, expected: str
) -> None:
    assert (
        normalize_node_type(
            value,
            default=default,
            use_default_when_falsey=use_default_when_falsey,
        )
        == expected
    )


@pytest.mark.parametrize(
    "value,default,use_default_when_falsey,expected",
    [
        (" 01 ", "", False, "01"),
        ("  ", "fallback", False, "fallback"),
        (None, "", True, ""),
        (None, "fallback", False, "None"),
        ("none", "", False, "none"),
    ],
)
def test_normalize_node_addr_cases(
    value: Any, default: str, use_default_when_falsey: bool, expected: str
) -> None:
    assert (
        normalize_node_addr(
            value,
            default=default,
            use_default_when_falsey=use_default_when_falsey,
        )
        == expected
    )


def test_entry_gateway_record_handles_invalid_sources() -> None:
    assert _entry_gateway_record(None, "entry") is None

    hass = types.SimpleNamespace(data={DOMAIN: {}})
    assert _entry_gateway_record(hass, None) is None

    hass = types.SimpleNamespace(data={DOMAIN: []})
    assert _entry_gateway_record(hass, "entry") is None

    hass = types.SimpleNamespace(data={DOMAIN: {"entry": []}})
    assert _entry_gateway_record(hass, "entry") is None


def test_entry_gateway_record_returns_valid_mapping() -> None:
    """A well-formed mapping should be returned unchanged."""

    entry_mapping = {"brand": "TermoWeb"}
    hass = types.SimpleNamespace(data={DOMAIN: {"entry": entry_mapping}})

    assert _entry_gateway_record(hass, "entry") is entry_mapping


def test_build_gateway_device_info_defaults_without_entry() -> None:
    hass = types.SimpleNamespace(data={})

    info = build_gateway_device_info(hass, "entry", "dev")

    assert info["identifiers"] == {(DOMAIN, "dev")}
    assert info["manufacturer"] == "TermoWeb"
    assert "sw_version" not in info


def test_build_gateway_device_info_uses_brand_and_version() -> None:
    hass = types.SimpleNamespace(
        data={DOMAIN: {"entry": {"brand": "  Ducaheat  ", "version": 7}}}
    )

    info = build_gateway_device_info(hass, "entry", "dev")

    assert info["manufacturer"] == "Ducaheat"
    assert info["sw_version"] == "7"


def test_build_gateway_device_info_respects_include_version_flag() -> None:
    hass = types.SimpleNamespace(
        data={
            DOMAIN: {
                "entry": {
                    "brand": "Ducaheat",
                    "version": "9.1",
                    "coordinator": types.SimpleNamespace(
                        data={"dev": {"raw": {"model": "Controller"}}}
                    ),
                }
            }
        }
    )

    info = build_gateway_device_info(hass, "entry", "dev", include_version=False)

    assert info["manufacturer"] == "Ducaheat"
    assert info["model"] == "Controller"
    assert "sw_version" not in info


def test_build_gateway_device_info_uses_gateway_model_from_coordinator() -> None:
    hass = types.SimpleNamespace(
        data={
            DOMAIN: {
                "entry": {
                    "coordinator": types.SimpleNamespace(
                        data={"dev": {"raw": {"model": "Controller"}}}
                    )
                }
            }
        }
    )

    info = build_gateway_device_info(hass, "entry", "dev")

    assert info["model"] == "Controller"


def test_build_gateway_device_info_ignores_non_mapping_coordinator_data() -> None:
    hass = types.SimpleNamespace(
        data={DOMAIN: {"entry": {"coordinator": types.SimpleNamespace(data=[1, 2, 3])}}}
    )

    info = build_gateway_device_info(hass, "entry", "dev")

    assert info["model"] == "Gateway/Controller"


def test_build_gateway_device_info_discards_truthy_non_mapping_data() -> None:
    class TruthyNonMapping:
        def __bool__(self) -> bool:
            return True

    hass = types.SimpleNamespace(
        data={
            DOMAIN: {
                "entry": {
                    "coordinator": types.SimpleNamespace(data=TruthyNonMapping()),
                }
            }
        }
    )

    info = build_gateway_device_info(hass, "entry", "dev")

    assert info["model"] == "Gateway/Controller"


def test_normalize_heater_addresses_with_none() -> None:
    mapping, aliases = normalize_heater_addresses(None)

    assert mapping == {"htr": []}
    assert aliases == {"htr": "htr"}


def test_normalize_heater_addresses_accepts_string_sources() -> None:
    """String inputs should be coerced into normalised heater maps."""

    mapping, aliases = normalize_heater_addresses(
        {"heater": " 1 ", "acm": ["2", "2", " "]}
    )

    assert mapping == {"htr": ["1"], "acm": ["2"]}
    assert aliases["heater"] == "htr"


def test_get_brand_api_base_fallback() -> None:
    from custom_components.termoweb import const

    assert const.get_brand_api_base("unknown-brand") == const.API_BASE


def test_build_heater_energy_unique_id_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object, dict[str, Any]]] = []

    original_normalize_type = inventory_module.normalize_node_type
    original_normalize_addr = inventory_module.normalize_node_addr

    def _record_type(value, **kwargs):
        calls.append(("type", value, kwargs))
        return original_normalize_type(value, **kwargs)

    def _record_addr(value, **kwargs):
        calls.append(("addr", value, kwargs))
        return original_normalize_addr(value, **kwargs)

    monkeypatch.setattr(inventory_module, "normalize_node_type", _record_type)
    monkeypatch.setattr(inventory_module, "normalize_node_addr", _record_addr)
    monkeypatch.setattr(identifiers_module, "normalize_node_type", _record_type)
    monkeypatch.setattr(identifiers_module, "normalize_node_addr", _record_addr)
    unique_id = build_heater_energy_unique_id(" dev ", " ACM ", " 01 ")

    assert unique_id == f"{DOMAIN}:dev:acm:01:energy"
    assert parse_heater_energy_unique_id(unique_id) == ("dev", "acm", "01")
    assert calls == [
        ("addr", " dev ", {}),
        ("type", " ACM ", {}),
        ("addr", " 01 ", {}),
    ]


@pytest.mark.parametrize(
    "dev_id, node_type, addr",
    [
        ("", "htr", "01"),
        ("dev", "", "01"),
        ("dev", "htr", ""),
        ("dev", " ", "01"),
        ("dev", "htr", "  "),
    ],
)
def test_build_heater_energy_unique_id_requires_components(
    dev_id: str, node_type: str, addr: str
) -> None:
    with pytest.raises(ValueError):
        build_heater_energy_unique_id(dev_id, node_type, addr)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "not-domain:dev:htr:01:energy",
        f"{DOMAIN}:dev:htr:01:power",
        f"{DOMAIN}:dev:htr:energy",
        f"{DOMAIN}:dev:htr",
        f"{DOMAIN}:dev::01:energy",
    ],
)
def test_parse_heater_energy_unique_id_invalid(value) -> None:
    assert parse_heater_energy_unique_id(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (0, 0.0),
        (5.5, 5.5),
        ("12", 12.0),
        ("15 W", 15.0),
        ("1.2kW", 1200.0),
        ({"power": "3.5"}, 3.5),
        (["skip", {"watts": 7}], 7.0),
    ),
)
def test_coerce_power_watts_variants(value: Any, expected: float) -> None:
    """Instant power coercion should normalise multiple payload shapes."""

    assert coerce_power_watts(value) == expected


@pytest.mark.parametrize(
    "value",
    (
        -5,
        "-4",
        float("nan"),
        float("inf"),
        "-1kW",
        {"value": -1},
    ),
)
def test_coerce_power_watts_rejects_invalid(value: Any) -> None:
    """Negative and non-finite inputs should be rejected."""

    assert coerce_power_watts(value) is None


def test_extract_power_watts_nested_payload() -> None:
    """Instant power extraction should drill into nested payloads."""

    payload = {
        "status": {
            "data": [
                {"value": "ignore"},
                {"power": {"watts": "45"}},
            ]
        }
    }

    assert extract_power_watts(payload) == 45.0


def test_extract_power_watts_handles_missing_values() -> None:
    """Payloads without power data should return ``None``."""

    assert extract_power_watts({"status": {"data": []}}) is None


def test_extract_power_timestamp_variants() -> None:
    """Timestamp extraction should handle integers, milliseconds, and strings."""

    iso_ts = "2024-01-01T00:00:05+00:00"
    payload = {
        "ts": 1_700_000_000_000,
        "status": {"metrics": [{"updatedAt": iso_ts}]},
    }

    assert extract_power_timestamp(payload) == 1_700_000_000.0

    assert extract_power_timestamp({"status": {"updated_at": iso_ts}}) == pytest.approx(
        extract_power_timestamp(iso_ts)
    )


def test_format_timestamp_iso_handles_values() -> None:
    """Timestamp formatter should convert seconds to an ISO string."""

    assert format_timestamp_iso(None) is None
    assert format_timestamp_iso("not-a-number") is None

    iso_value = format_timestamp_iso(1_700_000_000.0)
    assert isinstance(iso_value, str)
    assert iso_value.endswith("+00:00")


@pytest.mark.asyncio
async def test_async_get_integration_version() -> None:
    hass = types.SimpleNamespace(integration_requests=[])

    assert await async_get_integration_version(hass) == "test-version"
    assert hass.integration_requests == [DOMAIN]
