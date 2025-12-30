from __future__ import annotations

import logging

import pytest

from custom_components.termoweb.codecs.termoweb_codec import (
    decode_node_settings,
    decode_samples,
)


def test_decode_node_settings_formats_temperatures() -> None:
    raw = {
        "mode": "Auto",
        "stemp": 21,
        "mtemp": " 19.25 ",
        "ptemp": [18, "20", None],
    }

    decoded = decode_node_settings("htr", raw)

    assert decoded["mode"] == "Auto"
    assert decoded["stemp"] == "21.0"
    assert decoded["mtemp"] == "19.2"
    assert decoded["ptemp"] == ["18.0", "20.0", None]


def test_decode_node_settings_handles_status_block() -> None:
    raw = {
        "status": {
            "mode": "off",
            "stemp": 19,
            "prog": [0, 1, 2],
            "ptemp": ["7", "17.5", "21.0"],
        }
    }

    decoded = decode_node_settings("acm", raw)

    assert decoded["status"]["mode"] == "off"
    assert decoded["status"]["stemp"] == "19.0"
    assert decoded["status"]["prog"] == [0, 1, 2]
    assert decoded["status"]["ptemp"] == ["7.0", "17.5", "21.0"]


def test_decode_node_settings_preserves_short_prog() -> None:
    raw = {"prog": [0, 1, 2]}

    decoded = decode_node_settings("htr", raw)

    assert decoded["prog"] == [0, 1, 2]


def test_decode_samples_filters_invalid_items(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    raw = {
        "samples": [
            {"t": 1000, "counter": 1.5},
            {"timestamp": 2000, "value": 5},
            {"t": "bad", "counter": 3},
            {"t": 3000},
        ]
    }

    decoded = decode_samples(raw)

    assert decoded == [
        {"t": 1000, "counter": "1.5"},
        {"t": 2000, "counter": "5"},
    ]
    assert any("Unexpected htr sample shape" in rec.message for rec in caplog.records)


def test_decode_samples_applies_timestamp_divisor() -> None:
    raw = {"samples": [{"t": 1000.0, "counter": 5}]}

    decoded = decode_samples(raw, timestamp_divisor=10.0)

    assert decoded == [{"t": 100, "counter": "5"}]
