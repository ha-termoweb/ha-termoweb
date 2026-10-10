from __future__ import annotations

from custom_components.termoweb.codecs.termoweb_codec import (
    decode_devs_payload,
    decode_nodes_payload,
)


def test_decode_devs_payload_list_filters_non_dicts() -> None:
    raw = [{"dev_id": "abc"}, "ignore", {"name": "ok"}]

    result = decode_devs_payload(raw)

    assert result == [{"dev_id": "abc"}, {"name": "ok"}]


def test_decode_devs_payload_dict_variants() -> None:
    raw_devs = {"devs": [{"id": 1}, "bad"]}  # legacy shape
    raw_devices = {"devices": [{"dev_id": "abc"}, 123]}  # alternate shape

    assert decode_devs_payload(raw_devs) == [{"id": 1}]
    assert decode_devs_payload(raw_devices) == [{"dev_id": "abc"}]


def test_decode_devs_payload_unexpected_shape() -> None:
    assert decode_devs_payload("oops") == []
    assert decode_devs_payload({"weird": []}) == []


def test_decode_nodes_payload_dict_normalizes_addresses() -> None:
    raw = {"nodes": [{"type": "htr", "addr": 2, "name": "Heater"}]}

    decoded = decode_nodes_payload(raw)

    assert decoded == {"nodes": [{"type": "htr", "addr": "2", "name": "Heater"}]}


def test_decode_nodes_payload_list_passthrough() -> None:
    raw = [{"type": "htr", "addr": "1"}]

    assert decode_nodes_payload(raw) is raw
