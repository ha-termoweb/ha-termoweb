from __future__ import annotations

from custom_components.termoweb.codecs import ducaheat_codec, termoweb_codec


def test_termoweb_codec_placeholders_removed() -> None:
    """Ensure TermoWeb codec placeholder helpers are absent."""

    assert not hasattr(termoweb_codec, "decode_payload")
    assert not hasattr(termoweb_codec, "encode_payload")


def test_ducaheat_codec_placeholders_removed() -> None:
    """Ensure Ducaheat codec placeholder helpers are absent."""

    assert not hasattr(ducaheat_codec, "decode_payload")
    assert not hasattr(ducaheat_codec, "decode_status")
    assert not hasattr(ducaheat_codec, "decode_samples")
    assert not hasattr(ducaheat_codec, "decode_prog")
