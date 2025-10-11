"""Tests for the `_statistics_row_get` helper."""

from types import SimpleNamespace

from custom_components.termoweb.energy import _statistics_row_get


def test_statistics_row_get_handles_dicts_and_namespaces() -> None:
    """Ensure `_statistics_row_get` supports dict rows and attribute rows."""

    dict_row = {"start": "begin", "sum": 42}
    namespace_row = SimpleNamespace(start=object(), sum=5)

    assert _statistics_row_get(dict_row, "start") == "begin"
    assert _statistics_row_get(dict_row, "sum") == 42

    start_marker = namespace_row.start
    assert _statistics_row_get(namespace_row, "start") is start_marker
    assert _statistics_row_get(namespace_row, "sum") == 5
