"""Tests for Ducaheat read model parsing and coercion helpers."""

from __future__ import annotations

import pytest

from custom_components.termoweb.codecs.ducaheat_read_models import (
    DucaheatExtraOptions,
    DucaheatSegmentedSettings,
    DucaheatSetupSegment,
    DucaheatStatusSegment,
    DucaheatThermostatSettings,
    _coerce_bool,
    _coerce_int,
    _coerce_number,
    _coerce_percentage,
    _normalise_prog,
    _normalise_prog_temps,
    _safe_temperature,
)


# ---------------------------------------------------------------------------
# _coerce_bool
# ---------------------------------------------------------------------------


class TestCoerceBool:
    """Tests for the _coerce_bool helper."""

    def test_none_returns_none(self) -> None:
        assert _coerce_bool(None) is None

    def test_bool_passthrough(self) -> None:
        assert _coerce_bool(True) is True
        assert _coerce_bool(False) is False

    def test_int_coercion(self) -> None:
        assert _coerce_bool(1) is True
        assert _coerce_bool(0) is False

    def test_float_coercion(self) -> None:
        assert _coerce_bool(1.0) is True
        assert _coerce_bool(0.0) is False

    def test_string_truthy(self) -> None:
        for val in ("1", "true", "yes", "on", " TRUE ", " Yes "):
            assert _coerce_bool(val) is True

    def test_string_falsy(self) -> None:
        for val in ("0", "false", "no", "off", " FALSE ", " No "):
            assert _coerce_bool(val) is False

    def test_unrecognised_string_returns_none(self) -> None:
        assert _coerce_bool("maybe") is None

    def test_non_string_non_number_returns_none(self) -> None:
        assert _coerce_bool([1, 2, 3]) is None


# ---------------------------------------------------------------------------
# _coerce_number
# ---------------------------------------------------------------------------


class TestCoerceNumber:
    def test_int_passthrough(self) -> None:
        assert _coerce_number(42) == 42

    def test_float_passthrough(self) -> None:
        assert _coerce_number(3.14) == 3.14

    def test_none_returns_none(self) -> None:
        assert _coerce_number(None) is None

    def test_string_number(self) -> None:
        assert _coerce_number("  21.5 ") == 21.5

    def test_invalid_string_returns_none(self) -> None:
        assert _coerce_number("abc") is None


# ---------------------------------------------------------------------------
# _coerce_percentage
# ---------------------------------------------------------------------------


class TestCoercePercentage:
    def test_valid_percentage(self) -> None:
        assert _coerce_percentage(50) == 50

    def test_clamps_above_100(self) -> None:
        assert _coerce_percentage(150) == 100

    def test_clamps_below_0(self) -> None:
        assert _coerce_percentage(-10) == 0

    def test_none_returns_none(self) -> None:
        assert _coerce_percentage(None) is None

    def test_invalid_returns_none(self) -> None:
        assert _coerce_percentage("abc") is None

    def test_float_truncated(self) -> None:
        assert _coerce_percentage(75.9) == 75


# ---------------------------------------------------------------------------
# _coerce_int
# ---------------------------------------------------------------------------


class TestCoerceInt:
    def test_valid_int(self) -> None:
        assert _coerce_int(42) == 42

    def test_string_int(self) -> None:
        assert _coerce_int("7") == 7

    def test_none_returns_none(self) -> None:
        assert _coerce_int(None) is None

    def test_invalid_returns_none(self) -> None:
        assert _coerce_int("abc") is None


# ---------------------------------------------------------------------------
# _safe_temperature
# ---------------------------------------------------------------------------


class TestSafeTemperature:
    def test_none_returns_none(self) -> None:
        assert _safe_temperature(None) is None

    def test_valid_number(self) -> None:
        assert _safe_temperature(21.5) == "21.5"

    def test_invalid_string_returned_cleaned(self) -> None:
        # Strings that can't be parsed as floats are cleaned and returned
        assert _safe_temperature("warm") == "warm"

    def test_empty_string_returns_none(self) -> None:
        assert _safe_temperature("") is None


# ---------------------------------------------------------------------------
# _normalise_prog
# ---------------------------------------------------------------------------


class TestNormaliseProg:
    def test_valid_168_list(self) -> None:
        prog = [0, 1, 2] * 56
        assert _normalise_prog(prog) == prog

    def test_invalid_values_in_list(self) -> None:
        prog = [3] * 168  # 3 is not in (0, 1, 2)
        assert _normalise_prog(prog) is None

    def test_non_integer_list(self) -> None:
        prog = ["x"] * 168
        assert _normalise_prog(prog) is None

    def test_wrong_length_list(self) -> None:
        assert _normalise_prog([0] * 100) is None

    def test_non_iterable_returns_none(self) -> None:
        assert _normalise_prog(42) is None

    def test_mapping_with_days_section(self) -> None:
        data = {
            "days": {
                "mon": [0] * 24,
                "tue": [1] * 24,
                "wed": [0] * 24,
                "thu": [1] * 24,
                "fri": [0] * 24,
                "sat": [2] * 24,
                "sun": [0] * 24,
            }
        }
        result = _normalise_prog(data)
        assert result is not None
        assert len(result) == 168

    def test_mapping_with_numeric_keys(self) -> None:
        data = {str(i): [0] * 24 for i in range(7)}
        result = _normalise_prog(data)
        assert result is not None
        assert len(result) == 168

    def test_mapping_missing_day_fills_zeros(self) -> None:
        # Only 'mon' present, rest should be filled with zeros
        data = {"days": {"mon": [1] * 24}}
        result = _normalise_prog(data)
        assert result is not None
        assert len(result) == 168
        assert result[:24] == [1] * 24
        assert result[24:48] == [0] * 24

    def test_mapping_with_nested_slots(self) -> None:
        day_data = {"slots": [0] * 24}
        data = {"days": {day: day_data for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")}}
        result = _normalise_prog(data)
        assert result is not None
        assert len(result) == 168

    def test_mapping_with_values_key(self) -> None:
        day_data = {"values": [0] * 24}
        data = {"days": {day: day_data for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")}}
        result = _normalise_prog(data)
        assert result is not None

    def test_mapping_48_slot_downsampled(self) -> None:
        # 48-slot day should be downsampled to 24
        day_data = [0, 1] * 24  # 48 slots
        data = {"days": {day: day_data for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")}}
        result = _normalise_prog(data)
        assert result is not None
        assert len(result) == 168

    def test_mapping_short_day_padded(self) -> None:
        # < 24 slots should be padded with zeros
        day_data = [1] * 10
        data = {"days": {day: day_data for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")}}
        result = _normalise_prog(data)
        assert result is not None
        assert len(result) == 168
        assert result[10:24] == [0] * 14  # padded portion

    def test_mapping_long_day_truncated(self) -> None:
        day_data = [1] * 30
        data = {"days": {day: day_data for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")}}
        result = _normalise_prog(data)
        assert result is not None
        assert len(result) == 168

    def test_mapping_none_slot_entry(self) -> None:
        day_data = {"slots": None, "values": None}
        data = {"days": {day: day_data for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")}}
        result = _normalise_prog(data)
        assert result is None

    def test_mapping_prog_fallback(self) -> None:
        data = {"prog": {day: [0] * 24 for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")}}
        result = _normalise_prog(data)
        assert result is not None
        assert len(result) == 168

    def test_mapping_with_bad_day_data(self) -> None:
        data = {"days": {"mon": ["x"] * 24}}
        result = _normalise_prog(data)
        assert result is None


# ---------------------------------------------------------------------------
# _normalise_prog_temps
# ---------------------------------------------------------------------------


class TestNormaliseProgTemps:
    def test_non_mapping_returns_none(self) -> None:
        assert _normalise_prog_temps([1, 2, 3]) is None

    def test_valid_temps(self) -> None:
        data = {"antifrost": 7, "eco": 16, "comfort": 21}
        result = _normalise_prog_temps(data)
        assert result is not None
        assert len(result) == 3
        assert result[0] == "7.0"

    def test_alt_keys(self) -> None:
        data = {"cold": 5, "night": 17, "day": 22}
        result = _normalise_prog_temps(data)
        assert result is not None
        assert result[0] == "5.0"

    def test_none_values_produce_empty_strings(self) -> None:
        data = {"antifrost": None, "eco": 16, "comfort": None}
        result = _normalise_prog_temps(data)
        assert result is not None
        assert result[0] == ""
        assert result[2] == ""

    def test_invalid_temp_falls_back_to_str(self) -> None:
        data = {"antifrost": "warm", "eco": 16, "comfort": 21}
        result = _normalise_prog_temps(data)
        assert result is not None
        assert result[0] == "warm"


# ---------------------------------------------------------------------------
# DucaheatStatusSegment
# ---------------------------------------------------------------------------


class TestDucaheatStatusSegment:
    def test_mode_normalised_to_lowercase(self) -> None:
        seg = DucaheatStatusSegment.model_validate({"mode": "AUTO"})
        assert seg.mode == "auto"

    def test_empty_mode_returns_none(self) -> None:
        seg = DucaheatStatusSegment.model_validate({"mode": "  "})
        assert seg.mode is None

    def test_state_normalised_to_lowercase(self) -> None:
        seg = DucaheatStatusSegment.model_validate({"state": "ON"})
        assert seg.state == "on"

    def test_empty_state_returns_none(self) -> None:
        seg = DucaheatStatusSegment.model_validate({"state": "  "})
        assert seg.state is None

    def test_units_cleaned(self) -> None:
        seg = DucaheatStatusSegment.model_validate({"units": " f "})
        assert seg.units == "F"

    def test_invalid_units_returns_none(self) -> None:
        seg = DucaheatStatusSegment.model_validate({"units": "X"})
        assert seg.units is None

    def test_temperature_formatting(self) -> None:
        seg = DucaheatStatusSegment.model_validate({"stemp": 21.234, "mtemp": 19})
        assert seg.stemp == "21.2"
        assert seg.mtemp == "19.0"

    def test_boolean_coercion_for_lock(self) -> None:
        seg = DucaheatStatusSegment.model_validate({"lock": "yes"})
        assert seg.lock is True

    def test_boost_time_validation(self) -> None:
        seg = DucaheatStatusSegment.model_validate({"boost_time": 120})
        assert seg.boost_time == 120

    def test_boost_time_none(self) -> None:
        seg = DucaheatStatusSegment.model_validate({"boost_time": None})
        assert seg.boost_time is None

    def test_boost_time_non_allowed_returned_as_is(self) -> None:
        # 30 is not in ALLOWED_BOOST_MINUTES_SET, but returned anyway
        seg = DucaheatStatusSegment.model_validate({"boost_time": 30})
        assert seg.boost_time == 30

    def test_numeric_coercion_for_boost_remaining(self) -> None:
        seg = DucaheatStatusSegment.model_validate({"boost_remaining": "45"})
        assert seg.boost_remaining == 45.0

    def test_boost_end_day_min_coercion(self) -> None:
        seg = DucaheatStatusSegment.model_validate(
            {"boost_end_day": "3", "boost_end_min": "15"}
        )
        assert seg.boost_end_day == 3
        assert seg.boost_end_min == 15

    def test_charge_percentage_clamping(self) -> None:
        seg = DucaheatStatusSegment.model_validate(
            {"current_charge_per": 150, "target_charge_per": -5}
        )
        assert seg.current_charge_per == 100
        assert seg.target_charge_per == 0

    def test_boost_end_nested_mapping(self) -> None:
        seg = DucaheatStatusSegment.model_validate(
            {"boost_end": {"day": 2, "minute": 30}}
        )
        assert seg.boost_end_day == 2
        assert seg.boost_end_min == 30

    def test_boost_active_promoted_from_boost_flag(self) -> None:
        seg = DucaheatStatusSegment.model_validate({"boost": True})
        assert seg.boost_active is True

    def test_boost_active_not_overridden_when_explicit(self) -> None:
        seg = DucaheatStatusSegment.model_validate(
            {"boost_active": False, "boost": True}
        )
        assert seg.boost_active is False


# ---------------------------------------------------------------------------
# DucaheatExtraOptions
# ---------------------------------------------------------------------------


class TestDucaheatExtraOptions:
    def test_boost_time_none(self) -> None:
        opts = DucaheatExtraOptions.model_validate({"boost_time": None})
        assert opts.boost_time is None

    def test_boost_time_non_allowed(self) -> None:
        opts = DucaheatExtraOptions.model_validate({"boost_time": 30})
        assert opts.boost_time == 30

    def test_charging_coercion(self) -> None:
        opts = DucaheatExtraOptions.model_validate({"charging": "yes"})
        assert opts.charging is True

    def test_percentage_clamping(self) -> None:
        opts = DucaheatExtraOptions.model_validate(
            {"current_charge_per": 200, "target_charge_per": -1}
        )
        assert opts.current_charge_per == 100
        assert opts.target_charge_per == 0

    def test_boost_end_nested(self) -> None:
        opts = DucaheatExtraOptions.model_validate(
            {"boost_end": {"day": 5, "minute": 45}}
        )
        assert opts.boost_end_day == 5
        assert opts.boost_end_min == 45


# ---------------------------------------------------------------------------
# DucaheatSetupSegment
# ---------------------------------------------------------------------------


class TestDucaheatSetupSegment:
    def test_boost_time_none(self) -> None:
        setup = DucaheatSetupSegment.model_validate({"boost_time": None})
        assert setup.boost_time is None

    def test_boost_time_non_allowed_passthrough(self) -> None:
        setup = DucaheatSetupSegment.model_validate({"boost_time": 30})
        assert setup.boost_time == 30

    def test_charging_coercion(self) -> None:
        setup = DucaheatSetupSegment.model_validate({"charging": 1})
        assert setup.charging is True

    def test_percentage_clamping(self) -> None:
        setup = DucaheatSetupSegment.model_validate(
            {"current_charge_per": 110, "target_charge_per": -5}
        )
        assert setup.current_charge_per == 100
        assert setup.target_charge_per == 0

    def test_priority_coercion(self) -> None:
        setup = DucaheatSetupSegment.model_validate({"priority": "3"})
        assert setup.priority == 3

    def test_boost_end_nested(self) -> None:
        setup = DucaheatSetupSegment.model_validate(
            {"boost_end": {"day": 1, "minute": 10}}
        )
        assert setup.boost_end_day == 1
        assert setup.boost_end_min == 10


# ---------------------------------------------------------------------------
# DucaheatThermostatSettings
# ---------------------------------------------------------------------------


class TestDucaheatThermostatSettings:
    def test_mode_normalisation(self) -> None:
        thm = DucaheatThermostatSettings.model_validate({"mode": "AUTO"})
        assert thm.mode == "auto"

    def test_empty_mode_returns_none(self) -> None:
        thm = DucaheatThermostatSettings.model_validate({"mode": "  "})
        assert thm.mode is None

    def test_state_normalisation(self) -> None:
        thm = DucaheatThermostatSettings.model_validate({"state": "ON"})
        assert thm.state == "on"

    def test_empty_state_returns_none(self) -> None:
        thm = DucaheatThermostatSettings.model_validate({"state": "  "})
        assert thm.state is None

    def test_temp_coercion(self) -> None:
        thm = DucaheatThermostatSettings.model_validate(
            {"stemp": "21.5", "mtemp": 19}
        )
        assert thm.stemp == 21.5
        assert thm.mtemp == 19.0

    def test_temp_invalid_returns_none(self) -> None:
        thm = DucaheatThermostatSettings.model_validate({"stemp": "abc"})
        assert thm.stemp is None

    def test_units_normalisation(self) -> None:
        thm = DucaheatThermostatSettings.model_validate({"units": " c "})
        assert thm.units == "C"

    def test_invalid_units(self) -> None:
        thm = DucaheatThermostatSettings.model_validate({"units": "X"})
        assert thm.units is None

    def test_ptemp_from_mapping(self) -> None:
        thm = DucaheatThermostatSettings.model_validate(
            {"ptemp": {"antifrost": 7, "eco": 16, "comfort": 21}}
        )
        assert thm.ptemp is not None
        assert len(thm.ptemp) == 3

    def test_ptemp_from_list(self) -> None:
        thm = DucaheatThermostatSettings.model_validate(
            {"ptemp": [7.0, 16.0, 21.0]}
        )
        assert thm.ptemp == [7.0, 16.0, 21.0]

    def test_ptemp_with_invalid_item(self) -> None:
        thm = DucaheatThermostatSettings.model_validate(
            {"ptemp": [7.0, "abc", 21.0]}
        )
        assert thm.ptemp is None

    def test_ptemp_non_iterable_returns_none(self) -> None:
        thm = DucaheatThermostatSettings.model_validate({"ptemp": 42})
        assert thm.ptemp is None

    def test_prog_normalisation(self) -> None:
        thm = DucaheatThermostatSettings.model_validate({"prog": [0] * 168})
        assert thm.prog == [0] * 168

    def test_battery_level_clamped(self) -> None:
        thm = DucaheatThermostatSettings.model_validate({"batt_level": 10})
        assert thm.batt_level == 5  # clamped to max 5

    def test_battery_level_invalid(self) -> None:
        thm = DucaheatThermostatSettings.model_validate({"batt_level": "abc"})
        assert thm.batt_level is None

    def test_battery_level_clamped_low(self) -> None:
        thm = DucaheatThermostatSettings.model_validate({"batt_level": -3})
        assert thm.batt_level == 0


# ---------------------------------------------------------------------------
# DucaheatSegmentedSettings.to_flat_dict
# ---------------------------------------------------------------------------


class TestSegmentedToFlatDict:
    def test_heater_excludes_accumulator_fields(self) -> None:
        settings = DucaheatSegmentedSettings.model_validate(
            {
                "status": {
                    "mode": "auto",
                    "stemp": "21.0",
                    "boost_active": True,
                    "charging": True,
                },
            }
        )
        flat = settings.to_flat_dict(accumulator=False)
        assert "mode" in flat
        assert "boost_active" not in flat
        assert "charging" not in flat

    def test_accumulator_includes_setup_metadata(self) -> None:
        settings = DucaheatSegmentedSettings.model_validate(
            {
                "status": {"mode": "auto", "stemp": "21.0"},
                "setup": {
                    "boost_time": 120,
                    "current_charge_per": 50,
                    "extra_options": {"boost_time": 60, "charging": True},
                },
            }
        )
        flat = settings.to_flat_dict(accumulator=True)
        assert flat.get("mode") == "auto"

    def test_prog_included(self) -> None:
        settings = DucaheatSegmentedSettings.model_validate(
            {"prog": [0] * 168}
        )
        flat = settings.to_flat_dict(accumulator=False)
        assert flat["prog"] == [0] * 168

    def test_prog_temps_included(self) -> None:
        settings = DucaheatSegmentedSettings.model_validate(
            {"prog_temps": {"antifrost": 7, "eco": 16, "comfort": 21}}
        )
        flat = settings.to_flat_dict(accumulator=False)
        assert "ptemp" in flat

    def test_priority_from_setup(self) -> None:
        settings = DucaheatSegmentedSettings.model_validate(
            {"setup": {"priority": 3}}
        )
        flat = settings.to_flat_dict(accumulator=False)
        assert flat.get("priority") == 3

    def test_accumulator_with_extra_options_boost_metadata(self) -> None:
        """Extra options boost metadata is merged into the flat dict."""
        settings = DucaheatSegmentedSettings.model_validate(
            {
                "status": {"mode": "auto", "stemp": "21.0"},
                "setup": {
                    "extra_options": {
                        "boost_time": 60,
                        "boost_temp": "25.0",
                        "boost_end_day": 3,
                        "boost_end_min": 30,
                        "charging": True,
                        "current_charge_per": 80,
                        "target_charge_per": 90,
                    }
                },
            }
        )
        flat = settings.to_flat_dict(accumulator=True)
        assert flat.get("boost_time") == 60
        assert flat.get("boost_temp") == "25.0"

    def test_accumulator_setup_direct_metadata(self) -> None:
        """Setup-level boost and charge metadata is merged."""
        settings = DucaheatSegmentedSettings.model_validate(
            {
                "status": {"mode": "auto"},
                "setup": {
                    "boost_end_day": 2,
                    "boost_end_min": 15,
                    "charging": True,
                    "current_charge_per": 60,
                    "target_charge_per": 100,
                },
            }
        )
        flat = settings.to_flat_dict(accumulator=True)
        assert flat.get("boost_end_day") == 2
        assert flat.get("charging") is True


# ---------------------------------------------------------------------------
# _merge_boost_metadata
# ---------------------------------------------------------------------------


from custom_components.termoweb.codecs.ducaheat_read_models import (
    _merge_boost_metadata,
    _merge_accumulator_charge_metadata,
)


class TestMergeBoostMetadata:
    def test_non_mapping_source_noop(self) -> None:
        target: dict = {}
        _merge_boost_metadata(target, None)
        assert target == {}

    def test_boost_active_from_source(self) -> None:
        target: dict = {}
        _merge_boost_metadata(target, {"boost_active": True})
        assert target["boost_active"] is True

    def test_boost_promoted_from_boost_flag(self) -> None:
        target: dict = {}
        _merge_boost_metadata(target, {"boost": True})
        assert target["boost_active"] is True

    def test_boost_end_day_min_from_source(self) -> None:
        target: dict = {}
        _merge_boost_metadata(target, {"boost_end_day": 5, "boost_end_min": 30})
        assert target["boost_end_day"] == 5
        assert target["boost_end_min"] == 30

    def test_boost_end_nested_mapping(self) -> None:
        target: dict = {}
        _merge_boost_metadata(target, {"boost_end": {"day": 3, "minute": 45}})
        assert target["boost_end_day"] == 3
        assert target["boost_end_min"] == 45

    def test_prefer_existing_preserves_target(self) -> None:
        target: dict = {"boost_active": False}
        _merge_boost_metadata(
            target, {"boost_active": True}, prefer_existing=True
        )
        assert target["boost_active"] is False

    def test_prefer_existing_fills_none(self) -> None:
        target: dict = {"boost_active": None}
        _merge_boost_metadata(
            target, {"boost_active": True}, prefer_existing=True
        )
        assert target["boost_active"] is True

    def test_prefer_existing_both_none_no_overwrite(self) -> None:
        target: dict = {"boost_end_day": None}
        _merge_boost_metadata(
            target,
            {"boost_end": {"day": None, "minute": None}},
            prefer_existing=True,
        )
        # None values not assigned due to allow_none=False default
        assert target["boost_end_day"] is None


# ---------------------------------------------------------------------------
# _merge_accumulator_charge_metadata
# ---------------------------------------------------------------------------


class TestMergeAccumulatorChargeMetadata:
    def test_non_mapping_source_noop(self) -> None:
        target: dict = {}
        _merge_accumulator_charge_metadata(target, None)
        assert target == {}

    def test_charging_merged(self) -> None:
        target: dict = {}
        _merge_accumulator_charge_metadata(target, {"charging": True})
        assert target["charging"] is True

    def test_charge_percentages_merged(self) -> None:
        target: dict = {}
        _merge_accumulator_charge_metadata(
            target, {"current_charge_per": 50, "target_charge_per": 80}
        )
        assert target["current_charge_per"] == 50
        assert target["target_charge_per"] == 80

    def test_prefer_existing_preserves(self) -> None:
        target: dict = {"charging": False}
        _merge_accumulator_charge_metadata(
            target, {"charging": True}, prefer_existing=True
        )
        assert target["charging"] is False

    def test_prefer_existing_fills_none(self) -> None:
        target: dict = {"charging": None}
        _merge_accumulator_charge_metadata(
            target, {"charging": True}, prefer_existing=True
        )
        assert target["charging"] is True

    def test_invalid_percentage_skipped(self) -> None:
        target: dict = {}
        _merge_accumulator_charge_metadata(
            target, {"current_charge_per": "invalid"}
        )
        assert "current_charge_per" not in target


