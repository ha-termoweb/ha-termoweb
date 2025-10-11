# Test Failure Recovery Plan

The recent backend sanitisation changes tightened accumulator boost validation, requiring `boost_time` values to be between 60 and 600 minutes in 60-minute increments. Several legacy tests still expect the previous behaviour and now fail. The following focused tasks update the affected test suites. Each task edits a single test module so the fixes can be executed in parallel without conflicts.

1. **tests/test_api.py** – Update accumulator API helper tests to use valid `boost_time` increments and assert the new validation error text. Refresh fixtures and parametrised cases that currently rely on 15-, 30-, or 45-minute durations.
2. **tests/test_backend_ducaheat.py** – Revise backend contract tests so boost payloads and validation helpers expect 60-minute granularity. Adjust assertions around `validate_boost_minutes` and `build_acm_boost_payload` accordingly.
3. **tests/test_climate.py** – Align accumulator climate entity tests with the sanitised boost rules. Replace invalid durations in preset-mode scenarios, update mocked metadata, and revise `validate_boost_minutes` expectations.
4. **tests/test_ducaheat_acm_writes.py** – Rework Ducaheat accumulator write tests to supply compliant boost durations and update metadata calculations for the stricter validation helper.
5. **tests/test_select.py** – Ensure the boost duration select tests exclusively cover supported increments. Update coercion parametrisation to reflect the tightened validation and remove legacy expectations for 45-minute options.
6. **tests/test_coordinator.py** – Refresh coordinator boost metadata tests so they no longer construct payloads with disallowed durations. Add coverage for handling invalid boost data returned from the backend under the new sanitiser.
7. **tests/test_heater_entities.py** – Update heater entity boost runtime tests to align with the 60-minute increment rule and extend edge-case coverage for rejected durations.
8. **tests/test_heater_energy_sensor.py** – Adjust energy sensor boost fixtures to use compliant durations and update assertions tied to boost counters now that invalid increments are filtered out earlier in the flow.

Each task is limited to a single test file and avoids modifying integration code.
