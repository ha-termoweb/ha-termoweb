# AGENTS.md

## Overview
This repository contains a Home Assistant integration for the TermoWeb cloud. TermoWeb is a platform for monitoring and controlling electric heaters, thermostats and power monitors in residential and commercial installations. There are at least 2 separate instances (named Ducaheat and TermoWeb) of the TermoWeb backend with slightly different APIs. The TermoWeb platform is hosted by Tevolve, who make three different branded Android apps: TermoWeb, Tevolve and Ducaheat.  The apps allow a user to log in to the TermoWeb backend and control their home heaters. 

Each user installs a gatway device that connects wirelessly to each heater, thermostat or power monitor and feeds data to the TermoWeb back end. TermoWeb-enabled heaters and gateways are made by a variety of manufacturers such as Ducasa, ATC, Soler & Palau, Ecotermi and others. Most of these are in Europe. The TermoWeb Android and iOS apps connect to the TermoWeb backend via REST and Websocket protocols and allow users to program and control their home heaters. 

This Home Assistant (HA) integration connects HA to TermoWeb and allows users to control their heaters from their home automation platform, instead of the mobile apps.

## Terms
- TermoWeb is the platform (backend). There are at least two instances (TermoWeb and Ducaheat) of the backend on different URLs. 
- TermoWeb, Ducaheat and Tevolve are the names of the three mobile apps. The backends for the TermoWeb and Ducaheat apps are on different endpoints and have slightly different API structure. It appears that Ducaheat is more modern. It appears Tevolve uses the same backend as Ducaheat.
- In the API, "Device", identified by a hex dev_id is the single gateway that connects each user's heaters to the TermoWeb backend.
- In the API, "Node" is a heater (htr), accumulator (acm), power monitor (pmo) or thermostat (thm) that is connected to the gateway and visible/controllable in the app.
- Each node has an "addr" attribute, which is an integer address identifying the node (eg. addr=2)
- Each node has a "type", which is one of "htr","acm","pmo", or "thm" identifying the type of node.

## Fair/Reasonable Use
- The TermoWeb backend is a hosted system that supports hundreds or potentially thousands of users.
- This integration must maintain reasonable use of the backend and avoid resource abuse. That means, we cannot hit the REST API as often as we want because that is not consistent with the mobile app usage patters. Instead, we rely on Websocket updates (push) from the backend, once the initial inventory and startup are complete. The REST interface is used only as a backup if Websocket fails, until WS is re-established. REST calls and need to be rate limited.
- Rate limiting is especially important for the import_energy_history service, which brings historical data into the integration. Since energy data is hourly and we need to get a year's worth of data for seasonal variations, we have to very carefully pace the import so as not to overwhelm the backend. A user with 6 heaters will need to pull 51,840 energy data points - if more than a few users did import simultaneously, we would crash the backend. Current limit is 2 queries per second. Fortunately, import should only need to happen once per user.

## End User Description
The end user of this integration is a non-technical person who has a home automation system and heaters at home that use TermoWeb. They are not interested in the implementation details or architecture. They want to be able to monitor and control their heaters through home assistant, without too much hassle. Documentation should be user friendly, detailed, step-by-step, action focused and easy to understand in plain English (many users of this system will only speak English as a second language). 

## Docs
- termoweb_openapi.yaml, termoweb_api.md - Document the API for the backend serving the TermoWeb mobile app.
- ducaheat_openapi.yaml, ducaheat_api.md - Document the API for the backend serving the Ducaheat mobile app.
- architecture.md - Documents the architecture of this integration and the Python class hierarchy.

## Development Guidelines
- Use Python version specified in pyproject.toml and type hints for all new code.
- Add one-line docstring descriptions to every function.
- For each commit or task, make the absolute MINIMAL SURGICAL changes and only directly related to the task at hand.
- Don't Repeat Yourself (DRY) when coding.
- Use defensive programming - anticipate errors, invalid user input, protocol failures, connection problems. Handle and report errors.
- Log entry and exit of major functions or actions as INFO, log protocol calls and responses as DEBUG, log errors as ERROR. 
- Lint and reformat with ruff before committing.
- Run tests with `timeout 30s pytest --cov=custom_components.termoweb --cov-report=term-missing` (or the equivalent cross-platform command) and make sure they pass before committing, with 100% coverage, so the suite self-terminates after 30 seconds.
- When documenting work after a timed run, capture the partial log and note that a timeout counts as a test failure requiring investigation; developers may run focused, no-coverage subsets during debugging before rerunning the full timed command.
- pytest with coverage take no more than 30 seconds to complete for the entire codebase. If tests take longer, assume there is an async wait problem and interrupt. Do not wait longer than 30 seconds. 
- All tests must be meaningful. No cheating, no softball easy-to-pass tests. Test actual edge cases, exception handling, incorrect inputs.
- Focus testing on interfaces between components, and error handling.
- New features must be documented.
- Do not make changes to parts of the code that are unrelated to your current task.


## Pull Request Expectations
- Keep every PR focused on a single feature or test with minimal code changes as needed.
- Provide a brief summary of your changes and how they were tested.
- Include any relevant documentation updates when behavior changes.
