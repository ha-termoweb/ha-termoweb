# AGENTS.md

## Development Guidelines
- Use Python version specified in pyproject.toml and type hints for all new code.
- For each commit or task, make the absolute MINIMAL SURGICAL changes and only directly related to the task at hand.
- Don't Repeat Yourself (DRY) when coding.
- Use best practices and design patterns.
- Use defensive programming - anticipate errors, invalid user input, protocol failures, connection problems. Handle and report errors.
- Log entry and exit of major functions or actions as INFO, log protocol calls and responses as DEBUG, log errors as ERROR. 
- Lint and reformat with ruff before committing.
- Run tests with `pytest` and make sure they pass before committing.
- Do not make changes to parts of the code that are unrelated to your current task

## Pull Request Expectations
- Keep every PR focused on a single feature or test with minimal code changes as needed.
- Provide a brief summary of your changes and how they were tested.
- Include any relevant documentation updates when behavior changes.
