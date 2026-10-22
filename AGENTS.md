## Environment

- Use Pixi for Python environments and development tasks
- Tasks and environments are defined in `pyproject.toml`
- Run `pixi task list` to discover available commands

## Git

- Run `prek install` once to enable pre-commit checks

## Tests

- Run affected tests while iterating (`pixi run tests <test-path>`)
- Use `pixi run tests-fast` for broader checks (it excludes slow and JAX tests)
- For cross-cutting changes, run the full suite (`pixi run tests`) in a Pixi
  environment with the required dependencies

## Task-specific guidance

Consult the relevant sections of the documentation for your task:

- Contribution workflow: [contributor guide](docs/source/development/how_to_contribute.md).
- Code changes: [style guide](docs/source/development/styleguide.md).
- Typing and public interfaces: [EP-02](docs/source/development/ep-02-typing.md).
- GitHub activity: read the contributor guide's AI-use section and\
  follow its approval requirements before publishing.

## Generated code

Changes to optimizer definitions require regenerating `src/optimagic/algorithms.py`\
through the existing hook.
