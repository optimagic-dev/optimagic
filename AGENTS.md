## Environment

- Use Pixi for Python environments and development tasks
- Tasks and environments are defined in `pyproject.toml`
- Run `pixi task list` to discover available commands

## Git

- If pre-commit checks do not run when committing, run `prek install`

## Tests

- Run affected tests while iterating (`pixi run tests <test-path>`)
- Use `pixi run tests-fast` for broader checks (it excludes slow and JAX tests)
- For changes to shared code used across multiple modules, run the full suite
  (`pixi run tests`)

## Task-specific guidance

Consult the relevant sections of the documentation for your task:

- Contribution workflow: [contributor guide](docs/source/development/how_to_contribute.md).
- Code changes: [style guide](docs/source/development/styleguide.md).
- Typing and public interfaces: [EP-02](docs/source/development/ep-02-typing.md).
- GitHub activity: read the contributor guide's AI-use section and\
  follow its guidance before publishing.
