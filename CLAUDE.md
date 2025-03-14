# Folding Project Reference

## Core Commands
- Run tests: `python -m pytest tests/`
- Run single test: `python -m pytest tests/test_file.py::test_function`
- Install deps: `poetry install`
- Lint: `black folding/`
- Type check: `mypy folding/`

## Code Style
- PEP 8 with Black formatting (79 character line limit)
- 4 spaces indentation
- Import order: stdlib → third party → local
- Use f-strings for formatting
- Atomic function design with good error handling

## Naming Conventions
- Classes: CapWords (PascalCase)
- Functions/Variables: snake_case
- Constants: UPPER_SNAKE_CASE
- Private methods/variables: _single_leading_underscore
- "Private" class attributes: __double_leading_underscore

## Git Workflow
- Feature branches from staging: `feature/<ticket>/<description>`
- Main branches: main (protected), staging (development)
- Commit message format: imperative mood, < 50 chars summary
- PR target: staging branch