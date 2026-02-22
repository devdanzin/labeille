# CLAUDE.md — labeille development guide

## Project overview
labeille hunts CPython JIT bugs by running real-world PyPI package test suites against JIT-enabled builds.
Companion to [lafleur](https://github.com/devdanzin/lafleur) (evolutionary JIT fuzzer).

## Environment
- The `.venv` was created with a JIT-enabled CPython build at `~/projects/jit_cpython/python`
- This build has AddressSanitizer (ASAN) enabled — always set `ASAN_OPTIONS=detect_leaks=0`
- The JIT has known bugs that crash dev tools (e.g. mypy) — always set `PYTHON_JIT=0` for dev commands
- Standard prefix for all dev commands: `ASAN_OPTIONS=detect_leaks=0 PYTHON_JIT=0`

## Dev commands
```bash
# Lint and format
ASAN_OPTIONS=detect_leaks=0 PYTHON_JIT=0 .venv/bin/ruff format .
ASAN_OPTIONS=detect_leaks=0 PYTHON_JIT=0 .venv/bin/ruff check .

# Type checking (strict mode, configured in pyproject.toml)
ASAN_OPTIONS=detect_leaks=0 PYTHON_JIT=0 .venv/bin/mypy

# Tests (unittest, NOT pytest)
ASAN_OPTIONS=detect_leaks=0 PYTHON_JIT=0 .venv/bin/python -m unittest discover tests -v
```

## Code style
- Python >=3.13, developed on 3.15
- src layout: `src/labeille/`
- ruff line-length: 99
- mypy strict mode
- Tests use `unittest` with `unittest.mock` — never pytest
- Click CLI framework — Click 8.3+ (no `mix_stderr` in CliRunner)

## Architecture
- `cli.py` — Click CLI entry point with `resolve` and `run` subcommands
- `resolve.py` — PyPI metadata fetching, repo URL extraction, registry building
- `runner.py` — Clone repos, create venvs, run test suites, detect crashes
- `crash.py` — Crash detection from exit codes and stderr patterns
- `classifier.py` — Pure Python vs C extension classification from wheel tags
- `registry.py` — YAML registry I/O (index + per-package configs)
- `logging.py` — Structured logging setup

## Testing notes
- Integration tests mock `fetch_pypi_metadata` to avoid network calls
- Runner tests mock `subprocess.run` and `clone_repo` extensively
- 127 tests total across 6 test files

## Workflow
- Use `/task-workflow <description>` for the full issue → branch → code → test → commit → PR → merge cycle
