# CLAUDE.md — labeille development guide

## Project overview
labeille hunts CPython JIT bugs by running real-world PyPI package test suites against JIT-enabled builds.
Companion to [lafleur](https://github.com/devdanzin/lafleur) (evolutionary JIT fuzzer).

## Security note
labeille installs and runs arbitrary third-party code from PyPI. Never run batch operations on a machine with sensitive credentials or data. Use containers, VMs, or disposable cloud instances for batch runs. See README.md for detailed guidance.

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
- `cli.py` — Click CLI entry point with `resolve`, `run`, `scan-deps`, `registry`, and `analyze` subcommands
- `resolve.py` — PyPI metadata fetching, repo URL extraction, registry building
- `runner.py` — Clone repos, create venvs, run test suites, detect crashes
- `crash.py` — Crash detection from exit codes and stderr patterns
- `classifier.py` — Pure Python vs C extension classification from wheel tags
- `registry.py` — YAML registry I/O (index + per-package configs)
- `registry_cli.py` — Batch registry management CLI (add/remove/rename/set fields, validate, migrate)
- `migrations.py` — Registry migration framework (named transformations with logging and dry-run)
- `registry_ops.py` — Batch operations with filtering, atomic writes, dry-run previews
- `analyze.py` — Data loading and analysis (run data, registry stats, comparison, flaky detection)
- `analyze_cli.py` — Analysis CLI (registry, run, compare, history, package subcommands)
- `formatting.py` — Shared text formatting (tables, histograms, sparklines, durations)
- `summary.py` — Run summary formatting (uses formatting.py)
- `scan_deps.py` — AST-based test dependency scanner
- `import_map.py` — Import name to pip package mapping (100+ entries)
- `yaml_lines.py` — Line-level YAML manipulation preserving formatting
- `logging.py` — Structured logging setup

## Testing notes
- Integration tests mock `fetch_pypi_metadata` to avoid network calls
- Runner tests mock `subprocess.run` and `clone_repo` extensively
- 546 tests total across 15 test files

## Enriching packages

Registry lives in `registry/` — `index.yaml` (package list) + `packages/<name>.yaml` (per-package config).
Update index after editing package files: `load_index()` → `update_index_from_packages()` → `save_index()`.

### Process
1. Clone repo, examine pyproject.toml/setup.cfg/tox.ini/requirements*.txt for test deps
2. Determine install_command, test_command, test_framework, uses_xdist, timeout
3. Write the package YAML, set `enriched: true`

### Enrichment rules (learned from first 50 packages)
1. Cross-check test imports against installed deps — scan conftest.py and first test files for non-stdlib imports; common missed deps: trustme, uvicorn, trio, tomli_w, appdirs, wcag_contrast_ratio, installer, setuptools, flask
2. Check pytest config (pyproject.toml `[tool.pytest.ini_options]`, tox.ini `[pytest]`, pytest.ini) for plugin flags — if addopts uses `--cov`/`--timeout`/etc, install the plugin (pytest-cov, pytest-timeout)
3. Check filterwarnings for module references (e.g. `coverage.exceptions.CoverageWarning`) — pytest tries to import the module, so it must be installed
4. For setuptools-scm packages, add `git fetch --tags --depth 1` to install_command — shallow clones lose tags and version detection fails
5. When `[test]` extras pull in heavy/problematic transitive deps (numpy, rpds-py, pydantic-core), install deps manually: `pip install -e . && pip install pytest <specific-deps>` instead of `pip install -e ".[test]"`
6. For jsonschema dependency (pulls rpds-py via PyO3): pre-install `jsonschema<4.18` which uses pyrsistent instead
7. For packages whose main branch pins unreleased dependency versions, set `skip: true` with skip_reason
8. Never use `-x`/`--exitfirst` — a single unrelated failure shouldn't hide JIT crashes in later tests
9. Always disable xdist for JIT testing (`-p no:xdist`) — parallel workers mask JIT-specific crashes
10. For packages with src/ layout (e.g. pytz), verify install path — may need `pip install src/` not `pip install -e src/`

### Common 3.15 alpha blockers
- **PyO3/maturin** (rpds-py, pydantic-core, orjson): won't build until PyO3 supports 3.15
- **meson-python** (numpy, pandas): need `pip install meson-python meson cython ninja` before `--no-build-isolation`
- **moto[server]**: requires pydantic → pydantic-core (PyO3), blocks aiobotocore testing

## Performance and resource usage

### Parallel execution
- `--workers N` runs N packages in parallel (default: 1, sequential)
- Each worker spawns its own subprocesses (git, pip, pytest) — threads just wait on child processes, so the GIL is not a bottleneck
- Memory scales linearly with worker count; each test process uses the full interpreter + test suite + any ASAN overhead

### ASAN and resource pressure
- The dev .venv uses an ASAN-enabled build: ~2-3x memory per process
- For parallel runs with ASAN, --workers 2-3 is the practical limit
- For large batch runs (50+ packages), consider a non-ASAN JIT build: most JIT crashes (segfaults, aborts, assertion failures) reproduce identically without ASAN, and you can run --workers 4-8 comfortably
- Specific crashes can be reproduced under ASAN afterward for detailed memory error analysis

### Orphaned process cleanup
- labeille kills the entire process group on timeout (`os.killpg`), which catches most grandchild processes
- However, if you interrupt labeille itself (Ctrl+C), subprocesses may still survive — check with `ps aux | grep pytest` and clean up as needed

## Registry batch operations

- Always dry-run first (omit --apply), review the preview, then re-run with --apply.
- Use --lenient when resuming an interrupted operation or when you expect some files to already have the field.
- Use --after to control field placement in the YAML for readability.
- Run `labeille registry validate` after batch edits to catch issues.

## Registry migrations

- `labeille registry migrate --list` shows available migrations and their applied status.
- `labeille registry migrate <name>` previews changes (dry-run by default).
- `labeille registry migrate <name> --apply` applies changes and logs to `migrations.log`.
- Each migration runs once — re-application is blocked with the original date shown.
- Add new migrations by decorating a function with `@register_migration(name, description)` in `migrations.py`.

## Quick investigation overrides

These CLI options let you experiment without touching registry YAMLs:
- `--extra-deps "pkg1,pkg2"` — inject deps into every venv
- `--test-command-override "..."` — replace all test commands
- `--test-command-suffix "..."` — append flags to test commands
- `--repo-override "pkg=url"` — use a fork's repo (repeatable)
- `--packages=pkg@revision` — test at a specific git revision
- `--no-shallow` — full clone (needed for old revisions)

## Investigating a crash at a specific revision

If a crash was found at commit abc123:
1. Reproduce: `labeille run --packages=mypkg@abc123 --no-shallow ...`
2. Check if HEAD still crashes: `labeille run --packages=mypkg ...`
3. If HEAD doesn't crash, the package fixed the issue.
4. If HEAD still crashes, the issue persists — file upstream or investigate the JIT side.

## Workflow
- Use `/task-workflow <description>` for the full issue → branch → code → test → commit → PR → merge cycle
