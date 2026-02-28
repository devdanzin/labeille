# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- `bench` subpackage with `system.py` module for capturing system profiles (CPU, RAM, OS, disk) and Python interpreter profiles (version, JIT/GIL state, build flags) for benchmark reproducibility.
- `SystemProfile`, `PythonProfile`, `StabilityCheck`, and `SystemSnapshot` dataclasses with JSON serialization and terminal display formatting.
- `check_stability()` pre-benchmark validation (load average, available RAM).
- `labeille bisect` command to binary-search a package's git history and find the first commit that introduced a crash.
- `bisect.py` module with `BisectConfig`, `BisectStep`, `BisectResult` dataclasses and the `run_bisect` algorithm with skip-neighbor handling for unbuildable commits.
- Commit-aware run comparison: `analyze compare` and `analyze run` show git commit changes alongside status changes with heuristic annotations (e.g. "unchanged — likely a CPython/JIT regression").
- `PackageComparison` dataclass with `commit_changed`/`commit_unchanged` properties for per-package comparison data.
- New crash summary statistics in compare output showing repo unchanged/changed/unknown counts.

### Enhanced
- Switched build backend from setuptools to hatchling for better src layout support and lighter build dependencies.
- Added minimum version pins to runtime dependencies (click>=8.0, pyyaml>=6.0, requests>=2.28).
- Added `py.typed` marker for PEP 561 type checker support.
- Added sdist/wheel exclusion rules to keep distribution lean (no tests, registry, results, or docs).
- Added Installation section to README with pipx, pip, and from-source instructions.
- Added `Environment :: Console` and `Topic :: Software Development :: Quality Assurance` classifiers.
- Renamed `Issues` URL key to `Bug Tracker` in project metadata for PyPI display consistency.

### Fixed
- `run_meta.json` now stores actual CLI argument strings (`sys.argv[1:]`) instead of parameter names, making runs reproducible from metadata.
- `build_reproduce_command` uses `export PATH` for venv activation instead of fragile `.venv/bin/` prefix string replacement.
- Deduplicated `_signal_name` (3 copies → `format_signal_name` in `formatting.py`).
- Deduplicated `_result_detail` (3 copies → public `result_detail` in `analyze.py`).
- Made `_extract_minor_version` public as `extract_minor_version` in `analyze.py`.
- Removed redundant zero-check in `compare_runs` duration percentage calculation.
- Fixed timeout documentation (300s → 600s) in `doc/enrichment.md`.
- `_quote_yaml_scalar` now quotes all numeric strings (integers, scientific notation, octal-like), tilde, and additional YAML special characters.
- `find_field_extent` no longer consumes trailing blank lines after scalar fields, fixing `insert_field_after` placement near blank lines.
- Rewrote `batch_set_field` to use line-level manipulation instead of PyYAML round-trip, preserving YAML formatting.
- Added `set_field_value` to `yaml_lines.py` for in-place field value replacement.
- `format_yaml_value` and `parse_default_value` now handle `None`/`null` values.
- `_is_version_specific_skip` now uses word-boundary regex patterns to prevent false positives (e.g. "trust" no longer matches the "rust" pattern).
- `scan-deps` now warns about namespace packages (`google`, `azure`, `zope`, etc.) where pip resolution is uncertain, and tries full import paths before falling back to top-level modules.
- `IndexEntry` now tracks `skip_versions_keys` for fast version-skip filtering without loading full YAML files.
- `filter_packages` uses index-level `skip_versions_keys` to skip packages before loading YAML.
- `_dict_to_package` coerces `notes: null` to empty string for type safety.
- `_dict_to_package` logs unknown YAML keys at debug level to surface typos.
- `validate_registry` checks `uses_xdist`/`-p no:xdist` consistency in both directions.
- `check_jit_enabled` now uses explicit `sys.flags.jit` check instead of nonexistent `sys._jit`, with exact stdout comparison.
- `_parse_install_packages` now handles `python -m pip install`, `python3 -m pip install`, and path-qualified pip invocations.
- `_package_to_dict` accepts `omit_defaults` parameter to exclude default-valued fields from output.
- `run_test_command` and `install_package` now kill the entire process tree on timeout via `os.killpg`, preventing orphaned grandchild processes from accumulating during batch runs.
- `RunData.result_for()` now uses O(1) dict lookup instead of O(N) linear scan, with lazily-built `_results_by_pkg` cache.
- `compare_runs` and `_compute_status_changes` use `result_for()` instead of building ad-hoc dicts.
- Subprocess helpers (`build_env`, `check_jit_enabled`, `create_venv`, `validate_target_python`) now strip `PYTHONHOME`/`PYTHONPATH` via `_clean_env()` to prevent environment pollution.
- CLI warns when only one of `--repos-dir`/`--venvs-dir` is set, since the other will use a temporary directory.
- `update_index_from_packages` accepts optional `modified_packages` set to avoid O(N) disk reads when only a few packages changed.
- `_PLATFORM_INDICATORS` now detects bare `linux_x86_64`/`linux_aarch64` wheels in addition to `manylinux`/`musllinux`.
- `fetch_pypi_metadata` and `resolve_package` accept an optional `requests.Session` for connection reuse; `resolve_all` uses shared/thread-local sessions.
- `_is_import_error_handler` no longer treats `except Exception` as an import error handler, reducing false conditional import flags.
- `_parse_install_packages` uses a regex instead of chained `.split()` to handle all PEP 440 specifiers (`~=`, `!=`, `;` markers).
- `pull_repo` uses `git fetch` + `reset --hard FETCH_HEAD` + `clean -fdx` instead of `git pull --ff-only`, handling dirty working trees left by test suites.

### Added
- `--extra-deps` option to inject additional packages into every venv after the package's own dependencies.
- `--test-command-override` option to replace the test command for all packages in a run.
- `--test-command-suffix` option to append flags to each package's existing test command.
- `--repo-override PKG=URL` option (repeatable) to test forks or PR branches without modifying registry.
- `--clone-depth` and `--no-shallow` CLI options to override per-package clone depth; `--clone-depth=0` or `--no-shallow` for full clones.
- Per-package git revision support via `--packages=pkg@revision` syntax; accepts commit hashes, branches, tags, or relative refs like `HEAD~10`.
- `checkout_revision` helper for checking out specific git refs after cloning.
- `parse_package_specs` function for parsing `name@revision` package spec syntax.
- `requested_revision` field in `PackageResult` to distinguish explicitly requested revisions from HEAD.
- 350 enriched package configurations with full test commands, install commands, and metadata.
- Applied `skip-to-skip-versions` migration on 36 packages (PyO3, maturin, Cython, JIT crashes).
- Config fixes for python-dateutil, pyyaml, msgpack, hatchling, openai, numpy, pytz, sqlalchemy, and 3 archived google packages.
- `registry/migrations.log` tracking applied migration history.
- `labeille registry migrate` command with a migration framework for registry schema transformations.
- `skip-to-skip-versions` migration to convert 3.15-specific `skip:true` entries to `skip_versions["3.15"]`.
- Migration log (`migrations.log`) to track applied migrations and prevent re-application.
- Dry-run support for migrations with preview of affected packages.
- `labeille scan-deps` command for static test dependency discovery via AST-based import analysis.
- `import_map.py` module with 100+ import-name-to-pip-package mappings for common mismatches (PIL->Pillow, yaml->PyYAML, etc.).
- Three output formats for scan-deps: human-readable (default), JSON, and pip (for direct shell use).
- Automatic test directory detection and local module filtering in scan-deps.
- Comparison against existing install_command to identify missing deps.
- Registry cross-referencing for import_name resolution in scan-deps.
- `labeille analyze` CLI subgroup with five subcommands: `registry`, `run`, `compare`, `history`, `package`.
- `formatting.py` shared formatting module (tables, histograms, sparklines, duration, status icons).
- `analyze.py` data loading and analysis module (run data, registry stats, comparison, flaky detection).
- `labeille registry` CLI subgroup for batch registry management (add-field, remove-field, rename-field, set-field, validate, add-index-field, remove-index-field).
- Line-level YAML manipulation (`yaml_lines.py`) preserving exact formatting.
- Batch operations module (`registry_ops.py`) with filtering, atomic writes, and dry-run previews.
- Registry validation against the PackageEntry schema with `labeille registry validate`.
- `skip_versions` registry field for per-Python-version skip reasons (e.g. `3.15: "PyO3 not supported"`).
- `--force-run` flag to override `skip` and `skip_versions` for debugging.
- `--workers N` option for parallel package testing in `labeille run`.
- `--workers N` option for parallel PyPI resolution in `labeille resolve`.
- Cancellation support for `--stop-after-crash` in parallel mode.
- `clone_depth` registry field for packages needing git tags (e.g. setuptools-scm).
- `import_name` registry field for packages whose import name differs from PyPI name.
- `summary.py` module for formatting run results.
- Enrichment best practices documented in CONTRIBUTING.md.
- `--refresh-venvs` flag to delete and recreate existing venvs, ensuring updated install commands take effect.
- Initial project scaffolding.
- CLI skeleton with `resolve` and `run` subcommands.
- Registry schema and data structures.

### Documentation
- Added security warnings to README.md, runner.py module docstring, and CLAUDE.md.
- Added Gemini acknowledgment to CREDITS.md.
- Comprehensive enrichment guide with manual workflow, troubleshooting, and Claude Code prompts (doc/enrichment.md).
- Updated README with enrichment overview and link to guide.
- Parallel execution guidance, resource considerations, and ASAN vs non-ASAN trade-offs.

### Enhanced
- Refactored `summary.py` to use shared formatters from `formatting.py`.
- Improved repo URL resolution with secondary keys (bug tracker, issues, changelog) and legacy field fallbacks (home_page, download_url).
- Run summary shows version-skipped count separately when skip_versions is active.
- Progress reporting adapted for parallel execution with per-completion status lines.
- Rich end-of-run summary with per-package table, timing stats, and crash details.
- Quiet mode shows only crash information; default mode hides passing packages.
- Post-install import validation catches broken installs before running tests.
- Add `--work-dir`, `--repos-dir`, and `--venvs-dir` options to `run` for persistent
  clone/venv directories that survive across runs.
- Reuse existing repo clones (pull instead of re-clone) and venvs (skip create+install).
- Log repo and venv paths in default output for each package.
- Verbose mode (`-v`) now shows test subprocess stdout/stderr, resolved commands,
  install output, installed dependency list, git operations, and per-phase timing.
