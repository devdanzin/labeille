# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
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
