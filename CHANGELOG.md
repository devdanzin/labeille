# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
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

### Enhanced
- Rich end-of-run summary with per-package table, timing stats, and crash details.
- Quiet mode shows only crash information; default mode hides passing packages.
- Post-install import validation catches broken installs before running tests.
- Add `--work-dir`, `--repos-dir`, and `--venvs-dir` options to `run` for persistent
  clone/venv directories that survive across runs.
- Reuse existing repo clones (pull instead of re-clone) and venvs (skip create+install).
- Log repo and venv paths in default output for each package.
- Verbose mode (`-v`) now shows test subprocess stdout/stderr, resolved commands,
  install output, installed dependency list, git operations, and per-phase timing.
