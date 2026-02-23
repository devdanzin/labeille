# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- `--refresh-venvs` flag to delete and recreate existing venvs, ensuring updated install commands take effect.
- Initial project scaffolding.
- CLI skeleton with `resolve` and `run` subcommands.
- Registry schema and data structures.

### Enhanced
- Add `--work-dir`, `--repos-dir`, and `--venvs-dir` options to `run` for persistent
  clone/venv directories that survive across runs.
- Reuse existing repo clones (pull instead of re-clone) and venvs (skip create+install).
- Log repo and venv paths in default output for each package.
- Verbose mode (`-v`) now shows test subprocess stdout/stderr, resolved commands,
  install output, installed dependency list, git operations, and per-phase timing.
