# Contributing to labeille

Thanks for your interest in contributing! This guide covers everything you need to
get started.

## Reporting Bugs

### Bugs in labeille itself

Open an issue at [github.com/devdanzin/labeille/issues](https://github.com/devdanzin/labeille/issues)
with:

- What you were doing (command, package, configuration)
- What you expected to happen
- What actually happened (include tracebacks and logs)
- Your Python version and OS

### JIT bugs found by labeille

If labeille detects a crash in a JIT-enabled CPython build, that's a CPython bug!
To report it:

1. Verify the crash **does not** occur without the JIT (run with `PYTHON_JIT=0`).
2. Try to create a minimal reproducer if possible.
3. Open an issue on the [CPython GitHub](https://github.com/python/cpython/issues) with:
   - The crash signature and signal (e.g. SIGSEGV, SIGABRT)
   - The Python version and build configuration
   - The package and test that triggered the crash
   - Steps to reproduce

## Development Setup

```bash
# Clone the repository
git clone https://github.com/devdanzin/labeille.git
cd labeille

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with dev dependencies
pip install -e '.[dev]'
```

## Quality Checks

Run all of these before submitting a pull request:

```bash
# Format code
ruff format .

# Lint
ruff check .

# Type check
mypy

# Run tests
python -m unittest discover tests
```

All checks must pass. CI will run them automatically on pull requests.

## Pull Request Process

1. **Fork** the repository and create a branch from `main`.
2. **Make your changes.** Keep commits focused — one logical change per commit.
3. **Run quality checks** (see above). All must pass.
4. **Update documentation** if needed:
   - Add a line to `CHANGELOG.md` under `[Unreleased]`.
   - Update `CREDITS.md` if you're a new contributor.
5. **Submit a pull request** with a clear description of what and why.

## Commit Messages

We encourage (but don't require) [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add timeout support for test runner
fix: handle missing repo URL in PyPI metadata
docs: update quick start instructions
refactor: extract crash signature logic into separate module
```

The most important thing is that the message clearly describes what changed and why.

## Code Style

- **Formatter:** ruff format (line length 99)
- **Linter:** ruff check
- **Type checking:** mypy in strict mode
- All public functions need type annotations.
- Keep modules focused and imports clean.

## Adding Packages to the Registry

To add a new package to the test registry:

1. Create a YAML file in `registry/packages/{package_name}.yaml` following the
   schema in an existing package file.
2. Add an entry to `registry/index.yaml`.
3. Test that the package installs and its test suite runs locally before submitting.

## Enriching Registry Packages

After resolving a package with `labeille resolve`, the YAML file needs manual
enrichment to configure install and test commands correctly. Here are best practices
learned from enriching the first 50 packages:

1. **Cross-check test imports against installed deps.** After determining the install
   command, scan test files (`conftest.py`, early test files) for third-party imports
   that aren't covered by the install command. Common missed deps: `trustme`,
   `uvicorn`, `trio`, `tomli_w`, `appdirs`, `wcag_contrast_ratio`, `installer`,
   `pytest-cov`, `pytest-timeout`.

2. **Check for pytest plugin references.** Look at `pyproject.toml`
   `[tool.pytest.ini_options]` addopts and `tox.ini` `[pytest]` addopts. If addopts
   references `--cov`, `--timeout`, or other plugin-specific flags, either install
   the corresponding plugin or add `-o "addopts="` to `test_command` to override
   the config.

3. **Handle setuptools-scm and version-from-git-tags tools.** Set `clone_depth: 50`
   (or higher) so that git tags are available for version computation. Shallow clones
   lose tags and version detection fails.

4. **Avoid problematic transitive dependencies.** When test extras pull in heavy or
   problematic transitive dependencies (numpy, rpds-py, pydantic-core), use a minimal
   install approach: `pip install -e . && pip install pytest <specific-needed-test-deps>`
   instead of `pip install -e ".[test]"`.

5. **Skip packages with unreleased dependency pins.** For packages whose main branch
   pins unreleased dependency versions (like pydantic pinning unreleased pydantic-core),
   set `skip: true` with an explanatory `skip_reason`.

6. **Remove `-x` / `--exitfirst` from test_command.** A single unrelated failure
   shouldn't prevent us from seeing JIT crashes in later tests.

7. **Disable pytest-xdist.** Always set `uses_xdist: true` and add `-p no:xdist` to
   the test command if the project uses pytest-xdist, to ensure crashes propagate
   directly instead of being masked by worker processes.

## AI-Assisted Development

Contributors are welcome to use AI tools (Claude, Copilot, etc.) to help with their
contributions. However, you are responsible for:

- Understanding the code you submit
- Ensuring it passes all quality checks
- Reviewing AI-generated code for correctness and security
- Writing meaningful commit messages and PR descriptions

## Questions?

Open an issue or start a discussion. We're happy to help!
