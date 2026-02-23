# Enriching the Package Registry

## What is enrichment?

When you run `labeille resolve`, it creates skeleton YAML files for each package —
a repo URL, an extension type classification, and a bunch of empty fields. These
skeletons are enough to know *what* to test, but not *how* to test it. That's where
enrichment comes in.

Enrichment is the process of filling in the installation and test instructions for
each package: what commands install it, what dependencies its test suite needs, how
to invoke the tests, and any quirks to work around. Without enrichment, `labeille run`
has nothing to run.

You can enrich packages manually, with Claude Code, or with another AI coding agent.
The process is the same either way — read the project's configuration files, figure
out the right commands, write them into the YAML, and verify they work.

**Enrichment is iterative.** You rarely get it right on the first try. Test suites
have undocumented dependencies, pytest configurations that conflict with how labeille
runs tests, and build systems that need special handling. The workflow is:

1. Read the project's config files and fill in the YAML
2. Run the package through `labeille run`
3. Diagnose any failures from the output
4. Fix the YAML
5. Re-run until the test harness works

"Working" means the test suite runs — exit code 0 (all tests pass) or exit code 1
(some tests fail). Both are fine. What matters is that the harness is correctly
configured. Exit codes 2, 3, and 4 mean something is wrong with the configuration.


## Registry file format

Each package has a YAML file at `registry/packages/{name}.yaml`. Here's the full
schema with explanations.

### Identity fields

| Field | Type | Description |
|-------|------|-------------|
| `package` | string | PyPI package name (must match the filename) |
| `repo` | string | Source repository URL (set by `labeille resolve`) |
| `pypi_url` | string | Link to the PyPI page |
| `import_name` | string or null | Python import name when it differs from the package name. For example, `Pillow` imports as `PIL`, `python-dateutil` imports as `dateutil`. Used for post-install import validation. Null means the import name is derived from the package name automatically. |

### Classification

| Field | Type | Description |
|-------|------|-------------|
| `extension_type` | string | `pure` (pure Python), `extensions` (has C/Rust extensions), or `unknown`. Set by `labeille resolve` from wheel tag analysis. |
| `clone_depth` | int or null | Git clone depth. Null means shallow (depth=1). Set to 50 or higher for packages that use setuptools-scm or other version-from-git-tags tools, since shallow clones lose the tags needed for version detection. |

### Installation

| Field | Type | Description |
|-------|------|-------------|
| `install_method` | string | `pip`, `pip-extras`, or `custom`. Informational — the actual command is in `install_command`. |
| `install_command` | string | The exact shell command to install the package with its test dependencies. Runs inside an activated venv with the target Python. |

### Testing

| Field | Type | Description |
|-------|------|-------------|
| `test_command` | string | The exact shell command to run the test suite. `python` in this command gets replaced with the venv's Python automatically. Runs with `shell=True`, so quotes and pipes work. |
| `test_framework` | string | `pytest`, `unittest`, or `custom`. Informational. |
| `uses_xdist` | bool | Whether the project uses pytest-xdist for parallel testing. If true, the test command should include `-p no:xdist` to disable it — parallel workers mask JIT-specific crashes. |
| `timeout` | int or null | Per-package timeout in seconds. Null means use the runner's default (300s). Set higher for packages with large test suites. |

### Control

| Field | Type | Description |
|-------|------|-------------|
| `python_versions` | list | Python versions this package supports (informational). |
| `skip` | bool | Skip this package entirely during `labeille run`. |
| `skip_reason` | string or null | Why the package is skipped — build failures, missing 3.15 support, etc. |
| `notes` | string | Free-form notes about quirks, known issues, or enrichment decisions. |
| `enriched` | bool | Whether this file has been reviewed and filled in. Set to `true` once you've determined the install and test commands, even if some tests fail. |

### Complete example

Here's a well-enriched package — `click`, a pure Python CLI framework with a
straightforward test setup:

```yaml
package: click
repo: https://github.com/pallets/click/
pypi_url: https://pypi.org/project/click/
extension_type: pure
python_versions:
- '3.15'
- '3.14'
install_method: pip
install_command: pip install -e . && pip install pytest
test_command: python -m pytest tests/
test_framework: pytest
uses_xdist: false
timeout: null
skip: false
skip_reason: null
notes: Click CLI framework by Pallets. Uses flit_core build backend.
enriched: true
clone_depth: null
import_name: null
```

And here's one with more complexity — `setuptools-scm`, which needs git tags for
version detection and a pytest config override:

```yaml
package: setuptools-scm
repo: https://github.com/pypa/setuptools-scm/
pypi_url: https://pypi.org/project/setuptools-scm/
extension_type: pure
python_versions: []
install_method: pip
install_command: git fetch --tags --depth 1 && pip install -e . && pip install pytest setuptools tomli typing-extensions build
test_command: python -m pytest testing/ -o "strict_config=false"
test_framework: pytest
uses_xdist: false
timeout: 180
skip: false
skip_reason: null
notes: SCM-based version management. Needs git tags for version detection
  (git fetch --tags). Tests in testing/ directory.
enriched: true
clone_depth: 50
import_name: setuptools_scm
```


## Manual enrichment walkthrough

Let's walk through enriching a package from scratch using `packaging` as our
example — it's pure Python, popular, and has a clean test setup.

### Step 1: Examine the project

Start by looking at the project's configuration files to understand how it's
built and tested.

**Clone the repo and look at these files, in order:**

1. **`pyproject.toml`** — The most important file. Check:
   - `[project.optional-dependencies]` for test/dev dependency groups
   - `[tool.pytest.ini_options]` for test configuration, addopts, and plugin flags
   - `[build-system]` to understand the build backend
   - `[tool.setuptools-scm]` or similar — indicates version-from-git-tags

2. **`tox.ini` / `noxfile.py`** — The test automation config. Shows which
   dependencies are installed for testing and how tests are invoked.

3. **`CONTRIBUTING.md` / `README.md`** — Human-readable test instructions.
   Sometimes the only place unusual test requirements are documented.

4. **`requirements*.txt`** — Some projects keep test dependencies in
   `requirements-test.txt` or `requirements-dev.txt`.

5. **`conftest.py` and a few test files** — Check imports for dependencies that
   aren't listed anywhere else. This is the most common source of "mystery"
   `ModuleNotFoundError` failures.

For `packaging`, here's what we'd find:

- `pyproject.toml` has a `[project.optional-dependencies]` section with a `tests`
  group listing `pytest` and `pretend`
- The test directory is `tests/`
- No xdist, no special pytest plugins

### Step 2: Determine the install command

Follow this decision tree:

1. **Is there a `[test]` or `[testing]` extra?**
   Use `pip install -e ".[test]"` — but first check what it pulls in. If it
   includes heavy deps like numpy, rpds-py, or pydantic-core that won't build
   on 3.15, install deps manually instead.

2. **Is there a `[dev]` extra that includes test deps?**
   Use `pip install -e ".[dev]"` — same caveat about heavy transitive deps.

3. **Are test deps only in `tox.ini` or `requirements-test.txt`?**
   Use `pip install -e . && pip install -r requirements-test.txt`

4. **None of the above?**
   Use `pip install -e . && pip install pytest <deps-from-test-imports>`

For `packaging`, option 1 works cleanly:

```
pip install -e ".[tests]"
```

But we could also go explicit (and this is what the actual YAML uses, to avoid
any transitive dependency surprises):

```
pip install -e . && pip install pytest pretend tomli_w
```

**Common pitfalls to watch for:**

- **Transitive dependency hell on 3.15.** Test extras that pull in rpds-py (via
  jsonschema >= 4.18), pydantic-core, or numpy will fail because these packages
  use PyO3/Rust and don't support 3.15 yet. Fix: install deps manually, use
  `jsonschema<4.18` (which uses pyrsistent instead of rpds-py), or skip deps
  you don't need.

- **setuptools-scm packages.** If the project uses setuptools-scm (check
  `[tool.setuptools-scm]` in pyproject.toml), shallow clones lose the git tags
  needed for version detection. Fix: set `clone_depth: 50` and add
  `git fetch --tags --depth 1` to the install command.

- **Editable installs that break.** Some packages (especially monorepo
  subdirectories) don't support `pip install -e .`. Fix: use
  `pip install .` (non-editable) instead.

- **Packages needing build deps upfront.** Some Cython or meson-based packages
  need build tools installed first: `pip install setuptools cython && pip install --no-build-isolation -e .`

### Step 3: Determine the test command

Follow this decision tree:

1. **Does `pyproject.toml` have `[tool.pytest.ini_options]`?**
   It's a pytest project. Use `python -m pytest tests/` (or whatever test
   directory exists — check for `tests/`, `test/`, `testing/`, or test files
   in the package directory itself).

2. **Is there a `tests/` directory with `test_*.py` files importing unittest?**
   Use `python -m unittest discover tests`

3. **Something else?**
   Check `tox.ini` or `noxfile.py` for the actual test invocation.

**Important checks for the test command:**

- **xdist usage.** If `addopts` includes `-n auto`, `--numprocesses=auto`, or
  the project lists pytest-xdist as a dependency, set `uses_xdist: true` and add
  `-p no:xdist` to the test command. Parallel workers mask JIT crashes.

- **Plugin flags in addopts.** If addopts includes `--cov`, `--timeout`, or other
  plugin flags, you have two choices:
  - Install the corresponding plugin (pytest-cov, pytest-timeout, etc.)
  - Override addopts entirely: `-o "addopts="`

- **Never use `-x` or `--exitfirst`.** A single unrelated test failure shouldn't
  prevent us from seeing JIT crashes in later tests.

- **Check `filterwarnings`.** If pytest config has `filterwarnings` entries that
  reference specific modules (e.g. `coverage.exceptions.CoverageWarning`), pytest
  tries to import those modules. They must be installed or the warning filter
  entries will error out.

For `packaging`, the test command is simple:

```
python -m pytest tests/
```

No xdist, no special plugins, no addopts conflicts.

### Step 4: Test it locally

You can validate your enrichment manually:

```bash
# Clone and install
git clone --depth=1 https://github.com/pypa/packaging /tmp/test-packaging
cd /tmp/test-packaging
/path/to/jit-python -m venv /tmp/venv-packaging
source /tmp/venv-packaging/bin/activate
pip install -e . && pip install pytest pretend tomli_w

# Run tests with JIT enabled
PYTHON_JIT=1 PYTHONFAULTHANDLER=1 python -m pytest tests/
```

Or use labeille directly, which handles cloning, venv creation, and environment
setup for you:

```bash
labeille run --target-python /path/to/jit-python \
    --packages packaging \
    --repos-dir /tmp/repos --venvs-dir /tmp/venvs \
    -v
```

The `-v` flag shows verbose output including install logs and test output, which
is essential for diagnosing failures.

### Step 5: Update the YAML and iterate

If the test run fails, diagnose the error from the output:

| Exit code | Meaning | Typical cause |
|-----------|---------|---------------|
| 0 | All tests passed | You're done! |
| 1 | Some tests failed | Config is correct — tests just fail (expected on alpha Python) |
| 2 | Collection error | Missing test dependency (most common) |
| 3 | Internal pytest error | Strict config warnings, plugin conflicts |
| 4 | No tests collected | Wrong test path, unrecognized pytest arguments |
| 5 | No tests ran | Test discovery found nothing to run |
| 134 | SIGABRT | JIT crash — this is what we're looking for! |
| 139 | SIGSEGV | JIT crash — this is what we're looking for! |

For each failure type:

- **Exit 2 (ModuleNotFoundError):** Add the missing module to `install_command`
  and re-run with `--refresh-venvs`.
- **Exit 3 (internal error):** Usually a strict config issue. Try adding
  `-o "strict_config=false"` or `-o "addopts="` to the test command.
- **Exit 4 (no tests / bad args):** Check the test path. Also check if addopts
  includes flags that conflict with labeille's invocation.
- **Timeout:** Increase the `timeout` field, or test a subset of the test suite
  by specifying a subdirectory.

Set `enriched: true` once the test suite runs (exit 0 or 1). The package is
enriched even if some tests fail — what matters is that the harness works
correctly.

If you changed `install_command`, use `--refresh-venvs` to force venv recreation.
If you only changed `test_command`, you don't need `--refresh-venvs`.


## Common problems and solutions

A reference for frequent issues encountered during enrichment. These patterns
come from enriching ~350 packages.

### Missing test dependencies (exit code 2)

The most common failure. Tests import modules that aren't listed in the install
command.

| Missing module | What it is | Packages that need it |
|----------------|------------|----------------------|
| `hypothesis` | Property-based testing | yarl, mpmath, rfc3339-validator, coverage |
| `trustme` | Local TLS certificates | requests, httpx, uvicorn |
| `objgraph` | Object graph debugging | greenlet, multidict |
| `psutil` | Process utilities | multidict, ipykernel |
| `testpath` | Path testing utilities | jeepney |
| `aioresponses` | Async HTTP mocking | google-auth (aio tests) |
| `pytest-cov` | Coverage plugin | many packages reference `--cov` in addopts |
| `pytest-timeout` | Timeout plugin | packages with `--timeout` in addopts |
| `setuptools` | Provides `pkg_resources` | legacy packages that import `pkg_resources` |
| `build` | PEP 517 build frontend | poetry-core integration tests |
| `tomli_w` | TOML writing | packaging, build |
| `covdefaults` | Coverage defaults | yarl (referenced in .coveragerc) |

**Fix:** Add the module to `install_command` and re-run with `--refresh-venvs`.

### pytest addopts conflicts (exit code 3 or 4)

Many packages configure pytest addopts in `pyproject.toml` or `pytest.ini` with
flags that conflict with labeille's usage — usually xdist flags or plugin options.

| Package pattern | Problem | Fix |
|-----------------|---------|-----|
| `-n auto` or `--numprocesses=auto` in addopts | Requires pytest-xdist | Add `-p no:xdist -o "addopts="` |
| `--cov` in addopts | Requires pytest-cov | Install pytest-cov or `-o "addopts="` |
| `--timeout` in addopts | Requires pytest-timeout | Install pytest-timeout or `-o "addopts="` |
| `--strict-markers` in addopts | Errors on unknown markers | Keep it: `-o "addopts=--strict-markers"` |

The nuclear option is `-o "addopts="` which clears all configured addopts. This
fixes most pytest config conflicts but loses any intentional configuration. Use
it when you can't easily install all the required plugins.

### Strict config errors (exit code 3)

Some packages enable `strict_config = true` in their pytest config, which errors
on any unrecognized configuration option.

**Fix:** Add `-o "strict_config=false"` to the test command.

### Wrong rootdir detection

When a package repository has no pytest config file (`pyproject.toml`, `pytest.ini`,
etc.), pytest walks up the directory tree looking for one. If labeille's own
`pyproject.toml` is found first, pytest uses the wrong root directory.

**Fix:** Add `--rootdir .` to the test command.

### Wrong test paths (exit code 4)

The test directory might not be where you expect:

- `tests/` vs `test/` vs `testing/` (setuptools-scm uses `testing/`)
- Tests inside the package directory: `regex/tests/test_regex.py`
- Tests as a single file: `test_toml.py` in the project root

**Fix:** Check the actual directory structure and update the test command path.

### Ignoring problematic test files

Sometimes specific test files or directories import unavailable dependencies or
test features we don't need. Rather than installing everything, ignore them:

```yaml
test_command: python -m pytest tests/ --ignore=tests/transport/aio --ignore=tests/integration
```

Common cases:
- `--ignore=tests/integration` — integration tests that need running services
- `--ignore=tests/transport/aio` — async transport tests needing aiohttp extras
- `--ignore=tests/supervisors` — supervisor tests needing specific process setup

### Build failures on Python 3.15

Some packages can't be built on 3.15 at all:

| Blocker | Affected packages | Status |
|---------|-------------------|--------|
| PyO3/maturin (Rust) | rpds-py, pydantic-core, orjson | Waiting for PyO3 3.15 support |
| Changed C API | Some Cython packages | Check case-by-case |
| Unreleased dependency pins | Packages pinning pre-release versions | Skip with reason |

**Fix:** Set `skip: true` with a clear `skip_reason`.

### Timeout issues

Some test suites are very large or have tests that hang:

| Package | Typical runtime | Notes |
|---------|-----------------|-------|
| setuptools | ~28 minutes | Very large test suite |
| networkx | ~14 minutes | |
| pyparsing | ~9 minutes | |
| typer | ~9 minutes | |
| lz4 | ~8 minutes | |

**Fix:** Set `timeout` to an appropriate value (in seconds). For extremely large
suites, consider testing a subset: `python -m pytest tests/unit/` instead of
the full suite.

### YAML formatting tips

- **Keep commands on a single line.** Long lines are fine in YAML. Single-line
  commands are unambiguous and avoid multiline parsing confusion.
- **Verify YAML syntax after editing.** One broken YAML file crashes labeille
  for *all* packages (it loads all YAMLs at startup). Quick check:
  ```bash
  python -c "import yaml; yaml.safe_load(open('registry/packages/mypackage.yaml'))"
  ```
- **Watch nested quotes.** Values with inner quotes like
  `-o "addopts=" -o "markers=custom: my marker"` need careful handling. Test
  the YAML loads correctly after editing.


## Working with Claude Code

Claude Code is excellent at the mechanical parts of enrichment — reading project
configs, figuring out dependencies, iterating on failures. The human should
review the results and make judgment calls about skip decisions, unusual setups,
and whether a package is worth spending time on.

### Initial enrichment prompt

For enriching packages that have skeleton YAML files (from `labeille resolve`):

```
Enrich the labeille registry files for the following packages:
{package list or "all unenriched packages in registry/packages/"}

For each package:
1. Clone the repo (use the URL in the YAML file).
2. Read pyproject.toml, tox.ini, CONTRIBUTING.md, and test files to determine
   the install command, test command, test framework, and any special needs.
3. Pay close attention to:
   - All optional-dependency groups (test, testing, dev, all)
   - pytest addopts that reference plugins (--cov, --timeout, etc.)
   - filterwarnings that reference modules that must be installed
   - setuptools-scm or other version-from-git-tags tools (set clone_depth: 50)
   - Whether the project uses xdist (set uses_xdist: true, add -p no:xdist)
   - Never include -x or --exitfirst in the test command
4. Update the YAML file and set enriched: true.
5. If the package can't be tested (needs external services, unreleased deps,
   PyO3 on 3.15), set skip: true with a clear skip_reason.

Refer to doc/enrichment.md for the full field reference and common pitfalls.
```

### Iterative fix prompt

For fixing packages whose test runs failed:

```
Fix the labeille registry for these packages that failed their test runs:
{package list}

For each package, run:
  labeille run --target-python {path} --packages {name} \
      --repos-dir {repos-dir} --venvs-dir {venvs-dir} -v

Diagnose the failure from the verbose output:
- ModuleNotFoundError -> add the module to install_command
- pytest plugin/config error -> install the plugin or override addopts
- Build/install failure -> try different install approach
- Timeout -> increase the timeout field or test a subset

After fixing the YAML, re-run to verify. Use --refresh-venvs if you changed
the install_command (since the old venv has stale dependencies). Don't use
--refresh-venvs if you only changed the test_command.

Repeat until tests either run (pass or fail is fine -- we want the harness
working) or you determine the package can't be tested (set skip: true).

Refer to doc/enrichment.md for common problems and solutions.
```

### Full enrichment-and-fix prompt

For doing everything in one go — new packages, end to end:

```
Enrich and validate the labeille registry for these packages:
{package list}

Phase 1 -- Enrich: For each package, clone the repo and read the project
configuration to fill in the YAML fields. Set enriched: true. Refer to
doc/enrichment.md for field reference and enrichment rules.

Phase 2 -- Validate: Run each package through labeille:
  labeille run --target-python {path} --packages {name} \
      --repos-dir {repos-dir} --venvs-dir {venvs-dir} -v

If a package fails, diagnose the error, fix the YAML, and re-run. Use
--refresh-venvs when you change install_command. Iterate until the test
harness works or you determine it can't work (then set skip: true with
skip_reason).

Don't spend more than 3 attempts per package -- if it's still broken after
3 fixes, set skip: true with notes explaining what you tried.
```

In all prompts, substitute `{path}`, `{repos-dir}`, and `{venvs-dir}` with
your actual paths. The package list can come from the registry index — for
example, to find all unenriched packages:

```bash
grep "enriched: false" registry/index.yaml
```


## Tips for efficient enrichment

- **Start with pure Python packages.** Use `--skip-extensions` when running
  labeille. Pure Python packages are most likely to build on 3.15 and most
  useful for JIT testing (the JIT optimizes Python bytecode, not C extensions).

- **Batch similar packages.** Packages from the same organization (pallets,
  pypa, aio-libs, etc.) often have similar test setups, so you can reuse
  patterns.

- **Use persistent directories.** The `--repos-dir` and `--venvs-dir` options
  save significant time when iterating. You don't re-clone or re-create venvs
  on every run. Use `--refresh-venvs` only when you change install commands.

- **Set reasonable timeouts.** Use `--timeout` with 300-600 seconds to avoid
  getting stuck on packages with hanging tests.

- **Don't spend hours on one package.** When in doubt about a complex package,
  set `skip: true` with notes about what you tried, and move on. There are
  thousands of packages to test.

- **Scan test imports upfront.** Rather than using labeille as a dependency
  oracle (run, see ModuleNotFoundError, add dep, re-run), read the test files
  first and add all non-stdlib imports to the install command in one go. This
  saves hours of retry cycles across a large batch.

- **Run batches, not individual packages.** After enriching a batch, do a full
  run to catch any remaining issues:
  ```bash
  labeille run --target-python /path/to/jit-python \
      --packages pkg1,pkg2,pkg3 \
      --repos-dir /tmp/repos --venvs-dir /tmp/venvs \
      --timeout 600 -v
  ```

- **Use parallel execution for batch runs.** `--workers N` tests multiple
  packages in parallel, overlapping clone/install/test across packages:
  ```bash
  labeille run --target-python /path/to/python --workers 4
  ```
  Each parallel package test spawns its own subprocesses with a full venv.
  With N workers, expect roughly N times the memory usage of a single run.
  For ASAN builds, `--workers 2-3` is the practical limit. For non-ASAN
  builds, `--workers 4-8` works well on most machines.

- **Check for orphaned processes after killing labeille.** If you interrupt a
  labeille run, pytest subprocesses may survive and consume significant memory.
  Check with `ps aux | grep pytest` and clean up as needed.

- **Verify YAML after every edit.** One broken YAML file blocks all packages.
  A quick `python -c "import yaml; yaml.safe_load(open('file.yaml'))"` saves
  you from debugging a confusing "all packages failed" situation.

- **Update the registry index after editing package files:**
  ```python
  from labeille.registry import load_index, update_index_from_packages, save_index
  from pathlib import Path

  registry = Path("registry")
  index = load_index(registry)
  update_index_from_packages(index, registry)
  save_index(index, registry)
  ```
