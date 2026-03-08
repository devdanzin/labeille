# The Resolve-Run Workflow

## Overview

labeille works in two phases: **resolve** discovers packages and builds a registry,
then **run** executes their test suites against a JIT-enabled CPython build and
detects crashes. Between these phases, packages need to be **enriched** with
specific installation and test instructions.

```
labeille resolve  →  enrich registry  →  labeille run  →  analyze results
     │                     │                   │                  │
 Fetch PyPI           Fill in YAML         Clone, build,     Crashes, timing,
 metadata,            install/test         test, detect      comparisons
 find repos           commands             crashes
                       ↕
                   laruche (external registry repo)
```

The registry is maintained externally in
[laruche](https://github.com/devdanzin/laruche) and synced locally via
`labeille registry sync`. Resolve creates skeleton entries, enrichment fills
them in, and run consumes them. Each phase can be run independently and
repeatedly.


## Phase 1: Resolve

### What resolve does

`labeille resolve` fetches PyPI metadata for each package and extracts:

1. **Source repository URL** — from PyPI metadata fields (project_urls, home_page, etc.)
2. **Extension type** — `pure`, `extensions`, or `unknown`, determined from wheel tag analysis
3. **PyPI URL** — direct link to the package page

It writes two things to the registry directory (default: `~/.local/share/labeille/registry/`):
- `index.yaml` — a sorted list of all tracked packages with download counts
- `packages/{name}.yaml` — per-package YAML with the discovered metadata and
  empty fields for enrichment

### Package sources

You can feed packages to resolve from multiple sources:

```bash
# Positional arguments
labeille resolve requests click flask

# From a file (one package name per line)
labeille resolve --from-file packages.txt

# From a JSON file with download counts (e.g. from BigQuery)
labeille resolve --from-json top-pypi.json --top 500

# Combine sources — all are merged
labeille resolve requests --from-file extras.txt --from-json top-pypi.json --top 100
```

### Non-destructive updates

Resolve never overwrites package files that have been manually enriched
(`enriched: true`). Re-running resolve on an existing registry safely adds new
packages without touching ones you've already configured.

### Resolve options

| Option | Description |
|--------|-------------|
| `PACKAGES` | Package names (positional, multiple) |
| `--from-file FILE` | File with one package name per line |
| `--from-json FILE` | JSON file with download counts |
| `--top N` | Top N packages by downloads (requires `--from-json`) |
| `--registry-dir PATH` | Registry directory (default: `~/.local/share/labeille/registry/`) |
| `--workers N` | Parallel PyPI API requests (default: 1) |
| `--timeout SECONDS` | PyPI API request timeout (default: 10) |
| `--dry-run` | Show what would be done without writing files |
| `-v, --verbose` | Detailed output |
| `-q, --quiet` | Only show errors |
| `--log-file PATH` | Log file path (default: `labeille-resolve.log`) |


## Phase 2: Run

### What run does

`labeille run` reads the registry and, for each enriched package:

1. Clones the source repository (shallow by default)
2. Creates a virtual environment with the target Python
3. Installs the package using its `install_command`
4. Runs the test suite using its `test_command`
5. Detects crashes from exit codes, signals, and stderr patterns
6. Records results as JSONL with full metadata

The target Python has `PYTHON_JIT=1` and `PYTHONFAULTHANDLER=1` set automatically
to enable the JIT and get crash tracebacks.

### Basic usage

```bash
# Run all enriched packages
labeille run --target-python ~/cpython/python

# Run specific packages
labeille run --target-python ~/cpython/python --packages requests,click,flask

# Run top 50 by downloads, 4 workers
labeille run --target-python ~/cpython/python --top 50 --workers 4

# Test a specific git revision
labeille run --target-python ~/cpython/python --packages=urllib3@abc1234 --no-shallow
```

### Package filtering

| Option | Description |
|--------|-------------|
| `--packages CSV` | Comma-separated filter (supports `name@revision` syntax) |
| `--top N` | Top N packages by download count |
| `--skip-extensions` | Skip C extension packages (test pure Python only) |
| `--force-run` | Override `skip` and `skip_versions` flags |
| `--skip-completed` | Resume: skip already-tested packages |

### Result statuses

| Status | Meaning |
|--------|---------|
| `pass` | Test suite exited 0 (all tests passed) |
| `fail` | Test suite exited 1 (some tests failed — normal) |
| `crash` | Segfault, abort, or assertion failure detected |
| `timeout` | Test suite exceeded the time limit |
| `install_error` | Package installation failed |
| `clone_error` | Git clone failed |
| `error` | Other unexpected error |

### Crash detection

labeille detects crashes from:
- **Exit codes**: negative exit codes (signals), exit code 134 (SIGABRT), 139 (SIGSEGV)
- **Stderr patterns**: ASAN reports, Python fatal errors, assertion failures
- **Signal names**: extracted from the exit code for crash signature

Each crash gets a signature (signal + stderr context) for deduplication and
comparison across runs.

### Parallel execution

`--workers N` runs N packages in parallel. Each worker handles one package
end-to-end. Memory scales linearly with worker count. For ASAN-enabled builds,
2-3 workers is the practical limit due to ~2-3x memory overhead per process.

### Persistent directories

By default, repos and venvs are created in temporary directories and cleaned up
after each package. For faster repeated runs:

```bash
# Reuse repos and venvs across runs
labeille run --target-python ~/cpython/python --work-dir ~/labeille-work

# Or set them individually
labeille run --target-python ~/cpython/python \
    --repos-dir ~/repos --venvs-dir ~/venvs

# Force reinstall when install commands change
labeille run --target-python ~/cpython/python --work-dir ~/work --refresh-venvs
```

### Resuming interrupted runs

```bash
# Start a named run
labeille run --target-python ~/cpython/python --run-id my-batch

# Resume after interruption (skips already-tested packages)
labeille run --target-python ~/cpython/python --run-id my-batch --skip-completed
```

### Runtime overrides

Override registry settings without editing YAML files:

```bash
# Inject extra dependencies into every venv
labeille run --target-python ~/cpython/python --extra-deps "trustme,uvicorn"

# Override all test commands
labeille run --target-python ~/cpython/python --test-command-override "python -m pytest -x"

# Append flags to test commands
labeille run --target-python ~/cpython/python --test-command-suffix "--tb=long -v"

# Use a fork's repo
labeille run --target-python ~/cpython/python --repo-override "requests=https://github.com/me/requests"
```

### Run options

| Option | Description |
|--------|-------------|
| `--target-python PATH` | Python interpreter to test with (required) |
| `--registry-dir PATH` | Registry directory (default: `~/.local/share/labeille/registry/`) |
| `--results-dir PATH` | Output directory (default: `results`) |
| `--packages CSV` | Comma-separated filter (supports `name@revision`) |
| `--top N` | Top N packages by download count |
| `--workers N` | Parallel package execution (default: 1) |
| `--timeout SECONDS` | Per-package timeout (default: 600) |
| `--skip-extensions` | Skip C extension packages |
| `--skip-completed` | Resume: skip already-tested packages |
| `--force-run` | Override skip and skip_versions flags |
| `--stop-after-crash N` | Stop after N crashes found |
| `--run-id ID` | Custom run identifier (default: timestamp) |
| `--work-dir PATH` | Base directory for repos and venvs |
| `--repos-dir PATH` | Persistent repo clones |
| `--venvs-dir PATH` | Persistent venvs |
| `--keep-work-dirs` | Don't clean up temporary directories |
| `--refresh-venvs` | Delete and recreate existing venvs |
| `--extra-deps CSV` | Inject additional packages into every venv |
| `--test-command-override STR` | Replace all test commands |
| `--test-command-suffix STR` | Append flags to test commands |
| `--repo-override PKG=URL` | Override repo URL (repeatable) |
| `--clone-depth N` | Git clone depth |
| `--no-shallow` | Full clone (needed for old revisions) |
| `--installer {auto,uv,pip}` | Package installer backend (default: auto) |
| `--env KEY=VALUE` | Extra environment variables (repeatable) |
| `--dry-run` | Don't actually execute tests |
| `-v, --verbose` | Show all details |
| `-q, --quiet` | Only show crashes |


## The Registry Bridge

The registry connects resolve and run. It is maintained as a separate project,
[laruche](https://github.com/devdanzin/laruche), and synced locally via
`labeille registry sync`. By default, the registry lives at
`~/.local/share/labeille/registry/`.

1. **Resolve creates it** — writes skeleton YAML with repo URL and extension type
2. **Enrichment fills it in** — adds install_command, test_command, dependencies
3. **Run consumes it** — reads the commands and executes them

This separation means you resolve once and run many times against different
Python builds. See [enrichment.md](enrichment.md) for the enrichment guide and
the [laruche repository](https://github.com/devdanzin/laruche) for the full
field schema and enrichment documentation.


## Complete Workflow Example

```bash
# 1. Build a registry from the top 100 PyPI packages
labeille resolve --from-json top-pypi.json --top 100

# 2. Enrich packages (see doc/enrichment.md)
#    Fill in install_command, test_command, etc. for each package

# 3. Run tests against a JIT-enabled CPython build
labeille run --target-python ~/jit_cpython/python \
    --results-dir results \
    --work-dir ~/labeille-work \
    --workers 4

# 4. Analyze results
labeille analyze run                            # Summary
labeille analyze run -q                         # Crashes only
labeille analyze package requests               # Deep dive on one package

# 5. Compare with a previous run
labeille analyze compare run_001 run_002
```


## Investigation Workflow

When a crash is found:

```bash
# 1. Reproduce the crash
labeille run --target-python ~/jit_cpython/python --packages=urllib3 \
    --work-dir ~/investigate

# 2. Check if HEAD still crashes
labeille run --target-python ~/jit_cpython/python --packages=urllib3

# 3. Test at the specific revision where the crash was found
labeille run --target-python ~/jit_cpython/python \
    --packages=urllib3@abc1234 --no-shallow

# 4. Bisect across the package's git history
labeille bisect --target-python ~/jit_cpython/python \
    --package urllib3 --good v2.0.0 --bad HEAD
```

If the package code didn't change between runs but the crash appeared, the
regression is almost certainly on the CPython/JIT side.


## Results

Each run creates a directory under `results/{run_id}/`:

- **`run_meta.json`** — Run metadata: Python version, JIT status, hostname, timing
- **`results.jsonl`** — One JSON line per package with status, exit code, signal,
  crash signature, timing, and installed dependency versions
- **`crashes/`** — Full stderr captures for crashed packages
- **`run.log`** — Detailed debug log

Use `labeille analyze` commands to examine results, or process the JSONL directly.


## What's Next

Once you have the basic workflow running, labeille offers specialized testing modes:

- **[Benchmarking](benchmarking.md)** — Compare test suite performance across conditions
  (JIT vs no-JIT, different interpreters, with/without coverage)
- **[Free-threaded testing](free-threaded.md)** — Test packages against free-threaded
  CPython builds to detect crashes, deadlocks, and race conditions
- **[Compatibility analysis](compat.md)** — Survey C extension packages for build
  compatibility against any Python version
