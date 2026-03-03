# Free-Threaded Testing

## What is free-threaded testing?

Free-threaded CPython (PEP 703) removes the Global Interpreter Lock (GIL), enabling
true parallelism but introducing new failure modes: data races, deadlocks, and crashes
in code that relied on the GIL for thread safety.

`labeille ft` tests packages against free-threaded CPython builds by running each
package's test suite multiple times with `PYTHON_GIL=0`. Multiple iterations catch
intermittent failures — race conditions don't reproduce deterministically.

Each package is classified into a compatibility category: fully compatible, compatible
with GIL fallback, intermittent failures, incompatible, crash, or deadlock.


## Prerequisites

- A **free-threaded CPython build** (`./configure --disable-gil`)
- Optional: a **TSAN build** for race condition detection
  (`./configure --disable-gil --with-thread-sanitizer`)
- Packages enriched in the registry (see [workflow.md](workflow.md))


## Quick Start

### Basic survey

```bash
# Run each package 10 times with PYTHON_GIL=0
labeille ft run --target-python ~/cpython-ft/python \
    --registry-dir registry --work-dir ~/ft-work \
    --top 50

# View results
labeille ft show results/ft_*
```

### Quick coverage scan

Stop after the first passing iteration per package:

```bash
labeille ft run --target-python ~/cpython-ft/python \
    --registry-dir registry --work-dir ~/ft-work \
    --stop-on-first-pass --top 100
```

### Thorough analysis with GIL comparison

```bash
labeille ft run --target-python ~/cpython-ft/python \
    --registry-dir registry --work-dir ~/ft-work \
    --compare-with-gil --iterations 10 --top 50
```

### TSAN race detection

```bash
labeille ft run --target-python ~/cpython-tsan/python \
    --registry-dir registry --work-dir ~/ft-work \
    --tsan --iterations 5 --top 30
```


## Concepts

### Failure categories

Each package is classified based on its behavior across all iterations:

| Category | Description |
|----------|-------------|
| `compatible` | All iterations pass — fully compatible with free-threading |
| `compatible_gil_fallback` | Passes, but the GIL is re-enabled at runtime (C extension without `Py_mod_gil`) |
| `intermittent` | Some iterations pass, some fail — likely race conditions |
| `incompatible` | All iterations fail consistently (not a race, just broken) |
| `crash` | Segfault, abort, or assertion failure under free-threading |
| `deadlock` | Test suite stalls with no output (detected via stall threshold) |
| `tsan_warnings` | TSAN detects data races (may still pass functionally) |
| `install_failure` | Could not install the package |
| `import_failure` | Package installs but fails to import |
| `unknown` | Uncategorizable result |

### Iterations and reliability

Free-threading bugs are often non-deterministic. A single run may pass even if the
code has race conditions. The default of 10 iterations provides reasonable detection
probability:

- A race that manifests ~50% of the time will be caught with >99.9% probability
  in 10 runs
- A race that manifests ~10% of the time will be caught with ~65% probability

Increase `--iterations` for rare races, or use `--stop-on-first-pass` for quick
coverage scans where you only need to know if it *can* pass.

### Deadlock detection

`--stall-threshold SECONDS` (default: 60) monitors test suite output. If no output
is produced for the threshold duration, the iteration is classified as a deadlock
and the process is killed.

Increase the threshold for packages with naturally slow tests (heavy computation,
network waits).

### GIL comparison mode

`--compare-with-gil` runs each package twice:
1. With `PYTHON_GIL=0` (free-threaded)
2. With `PYTHON_GIL=1` (GIL enabled)

This isolates free-threading-specific failures: if a package fails with both
GIL=0 and GIL=1, the failure is unrelated to free-threading (e.g., a pre-existing
test bug). Only failures unique to GIL=0 are true free-threading issues.

### Extension GIL compatibility

When `--detect-extensions` is enabled (default), labeille probes each package's
C extensions for free-threading compatibility:

- Checks if extensions declare `Py_mod_gil` (the opt-in for free-threading)
- Monitors `sys._is_gil_enabled()` at runtime to detect GIL fallback
- Reports which packages trigger GIL re-enablement due to incompatible extensions


## Running Tests

### Package selection

```bash
--packages requests,click    # Specific packages
--top 50                     # Top N by downloads
--registry-dir registry      # Required
```

### Iteration control

```bash
--iterations 10              # Measured iterations per package (default: 10)
--stop-on-first-pass         # Stop after first passing iteration
```

### Timeouts and stalls

```bash
--timeout 600                # Per-iteration timeout in seconds (default: 600)
--stall-threshold 60         # Seconds without output before deadlock (default: 60)
```

### Extra dependencies and overrides

```bash
--extra-deps "trustme,uvicorn"         # Inject deps into every venv
--test-command-suffix "--tb=short"     # Append to test commands
--test-command-override "python -m pytest tests/" # Replace all test commands
```

### Environment variables

```bash
--env PYTHONMALLOC=debug     # Extra env vars (repeatable)
--env PYTHONTRACEMALLOC=5
```

`PYTHON_GIL=0` is set automatically — you don't need to specify it.

### Persistent directories

```bash
--repos-dir ~/repos          # Reuse repo clones
--venvs-dir ~/venvs          # Reuse venvs
--results-dir results        # Output directory (default: results)
```

### Full options

| Option | Description |
|--------|-------------|
| `--target-python PATH` | Free-threaded Python build (required) |
| `--iterations N` | Iterations per package (default: 10) |
| `--timeout SECONDS` | Per-iteration timeout (default: 600) |
| `--stall-threshold SECONDS` | Deadlock detection threshold (default: 60) |
| `--packages CSV` | Comma-separated package filter |
| `--top N` | Top N packages by downloads |
| `--compare-with-gil` | Also run with GIL enabled for comparison |
| `--stop-on-first-pass` | Stop after first passing iteration |
| `--detect-extensions` | Check extension GIL compatibility (default: on) |
| `--tsan` | Parse TSAN warnings from stderr |
| `--check-stability` | Check system stability before starting |
| `--extra-deps CSV` | Extra dependencies for every venv |
| `--test-command-suffix STR` | Append to test commands |
| `--test-command-override STR` | Override all test commands |
| `--env KEY=VALUE` | Extra environment variables (repeatable) |
| `--registry-dir PATH` | Registry directory (default: `registry`) |
| `--repos-dir PATH` | Persistent repos (default: `repos`) |
| `--venvs-dir PATH` | Persistent venvs (default: `venvs`) |
| `--results-dir PATH` | Output directory (default: `results`) |
| `-v, --verbose` | Detailed output |


## Viewing Results

### ft show

Display a summary of free-threading test results:

```bash
labeille ft show results/ft_20260303_140000
labeille ft show results/ft_20260303_140000 --sort pass_rate
labeille ft show results/ft_20260303_140000 --limit 20
```

Shows: system and Python info, compatibility summary (counts per category),
and a per-package table with category, pass rate, and iteration details.

Sorting options: `category` (default), `pass_rate`, `name`.

### Flakiness analysis

Investigate intermittent failures in detail:

```bash
# Overview of all flaky packages
labeille ft flaky results/ft_20260303_140000

# Deep dive on one package
labeille ft flaky results/ft_20260303_140000 --package urllib3
```

Shows which tests fail intermittently, failure patterns across iterations, and
crash signature consistency.

### Extension compatibility

Check which packages have C extensions and their GIL compatibility status:

```bash
labeille ft compat results/ft_20260303_140000
labeille ft compat results/ft_20260303_140000 --extensions-only
```

Reports `Py_mod_gil` declarations, GIL fallback status, and whether extensions
are free-threading-safe.


## Comparing Runs

Track free-threading compatibility across CPython releases:

```bash
labeille ft compare results/ft_314a1 results/ft_314b2
```

Shows:
- Category transitions (e.g., `crash` -> `compatible`)
- Pass rate changes
- New issues and resolved issues

Use case: run the same packages against CPython 3.14a1 and 3.14b2 to track
which packages gain or lose free-threading compatibility.


## Reports and Exports

### Comprehensive report

Generate a full compatibility report for sharing:

```bash
labeille ft report results/ft_20260303_140000
labeille ft report results/ft_20260303_140000 --format markdown -o report.md
labeille ft report results/ft_20260303_140000 --format text
```

### Export for analysis

```bash
# CSV — one row per package
labeille ft export results/ft_20260303_140000 --format csv -o results.csv

# JSON — structured data
labeille ft export results/ft_20260303_140000 --format json -o results.json
```


## Output Files

A free-threading run produces:

```
results/ft_20260303_140000/
├── ft_meta.json          # Run metadata (target Python, config, system info)
└── ft_results.jsonl      # One JSON line per package with all iterations
```

`ft_results.jsonl` contains per-package:
- Failure category classification
- Per-iteration outcomes (pass/fail/crash/deadlock, exit code, duration)
- Pass rate and iteration count
- Extension GIL compatibility info (if `--detect-extensions`)
- TSAN warnings (if `--tsan`)
- GIL-enabled comparison results (if `--compare-with-gil`)


## Examples

### Ecosystem-wide survey

```bash
# Quick scan of top 100 packages
labeille ft run --target-python ~/cpython-ft/python \
    --registry-dir registry --work-dir ~/ft-work \
    --stop-on-first-pass --top 100

# View summary
labeille ft show results/ft_*

# Export report for the free-threading compatibility tracker
labeille ft report results/ft_* --format markdown -o ft-compat-report.md
```

### Package deep-dive

```bash
# Run many iterations for a specific package
labeille ft run --target-python ~/cpython-ft/python \
    --registry-dir registry --work-dir ~/ft-work \
    --packages urllib3 --iterations 50

# Analyze flakiness
labeille ft flaky results/ft_* --package urllib3
```

### Version-to-version tracking

```bash
# Run against 3.14a1
labeille ft run --target-python ~/cpython-314a1/python \
    --registry-dir registry --work-dir ~/ft-work --top 50 \
    --results-dir results/ft_314a1

# Run against 3.14b2
labeille ft run --target-python ~/cpython-314b2/python \
    --registry-dir registry --work-dir ~/ft-work --top 50 \
    --results-dir results/ft_314b2

# Compare
labeille ft compare results/ft_314a1 results/ft_314b2
```

### TSAN race detection

```bash
# Use a TSAN-instrumented build
labeille ft run --target-python ~/cpython-tsan/python \
    --registry-dir registry --work-dir ~/ft-work \
    --tsan --iterations 5 --top 30

# Check for packages with tsan_warnings category
labeille ft show results/ft_* --sort category
```


## Interpreting Results

### Compatible packages

Packages classified as `compatible` pass all iterations with `PYTHON_GIL=0`. They
are safe to use with free-threaded CPython.

`compatible_gil_fallback` means the package works, but the GIL is re-enabled at
runtime because a C extension doesn't declare `Py_mod_gil`. The package is
functionally compatible but doesn't benefit from true parallelism.

### Investigating intermittent failures

`intermittent` packages have race conditions. Use `ft flaky --package NAME` to see:
- Which specific tests fail and how often
- Whether failures are concentrated in specific tests or spread across the suite
- Whether crash signatures are consistent (same race) or varied (multiple races)

### Investigating crashes

`crash` packages segfault or abort under free-threading. Reproduce with:

```bash
PYTHON_GIL=0 ~/cpython-ft/python -m pytest tests/
```

If the crash doesn't reproduce, increase iterations — it may be intermittent.

### Investigating deadlocks

`deadlock` packages stall without producing output. Reproduce manually and attach
a debugger to inspect thread states. Common causes: lock ordering inversions,
missing lock releases on exception paths, busy-wait loops on GIL-protected state.


## Troubleshooting

### All packages show install_failure

The free-threaded Python build may not be properly configured. Verify:
```bash
~/cpython-ft/python -c "import sys; print(sys._is_gil_enabled())"
```
Should print `False` if the GIL is disabled by default.

### Too many false deadlocks

Increase `--stall-threshold` for packages with slow tests. The default 60 seconds
may be too short for heavy computation or network-dependent tests.

### TSAN not detecting anything

TSAN requires a TSAN-instrumented CPython build (`--with-thread-sanitizer`). The
`--tsan` flag only tells labeille to parse TSAN output — it doesn't enable TSAN
instrumentation in CPython itself.
