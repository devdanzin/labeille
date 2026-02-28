# labeille

**Hunt for CPython JIT bugs by running real-world test suites.**

labeille is a companion to [lafleur](https://github.com/devdanzin/lafleur), an
evolutionary fuzzer for CPython's JIT compiler. Where lafleur generates synthetic
programs to find structural bugs, labeille takes a complementary approach: it runs
the test suites of popular PyPI packages against JIT-enabled CPython builds to find
crashes — segfaults, aborts, and assertion failures that only surface with real-world
code patterns.

## Why?

Fuzzers are great at finding crashes triggered by unusual code structures, but they
rarely produce code that resembles real-world usage. Meanwhile, the test suites of
popular packages exercise well-established code patterns, library interactions, and
edge cases that package authors have accumulated over years. Running these suites
against a JIT-enabled CPython catches bugs that synthetic programs miss — semantic
errors, optimization regressions, and interaction effects between the JIT and native
extensions.

## Status

**Early development (alpha).** Both the `resolve` and `run` subcommands are
implemented. The tool can resolve PyPI packages to source repositories, classify
them by extension type, and run their test suites against a JIT-enabled Python
build. Crash detection, signature extraction, and JSONL result recording are
functional. The registry format and CLI interface may change.

## Security Considerations

Labeille installs PyPI packages and runs their test suites, which means
executing arbitrary third-party code on your machine. This is inherent to the
task, not a bug — `setup.py`, build scripts, post-install hooks, and test code
all run with your user's privileges.

**Run labeille in a disposable, isolated environment**, especially when testing
beyond the most popular, well-audited packages. Even for well-known packages,
supply chain attacks (typosquatting, compromised maintainer accounts, malicious
updates) are a real and growing threat.

Recommended isolation strategies, from simplest to strongest:

- **Docker or Podman container** — easiest to set up, good process isolation
- **Dedicated VM** — stronger isolation from host filesystem and network
- **Ephemeral cloud instance** torn down after each batch run — strongest
  guarantee of a clean slate
- At minimum, avoid running as root and use a dedicated user account

When using `--repos-dir` or `--venvs-dir` for persistent directories, cached
repos and venvs from previous runs persist on disk. A compromised package's
artifacts survive across runs unless the directories are cleaned.

## Installation

```bash
pipx install labeille
```

Or with pip:

```bash
pip install labeille
```

### From source

```bash
git clone https://github.com/devdanzin/labeille
cd labeille
pip install -e '.[dev]'
```

## Quick Start

```bash
# Step 1: Resolve packages — build the test registry from a PyPI top-packages dump
labeille resolve --from-json top-pypi-packages.json --top 50 --registry-dir registry

# Or resolve specific packages by name
labeille resolve requests click flask --registry-dir registry

# Step 2: Run test suites against a JIT-enabled Python build
labeille run --target-python /path/to/jit-python --registry-dir registry

# Dry-run to see what would be tested without actually running anything
labeille run --target-python /path/to/jit-python --dry-run

# Run only pure-Python packages (skip C extensions)
labeille run --target-python /path/to/jit-python --skip-extensions

# Stop after finding the first crash
labeille run --target-python /path/to/jit-python --stop-after-crash 1

# Run tests in parallel (4 workers)
labeille run --target-python /path/to/jit-python --workers 4

# Test a specific package at a specific git revision
labeille run --target-python /path/to/jit-python \
    --packages=requests@abc1234 --no-shallow
```

### Testing specific revisions

To test a package at a specific git revision (useful for reproducing
crashes or bisecting regressions):

```bash
labeille run --target-python /path/to/python \
    --packages=requests@abc1234 --no-shallow
```

The `@revision` accepts any git ref: commit hashes, branch names,
tags, or relative refs like `HEAD~10`. Use `--no-shallow` (or
`--clone-depth=0`) when the target revision may be beyond the default
shallow clone depth.

Revision overrides are ephemeral — they apply to the current run only
and are not written back to the registry. The exact CLI invocation is
recorded in `run_meta.json` for reproducibility.

### Runtime customization

Override test behavior without modifying the registry:

```bash
# Run with coverage
labeille run --extra-deps coverage \
    --test-command-override "coverage run -m pytest"

# Add verbose output to all test commands
labeille run --test-command-suffix "--tb=long -v"

# Test a fork
labeille run --packages=requests \
    --repo-override "requests=https://github.com/fork/requests"

# Combine: test a specific revision of a fork with extra deps
labeille run --packages=requests@fix-branch \
    --repo-override "requests=https://github.com/fork/requests" \
    --extra-deps "coverage" --no-shallow
```

### Bisecting crashes

When a crash is found, bisect the package's git history to pinpoint the
first commit that introduced it:

```bash
# Find which commit introduced a SIGSEGV in requests
labeille bisect requests \
    --good=v2.30.0 --bad=v2.31.0 \
    --target-python /path/to/jit-python

# Filter by crash signature
labeille bisect requests \
    --good=v2.30.0 --bad=v2.31.0 \
    --target-python /path/to/jit-python \
    --crash-signature "SIGSEGV"

# Use a persistent work directory (avoids re-cloning)
labeille bisect requests \
    --good=v2.30.0 --bad=v2.31.0 \
    --target-python /path/to/jit-python \
    --work-dir /tmp/bisect-work
```

The bisect algorithm clones the repo at full depth, verifies the good and
bad revisions, then binary-searches to find the first bad commit. Commits
that fail to build are automatically skipped by trying neighboring commits.

### Platform support

System profiling works on Linux and macOS. Platform-specific details:

- **Linux**: CPU info from `/proc/cpuinfo`, memory from `/proc/meminfo`,
  disk type from `/sys/block/`.
- **macOS**: CPU info from `sysctl`, memory from `vm_stat`, disk type
  from `diskutil`.

All other features (registry, runner, analysis, bisect) work identically
on both platforms.

## How It Works

labeille operates in two phases:

### Phase 1: Resolve (`labeille resolve`)

Builds a registry of packages to test:

1. Reads package names from CLI arguments, a text file, or a PyPI top-packages
   JSON dump.
2. Queries the PyPI JSON API for each package to find its source repository URL.
3. Classifies each package as pure Python, C extension, or unknown by inspecting
   wheel tags.
4. Creates a YAML configuration file per package in the registry.
5. Updates the registry index sorted by download count.

Resolve is **non-destructive**: it never overwrites package files that have been
manually enriched (`enriched: true`).

### Phase 2: Run (`labeille run`)

Runs test suites and detects crashes:

1. Reads the registry and filters packages based on CLI options.
2. For each package: clones the repo, creates a venv with the target Python,
   installs the package, and runs its test command.
3. Sets `PYTHON_JIT=1` and `PYTHONFAULTHANDLER=1` to enable the JIT and get
   crash tracebacks.
4. Classifies each result as pass, fail, crash, timeout, or error.
5. For crashes: extracts a signature (signal + stderr context) and saves the
   full stderr output.
6. Writes results as JSONL for analysis, with full metadata for reproducibility.

Runs can execute packages in parallel with `--workers N` for faster batch
testing. Each worker handles one package end-to-end with results collected
thread-safely.

Runs are **resumable**: use `--skip-completed` with the same `--run-id` to
continue after an interruption.

## Dependency Scanning

Before enriching a package, scan its test imports to discover dependencies:

```bash
# Clone and scan
git clone --depth=1 https://github.com/psf/requests /tmp/requests
labeille scan-deps /tmp/requests --package-name requests

# Compare against existing install_command
labeille scan-deps /tmp/requests --package-name requests \
    --install-command "pip install -e '.[dev]'"

# Get just the pip install line for missing deps
labeille scan-deps /tmp/requests --format pip

# JSON output for scripting
labeille scan-deps /tmp/requests --format json
```

## Enriching Packages

After `labeille resolve` creates skeleton registry files, each package needs
to be *enriched* with specific installation and test instructions. This is
the most important step — without accurate enrichment, test runs will fail
with missing dependencies, broken installs, or pytest configuration errors.

Enrichment can be done manually, with Claude Code, or with another AI coding
agent. The process is iterative: fill in the YAML fields, run the tests,
diagnose any failures, fix the YAML, and re-run until the test harness works.

For the complete guide — including field reference, step-by-step walkthrough,
common problems, and ready-to-use Claude Code prompts — see
**[doc/enrichment.md](doc/enrichment.md)**.

## Analyzing Results

Analyze registry composition and run results:

```bash
# Registry overview (counts by type, framework, skip reasons)
labeille analyze registry

# Registry as a table, filtered
labeille analyze registry --format table --where extension_type:pure

# Single run summary (aggregate stats, crash detail, reproduce commands)
labeille analyze run

# Specific run, quiet mode (crashes only)
labeille analyze run 2026-02-23T08-01-05 -q

# Compare two runs (status changes, timing deltas)
labeille analyze compare 2026-02-20T10-00-00 2026-02-22T10-00-00

# Run history with trends and flaky package detection
labeille analyze history --last 5

# Deep dive on a specific package
labeille analyze package requests
```

### Commit-aware comparison

When comparing runs, labeille shows whether each package's repository
changed between runs:

```
labeille analyze compare run_001 run_002

Status changes:
  requests: PASS → CRASH
    Repo: abc1234 → abc1234 (unchanged — likely a CPython/JIT regression)
```

This helps triage new crashes: if the package code didn't change,
the regression is almost certainly on the CPython/JIT side.

## Registry Management

Batch operations for managing the package registry:

```bash
# Preview adding a new field (dry run)
labeille registry add-field skip_versions --type dict --after skip_reason

# Apply the change
labeille registry add-field skip_versions --type dict --after skip_reason --apply

# Resume after an interrupted operation
labeille registry add-field skip_versions --type dict --after skip_reason --apply --lenient

# Set a field on filtered packages
labeille registry set-field timeout 600 --where extension_type=extensions --apply

# Validate registry against schema
labeille registry validate

# Remove a deprecated field
labeille registry remove-field old_field --apply --lenient
```

## Registry Format

### Package file (`registry/packages/{name}.yaml`)

```yaml
package: requests
repo: "https://github.com/psf/requests"
pypi_url: "https://pypi.org/project/requests/"
extension_type: pure       # pure | extensions | unknown
python_versions: []
install_method: pip
install_command: "pip install -e '.[dev]'"
test_command: "python -m pytest tests/"
test_framework: pytest
uses_xdist: false
timeout: null
skip: false
skip_reason: null
skip_versions:             # per-version skip reasons (empty = no version skips)
  "3.15": "PyO3 not supported on 3.15"
notes: ""
enriched: false            # set to true after manual review
clone_depth: null           # null = shallow (depth=1); set higher for setuptools-scm
import_name: null           # null = derived from package name; override when different
```

### Index file (`registry/index.yaml`)

```yaml
last_updated: "2026-02-22T14:30:00"
packages:
  - name: boto3
    download_count: 1611866263
    extension_type: unknown
    enriched: false
    skip: false
```

## Results

Each run creates a directory under `results/{run_id}/` containing:

- **`run_meta.json`** — Run metadata: Python version, JIT status, hostname, timing.
- **`results.jsonl`** — One JSON line per package with status, exit code, signal,
  crash signature, timing, and installed dependency versions.
- **`crashes/`** — Full stderr captures for crashed packages.
- **`run.log`** — Detailed debug log.

Result statuses: `pass`, `fail`, `crash`, `timeout`, `install_error`,
`clone_error`, `error`.

## Project Structure

```
labeille/
├── src/labeille/        # Main package
│   ├── cli.py           # Click CLI with resolve, run, bisect, scan-deps, registry, and analyze subcommands
│   ├── resolve.py       # Resolve PyPI packages to source repositories
│   ├── runner.py        # Run test suites and capture results
│   ├── bisect.py        # Automated crash bisection across git history
│   ├── registry.py      # Registry reading/writing/schema
│   ├── registry_cli.py  # Batch registry management CLI
│   ├── registry_ops.py  # Batch operations (add/remove/rename/set/validate)
│   ├── analyze.py       # Data loading and analysis functions
│   ├── analyze_cli.py   # Analysis CLI (registry, run, compare, history, package)
│   ├── formatting.py    # Shared text formatting (tables, histograms, sparklines)
│   ├── summary.py       # Run summary formatting
│   ├── yaml_lines.py    # Line-level YAML manipulation
│   ├── classifier.py    # Pure Python / C extension detection
│   ├── scan_deps.py     # AST-based test dependency scanner
│   ├── import_map.py    # Import name to pip package mapping
│   ├── crash.py         # Crash detection and signature extraction
│   └── logging.py       # Structured logging setup
├── doc/                 # Documentation
│   └── enrichment.md    # Package enrichment guide
├── tests/               # Unit and integration tests
├── registry/            # Package test configurations
│   ├── index.yaml       # Index of tracked packages
│   └── packages/        # Per-package YAML configs
└── results/             # Test run output (gitignored)
```

## Relationship to lafleur

[lafleur](https://github.com/devdanzin/lafleur) and labeille are complementary tools
for finding CPython JIT bugs:

| | lafleur | labeille |
|---|---|---|
| **Approach** | Evolutionary fuzzing | Real-world test suites |
| **Input** | Generated synthetic programs | Existing package tests |
| **Finds** | Structural JIT bugs | Semantic bugs, regressions |
| **Coverage** | Broad, random exploration | Targeted, real usage patterns |

Used together, they provide broad coverage of the JIT's behavior under both synthetic
and real-world workloads.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and
the pull request process.

## Acknowledgments

[Anthropic](https://www.anthropic.com/) provided financial support that enabled access to
advanced AI capabilities for labeille's development. See [CREDITS.md](CREDITS.md) for full
details.

## License

MIT — see [LICENSE](LICENSE) for details.
