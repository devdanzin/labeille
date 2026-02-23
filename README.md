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

## Quick Start

```bash
# Install in development mode
pip install -e '.[dev]'

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
```

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
│   ├── cli.py           # Click CLI with resolve and run subcommands
│   ├── resolve.py       # Resolve PyPI packages to source repositories
│   ├── runner.py        # Run test suites and capture results
│   ├── registry.py      # Registry reading/writing/schema
│   ├── classifier.py    # Pure Python / C extension detection
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

## License

MIT — see [LICENSE](LICENSE) for details.
