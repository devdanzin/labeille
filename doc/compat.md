# C Extension Compatibility Analysis

## What is compat?

`labeille compat` surveys C extension packages for build compatibility against any
Python version. It builds each package, captures the full build output, and classifies
failures into fine-grained categories using 40+ built-in error patterns.

This answers questions like:
- How many packages build successfully on Python 3.15?
- Which removed C APIs are blocking the most packages?
- Did a new CPython release fix or break anything compared to the last one?
- What are the main categories of build failures?


## Quick Start

### Registry-based survey

```bash
# Survey all C extension packages in the registry
labeille compat survey \
    --target-python ~/cpython-315/python \
    --registry-dir registry \
    --extensions-only

# Survey top 50 by downloads
labeille compat survey \
    --target-python ~/cpython-315/python \
    --registry-dir registry \
    --top 50
```

### Ad-hoc package list

```bash
# Inline package list
labeille compat survey \
    --target-python ~/cpython-315/python \
    --packages numpy,scipy,pandas

# From a file (one per line)
labeille compat survey \
    --target-python ~/cpython-315/python \
    --packages-file packages.txt
```

### View and compare results

```bash
# Display results
labeille compat show results/compat_20260303

# Compare two surveys (e.g., Python 3.14 vs 3.15)
labeille compat diff results/compat_314 results/compat_315
```


## Concepts

### Build modes

| Mode | Flag | What it does |
|------|------|-------------|
| **sdist** (default) | `--from sdist` | Downloads sdist from PyPI, builds with `pip install --no-binary <pkg> <pkg>` |
| **source** | `--from source` | Clones git repo, builds with the registry's `install_command` |
| **no-binary-all** | `--no-binary-all` | Builds everything from source including dependencies: `pip install --no-binary :all: <pkg>` |

**sdist** is the most common mode — it tests what PyPI users experience when
installing from source distributions. **source** tests the latest git HEAD.
**no-binary-all** is the strictest, forcing all transitive dependencies to build
from source too (much slower, more likely to time out).

### Error classification

Build output is matched against 40+ regular expression patterns organized into
categories and subcategories. When a build fails, all matching patterns are
recorded. The **primary category** is the first match (most specific patterns
are listed first).

### Error categories

| Category | Description | Since |
|----------|-------------|-------|
| `python_header` | Python.h not found (dev headers missing) | — |
| `removed_c_api` | Functions/macros removed in recent Python versions | 3.12+ |
| `changed_struct` | Direct struct member access (use accessor macros instead) | 3.12+ |
| `cython_incompatible` | Cython version too old for target Python | — |
| `pyo3_incompatible` | PyO3/Maturin doesn't support target Python | — |
| `numpy_c_api` | NumPy C API/ABI version mismatch | — |
| `missing_system_lib` | Missing system headers or libraries | — |
| `setuptools_distutils` | distutils removed, pkg_resources missing | 3.12+ |
| `build_backend` | PEP 517 backend errors, Meson, CMake | — |
| `compiler_error` | Generic C/C++ compilation or linker errors | — |
| `import_failure` | Undefined symbols or ABI mismatch at import time | — |

Each category has multiple subcategories. For example, `removed_c_api` includes:
`PyUnicode_READY`, `Py_UNICODE`, `PyUnicode_AS_UNICODE`, `tp_print`,
`PyEval_CallObject`, and more.

### Import probing

After a successful build, labeille imports the package to detect runtime issues
that don't surface during compilation (e.g., undefined symbols from ABI mismatches).
Import failures are classified as `import_fail` or `import_crash`.

### Crash detection

If the build or import process crashes (segfault, abort), labeille detects it
using the same crash detection module as `labeille run`.


## Running Surveys

### Package sources

Packages can come from three sources, which are merged:

```bash
--registry-dir registry      # All packages in the registry
--packages numpy,scipy       # Inline list
--packages-file packages.txt # One per line
```

Filter to C extension packages only (default for registry-based):

```bash
--extensions-only   # Only packages with extension_type=extensions (default)
--all-types         # Include pure Python packages too
```

### Build modes

```bash
--from sdist              # Build from PyPI sdist (default)
--from source             # Clone and build from git repo
--no-binary-all           # Force compile everything including deps
```

For source mode, you can specify a persistent repo directory:

```bash
--repos-dir ~/compat-repos   # Reuse repo clones
```

### Custom error patterns

Extend or override the built-in patterns with a YAML file:

```bash
labeille compat survey --extra-patterns my-patterns.yaml ...
```

### Parallel execution

```bash
--workers 4    # Run 4 package builds in parallel
```

### Other options

| Option | Description |
|--------|-------------|
| `--target-python PATH` | Python interpreter to build against (required) |
| `--output-dir PATH` | Output directory (default: `compat-results`) |
| `--timeout SECONDS` | Build timeout per package (default: 600) |
| `--workers N` | Parallel builds (default: 1) |
| `--installer {auto,uv,pip}` | Package installer backend (default: auto) |
| `--export-markdown` | Also export results as a markdown file |
| `-v, --verbose` | Detailed output |
| `-q, --quiet` | Only show errors |


## Viewing Results

### compat show

Display a saved survey:

```bash
labeille compat show results/compat_20260303

# Filter by status
labeille compat show results/compat_20260303 --status build_fail

# Filter by failure category
labeille compat show results/compat_20260303 --category removed_c_api

# Markdown format for sharing
labeille compat show results/compat_20260303 --format markdown
```

### Listing patterns

See all available error classification patterns:

```bash
labeille compat patterns

# Filter by category
labeille compat patterns --category removed_c_api

# Include custom patterns
labeille compat patterns --extra-patterns my-patterns.yaml
```


## Comparing Surveys

Compare two surveys to track compatibility changes:

```bash
labeille compat diff results/compat_314 results/compat_315
```

Shows:
- **Regressions**: packages that went from `build_ok` to a failure
- **Fixes**: packages that went from a failure to `build_ok`
- **Category changes**: packages whose failure category changed

Use case: run surveys against Python 3.14 and 3.15, then diff to see what the
new version broke or fixed.


## Custom Error Patterns

### YAML format

```yaml
patterns:
  - category: my_project
    subcategory: special_api
    since: "3.15"
    pattern: "error.*MySpecialAPI"
    description: "MySpecialAPI removed in 3.15"

  - category: removed_c_api
    subcategory: PyFrameObject_fields
    since: "3.15"
    pattern: "error.*PyFrameObject.*f_lineno"
    description: "Direct PyFrameObject field access removed"
```

### Overriding built-in patterns

If a custom pattern has the same `category` and `subcategory` as a built-in
pattern, the custom version takes precedence. This lets you refine built-in
patterns without modifying labeille's source code.


## Output Files

A survey produces:

```
compat-results/compat_20260303_140000/
├── compat_meta.json      # Survey metadata (target Python, mode, timing)
├── compat_results.jsonl  # One JSON line per package with status and matches
└── build_logs/           # Full stdout/stderr for failed builds
    ├── numpy_stdout.txt
    ├── numpy_stderr.txt
    └── ...
```

`compat_results.jsonl` contains per-package:
- Build status (`build_ok`, `build_fail`, `import_fail`, `import_crash`,
  `no_sdist`, `timeout`, `skip`)
- Exit code and duration
- Error matches (category, subcategory, matched line, line number)
- Primary category and subcategory
- Crash signature (if applicable)


## Examples

### Python 3.15 compatibility survey

```bash
# Survey all C extension packages in the registry
labeille compat survey \
    --target-python ~/cpython-315/python \
    --registry-dir registry \
    --extensions-only \
    --workers 4 \
    --export-markdown

# View the results
labeille compat show compat-results/compat_*

# Check which removed APIs are causing the most failures
labeille compat show compat-results/compat_* --category removed_c_api
```

### Cross-version tracking

```bash
# Survey against 3.14
labeille compat survey \
    --target-python ~/cpython-314/python \
    --packages-file c-ext-packages.txt \
    --output-dir compat-314

# Survey against 3.15
labeille compat survey \
    --target-python ~/cpython-315/python \
    --packages-file c-ext-packages.txt \
    --output-dir compat-315

# Diff: what changed?
labeille compat diff compat-314/compat_* compat-315/compat_*
```

### Source mode for unreleased code

Test the latest git HEAD of packages against the target Python:

```bash
labeille compat survey \
    --target-python ~/cpython-315/python \
    --registry-dir registry \
    --from source \
    --repos-dir ~/compat-repos \
    --extensions-only \
    --workers 2
```


## Troubleshooting

### All packages fail with python_header

The target Python's development headers are not installed or not in the expected
location. Verify:
```bash
ls $(~/cpython-315/python -c "import sysconfig; print(sysconfig.get_path('include'))")/Python.h
```

### sdist not available

Some packages only distribute wheels on PyPI. Use `--from source` to clone and
build from the git repository instead.

### Timeouts on large builds

Packages like numpy, scipy, and tensorflow can take 10+ minutes to build from
source. Increase `--timeout` accordingly, or exclude them from the survey.

### Many unclassified failures

If many packages show `unclassified/unknown`, the build errors don't match any
built-in patterns. Check the build logs in `build_logs/` and consider adding
custom patterns via `--extra-patterns`.
