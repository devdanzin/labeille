# Benchmarking

## What is benchmarking in labeille?

labeille's benchmarking system measures **whole test suite execution time** across
different conditions — JIT vs no-JIT, different interpreters, with/without coverage,
varying resource constraints. It is not a microbenchmark tool: it runs the same test
suites you use with `labeille run`, collects wall/user/sys time and peak RSS for each
iteration, and produces statistical comparisons.

This answers questions like:
- How much overhead does the JIT add to test suite execution?
- Is a new CPython build faster or slower than the previous one?
- Which packages show the most JIT overhead?
- Are there performance regressions over time?


## Quick Start

### Inline conditions (simplest)

Compare JIT-enabled vs JIT-disabled with two inline conditions:

```bash
labeille bench run \
    --condition "jit:target_python=/opt/cpython-jit/python,env.PYTHON_JIT=1" \
    --condition "nojit:target_python=/opt/cpython-jit/python,env.PYTHON_JIT=0" \
    --registry-dir registry \
    --work-dir ~/bench-work \
    --packages requests,click,flask
```

### Quick mode

For a fast sanity check — 3 iterations, no warmup, top 20 packages:

```bash
labeille bench run --quick \
    --condition "jit:target_python=/opt/cpython-jit/python,env.PYTHON_JIT=1" \
    --condition "nojit:target_python=/opt/cpython-jit/python,env.PYTHON_JIT=0" \
    --registry-dir registry --work-dir ~/bench-work
```

### YAML profile (recommended for repeated use)

```bash
labeille bench run --profile jit-overhead.yaml \
    --registry-dir registry --work-dir ~/bench-work
```

### View and compare results

```bash
# Display results
labeille bench show results/bench_20260303_140000

# Compare conditions within a run
labeille bench compare results/bench_20260303_140000

# Compare across runs
labeille bench compare results/bench_run1 results/bench_run2
```


## Concepts

### Conditions

A **condition** is a named configuration that defines how to run tests. Each benchmark
compares one or more conditions. A condition can specify:

- `target_python` — path to the Python interpreter
- `env` — environment variables (e.g., `PYTHON_JIT=1`)
- `extra_deps` — additional packages to install
- `test_command_prefix` — prepend to test commands (e.g., `coverage run -m`)
- `test_command_suffix` — append to test commands
- `test_command_override` — replace test commands entirely
- `install_command` — override install commands
- `constraints` — resource limits (memory, CPU affinity, CPU time)

### Iterations and warmup

Each package's test suite runs multiple times per condition:

- **Warmup iterations** (default: 1) — not included in statistics, allows caches and
  JIT compilation to stabilize
- **Measured iterations** (default: 5, minimum: 3) — collected for statistical analysis

More iterations improve statistical confidence but increase total runtime linearly.

### Execution strategies

When comparing multiple conditions:

- **Alternating** (default for multi-condition): runs condition A then B for package 1,
  then A then B for package 2, etc. Reduces systematic bias from time-varying factors
  (thermal throttling, background load).
- **Block**: runs all iterations of condition A for all packages, then all of condition B.
  Faster but more susceptible to systematic bias.
- **Interleaved** (`--interleave`): interleaves packages across iterations. Useful when
  you want to distribute cache/memory effects across the run.


## YAML Profile Format

A profile defines conditions and shared settings in a YAML file:

```yaml
name: JIT overhead measurement
description: Compare JIT-enabled vs JIT-disabled CPython

iterations: 7
warmup: 2
timeout: 600

conditions:
  jit:
    target_python: /opt/cpython-jit/python
    env:
      PYTHON_JIT: "1"

  nojit:
    target_python: /opt/cpython-jit/python
    env:
      PYTHON_JIT: "0"

# Shared settings applied to all conditions unless overridden
default_env:
  ASAN_OPTIONS: "detect_leaks=0"

default_extra_deps:
  - pytest-timeout

# Optional: package filtering
packages:
  - requests
  - click
  - flask

# Optional: resource constraints applied to all conditions
default_constraints:
  cpu_affinity: "0,1"
  memory_limit_mb: 4096
```

### Inline condition syntax

The `--condition` flag uses the format `name:key=value,key=value`:

```bash
--condition "jit:target_python=/opt/python,env.PYTHON_JIT=1"
--condition "nojit:target_python=/opt/python,env.PYTHON_JIT=0"
```

Supported keys: `target_python`, `env.KEY`, `extra_deps`, `test_command_prefix`,
`test_command_suffix`, `test_command_override`, `install_command`.


## Running Benchmarks

### System stability

For reliable results, ensure the system is quiet:

```bash
# Check stability before starting (warns if load is high)
labeille bench run --check-stability --profile profile.yaml ...

# Wait for system to stabilize before starting
labeille bench run --wait-for-stability --profile profile.yaml ...
```

### Package selection

Same options as `labeille run`:

```bash
--packages requests,click    # Specific packages
--top 50                     # Top N by downloads
--registry-dir registry      # Required
```

### Persistent directories

Reuse repos and venvs across benchmark runs:

```bash
--work-dir ~/bench-work      # Sets both repos-dir and venvs-dir
--repos-dir ~/repos           # Or set individually
--venvs-dir ~/venvs
```

### Resource constraints

Control resource usage per iteration:

```bash
# Memory limit (ulimit -v)
labeille bench run --memory-limit 4096 ...

# CPU affinity (taskset) — pin to specific cores
labeille bench run --cpu-affinity "0,1" ...

# CPU time limit (ulimit -t)
labeille bench run --cpu-time-limit 300 ...
```

Constraints can also be set per-condition in a YAML profile.

### Per-test timing

Capture individual test timings via `pytest --durations=0`:

```bash
labeille bench run --per-test-timing --profile profile.yaml ...
```

This enables `--per-test` in `bench show` and `bench compare` to identify which
specific tests contribute most to overhead.

### Cache dropping

For cold-start benchmarks, drop filesystem caches between iterations:

```bash
# First: set up the cache-drop helper (requires sudo configuration)
labeille bench setup-cache-drop

# Then run with cache dropping
labeille bench run --drop-caches --profile profile.yaml ...

# Or compare warm vs cold automatically
labeille bench run --warm-vs-cold --profile profile.yaml ...
```


## Viewing Results

### bench show

Display results from a benchmark run:

```bash
labeille bench show results/bench_20260303_140000
```

Shows system profile, conditions defined, and a per-package table with median
wall time, IQR, coefficient of variation, and status for each condition.

### Anomaly detection

Flag measurement anomalies — high variance, bimodal distributions, outliers:

```bash
labeille bench show results/bench_20260303_140000 --anomalies
```

Anomaly types: `high_cv` (coefficient of variation > threshold), `bimodal`
(suspected multimodal distribution), `outlier_heavy` (many outlier iterations),
`status_mixed` (some iterations pass, some fail), `trend` (monotonic drift).

### Per-test timing

Show individual test timings for a specific package:

```bash
labeille bench show results/bench_20260303_140000 --per-test requests
```


## Comparing Results

### Within a single run (multi-condition)

Compare conditions defined in the same benchmark:

```bash
labeille bench compare results/bench_20260303_140000
```

Shows overhead percentage, confidence intervals, and statistical significance
(Welch's t-test) for each package.

### Across separate runs

Compare results from different benchmark executions:

```bash
labeille bench compare results/bench_run1 results/bench_run2
```

### Per-test comparison

Identify which tests contribute most to overhead:

```bash
labeille bench compare results/bench_20260303_140000 --per-test requests
```

### Choosing a metric

```bash
--metric wall    # Wall clock time (default)
--metric cpu     # User + sys CPU time
--metric rss     # Peak resident set size
```


## Longitudinal Tracking

Track benchmark performance over time with tracking series.

### Creating a series

```bash
labeille bench track init jit-perf --description "JIT performance over CPython commits"
```

### Adding runs to a series

```bash
labeille bench track add jit-perf results/bench_20260303_140000 \
    --notes "CPython main @ abc1234" \
    --commit sha=abc1234,branch=main
```

### Viewing series history

```bash
labeille bench track show jit-perf          # All runs
labeille bench track show jit-perf --last 5 # Last 5 runs
```

### Pinning a baseline

Pin a specific run as the reference point for trend analysis:

```bash
labeille bench track pin jit-perf bench_20260301_100000
labeille bench track unpin jit-perf   # Remove pin
```

### Trend analysis

Detect performance trends and regressions across the series:

```bash
labeille bench track trend jit-perf
labeille bench track trend jit-perf --condition jit --format markdown
```

Classifies each package as: `stable`, `improving`, `regressing`, or `volatile`.

Thresholds are configurable:
- `--regression-threshold 0.02` — per-run change threshold (fraction)
- `--trend-threshold 0.05` — overall slope threshold for classification

### Regression alerts

Check for new regressions compared to the baseline and previous run:

```bash
labeille bench track alert jit-perf
```

### Listing all series

```bash
labeille bench track list
```


## Exporting Results

### CSV (raw data)

One row per package per condition per iteration — for pandas, R, or spreadsheets:

```bash
labeille bench export results/bench_20260303_140000 --format csv
labeille bench export results/bench_20260303_140000 --format csv -o data.csv
```

### CSV summary

One row per package per condition with aggregated statistics:

```bash
labeille bench export results/bench_20260303_140000 --format csv-summary
```

### Markdown

Summary table suitable for GitHub issues and reports:

```bash
labeille bench export results/bench_20260303_140000 --format markdown
```


## Output Files

A benchmark run produces:

```
results/bench_20260303_140000/
├── bench_meta.json       # System profile, Python profile, conditions, timing
└── bench_results.jsonl   # One JSON line per package with per-condition data
```

`bench_meta.json` contains:
- System characterization (CPU, RAM, OS, kernel)
- Python profile for each condition (version, JIT status, GIL, build flags)
- Condition definitions as resolved
- Execution timestamps

`bench_results.jsonl` contains per-package:
- Per-condition iteration timings (wall, user, sys, RSS)
- Descriptive statistics (mean, median, std, percentiles, IQR, CV)
- Outlier flags
- Per-test timings (if `--per-test-timing` was used)


## System Profiling

Print system characterization for documentation and reproducibility:

```bash
labeille bench system
labeille bench system --target-python /opt/cpython/python
labeille bench system --json
```


## Examples

### JIT overhead measurement

```bash
# Create a profile
cat > jit-profile.yaml << 'EOF'
name: JIT overhead
iterations: 7
warmup: 2
conditions:
  jit:
    target_python: /opt/cpython-jit/python
    env: { PYTHON_JIT: "1" }
  nojit:
    target_python: /opt/cpython-jit/python
    env: { PYTHON_JIT: "0" }
EOF

# Run the benchmark
labeille bench run --profile jit-profile.yaml \
    --registry-dir registry --work-dir ~/bench-work --top 30

# View results
labeille bench show results/bench_*

# Compare conditions
labeille bench compare results/bench_*
```

### Tracking JIT performance over time

```bash
# Initialize a series
labeille bench track init jit-tracking -d "Track JIT overhead across CPython commits"

# After each CPython build, run and add:
labeille bench run --profile jit-profile.yaml --registry-dir registry --work-dir ~/bench
labeille bench track add jit-tracking results/bench_* --commit sha=$(git -C ~/cpython rev-parse HEAD)

# Check for regressions
labeille bench track trend jit-tracking
labeille bench track alert jit-tracking
```


## Troubleshooting

### High variance in measurements

- Close other applications and background processes
- Use `--check-stability` or `--wait-for-stability`
- Pin CPU cores with `--cpu-affinity` to avoid migration
- Check for thermal throttling (sustained heavy loads)
- Increase `--iterations` for better statistical confidence

### Mixed pass/fail across iterations

Flaky tests pollute timing data. Use `--anomalies` with `bench show` to identify
packages with `status_mixed` anomalies. Consider adding `--test-command-suffix "-k
'not flaky_test'"` to exclude known flaky tests.

### OOM kills

- ASAN-enabled builds use ~2-3x memory; increase `--memory-limit` or use a non-ASAN build
- Reduce `--workers` if running other processes
- Check `oom_detected` field in results for confirmation
