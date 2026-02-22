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

**Early development — not yet functional.** The project structure, CLI skeleton, and
registry schema are in place, but the core logic is not yet implemented.

## Quick Start

```bash
# Install in development mode
pip install -e '.[dev]'

# Resolve top PyPI packages to source repositories (not yet implemented)
labeille resolve --top 100

# Run test suites against a JIT-enabled Python build (not yet implemented)
labeille run --python /path/to/jit-python --packages registry/packages/
```

## Project Structure

```
labeille/
├── src/labeille/        # Main package
│   ├── cli.py           # Click CLI with resolve and run subcommands
│   ├── resolve.py       # Resolve PyPI packages to source repositories
│   ├── runner.py        # Run test suites and capture results
│   ├── registry.py      # Registry reading/writing/schema
│   ├── classifier.py    # Pure Python / C extension detection
│   ├── crash.py         # Crash signature extraction
│   └── logging.py       # Structured logging setup
├── tests/               # Unit tests
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
