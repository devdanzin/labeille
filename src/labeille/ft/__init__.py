"""Free-threading compatibility testing for labeille.

Tests real-world PyPI packages against free-threaded CPython builds,
detecting crashes, deadlocks, race conditions, and GIL fallback
behavior across multiple runs.
"""
