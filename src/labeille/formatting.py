"""Shared text formatting helpers for labeille.

Provides functions for formatting durations, tables, histograms, sparklines,
and other text output used across CLI commands and summaries.
"""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration.

    Examples: ``'8s'``, ``'1m 23s'``, ``'1h 12m 34s'``. Always whole seconds
    (truncated, not rounded).
    """
    total = int(seconds)
    if total >= 3600:
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h}h {m:2d}m {s:2d}s"
    if total >= 60:
        m = total // 60
        s = total % 60
        return f"{m}m {s:2d}s"
    return f"{total}s"


def format_status_icon(status: str) -> str:
    """Return a visual status indicator for the given status string."""
    icons: dict[str, str] = {
        "crash": "\u2717 CRASH",
        "timeout": "\u23f1 TIMEOUT",
        "fail": "\u2717 FAIL",
        "install_error": "\u26a0 ERROR",
        "clone_error": "\u26a0 ERROR",
        "error": "\u26a0 ERROR",
        "pass": "\u2713 PASS",
        "skip": "\u2298 SKIP",
    }
    return icons.get(status, status.upper())


def format_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    alignments: list[str] | None = None,
    max_col_width: dict[int, int] | None = None,
    indent: int = 2,
) -> str:
    """Format a list of rows as an aligned text table.

    Auto-calculates column widths from content. Truncates columns that
    exceed *max_col_width* (with ``'...'`` suffix). Right-aligns columns
    marked ``'r'`` in *alignments*.

    Args:
        headers: Column header strings.
        rows: List of rows, each a list of cell strings.
        alignments: Per-column alignment: ``'l'``, ``'r'``, or ``'c'``.
        max_col_width: Column index to max width mapping.
        indent: Number of leading spaces per line.

    Returns:
        The formatted table as a string.
    """
    if not headers:
        return ""

    ncols = len(headers)
    if alignments is None:
        alignments = ["l"] * ncols
    while len(alignments) < ncols:
        alignments.append("l")

    max_widths = max_col_width or {}

    # Truncate cells and compute column widths.
    def _trunc(text: str, max_w: int) -> str:
        if len(text) <= max_w:
            return text
        return text[: max_w - 3] + "..."

    proc_headers = list(headers)
    proc_rows: list[list[str]] = []
    for row in rows:
        padded = list(row) + [""] * (ncols - len(row))
        proc_rows.append(padded[:ncols])

    # Apply max_col_width truncation.
    for ci, max_w in max_widths.items():
        if ci < ncols:
            proc_headers[ci] = _trunc(proc_headers[ci], max_w)
            for row in proc_rows:
                row[ci] = _trunc(row[ci], max_w)

    # Compute widths.
    widths = [len(h) for h in proc_headers]
    for row in proc_rows:
        for ci, cell in enumerate(row):
            widths[ci] = max(widths[ci], len(cell))

    prefix = " " * indent

    def _format_cell(text: str, width: int, align: str) -> str:
        if align == "r":
            return text.rjust(width)
        if align == "c":
            return text.center(width)
        return text.ljust(width)

    lines: list[str] = []
    header_line = "  ".join(
        _format_cell(proc_headers[i], widths[i], alignments[i]) for i in range(ncols)
    )
    lines.append(prefix + header_line)

    for row in proc_rows:
        row_line = "  ".join(_format_cell(row[i], widths[i], alignments[i]) for i in range(ncols))
        lines.append(prefix + row_line)

    return "\n".join(lines)


def format_histogram(
    buckets: list[tuple[str, int]],
    *,
    max_bar_width: int = 40,
    show_percentages: bool = True,
    total: int | None = None,
) -> str:
    """Format a text histogram using block characters.

    Each bucket is ``(label, count)``. Labels are left-aligned, bars are
    proportional to the largest bucket.

    Args:
        buckets: List of (label, count) tuples.
        max_bar_width: Maximum width of the bar in characters.
        show_percentages: Whether to show percentage after count.
        total: Total for percentage calculation. If None, computed from buckets.

    Returns:
        The formatted histogram as a string.
    """
    if not buckets:
        return ""

    if total is None:
        total = sum(count for _, count in buckets)

    max_count = max((count for _, count in buckets), default=0)
    label_width = max((len(label) for label, _ in buckets), default=0)

    lines: list[str] = []
    for label, count in buckets:
        if max_count > 0:
            bar_len = int(count / max_count * max_bar_width)
            bar = "\u2588" * bar_len if bar_len > 0 else "\u258f"
        else:
            bar = ""

        count_str = f"{count:3d}"
        if show_percentages and total > 0:
            pct = count / total * 100
            pct_str = f"({pct:4.0f}%)"
        elif show_percentages:
            pct_str = "(  -%)"
        else:
            pct_str = ""

        line = f"  {label:>{label_width}s}   {bar:<{max_bar_width}s}  {count_str}  {pct_str}"
        lines.append(line.rstrip())

    return "\n".join(lines)


_SPARK_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def format_sparkline(values: list[float], width: int = 10) -> str:
    """Format a series of values as a sparkline using block characters.

    Scales values to the block character range. Returns a string of
    *width* characters.

    Args:
        values: Numeric values to display.
        width: Output width in characters.

    Returns:
        The sparkline string.
    """
    if not values:
        return ""

    # Resample to width if needed.
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    elif len(values) < width:
        sampled = list(values)
        while len(sampled) < width:
            sampled.append(values[-1])
    else:
        sampled = list(values)

    lo = min(sampled)
    hi = max(sampled)
    span = hi - lo

    result: list[str] = []
    for v in sampled:
        if span == 0:
            idx = len(_SPARK_CHARS) // 2
        else:
            idx = int((v - lo) / span * (len(_SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(_SPARK_CHARS) - 1))
        result.append(_SPARK_CHARS[idx])

    return "".join(result)


def format_section_header(title: str, width: int = 80) -> str:
    """Format a section header: ``'\u2500\u2500\u2500 Title \u2500\u2500...'``."""
    prefix = "\u2500\u2500\u2500 "
    suffix_len = width - len(prefix) - len(title) - 1
    suffix = " " + "\u2500" * max(0, suffix_len)
    return prefix + title + suffix


def truncate(text: str, max_len: int, suffix: str = "...") -> str:
    """Truncate text to *max_len*, adding *suffix* if truncated."""
    if len(text) <= max_len:
        return text
    if max_len <= len(suffix):
        return suffix[:max_len]
    return text[: max_len - len(suffix)] + suffix


def format_signal_name(sig: int | None) -> str:
    """Convert a signal number to its name, e.g. 11 → 'SIGSEGV'.

    Returns ``""`` if *sig* is ``None``.
    """
    if sig is None:
        return ""
    import signal as signal_module

    try:
        return signal_module.Signals(sig).name
    except (ValueError, AttributeError):
        return f"SIG{sig}"


def format_percentage(count: int, total: int) -> str:
    """Format as percentage: ``'44.2%'``. Returns ``'-'`` if *total* is 0."""
    if total == 0:
        return "-"
    return f"{count / total * 100:.1f}%"
