"""Structured logging setup for labeille.

This module configures structured logging with support for both human-readable
console output and a file handler that always logs at DEBUG level. It provides
consistent formatting across all labeille modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOGGER_NAME = "labeille"
_DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_CONSOLE_FORMAT = "%(levelname)-8s %(message)s"


def setup_logging(
    *,
    verbose: bool = False,
    quiet: bool = False,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure and return the root labeille logger.

    Sets up a console handler whose level is controlled by *verbose*/*quiet*,
    and an optional file handler that always logs at DEBUG.

    Args:
        verbose: If True, set console log level to DEBUG.
        quiet: If True, set console log level to WARNING. Ignored if *verbose* is True.
        log_file: If provided, add a file handler at DEBUG level to this path.

    Returns:
        The configured root logger for labeille.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers to allow reconfiguration.
    logger.handlers.clear()

    # Console handler.
    console = logging.StreamHandler()
    if verbose:
        console.setLevel(logging.DEBUG)
    elif quiet:
        console.setLevel(logging.WARNING)
    else:
        console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    logger.addHandler(console)

    # File handler (always DEBUG).
    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a named child logger under the labeille namespace.

    Args:
        name: The logger name (will be prefixed with ``labeille.``).

    Returns:
        A configured logger instance.
    """
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
