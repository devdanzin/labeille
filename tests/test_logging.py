"""Tests for labeille.logging module."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from labeille.logging import get_logger, setup_logging


class TestSetupLogging(unittest.TestCase):
    """Tests for setup_logging configuration."""

    def setUp(self) -> None:
        # Close and remove any existing handlers to avoid resource leaks.
        logger = logging.getLogger("labeille")
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    def test_default_level_is_info(self) -> None:
        logger = setup_logging()
        self.assertEqual(len(logger.handlers), 1)
        console = logger.handlers[0]
        self.assertEqual(console.level, logging.INFO)

    def test_verbose_sets_debug(self) -> None:
        logger = setup_logging(verbose=True)
        console = logger.handlers[0]
        self.assertEqual(console.level, logging.DEBUG)

    def test_quiet_sets_warning(self) -> None:
        logger = setup_logging(quiet=True)
        console = logger.handlers[0]
        self.assertEqual(console.level, logging.WARNING)

    def test_verbose_overrides_quiet(self) -> None:
        logger = setup_logging(verbose=True, quiet=True)
        console = logger.handlers[0]
        self.assertEqual(console.level, logging.DEBUG)

    def test_log_file_adds_file_handler(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = Path(f.name)
        try:
            logger = setup_logging(log_file=log_path)
            self.assertEqual(len(logger.handlers), 2)
            file_handler = logger.handlers[1]
            self.assertIsInstance(file_handler, logging.FileHandler)
            self.assertEqual(file_handler.level, logging.DEBUG)
        finally:
            log_path.unlink(missing_ok=True)

    def test_repeated_calls_clear_handlers(self) -> None:
        setup_logging()
        setup_logging()
        logger = logging.getLogger("labeille")
        self.assertEqual(len(logger.handlers), 1)

    def test_root_logger_level_is_debug(self) -> None:
        logger = setup_logging()
        self.assertEqual(logger.level, logging.DEBUG)


class TestGetLogger(unittest.TestCase):
    """Tests for get_logger."""

    def test_returns_child_logger(self) -> None:
        child = get_logger("test.module")
        self.assertEqual(child.name, "labeille.test.module")

    def test_child_is_under_labeille_namespace(self) -> None:
        child = get_logger("test.inherit")
        self.assertTrue(child.name.startswith("labeille."))


if __name__ == "__main__":
    unittest.main()
