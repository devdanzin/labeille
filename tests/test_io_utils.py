"""Tests for labeille.io_utils module."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from labeille.io_utils import atomic_write_text, utc_now_iso


class TestAtomicWriteText(unittest.TestCase):
    """Tests for atomic_write_text."""

    def test_writes_content_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "test.txt"
            atomic_write_text(target, "hello world")
            self.assertEqual(target.read_text(encoding="utf-8"), "hello world")

    def test_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "test.txt"
            target.write_text("old content", encoding="utf-8")
            atomic_write_text(target, "new content")
            self.assertEqual(target.read_text(encoding="utf-8"), "new content")

    def test_cleans_up_temp_on_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "test.txt"
            with patch("labeille.io_utils.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    atomic_write_text(target, "content")
            # Target should not exist since replace failed.
            self.assertFalse(target.exists())
            # Temp file should have been cleaned up.
            remaining = list(Path(tmp).glob("*.tmp"))
            self.assertEqual(remaining, [])

    def test_respects_encoding_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "test.txt"
            atomic_write_text(target, "caf\u00e9", encoding="utf-8")
            self.assertEqual(target.read_text(encoding="utf-8"), "caf\u00e9")

    def test_preserves_original_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "test.txt"
            target.write_text("original", encoding="utf-8")
            with patch("labeille.io_utils.os.replace", side_effect=OSError("fail")):
                with self.assertRaises(OSError):
                    atomic_write_text(target, "replacement")
            self.assertEqual(target.read_text(encoding="utf-8"), "original")


class TestUtcNowIso(unittest.TestCase):
    """Tests for utc_now_iso."""

    def test_returns_string(self) -> None:
        result = utc_now_iso()
        self.assertIsInstance(result, str)

    def test_format_matches_iso8601_with_z(self) -> None:
        result = utc_now_iso()
        # Must be YYYY-MM-DDTHH:MM:SSZ (second precision, Z suffix)
        self.assertRegex(result, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_timestamp_is_close_to_now(self) -> None:
        before = datetime.now(timezone.utc)
        result = utc_now_iso()
        after = datetime.now(timezone.utc)
        parsed = datetime.strptime(result, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        self.assertGreaterEqual(parsed, before.replace(microsecond=0))
        # Allow 1 second tolerance for second truncation
        from datetime import timedelta

        self.assertLessEqual(parsed, after + timedelta(seconds=1))

    def test_ends_with_z_not_offset(self) -> None:
        result = utc_now_iso()
        self.assertTrue(result.endswith("Z"))
        self.assertNotIn("+", result)

    def test_no_microseconds(self) -> None:
        result = utc_now_iso()
        # Should not contain a dot (microsecond separator)
        self.assertNotIn(".", result)


if __name__ == "__main__":
    unittest.main()
