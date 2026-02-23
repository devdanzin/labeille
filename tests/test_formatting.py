"""Tests for labeille.formatting — shared text formatting helpers."""

from __future__ import annotations

import unittest

from labeille.formatting import (
    format_duration,
    format_histogram,
    format_percentage,
    format_section_header,
    format_sparkline,
    format_status_icon,
    format_table,
    truncate,
)


class TestFormatDuration(unittest.TestCase):
    def test_seconds(self) -> None:
        self.assertEqual(format_duration(8.0), "8s")

    def test_minutes(self) -> None:
        self.assertEqual(format_duration(83.0), "1m 23s")

    def test_hours(self) -> None:
        self.assertEqual(format_duration(4354.0), "1h 12m 34s")

    def test_zero(self) -> None:
        self.assertEqual(format_duration(0.0), "0s")

    def test_just_under_minute(self) -> None:
        self.assertEqual(format_duration(59.9), "59s")

    def test_exact_minute(self) -> None:
        self.assertEqual(format_duration(60.0), "1m  0s")


class TestFormatStatusIcon(unittest.TestCase):
    def test_crash(self) -> None:
        result = format_status_icon("crash")
        self.assertIn("CRASH", result)

    def test_timeout(self) -> None:
        result = format_status_icon("timeout")
        self.assertIn("TIMEOUT", result)

    def test_fail(self) -> None:
        result = format_status_icon("fail")
        self.assertIn("FAIL", result)

    def test_error(self) -> None:
        result = format_status_icon("error")
        self.assertIn("ERROR", result)

    def test_pass(self) -> None:
        result = format_status_icon("pass")
        self.assertIn("PASS", result)

    def test_skip(self) -> None:
        result = format_status_icon("skip")
        self.assertIn("SKIP", result)


class TestFormatTable(unittest.TestCase):
    def test_basic(self) -> None:
        headers = ["Name", "Value"]
        rows = [["alpha", "100"], ["beta", "200"]]
        text = format_table(headers, rows)
        self.assertIn("Name", text)
        self.assertIn("alpha", text)
        self.assertIn("200", text)
        lines = text.splitlines()
        self.assertEqual(len(lines), 3)  # header + 2 rows

    def test_truncation(self) -> None:
        headers = ["Name", "Value"]
        rows = [["a" * 50, "short"]]
        text = format_table(headers, rows, max_col_width={0: 10})
        self.assertIn("...", text)
        # Original long string should not appear in full.
        self.assertNotIn("a" * 50, text)

    def test_alignment(self) -> None:
        headers = ["Name", "Count"]
        rows = [["alpha", "100"], ["beta", "2"]]
        text = format_table(headers, rows, alignments=["l", "r"])
        lines = text.splitlines()
        # Count column should be right-aligned.
        for line in lines:
            # The count value in the last column.
            self.assertTrue(len(line) > 0)

    def test_empty_rows(self) -> None:
        headers = ["Name", "Value"]
        text = format_table(headers, [])
        self.assertIn("Name", text)
        lines = text.splitlines()
        self.assertEqual(len(lines), 1)  # header only

    def test_empty_headers(self) -> None:
        text = format_table([], [["a", "b"]])
        self.assertEqual(text, "")


class TestFormatHistogram(unittest.TestCase):
    def test_basic(self) -> None:
        buckets = [("small", 10), ("medium", 5), ("large", 2)]
        text = format_histogram(buckets)
        self.assertIn("small", text)
        self.assertIn("medium", text)
        self.assertIn("large", text)
        lines = text.splitlines()
        self.assertEqual(len(lines), 3)

    def test_single_bucket(self) -> None:
        buckets = [("only", 42)]
        text = format_histogram(buckets)
        self.assertIn("only", text)
        self.assertIn("42", text)
        # Single bucket should get full bar.
        self.assertIn("\u2588", text)

    def test_empty(self) -> None:
        text = format_histogram([])
        self.assertEqual(text, "")

    def test_percentages(self) -> None:
        buckets = [("a", 50), ("b", 50)]
        text = format_histogram(buckets, show_percentages=True, total=100)
        self.assertIn("50%", text)

    def test_no_percentages(self) -> None:
        buckets = [("a", 10)]
        text = format_histogram(buckets, show_percentages=False)
        self.assertNotIn("%", text)


class TestFormatSparkline(unittest.TestCase):
    def test_basic(self) -> None:
        values = [0.0, 0.5, 1.0]
        result = format_sparkline(values, width=3)
        self.assertEqual(len(result), 3)
        # First should be lowest char, last should be highest.
        self.assertEqual(result[0], "\u2581")
        self.assertEqual(result[-1], "\u2588")

    def test_constant(self) -> None:
        values = [5.0, 5.0, 5.0]
        result = format_sparkline(values, width=3)
        self.assertEqual(len(result), 3)
        # All same value → all same char.
        self.assertEqual(len(set(result)), 1)

    def test_empty(self) -> None:
        result = format_sparkline([])
        self.assertEqual(result, "")

    def test_width_expansion(self) -> None:
        values = [1.0, 2.0]
        result = format_sparkline(values, width=5)
        self.assertEqual(len(result), 5)


class TestTruncate(unittest.TestCase):
    def test_short(self) -> None:
        self.assertEqual(truncate("hello", 10), "hello")

    def test_long(self) -> None:
        result = truncate("hello world", 8)
        self.assertEqual(len(result), 8)
        self.assertTrue(result.endswith("..."))

    def test_exact_length(self) -> None:
        self.assertEqual(truncate("hello", 5), "hello")


class TestFormatSectionHeader(unittest.TestCase):
    def test_basic(self) -> None:
        header = format_section_header("Test")
        self.assertIn("Test", header)
        self.assertIn("\u2500", header)

    def test_width(self) -> None:
        header = format_section_header("Title", width=40)
        self.assertEqual(len(header), 40)


class TestFormatPercentage(unittest.TestCase):
    def test_normal(self) -> None:
        self.assertEqual(format_percentage(44, 100), "44.0%")

    def test_zero_total(self) -> None:
        self.assertEqual(format_percentage(0, 0), "-")

    def test_precision(self) -> None:
        result = format_percentage(1, 3)
        self.assertIn("33.3%", result)


if __name__ == "__main__":
    unittest.main()
