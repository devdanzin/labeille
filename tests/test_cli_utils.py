"""Tests for labeille.cli_utils — shared CLI parsing helpers."""

from __future__ import annotations

import unittest

import click

from labeille.cli_utils import parse_csv_list, parse_env_pairs


class TestParseEnvPairs(unittest.TestCase):
    def test_single_pair(self) -> None:
        result = parse_env_pairs(("FOO=bar",))
        self.assertEqual(result, {"FOO": "bar"})

    def test_multiple_pairs(self) -> None:
        result = parse_env_pairs(("A=1", "B=2", "C=3"))
        self.assertEqual(result, {"A": "1", "B": "2", "C": "3"})

    def test_empty_tuple(self) -> None:
        result = parse_env_pairs(())
        self.assertEqual(result, {})

    def test_equals_in_value(self) -> None:
        result = parse_env_pairs(("FOO=bar=baz",))
        self.assertEqual(result, {"FOO": "bar=baz"})

    def test_empty_value(self) -> None:
        result = parse_env_pairs(("FOO=",))
        self.assertEqual(result, {"FOO": ""})

    def test_missing_equals_raises_usage_error(self) -> None:
        with self.assertRaises(click.UsageError) as ctx:
            parse_env_pairs(("NOEQ",))
        self.assertIn("KEY=VALUE", str(ctx.exception))
        self.assertIn("NOEQ", str(ctx.exception))

    def test_missing_equals_among_valid_raises(self) -> None:
        with self.assertRaises(click.UsageError):
            parse_env_pairs(("A=1", "BAD", "C=3"))

    def test_duplicate_key_last_wins(self) -> None:
        result = parse_env_pairs(("FOO=first", "FOO=second"))
        self.assertEqual(result, {"FOO": "second"})


class TestParseCsvList(unittest.TestCase):
    def test_none_returns_empty(self) -> None:
        self.assertEqual(parse_csv_list(None), [])

    def test_empty_string_returns_empty(self) -> None:
        self.assertEqual(parse_csv_list(""), [])

    def test_single_item(self) -> None:
        self.assertEqual(parse_csv_list("foo"), ["foo"])

    def test_multiple_items(self) -> None:
        self.assertEqual(parse_csv_list("a,b,c"), ["a", "b", "c"])

    def test_strips_whitespace(self) -> None:
        self.assertEqual(parse_csv_list(" a , b , c "), ["a", "b", "c"])

    def test_trailing_comma_ignored(self) -> None:
        self.assertEqual(parse_csv_list("a,b,"), ["a", "b"])

    def test_leading_comma_ignored(self) -> None:
        self.assertEqual(parse_csv_list(",a,b"), ["a", "b"])

    def test_whitespace_only_items_filtered(self) -> None:
        self.assertEqual(parse_csv_list("a, ,b, ,c"), ["a", "b", "c"])

    def test_all_whitespace_returns_empty(self) -> None:
        self.assertEqual(parse_csv_list("  ,  ,  "), [])


if __name__ == "__main__":
    unittest.main()
