"""Tests for labeille.io_utils module."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from labeille.io_utils import (
    append_jsonl,
    atomic_write_text,
    dataclass_from_dict,
    extract_minor_version,
    generate_run_id,
    iter_jsonl,
    load_jsonl,
    load_yaml_strict,
    safe_load_yaml,
    utc_now_iso,
    write_meta_json,
)


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


class TestSafeLoadYaml(unittest.TestCase):
    """Tests for safe_load_yaml."""

    def test_valid_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.yaml"
            p.write_text("key: value\n", encoding="utf-8")
            result = safe_load_yaml(p)
            self.assertEqual(result, {"key": "value"})

    def test_malformed_yaml_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.yaml"
            p.write_text(":\n  - :\n    bad: [unterminated\n", encoding="utf-8")
            result = safe_load_yaml(p)
            self.assertIsNone(result)

    def test_non_dict_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "list.yaml"
            p.write_text("- item1\n- item2\n", encoding="utf-8")
            result = safe_load_yaml(p)
            self.assertIsNone(result)

    def test_empty_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "empty.yaml"
            p.write_text("", encoding="utf-8")
            result = safe_load_yaml(p)
            self.assertIsNone(result)

    def test_scalar_yaml_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scalar.yaml"
            p.write_text("just a string\n", encoding="utf-8")
            result = safe_load_yaml(p)
            self.assertIsNone(result)


class TestLoadJsonFile(unittest.TestCase):
    """Tests for load_json_file."""

    def test_valid_json(self) -> None:
        from labeille.io_utils import load_json_file

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.json"
            p.write_text('{"key": "value"}', encoding="utf-8")
            result = load_json_file(p)
            self.assertEqual(result, {"key": "value"})

    def test_malformed_json_raises_valueerror(self) -> None:
        from labeille.io_utils import load_json_file

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text("{truncated", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_json_file(p)
            self.assertIn("Invalid JSON", str(ctx.exception))

    def test_non_dict_json_raises_valueerror(self) -> None:
        from labeille.io_utils import load_json_file

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "list.json"
            p.write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_json_file(p)
            self.assertIn("Expected JSON object", str(ctx.exception))

    def test_missing_file_raises_oserror(self) -> None:
        from labeille.io_utils import load_json_file

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "missing.json"
            with self.assertRaises(OSError):
                load_json_file(p)


class TestLoadYamlStrict(unittest.TestCase):
    """Tests for load_yaml_strict."""

    def test_valid_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.yaml"
            p.write_text("key: value\ncount: 42\n", encoding="utf-8")
            result = load_yaml_strict(p)
            self.assertEqual(result, {"key": "value", "count": 42})

    def test_malformed_yaml_raises_valueerror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.yaml"
            p.write_text(":\n  - :\n    bad: [unterminated\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_yaml_strict(p)
            self.assertIn("Invalid YAML", str(ctx.exception))

    def test_non_dict_raises_valueerror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "list.yaml"
            p.write_text("- item1\n- item2\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_yaml_strict(p)
            self.assertIn("Expected YAML mapping", str(ctx.exception))

    def test_empty_file_raises_valueerror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "empty.yaml"
            p.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_yaml_strict(p)
            self.assertIn("Expected YAML mapping", str(ctx.exception))

    def test_scalar_raises_valueerror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scalar.yaml"
            p.write_text("just a string\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_yaml_strict(p)
            self.assertIn("Expected YAML mapping", str(ctx.exception))


class TestIterJsonl(unittest.TestCase):
    """Tests for iter_jsonl."""

    def test_valid_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_text(
                '{"name": "a", "val": 1}\n{"name": "b", "val": 2}\n',
                encoding="utf-8",
            )
            results = list(iter_jsonl(p, lambda d: d))
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["name"], "a")
            self.assertEqual(results[1]["name"], "b")

    def test_skips_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_text(
                '{"name": "a"}\n{truncated\n{"name": "b"}\n',
                encoding="utf-8",
            )
            results = list(iter_jsonl(p, lambda d: d))
            self.assertEqual(len(results), 2)

    def test_skips_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_text('{"x": 1}\n\n\n{"x": 2}\n', encoding="utf-8")
            results = list(iter_jsonl(p, lambda d: d))
            self.assertEqual(len(results), 2)

    def test_deserialize_error_skipped(self) -> None:
        def bad_deserialize(d: dict) -> str:  # type: ignore[type-arg]
            return d["missing_key"]

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_text('{"name": "a"}\n', encoding="utf-8")
            results = list(iter_jsonl(p, bad_deserialize))
            self.assertEqual(results, [])

    def test_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "empty.jsonl"
            p.write_text("", encoding="utf-8")
            results = list(iter_jsonl(p, lambda d: d))
            self.assertEqual(results, [])


class TestLoadJsonl(unittest.TestCase):
    """Tests for load_jsonl."""

    def test_returns_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
            results = load_jsonl(p, lambda d: d)
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 2)

    def test_with_custom_deserializer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_text('{"val": 10}\n{"val": 20}\n', encoding="utf-8")
            results = load_jsonl(p, lambda d: d["val"] * 2)
            self.assertEqual(results, [20, 40])


class TestAppendJsonl(unittest.TestCase):
    """Tests for append_jsonl."""

    def test_creates_file_if_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "new.jsonl"
            append_jsonl(p, {"key": "value"})
            self.assertTrue(p.exists())
            content = p.read_text(encoding="utf-8")
            self.assertEqual(json.loads(content.strip()), {"key": "value"})

    def test_appends_to_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "existing.jsonl"
            p.write_text('{"a": 1}\n', encoding="utf-8")
            append_jsonl(p, {"a": 2})
            lines = p.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[1]), {"a": 2})

    def test_each_append_is_one_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "multi.jsonl"
            append_jsonl(p, {"x": 1})
            append_jsonl(p, {"x": 2})
            append_jsonl(p, {"x": 3})
            lines = [ln for ln in p.read_text(encoding="utf-8").split("\n") if ln.strip()]
            self.assertEqual(len(lines), 3)


class TestDataclassFromDict(unittest.TestCase):
    """Tests for dataclass_from_dict."""

    def test_basic_creation(self) -> None:
        @dataclass
        class Point:
            x: int
            y: int

        result = dataclass_from_dict(Point, {"x": 1, "y": 2})
        self.assertEqual(result.x, 1)
        self.assertEqual(result.y, 2)

    def test_ignores_unknown_keys(self) -> None:
        @dataclass
        class Simple:
            name: str

        result = dataclass_from_dict(Simple, {"name": "test", "extra": 42, "other": True})
        self.assertEqual(result.name, "test")

    def test_missing_required_field_raises(self) -> None:
        @dataclass
        class Required:
            name: str
            value: int

        with self.assertRaises(TypeError):
            dataclass_from_dict(Required, {"name": "test"})

    def test_with_defaults(self) -> None:
        @dataclass
        class WithDefault:
            name: str
            count: int = 0

        result = dataclass_from_dict(WithDefault, {"name": "test"})
        self.assertEqual(result.count, 0)


class TestExtractMinorVersion(unittest.TestCase):
    """Tests for extract_minor_version."""

    def test_full_version_string(self) -> None:
        self.assertEqual(extract_minor_version("3.15.0a5+ (heads/main:abc1234)"), "3.15")

    def test_release_version(self) -> None:
        self.assertEqual(extract_minor_version("3.13.2"), "3.13")

    def test_simple_major_minor(self) -> None:
        self.assertEqual(extract_minor_version("3.14"), "3.14")

    def test_alpha_suffix(self) -> None:
        self.assertEqual(extract_minor_version("3.15a1"), "3.15")

    def test_single_component_returns_original(self) -> None:
        self.assertEqual(extract_minor_version("3"), "3")

    def test_non_numeric_major_returns_original(self) -> None:
        self.assertEqual(extract_minor_version("abc.15.0"), "abc.15.0")

    def test_empty_string_returns_original(self) -> None:
        self.assertEqual(extract_minor_version(""), "")

    def test_strips_whitespace(self) -> None:
        self.assertEqual(extract_minor_version("  3.14.1  "), "3.14")


class TestGenerateRunId(unittest.TestCase):
    """Tests for generate_run_id."""

    def test_starts_with_prefix(self) -> None:
        result = generate_run_id("bench")
        self.assertTrue(result.startswith("bench_"))

    def test_format_matches_pattern(self) -> None:
        result = generate_run_id("ft")
        self.assertRegex(result, r"^ft_\d{8}_\d{6}$")

    def test_different_prefix(self) -> None:
        result = generate_run_id("test_run")
        self.assertTrue(result.startswith("test_run_"))

    def test_uses_utc(self) -> None:
        with patch("labeille.io_utils.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 11, 14, 30, 45, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = generate_run_id("run")
            mock_dt.now.assert_called_once_with(timezone.utc)
            self.assertEqual(result, "run_20260311_143045")


class TestWriteMetaJson(unittest.TestCase):
    """Tests for write_meta_json."""

    def test_writes_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "meta.json"
            write_meta_json(p, {"run_id": "test_001", "count": 5})
            content = p.read_text(encoding="utf-8")
            data = json.loads(content)
            self.assertEqual(data["run_id"], "test_001")
            self.assertEqual(data["count"], 5)

    def test_output_is_indented(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "meta.json"
            write_meta_json(p, {"key": "value"})
            content = p.read_text(encoding="utf-8")
            self.assertIn("  ", content)

    def test_output_ends_with_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "meta.json"
            write_meta_json(p, {"key": "value"})
            content = p.read_text(encoding="utf-8")
            self.assertTrue(content.endswith("\n"))

    def test_overwrites_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "meta.json"
            write_meta_json(p, {"old": True})
            write_meta_json(p, {"new": True})
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertNotIn("old", data)
            self.assertIn("new", data)


if __name__ == "__main__":
    unittest.main()
