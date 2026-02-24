"""Tests for labeille.yaml_lines — line-level YAML manipulation."""

from __future__ import annotations

import unittest

from labeille.yaml_lines import (
    _quote_yaml_scalar,
    find_field_extent,
    find_field_line,
    format_yaml_value,
    has_field,
    insert_field_after,
    parse_default_value,
    remove_field,
    rename_field,
)

# A realistic package YAML for testing.
SAMPLE_YAML = """\
package: testpkg
repo: "https://github.com/user/testpkg"
pypi_url: "https://pypi.org/project/testpkg/"
extension_type: pure
python_versions: []
install_method: pip
install_command: "pip install -e '.[dev]'"
test_command: "python -m pytest tests/"
test_framework: pytest
uses_xdist: false
timeout: null
skip: false
skip_reason: null
skip_versions: {}
notes: ""
enriched: true
clone_depth: null
import_name: null
"""

SAMPLE_WITH_DICT = """\
package: dictpkg
skip: false
skip_reason: null
skip_versions:
  "3.15": "PyO3 not supported"
  "3.14": "broken build"
notes: ""
enriched: true
"""

SAMPLE_WITH_LIST = """\
package: listpkg
python_versions:
- "3.15"
- "3.14"
- "3.13"
extension_type: pure
enriched: true
"""


SAMPLE_WITH_BLANK_LINES = """\
package: blankpkg
skip_reason: null

skip_versions: {}
notes: ""
enriched: true
"""

SAMPLE_BLOCK_WITH_INTERNAL_BLANK = """\
package: blockblank
skip_versions:
  "3.15": "PyO3 not supported"

  "3.14": "broken build"
notes: ""
"""


class TestFindFieldLine(unittest.TestCase):
    def test_finds_existing_field(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        idx = find_field_line(lines, "package")
        self.assertEqual(idx, 0)

    def test_finds_middle_field(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        idx = find_field_line(lines, "skip")
        self.assertEqual(idx, 11)

    def test_returns_none_for_missing(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        idx = find_field_line(lines, "nonexistent")
        self.assertIsNone(idx)

    def test_does_not_match_substring(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        # "skip_reason" contains "skip" but find_field_line("skip") should
        # NOT match line "skip_reason: null"
        idx = find_field_line(lines, "skip")
        self.assertIsNotNone(idx)
        self.assertIn("skip:", lines[idx])
        self.assertNotIn("skip_reason", lines[idx])


class TestFindFieldExtent(unittest.TestCase):
    def test_scalar_field(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        idx = find_field_line(lines, "skip")
        self.assertIsNotNone(idx)
        start, end = find_field_extent(lines, idx)
        self.assertEqual(end - start, 1)

    def test_dict_field(self) -> None:
        lines = SAMPLE_WITH_DICT.splitlines(True)
        idx = find_field_line(lines, "skip_versions")
        self.assertIsNotNone(idx)
        start, end = find_field_extent(lines, idx)
        # skip_versions: + two indented sub-keys = 3 lines
        self.assertEqual(end - start, 3)

    def test_list_field(self) -> None:
        lines = SAMPLE_WITH_LIST.splitlines(True)
        idx = find_field_line(lines, "python_versions")
        self.assertIsNotNone(idx)
        start, end = find_field_extent(lines, idx)
        # python_versions: + three "- ..." items = 4 lines
        self.assertEqual(end - start, 4)

    def test_empty_dict_field(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        idx = find_field_line(lines, "skip_versions")
        self.assertIsNotNone(idx)
        start, end = find_field_extent(lines, idx)
        self.assertEqual(end - start, 1)  # "skip_versions: {}" is one line


class TestInsertFieldAfter(unittest.TestCase):
    def test_insert_after_scalar(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        result = insert_field_after(lines, "skip_reason", "new_field", '"hello"')
        text = "".join(result)
        self.assertIn('new_field: "hello"\n', text)
        # Should appear after skip_reason
        sr_pos = text.index("skip_reason:")
        nf_pos = text.index("new_field:")
        self.assertGreater(nf_pos, sr_pos)

    def test_insert_after_multiline(self) -> None:
        lines = SAMPLE_WITH_DICT.splitlines(True)
        result = insert_field_after(lines, "skip_versions", "new_field", "42")
        text = "".join(result)
        self.assertIn("new_field: 42\n", text)
        # Should appear after the skip_versions block (after "broken build" line)
        broken_pos = text.index("broken build")
        nf_pos = text.index("new_field:")
        self.assertGreater(nf_pos, broken_pos)

    def test_insert_at_end(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        result = insert_field_after(lines, "import_name", "new_field", "false")
        text = "".join(result)
        self.assertIn("new_field: false\n", text)

    def test_raises_if_after_field_missing(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        with self.assertRaises(ValueError):
            insert_field_after(lines, "nonexistent", "new_field", "value")


class TestRemoveField(unittest.TestCase):
    def test_remove_scalar(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        result = remove_field(lines, "skip")
        text = "".join(result)
        self.assertNotIn("skip:", text)
        # skip_reason should still be there
        self.assertIn("skip_reason:", text)

    def test_remove_multiline_dict(self) -> None:
        lines = SAMPLE_WITH_DICT.splitlines(True)
        result = remove_field(lines, "skip_versions")
        text = "".join(result)
        self.assertNotIn("skip_versions:", text)
        self.assertNotIn("PyO3 not supported", text)
        self.assertNotIn("broken build", text)
        # notes: should still be there
        self.assertIn("notes:", text)

    def test_raises_if_missing(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        with self.assertRaises(ValueError):
            remove_field(lines, "nonexistent")


class TestRenameField(unittest.TestCase):
    def test_rename_preserves_value(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        result = rename_field(lines, "skip_reason", "skip_note")
        text = "".join(result)
        self.assertIn("skip_note: null\n", text)
        self.assertNotIn("skip_reason:", text)

    def test_rename_preserves_other_fields(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        before_count = len(lines)
        result = rename_field(lines, "skip_reason", "skip_note")
        self.assertEqual(len(result), before_count)

    def test_raises_if_missing(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        with self.assertRaises(ValueError):
            rename_field(lines, "nonexistent", "new_name")


class TestHasField(unittest.TestCase):
    def test_existing_field(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        self.assertTrue(has_field(lines, "package"))

    def test_missing_field(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        self.assertFalse(has_field(lines, "nonexistent"))


class TestFormatYamlValue(unittest.TestCase):
    def test_empty_string(self) -> None:
        self.assertEqual(format_yaml_value("", "str"), '""')

    def test_simple_string(self) -> None:
        self.assertEqual(format_yaml_value("hello", "str"), "hello")

    def test_string_with_special_chars(self) -> None:
        result = format_yaml_value("key: value", "str")
        self.assertIn('"', result)

    def test_int(self) -> None:
        self.assertEqual(format_yaml_value(42, "int"), "42")

    def test_bool_true(self) -> None:
        self.assertEqual(format_yaml_value(True, "bool"), "true")

    def test_bool_false(self) -> None:
        self.assertEqual(format_yaml_value(False, "bool"), "false")

    def test_empty_list(self) -> None:
        self.assertEqual(format_yaml_value([], "list"), "[]")

    def test_empty_dict(self) -> None:
        self.assertEqual(format_yaml_value({}, "dict"), "{}")

    def test_non_empty_dict(self) -> None:
        result = format_yaml_value({"3.15": "broken"}, "dict")
        self.assertIn("3.15", result)
        self.assertIn("broken", result)


class TestParseDefaultValue(unittest.TestCase):
    def test_str_default(self) -> None:
        self.assertEqual(parse_default_value(None, "str"), "")

    def test_int_default(self) -> None:
        self.assertEqual(parse_default_value(None, "int"), 0)

    def test_bool_default(self) -> None:
        self.assertFalse(parse_default_value(None, "bool"))

    def test_list_default(self) -> None:
        self.assertEqual(parse_default_value(None, "list"), [])

    def test_dict_default(self) -> None:
        self.assertEqual(parse_default_value(None, "dict"), {})

    def test_str_value(self) -> None:
        self.assertEqual(parse_default_value("hello", "str"), "hello")

    def test_int_value(self) -> None:
        self.assertEqual(parse_default_value("42", "int"), 42)

    def test_bool_true(self) -> None:
        self.assertTrue(parse_default_value("true", "bool"))

    def test_bool_false(self) -> None:
        self.assertFalse(parse_default_value("false", "bool"))

    def test_json_dict(self) -> None:
        result = parse_default_value('{"3.15": "broken"}', "dict")
        self.assertEqual(result, {"3.15": "broken"})

    def test_json_list(self) -> None:
        result = parse_default_value('["3.15", "3.14"]', "list")
        self.assertEqual(result, ["3.15", "3.14"])


class TestQuoteYamlScalar(unittest.TestCase):
    def test_quote_bare_integer(self) -> None:
        self.assertEqual(_quote_yaml_scalar("42"), '"42"')

    def test_quote_bare_negative(self) -> None:
        self.assertEqual(_quote_yaml_scalar("-1"), '"-1"')

    def test_quote_float_with_dot(self) -> None:
        self.assertEqual(_quote_yaml_scalar("3.15"), '"3.15"')

    def test_quote_scientific(self) -> None:
        self.assertEqual(_quote_yaml_scalar("1e10"), '"1e10"')

    def test_quote_octal_like(self) -> None:
        self.assertEqual(_quote_yaml_scalar("0123"), '"0123"')

    def test_quote_tilde(self) -> None:
        self.assertEqual(_quote_yaml_scalar("~"), '"~"')

    def test_quote_boolean_yes(self) -> None:
        self.assertEqual(_quote_yaml_scalar("yes"), '"yes"')

    def test_quote_normal_string(self) -> None:
        self.assertEqual(_quote_yaml_scalar("hello"), "hello")

    def test_quote_empty(self) -> None:
        self.assertEqual(_quote_yaml_scalar(""), '""')

    def test_quote_special_chars(self) -> None:
        self.assertEqual(_quote_yaml_scalar("key: value"), '"key: value"')


class TestFindFieldExtentDetailed(unittest.TestCase):
    def test_extent_scalar_no_trailing_blank(self) -> None:
        lines = SAMPLE_YAML.splitlines(True)
        idx = find_field_line(lines, "skip")
        self.assertIsNotNone(idx)
        start, end = find_field_extent(lines, idx)
        self.assertEqual(end - start, 1)

    def test_extent_scalar_with_trailing_blank(self) -> None:
        lines = SAMPLE_WITH_BLANK_LINES.splitlines(True)
        idx = find_field_line(lines, "skip_reason")
        self.assertIsNotNone(idx)
        start, end = find_field_extent(lines, idx)
        # Extent should be just the one line, NOT including the blank line.
        self.assertEqual(end - start, 1)

    def test_extent_block_dict(self) -> None:
        lines = SAMPLE_WITH_DICT.splitlines(True)
        idx = find_field_line(lines, "skip_versions")
        self.assertIsNotNone(idx)
        start, end = find_field_extent(lines, idx)
        self.assertEqual(end - start, 3)

    def test_extent_block_dict_with_internal_blank(self) -> None:
        lines = SAMPLE_BLOCK_WITH_INTERNAL_BLANK.splitlines(True)
        idx = find_field_line(lines, "skip_versions")
        self.assertIsNotNone(idx)
        start, end = find_field_extent(lines, idx)
        # skip_versions: + "3.15" line + blank + "3.14" line = 4 lines
        self.assertEqual(end - start, 4)

    def test_extent_block_dict_trailing_blank(self) -> None:
        yaml_text = 'skip_versions:\n  "3.15": broken\n\nnotes: ""\n'
        lines = yaml_text.splitlines(True)
        start, end = find_field_extent(lines, 0)
        # Extent should be skip_versions: + one entry = 2 lines.
        # The trailing blank line is NOT included.
        self.assertEqual(end - start, 2)

    def test_extent_block_list(self) -> None:
        lines = SAMPLE_WITH_LIST.splitlines(True)
        idx = find_field_line(lines, "python_versions")
        self.assertIsNotNone(idx)
        start, end = find_field_extent(lines, idx)
        self.assertEqual(end - start, 4)


class TestInsertAfterBlankLine(unittest.TestCase):
    def test_insert_after_scalar_before_blank_line(self) -> None:
        lines = SAMPLE_WITH_BLANK_LINES.splitlines(True)
        result = insert_field_after(lines, "skip_reason", "new_field", '"value"')
        text = "".join(result)
        # new_field should be between skip_reason and the blank line.
        result_lines = text.splitlines(True)
        sr_idx = next(i for i, ln in enumerate(result_lines) if ln.startswith("skip_reason:"))
        nf_idx = next(i for i, ln in enumerate(result_lines) if ln.startswith("new_field:"))
        self.assertEqual(nf_idx, sr_idx + 1)
        # The blank line should still be present after new_field.
        self.assertEqual(result_lines[nf_idx + 1].strip(), "")


if __name__ == "__main__":
    unittest.main()
