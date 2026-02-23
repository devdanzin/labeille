"""Line-level YAML manipulation helpers.

These functions operate on file content as a list of lines, performing
insertions, removals, and renames without round-tripping through PyYAML.
This preserves exact formatting of existing fields.
"""

from __future__ import annotations

import json
import re
from typing import Any


def find_field_line(lines: list[str], field_name: str) -> int | None:
    """Find the line index of a top-level YAML field.

    Returns ``None`` if the field is not found.
    """
    pattern = re.compile(rf"^{re.escape(field_name)}:")
    for i, line in enumerate(lines):
        if pattern.match(line):
            return i
    return None


def find_field_extent(lines: list[str], start: int) -> tuple[int, int]:
    """Find the start and end (exclusive) line indices for a field.

    Includes the key line and any continuation lines (indented sub-values
    for dicts/lists).

    Args:
        lines: The file lines.
        start: The line index of the field key.

    Returns:
        A ``(start, end)`` tuple where ``end`` is exclusive.
    """
    # Check if the key line has a block-style value (value after colon is empty
    # or just whitespace, meaning the actual values are on subsequent lines).
    key_line = lines[start]
    colon_idx = key_line.index(":")
    after_colon = key_line[colon_idx + 1 :].strip()
    is_block = after_colon == "" or after_colon.startswith("#")

    end = start + 1
    while end < len(lines):
        line = lines[end]
        # Blank lines or lines starting with whitespace are continuations
        if line.strip() == "":
            end += 1
            continue
        if line[0] in (" ", "\t"):
            end += 1
            continue
        # Block-style list items at column 0 (e.g. "- value")
        if is_block and line.startswith("- "):
            end += 1
            continue
        # Non-indented, non-blank line is the next field
        break

    # Trim trailing blank lines from the extent
    while end > start + 1 and lines[end - 1].strip() == "":
        end -= 1

    return (start, end)


def insert_field_after(
    lines: list[str],
    after_field: str,
    new_field: str,
    new_value_text: str,
) -> list[str]:
    """Insert a new field after an existing field.

    Args:
        lines: The file lines.
        after_field: The field after which to insert.
        new_field: The new field name.
        new_value_text: The formatted YAML value text (from :func:`format_yaml_value`).

    Returns:
        Modified lines with the new field inserted.

    Raises:
        ValueError: If *after_field* is not found.
    """
    idx = find_field_line(lines, after_field)
    if idx is None:
        raise ValueError(f"Field '{after_field}' not found")

    _, extent_end = find_field_extent(lines, idx)
    new_line = f"{new_field}: {new_value_text}\n"
    result = lines[:extent_end] + [new_line] + lines[extent_end:]
    return result


def remove_field(lines: list[str], field_name: str) -> list[str]:
    """Remove a field and its continuation lines.

    Args:
        lines: The file lines.
        field_name: The field to remove.

    Returns:
        Modified lines with the field removed.

    Raises:
        ValueError: If the field is not found.
    """
    idx = find_field_line(lines, field_name)
    if idx is None:
        raise ValueError(f"Field '{field_name}' not found")

    start, end = find_field_extent(lines, idx)
    return lines[:start] + lines[end:]


def rename_field(lines: list[str], old_name: str, new_name: str) -> list[str]:
    """Rename a field key, preserving its value.

    Args:
        lines: The file lines.
        old_name: The current field name.
        new_name: The new field name.

    Returns:
        Modified lines with the field renamed.

    Raises:
        ValueError: If *old_name* is not found.
    """
    idx = find_field_line(lines, old_name)
    if idx is None:
        raise ValueError(f"Field '{old_name}' not found")

    line = lines[idx]
    # Replace just the key portion (everything before the first colon)
    new_line = re.sub(rf"^{re.escape(old_name)}:", f"{new_name}:", line, count=1)
    result = list(lines)
    result[idx] = new_line
    return result


def format_yaml_value(value: Any, field_type: str) -> str:
    """Format a Python value as a YAML string for inline insertion.

    Handles: str, int, bool, list (``[]`` or block), dict (``{}`` or block).

    Args:
        value: The Python value to format.
        field_type: One of ``"str"``, ``"int"``, ``"bool"``, ``"list"``, ``"dict"``.

    Returns:
        The YAML text representation.
    """
    if field_type == "bool":
        return "true" if value else "false"
    if field_type == "int":
        return str(int(value))
    if field_type == "str":
        s = str(value)
        if s == "":
            return '""'
        # Quote if the value contains special YAML characters
        if any(c in s for c in (":", "#", "{", "}", "[", "]", ",", "&", "*", "?", "|", ">", "'")):
            return f'"{s}"'
        if s.lower() in ("true", "false", "null", "yes", "no", "on", "off"):
            return f'"{s}"'
        return s
    if field_type == "list":
        if not value:
            return "[]"
        # Block style for non-empty lists
        items = [f"\n- {_quote_yaml_scalar(item)}" for item in value]
        return "".join(items)
    if field_type == "dict":
        if not value:
            return "{}"
        # Block style for non-empty dicts
        items = [f"\n  {_quote_yaml_scalar(k)}: {_quote_yaml_scalar(v)}" for k, v in value.items()]
        return "".join(items)
    return str(value)


def _quote_yaml_scalar(value: Any) -> str:
    """Quote a scalar value for YAML if needed."""
    s = str(value)
    # Numeric strings that look like floats should be quoted
    try:
        float(s)
        if "." in s or s.lower() in ("inf", "-inf", "nan"):
            return f'"{s}"'
    except ValueError:
        pass
    if s.lower() in ("true", "false", "null", "yes", "no", "on", "off"):
        return f'"{s}"'
    if any(c in s for c in (":", "#", "{", "}", "[", "]", ",", "&", "*", "?", "|", ">", "'")):
        return f'"{s}"'
    return s


def has_field(lines: list[str], field_name: str) -> bool:
    """Check if a top-level YAML field exists."""
    return find_field_line(lines, field_name) is not None


def parse_default_value(default_str: str | None, field_type: str) -> Any:
    """Parse a default value string according to the field type.

    Args:
        default_str: The raw default value string, or ``None`` for type default.
        field_type: One of ``"str"``, ``"int"``, ``"bool"``, ``"list"``, ``"dict"``.

    Returns:
        The parsed Python value.
    """
    if default_str is None:
        defaults: dict[str, Any] = {
            "str": "",
            "int": 0,
            "bool": False,
            "list": [],
            "dict": {},
        }
        return defaults.get(field_type, "")

    if field_type == "bool":
        return default_str.lower() in ("true", "1", "yes")
    if field_type == "int":
        return int(default_str)
    if field_type in ("list", "dict"):
        return json.loads(default_str)
    return default_str
