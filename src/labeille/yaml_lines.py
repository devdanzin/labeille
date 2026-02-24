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
    key_line = lines[start]
    colon_idx = key_line.index(":")
    after_colon = key_line[colon_idx + 1 :].strip()
    is_block = after_colon == "" or after_colon.startswith("#")

    end = start + 1
    if is_block:
        # Block-style value: include indented continuation lines.
        while end < len(lines):
            line = lines[end]
            if line.strip() == "":
                # Blank line might be mid-block or trailing — peek ahead.
                # If the next non-blank line is indented, it's mid-block.
                peek = end + 1
                while peek < len(lines) and lines[peek].strip() == "":
                    peek += 1
                if peek < len(lines) and lines[peek][0] in (" ", "\t"):
                    end = peek  # skip blank lines within block
                    continue
                else:
                    break  # trailing blank line, not part of block
            if line[0] in (" ", "\t"):
                end += 1
                continue
            if line.startswith("- "):
                end += 1
                continue
            break
    # For scalar fields, extent is just the single line.
    # Don't consume trailing blank lines.

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


def set_field_value(lines: list[str], field_name: str, new_value: str) -> list[str]:
    """Replace the value of an existing top-level YAML field.

    Handles scalar fields (``skip: true`` → ``skip: false``), null fields,
    quoted strings, and block-style fields (dicts/lists with indented
    continuation lines).

    Args:
        lines: The file lines.
        field_name: The field to modify.
        new_value: The new value, already formatted via :func:`format_yaml_value`.
            For block-style values (dicts/lists), this should include
            leading newlines for sub-items.

    Returns:
        Modified lines with the field value replaced.

    Raises:
        ValueError: If the field is not found.
    """
    idx = find_field_line(lines, field_name)
    if idx is None:
        raise ValueError(f"Field '{field_name}' not found")

    start, end = find_field_extent(lines, idx)

    # Build replacement lines.
    if new_value.startswith("\n"):
        # Block-style value (dict/list with indented sub-items).
        # Split into key line + continuation lines.
        parts = new_value.split("\n")
        # parts[0] is "" (before leading \n), rest are the sub-items.
        replacement = [f"{field_name}:\n"]
        for part in parts[1:]:
            replacement.append(f"{part}\n")
    else:
        replacement = [f"{field_name}: {new_value}\n"]

    return lines[:start] + replacement + lines[end:]


def format_yaml_value(value: Any, field_type: str) -> str:
    """Format a Python value as a YAML string for inline insertion.

    Handles: ``None``, str, int, bool, list (``[]`` or block),
    dict (``{}`` or block).

    Args:
        value: The Python value to format.
        field_type: One of ``"str"``, ``"int"``, ``"bool"``, ``"list"``, ``"dict"``.

    Returns:
        The YAML text representation.
    """
    if value is None:
        return "null"
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
    """Quote a scalar value for YAML if needed.

    Quotes values that YAML would interpret as non-string types:
    integers, floats, booleans, null, and special float values.
    """
    s = str(value)

    # Empty string must be quoted.
    if not s:
        return '""'

    # YAML boolean/null keywords.
    if s.lower() in ("true", "false", "null", "yes", "no", "on", "off", "~"):
        return f'"{s}"'

    # Strings that YAML would parse as numbers (int or float).
    # This includes: "42", "3.15", "1e10", "0x1F", "0o17", "0b101",
    # ".inf", "-.inf", ".nan"
    try:
        # If Python can parse it as int or float, YAML probably will too.
        if s.isdigit():
            return f'"{s}"'
        float(s)
        return f'"{s}"'
    except ValueError:
        pass

    # Strings starting with 0 followed by digits (octal-like: "0123").
    if len(s) > 1 and s[0] == "0" and s[1:].isdigit():
        return f'"{s}"'

    # YAML special characters that require quoting.
    if any(
        c in s
        for c in (
            ":",
            "#",
            "{",
            "}",
            "[",
            "]",
            ",",
            "&",
            "*",
            "?",
            "|",
            ">",
            "'",
            "!",
            "%",
            "@",
            "`",
        )
    ):
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

    # Allow setting a field to null explicitly.
    if default_str.lower() == "null":
        return None

    if field_type == "bool":
        return default_str.lower() in ("true", "1", "yes")
    if field_type == "int":
        return int(default_str)
    if field_type in ("list", "dict"):
        return json.loads(default_str)
    return default_str
