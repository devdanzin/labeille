"""Tests for labeille.registry_ops — batch operations on the package registry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from labeille.registry_ops import (
    PROTECTED_FIELDS,
    PROTECTED_INDEX_FIELDS,
    DryRunPreview,
    FieldFilter,
    OpResult,
    batch_add_field,
    batch_remove_field,
    batch_rename_field,
    batch_set_field,
    matches,
    parse_where,
    rebuild_index,
    remove_index_field,
    validate_registry,
)


def _write_package(
    registry_dir: Path,
    name: str,
    *,
    extra_fields: dict[str, object] | None = None,
) -> Path:
    """Create a minimal package YAML file in the registry."""
    data: dict[str, object] = {
        "package": name,
        "repo": f"https://github.com/user/{name}",
        "pypi_url": f"https://pypi.org/project/{name}/",
        "extension_type": "pure",
        "python_versions": [],
        "install_method": "pip",
        "install_command": "pip install -e '.[dev]'",
        "test_command": "python -m pytest tests/",
        "test_framework": "pytest",
        "uses_xdist": False,
        "timeout": None,
        "skip": False,
        "skip_reason": None,
        "skip_versions": {},
        "notes": "",
        "enriched": True,
        "clone_depth": None,
        "import_name": None,
    }
    if extra_fields:
        data.update(extra_fields)
    pkg_dir = registry_dir / "packages"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    p = pkg_dir / f"{name}.yaml"
    p.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return p


def _write_index(registry_dir: Path, names: list[str]) -> None:
    """Create a minimal index.yaml."""
    registry_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "last_updated": "2026-02-23T00:00:00",
        "packages": [
            {"name": n, "extension_type": "pure", "enriched": True, "skip": False} for n in names
        ],
    }
    (registry_dir / "index.yaml").write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


class TestParseWhere(unittest.TestCase):
    def test_exact_match(self) -> None:
        f = parse_where("extension_type=pure")
        self.assertEqual(f.field, "extension_type")
        self.assertEqual(f.op, "=")
        self.assertEqual(f.value, "pure")

    def test_substring_match(self) -> None:
        f = parse_where("repo~=github")
        self.assertEqual(f.field, "repo")
        self.assertEqual(f.op, "~=")
        self.assertEqual(f.value, "github")

    def test_bool_true(self) -> None:
        f = parse_where("enriched:true")
        self.assertEqual(f.field, "enriched")
        self.assertEqual(f.op, ":true")

    def test_bool_false(self) -> None:
        f = parse_where("skip:false")
        self.assertEqual(f.field, "skip")
        self.assertEqual(f.op, ":false")

    def test_null_check(self) -> None:
        f = parse_where("timeout:null")
        self.assertEqual(f.field, "timeout")
        self.assertEqual(f.op, ":null")

    def test_notnull_check(self) -> None:
        f = parse_where("repo:notnull")
        self.assertEqual(f.field, "repo")
        self.assertEqual(f.op, ":notnull")

    def test_invalid_expr(self) -> None:
        with self.assertRaises(ValueError):
            parse_where("noop")


class TestMatches(unittest.TestCase):
    def test_exact_match(self) -> None:
        entry = {"extension_type": "pure", "skip": False}
        f = FieldFilter(field="extension_type", op="=", value="pure")
        self.assertTrue(matches(entry, [f]))

    def test_exact_no_match(self) -> None:
        entry = {"extension_type": "extensions"}
        f = FieldFilter(field="extension_type", op="=", value="pure")
        self.assertFalse(matches(entry, [f]))

    def test_substring_match(self) -> None:
        entry = {"repo": "https://github.com/user/pkg"}
        f = FieldFilter(field="repo", op="~=", value="github")
        self.assertTrue(matches(entry, [f]))

    def test_bool_true(self) -> None:
        entry = {"enriched": True}
        f = FieldFilter(field="enriched", op=":true", value=None)
        self.assertTrue(matches(entry, [f]))

    def test_bool_false(self) -> None:
        entry = {"skip": False}
        f = FieldFilter(field="skip", op=":false", value=None)
        self.assertTrue(matches(entry, [f]))

    def test_null_check(self) -> None:
        entry = {"timeout": None}
        f = FieldFilter(field="timeout", op=":null", value=None)
        self.assertTrue(matches(entry, [f]))

    def test_notnull_check(self) -> None:
        entry = {"repo": "https://example.com"}
        f = FieldFilter(field="repo", op=":notnull", value=None)
        self.assertTrue(matches(entry, [f]))

    def test_combined_and(self) -> None:
        entry = {"extension_type": "pure", "enriched": True}
        filters = [
            FieldFilter(field="extension_type", op="=", value="pure"),
            FieldFilter(field="enriched", op=":true", value=None),
        ]
        self.assertTrue(matches(entry, filters))

    def test_combined_and_fails(self) -> None:
        entry = {"extension_type": "pure", "enriched": False}
        filters = [
            FieldFilter(field="extension_type", op="=", value="pure"),
            FieldFilter(field="enriched", op=":true", value=None),
        ]
        self.assertFalse(matches(entry, filters))


class TestBatchAddField(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        for name in ["alpha", "beta", "gamma"]:
            _write_package(self.registry, name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_add_field_basic(self) -> None:
        result = batch_add_field(
            self.registry,
            "priority",
            field_type="int",
            default="5",
            after="skip_reason",
            dry_run=False,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.modified), 3)
        for name in ["alpha", "beta", "gamma"]:
            p = self.registry / "packages" / f"{name}.yaml"
            data = yaml.safe_load(p.read_text())
            self.assertEqual(data["priority"], 5)

    def test_add_field_exists_strict(self) -> None:
        result = batch_add_field(
            self.registry,
            "skip",
            field_type="bool",
            dry_run=False,
            lenient=False,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.errors), 3)
        self.assertEqual(len(result.modified), 0)

    def test_add_field_exists_lenient(self) -> None:
        result = batch_add_field(
            self.registry,
            "skip",
            field_type="bool",
            dry_run=False,
            lenient=True,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.skipped), 3)
        self.assertEqual(len(result.modified), 0)

    def test_add_field_partial_resume(self) -> None:
        # First add to alpha only via --packages
        batch_add_field(
            self.registry,
            "priority",
            field_type="int",
            default="5",
            after="skip_reason",
            packages_list=["alpha"],
            dry_run=False,
        )
        # Now run on all with --lenient; alpha already has it
        result = batch_add_field(
            self.registry,
            "priority",
            field_type="int",
            default="5",
            after="skip_reason",
            dry_run=False,
            lenient=True,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.skipped), 1)  # alpha
        self.assertEqual(len(result.modified), 2)  # beta, gamma

    def test_dry_run_no_writes(self) -> None:
        # Read original content
        original = {}
        for name in ["alpha", "beta", "gamma"]:
            p = self.registry / "packages" / f"{name}.yaml"
            original[name] = p.read_text()

        result = batch_add_field(
            self.registry,
            "priority",
            field_type="int",
            default="5",
            after="skip_reason",
            dry_run=True,
        )
        self.assertIsInstance(result, DryRunPreview)
        self.assertEqual(result.affected_count, 3)

        # Verify nothing was written
        for name in ["alpha", "beta", "gamma"]:
            p = self.registry / "packages" / f"{name}.yaml"
            self.assertEqual(p.read_text(), original[name])

    def test_atomic_write(self) -> None:
        """Verify that os.replace is used for atomic writes."""
        import os

        real_replace = os.replace
        with patch("labeille.registry_ops.os.replace") as mock_replace:
            mock_replace.side_effect = real_replace
            batch_add_field(
                self.registry,
                "priority",
                field_type="int",
                default="5",
                after="skip_reason",
                dry_run=False,
            )
            self.assertEqual(mock_replace.call_count, 3)


class TestBatchRemoveField(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        for name in ["alpha", "beta", "gamma"]:
            _write_package(self.registry, name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_remove_field_basic(self) -> None:
        result = batch_remove_field(
            self.registry,
            "notes",
            dry_run=False,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.modified), 3)
        for name in ["alpha", "beta", "gamma"]:
            p = self.registry / "packages" / f"{name}.yaml"
            data = yaml.safe_load(p.read_text())
            self.assertNotIn("notes", data)

    def test_remove_field_missing_strict(self) -> None:
        result = batch_remove_field(
            self.registry,
            "nonexistent",
            dry_run=False,
            lenient=False,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.errors), 3)

    def test_remove_field_missing_lenient(self) -> None:
        result = batch_remove_field(
            self.registry,
            "nonexistent",
            dry_run=False,
            lenient=True,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.skipped), 3)

    def test_remove_field_protected(self) -> None:
        for field in PROTECTED_FIELDS:
            with self.assertRaises(ValueError, msg=f"Expected ValueError for '{field}'"):
                batch_remove_field(self.registry, field, dry_run=False)


class TestBatchRenameField(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        for name in ["alpha", "beta", "gamma"]:
            _write_package(self.registry, name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_rename_field_basic(self) -> None:
        result = batch_rename_field(
            self.registry,
            "notes",
            "comments",
            dry_run=False,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.modified), 3)
        for name in ["alpha", "beta", "gamma"]:
            p = self.registry / "packages" / f"{name}.yaml"
            data = yaml.safe_load(p.read_text())
            self.assertIn("comments", data)
            self.assertNotIn("notes", data)


class TestBatchSetField(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        _write_package(self.registry, "pure_pkg", extra_fields={"extension_type": "pure"})
        _write_package(self.registry, "ext_pkg", extra_fields={"extension_type": "extensions"})
        _write_package(self.registry, "unknown_pkg", extra_fields={"extension_type": "unknown"})

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_set_field_with_filter(self) -> None:
        filters = [FieldFilter(field="extension_type", op="=", value="extensions")]
        result = batch_set_field(
            self.registry,
            "timeout",
            "600",
            field_type="int",
            filters=filters,
            dry_run=False,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.modified), 1)
        p = self.registry / "packages" / "ext_pkg.yaml"
        data = yaml.safe_load(p.read_text())
        self.assertEqual(data["timeout"], 600)

    def test_set_field_requires_all_or_where(self) -> None:
        # set_field without filters should still work if require_all is True
        result = batch_set_field(
            self.registry,
            "timeout",
            "600",
            field_type="int",
            require_all=True,
            dry_run=False,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.modified), 3)


HAND_CRAFTED_YAML = """\
package: handcrafted
repo: "https://github.com/user/handcrafted"
pypi_url: "https://pypi.org/project/handcrafted/"
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


def _write_handcrafted(registry_dir: Path, name: str, content: str) -> Path:
    """Write a hand-crafted YAML file directly (no yaml.dump)."""
    pkg_dir = registry_dir / "packages"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    p = pkg_dir / f"{name}.yaml"
    p.write_text(content, encoding="utf-8")
    return p


class TestBatchSetFieldFormatting(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_set_field_preserves_formatting(self) -> None:
        _write_handcrafted(self.registry, "handcrafted", HAND_CRAFTED_YAML)
        result = batch_set_field(
            self.registry,
            "timeout",
            "600",
            field_type="int",
            require_all=True,
            dry_run=False,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.modified), 1)
        p = self.registry / "packages" / "handcrafted.yaml"
        after = p.read_text(encoding="utf-8")
        # The timeout line should be updated.
        self.assertIn("timeout: 600\n", after)
        # All other lines should be byte-identical.
        before_lines = HAND_CRAFTED_YAML.splitlines(True)
        after_lines = after.splitlines(True)
        for i, (bl, al) in enumerate(zip(before_lines, after_lines)):
            if bl.startswith("timeout:"):
                continue
            self.assertEqual(bl, al, f"Line {i} differs: {bl!r} vs {al!r}")

    def test_set_field_null_value(self) -> None:
        content = HAND_CRAFTED_YAML.replace("timeout: null", "timeout: 300")
        _write_handcrafted(self.registry, "pkg", content)
        result = batch_set_field(
            self.registry,
            "timeout",
            "null",
            field_type="int",
            require_all=True,
            dry_run=False,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.modified), 1)
        p = self.registry / "packages" / "pkg.yaml"
        after = p.read_text(encoding="utf-8")
        self.assertIn("timeout: null\n", after)

    def test_set_field_preserves_quoted_strings(self) -> None:
        _write_handcrafted(self.registry, "quoted", HAND_CRAFTED_YAML)
        result = batch_set_field(
            self.registry,
            "skip",
            "true",
            field_type="bool",
            require_all=True,
            dry_run=False,
        )
        self.assertIsInstance(result, OpResult)
        p = self.registry / "packages" / "quoted.yaml"
        after = p.read_text(encoding="utf-8")
        self.assertIn("skip: true\n", after)
        # Quoted strings in other fields should be preserved.
        self.assertIn('repo: "https://github.com/user/handcrafted"\n', after)
        self.assertIn("install_command: \"pip install -e '.[dev]'\"\n", after)


class TestValidateRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_validate_clean(self) -> None:
        for name in ["alpha", "beta"]:
            _write_package(self.registry, name)
        issues = validate_registry(self.registry)
        errors = [i for i in issues if i.level == "error"]
        self.assertEqual(len(errors), 0)

    def test_validate_missing_required(self) -> None:
        pkg_dir = self.registry / "packages"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        p = pkg_dir / "broken.yaml"
        p.write_text(yaml.dump({"repo": "https://example.com"}), encoding="utf-8")
        issues = validate_registry(self.registry)
        error_msgs = [i.message for i in issues if i.level == "error"]
        self.assertTrue(any("'package'" in m for m in error_msgs))
        self.assertTrue(any("'enriched'" in m for m in error_msgs))

    def test_validate_float_key_warning(self) -> None:
        pkg_dir = self.registry / "packages"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "package": "floatpkg",
            "enriched": False,
            "skip_versions": {3.15: "broken"},
        }
        p = pkg_dir / "floatpkg.yaml"
        p.write_text(yaml.dump(data), encoding="utf-8")
        issues = validate_registry(self.registry)
        warnings = [i for i in issues if i.level == "warning"]
        self.assertTrue(any("float" in w.message for w in warnings))

    def test_validate_unknown_field(self) -> None:
        _write_package(self.registry, "extra", extra_fields={"experimental_flag": True})
        issues = validate_registry(self.registry)
        warnings = [i for i in issues if i.level == "warning"]
        self.assertTrue(any("unknown field" in w.message for w in warnings))

    def test_validate_unknown_field_strict(self) -> None:
        _write_package(self.registry, "extra", extra_fields={"experimental_flag": True})
        issues = validate_registry(self.registry, strict=True)
        errors = [i for i in issues if i.level == "error"]
        self.assertTrue(any("unknown field" in e.message for e in errors))

    def test_validate_uses_xdist_without_no_xdist(self) -> None:
        """uses_xdist: true without -p no:xdist in test_command → warning."""
        _write_package(
            self.registry,
            "xdistpkg",
            extra_fields={
                "uses_xdist": True,
                "test_command": "python -m pytest tests/",
            },
        )
        issues = validate_registry(self.registry)
        warnings = [i for i in issues if i.level == "warning"]
        xdist_warnings = [w for w in warnings if "uses_xdist" in w.message]
        self.assertTrue(
            any("does not include" in w.message for w in xdist_warnings),
            f"Expected uses_xdist warning, got: {xdist_warnings}",
        )

    def test_validate_no_xdist_flag_without_uses_xdist(self) -> None:
        """-p no:xdist in test_command but uses_xdist: false → warning."""
        _write_package(
            self.registry,
            "noxdistpkg",
            extra_fields={
                "uses_xdist": False,
                "test_command": "python -m pytest -p no:xdist tests/",
            },
        )
        issues = validate_registry(self.registry)
        warnings = [i for i in issues if i.level == "warning"]
        xdist_warnings = [w for w in warnings if "no:xdist" in w.message]
        self.assertTrue(
            any("uses_xdist is false" in w.message for w in xdist_warnings),
            f"Expected reverse xdist warning, got: {xdist_warnings}",
        )

    def test_validate_uses_xdist_with_no_xdist(self) -> None:
        """uses_xdist: true with -p no:xdist → no warning."""
        _write_package(
            self.registry,
            "goodpkg",
            extra_fields={
                "uses_xdist": True,
                "test_command": "python -m pytest -p no:xdist tests/",
            },
        )
        issues = validate_registry(self.registry)
        xdist_issues = [i for i in issues if "xdist" in i.message.lower()]
        self.assertEqual(len(xdist_issues), 0)

    def test_validate_skipped_package_no_xdist_check(self) -> None:
        """Skipped packages don't trigger xdist validation."""
        _write_package(
            self.registry,
            "skippkg",
            extra_fields={
                "skip": True,
                "uses_xdist": True,
                "test_command": "python -m pytest tests/",
            },
        )
        issues = validate_registry(self.registry)
        xdist_issues = [i for i in issues if "xdist" in i.message.lower()]
        self.assertEqual(len(xdist_issues), 0)


class TestFilterExactMatch(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        _write_package(self.registry, "pure_pkg", extra_fields={"extension_type": "pure"})
        _write_package(self.registry, "ext_pkg", extra_fields={"extension_type": "extensions"})

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_filter_exact_match(self) -> None:
        filters = [FieldFilter(field="extension_type", op="=", value="pure")]
        result = batch_add_field(
            self.registry,
            "priority",
            field_type="int",
            default="1",
            after="skip_reason",
            filters=filters,
            dry_run=False,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.modified), 1)

    def test_filter_substring_match(self) -> None:
        filters = [FieldFilter(field="extension_type", op="~=", value="ext")]
        result = batch_add_field(
            self.registry,
            "priority",
            field_type="int",
            default="1",
            after="skip_reason",
            filters=filters,
            dry_run=False,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.modified), 1)

    def test_filter_bool_match(self) -> None:
        filters = [FieldFilter(field="enriched", op=":true", value=None)]
        result = batch_add_field(
            self.registry,
            "priority",
            field_type="int",
            default="1",
            after="skip_reason",
            filters=filters,
            dry_run=False,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.modified), 2)

    def test_filter_null_check(self) -> None:
        filters = [FieldFilter(field="timeout", op=":null", value=None)]
        result = batch_add_field(
            self.registry,
            "priority",
            field_type="int",
            default="1",
            after="skip_reason",
            filters=filters,
            dry_run=False,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.modified), 2)

    def test_filter_notnull_check(self) -> None:
        _write_package(self.registry, "timeout_pkg", extra_fields={"timeout": 300})
        filters = [FieldFilter(field="timeout", op=":notnull", value=None)]
        result = batch_add_field(
            self.registry,
            "priority",
            field_type="int",
            default="1",
            after="skip_reason",
            filters=filters,
            dry_run=False,
        )
        self.assertIsInstance(result, OpResult)
        self.assertEqual(len(result.modified), 1)


class TestRebuildIndex(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        for name in ["alpha", "beta"]:
            _write_package(self.registry, name)
        _write_index(self.registry, ["alpha", "beta"])

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_rebuild_index(self) -> None:
        count = rebuild_index(self.registry)
        self.assertEqual(count, 2)

    def test_rebuild_picks_up_new_packages(self) -> None:
        _write_package(self.registry, "gamma")
        count = rebuild_index(self.registry)
        self.assertEqual(count, 3)


class TestRemoveIndexField(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        for name in ["alpha", "beta"]:
            _write_package(self.registry, name)
        _write_index(self.registry, ["alpha", "beta"])

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_remove_index_field_protected(self) -> None:
        for field in PROTECTED_INDEX_FIELDS:
            with self.assertRaises(ValueError, msg=f"Expected ValueError for '{field}'"):
                remove_index_field(self.registry, field)


if __name__ == "__main__":
    unittest.main()
