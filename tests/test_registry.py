"""Tests for labeille.registry."""

import tempfile
import unittest
from pathlib import Path

from labeille.registry import (
    Index,
    IndexEntry,
    PackageEntry,
    _dict_to_package,
    _package_to_dict,
    load_index,
    load_package,
    package_exists,
    save_index,
    save_package,
    sort_index,
    update_index_from_packages,
)


class TestPackageIO(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        (self.registry / "packages").mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_save_and_load_package(self) -> None:
        entry = PackageEntry(
            package="requests",
            repo="https://github.com/psf/requests",
            pypi_url="https://pypi.org/project/requests/",
            extension_type="pure",
            python_versions=["3.14", "3.13"],
            install_command="pip install -e '.[dev]'",
            test_command="python -m pytest tests/",
        )
        save_package(entry, self.registry)
        loaded = load_package("requests", self.registry)
        self.assertEqual(loaded.package, "requests")
        self.assertEqual(loaded.repo, "https://github.com/psf/requests")
        self.assertEqual(loaded.extension_type, "pure")
        self.assertEqual(loaded.python_versions, ["3.14", "3.13"])
        self.assertEqual(loaded.install_command, "pip install -e '.[dev]'")
        self.assertEqual(loaded.test_command, "python -m pytest tests/")
        self.assertFalse(loaded.enriched)

    def test_package_exists(self) -> None:
        self.assertFalse(package_exists("nonexistent", self.registry))
        entry = PackageEntry(package="click")
        save_package(entry, self.registry)
        self.assertTrue(package_exists("click", self.registry))

    def test_load_nonexistent_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_package("nonexistent", self.registry)

    def test_roundtrip_preserves_all_fields(self) -> None:
        entry = PackageEntry(
            package="mypackage",
            repo="https://github.com/user/mypackage",
            pypi_url="https://pypi.org/project/mypackage/",
            extension_type="extensions",
            python_versions=["3.15"],
            install_method="custom",
            install_command="make install",
            test_command="make test",
            test_framework="unittest",
            uses_xdist=True,
            timeout=300,
            skip=True,
            skip_reason="broken on 3.15",
            notes="Some notes here",
            enriched=True,
        )
        save_package(entry, self.registry)
        loaded = load_package("mypackage", self.registry)
        self.assertEqual(loaded.package, entry.package)
        self.assertEqual(loaded.repo, entry.repo)
        self.assertEqual(loaded.pypi_url, entry.pypi_url)
        self.assertEqual(loaded.extension_type, entry.extension_type)
        self.assertEqual(loaded.python_versions, entry.python_versions)
        self.assertEqual(loaded.install_method, entry.install_method)
        self.assertEqual(loaded.install_command, entry.install_command)
        self.assertEqual(loaded.test_command, entry.test_command)
        self.assertEqual(loaded.test_framework, entry.test_framework)
        self.assertEqual(loaded.uses_xdist, entry.uses_xdist)
        self.assertEqual(loaded.timeout, entry.timeout)
        self.assertEqual(loaded.skip, entry.skip)
        self.assertEqual(loaded.skip_reason, entry.skip_reason)
        self.assertEqual(loaded.notes, entry.notes)
        self.assertEqual(loaded.enriched, entry.enriched)

    def test_save_package_with_none_repo(self) -> None:
        entry = PackageEntry(package="mystery", repo=None)
        save_package(entry, self.registry)
        loaded = load_package("mystery", self.registry)
        self.assertIsNone(loaded.repo)

    def test_clone_depth_roundtrip(self) -> None:
        entry = PackageEntry(package="scm-pkg", clone_depth=50)
        save_package(entry, self.registry)
        loaded = load_package("scm-pkg", self.registry)
        self.assertEqual(loaded.clone_depth, 50)

    def test_clone_depth_none_default(self) -> None:
        entry = PackageEntry(package="default-pkg")
        save_package(entry, self.registry)
        loaded = load_package("default-pkg", self.registry)
        self.assertIsNone(loaded.clone_depth)

    def test_import_name_roundtrip(self) -> None:
        entry = PackageEntry(package="python-dateutil", import_name="dateutil")
        save_package(entry, self.registry)
        loaded = load_package("python-dateutil", self.registry)
        self.assertEqual(loaded.import_name, "dateutil")

    def test_import_name_none_default(self) -> None:
        entry = PackageEntry(package="simple")
        save_package(entry, self.registry)
        loaded = load_package("simple", self.registry)
        self.assertIsNone(loaded.import_name)

    def test_skip_versions_roundtrip(self) -> None:
        entry = PackageEntry(
            package="mypkg",
            skip_versions={"3.15": "PyO3 not supported", "3.14": "build broken"},
        )
        save_package(entry, self.registry)
        loaded = load_package("mypkg", self.registry)
        self.assertEqual(
            loaded.skip_versions, {"3.15": "PyO3 not supported", "3.14": "build broken"}
        )

    def test_skip_versions_empty_default(self) -> None:
        entry = PackageEntry(package="noskips")
        save_package(entry, self.registry)
        loaded = load_package("noskips", self.registry)
        self.assertEqual(loaded.skip_versions, {})

    def test_skip_versions_float_key_coercion(self) -> None:
        """PyYAML parses bare 3.15 as a float; keys should be coerced to strings."""
        import yaml

        p = self.registry / "packages" / "floatpkg.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write YAML with a float key (simulates what PyYAML produces for bare 3.15).
        data = {"package": "floatpkg", "skip_versions": {3.15: "broken", 3.14: "also broken"}}
        p.write_text(yaml.dump(data), encoding="utf-8")
        loaded = load_package("floatpkg", self.registry)
        self.assertIn("3.15", loaded.skip_versions)
        self.assertIn("3.14", loaded.skip_versions)
        self.assertEqual(loaded.skip_versions["3.15"], "broken")

    def test_new_fields_tolerated_when_missing(self) -> None:
        """Old YAML files without new fields load with defaults."""
        import yaml

        # Write a YAML file without clone_depth, import_name, and skip_versions
        p = self.registry / "packages" / "oldpkg.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"package": "oldpkg", "repo": "https://github.com/x/y", "enriched": True}
        p.write_text(yaml.dump(data), encoding="utf-8")
        loaded = load_package("oldpkg", self.registry)
        self.assertEqual(loaded.package, "oldpkg")
        self.assertIsNone(loaded.clone_depth)
        self.assertIsNone(loaded.import_name)
        self.assertEqual(loaded.skip_versions, {})


class TestIndexIO(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        self.registry.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_load_empty_index(self) -> None:
        index = load_index(self.registry)
        self.assertEqual(index.packages, [])
        self.assertEqual(index.last_updated, "")

    def test_save_and_load_index(self) -> None:
        index = Index(
            packages=[
                IndexEntry(name="boto3", download_count=1_000_000),
                IndexEntry(name="requests", download_count=500_000, extension_type="pure"),
            ]
        )
        save_index(index, self.registry)
        loaded = load_index(self.registry)
        self.assertEqual(len(loaded.packages), 2)
        self.assertEqual(loaded.packages[0].name, "boto3")
        self.assertEqual(loaded.packages[0].download_count, 1_000_000)
        self.assertEqual(loaded.packages[1].name, "requests")
        self.assertEqual(loaded.packages[1].extension_type, "pure")
        self.assertNotEqual(loaded.last_updated, "")

    def test_index_roundtrip_preserves_data(self) -> None:
        index = Index(
            packages=[
                IndexEntry(
                    name="pkg1",
                    download_count=100,
                    extension_type="pure",
                    enriched=True,
                    skip=False,
                ),
                IndexEntry(
                    name="pkg2",
                    download_count=50,
                    extension_type="extensions",
                    enriched=False,
                    skip=True,
                ),
            ]
        )
        save_index(index, self.registry)
        loaded = load_index(self.registry)
        self.assertEqual(len(loaded.packages), 2)
        for orig, got in zip(index.packages, loaded.packages):
            self.assertEqual(got.name, orig.name)
            self.assertEqual(got.download_count, orig.download_count)
            self.assertEqual(got.extension_type, orig.extension_type)
            self.assertEqual(got.enriched, orig.enriched)
            self.assertEqual(got.skip, orig.skip)


class TestIndexSorting(unittest.TestCase):
    def test_sort_by_download_count_descending(self) -> None:
        index = Index(
            packages=[
                IndexEntry(name="small", download_count=10),
                IndexEntry(name="big", download_count=1000),
                IndexEntry(name="medium", download_count=100),
            ]
        )
        sort_index(index)
        names = [e.name for e in index.packages]
        self.assertEqual(names, ["big", "medium", "small"])

    def test_nulls_last(self) -> None:
        index = Index(
            packages=[
                IndexEntry(name="unknown", download_count=None),
                IndexEntry(name="known", download_count=500),
            ]
        )
        sort_index(index)
        self.assertEqual(index.packages[0].name, "known")
        self.assertEqual(index.packages[1].name, "unknown")

    def test_nulls_sorted_by_name(self) -> None:
        index = Index(
            packages=[
                IndexEntry(name="zebra", download_count=None),
                IndexEntry(name="alpha", download_count=None),
            ]
        )
        sort_index(index)
        self.assertEqual(index.packages[0].name, "alpha")
        self.assertEqual(index.packages[1].name, "zebra")


class TestIndexSkipVersionsKeys(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        self.registry.mkdir(parents=True, exist_ok=True)
        (self.registry / "packages").mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_index_entry_skip_versions_keys_roundtrip(self) -> None:
        """IndexEntry with skip_versions_keys survives save/load round-trip."""
        index = Index(
            packages=[
                IndexEntry(
                    name="mypkg",
                    download_count=100,
                    skip_versions_keys=["3.14", "3.15"],
                ),
            ]
        )
        save_index(index, self.registry)
        loaded = load_index(self.registry)
        self.assertEqual(loaded.packages[0].skip_versions_keys, ["3.14", "3.15"])

    def test_update_index_from_packages_syncs_skip_versions(self) -> None:
        """update_index_from_packages copies skip_versions keys to index."""
        pkg = PackageEntry(
            package="verpkg",
            skip_versions={"3.15": "PyO3 not supported", "3.14": "build broken"},
        )
        save_package(pkg, self.registry)
        index = Index(packages=[IndexEntry(name="verpkg", download_count=100)])
        update_index_from_packages(index, self.registry)
        self.assertEqual(index.packages[0].skip_versions_keys, ["3.14", "3.15"])

    def test_load_index_tolerates_missing_skip_versions_keys(self) -> None:
        """Index YAML without skip_versions_keys loads with default []."""
        import yaml

        data = {
            "last_updated": "2026-02-24T00:00:00",
            "packages": [
                {"name": "oldpkg", "download_count": 50, "enriched": True, "skip": False}
            ],
        }
        (self.registry / "index.yaml").write_text(
            yaml.dump(data, default_flow_style=False), encoding="utf-8"
        )
        loaded = load_index(self.registry)
        self.assertEqual(loaded.packages[0].skip_versions_keys, [])

    def test_index_entry_default_empty_list(self) -> None:
        """IndexEntry defaults to empty skip_versions_keys."""
        entry = IndexEntry(name="pkg")
        self.assertEqual(entry.skip_versions_keys, [])


class TestDictToPackageCoercion(unittest.TestCase):
    def test_coerces_null_notes(self) -> None:
        """notes: null in YAML → empty string."""
        data = {"package": "nullnotes", "notes": None, "enriched": False}
        pkg = _dict_to_package(data)
        self.assertEqual(pkg.notes, "")
        self.assertIsInstance(pkg.notes, str)

    def test_empty_notes_preserved(self) -> None:
        """notes: '' stays as empty string."""
        data = {"package": "emptynotes", "notes": "", "enriched": False}
        pkg = _dict_to_package(data)
        self.assertEqual(pkg.notes, "")

    def test_notes_string_preserved(self) -> None:
        """notes with actual content stays unchanged."""
        data = {"package": "pkg", "notes": "some note", "enriched": False}
        pkg = _dict_to_package(data)
        self.assertEqual(pkg.notes, "some note")

    def test_unknown_keys_logged(self) -> None:
        """Unknown fields produce debug log messages."""
        data = {
            "package": "badpkg",
            "enriched": False,
            "skip_reaason": "typo field",
            "experimental": True,
        }
        with self.assertLogs("labeille.registry", level="DEBUG") as cm:
            pkg = _dict_to_package(data)
        self.assertEqual(pkg.package, "badpkg")
        log_output = "\n".join(cm.output)
        self.assertIn("unknown fields", log_output)
        self.assertIn("skip_reaason", log_output)
        self.assertIn("experimental", log_output)

    def test_no_log_for_known_keys(self) -> None:
        """No debug log when all keys are known."""
        import logging

        data = {"package": "goodpkg", "enriched": True, "notes": "ok"}
        logger = logging.getLogger("labeille.registry")
        with self.assertNoLogs(logger, level="DEBUG"):
            _dict_to_package(data)


class TestPackageToDictOmitDefaults(unittest.TestCase):
    def test_omit_defaults_removes_default_fields(self) -> None:
        """omit_defaults=True removes fields matching PackageEntry defaults."""
        entry = PackageEntry(
            package="mypkg",
            repo="https://github.com/user/mypkg",
            enriched=True,
            timeout=300,
        )
        data = _package_to_dict(entry, omit_defaults=True)
        # Non-default fields should be present.
        self.assertEqual(data["package"], "mypkg")
        self.assertEqual(data["repo"], "https://github.com/user/mypkg")
        self.assertTrue(data["enriched"])
        self.assertEqual(data["timeout"], 300)
        # Default fields should be absent.
        self.assertNotIn("notes", data)
        self.assertNotIn("clone_depth", data)
        self.assertNotIn("import_name", data)
        self.assertNotIn("uses_xdist", data)

    def test_omit_defaults_false_includes_all(self) -> None:
        """omit_defaults=False (default) includes all fields."""
        entry = PackageEntry(package="mypkg")
        data = _package_to_dict(entry, omit_defaults=False)
        self.assertIn("notes", data)
        self.assertIn("clone_depth", data)
        self.assertIn("timeout", data)
        self.assertIn("uses_xdist", data)

    def test_omit_defaults_always_includes_package(self) -> None:
        """package field is always included even with omit_defaults."""
        entry = PackageEntry(package="mypkg")
        data = _package_to_dict(entry, omit_defaults=True)
        self.assertIn("package", data)
        self.assertEqual(data["package"], "mypkg")

    def test_default_parameter_is_false(self) -> None:
        """Default call without omit_defaults includes everything."""
        entry = PackageEntry(package="mypkg")
        data = _package_to_dict(entry)
        self.assertIn("notes", data)
        self.assertIn("clone_depth", data)

    def test_skip_versions_keys_coerced_with_omit_defaults(self) -> None:
        """skip_versions keys are coerced to strings even with omit_defaults."""
        entry = PackageEntry(
            package="mypkg",
            skip_versions={"3.15": "broken"},
        )
        data = _package_to_dict(entry, omit_defaults=True)
        self.assertIn("skip_versions", data)
        self.assertIn("3.15", data["skip_versions"])
        self.assertIsInstance(list(data["skip_versions"].keys())[0], str)


if __name__ == "__main__":
    unittest.main()
