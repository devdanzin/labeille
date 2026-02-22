"""Tests for labeille.registry."""

import tempfile
import unittest
from pathlib import Path

from labeille.registry import (
    Index,
    IndexEntry,
    PackageEntry,
    load_index,
    load_package,
    package_exists,
    save_index,
    save_package,
    sort_index,
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


if __name__ == "__main__":
    unittest.main()
