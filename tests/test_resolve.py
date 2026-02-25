"""Tests for labeille.resolve."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from labeille.registry import PackageEntry, load_index, load_package, save_package
from labeille.resolve import (
    PackageInput,
    _normalize_github_url,
    extract_repo_url,
    fetch_pypi_metadata,
    merge_inputs,
    read_packages_from_args,
    read_packages_from_file,
    read_packages_from_json,
    resolve_all,
    resolve_package,
)


def _make_pypi_response(
    project_urls: dict[str, str] | None = None,
    urls: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    """Build a minimal PyPI JSON API response for testing."""
    return {
        "info": {
            "name": "testpkg",
            "project_urls": project_urls,
        },
        "urls": urls or [],
    }


def _mock_response(status_code: int = 200, json_data: object = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    return resp


class TestExtractRepoUrl(unittest.TestCase):
    def test_source_key(self) -> None:
        meta = _make_pypi_response(project_urls={"Source": "https://github.com/user/repo"})
        self.assertEqual(extract_repo_url(meta), "https://github.com/user/repo")

    def test_source_code_key(self) -> None:
        meta = _make_pypi_response(project_urls={"Source Code": "https://github.com/user/repo"})
        self.assertEqual(extract_repo_url(meta), "https://github.com/user/repo")

    def test_repository_key(self) -> None:
        meta = _make_pypi_response(project_urls={"Repository": "https://github.com/user/repo"})
        self.assertEqual(extract_repo_url(meta), "https://github.com/user/repo")

    def test_github_key(self) -> None:
        meta = _make_pypi_response(project_urls={"GitHub": "https://github.com/user/repo"})
        self.assertEqual(extract_repo_url(meta), "https://github.com/user/repo")

    def test_code_key(self) -> None:
        meta = _make_pypi_response(project_urls={"Code": "https://github.com/user/repo"})
        self.assertEqual(extract_repo_url(meta), "https://github.com/user/repo")

    def test_case_insensitive(self) -> None:
        meta = _make_pypi_response(project_urls={"SOURCE CODE": "https://github.com/user/repo"})
        self.assertEqual(extract_repo_url(meta), "https://github.com/user/repo")

    def test_homepage_github(self) -> None:
        meta = _make_pypi_response(project_urls={"Homepage": "https://github.com/user/repo"})
        self.assertEqual(extract_repo_url(meta), "https://github.com/user/repo")

    def test_homepage_gitlab(self) -> None:
        meta = _make_pypi_response(project_urls={"Homepage": "https://gitlab.com/user/repo"})
        self.assertEqual(extract_repo_url(meta), "https://gitlab.com/user/repo")

    def test_homepage_non_forge_ignored(self) -> None:
        meta = _make_pypi_response(project_urls={"Homepage": "https://example.com/docs"})
        self.assertIsNone(extract_repo_url(meta))

    def test_no_project_urls(self) -> None:
        meta = _make_pypi_response(project_urls=None)
        self.assertIsNone(extract_repo_url(meta))

    def test_empty_project_urls(self) -> None:
        meta = _make_pypi_response(project_urls={})
        self.assertIsNone(extract_repo_url(meta))

    def test_no_info_key(self) -> None:
        self.assertIsNone(extract_repo_url({}))

    def test_priority_source_over_homepage(self) -> None:
        meta = _make_pypi_response(
            project_urls={
                "Homepage": "https://github.com/user/old",
                "Source": "https://github.com/user/new",
            }
        )
        self.assertEqual(extract_repo_url(meta), "https://github.com/user/new")

    def test_secondary_key_bug_tracker(self) -> None:
        """Bug tracker URL on GitHub is normalised to repo root."""
        meta = _make_pypi_response(
            project_urls={"Bug Tracker": "https://github.com/user/repo/issues"}
        )
        self.assertEqual(extract_repo_url(meta), "https://github.com/user/repo")

    def test_secondary_key_changelog(self) -> None:
        """Changelog URL on GitHub is normalised to repo root."""
        meta = _make_pypi_response(
            project_urls={"Changelog": "https://github.com/user/repo/blob/main/CHANGELOG.md"}
        )
        self.assertEqual(extract_repo_url(meta), "https://github.com/user/repo")

    def test_secondary_key_non_github_ignored(self) -> None:
        """Secondary keys pointing to non-GitHub hosts are ignored."""
        meta = _make_pypi_response(
            project_urls={"Bug Tracker": "https://bugs.example.com/repo/issues"}
        )
        self.assertIsNone(extract_repo_url(meta))

    def test_legacy_home_page_field(self) -> None:
        """Legacy info.home_page field is used as last resort."""
        meta: dict[str, object] = {
            "info": {
                "name": "oldpkg",
                "project_urls": None,
                "home_page": "https://github.com/user/repo",
            },
            "urls": [],
        }
        self.assertEqual(extract_repo_url(meta), "https://github.com/user/repo")

    def test_legacy_download_url_field(self) -> None:
        """Legacy info.download_url field is used as last resort."""
        meta: dict[str, object] = {
            "info": {
                "name": "oldpkg",
                "project_urls": None,
                "download_url": "https://github.com/user/repo/archive/v1.0.tar.gz",
            },
            "urls": [],
        }
        self.assertEqual(extract_repo_url(meta), "https://github.com/user/repo")

    def test_legacy_non_forge_ignored(self) -> None:
        """Legacy fields pointing to non-forge hosts are ignored."""
        meta: dict[str, object] = {
            "info": {
                "name": "oldpkg",
                "project_urls": None,
                "home_page": "https://example.com/docs",
            },
            "urls": [],
        }
        self.assertIsNone(extract_repo_url(meta))


class TestNormalizeGithubUrl(unittest.TestCase):
    def test_plain_repo(self) -> None:
        self.assertEqual(
            _normalize_github_url("https://github.com/user/repo"),
            "https://github.com/user/repo",
        )

    def test_issues_url(self) -> None:
        self.assertEqual(
            _normalize_github_url("https://github.com/user/repo/issues"),
            "https://github.com/user/repo",
        )

    def test_blob_url(self) -> None:
        self.assertEqual(
            _normalize_github_url("https://github.com/user/repo/blob/main/CHANGELOG.md"),
            "https://github.com/user/repo",
        )

    def test_tree_url(self) -> None:
        self.assertEqual(
            _normalize_github_url("https://github.com/user/repo/tree/v2.0"),
            "https://github.com/user/repo",
        )

    def test_wiki_url(self) -> None:
        self.assertEqual(
            _normalize_github_url("https://github.com/user/repo/wiki"),
            "https://github.com/user/repo",
        )

    def test_dot_git_suffix(self) -> None:
        self.assertEqual(
            _normalize_github_url("https://github.com/user/repo.git"),
            "https://github.com/user/repo",
        )

    def test_non_github_returns_none(self) -> None:
        self.assertIsNone(_normalize_github_url("https://gitlab.com/user/repo"))

    def test_orgs_path_returns_none(self) -> None:
        self.assertIsNone(_normalize_github_url("https://github.com/orgs/myorg/repositories"))

    def test_archive_url(self) -> None:
        self.assertEqual(
            _normalize_github_url("https://github.com/user/repo/archive/v1.0.tar.gz"),
            "https://github.com/user/repo",
        )


class TestFetchPypiMetadata(unittest.TestCase):
    @patch("labeille.resolve.requests.get")
    def test_success(self, mock_get: MagicMock) -> None:
        data = _make_pypi_response(project_urls={"Source": "https://github.com/u/r"})
        mock_get.return_value = _mock_response(200, data)
        result = fetch_pypi_metadata("testpkg")
        self.assertIsNotNone(result)
        mock_get.assert_called_once()

    @patch("labeille.resolve.requests.get")
    def test_404(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response(404)
        result = fetch_pypi_metadata("nonexistent")
        self.assertIsNone(result)

    @patch("labeille.resolve.requests.get")
    def test_429_rate_limit(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response(429)
        result = fetch_pypi_metadata("testpkg")
        self.assertIsNone(result)

    @patch("labeille.resolve.requests.get")
    def test_500_server_error(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response(500)
        result = fetch_pypi_metadata("testpkg")
        self.assertIsNone(result)

    @patch("labeille.resolve.requests.get")
    def test_connection_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.ConnectionError("connection refused")
        result = fetch_pypi_metadata("testpkg")
        self.assertIsNone(result)

    @patch("labeille.resolve.requests.get")
    def test_timeout_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.Timeout("timed out")
        result = fetch_pypi_metadata("testpkg")
        self.assertIsNone(result)

    @patch("labeille.resolve.requests.get")
    def test_user_agent_header(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response(200, {})
        fetch_pypi_metadata("testpkg", timeout=5.0)
        _, kwargs = mock_get.call_args
        self.assertIn("User-Agent", kwargs["headers"])
        self.assertIn("labeille/", kwargs["headers"]["User-Agent"])

    def test_with_session(self) -> None:
        """When a session is passed, session.get() is called instead of requests.get()."""
        session = MagicMock(spec=requests.Session)
        session.get.return_value = _mock_response(200, {"info": {}})
        result = fetch_pypi_metadata("testpkg", session=session)
        session.get.assert_called_once()
        self.assertIsNotNone(result)

    @patch("labeille.resolve.requests.get")
    def test_without_session(self, mock_get: MagicMock) -> None:
        """Without a session, requests.get() is called (existing behaviour)."""
        mock_get.return_value = _mock_response(200, {"info": {}})
        fetch_pypi_metadata("testpkg")
        mock_get.assert_called_once()


class TestInputReading(unittest.TestCase):
    def test_read_from_args(self) -> None:
        result = read_packages_from_args(("Requests", "Click"))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "requests")
        self.assertEqual(result[1].name, "click")
        self.assertIsNone(result[0].download_count)

    def test_read_from_args_strips_whitespace(self) -> None:
        result = read_packages_from_args(("  flask  ", ""))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "flask")

    def test_read_from_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("requests\n# comment\n\nclick\nflask\n")
            f.flush()
            result = read_packages_from_file(Path(f.name))
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].name, "requests")
        self.assertEqual(result[1].name, "click")
        self.assertEqual(result[2].name, "flask")

    def test_read_from_json(self) -> None:
        data = {
            "rows": [
                {"project": "boto3", "download_count": 1_000_000},
                {"project": "requests", "download_count": 500_000},
                {"project": "click", "download_count": 200_000},
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = read_packages_from_json(Path(f.name))
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].name, "boto3")
        self.assertEqual(result[0].download_count, 1_000_000)

    def test_read_from_json_with_top(self) -> None:
        data = {
            "rows": [
                {"project": "boto3", "download_count": 1_000_000},
                {"project": "requests", "download_count": 500_000},
                {"project": "click", "download_count": 200_000},
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = read_packages_from_json(Path(f.name), top_n=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].name, "boto3")
        self.assertEqual(result[1].name, "requests")


class TestMergeInputs(unittest.TestCase):
    def test_deduplication(self) -> None:
        a = [PackageInput("requests"), PackageInput("click")]
        b = [PackageInput("requests"), PackageInput("flask")]
        result = merge_inputs(a, b)
        names = {p.name for p in result}
        self.assertEqual(names, {"requests", "click", "flask"})

    def test_prefers_download_count(self) -> None:
        a = [PackageInput("requests")]
        b = [PackageInput("requests", download_count=500_000)]
        result = merge_inputs(a, b)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].download_count, 500_000)


class TestResolvePackage(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        (self.registry / "packages").mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    @patch("labeille.resolve.fetch_pypi_metadata")
    def test_creates_new_package(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = _make_pypi_response(
            project_urls={"Source": "https://github.com/psf/requests"},
            urls=[{"filename": "requests-2.31.0-py3-none-any.whl"}],
        )
        pkg = PackageInput("requests")
        result = resolve_package(pkg, self.registry)
        self.assertEqual(result.action, "created")
        self.assertEqual(result.repo_url, "https://github.com/psf/requests")
        self.assertEqual(result.extension_type, "pure")
        loaded = load_package("requests", self.registry)
        self.assertEqual(loaded.repo, "https://github.com/psf/requests")

    @patch("labeille.resolve.fetch_pypi_metadata")
    def test_updates_non_enriched_package(self, mock_fetch: MagicMock) -> None:
        existing = PackageEntry(
            package="requests",
            repo=None,
            extension_type="unknown",
            test_command="make test",  # manually edited field
        )
        save_package(existing, self.registry)

        mock_fetch.return_value = _make_pypi_response(
            project_urls={"Source": "https://github.com/psf/requests"},
            urls=[{"filename": "requests-2.31.0-py3-none-any.whl"}],
        )
        pkg = PackageInput("requests")
        result = resolve_package(pkg, self.registry)
        self.assertEqual(result.action, "updated")
        loaded = load_package("requests", self.registry)
        self.assertEqual(loaded.repo, "https://github.com/psf/requests")
        self.assertEqual(loaded.test_command, "make test")  # preserved

    @patch("labeille.resolve.fetch_pypi_metadata")
    def test_skips_enriched_package(self, mock_fetch: MagicMock) -> None:
        existing = PackageEntry(
            package="requests",
            repo="https://github.com/psf/requests",
            extension_type="pure",
            enriched=True,
        )
        save_package(existing, self.registry)
        pkg = PackageInput("requests")
        result = resolve_package(pkg, self.registry)
        self.assertEqual(result.action, "skipped_enriched")
        mock_fetch.assert_not_called()

    @patch("labeille.resolve.fetch_pypi_metadata")
    def test_handles_no_repo_url(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = _make_pypi_response(
            project_urls={},
            urls=[{"filename": "pkg-1.0-py3-none-any.whl"}],
        )
        pkg = PackageInput("mystery")
        result = resolve_package(pkg, self.registry)
        self.assertEqual(result.action, "created")
        self.assertIsNone(result.repo_url)
        loaded = load_package("mystery", self.registry)
        self.assertIsNone(loaded.repo)

    @patch("labeille.resolve.fetch_pypi_metadata")
    def test_handles_pypi_failure(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = None
        pkg = PackageInput("broken")
        result = resolve_package(pkg, self.registry)
        self.assertEqual(result.action, "failed")
        self.assertIsNotNone(result.error)

    @patch("labeille.resolve.fetch_pypi_metadata")
    def test_dry_run_no_files(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = _make_pypi_response(
            project_urls={"Source": "https://github.com/user/pkg"},
            urls=[{"filename": "pkg-1.0-py3-none-any.whl"}],
        )
        pkg = PackageInput("newpkg")
        result = resolve_package(pkg, self.registry, dry_run=True)
        self.assertEqual(result.action, "skipped")
        self.assertFalse((self.registry / "packages" / "newpkg.yaml").exists())


class TestResolveAll(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        (self.registry / "packages").mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    @patch("labeille.resolve.fetch_pypi_metadata")
    def test_resolve_multiple_packages(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = [
            _make_pypi_response(
                project_urls={"Source": "https://github.com/psf/requests"},
                urls=[{"filename": "requests-2.31.0-py3-none-any.whl"}],
            ),
            _make_pypi_response(
                project_urls={"Source": "https://github.com/pallets/click"},
                urls=[{"filename": "click-8.0-py3-none-any.whl"}],
            ),
        ]
        packages = [PackageInput("requests", 1_000_000), PackageInput("click", 500_000)]
        results, summary = resolve_all(packages, self.registry)
        self.assertEqual(summary.total, 2)
        self.assertEqual(summary.resolved, 2)
        self.assertEqual(summary.created, 2)
        self.assertEqual(summary.failed, 0)
        # Index should be created.
        index = load_index(self.registry)
        self.assertEqual(len(index.packages), 2)

    @patch("labeille.resolve.fetch_pypi_metadata")
    def test_partial_failure_continues(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = [
            None,  # first fails
            _make_pypi_response(
                project_urls={"Source": "https://github.com/pallets/click"},
                urls=[{"filename": "click-8.0-py3-none-any.whl"}],
            ),
        ]
        packages = [PackageInput("broken"), PackageInput("click")]
        results, summary = resolve_all(packages, self.registry)
        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.resolved, 1)

    @patch("labeille.resolve.fetch_pypi_metadata")
    def test_dry_run_no_index_changes(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = _make_pypi_response(
            project_urls={"Source": "https://github.com/user/pkg"},
            urls=[{"filename": "pkg-1.0-py3-none-any.whl"}],
        )
        packages = [PackageInput("newpkg")]
        _, summary = resolve_all(packages, self.registry, dry_run=True)
        self.assertEqual(summary.skipped, 1)
        # No index file should be created.
        self.assertFalse((self.registry / "index.yaml").exists())

    @patch("labeille.resolve.fetch_pypi_metadata")
    def test_index_preserves_existing_entries(self, mock_fetch: MagicMock) -> None:
        """Existing index entries for packages we didn't resolve are preserved."""
        from labeille.registry import Index, IndexEntry, save_index

        existing_index = Index(packages=[IndexEntry(name="preexisting", download_count=999)])
        save_index(existing_index, self.registry)

        mock_fetch.return_value = _make_pypi_response(
            project_urls={"Source": "https://github.com/user/new"},
            urls=[{"filename": "new-1.0-py3-none-any.whl"}],
        )
        packages = [PackageInput("newpkg", 100)]
        resolve_all(packages, self.registry)

        index = load_index(self.registry)
        names = {e.name for e in index.packages}
        self.assertIn("preexisting", names)
        self.assertIn("newpkg", names)

    @patch("labeille.resolve.fetch_pypi_metadata")
    def test_index_sorted_by_download_count(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = [
            _make_pypi_response(
                project_urls={"Source": "https://github.com/user/small"},
                urls=[{"filename": "small-1.0-py3-none-any.whl"}],
            ),
            _make_pypi_response(
                project_urls={"Source": "https://github.com/user/big"},
                urls=[{"filename": "big-1.0-py3-none-any.whl"}],
            ),
        ]
        packages = [PackageInput("small", 100), PackageInput("big", 10_000)]
        resolve_all(packages, self.registry)
        index = load_index(self.registry)
        self.assertEqual(index.packages[0].name, "big")
        self.assertEqual(index.packages[1].name, "small")


class TestResolveAllParallel(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmpdir.name)
        (self.registry / "packages").mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    @patch("labeille.resolve.fetch_pypi_metadata")
    def test_parallel_resolve_collects_all(self, mock_fetch: MagicMock) -> None:
        """Parallel resolve collects results from all packages."""
        mock_fetch.side_effect = [
            _make_pypi_response(
                project_urls={"Source": "https://github.com/psf/requests"},
                urls=[{"filename": "requests-2.31.0-py3-none-any.whl"}],
            ),
            _make_pypi_response(
                project_urls={"Source": "https://github.com/pallets/click"},
                urls=[{"filename": "click-8.0-py3-none-any.whl"}],
            ),
        ]
        packages = [PackageInput("requests", 1_000_000), PackageInput("click", 500_000)]
        results, summary = resolve_all(packages, self.registry, workers=2)
        self.assertEqual(summary.total, 2)
        self.assertEqual(summary.resolved, 2)
        self.assertEqual(summary.created, 2)
        # Index should be created with both packages.
        index = load_index(self.registry)
        self.assertEqual(len(index.packages), 2)

    @patch("labeille.resolve.fetch_pypi_metadata")
    def test_parallel_partial_failure(self, mock_fetch: MagicMock) -> None:
        """Parallel resolve handles partial failures."""
        mock_fetch.side_effect = [
            None,  # first fails
            _make_pypi_response(
                project_urls={"Source": "https://github.com/pallets/click"},
                urls=[{"filename": "click-8.0-py3-none-any.whl"}],
            ),
        ]
        packages = [PackageInput("broken"), PackageInput("click")]
        results, summary = resolve_all(packages, self.registry, workers=2)
        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.resolved, 1)


if __name__ == "__main__":
    unittest.main()
