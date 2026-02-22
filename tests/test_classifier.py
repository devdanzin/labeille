"""Tests for labeille.classifier."""

import unittest

from labeille.classifier import classify_from_urls, has_platform_wheel, is_pure_wheel


class TestIsPureWheel(unittest.TestCase):
    def test_py3_none_any(self) -> None:
        self.assertTrue(is_pure_wheel("requests-2.31.0-py3-none-any.whl"))

    def test_py2_py3_none_any(self) -> None:
        self.assertTrue(is_pure_wheel("six-1.16.0-py2.py3-none-any.whl"))

    def test_manylinux_wheel(self) -> None:
        self.assertFalse(is_pure_wheel("numpy-1.26.0-cp312-cp312-manylinux_2_17_x86_64.whl"))

    def test_macosx_wheel(self) -> None:
        self.assertFalse(is_pure_wheel("numpy-1.26.0-cp312-cp312-macosx_11_0_arm64.whl"))

    def test_sdist_not_wheel(self) -> None:
        self.assertFalse(is_pure_wheel("requests-2.31.0.tar.gz"))


class TestHasPlatformWheel(unittest.TestCase):
    def test_manylinux(self) -> None:
        files = ["pkg-1.0-cp312-cp312-manylinux_2_17_x86_64.whl"]
        self.assertTrue(has_platform_wheel(files))

    def test_musllinux(self) -> None:
        files = ["pkg-1.0-cp312-cp312-musllinux_1_1_x86_64.whl"]
        self.assertTrue(has_platform_wheel(files))

    def test_macosx(self) -> None:
        files = ["pkg-1.0-cp312-cp312-macosx_11_0_arm64.whl"]
        self.assertTrue(has_platform_wheel(files))

    def test_win(self) -> None:
        files = ["pkg-1.0-cp312-cp312-win_amd64.whl"]
        self.assertTrue(has_platform_wheel(files))

    def test_pure_only(self) -> None:
        files = ["pkg-1.0-py3-none-any.whl"]
        self.assertFalse(has_platform_wheel(files))

    def test_empty_list(self) -> None:
        self.assertFalse(has_platform_wheel([]))


class TestClassifyFromUrls(unittest.TestCase):
    def test_pure_python(self) -> None:
        urls = [
            {"filename": "requests-2.31.0-py3-none-any.whl"},
            {"filename": "requests-2.31.0.tar.gz"},
        ]
        self.assertEqual(classify_from_urls(urls), "pure")

    def test_extensions_manylinux(self) -> None:
        urls = [
            {"filename": "numpy-1.26.0-cp312-cp312-manylinux_2_17_x86_64.whl"},
            {"filename": "numpy-1.26.0-cp312-cp312-macosx_11_0_arm64.whl"},
            {"filename": "numpy-1.26.0.tar.gz"},
        ]
        self.assertEqual(classify_from_urls(urls), "extensions")

    def test_mixed_pure_and_platform(self) -> None:
        """A package with both pure and platform wheels → extensions."""
        urls = [
            {"filename": "pkg-1.0-py3-none-any.whl"},
            {"filename": "pkg-1.0-cp312-cp312-manylinux_2_17_x86_64.whl"},
        ]
        self.assertEqual(classify_from_urls(urls), "extensions")

    def test_sdist_only(self) -> None:
        urls = [{"filename": "pkg-1.0.tar.gz"}]
        self.assertEqual(classify_from_urls(urls), "unknown")

    def test_empty_urls(self) -> None:
        self.assertEqual(classify_from_urls([]), "unknown")

    def test_no_filename_key(self) -> None:
        urls = [{"url": "https://example.com/pkg-1.0.tar.gz"}]
        self.assertEqual(classify_from_urls(urls), "unknown")


if __name__ == "__main__":
    unittest.main()
