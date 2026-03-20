"""Tests for labeille.cext_build_cli — CLI for cext-build command."""

from __future__ import annotations

import unittest

from click.testing import CliRunner

from labeille.cli import main


class TestCextBuildCLI(unittest.TestCase):
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["cext-build", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--target-python", result.output)
        self.assertIn("--packages", result.output)
        self.assertIn("--registry-dir", result.output)
        self.assertIn("--bear-path", result.output)
        self.assertIn("compile_commands.json", result.output)

    def test_command_registered(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        self.assertIn("cext-build", result.output)

    def test_requires_target_python(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["cext-build", "--packages", "numpy"])
        self.assertNotEqual(result.exit_code, 0)

    def test_requires_packages_or_registry(self) -> None:
        runner = CliRunner()
        # Use a fake python path that exists.
        result = runner.invoke(main, ["cext-build", "--target-python", "/bin/sh"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("provide", result.output.lower() + (result.output if result.output else ""))
