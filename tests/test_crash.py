"""Tests for labeille.crash."""

import signal
import unittest

from labeille.crash import (
    detect_crash,
    extract_crash_signature,
    signal_name,
)


class TestSignalName(unittest.TestCase):
    def test_sigsegv(self) -> None:
        self.assertEqual(signal_name(signal.SIGSEGV), "SIGSEGV")

    def test_sigabrt(self) -> None:
        self.assertEqual(signal_name(signal.SIGABRT), "SIGABRT")

    def test_sigterm(self) -> None:
        self.assertEqual(signal_name(signal.SIGTERM), "SIGTERM")

    def test_unknown_signal(self) -> None:
        self.assertEqual(signal_name(254), "SIG254")


class TestDetectCrash(unittest.TestCase):
    # --- Signal-based detection ---

    def test_sigsegv_negative_exit_code(self) -> None:
        crash = detect_crash(-signal.SIGSEGV, "")
        self.assertIsNotNone(crash)
        assert crash is not None
        self.assertEqual(crash.signal_number, signal.SIGSEGV)
        self.assertEqual(crash.signal_name, "SIGSEGV")

    def test_sigabrt_negative_exit_code(self) -> None:
        crash = detect_crash(-signal.SIGABRT, "")
        self.assertIsNotNone(crash)
        assert crash is not None
        self.assertEqual(crash.signal_number, signal.SIGABRT)
        self.assertEqual(crash.signal_name, "SIGABRT")

    def test_exit_code_134_sigabrt(self) -> None:
        crash = detect_crash(134, "")
        self.assertIsNotNone(crash)
        assert crash is not None
        self.assertEqual(crash.signal_number, signal.SIGABRT)

    def test_exit_code_139_sigsegv(self) -> None:
        crash = detect_crash(139, "")
        self.assertIsNotNone(crash)
        assert crash is not None
        self.assertEqual(crash.signal_number, signal.SIGSEGV)

    # --- Stderr pattern detection ---

    def test_segfault_in_stderr(self) -> None:
        crash = detect_crash(1, "Segmentation fault (core dumped)\n")
        self.assertIsNotNone(crash)
        assert crash is not None
        self.assertEqual(crash.signal_number, signal.SIGSEGV)
        self.assertIn("Segmentation fault", crash.signature)

    def test_fatal_python_error(self) -> None:
        stderr = (
            "Fatal Python error: Segmentation fault\n"
            "\n"
            "Current thread 0x00007f... (most recent call first):\n"
        )
        crash = detect_crash(1, stderr)
        self.assertIsNotNone(crash)
        assert crash is not None
        self.assertIn("Fatal Python error", crash.signature)

    def test_assertion_failure(self) -> None:
        stderr = (
            "python: Python/optimizer_symbols.c:1316: "
            "_Py_uop_frame_new: Assertion `co != NULL' failed.\n"
            "Aborted (core dumped)\n"
        )
        crash = detect_crash(134, stderr)
        self.assertIsNotNone(crash)
        assert crash is not None
        self.assertTrue(crash.is_assertion)
        self.assertIn("Assertion", crash.signature)

    def test_abort_in_stderr(self) -> None:
        crash = detect_crash(1, "Aborted\n")
        self.assertIsNotNone(crash)
        assert crash is not None
        self.assertEqual(crash.signal_number, signal.SIGABRT)

    # --- Non-crash cases ---

    def test_normal_exit_zero(self) -> None:
        crash = detect_crash(0, "")
        self.assertIsNone(crash)

    def test_normal_test_failure(self) -> None:
        stderr = "FAILED tests/test_foo.py::test_bar - AssertionError: 1 != 2\n"
        crash = detect_crash(1, stderr)
        self.assertIsNone(crash)

    def test_abort_error_not_crash(self) -> None:
        """AbortError from test frameworks should NOT trigger crash detection."""
        stderr = "AbortError: The operation was aborted\nFAILED 3 tests\n"
        crash = detect_crash(1, stderr)
        self.assertIsNone(crash)

    def test_exit_code_1_clean_stderr(self) -> None:
        crash = detect_crash(1, "1 failed, 49 passed\n")
        self.assertIsNone(crash)

    def test_exit_code_2_no_crash_patterns(self) -> None:
        crash = detect_crash(2, "ERROR: no tests ran\n")
        self.assertIsNone(crash)

    # --- Combined signal + stderr ---

    def test_sigsegv_with_faulthandler(self) -> None:
        stderr = (
            "Fatal Python error: Segmentation fault\n"
            "\n"
            "Current thread 0x00007f... (most recent call first):\n"
            '  File "test.py", line 42 in foo\n'
        )
        crash = detect_crash(-signal.SIGSEGV, stderr)
        self.assertIsNotNone(crash)
        assert crash is not None
        self.assertEqual(crash.signal_number, signal.SIGSEGV)
        self.assertIn("Fatal Python error", crash.signature)


class TestExtractCrashSignature(unittest.TestCase):
    def test_with_matched_line(self) -> None:
        sig = extract_crash_signature("SIGSEGV", "Segmentation fault (core dumped)")
        self.assertEqual(sig, "SIGSEGV: Segmentation fault (core dumped)")

    def test_signal_only(self) -> None:
        sig = extract_crash_signature("SIGSEGV", "")
        self.assertEqual(sig, "SIGSEGV")

    def test_assertion_line(self) -> None:
        line = "python: file.c:42: func: Assertion `x != NULL' failed."
        sig = extract_crash_signature("SIGABRT", line)
        self.assertIn("SIGABRT", sig)
        self.assertIn("Assertion", sig)


if __name__ == "__main__":
    unittest.main()
