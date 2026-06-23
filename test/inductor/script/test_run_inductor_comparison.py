"""
Unit tests for the comparison harness helpers.

Tests pure functions and the new _save_results() and _load_results() functions.
Runs standalone -- no torch_npu needed.
"""

import subprocess
import sys
import textwrap
from pathlib import Path
import signal
from unittest.mock import MagicMock, call, patch

import pytest

# Import functions directly from the harness script -- no torch_npu required.
# Add the scripts directory to sys.path so we can import the sibling module.
_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir))

import run_inductor_comparison

from run_inductor_comparison import (
    _classify_status,
    _load_results,
    _parse_test_status,
    _save_results,
    _sanitize_test_name,
    _worker_run_tests,
    extract_generated_code_from_cache,
    generate_standalone_report,
    graceful_kill_group,
    run_single_test,
    worker_debug_dir,
)

# ── _classify_status() ───────────────────────────────────────────────────────


class TestClassifyStatus:
    """Test all status combinations for _classify_status()."""

    # -- Both passed --
    def test_both_passed(self):
        assert _classify_status("passed", "passed") == "OK"

    # -- Old passed, new fails --
    def test_regression_failed(self):
        assert _classify_status("passed", "failed") == "REGRESSION"

    def test_regression_error(self):
        assert _classify_status("passed", "error") == "REGRESSION"

    # -- Old passed, new times out --
    def test_timeout_new(self):
        assert _classify_status("passed", "timeout") == "TIMEOUT_NEW"

    # -- Old passed, new skipped --
    def test_skipped_new(self):
        assert _classify_status("passed", "skipped") == "SKIPPED_NEW"

    # -- Old failed, new passed --
    def test_improved_from_failed(self):
        assert _classify_status("failed", "passed") == "IMPROVED"

    def test_improved_from_error(self):
        assert _classify_status("error", "passed") == "IMPROVED"

    # -- Old timeout, new passed --
    def test_timeout_old(self):
        assert _classify_status("timeout", "passed") == "TIMEOUT_OLD"

    # -- Old skipped, new passed --
    def test_skipped_old(self):
        assert _classify_status("skipped", "passed") == "SKIPPED_OLD"

    # -- Both failed --
    def test_both_failed(self):
        assert _classify_status("failed", "failed") == "BOTH_FAILED"

    def test_both_error(self):
        assert _classify_status("error", "error") == "BOTH_FAILED"

    def test_failed_and_error(self):
        assert _classify_status("failed", "error") == "BOTH_FAILED"

    def test_error_and_failed(self):
        assert _classify_status("error", "failed") == "BOTH_FAILED"

    # -- Both timeout --
    def test_both_timeout(self):
        assert _classify_status("timeout", "timeout") == "BOTH_TIMEOUT"

    # -- Old skipped, new is not passed --
    def test_skipped_old_failed(self):
        assert _classify_status("skipped", "failed") == "SKIPPED_OLD"

    def test_skipped_old_timeout(self):
        assert _classify_status("skipped", "timeout") == "SKIPPED_OLD"

    def test_both_skipped(self):
        assert _classify_status("skipped", "skipped") == "BOTH_SKIPPED"

    # -- New skipped, old is not passed --
    def test_skipped_new_failed(self):
        assert _classify_status("failed", "skipped") == "SKIPPED_NEW"

    def test_skipped_new_timeout(self):
        assert _classify_status("timeout", "skipped") == "SKIPPED_NEW"

    # -- Mixed / fallback --
    def test_mixed_timeout_and_failed(self):
        assert _classify_status("timeout", "failed") == "MISMATCH"

    def test_mixed_timeout_and_error(self):
        assert _classify_status("error", "timeout") == "MISMATCH"

    # -- Unknown status falls through to MISMATCH --
    def test_unknown_status(self):
        assert _classify_status("unknown", "unknown") == "MISMATCH"


# ── _sanitize_test_name() ────────────────────────────────────────────────────


class TestSanitizeTestName:
    """Test _sanitize_test_name() string transformation."""

    def test_simple_name(self):
        assert _sanitize_test_name("test/inductor/test_torchinductor.py::TestSuite::test_add") == "test_add"

    def test_with_brackets(self):
        assert _sanitize_test_name(
            "test/inductor/test_torchinductor.py::TestSuite::test_add[foo-bar]"
        ) == "test_add_foo_bar"

    def test_with_parens(self):
        assert _sanitize_test_name(
            "test/inductor/test_torchinductor.py::TestSuite::test_func(param1)"
        ) == "test_func_param1"

    def test_with_angle_brackets(self):
        assert _sanitize_test_name(
            "test/inductor/test_torchinductor.py::TestSuite::test_lt<int>"
        ) == "test_lt_int"

    def test_multiple_special_chars(self):
        assert _sanitize_test_name(
            "test/inductor/test_torchinductor.py::TestSuite::test_x[a-b](c<d>)"
        ) == "test_x_a_b_c_d"

    def test_underscores_collapsed(self):
        assert _sanitize_test_name(
            "test/inductor/test_torchinductor.py::TestSuite::test___double___under"
        ) == "test_double_under"

    def test_leading_trailing_underscores_stripped(self):
        assert _sanitize_test_name(
            "test/inductor/test_torchinductor.py::TestSuite::test_[foo]"
        ) == "test_foo"

    def test_only_special_chars(self):
        assert _sanitize_test_name(
            "test/inductor/test_torchinductor.py::TestSuite::test_[]()"
        ) == "test"

    def test_no_path_prefix(self):
        assert _sanitize_test_name("test_conv2d") == "test_conv2d"


# ── _parse_test_status() ─────────────────────────────────────────────────────


class TestParseTestStatus:
    """Test _parse_test_status() parses pytest -v stdout."""

    def test_passed(self):
        stdout = "test/inductor/test_torchinductor.py::TestA::test_add PASSED [ 50%]\n"
        assert _parse_test_status(stdout) == "passed"

    def test_failed(self):
        stdout = "test/inductor/test_torchinductor.py::TestA::test_sub FAILED [100%]\n"
        assert _parse_test_status(stdout) == "failed"

    def test_skipped(self):
        stdout = "test/inductor/test_torchinductor.py::TestA::test_skip SKIPPED [100%]\n"
        assert _parse_test_status(stdout) == "skipped"

    def test_error(self):
        stdout = "test/inductor/test_torchinductor.py::TestA::test_timeout ERROR [100%]\n"
        assert _parse_test_status(stdout) == "failed"

    def test_empty_stdout(self):
        assert _parse_test_status("") == "error"

    def test_none_stdout(self):
        assert _parse_test_status(None) == "error"

    def test_no_marker(self):
        stdout = "some random output without markers\n"
        assert _parse_test_status(stdout) == "error"

    def test_passed_with_trailing_output(self):
        stdout = "=========================== short test summary info ============================\nPASSED [100%]\n"
        assert _parse_test_status(stdout) == "passed"

    def test_failed_in_long_output(self):
        stdout = "================ test session starts =================\nplatform linux -- Python 3.11.0\ncollected 1 item\n\ntest_torchinductor.py::TestA::test_x FAILED\n\n================================ FAILURES =================================\n___________________________ TestA.test_x ___________________________\nAssertionError: assert False\n\n=========================== short test summary info ============================\nFAILED test_torchinductor.py::TestA::test_x\n======================= 1 failed in 0.5s =======================\n"
        assert _parse_test_status(stdout) == "failed"

    def test_returncode_fallback_passed(self):
        assert _parse_test_status("no markers here", returncode=0) == "passed"

    def test_returncode_fallback_failed(self):
        assert _parse_test_status("no markers here", returncode=1) == "failed"

    def test_returncode_fallback_skipped(self):
        assert _parse_test_status("no markers here", returncode=5) == "skipped"

    def test_returncode_fallback_none(self):
        assert _parse_test_status("no markers here", returncode=None) == "error"

    def test_stdout_takes_precedence_over_returncode(self):
        assert _parse_test_status("test_torchinductor.py::TestA::test_x PASSED [100%]", returncode=1) == "passed"

    def test_no_marker_in_traceback(self):
        stdout = "AssertionError: assert FAILED in test message\n"
        assert _parse_test_status(stdout) == "error"

    def test_no_marker_in_error_log(self):
        stdout = "ERROR: could not open file\n"
        assert _parse_test_status(stdout) == "error"


# ── _save_results() / _load_results() ────────────────────────────────────────


class TestSaveLoadResults:
    """Test _save_results() and _load_results() round-trip."""

    def _sample_results(self):
        return {
            "path/to/test.py::TestA::test_add": {"status": "passed", "elapsed": 0.5},
            "path/to/test.py::TestA::test_sub": {"status": "failed", "elapsed": 1.2},
            "path/to/test.py::TestA::test_mul": {"status": "skipped", "elapsed": 0.0},
            "path/to/test.py::TestA::test_div": {"status": "timeout", "elapsed": 300.0},
        }

    def test_save_results_creates_file(self, tmp_path):
        out = tmp_path / "results.txt"
        _save_results(self._sample_results(), out)
        assert out.exists()
        content = out.read_text()
        assert "test_add" in content
        assert "test_sub" in content
        assert "passed" in content
        assert "failed" in content

    def test_save_results_line_count(self, tmp_path):
        out = tmp_path / "results.txt"
        _save_results(self._sample_results(), out)
        lines = [l for l in out.read_text().strip().split("\n") if l.strip()]
        assert len(lines) == 4

    def test_save_load_roundtrip(self, tmp_path):
        backend_dir = tmp_path / "old_backend"
        backend_dir.mkdir()
        results_file = backend_dir / "results.txt"
        results = self._sample_results()
        _save_results(results, results_file)
        loaded = _load_results(str(tmp_path), "old")
        assert loaded == results

    def test_empty_results(self, tmp_path):
        out = tmp_path / "results.txt"
        _save_results({}, out)
        assert out.read_text().strip() == ""

    def test_load_results_old_backend(self, tmp_path):
        backend_dir = tmp_path / "old_backend"
        backend_dir.mkdir()
        results_file = backend_dir / "results.txt"
        results_file.write_text("test_a passed 0.5\ntest_b failed 1.0\ntest_c skipped 0.0\n")
        results = _load_results(str(tmp_path), "old")
        assert len(results) == 3
        assert results["test_a"] == {"status": "passed", "elapsed": 0.5}
        assert results["test_b"] == {"status": "failed", "elapsed": 1.0}
        assert results["test_c"] == {"status": "skipped", "elapsed": 0.0}

    def test_load_results_new_backend(self, tmp_path):
        backend_dir = tmp_path / "new_backend"
        backend_dir.mkdir()
        results_file = backend_dir / "results.txt"
        results_file.write_text("test_x passed 0.3\ntest_y timeout 300.0\n")
        results = _load_results(str(tmp_path), "new")
        assert len(results) == 2
        assert results["test_x"] == {"status": "passed", "elapsed": 0.3}
        assert results["test_y"] == {"status": "timeout", "elapsed": 300.0}

    def test_load_returns_empty_when_file_missing(self, tmp_path, capsys):
        result = _load_results(str(tmp_path), "old")
        assert result == {}
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_load_returns_empty_when_dir_missing(self, tmp_path, capsys):
        result = _load_results(str(tmp_path / "nonexistent"), "old")
        assert result == {}
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_load_elapsed_rounding(self, tmp_path):
        backend_dir = tmp_path / "old_backend"
        backend_dir.mkdir()
        (backend_dir / "results.txt").write_text("test_r passed 1.234567\n")
        results = _load_results(str(tmp_path), "old")
        assert results["test_r"]["elapsed"] == pytest.approx(1.23)


# ── generate_standalone_report() ─────────────────────────────────────────────


class TestGenerateStandaloneReport:
    """Test generate_standalone_report() produces correct output."""

    def _sample_results(self):
        return {
            "test_a.py::TestA::test_add": {"status": "passed", "elapsed": 0.5},
            "test_a.py::TestA::test_sub": {"status": "failed", "elapsed": 1.2},
            "test_b.py::TestB::test_mul": {"status": "timeout", "elapsed": 300.0},
            "test_c.py::TestC::test_div": {"status": "skipped", "elapsed": 0.0},
        }

    def test_generates_report_file(self, tmp_path):
        results = self._sample_results()
        report = generate_standalone_report(results, tmp_path)
        report_path = tmp_path / "comparison_report.txt"
        assert report_path.exists()
        assert report_path.read_text() == report

    def test_report_contains_header(self, tmp_path):
        report = generate_standalone_report(self._sample_results(), tmp_path)
        assert "Standalone Report" in report
        assert "Test" in report
        assert "Status" in report
        assert "Elapsed" in report

    def test_report_contains_summary(self, tmp_path):
        report = generate_standalone_report(self._sample_results(), tmp_path)
        assert "Total tests:      4" in report
        assert "passed:           1" in report
        assert "failed:           1" in report
        assert "timeout:          1" in report
        assert "skipped:          1" in report
        assert "error:            0" in report

    def test_report_with_backend_label(self, tmp_path):
        report = generate_standalone_report(self._sample_results(), tmp_path, "new")
        assert "new_backend" in report
        assert "TORCHINDUCTOR_NPU_BACKEND=new" in report

    def test_report_without_backend_label(self, tmp_path):
        report = generate_standalone_report(self._sample_results(), tmp_path)
        assert "Backend:" not in report

    def test_report_contains_all_tests(self, tmp_path):
        report = generate_standalone_report(self._sample_results(), tmp_path)
        assert "test_add" in report
        assert "test_sub" in report
        assert "test_mul" in report
        assert "test_div" in report

    def test_empty_results(self, tmp_path):
        report = generate_standalone_report({}, tmp_path)
        assert "Total tests:      0" in report
        assert "passed:           0" in report

    def test_prints_to_stdout(self, tmp_path, capsys):
        generate_standalone_report(self._sample_results(), tmp_path)
        captured = capsys.readouterr()
        assert "Standalone Report" in captured.out


# ── worker_debug_dir() ───────────────────────────────────────────────────────


class TestWorkerDebugDir:
    """Test worker_debug_dir() returns correct paths."""

    @pytest.fixture(autouse=True)
    def _setup_globals(self, tmp_path):
        cwd = tmp_path / "debug"
        cwd.mkdir()
        run_inductor_comparison.CWD_DIR = cwd

    def test_worker_0(self, tmp_path):
        assert worker_debug_dir(0) == tmp_path / "debug" / "torch_compile_debug_worker_0"

    def test_worker_1(self, tmp_path):
        assert worker_debug_dir(1) == tmp_path / "debug" / "torch_compile_debug_worker_1"

    def test_worker_3(self, tmp_path):
        assert worker_debug_dir(3) == tmp_path / "debug" / "torch_compile_debug_worker_3"


# ── graceful_kill_group() ────────────────────────────────────────────────────


class TestGracefulKillGroup:
    """Test graceful_kill_group() sends SIGTERM first, escalates to SIGKILL."""

    def test_fast_path_sigterm_only(self):
        """Process exits on SIGTERM — no SIGKILL sent."""
        mock_proc = MagicMock()
        mock_proc.pid = 9999
        with patch("run_inductor_comparison.os.getpgid", return_value=9999), \
             patch("run_inductor_comparison.os.killpg") as mock_kill:
            graceful_kill_group(mock_proc, grace_period=2)

        mock_proc.wait.assert_called_once_with(timeout=2)
        # Only one killpg call, with SIGTERM
        mock_kill.assert_called_once_with(9999, signal.SIGTERM)

    def test_slow_path_sigterm_then_sigkill(self):
        """Process ignores SIGTERM — SIGKILL sent after grace period."""
        mock_proc = MagicMock()
        mock_proc.pid = 9999
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="pytest", timeout=2),
            -9,  # second wait() after SIGKILL succeeds
        ]
        with patch("run_inductor_comparison.os.getpgid", return_value=9999), \
             patch("run_inductor_comparison.os.killpg") as mock_kill:
            graceful_kill_group(mock_proc, grace_period=2)

        mock_proc.wait.assert_called()
        assert mock_proc.wait.call_count == 2
        calls = mock_kill.call_args_list
        assert len(calls) == 2
        assert calls[0] == call(9999, signal.SIGTERM)
        assert calls[1] == call(9999, signal.SIGKILL)

    def test_custom_grace_period(self):
        """grace_period parameter controls wait timeout."""
        mock_proc = MagicMock()
        mock_proc.pid = 9999
        with patch("run_inductor_comparison.os.getpgid", return_value=9999), \
             patch("run_inductor_comparison.os.killpg") as mock_kill:
            graceful_kill_group(mock_proc, grace_period=10)

        mock_proc.wait.assert_called_once_with(timeout=10)


# ── TimeoutExpired handling ──────────────────────────────────────────────────


class TestTimeoutExpiredHandling:
    """Test that subprocess.TimeoutExpired is handled correctly via Popen."""

    def _make_worker_args(self):
        return (0, 0, ["fake::test_foo"], "new", 5, "/tmp/_fake_output", 5)

    def test_worker_run_tests_handles_timeout_expired(self):
        """_worker_run_tests should use graceful_kill_group on timeout."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired(
            cmd="pytest", timeout=35
        )
        mock_proc.stdout.read.return_value = b""
        mock_proc.stderr.read.return_value = b"timeout"
        with patch("run_inductor_comparison.graceful_kill_group"), \
             patch("run_inductor_comparison.subprocess.Popen", return_value=mock_proc), \
             patch("run_inductor_comparison._build_env", return_value={}), \
             patch("run_inductor_comparison.clear_debug_dir"):
            results = _worker_run_tests(self._make_worker_args())
        assert "fake::test_foo" in results
        assert results["fake::test_foo"]["status"] == "timeout"

    def test_run_single_test_handles_timeout_expired(self):
        """run_single_test should decode stdout/stderr on timeout."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired(
            cmd="pytest", timeout=35
        )
        mock_proc.stdout.read.return_value = b"stdout data"
        mock_proc.stderr.read.return_value = b"stderr data"
        with patch("run_inductor_comparison.graceful_kill_group"), \
             patch("run_inductor_comparison.subprocess.Popen", return_value=mock_proc), \
             patch("run_inductor_comparison._build_env", return_value={}), \
             patch("run_inductor_comparison.clear_debug_dir"):
            result = run_single_test("fake::test_bar", "new", 5)
        assert result["status"] == "timeout"
        assert result["stdout"] == "stdout data"
        assert result["stderr"] == "stderr data"


# ── extract_generated_code_from_cache() ──────────────────────────────────────


class TestExtractGeneratedCodeFromCache:
    """Test extract_generated_code_from_cache() with temp dirs."""

    @pytest.fixture(autouse=True)
    def _setup_globals(self, tmp_path):
        """Set up CWD_DIR required by other tests in this module."""
        run_inductor_comparison.CWD_DIR = tmp_path / "debug"
        (tmp_path / "debug").mkdir()

    def _create_output_dir(self, tmp_path, backend_key):
        """Create the output_dir structure with a backend label."""
        backend_label = run_inductor_comparison.BACKENDS[backend_key]["label"]
        output_dir = tmp_path / "output"
        (output_dir / backend_label / "debug_staging").mkdir(parents=True)
        return output_dir

    def _create_cache_file(self, tmp_path, subdir, filename, content):
        """Create a fake inductor cache file and return its path."""
        cache_dir = tmp_path / "inductor_cache" / subdir
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / filename
        cache_file.write_text(content)
        return str(cache_file)

    def test_extracts_single_file(self, tmp_path):
        output_dir = self._create_output_dir(tmp_path, "old")
        cache_file = self._create_cache_file(tmp_path, "nc", "abc123.py", "def foo(): pass")
        test_safe = "test_add"
        staging = (output_dir / "old_backend" / "debug_staging" / test_safe)
        staging.mkdir(parents=True)
        (staging / "test.log").write_text(f"Error loading {cache_file}\n")

        result = extract_generated_code_from_cache(test_safe, "old", output_dir)

        assert len(result) == 1
        assert (staging / "torchinductor" / "codecache" / "nc" / "abc123.py").exists()
        assert (staging / "torchinductor" / "codecache" / "cache_sources.txt").exists()

    def test_extracts_multiple_files(self, tmp_path):
        output_dir = self._create_output_dir(tmp_path, "new")
        f1 = self._create_cache_file(tmp_path, "nc", "abc.py", "code_a")
        f2 = self._create_cache_file(tmp_path, "sb", "def.py", "code_b")
        test_safe = "test_conv"
        staging = output_dir / "new_backend" / "debug_staging" / test_safe
        staging.mkdir(parents=True)
        (staging / "test.log").write_text(f"Loaded {f1}\nThen {f2}\n")

        result = extract_generated_code_from_cache(test_safe, "new", output_dir)

        assert len(result) == 2
        assert (staging / "torchinductor" / "codecache" / "nc" / "abc.py").exists()
        assert (staging / "torchinductor" / "codecache" / "sb" / "def.py").exists()

    def test_deduplicates_paths(self, tmp_path):
        output_dir = self._create_output_dir(tmp_path, "old")
        cache_file = self._create_cache_file(tmp_path, "nc", "abc.py", "code")
        test_safe = "test_dup"
        staging = output_dir / "old_backend" / "debug_staging" / test_safe
        staging.mkdir(parents=True)
        (staging / "test.log").write_text(f"{cache_file}\n{cache_file}\n{cache_file}\n")

        result = extract_generated_code_from_cache(test_safe, "old", output_dir)

        assert len(result) == 1
        assert (staging / "torchinductor" / "codecache" / "nc" / "abc.py").exists()

    def test_skips_nonexistent_files(self, tmp_path):
        output_dir = self._create_output_dir(tmp_path, "old")
        test_safe = "test_missing"
        staging = output_dir / "old_backend" / "debug_staging" / test_safe
        staging.mkdir(parents=True)
        (staging / "test.log").write_text("/tmp/fake/inductor_cache/nc/doesnotexist.py\n")

        result = extract_generated_code_from_cache(test_safe, "old", output_dir)

        assert len(result) == 0

    def test_empty_test_log(self, tmp_path):
        output_dir = self._create_output_dir(tmp_path, "old")
        test_safe = "test_empty"
        staging = output_dir / "old_backend" / "debug_staging" / test_safe
        staging.mkdir(parents=True)
        (staging / "test.log").write_text("")

        result = extract_generated_code_from_cache(test_safe, "old", output_dir)

        assert result == []

    def test_no_test_log(self, tmp_path):
        output_dir = self._create_output_dir(tmp_path, "old")
        test_safe = "test_no_log"
        staging = output_dir / "old_backend" / "debug_staging" / test_safe
        staging.mkdir(parents=True)
        # No test.log created

        result = extract_generated_code_from_cache(test_safe, "old", output_dir)

        assert result == []

    def test_ignores_non_python_paths(self, tmp_path):
        output_dir = self._create_output_dir(tmp_path, "old")
        cache_file = self._create_cache_file(tmp_path, "nc", "abc.py", "code")
        test_safe = "test_filter"
        staging = output_dir / "old_backend" / "debug_staging" / test_safe
        staging.mkdir(parents=True)
        (staging / "test.log").write_text(
            f"Loaded {cache_file}\n"
            "/tmp/inductor_cache/nc/somefile.txt\n"
            "/some/random/path.py\n"
        )

        result = extract_generated_code_from_cache(test_safe, "old", output_dir)

        assert len(result) == 1
        assert (staging / "torchinductor" / "codecache" / "nc" / "abc.py").exists()

    def test_cache_sources_written(self, tmp_path):
        output_dir = self._create_output_dir(tmp_path, "old")
        f1 = self._create_cache_file(tmp_path, "nc", "abc.py", "code_a")
        f2 = self._create_cache_file(tmp_path, "sb", "def.py", "code_b")
        test_safe = "test_sources"
        staging = output_dir / "old_backend" / "debug_staging" / test_safe
        staging.mkdir(parents=True)
        (staging / "test.log").write_text(f"Loaded {f1}\nThen {f2}\n")

        extract_generated_code_from_cache(test_safe, "old", output_dir)

        sources = (staging / "torchinductor" / "codecache" / "cache_sources.txt").read_text()
        assert f1 in sources
        assert f2 in sources
        assert "abc.py" in sources
        assert "def.py" in sources
        assert "->" in sources
        lines = sources.strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            assert " -> " in line

    def test_worker_id_parameter(self, tmp_path):
        """worker_id parameter is accepted but doesn't affect staging path."""
        output_dir = self._create_output_dir(tmp_path, "old")
        cache_file = self._create_cache_file(tmp_path, "nc", "abc.py", "code")
        test_safe = "test_worker"
        staging = output_dir / "old_backend" / "debug_staging" / test_safe
        staging.mkdir(parents=True)
        (staging / "test.log").write_text(f"Error loading {cache_file}\n")

        result = extract_generated_code_from_cache(test_safe, "old", output_dir, worker_id=42)

        assert len(result) == 1
        assert (staging / "torchinductor" / "codecache" / "nc" / "abc.py").exists()


# ── _resolve_test_files() ────────────────────────────────────────────────────


class TestResolveTestFiles:
    """Test _resolve_test_files() glob expansion and path resolution."""

    def test_single_file(self, tmp_path):
        f = tmp_path / "test_a.py"
        f.write_text("dummy")
        files = run_inductor_comparison._resolve_test_files([str(f)])
        assert files == [f]

    def test_glob_pattern(self, tmp_path):
        (tmp_path / "test_a.py").write_text("x")
        (tmp_path / "test_b.py").write_text("x")
        (tmp_path / "other.py").write_text("x")
        with patch.object(run_inductor_comparison, "REPO_ROOT", tmp_path):
            files = run_inductor_comparison._resolve_test_files(["test_*.py"])
        assert len(files) == 2
        assert tmp_path / "test_a.py" in files
        assert tmp_path / "test_b.py" in files

    def test_glob_no_match(self, tmp_path):
        with patch.object(run_inductor_comparison, "REPO_ROOT", tmp_path):
            with pytest.raises(SystemExit):
                run_inductor_comparison._resolve_test_files(["no_match_*.py"])

    def test_absolute_path(self, tmp_path):
        f = tmp_path / "test_abs.py"
        f.write_text("x")
        files = run_inductor_comparison._resolve_test_files([str(f)])
        assert files == [f]

    def test_directory_path(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        (d / "test_a.py").write_text("x")
        (d / "test_b.py").write_text("x")
        files = run_inductor_comparison._resolve_test_files([str(d)])
        assert len(files) == 1
        assert files[0] == d

    def test_multiple_patterns(self, tmp_path):
        (tmp_path / "test_a.py").write_text("x")
        (tmp_path / "other.py").write_text("x")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "test_b.py").write_text("x")
        files = run_inductor_comparison._resolve_test_files([str(tmp_path / "test_a.py"), str(subdir)])
        assert len(files) == 2
        assert tmp_path / "test_a.py" in files
        assert subdir in files

    def test_relative_path_from_cwd(self, tmp_path):
        """Relative plain paths resolve relative to CWD."""
        (tmp_path / "test_a.py").write_text("x")
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            files = run_inductor_comparison._resolve_test_files(["test_a.py"])
        assert files == [tmp_path / "test_a.py"]

    def test_deduplicates(self, tmp_path):
        f = tmp_path / "test_a.py"
        f.write_text("x")
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            files = run_inductor_comparison._resolve_test_files(["test_a.py", "test_a.py"])
        assert len(files) == 1

    def test_absolute_glob_pattern(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "test_a.py").write_text("x")
        (subdir / "test_b.py").write_text("x")
        (subdir / "other.py").write_text("x")
        files = run_inductor_comparison._resolve_test_files([str(subdir / "test_*.py")])
        assert len(files) == 2
        assert subdir / "test_a.py" in files
        assert subdir / "test_b.py" in files


# ── collect_tests() ──────────────────────────────────────────────────────────


class TestCollectTests:
    """Test collect_tests() with multiple test files."""

    @pytest.fixture(autouse=True)
    def _setup_globals(self, tmp_path):
        run_inductor_comparison.CWD_DIR = tmp_path / "debug"
        (tmp_path / "debug").mkdir()

    def test_single_file(self, tmp_path):
        """collect_tests with a single file path."""
        f = tmp_path / "test_a.py"
        f.write_text("import unittest\nclass TestA(unittest.TestCase):\n    def test_x(self):\n        pass\n")
        with patch("run_inductor_comparison.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                [], 0, stdout=f"{f}::TestA::test_x\n", stderr=""
            )
            with patch("run_inductor_comparison._build_env", return_value={}):
                ids = run_inductor_comparison.collect_tests([f])
        assert len(ids) == 1

    def test_multiple_files(self, tmp_path):
        """collect_tests combines results from multiple files."""
        f1 = tmp_path / "test_a.py"
        f1.write_text("x")
        f2 = tmp_path / "test_b.py"
        f2.write_text("x")
        calls = []
        def side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            test_path = cmd[-1]
            if "test_a" in test_path:
                calls.append("a")
                return subprocess.CompletedProcess([], 0, stdout=f"{f1}::TestA::test_x\n", stderr="")
            else:
                calls.append("b")
                return subprocess.CompletedProcess([], 0, stdout=f"{f2}::TestB::test_y\n", stderr="")
        with patch("run_inductor_comparison.subprocess.run", side_effect=side_effect), \
             patch("run_inductor_comparison._build_env", return_value={}):
            ids = run_inductor_comparison.collect_tests([f1, f2])
        assert len(calls) == 2
        assert "test_x" in str(ids)
        assert "test_y" in str(ids)

    def test_directory_path(self, tmp_path):
        """collect_tests passes directories to pytest."""
        d = tmp_path / "tests"
        d.mkdir()
        with patch("run_inductor_comparison.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                [], 0, stdout=f"{d}/test_a.py::TestA::test_x\n", stderr=""
            )
            with patch("run_inductor_comparison._build_env", return_value={}):
                run_inductor_comparison.collect_tests([d])
            # Verify pytest command includes the directory path
            cmd = mock_run.call_args[0][0]
            assert str(d) in cmd[-1]
