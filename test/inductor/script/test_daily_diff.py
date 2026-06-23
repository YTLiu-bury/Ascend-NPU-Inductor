"""
Unit tests for daily_diff.py pure functions.
Covers: parse_results_xml, parse_comparison_report, parse_env_info,
resolve_truncated_names, load_new_backend_results, classify.
Standalone -- no torch_npu needed.
"""

import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import pytest

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir))

import daily_diff

from daily_diff import (
    classify,
    load_new_backend_results,
    parse_comparison_report,
    parse_env_info,
    parse_results_xml,
    resolve_truncated_names,
)


# ── parse_results_xml ──────────────────────────────────────────────────────


class TestParseResultsXml:
    def _write_xml(self, tmp_path, testcases):
        """Helper: write a JUnit XML with given testcases."""
        root = ET.Element("testsuite")
        for name, status in testcases:
            tc = ET.SubElement(root, "testcase", name=name, time="1.0")
            if status == "failed":
                ET.SubElement(tc, "failure", message="fail")
            elif status == "timeout":
                ET.SubElement(tc, "error", message="timeout")
            elif status == "skipped":
                ET.SubElement(tc, "skipped", message="skip")
        tree = ET.ElementTree(root)
        xml_path = tmp_path / "results.xml"
        tree.write(str(xml_path), encoding="unicode", xml_declaration=True)
        return xml_path

    def test_passed(self, tmp_path):
        xml = self._write_xml(tmp_path, [("test_a_npu", "passed")])
        r = parse_results_xml(xml)
        assert r == {"test_a_npu": "passed"}

    def test_failed(self, tmp_path):
        xml = self._write_xml(tmp_path, [("test_b_npu", "failed")])
        r = parse_results_xml(xml)
        assert r == {"test_b_npu": "failed"}

    def test_timeout(self, tmp_path):
        xml = self._write_xml(tmp_path, [("test_c_npu", "timeout")])
        r = parse_results_xml(xml)
        assert r == {"test_c_npu": "timeout"}

    def test_skipped(self, tmp_path):
        xml = self._write_xml(tmp_path, [("test_d_npu", "skipped")])
        r = parse_results_xml(xml)
        assert r == {"test_d_npu": "skipped"}

    def test_missing_file(self, tmp_path):
        r = parse_results_xml(tmp_path / "nonexistent.xml")
        assert r == {}

    def test_multiple(self, tmp_path):
        xml = self._write_xml(tmp_path, [
            ("test_a_npu", "passed"),
            ("test_b_npu", "failed"),
            ("test_c_npu", "timeout"),
        ])
        r = parse_results_xml(xml)
        assert r == {
            "test_a_npu": "passed",
            "test_b_npu": "failed",
            "test_c_npu": "timeout",
        }


# ── parse_comparison_report ────────────────────────────────────────────────


class TestParseComparisonReport:
    def _write_report(self, tmp_path, lines):
        path = tmp_path / "comparison_report.txt"
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_basic_statuses(self, tmp_path):
        path = self._write_report(tmp_path, [
            "test_abs_npu                             passed       failed       REGRESSION",
            "test_add_npu                              passed       passed       OK",
            "test_bmm_npu                              failed       passed       IMPROVED",
            "test_cumsum_npu                           failed       failed       BOTH_FAILED",
        ])
        r = parse_comparison_report(path)
        assert r == {
            "test_abs_npu": "failed",
            "test_add_npu": "passed",
            "test_bmm_npu": "passed",
            "test_cumsum_npu": "failed",
        }

    def test_timeout_and_skipped(self, tmp_path):
        path = self._write_report(tmp_path, [
            "test_slow_npu                            passed       timeout      TIMEOUT_NEW",
            "test_skip_npu                            skipped      skipped      SKIPPED_OLD",
        ])
        r = parse_comparison_report(path)
        assert r == {
            "test_slow_npu": "timeout",
            "test_skip_npu": "skipped",
        }

    def test_truncated_name(self, tmp_path):
        path = self._write_report(tmp_path, [
            "test_adaptive_avg_pool_errors_with_lon   passed       failed       REGRESSION",
        ])
        r = parse_comparison_report(path)
        assert "test_adaptive_avg_pool_errors_with_lon" in r
        assert r["test_adaptive_avg_pool_errors_with_lon"] == "failed"

    def test_ignores_header_and_summary(self, tmp_path):
        path = self._write_report(tmp_path, [
            "================================================================================",
            " NPU Inductor Backend Comparison Report",
            " Old: TORCHINDUCTOR_NPU_BACKEND=default",
            "--------------------------------------------------------------------------------",
            "Test                                     Old          New          Status",
            "--------------------------------------------------------------------------------",
            "test_abs_npu                             passed       passed       OK",
            "================================================================================",
            " Summary:",
            "  Total tests:      1",
            "================================================================================",
        ])
        r = parse_comparison_report(path)
        assert r == {"test_abs_npu": "passed"}

    def test_missing_file(self, tmp_path):
        r = parse_comparison_report(tmp_path / "nonexistent.txt")
        assert r == {}


# ── parse_env_info ─────────────────────────────────────────────────────────


class TestParseEnvInfo:
    def test_basic_parsing(self, tmp_path):
        path = tmp_path / "env_info.txt"
        path.write_text(textwrap.dedent("""\
            [Environment]
            Date: 2026-05-15 14:30:00
            Python: Python 3.11.11

            [torch-npu]
            Version: 2.7.1.post4
            Commit: 9a3b47264
            Branch: v2.7.1-26.0.0
        """))
        r = parse_env_info(path)
        assert r["Environment"]["Date"] == "2026-05-15 14:30:00"
        assert r["Environment"]["Python"] == "Python 3.11.11"
        assert r["torch-npu"]["Version"] == "2.7.1.post4"
        assert r["torch-npu"]["Commit"] == "9a3b47264"

    def test_missing_file(self, tmp_path):
        r = parse_env_info(tmp_path / "nonexistent.txt")
        assert r == {}

    def test_value_with_colon(self, tmp_path):
        """Values may contain colons (e.g. paths like C:\\...)."""
        path = tmp_path / "env_info.txt"
        path.write_text("[Section]\nKey: http://example.com:8080\n")
        r = parse_env_info(path)
        assert r["Section"]["Key"] == "http://example.com:8080"


# ── resolve_truncated_names ────────────────────────────────────────────────


class TestResolveTruncatedNames:
    def test_names_already_match(self):
        short = {"test_a_npu": "passed", "test_b_npu": "failed"}
        full = {"test_a_npu": "passed", "test_b_npu": "failed"}
        assert resolve_truncated_names(short, full) == short

    def test_suffix_matching(self):
        """Short names should match full pytest node IDs by ::suffix."""
        short = {"test_abs_npu": "failed"}
        full = {"path/test.py::NPUTests::test_abs_npu": "passed"}
        resolved = resolve_truncated_names(short, full)
        assert "path/test.py::NPUTests::test_abs_npu" in resolved
        assert resolved["path/test.py::NPUTests::test_abs_npu"] == "failed"

    def test_truncated_matching(self):
        """Truncated comparison_report name matches full name."""
        short = {"test_adaptive_avg_pool_errors_with_lon": "failed"}
        full = {"path/test.py::NPUTests::test_adaptive_avg_pool_errors_with_long_npu": "passed"}
        resolved = resolve_truncated_names(short, full)
        assert "path/test.py::NPUTests::test_adaptive_avg_pool_errors_with_long_npu" in resolved

    def test_no_match_keeps_original(self):
        short = {"test_unknown_npu": "failed"}
        full = {"path/test.py::NPUTests::test_other_npu": "passed"}
        resolved = resolve_truncated_names(short, full)
        assert "test_unknown_npu" in resolved

    def test_empty_input(self):
        assert resolve_truncated_names({}, {"a": "passed"}) == {}
        assert resolve_truncated_names({"a": "passed"}, {}) == {"a": "passed"}


# ── classify ───────────────────────────────────────────────────────────────


class TestClassify:
    def test_still_passing(self):
        assert classify("passed", "passed") == "STILL_PASSING"

    def test_regression(self):
        assert classify("passed", "failed") == "REGRESSION"
        assert classify("passed", "timeout") == "REGRESSION"

    def test_improved(self):
        assert classify("failed", "passed") == "IMPROVED"
        assert classify("timeout", "passed") == "IMPROVED"

    def test_still_failing(self):
        assert classify("failed", "failed") == "STILL_FAILING"
        assert classify("failed", "timeout") == "STILL_FAILING"
        assert classify("timeout", "failed") == "STILL_FAILING"
        assert classify("timeout", "timeout") == "STILL_FAILING"

    def test_other(self):
        assert classify("skipped", "passed") == "OTHER"
        assert classify("passed", "skipped") == "OTHER"
        assert classify("missing", "failed") == "OTHER"


# ── load_new_backend_results integration ───────────────────────────────────


class TestLoadNewBackendResults:
    def test_prefers_xml(self, tmp_path):
        """When both results.xml and comparison_report.txt exist, uses XML."""
        xml_dir = tmp_path / "comparison" / "new_backend"
        xml_dir.mkdir(parents=True)
        root = ET.Element("testsuite")
        tc = ET.SubElement(root, "testcase", name="test_from_xml_npu", time="1.0")
        ET.ElementTree(root).write(str(xml_dir / "results.xml"), encoding="unicode")

        report_dir = tmp_path / "comparison"
        (report_dir / "comparison_report.txt").write_text(
            "test_from_report_npu                      passed       failed       REGRESSION\n"
        )

        r = load_new_backend_results(tmp_path)
        assert "test_from_xml_npu" in r
        assert "test_from_report_npu" not in r

    def test_fallback_to_comparison_report(self, tmp_path):
        """When results.xml missing, falls back to comparison_report.txt."""
        comp_dir = tmp_path / "comparison"
        comp_dir.mkdir()
        (comp_dir / "comparison_report.txt").write_text(
            "test_abs_npu                             passed       failed       REGRESSION\n"
        )
        r = load_new_backend_results(tmp_path)
        assert r == {"test_abs_npu": "failed"}

    def test_fallback_to_root_comparison_report(self, tmp_path):
        """Falls back to comparison_report.txt in run dir root."""
        (tmp_path / "comparison_report.txt").write_text(
            "test_abs_npu                             passed       passed       OK\n"
        )
        r = load_new_backend_results(tmp_path)
        assert r == {"test_abs_npu": "passed"}

    def test_nothing_found(self, tmp_path):
        r = load_new_backend_results(tmp_path)
        assert r == {}


# ── end-to-end: daily_diff.py main() ──────────────────────────────────────


class TestDailyDiffMain:
    def _setup_run_dir(self, tmp_path, name, xml_testcases, report_lines=None, env_sections=None):
        """Create a fake run directory with results."""
        run_dir = tmp_path / name
        run_dir.mkdir()

        # Write results.xml
        xml_dir = run_dir / "comparison" / "new_backend"
        xml_dir.mkdir(parents=True)
        root = ET.Element("testsuite")
        for tc_name, tc_status in xml_testcases:
            tc = ET.SubElement(root, "testcase", name=tc_name, time="1.0")
            if tc_status == "failed":
                ET.SubElement(tc, "failure", message="fail")
            elif tc_status == "timeout":
                ET.SubElement(tc, "error", message="timeout")
            elif tc_status == "skipped":
                ET.SubElement(tc, "skipped", message="skip")
        ET.ElementTree(root).write(str(xml_dir / "results.xml"), encoding="unicode")

        # Write comparison_report.txt fallback (uses short names)
        if report_lines is not None:
            comp_dir = run_dir / "comparison"
            comp_dir.mkdir(exist_ok=True)
            (comp_dir / "comparison_report.txt").write_text("\n".join(report_lines) + "\n")

        # Write env_info.txt
        if env_sections:
            lines = []
            for section, kv in env_sections.items():
                lines.append(f"[{section}]")
                for k, v in kv.items():
                    lines.append(f"{k}: {v}")
                lines.append("")
            (run_dir / "env_info.txt").write_text("\n".join(lines))

        return run_dir

    def test_xml_vs_xml(self, tmp_path):
        """Both runs have results.xml with full pytest node IDs."""
        prev = self._setup_run_dir(tmp_path, "prev_run", [
            ("path/test.py::NPUTests::test_a_npu", "passed"),
            ("path/test.py::NPUTests::test_b_npu", "failed"),
        ])
        today = self._setup_run_dir(tmp_path, "today_run", [
            ("path/test.py::NPUTests::test_a_npu", "failed"),
            ("path/test.py::NPUTests::test_b_npu", "passed"),
        ])

        with patch("sys.argv", ["daily_diff.py", str(prev), str(today)]):
            daily_diff.main()

        report = (today / "daily_diff_report.md").read_text()
        assert "test_a_npu" in report
        assert "REGRESSION" in report
        assert "IMPROVED" in report
        assert "Previous: prev_run" in report
        assert "Current: today_run" in report

    def test_report_vs_xml_cross_source(self, tmp_path):
        """Previous run has only comparison_report.txt, today has results.xml."""
        # Setup prev manually: NO xml, only comparison_report.txt
        prev = tmp_path / "prev_run"
        prev.mkdir()
        comp_dir = prev / "comparison"
        comp_dir.mkdir()
        (comp_dir / "comparison_report.txt").write_text(
            "test_a_npu                             passed       passed       OK\n"
            "test_b_npu                             passed       failed       REGRESSION\n"
        )

        today = self._setup_run_dir(tmp_path, "today_run", [
            ("path/test.py::NPUTests::test_a_npu", "passed"),
            ("path/test.py::NPUTests::test_b_npu", "passed"),
        ])

        with patch("sys.argv", ["daily_diff.py", str(prev), str(today)]):
            daily_diff.main()

        report = (today / "daily_diff_report.md").read_text()
        assert "IMPROVED" in report  # test_b went from failed to passed
        assert "STILL PASSING" in report

    def test_env_info_displayed(self, tmp_path):
        """Environment section appears in report when env_info.txt exists."""
        prev = self._setup_run_dir(tmp_path, "prev_run",
            xml_testcases=[("path/test.py::NPUTests::test_a_npu", "passed")],
            env_sections={"torch-npu": {"Commit": "abc123"}})
        today = self._setup_run_dir(tmp_path, "today_run",
            xml_testcases=[("path/test.py::NPUTests::test_a_npu", "passed")],
            env_sections={
                "torch-npu": {"Version": "2.7.1", "Commit": "def456", "Branch": "main"},
                "triton-ascend": {"Version": "3.2.2", "Commit": "ghi789", "Branch": "dev"},
            })

        with patch("sys.argv", ["daily_diff.py", str(prev), str(today)]):
            daily_diff.main()

        report = (today / "daily_diff_report.md").read_text()
        assert "2.7.1" in report
        assert "def456" in report
        assert "abc123 -> def456" in report
