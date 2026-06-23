#!/usr/bin/env python3
"""
Cross-day comparison of new backend test results.

Usage:
    python daily_diff.py [--label LABEL] <prev_dir> <today_dir>

Reads comparison/new_backend/results.xml from both directories.
Falls back to comparison/comparison_report.txt or run_dir/comparison_report.txt if results.xml missing.
Outputs: <today_dir>/daily_diff_report.md
"""

import sys
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime


def parse_results_xml(xml_path):
    """Parse JUnit XML into dict of test_name -> status."""
    results = {}
    if not Path(xml_path).exists():
        return results
    tree = ET.parse(str(xml_path))
    for tc in tree.getroot().findall(".//testcase"):
        name = tc.get("name", "")
        if tc.find("failure") is not None:
            status = "failed"
        elif tc.find("error") is not None:
            status = "timeout"
        elif tc.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"
        results[name] = status
    return results


def parse_comparison_report(path):
    """Parse comparison_report.txt to extract new backend test statuses.

    Format per line: test_name  old_status  new_status  OVERALL_STATUS
    We extract new_status (third-to-last column).
    """
    results = {}
    valid = {"passed", "failed", "timeout", "skipped"}
    if not Path(path).exists():
        return results
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            status = parts[-1]
            if status not in ("OK", "REGRESSION", "IMPROVED", "BOTH_FAILED",
                              "BOTH_TIMEOUT", "TIMEOUT_OLD", "TIMEOUT_NEW",
                              "SKIPPED_OLD", "SKIPPED_NEW", "MISMATCH"):
                continue
            new_status = parts[-2]
            if new_status not in valid:
                continue
            test_name = " ".join(parts[:-3])
            # Normalize timeout -> timeout for consistency
            results[test_name] = new_status
    return results


def load_new_backend_results(run_dir):
    """Load new backend results, trying results.xml first, then comparison_report.txt."""
    run_dir = Path(run_dir)

    # Try results.xml first
    xml_path = run_dir / "comparison" / "new_backend" / "results.xml"
    if xml_path.exists():
        return parse_results_xml(xml_path)

    # Fallback: parse comparison_report.txt from comparison/ subdir
    report_path = run_dir / "comparison" / "comparison_report.txt"
    if report_path.exists():
        return parse_comparison_report(report_path)

    # Fallback: parse comparison_report.txt from run_dir root
    report_path = run_dir / "comparison_report.txt"
    if report_path.exists():
        return parse_comparison_report(report_path)

    return {}


def classify(yesterday_status, today_status):
    if yesterday_status == "passed" and today_status == "passed":
        return "STILL_PASSING"
    if yesterday_status == "passed" and today_status in ("failed", "timeout"):
        return "REGRESSION"
    if yesterday_status in ("failed", "timeout") and today_status == "passed":
        return "IMPROVED"
    if yesterday_status in ("failed", "timeout") and today_status in ("failed", "timeout"):
        return "STILL_FAILING"
    return "OTHER"


def parse_env_info(env_path):
    """Parse env_info.txt into a dict of section -> {key: value}."""
    sections = {}
    current_section = None
    if not Path(env_path).exists():
        return sections
    for line in open(env_path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            sections[current_section] = {}
        elif current_section and ":" in line:
            key, val = line.split(":", 1)
            sections[current_section][key.strip()] = val.strip()
    return sections


def resolve_truncated_names(short_results, full_results):
    """Map truncated/short test names to full pytest node IDs.

    comparison_report.txt has short/truncated names like 'test_adaptive_avg_pool_errors_with_lon'.
    results.xml has full node IDs like 'path/to/test.py::Class::test_adaptive_avg_pool_errors_with_long_npu'.

    Matching strategy:
    1. Exact match first
    2. Suffix match: full_name ends with '::' + short_name
    3. Prefix match: full_name's test part starts with short_name
    """
    if not short_results or not full_results:
        return short_results

    # Check if names are already identical (both from same source)
    if short_results.keys() == full_results.keys():
        return short_results

    resolved = {}
    for short_name, status in short_results.items():
        if short_name in full_results:
            resolved[short_name] = status
            continue
        # Suffix match: full_name ends with ::short_name
        suffix = "::" + short_name
        matches = [f for f in full_results if f.endswith(suffix)]
        if len(matches) == 1:
            resolved[matches[0]] = status
            continue
        # Prefix match: truncated name is a prefix of the full test name
        prefix_matches = [f for f in full_results
                          if f.split("::")[-1].startswith(short_name)]
        if len(prefix_matches) == 1:
            resolved[prefix_matches[0]] = status
        else:
            resolved[short_name] = status
    return resolved


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-day comparison of new backend test results.")
    parser.add_argument("--label", default=None, help="Suite label for report title (e.g. inductor_tests)")
    parser.add_argument("prev_dir", help="Previous run directory")
    parser.add_argument("today_dir", help="Today's run directory")
    args_ns = parser.parse_args()

    prev_dir = Path(args_ns.prev_dir)
    today_dir = Path(args_ns.today_dir)
    label = args_ns.label

    prev_label = prev_dir.name
    today_label = today_dir.name

    # Load env info
    today_env = parse_env_info(today_dir / "env_info.txt")
    prev_env = parse_env_info(prev_dir / "env_info.txt")

    # Load results
    prev_results = load_new_backend_results(prev_dir)
    today_results = load_new_backend_results(today_dir)

    if not prev_results:
        print(f"Error: No results found in {prev_dir}")
        sys.exit(1)
    if not today_results:
        print(f"Error: No results found in {today_dir}")
        sys.exit(1)

    # Resolve truncated names if sources differ
    prev_results = resolve_truncated_names(prev_results, today_results)
    today_results = resolve_truncated_names(today_results, prev_results)

    # Classify all tests
    all_tests = sorted(set(prev_results.keys()) | set(today_results.keys()))
    categories = {
        "REGRESSION": [],
        "IMPROVED": [],
        "STILL_FAILING": [],
        "STILL_PASSING": [],
        "OTHER": [],
    }

    for test in all_tests:
        y_status = prev_results.get(test, "missing")
        t_status = today_results.get(test, "missing")
        cat = classify(y_status, t_status)
        categories[cat].append((test, y_status, t_status))

    # Count summary for each day
    y_counts = {"passed": 0, "failed": 0, "timeout": 0, "skipped": 0}
    t_counts = {"passed": 0, "failed": 0, "timeout": 0, "skipped": 0}
    for s in prev_results.values():
        if s in y_counts:
            y_counts[s] += 1
    for s in today_results.values():
        if s in t_counts:
            t_counts[s] += 1

    # Build report
    lines = []
    lines.append("# NPU Inductor Daily Diff Report (new backend)" + (f" - {label}" if label else ""))
    lines.append("")
    lines.append(f"Previous: {prev_label} | Current: {today_label}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Environment info
    if today_env:
        lines.append("## Environment")
        lines.append("")
        for section in ["torch-npu", "triton-ascend"]:
            if section in today_env:
                s = today_env[section]
                lines.append(f"**{section}**: {s.get('Version', 'N/A')} (`{s.get('Commit', 'N/A')}` on `{s.get('Branch', 'N/A')}`)")
        if prev_env:
            y_npu = prev_env.get("torch-npu", {}).get("Commit", "")
            t_npu = today_env.get("torch-npu", {}).get("Commit", "")
            y_ta = prev_env.get("triton-ascend", {}).get("Commit", "")
            t_ta = today_env.get("triton-ascend", {}).get("Commit", "")
            if y_npu and t_npu:
                npu_change = "unchanged" if y_npu == t_npu else f"{y_npu} -> {t_npu}"
                lines.append(f"- torch-npu commit: {npu_change}")
            if y_ta and t_ta:
                ta_change = "unchanged" if y_ta == t_ta else f"{y_ta} -> {t_ta}"
                lines.append(f"- triton-ascend commit: {ta_change}")
        lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Category | Previous | Current | Delta |")
    lines.append("|----------|----------|---------|-------|")
    for key in ["passed", "failed", "timeout", "skipped"]:
        y_val = y_counts[key]
        t_val = t_counts[key]
        delta = t_val - y_val
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        lines.append(f"| {key} | {y_val} | {t_val} | {delta_str} |")
    lines.append(f"| total | {len(prev_results)} | {len(today_results)} | {len(today_results) - len(prev_results):+d} |")
    lines.append("")

    # Net improvement
    net = len(categories["IMPROVED"]) - len(categories["REGRESSION"])
    net_str = f"+{net}" if net > 0 else str(net)
    lines.append(f"**Net change**: IMPROVED({len(categories['IMPROVED'])}) - REGRESSION({len(categories['REGRESSION'])}) = **{net_str}**")
    lines.append("")

    # REGRESSION
    if categories["REGRESSION"]:
        lines.append(f"## REGRESSION ({len(categories['REGRESSION'])} - previous passed, current failed)")
        lines.append("")
        lines.append("| # | Test | Previous | Current |")
        lines.append("|---|------|----------|---------|")
        for i, (test, y_s, t_s) in enumerate(categories["REGRESSION"], 1):
            test_short = test.split("::")[-1]
            lines.append(f"| {i} | {test_short} | {y_s} | {t_s} |")
        lines.append("")

    # IMPROVED
    if categories["IMPROVED"]:
        lines.append(f"## IMPROVED ({len(categories['IMPROVED'])} - previous failed, current passed)")
        lines.append("")
        lines.append("| # | Test | Previous | Current |")
        lines.append("|---|------|----------|---------|")
        for i, (test, y_s, t_s) in enumerate(categories["IMPROVED"], 1):
            test_short = test.split("::")[-1]
            lines.append(f"| {i} | {test_short} | {y_s} | {t_s} |")
        lines.append("")

    # STILL FAILING
    if categories["STILL_FAILING"]:
        lines.append(f"## STILL FAILING ({len(categories['STILL_FAILING'])})")
        lines.append("")
        lines.append("| # | Test | Previous | Current |")
        lines.append("|---|------|----------|---------|")
        for i, (test, y_s, t_s) in enumerate(categories["STILL_FAILING"], 1):
            test_short = test.split("::")[-1]
            lines.append(f"| {i} | {test_short} | {y_s} | {t_s} |")
        lines.append("")

    # STILL PASSING
    lines.append(f"## STILL PASSING ({len(categories['STILL_PASSING'])})")
    lines.append("")
    lines.append(f"{len(categories['STILL_PASSING'])} tests passed on both runs.")
    lines.append("")

    # Write report
    report = "\n".join(lines) + "\n"
    report_path = today_dir / "daily_diff_report.md"
    report_path.write_text(report)
    print(report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
