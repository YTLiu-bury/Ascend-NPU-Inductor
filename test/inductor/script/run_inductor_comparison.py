#!/usr/bin/env python
"""
Comparison test harness for old vs new NPU inductor backends.

Runs test_torchinductor.py tests sequentially on both backends,
collects debug artifacts for failures, and produces a comparison report.

Usage:
    python test/inductor/scripts/run_inductor_comparison.py [OPTIONS]
"""

import argparse
import multiprocessing
import os
import re
import signal
import shutil
import subprocess
import sys
import time

from pathlib import Path
from datetime import datetime


# ── Constants ────────────────────────────────────────────────────────────────

BACKENDS = {
    "old": {"env": "default", "label": "old_backend"},
    "new": {"env": "new", "label": "new_backend"},
}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BISHENGIR_COMPILE_DIR = REPO_ROOT / "third_party" / "triton-ascend" / "third_party" / "ascend" / "AscendNPU-IR" / "build" / "bin"
DEBUG_DIR = None
CWD_DIR = None

def worker_debug_dir(worker_id):
    """Return the per-worker debug directory for a given worker ID."""
    return CWD_DIR / f"torch_compile_debug_worker_{worker_id}"

# Use the Python interpreter that ran this script, unless overridden via --python.
_DEFAULT_PYTHON = sys.executable


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare old vs new NPU inductor backend test results"
    )
    parser.add_argument(
        "-k", "--test",
        action="append",
        metavar="PATTERN",
        help="pytest -k pattern to filter tests (can be specified multiple times; combined with OR logic)",
    )
    parser.add_argument(
        "--test-path",
        action="append",
        metavar="NODE_ID",
        help="Exact pytest node ID to run (e.g. test_torchinductor.py::TestNew::test_conv2d; can be specified multiple times)",
    )
    parser.add_argument(
        "-f", "--test-file",
        action="append",
        metavar="PATTERN",
        default=None,
        help="Test file or glob pattern to run (repeatable; resolves relative to CWD; "
             "defaults to test_torchinductor.py if omitted)",
    )
    parser.add_argument(
        "--backend",
        choices=["old", "new"],
        default=None,
        help="Run only the specified backend (skips the other; use with --compare-with for diffing)",
    )
    parser.add_argument(
        "--compare-with",
        metavar="DIR",
        default=None,
        help="Path to a previous run's output directory; compare single-backend results against it (requires --backend)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: ./comparison_results_<timestamp>/)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-test timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--grace-period",
        type=int,
        default=5,
        help="Seconds to wait after SIGTERM before SIGKILL on timeout (default: 5)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1; each worker uses one NPU device)",
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Limit to the first N tests (applied after -k/--test-path filtering; affects both backends)",
    )
    parser.add_argument(
        "--debug-dir",
        metavar="DIR",
        default="/tmp",
        help="Root directory for torch compile debug artifacts, triton/inductor cache, and subprocess CWD (default: /tmp)",
    )
    parser.add_argument(
        "--blacklist",
        metavar="FILE",
        default=None,
        help="File with test names to skip (one per line, substring matching)",
    )
    args = parser.parse_args()
    # Validation
    if args.compare_with and not args.backend:
        parser.error("--compare-with requires --backend")
    if args.compare_with and not Path(args.compare_with).is_dir():
        parser.error(f"--compare-with path is not a directory: {args.compare_with}")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.workers > 1:
        import torch
        import torch.npu
        if not torch.npu.is_available():
            parser.error("--workers > 1 requires NPU to be available")
        if args.workers > torch.npu.device_count():
            parser.error(f"--workers ({args.workers}) exceeds available NPU devices ({torch.npu.device_count()})")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


def _setup_dirs(debug_dir):
    """Initialize CWD_DIR and DEBUG_DIR, creating all subdirectories.

    Returns (CWD_DIR, DEBUG_DIR).
    """
    cwd = Path(debug_dir)
    if cwd != Path("/tmp"):
        cwd = cwd / f"debug_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    cwd.mkdir(parents=True, exist_ok=True)
    for sub in ("tmp", "triton_cache", "inductor_cache", "torch_compile_debug"):
        (cwd / sub).mkdir(parents=True, exist_ok=True)
    return cwd, cwd / "torch_compile_debug"


def _resolve_test_id(node_id):
    """Resolve a relative test node ID to an absolute path."""
    parts = node_id.split("::")
    if not Path(parts[0]).is_absolute():
        parts[0] = str(REPO_ROOT / parts[0])
    return "::".join(parts)


def _resolve_test_files(patterns):
    """Resolve glob patterns and paths into a list of absolute Path objects.

    Patterns resolve relative to REPO_ROOT. Files and directories are returned as-is.
    Globs use Path.glob() for single-level patterns. Errors if no files match
    a given pattern.

    Returns deduplicated list of absolute Paths.
    """
    resolved = []
    cwd = Path.cwd()
    base = REPO_ROOT
    for pattern in patterns:
        p = Path(pattern)
        if any(c in pattern for c in ("*", "?", "[")):
            # Glob pattern — use parent dir to expand
            if p.is_absolute():
                base = p.parent
            matches = list(base.glob(p.name))
            if not matches:
                print(f"Error: no files match pattern: {pattern}")
                sys.exit(1)
            resolved.extend([m.resolve() for m in matches])
        elif p.is_absolute():
            resolved.append(p.resolve())
        else:
            # Plain relative path — resolve relative to CWD
            full = (cwd / p).resolve()
            resolved.append(full)

    seen = set()
    unique = []
    for p in resolved:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def apply_blacklist(test_ids, blacklist_path):
    """Filter out tests matching entries in a blacklist file.

    Args:
        test_ids: List of test IDs to filter.
        blacklist_path: Path to a file with one test name pattern per line.

    Returns:
        (filtered_ids, skipped_patterns) tuple.
    """
    if not blacklist_path or not Path(blacklist_path).exists():
        return test_ids, []
    skipped = [line.strip() for line in open(blacklist_path) if line.strip()]
    filtered = [t for t in test_ids if not any(s in t for s in skipped)]
    return filtered, skipped


def collect_tests(test_files, patterns=None, test_paths=None):
    """Collect test IDs using pytest --collect-only.

    Args:
        test_files: List of Path objects (files or directories) to collect from.
        patterns: List of pytest -k patterns (combined with OR logic).
        test_paths: List of exact pytest node IDs to run directly.
    """
    if test_paths and patterns:
        print("Warning: --test-path takes precedence; ignoring -k patterns.")
    if test_paths:
        return [_resolve_test_id(tp) for tp in test_paths]

    all_ids = []
    for test_file in test_files:
        test_file_abs = str(test_file.resolve())
        cmd = [_DEFAULT_PYTHON, "-m", "pytest", "--collect-only", "-q", test_file_abs]
        if patterns:
            combined = " or ".join(f"({p})" for p in patterns)
            cmd.extend(["-k", combined])

        env = _build_env()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(CWD_DIR),
            check=True,
            env=env,
        )
        for line in result.stdout.strip().split("\n"):
            match = re.match(r"^(.+::.+)$", line.strip())
            if match:
                test_id = _resolve_test_id(match.group(1))
                all_ids.append(test_id)

    return [t for t in all_ids if "::" in t]


def clear_debug_dir(worker_id=None):
    """Remove torch_compile_debug/ (or per-worker variant) to ensure fresh artifacts."""
    target = DEBUG_DIR if worker_id is None else worker_debug_dir(worker_id)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


def graceful_kill_group(proc, grace_period=5):
    """Terminate a process group gracefully, falling back to SIGKILL.

    Sends SIGTERM to allow Python atexit handlers and NPU runtime cleanup
    code to run. If the process group does not exit within `grace_period`
    seconds, escalates to SIGKILL.

    Args:
        proc: subprocess.Popen object
        grace_period: seconds to wait after SIGTERM before SIGKILL (default: 5)
    """
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        # Process already exited, nothing to clean up
        return
    try:
        proc.wait(timeout=grace_period)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        proc.wait()



def _build_env(extra=None):
    """Build a clean environment for subprocess calls.

    Removes the repo root from sys.path so that the installed torch_npu
    package (with _C.so) is used instead of the source directory.
    """
    env = os.environ.copy()
    # Use locally built bishengir-compile (same as build-triton-ascend.sh)
    if BISHENGIR_COMPILE_DIR.exists():
        pp_path = env.get("PATH", "")
        if str(BISHENGIR_COMPILE_DIR) not in pp_path:
            env["PATH"] = f"{BISHENGIR_COMPILE_DIR}{os.pathsep}{pp_path}"
    # Remove repo root from PYTHONPATH to avoid shadowing the installed torch_npu
    pp = env.get("PYTHONPATH", "")
    parts = [p for p in pp.split(os.pathsep) if p and Path(p).resolve() != REPO_ROOT.resolve()]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    # Ensure cwd is not accidentally on sys.path by setting PYTHONDONTWRITEBYTECODE
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Redirect triton/inductor compile cache and python tempfile to avoid /tmp inode exhaustion
    env["TRITON_CACHE_DIR"] = str(CWD_DIR / "triton_cache")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(CWD_DIR / "inductor_cache")
    env["TMPDIR"] = str(CWD_DIR / "tmp")
    if extra:
        env.update(extra)
    return env


def run_single_test(test_id, backend_key, timeout, worker_id=None, grace_period=5):
    """Run a single test via pytest subprocess.
    Returns dict with: status, elapsed, stdout, stderr
    """
    backend_val = BACKENDS[backend_key]["env"]
    clear_debug_dir(worker_id)

    env = _build_env({
        "TORCHINDUCTOR_NPU_BACKEND": backend_val,
        "TORCH_COMPILE_DEBUG": "1",
    })
    if worker_id is not None:
        env["TORCH_COMPILE_DEBUG_DIR"] = str(worker_debug_dir(worker_id))

    cmd = [
        _DEFAULT_PYTHON, "-m", "pytest",
        f"{test_id}",
        "-v",
        "--tb=long",
    ]

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(CWD_DIR),
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        graceful_kill_group(proc, grace_period=grace_period)
        stdout, stderr = proc.stdout.read(), proc.stderr.read()
        return {
            "status": "timeout",
            "elapsed": round(elapsed, 2),
            "stdout": stdout.decode() if stdout else "",
            "stderr": stderr.decode() if stderr else "",
        }
    elapsed = time.time() - start
    status = _parse_test_status(stdout.decode() if stdout else "", proc.returncode)
    return {
        "status": status,
        "elapsed": round(elapsed, 2),
        "stdout": stdout.decode() if stdout else "",
        "stderr": stderr.decode() if stderr else "",
    }



def _parse_test_status(stdout, returncode=None):
    """Parse pytest -v stdout to determine test status.

    Matches actual test result lines (e.g. "test_foo PASSED [100%]") rather
    than scanning the entire output for status keywords.  Falls back to
    returncode when stdout doesn't contain enough info.

    Timeout is handled by subprocess.TimeoutExpired, so this function
    never sees a timeout case.
    """
    if not stdout:
        return _status_from_returncode(returncode)
    # Match test result lines like "test_foo PASSED [100%]" or "test_foo FAILED"
    # Also match summary lines like "PASSED [100%] test_torchinductor.py::TestA::test_x"
    match = re.search(
        r"(?:::\S*\s+(PASSED|FAILED|SKIPPED|ERROR)\b|(?:^|\n)(PASSED|FAILED|SKIPPED|ERROR)\s+\[)",
        stdout,
    )
    if match:
        status = match.group(1) or match.group(2)
        status_map = {"PASSED": "passed", "FAILED": "failed", "SKIPPED": "skipped", "ERROR": "failed"}
        return status_map[status]
    return _status_from_returncode(returncode)


def _status_from_returncode(returncode):
    """Map a pytest return code to a status string."""
    if returncode is None:
        return "error"
    if returncode == 0:
        return "passed"
    if returncode == 5:
        return "skipped"
    return "failed"


def _save_results(results, output_path):
    """Write results dict to a one-line-per-test text file."""
    lines = []
    for test_id, result in sorted(results.items()):
        lines.append(f"{test_id} {result['status']} {result['elapsed']}")
    Path(output_path).write_text("\n".join(lines) + "\n")


def _load_results(compare_dir, backend_key):
    """Load results from a previous run's text file.

    Returns a dict of test_id -> {"status": ..., "elapsed": ...}
    """
    backend_label = BACKENDS[backend_key]["label"]
    results_path = Path(compare_dir) / backend_label / "results.txt"
    if not results_path.exists():
        print(f"Warning: Previous results not found at {results_path}")
        return {}

    results = {}
    for line in results_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.rsplit(" ", 2)
        if len(parts) != 3:
            continue
        test_id, status, elapsed = parts
        results[test_id] = {"status": status, "elapsed": round(float(elapsed), 2)}
    return results


def copy_debug_artifacts(test_safe_name, backend_key, output_dir, worker_id=None):
    """Copy torch_compile_debug/run_*/torchinductor/ artifacts to staging area."""
    staging_base = output_dir / BACKENDS[backend_key]["label"] / "debug_staging" / test_safe_name
    if worker_id is not None:
        source_debug = worker_debug_dir(worker_id) / "torch_compile_debug"
    else:
        source_debug = DEBUG_DIR
    if not source_debug.exists():
        staging_base.mkdir(parents=True, exist_ok=True)
        return

    # Find run_* directories
    for run_dir in sorted(source_debug.glob("run_*")):
        torchinductor_dir = run_dir / "torchinductor"
        if torchinductor_dir.is_dir():
            dest = staging_base / "torchinductor"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(str(torchinductor_dir), str(dest))
            # Also copy any sibling dirs/files in run_* that are relevant
            for item in run_dir.iterdir():
                if item.name != "torchinductor" and item.is_dir():
                    sib_dest = staging_base / item.name
                    if sib_dest.exists():
                        shutil.rmtree(sib_dest)
                    shutil.copytree(str(item), str(sib_dest))


_INDUCTOR_CACHE_PAT = re.compile(r'(/\S*inductor_cache/\S+\.py)')


def extract_generated_code_from_cache(
    test_safe_name,
    backend_key,
    output_dir,
    worker_id=None,
):
    """Scan test.log for inductor cache .py paths and copy them to staging.

    Only useful when torchinductor/ artifacts are incomplete (e.g., early
    compilation failure).  Copies discovered files into staging_base/torchinductor/
    and writes cache_sources.txt listing what was extracted.

    Returns the list of destination Paths that were copied.
    """
    staging_base = output_dir / BACKENDS[backend_key]["label"] / "debug_staging" / test_safe_name
    test_log = staging_base / "test.log"

    if not test_log.exists():
        return []

    found_paths = set()
    for match in _INDUCTOR_CACHE_PAT.finditer(test_log.read_text()):
        found_paths.add(match.group(1))

    if not found_paths:
        return []

    codecache_dest = staging_base / "torchinductor" / "codecache"
    codecache_dest.mkdir(parents=True, exist_ok=True)

    copied = []
    source_map = []
    for src_path in sorted(found_paths):
        src = Path(src_path)
        if not src.is_file():
            continue
        dest = codecache_dest / src.parent.name / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        copied.append(dest)
        source_map.append(f"{src} -> codecache/{src.parent.name}/{src.name}")

    if source_map:
        (codecache_dest / "cache_sources.txt").write_text(
            "\n".join(source_map) + "\n"
        )

    return copied


def _worker_run_tests(args):
    """Worker subprocess entry point.

    Args:
        args: Tuple of (worker_id, device_id, test_ids, backend_key, timeout, output_dir, grace_period)

    Returns:
        dict of test_id -> result dict
    """
    worker_id, device_id, test_ids, backend_key, timeout, output_dir, grace_period = args
    output_dir = Path(output_dir)

    results = {}
    for i, test_id in enumerate(test_ids, 1):
        test_name_short = test_id.split("::")[-1]

        # Probe device health before launching test to detect stuck aicpu_scheduler.
        # Retries up to 3 times; if all attempts fail, print a warning but still
        # proceed with the test (probe is advisory, not a gate).
        # Timeout is 45s because cold import torch (9.5s) + torch_npu init +
        # NPU context creation take ~17s even with warm filesystem cache.
        PROBE_TIMEOUT = 45

        def _run_probe(device_id):
            probe_env = _build_env({"ASCEND_RT_VISIBLE_DEVICES": str(device_id)})
            probe_cmd = [
                _DEFAULT_PYTHON, "-c",
                "import torch; import torch_npu; "
                "t = torch.tensor([1.0]).npu(); print('ok')",
            ]
            proc = subprocess.Popen(
                probe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(CWD_DIR),
                env=probe_env,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=PROBE_TIMEOUT)
            except subprocess.TimeoutExpired:
                graceful_kill_group(proc)
                stdout, stderr = proc.stdout.read(), proc.stderr.read()
                raise
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(
                    proc.returncode, probe_cmd,
                    output=stdout, stderr=stderr,
                )
            return True

        for probe_attempt in range(1, 4):
            try:
                _run_probe(device_id)
                break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                if probe_attempt < 3:
                    print(f"[worker {worker_id}] {test_name_short} probe attempt {probe_attempt}/3 failed, retrying in 60s...", flush=True)
                    time.sleep(60)
                else:
                    print(f"[worker {worker_id}] {test_name_short} probe failed after 3 attempts, waiting 180s before proceeding...", flush=True)
                    time.sleep(180)

        clear_debug_dir(worker_id)

        backend_val = BACKENDS[backend_key]["env"]
        w_env = _build_env({
            "TORCHINDUCTOR_NPU_BACKEND": backend_val,
            "TORCH_COMPILE_DEBUG": "1",
            "TORCH_COMPILE_DEBUG_DIR": str(worker_debug_dir(worker_id)),
            "ASCEND_RT_VISIBLE_DEVICES": str(device_id),
        })

        cmd = [
            _DEFAULT_PYTHON, "-m", "pytest",
            f"{test_id}",
            "-v",
            "--tb=long",
        ]

        # Run test with one retry for infrastructure failures
        max_attempts = 2
        for attempt in range(max_attempts):
            start = time.time()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(CWD_DIR),
                env=w_env,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout + 30)
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                graceful_kill_group(proc, grace_period=grace_period)
                stdout, stderr = proc.stdout.read(), proc.stderr.read()
                status = "timeout"
                stdout = stdout.decode() if stdout else ""
                stderr = stderr.decode() if stderr else ""
            else:
                elapsed = time.time() - start
                status = _parse_test_status(stdout.decode() if stdout else "", proc.returncode)
                stdout = stdout.decode() if stdout else ""
                stderr = stderr.decode() if stderr else ""

            if status == "failed" and attempt < max_attempts - 1:
                is_infra = (
                    "507033" in stderr
                    or "ctx is NULL" in stderr
                    or "device retain error" in stderr
                )
                if is_infra:
                    print(f"[worker {worker_id}] {test_name_short} infra failure (attempt {attempt+1}/{max_attempts}), retrying...", flush=True)
                    time.sleep(5)
                    clear_debug_dir(worker_id)
                    continue
            break

        res = {
            "status": status,
            "elapsed": round(elapsed, 2),
            "stdout": stdout,
            "stderr": stderr,
        }
        print(f"[worker {worker_id}] [{i}/{len(test_ids)}] {test_name_short} ... {status} ({res['elapsed']}s)", flush=True)
        results[test_id] = res

        # Immediately persist debug artifacts and logs for this test
        # before the next iteration clears the debug directory.
        test_safe = _sanitize_test_name(test_id)
        staging_dir = output_dir / BACKENDS[backend_key]["label"] / "debug_staging" / test_safe
        staging_dir.mkdir(parents=True, exist_ok=True)
        with open(staging_dir / "test.log", "w") as f:
            f.write("=== STDOUT ===\n")
            f.write(stdout if isinstance(stdout, str) else stdout.decode() if stdout else "")
            f.write("\n=== STDERR ===\n")
            f.write(stderr if isinstance(stderr, str) else stderr.decode() if stderr else "")
        copy_debug_artifacts(test_safe, backend_key, output_dir, worker_id=worker_id)

        if status != "passed":
            extract_generated_code_from_cache(test_safe, backend_key, output_dir, worker_id=worker_id)

    return results


def _sanitize_test_name(test_id):
    """Convert test ID to a safe directory name."""
    # e.g. test/inductor/test_torchinductor.py::TestCase::test_add[foo-bar]
    # -> test_add_foo_bar
    name = test_id.split("::")[-1]
    # Remove brackets and special chars
    name = re.sub(r"[\[\]()<>]", "_", name)
    name = re.sub(r"[_-]+", "_", name)
    return name.strip("_")



def cross_reference_single_backend_failures(test_results, output_dir, backend_key):
    """Build failure artifacts for a single-backend run."""
    failed_tests = {
        test_id for test_id, result in test_results.items()
        if result["status"] != "passed"
    }
    if not failed_tests:
        return

    failures_dir = output_dir / "failures"
    backend_label = BACKENDS[backend_key]["label"]

    for test_id in failed_tests:
        test_safe = _sanitize_test_name(test_id)
        test_failure_dir = failures_dir / test_safe
        staging_path = output_dir / backend_label / "debug_staging" / test_safe
        dest_path = test_failure_dir / backend_label

        if staging_path.exists():
            shutil.copytree(str(staging_path), str(dest_path))
        else:
            dest_path.mkdir(parents=True, exist_ok=True)
            Path(dest_path / "NO_ARTIFACTS.txt").write_text(
                f"No debug artifacts found for {backend_key} backend"
            )

    print(f"\nFailure artifacts saved to: {failures_dir}/")
    print(f"Total failing tests: {len(failed_tests)}")


def cross_reference_failures(old_results, new_results, output_dir):
    """Build final failure artifacts for tests that failed on either backend."""
    all_test_ids = set(old_results.keys()) | set(new_results.keys())
    failed_tests = set()

    for test_id in all_test_ids:
        old_status = old_results.get(test_id, {}).get("status", "error")
        new_status = new_results.get(test_id, {}).get("status", "error")
        if old_status != "passed" or new_status != "passed":
            failed_tests.add(test_id)

    failures_dir = output_dir / "failures"
    for test_id in failed_tests:
        test_safe = _sanitize_test_name(test_id)
        test_failure_dir = failures_dir / test_safe

        for backend_key in ["old", "new"]:
            backend_label = BACKENDS[backend_key]["label"]
            staging_path = output_dir / backend_label / "debug_staging" / test_safe
            dest_path = test_failure_dir / backend_label

            if staging_path.exists():
                shutil.copytree(str(staging_path), str(dest_path), dirs_exist_ok=True)
            else:
                dest_path.mkdir(parents=True, exist_ok=True)
                Path(dest_path / "NO_ARTIFACTS.txt").write_text(
                    f"No debug artifacts found for {backend_key} backend"
                )

    print(f"\nFailure artifacts saved to: {failures_dir}/")
    print(f"Total failing tests: {len(failed_tests)}")


def cleanup_staging(output_dir):
    """Remove debug_staging directories to save disk space."""
    for backend_key in ["old", "new"]:
        staging = output_dir / BACKENDS[backend_key]["label"] / "debug_staging"
        if staging.exists():
            shutil.rmtree(staging)
            print(f"Cleaned up staging for {backend_key}")


def _classify_status(old_status, new_status):
    """Classify combined status for a test."""
    if old_status == "passed" and new_status == "passed":
        return "OK"
    if old_status == "passed" and new_status in ("failed", "error"):
        return "REGRESSION"
    if old_status == "passed" and new_status == "timeout":
        return "TIMEOUT_NEW"
    if old_status == "passed" and new_status == "skipped":
        return "SKIPPED_NEW"
    if old_status in ("failed", "error") and new_status == "passed":
        return "IMPROVED"
    if old_status == "timeout" and new_status == "passed":
        return "TIMEOUT_OLD"
    if old_status == "skipped" and new_status == "passed":
        return "SKIPPED_OLD"
    if old_status in ("failed", "error") and new_status in ("failed", "error"):
        return "BOTH_FAILED"
    if old_status == "timeout" and new_status == "timeout":
        return "BOTH_TIMEOUT"
    if old_status == "skipped" and new_status == "skipped":
        return "BOTH_SKIPPED"
    # Mixed cases
    if old_status == "skipped":
        return "SKIPPED_OLD"
    if new_status == "skipped":
        return "SKIPPED_NEW"
    return "MISMATCH"


def generate_standalone_report(current_results, output_dir, backend_key=None):
    """Generate a standalone report for a single-backend run."""
    all_test_ids = sorted(current_results.keys())
    lines = []
    sep = "=" * 160
    lines.append(sep)
    lines.append(" NPU Inductor Backend - Standalone Report")
    if backend_key:
        lines.append(f" Backend: {BACKENDS[backend_key]['label']} (TORCHINDUCTOR_NPU_BACKEND={BACKENDS[backend_key]['env']})")
    lines.append(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")

    header = f"{'Test':<80} {'Status':<12} {'Elapsed (s)'}"
    lines.append(header)
    lines.append("-" * 160)

    counts = {}
    for test_id in all_test_ids:
        result = current_results[test_id]
        status = result["status"]
        elapsed = result["elapsed"]
        test_name = test_id.split("::")[-1]
        short_name = test_name[:78] if len(test_name) > 78 else test_name
        lines.append(f"{short_name:<80} {status:<12} {elapsed}")
        counts[status] = counts.get(status, 0) + 1

    lines.append("")
    lines.append(sep)
    lines.append(" Summary:")
    lines.append(f"  Total tests:      {len(all_test_ids)}")
    lines.append(f"  passed:           {counts.get('passed', 0)}")
    lines.append(f"  failed:           {counts.get('failed', 0)}")
    lines.append(f"  timeout:          {counts.get('timeout', 0)}")
    lines.append(f"  skipped:          {counts.get('skipped', 0)}")
    lines.append(f"  error:            {counts.get('error', 0)}")
    lines.append(sep)

    report = "\n".join(lines) + "\n"
    report_path = output_dir / "comparison_report.txt"
    report_path.write_text(report)
    print("\n" + report)
    return report


def generate_report(old_results, new_results, output_dir):
    """Generate ASCII comparison report."""
    all_test_ids = sorted(set(old_results.keys()) | set(new_results.keys()))
    rows = []
    counts = {}

    for test_id in all_test_ids:
        old_status = old_results.get(test_id, {}).get("status", "error")
        new_status = new_results.get(test_id, {}).get("status", "error")
        combined = _classify_status(old_status, new_status)
        test_name = test_id.split("::")[-1]
        rows.append((test_name, old_status, new_status, combined))
        counts[combined] = counts.get(combined, 0) + 1

    # Build table
    lines = []
    sep = "=" * 160
    lines.append(sep)
    lines.append(" NPU Inductor Backend Comparison Report")
    lines.append(f" Old: TORCHINDUCTOR_NPU_BACKEND=default")
    lines.append(f" New: TORCHINDUCTOR_NPU_BACKEND=new")
    lines.append(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")

    # Header
    header = f"{'Test':<80} {'Old':<12} {'New':<12} {'Status'}"
    lines.append(header)
    lines.append("-" * 160)

    # Rows
    for test_name, old_s, new_s, combined in rows:
        short_name = test_name[:78] if len(test_name) > 78 else test_name
        lines.append(f"{short_name:<80} {old_s:<12} {new_s:<12} {combined}")

    # Summary
    lines.append("")
    lines.append(sep)
    lines.append(" Summary:")
    lines.append(f"  Total tests:      {len(all_test_ids)}")
    lines.append(f"  OK:               {counts.get('OK', 0)}")
    lines.append(f"  REGRESSION:       {counts.get('REGRESSION', 0)}")
    lines.append(f"  IMPROVED:         {counts.get('IMPROVED', 0)}")
    lines.append(f"  BOTH_FAILED:      {counts.get('BOTH_FAILED', 0)}")
    lines.append(f"  BOTH_TIMEOUT:     {counts.get('BOTH_TIMEOUT', 0)}")
    lines.append(f"  BOTH_SKIPPED:     {counts.get('BOTH_SKIPPED', 0)}")
    lines.append(f"  TIMEOUT_OLD:      {counts.get('TIMEOUT_OLD', 0)}")
    lines.append(f"  TIMEOUT_NEW:      {counts.get('TIMEOUT_NEW', 0)}")
    lines.append(f"  SKIPPED_OLD:      {counts.get('SKIPPED_OLD', 0)}")
    lines.append(f"  SKIPPED_NEW:      {counts.get('SKIPPED_NEW', 0)}")
    lines.append(f"  MISMATCH:         {counts.get('MISMATCH', 0)}")
    lines.append(sep)

    report = "\n".join(lines) + "\n"

    # Save report
    report_path = output_dir / "comparison_report.txt"
    report_path.write_text(report)

    # Print to stdout
    print("\n" + report)

    return report


def main():
    args = parse_args()

    global CWD_DIR, DEBUG_DIR
    CWD_DIR, DEBUG_DIR = _setup_dirs(args.debug_dir)

    output_dir = Path(args.output) if args.output else Path(
        f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine test patterns
    patterns = args.test if args.test else []
    test_paths = args.test_path if args.test_path else []

    if patterns:
        combined_pattern = " or ".join(f"({p})" for p in patterns)
        print(f"Patterns: {combined_pattern}")
    elif test_paths:
        print(f"Test paths: {len(test_paths)} exact node IDs")
    else:
        print("Pattern: (all)")

    print(f"Timeout: {args.timeout}s")
    print(f"Workers: {args.workers}")
    print(f"Limit: {args.limit}")
    print(f"Output: {output_dir}")

    # Determine backends to run
    if args.backend:
        backends_to_run = [args.backend]
        print(f"Single-backend mode: {args.backend}")
    else:
        backends_to_run = ["new", "old"]

    # Resolve test files
    if args.test_file:
        test_files = _resolve_test_files(args.test_file)
        print(f"Test files: {[str(f) for f in test_files]}")
    else:
        default_test_file = REPO_ROOT / "test" / "inductor" / "test_torchinductor.py"
        if not default_test_file.exists():
            default_test_file = Path("/root/torch-npu-internal/test/inductor/test_torchinductor.py")
        test_files = [default_test_file]

    # Collect tests
    print("\nCollecting tests...")
    test_ids = collect_tests(test_files, patterns=patterns, test_paths=test_paths)
    print(f"Collected {len(test_ids)} tests")

    # Apply blacklist
    if args.blacklist:
        test_ids, skipped_patterns = apply_blacklist(test_ids, args.blacklist)
        print(f"Blacklist: removed {len(skipped_patterns)} patterns, {len(test_ids)} tests remaining")

    # Apply limit if specified
    if args.limit:
        test_ids = test_ids[:args.limit]
        print(f"Limited to first {args.limit} tests")

    # Save test list
    with open(output_dir / "test_list.txt", "w") as f:
        for tid in test_ids:
            f.write(tid + "\n")

    # Run tests for each backend
    old_results = {}
    new_results = {}

    if args.workers == 1:
        # Single-worker path (unchanged behavior)
        for backend_key in backends_to_run:
            backend_label = BACKENDS[backend_key]["label"]
            print(f"\n{'='*60}")
            print(f"Running {backend_label} (TORCHINDUCTOR_NPU_BACKEND={BACKENDS[backend_key]['env']})")
            print(f"{'='*60}")

            backend_dir = output_dir / backend_label
            backend_dir.mkdir(parents=True, exist_ok=True)
            results = {}

            for i, test_id in enumerate(test_ids, 1):
                test_safe = _sanitize_test_name(test_id)
                test_name_short = test_id.split("::")[-1]

                print(f"  [{i}/{len(test_ids)}] {test_name_short}...", end=" ", flush=True)
                result = run_single_test(test_id, backend_key, args.timeout, grace_period=args.grace_period)
                status = result["status"]
                print(f"{status} ({result['elapsed']}s)")

                results[test_id] = result

                log_dir = backend_dir / "debug_staging" / test_safe
                log_dir.mkdir(parents=True, exist_ok=True)
                with open(log_dir / "test.log", "w") as f:
                    f.write("=== STDOUT ===\n")
                    f.write(result["stdout"])
                    f.write("\n=== STDERR ===\n")
                    f.write(result["stderr"])

                copy_debug_artifacts(test_safe, backend_key, output_dir)

                if result["status"] != "passed":
                    extract_generated_code_from_cache(test_safe, backend_key, output_dir)

            _save_results(results, backend_dir / "results.txt")

            print(f"\n{backend_label} complete.")

            if backend_key == "old":
                old_results = results
            else:
                new_results = results
    else:
        # Multi-worker path
        # Split test IDs into chunks (round-robin distribution)
        chunks = [test_ids[i::args.workers] for i in range(args.workers)]

        for backend_key in backends_to_run:
            backend_label = BACKENDS[backend_key]["label"]
            print(f"\n{'='*60}")
            print(f"Running {backend_label} (TORCHINDUCTOR_NPU_BACKEND={BACKENDS[backend_key]['env']}) with {args.workers} workers")
            print(f"{'='*60}")

            backend_dir = output_dir / backend_label
            backend_dir.mkdir(parents=True, exist_ok=True)

            # Map worker IDs to physical device IDs from ASCEND_RT_VISIBLE_DEVICES
            parent_vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
            if parent_vis:
                available_devices = [d.strip() for d in parent_vis.split(",") if d.strip()]
            else:
                available_devices = [str(i) for i in range(args.workers)]

            print(f"Device mapping: workers -> {available_devices[:args.workers]}")

            worker_args = [
                (w_id, available_devices[w_id], chunks[w_id], backend_key, args.timeout, str(output_dir), args.grace_period)
                for w_id in range(args.workers)
            ]

            results = {}
            try:
                with multiprocessing.Pool(processes=args.workers) as pool:
                    worker_results = pool.map(_worker_run_tests, worker_args)
                for wr in worker_results:
                    results.update(wr)
            except Exception:
                total = sum(len(c) for c in chunks)
                print(f"\nError: One or more workers failed during {backend_label}. Partial results: {len(results)}/{total} tests.")
                raise

            _save_results(results, backend_dir / "results.txt")

            print(f"\n{backend_label} complete ({len(results)} tests).")

            if backend_key == "old":
                old_results = results
            else:
                new_results = results

    # Load previous results if --compare-with is set
    if args.compare_with:
        print(f"\nLoading previous results from: {args.compare_with}")
        # Compare current run against the SAME backend from the previous run
        # If --backend new, load previous new results into old_results slot
        # so the comparison reads as "old (previous) vs new (current)"
        if args.backend == "new":
            prev_results = _load_results(args.compare_with, "new")
            if not prev_results:
                prev_results = _load_results(args.compare_with, "old")
                if prev_results:
                    print("Warning: Fallback to old_backend results for comparison")
            old_results = prev_results
        else:
            prev_results = _load_results(args.compare_with, "old")
            if not prev_results:
                prev_results = _load_results(args.compare_with, "new")
                if prev_results:
                    print("Warning: Fallback to new_backend results for comparison")
            new_results = prev_results

    # Determine comparison mode
    comparing = len(old_results) > 0 and len(new_results) > 0

    if comparing:
        # Cross-reference failures and build final artifacts
        print("\nCross-referencing failures...")
        cross_reference_failures(old_results, new_results, output_dir)

        # Clean up staging
        print("\nCleaning up staging...")
        cleanup_staging(output_dir)

        # Generate comparison report
        print("\nGenerating comparison report...")
        generate_report(old_results, new_results, output_dir)
    else:
        # Standalone report (single backend, no comparison)
        if args.compare_with:
            print("Warning: No matching previous results found. Falling back to standalone report.")
        current_results = old_results if args.backend == "old" else new_results
        backend_key = args.backend

        # Cross-reference failures and build final artifacts
        print("\nCross-referencing failures...")
        cross_reference_single_backend_failures(current_results, output_dir, backend_key)

        # Clean up staging
        print("\nCleaning up staging...")
        staging = output_dir / BACKENDS[backend_key]["label"] / "debug_staging"
        if staging.exists():
            shutil.rmtree(staging)
            print(f"Cleaned up staging for {backend_key}")

        # Generate standalone report
        print("\nGenerating standalone report...")
        generate_standalone_report(current_results, output_dir, backend_key)

    print(f"\nDone. All artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
