#!/usr/bin/env python
"""Run test_torchinductor.py with a directly imported NPU inductor backend.

This runner is intentionally single-backend.  It keeps the useful shape of
run_inductor_comparison.py -- collect individual pytest node ids, run each test
in an isolated subprocess, and write pass-rate reports -- but it does not run
or compare the old backend.

Each pytest subprocess imports ``npu_inductor`` before pytest imports
test_torchinductor.py, so the tests stay unchanged and still exercise the real
test cases in test/inductor/test_torchinductor.py.
"""

import argparse
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_TEST_FILE = REPO_ROOT / "test" / "inductor" / "test_torchinductor.py"
BACKEND_LABEL = "npu_inductor_backend"
DEFAULT_BACKEND_MODULE = "npu_inductor"
PYTEST_BOOTSTRAP = (
    "import importlib, pytest, sys; "
    "mod = sys.argv[1]; "
    "None if mod == '__none__' else importlib.import_module(mod); "
    "raise SystemExit(pytest.main(sys.argv[2:]))"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run test_torchinductor.py with npu_inductor and report pass rate"
    )
    parser.add_argument(
        "-k",
        "--test",
        action="append",
        metavar="PATTERN",
        help="pytest -k pattern. Repeat to combine patterns with OR.",
    )
    parser.add_argument(
        "--test-path",
        action="append",
        metavar="NODE_ID",
        help=(
            "Exact pytest node id. You may pass either "
            "test/inductor/test_torchinductor.py::Class::test_name or "
            "Class::test_name."
        ),
    )
    parser.add_argument(
        "-f",
        "--test-file",
        default=str(DEFAULT_TEST_FILE),
        help="Test file to run. Defaults to test/inductor/test_torchinductor.py.",
    )
    parser.add_argument(
        "--backend-module",
        default=DEFAULT_BACKEND_MODULE,
        help="Module imported before pytest loads tests. Defaults to npu_inductor.",
    )
    parser.add_argument(
        "--no-backend-import",
        action="store_true",
        help="Do not import a backend module before pytest. Useful for collection debugging.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to ./npu_inductor_results_<timestamp>/.",
    )
    parser.add_argument(
        "--debug-dir",
        default="/tmp",
        help="Root directory for subprocess CWD, temp files, and compile caches.",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="ASCEND_RT_VISIBLE_DEVICES value, for example 0 or 0,1,2,3.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers. Each worker uses one visible NPU.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-test timeout in seconds.",
    )
    parser.add_argument(
        "--grace-period",
        type=int,
        default=5,
        help="Seconds to wait after SIGTERM before SIGKILL on timeout.",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help="Limit to the first N tests after filtering and blacklist.",
    )
    parser.add_argument(
        "--blacklist",
        default=None,
        help="File with test-name substrings to skip, one per line.",
    )
    parser.add_argument(
        "--no-compile-debug",
        action="store_true",
        help="Do not set TORCH_COMPILE_DEBUG=1 for test subprocesses.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect tests and write test_list.txt.",
    )
    args, pytest_args = parser.parse_known_args()
    if args.test_path and args.test:
        parser.error("--test-path and -k/--test cannot be used together")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args, pytest_args


def setup_runtime_dirs(debug_dir):
    cwd = Path(debug_dir)
    if cwd == Path("/tmp"):
        cwd = cwd / f"npu_inductor_tests_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    cwd.mkdir(parents=True, exist_ok=True)
    for subdir in ("tmp", "triton_cache", "inductor_cache", "torch_compile_debug"):
        (cwd / subdir).mkdir(parents=True, exist_ok=True)
    return cwd


def resolve_test_file(path):
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def resolve_node_id(node_id, default_test_file):
    first = node_id.split("::", 1)[0]
    looks_like_path = (
        first.endswith(".py")
        or "/" in first
        or "\\" in first
        or Path(first).is_absolute()
    )
    if not looks_like_path:
        return f"{default_test_file}::{node_id}"

    parts = node_id.split("::")
    path_part = Path(parts[0])
    if path_part.is_absolute():
        parts[0] = str(path_part.resolve())
        return "::".join(parts)

    for candidate in (
        REPO_ROOT / path_part,
        default_test_file.parent / path_part,
        Path.cwd() / path_part,
    ):
        if candidate.exists():
            parts[0] = str(candidate.resolve())
            return "::".join(parts)

    parts[0] = str((REPO_ROOT / path_part).resolve())
    return "::".join(parts)


def build_env(base_cwd, backend_module, worker_id=None, device_id=None, no_debug=False):
    env = os.environ.copy()

    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        repo = REPO_ROOT.resolve()
        parts = [
            p for p in pythonpath.split(os.pathsep)
            if p and Path(p).resolve() != repo
        ]
        env["PYTHONPATH"] = os.pathsep.join(parts)

    bishengir_dir = (
        REPO_ROOT
        / "third_party"
        / "triton-ascend"
        / "third_party"
        / "ascend"
        / "AscendNPU-IR"
        / "build"
        / "bin"
    )
    if bishengir_dir.exists():
        env["PATH"] = f"{bishengir_dir}{os.pathsep}{env.get('PATH', '')}"
        env.setdefault(
            "TRITON_NPU_COMPILER_PATH",
            str(bishengir_dir / "bishengir-compile"),
        )

    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(base_cwd / "tmp")
    env["TRITON_CACHE_DIR"] = str(base_cwd / "triton_cache")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(base_cwd / "inductor_cache")
    env["NPU_INDUCTOR_TEST_BACKEND_MODULE"] = backend_module
    if no_debug:
        env.pop("TORCH_COMPILE_DEBUG", None)
        env.pop("TORCH_COMPILE_DEBUG_DIR", None)
    else:
        env["TORCH_COMPILE_DEBUG"] = "1"
        debug_name = (
            f"torch_compile_debug_worker_{worker_id}"
            if worker_id is not None
            else "torch_compile_debug"
        )
        env["TORCH_COMPILE_DEBUG_DIR"] = str(base_cwd / debug_name)
    if device_id is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    return env


def bootstrap_cmd(backend_module, pytest_args):
    module = backend_module or "__none__"
    return [sys.executable, "-c", PYTEST_BOOTSTRAP, module] + pytest_args


def collect_tests(test_file, patterns, test_paths, backend_module, cwd, env):
    if test_paths:
        return [resolve_node_id(tp, test_file) for tp in test_paths]

    pytest_args = ["--collect-only", "-q", str(test_file)]
    if patterns:
        pytest_args += ["-k", " or ".join(f"({p})" for p in patterns)]

    cmd = bootstrap_cmd(backend_module, pytest_args)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        message = [
            "pytest collection failed.",
            f"Command: {' '.join(cmd)}",
            f"Return code: {proc.returncode}",
            "=== STDOUT ===",
            proc.stdout or "",
            "=== STDERR ===",
            proc.stderr or "",
        ]
        raise RuntimeError("\n".join(message))

    test_ids = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if re.match(r"^.+\.py::.+$", line):
            test_ids.append(resolve_node_id(line, test_file))
    return test_ids


def apply_blacklist(test_ids, blacklist_path):
    if not blacklist_path or not Path(blacklist_path).exists():
        return test_ids, []
    skipped = [
        line.strip()
        for line in Path(blacklist_path).read_text().splitlines()
        if line.strip()
    ]
    return [t for t in test_ids if not any(s in t for s in skipped)], skipped


def sanitize_test_name(test_id):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", test_id.split("::")[-1])[:180]


def graceful_kill_group(proc, grace_period):
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=grace_period)
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, OSError):
            pass
        proc.wait()


def status_from_returncode(returncode):
    return "passed" if returncode == 0 else "failed"


def parse_test_status(stdout, returncode):
    if not stdout:
        return status_from_returncode(returncode)
    match = re.search(
        r"(?:::\S*\s+(PASSED|FAILED|SKIPPED|ERROR)\b|"
        r"(?:^|\n)(PASSED|FAILED|SKIPPED|ERROR)\s+\[)",
        stdout,
    )
    if match:
        status = match.group(1) or match.group(2)
        return {
            "PASSED": "passed",
            "FAILED": "failed",
            "SKIPPED": "skipped",
            "ERROR": "failed",
        }[status]
    return status_from_returncode(returncode)


def run_one_test(test_id, backend_module, timeout, cwd, env, grace_period):
    cmd = bootstrap_cmd(backend_module, [test_id, "-v", "--tb=long"])
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout_b, stderr_b = proc.communicate(timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        graceful_kill_group(proc, grace_period)
        stdout_b = proc.stdout.read() if proc.stdout else b""
        stderr_b = proc.stderr.read() if proc.stderr else b""
        return {
            "status": "timeout",
            "elapsed": round(elapsed, 2),
            "stdout": stdout_b.decode(errors="replace") if stdout_b else "",
            "stderr": stderr_b.decode(errors="replace") if stderr_b else "",
        }

    elapsed = time.time() - start
    stdout = stdout_b.decode(errors="replace") if stdout_b else ""
    stderr = stderr_b.decode(errors="replace") if stderr_b else ""
    return {
        "status": parse_test_status(stdout, proc.returncode),
        "elapsed": round(elapsed, 2),
        "stdout": stdout,
        "stderr": stderr,
    }


def persist_result(output_dir, test_id, result):
    log_dir = output_dir / BACKEND_LABEL / "logs" / sanitize_test_name(test_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "test.log").write_text(
        "=== STDOUT ===\n"
        + result.get("stdout", "")
        + "\n=== STDERR ===\n"
        + result.get("stderr", ""),
        errors="replace",
    )


def save_results(results, output_path):
    lines = [
        f"{test_id}\t{result['status']}\t{result['elapsed']}"
        for test_id, result in sorted(results.items())
    ]
    output_path.write_text("\n".join(lines) + "\n")


def generate_report(results, output_dir, backend_module):
    total = len(results)
    counts = {}
    for result in results.values():
        counts[result["status"]] = counts.get(result["status"], 0) + 1

    passed = counts.get("passed", 0)
    skipped = counts.get("skipped", 0)
    runnable = max(total - skipped, 0)
    pass_rate_all = (passed / total * 100.0) if total else 0.0
    pass_rate_runnable = (passed / runnable * 100.0) if runnable else 0.0

    lines = []
    sep = "=" * 160
    lines.append(sep)
    lines.append(" NPU Inductor UT Report")
    lines.append(f" Backend module: {backend_module}")
    lines.append(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")
    lines.append(f"{'Test':<88} {'Status':<12} {'Elapsed(s)'}")
    lines.append("-" * 160)
    for test_id, result in sorted(results.items()):
        test_name = test_id.split("::")[-1]
        short_name = test_name[:86] if len(test_name) > 86 else test_name
        lines.append(
            f"{short_name:<88} {result['status']:<12} {result['elapsed']}"
        )
    lines.append("")
    lines.append(sep)
    lines.append(" Summary:")
    lines.append(f"  Total tests:                {total}")
    lines.append(f"  passed:                     {counts.get('passed', 0)}")
    lines.append(f"  failed:                     {counts.get('failed', 0)}")
    lines.append(f"  timeout:                    {counts.get('timeout', 0)}")
    lines.append(f"  skipped:                    {counts.get('skipped', 0)}")
    lines.append(f"  error:                      {counts.get('error', 0)}")
    lines.append(f"  pass rate (all):            {pass_rate_all:.2f}%")
    lines.append(f"  pass rate (excluding skip): {pass_rate_runnable:.2f}%")
    lines.append(sep)

    report = "\n".join(lines) + "\n"
    (output_dir / "comparison_report.txt").write_text(report)
    print("\n" + report)
    return report


def worker_run(args):
    (
        worker_id,
        device_id,
        test_ids,
        backend_module,
        timeout,
        cwd,
        output_dir,
        grace_period,
        no_debug,
    ) = args
    cwd = Path(cwd)
    output_dir = Path(output_dir)
    env = build_env(
        cwd,
        backend_module,
        worker_id=worker_id,
        device_id=device_id,
        no_debug=no_debug,
    )

    results = {}
    for index, test_id in enumerate(test_ids, 1):
        name = test_id.split("::")[-1]
        result = run_one_test(test_id, backend_module, timeout, cwd, env, grace_period)
        results[test_id] = result
        persist_result(output_dir, test_id, result)
        print(
            f"[worker {worker_id}] [{index}/{len(test_ids)}] "
            f"{name} ... {result['status']} ({result['elapsed']}s)",
            flush=True,
        )
    return results


def visible_devices(devices, workers):
    if devices:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = devices
    parent_visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if parent_visible:
        parsed = [d.strip() for d in parent_visible.split(",") if d.strip()]
    else:
        parsed = [str(i) for i in range(workers)]
    if workers > len(parsed):
        raise RuntimeError(
            f"--workers ({workers}) exceeds visible devices ({','.join(parsed)})"
        )
    return parsed


def main():
    args, extra_pytest_args = parse_args()
    if args.no_backend_import:
        args.backend_module = None
    test_file = resolve_test_file(args.test_file)
    if not test_file.exists():
        raise FileNotFoundError(f"test file does not exist: {test_file}")

    output_dir = Path(args.output) if args.output else Path(
        f"npu_inductor_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / BACKEND_LABEL).mkdir(parents=True, exist_ok=True)

    cwd = setup_runtime_dirs(args.debug_dir)
    devices = visible_devices(args.devices, args.workers)
    collect_env = build_env(
        cwd,
        args.backend_module,
        device_id=devices[0],
        no_debug=args.no_compile_debug,
    )

    print(f"Backend module: {args.backend_module}")
    print(f"Test file: {test_file}")
    print(f"Output: {output_dir}")
    print(f"Working directory: {cwd}")
    print(f"Workers: {args.workers}")
    print(f"Devices: {devices[:args.workers]}")
    if args.test:
        print("Pattern:", " or ".join(f"({p})" for p in args.test))
    elif args.test_path:
        print(f"Exact test paths: {len(args.test_path)}")
    else:
        print("Pattern: (all)")

    print("\nCollecting tests...")
    test_ids = collect_tests(
        test_file,
        args.test,
        args.test_path,
        args.backend_module,
        cwd,
        collect_env,
    )
    test_ids, skipped_patterns = apply_blacklist(test_ids, args.blacklist)
    if args.blacklist:
        print(
            f"Blacklist: removed {len(skipped_patterns)} patterns, "
            f"{len(test_ids)} tests remaining"
        )
    if args.limit:
        test_ids = test_ids[: args.limit]
        print(f"Limited to first {args.limit} tests")
    print(f"Collected {len(test_ids)} tests")

    (output_dir / "test_list.txt").write_text("\n".join(test_ids) + "\n")
    if args.collect_only:
        print(f"Done. Test list saved to: {output_dir / 'test_list.txt'}")
        return 0

    if extra_pytest_args:
        print(
            "Warning: extra pytest args are ignored by this per-test runner:",
            " ".join(extra_pytest_args),
        )

    chunks = [test_ids[i::args.workers] for i in range(args.workers)]
    worker_args = [
        (
            worker_id,
            devices[worker_id],
            chunks[worker_id],
            args.backend_module,
            args.timeout,
            str(cwd),
            str(output_dir),
            args.grace_period,
            args.no_compile_debug,
        )
        for worker_id in range(args.workers)
    ]

    if args.workers == 1:
        results = worker_run(worker_args[0])
    else:
        with multiprocessing.Pool(processes=args.workers) as pool:
            worker_results = pool.map(worker_run, worker_args)
        results = {}
        for worker_result in worker_results:
            results.update(worker_result)

    backend_dir = output_dir / BACKEND_LABEL
    save_results(results, backend_dir / "results.txt")
    generate_report(results, output_dir, args.backend_module)
    print(f"\nDone. Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
