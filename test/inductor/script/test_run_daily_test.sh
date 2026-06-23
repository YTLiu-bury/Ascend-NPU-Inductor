#!/bin/bash
# Quick smoke test for run_daily_test.sh logic (no build, no NPU).
# Validates: timestamp format, directory structure, PREV finding, daily_diff invocation.
# Usage: bash test_run_daily_test.sh

set -e

# ── Setup ──────────────────────────────────────────────────────────────────

WORKDIR=$(mktemp -d)
trap "rm -rf $WORKDIR" EXIT

# Simulate analysis_report directory with fake previous runs
ANALYSIS=$WORKDIR/analysis_report
mkdir -p $ANALYSIS

echo "--- Test 1: Timestamp format ---"
TS=$(date +%Y%m%d_%H%M)
if [[ $TS =~ ^[0-9]{8}_[0-9]{3,4}$ ]]; then
    echo "PASS: TS=$TS matches YYYYMMDD_HHMM"
else
    echo "FAIL: TS=$TS does not match expected format"
    exit 1
fi

echo ""
echo "--- Test 2: Find previous run (empty dir) ---"
PREV=$(ls -d $ANALYSIS/20* 2>/dev/null | sort | grep -v "$TS" | tail -1)
if [ -z "$PREV" ]; then
    echo "PASS: No previous run found (empty dir)"
else
    echo "FAIL: Found unexpected previous run: $PREV"
    exit 1
fi

echo ""
echo "--- Test 3: Find previous run (multiple runs) ---"
mkdir -p $ANALYSIS/20260513_2003
mkdir -p $ANALYSIS/20260514_0918
mkdir -p $ANALYSIS/$TS
PREV=$(ls -d $ANALYSIS/20* 2>/dev/null | sort | grep -v "$TS" | tail -1)
if [ "$PREV" = "$ANALYSIS/20260514_0918" ]; then
    echo "PASS: Previous run = 20260514_0918 (most recent before current)"
else
    echo "FAIL: Expected $ANALYSIS/20260514_0918, got $PREV"
    exit 1
fi

echo ""
echo "--- Test 4: Same-day runs don't collide ---"
TS1="20260515_0918"
TS2="20260515_1430"
mkdir -p $ANALYSIS/$TS1
mkdir -p $ANALYSIS/$TS2
count=$(ls -d $ANALYSIS/20260515_* 2>/dev/null | grep -v "$TS" | wc -l)
if [ "$count" -eq 2 ]; then
    echo "PASS: Two same-day runs coexist ($TS1 and $TS2)"
else
    echo "FAIL: Expected 2 same-day dirs, found $count"
    exit 1
fi

echo ""
echo "--- Test 5: env_info.txt generation ---"
TORCH_NPU_VER="2.7.1.post4"
TORCH_NPU_COMMIT="abc1234"
TORCH_NPU_BRANCH="v2.7.1-26.0.0"
TRITON_ASCEND_VER="3.2.2"
TRITON_ASCEND_COMMIT="def5678"
TRITON_ASCEND_BRANCH="3.2.2_dev"
TORCH_VER="2.7.1"
CANN_VER="/root/Ascend/cann-9.0.0"
PYTHON_VER="Python 3.11.11"

DAILY_DIR=$ANALYSIS/$TS
mkdir -p $DAILY_DIR
cat > $DAILY_DIR/env_info.txt << EOF
[Environment]
Date: $(date '+%Y-%m-%d %H:%M:%S')
Python: $PYTHON_VER

[torch-npu]
Version: $TORCH_NPU_VER
Commit: $TORCH_NPU_COMMIT
Branch: $TORCH_NPU_BRANCH

[triton-ascend]
Version: $TRITON_ASCEND_VER
Commit: $TRITON_ASCEND_COMMIT
Branch: $TRITON_ASCEND_BRANCH

[torch]
Version: $TORCH_VER

[CANN]
Path: $CANN_VER
EOF

if grep -q "Commit: $TORCH_NPU_COMMIT" $DAILY_DIR/env_info.txt && \
   grep -q "Version: $TORCH_VER" $DAILY_DIR/env_info.txt; then
    echo "PASS: env_info.txt generated correctly"
else
    echo "FAIL: env_info.txt content mismatch"
    cat $DAILY_DIR/env_info.txt
    exit 1
fi

echo ""
echo "--- Test 6: daily_diff.py end-to-end with fake data ---"
SCRIPTS_DIR="/models/torch-inductor/h00925030/test-daily/pytorch/test/inductor/scripts"
source /models/torch-inductor/h00925030/test-daily/venv/bin/activate

# Create fake prev run with comparison_report.txt
PREV_DIR=$ANALYSIS/20260514_0918
mkdir -p $PREV_DIR/comparison
cat > $PREV_DIR/comparison/comparison_report.txt << 'REPORT'
================================================================================
 NPU Inductor Backend Comparison Report
================================================================================

Test                                     Old          New          Status
--------------------------------------------------------------------------------
test_abs_npu                             passed       passed       OK
test_addmm_npu                           passed       failed       REGRESSION
test_bmm_npu                             failed       passed       IMPROVED
test_cumsum_npu                          failed       failed       BOTH_FAILED
================================================================================
 Summary:
  Total tests:      4
  OK:               1
================================================================================
REPORT
cat > $PREV_DIR/env_info.txt << 'EOF'
[torch-npu]
Version: 2.7.1.post3
Commit: xyz9999
Branch: v2.7.1-26.0.0
EOF

# Create fake current run with results.xml
CUR_DIR=$ANALYSIS/20260515_0918
mkdir -p $CUR_DIR/comparison/new_backend
python3 -c "
import xml.etree.ElementTree as ET
root = ET.Element('testsuite')
for name, status in [
    ('p/t.py::NPUTests::test_abs_npu', 'passed'),
    ('p/t.py::NPUTests::test_addmm_npu', 'passed'),
    ('p/t.py::NPUTests::test_bmm_npu', 'failed'),
    ('p/t.py::NPUTests::test_cumsum_npu', 'passed'),
]:
    tc = ET.SubElement(root, 'testcase', name=name, time='1.0')
    if status == 'failed':
        ET.SubElement(tc, 'failure', message='fail')
ET.ElementTree(root).write('$CUR_DIR/comparison/new_backend/results.xml', encoding='unicode')
"
cp $DAILY_DIR/env_info.txt $CUR_DIR/env_info.txt

# Run daily_diff.py
python $SCRIPTS_DIR/daily_diff.py $PREV_DIR $CUR_DIR 2>&1 | head -25

if [ -f "$CUR_DIR/daily_diff_report.md" ]; then
    echo ""
    # Verify key content:
    # - test_addmm_npu: prev new=failed, cur new=passed -> IMPROVED
    # - test_cumsum_npu: prev new=failed, cur new=passed -> IMPROVED
    # - test_bmm_npu: prev new=passed, cur new=failed -> REGRESSION
    # - test_abs_npu: prev new=passed, cur new=passed -> STILL PASSING
    if grep -q "test_addmm_npu.*failed.*passed" $CUR_DIR/daily_diff_report.md && \
       grep -q "test_cumsum_npu.*failed.*passed" $CUR_DIR/daily_diff_report.md && \
       grep -q "test_bmm_npu.*passed.*failed" $CUR_DIR/daily_diff_report.md && \
       grep -q "IMPROVED (2" $CUR_DIR/daily_diff_report.md && \
       grep -q "REGRESSION (1" $CUR_DIR/daily_diff_report.md && \
       grep -q "STILL PASSING (1)" $CUR_DIR/daily_diff_report.md && \
       grep -q "xyz9999 -> abc1234" $CUR_DIR/daily_diff_report.md; then
        echo "PASS: daily_diff.py generated correct cross-source report"
    else
        echo "FAIL: daily_diff.py report content unexpected"
        cat $CUR_DIR/daily_diff_report.md
        exit 1
    fi
else
    echo "FAIL: daily_diff_report.md not generated"
    exit 1
fi

echo ""
echo "=========================================="
echo "All 6 tests PASSED"
echo "=========================================="
