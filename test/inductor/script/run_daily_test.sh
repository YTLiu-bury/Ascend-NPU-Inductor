#!/bin/bash
# Daily automated test script
# Calls init_workspace.sh to prepare environment, then runs comparison tests
# Usage: bash run_daily_test.sh [workspace_path] [OPTIONS]
#   workspace_path: optional, defaults to /models/torch-inductor/h00925030/test-daily
# Options:
#   --skip-inductor-tests   Skip test_torchinductor.py (enabled by default)
#   --with-dynamic-shapes   Also run test_torchinductor_dynamic_shapes.py
#   --with-opinfo           Also run test_torchinductor_opinfo.py
#
# Examples:
#   # Run default suite only (inductor_tests)
#   bash run_daily_test.sh
#
#   # Run all three suites
#   bash run_daily_test.sh --with-dynamic-shapes --with-opinfo
#
#   # Run only dynamic_shapes + opinfo (skip default)
#   bash run_daily_test.sh --skip-inductor-tests --with-dynamic-shapes --with-opinfo
#
# Run in background:
#   nohup bash run_daily_test.sh [workspace_path] [OPTIONS] &>run.log &

set -e

# ── Parse arguments ─────────────────────────────────────────────────────────
SKIP_INDUCTOR=false
WITH_DYNAMIC=false
WITH_OPINFO=false
BASE=""
for arg in "$@"; do
    case "$arg" in
        --skip-inductor-tests) SKIP_INDUCTOR=true ;;
        --with-dynamic-shapes) WITH_DYNAMIC=true ;;
        --with-opinfo)         WITH_OPINFO=true ;;
        --help|-h)
            echo "Usage: $0 [workspace_path] [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-inductor-tests   Skip test_torchinductor.py (enabled by default)"
            echo "  --with-dynamic-shapes   Also run test_torchinductor_dynamic_shapes.py"
            echo "  --with-opinfo           Also run test_torchinductor_opinfo.py"
            exit 0
            ;;
        *) BASE="$arg" ;;
    esac
done
BASE=${BASE:-/models/torch-inductor/h00925030/test-daily}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# 1. Build/update environment (clone or pull + rebuild)
bash $SCRIPT_DIR/init_workspace.sh $BASE --daily

# 2. Source environment for test run
REPO=$BASE/pytorch
TRITON=$REPO/third_party/triton-ascend
ASCENDNPU_IR=$TRITON/third_party/ascend/AscendNPU-IR
ANALYSIS=$BASE/pytorch/analysis_report
TS=$(date +%Y%m%d_%H%M)
DAILY_DIR=$ANALYSIS/$TS
SCRIPTS=$REPO/test/inductor/scripts

source /root/Ascend/ascend-toolkit/set_env.sh
source /root/Ascend/cann-9.0.0/set_env.sh
source $BASE/venv/bin/activate
export PATH="$ASCENDNPU_IR/build/bin:$PATH"
unset PYTHONPATH

# Set visible NPU card
export ASCEND_RT_VISIBLE_DEVICES="6,7"

# 3. Create timestamped directory
mkdir -p $DAILY_DIR

# 4. Collect version info
TORCH_NPU_VER=$(pip show torch-npu 2>/dev/null | grep Version | awk '{print $2}')
TORCH_NPU_COMMIT=$(cd $REPO && git rev-parse --short HEAD)
TORCH_NPU_BRANCH=$(cd $REPO && git branch --show-current)
TRITON_ASCEND_VER=$(pip show triton-ascend 2>/dev/null | grep Version | awk '{print $2}')
TRITON_ASCEND_COMMIT=$(cd $TRITON && git rev-parse --short HEAD)
TRITON_ASCEND_BRANCH=$(cd $TRITON && git branch --show-current)
TORCH_VER=$(pip show torch 2>/dev/null | grep Version | awk '{print $2}')
CANN_VER=$ASCEND_HOME_PATH
PYTHON_VER=$(python --version 2>&1)
ASCENDNPU_IR_COMMIT=$(cd $ASCENDNPU_IR && git rev-parse --short HEAD)

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

[AscendNPU-IR]
Commit: $ASCENDNPU_IR_COMMIT
EOF

# 5. Build test suite list
SUITES=()
if ! $SKIP_INDUCTOR; then
    SUITES+=("inductor_tests|test/inductor/test_torchinductor.py")
fi
if $WITH_DYNAMIC; then
    SUITES+=("dynamic_shapes|test/inductor/test_torchinductor_dynamic_shapes.py")
fi
if $WITH_OPINFO; then
    SUITES+=("opinfo|test/inductor/test_torchinductor_opinfo.py")
fi

if [ ${#SUITES[@]} -eq 0 ]; then
    echo "Error: No test suites enabled. At least one suite must run."
    echo "  Default: inductor_tests runs automatically"
    echo "  Use --skip-inductor-tests to disable, --with-dynamic-shapes / --with-opinfo to add"
    exit 1
fi

echo "Test suites to run:"
for entry in "${SUITES[@]}"; do
    echo "  - ${entry%%|*}"
done

# 6. Find previous run for cross-day diff
PREV=$(ls -d $ANALYSIS/20* 2>/dev/null | sort | grep -v "$TS" | tail -1)
if [ -n "$PREV" ]; then
    echo "Previous run found: $PREV"
else
    echo "No previous run found, skipping diffs"
fi

# 7. Run comparison tests and per-suite diff
cd $BASE
FIRST_SUITE_LABEL="${SUITES[0]%%|*}"

for entry in "${SUITES[@]}"; do
    LABEL="${entry%%|*}"
    TEST_FILE="${entry##*|}"

    echo ""
    echo "========================================"
    echo "  Suite: $LABEL ($TEST_FILE)"
    echo "========================================"

    OUTPUT_DIR="$DAILY_DIR/$LABEL"

    python $SCRIPTS/run_inductor_comparison.py \
        --workers 2 \
        --timeout 3000 \
        --blacklist $SCRIPTS/blacklist.txt \
        --test-file "pytorch/$TEST_FILE" \
        --output "$OUTPUT_DIR" \
        --debug-dir "$DAILY_DIR"

    # Cross-day diff — backtrack to find most recent run with this label
    # Old date dirs use comparison/ (maps to inductor_tests); new ones use $LABEL/
    PREV_PATH=""
    for prev_dir in $(ls -d $ANALYSIS/20* 2>/dev/null | sort -r | grep -v "$TS"); do
        if [ -d "$prev_dir/$LABEL" ]; then
            PREV_PATH="$prev_dir/$LABEL"
            break
        fi
        # Fallback: old flat structure (comparison/ ≡ inductor_tests)
        if [ "$LABEL" = "inductor_tests" ] && [ -d "$prev_dir/comparison" ]; then
            PREV_PATH="$prev_dir/comparison"
            break
        fi
    done
    if [ -n "$PREV_PATH" ]; then
        echo "  Diffing $LABEL against $PREV_PATH ..."
        python $SCRIPTS/daily_diff.py --label "$LABEL" "$PREV_PATH" "$OUTPUT_DIR"
    else
        echo "  No previous data for $LABEL (searched all past runs), skipping diff"
    fi
done

# 8. Update analysis_report root with per-suite labeled copies (Layer 3)
#    Each suite gets prefixed copies: <file>_<label>.md
#    No unlabeled root copies — all access via labeled files.
#    failure_analysis_report.md / failure_table.md at root are reserved
#    for multi-suite merged reports (Layer 2), written by daily-test-analysis skill.
for entry in "${SUITES[@]}"; do
    LABEL="${entry%%|*}"
    SUITE_DIR="$DAILY_DIR/$LABEL"

    # Layer 3: labeled copies — every suite gets its own prefixed copy
    for stem in comparison_report daily_diff_report failure_analysis_report failure_table; do
        src=""
        case "$stem" in
            comparison_report)       src="$SUITE_DIR/comparison_report.txt" ; dst="$ANALYSIS/comparison_report_${LABEL}.txt" ;;
            daily_diff_report)       src="$SUITE_DIR/daily_diff_report.md"  ; dst="$ANALYSIS/daily_diff_report_${LABEL}.md"  ;;
            failure_analysis_report) src="$SUITE_DIR/failure_analysis_report.md" ; dst="$ANALYSIS/failure_analysis_report_${LABEL}.md" ;;
            failure_table)           src="$SUITE_DIR/failure_table.md"      ; dst="$ANALYSIS/failure_table_${LABEL}.md"      ;;
        esac
        if [ -f "$src" ]; then
            if cp -f "$src" "$dst"; then
                echo "  [ok] $dst"
            else
                echo "  [FAIL] cp $src -> $dst" >&2
            fi
        fi
    done
done

# No backward-compat root copies — use labeled copies instead:
#   analysis_report/comparison_report_<label>.txt
#   analysis_report/daily_diff_report_<label>.md
# failure_analysis_report.md / failure_table.md at root are reserved
# for multi-suite merged reports (Layer 2), written by daily-test-analysis skill.
