#!/usr/bin/env bash
set -u

source /home/l00971669/miniconda3/envs/lyt310_ci/bin/activate
source /usr/local/Ascend/cann-9.1.0-beta.1/set_env.sh

cd /home/nlucci/workspace_pr/BenchBoard/auto_board
mkdir -p ../board_file/auto_board_debug

DATE_ARG=today
VERSION=2130
BACKEND=triton_experimental

run_suite_group() {
  local device="$1"
  local log="$2"
  shift 2

  echo "start group: device=${device}, suites=$*, date=${DATE_ARG}, backend=${BACKEND}, version=${VERSION}, time=$(date)" >> "$log"

  for TYPE in "$@"; do
    echo "===== start ${TYPE} device=${device} $(date) =====" >> "$log"

    ASCEND_RT_VISIBLE_DEVICES=${device} bash auto_board_run.sh \
      -v ${VERSION} \
      -t ${TYPE} \
      -b ${BACKEND} \
      -d ${DATE_ARG} \
      >> "$log" 2>&1

    status=$?
    echo "===== finish ${TYPE} device=${device} exit=${status} $(date) =====" >> "$log"
  done

  echo "done group: device=${device}, time=$(date)" >> "$log"
}

run_suite_group 6 ../board_file/auto_board_debug/v2130_triton_experimental_card6_bench_hf_timm.log bench huggingface timm &
pid6=$!

run_suite_group 7 ../board_file/auto_board_debug/v2130_triton_experimental_card7_llm.log llm &
pid7=$!

wait "$pid6"
status6=$?
wait "$pid7"
status7=$?

echo "all done: card6=${status6}, card7=${status7}, time=$(date)" \
  >> ../board_file/auto_board_debug/v2130_triton_experimental_card6_7.summary.log

exit 0
