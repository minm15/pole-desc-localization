#!/usr/bin/env bash
set -e

COMMON_ARGS="--nlist 128 --nprobe 8 --mode localize --session_start 23 --session_end 27 --quant 4"

# alias
LEARNING_CMD="python src/ncltpoles_learning.py"

# output log file
LOG_FILE="ivf_log"

: > "${LOG_FILE}"

echo "========================================"
echo " IVF Method Comparison Experiment"
echo " Session: 24-25, Quant: 4"
echo " Log file: ${LOG_FILE}"
echo "========================================"

# ---------- 1. Baseline Hamming ----------
./scripts/clear.sh
METHOD="baseline_hamming"
echo "[Experiment] Running with ivf_method: ${METHOD}" | tee -a "${LOG_FILE}"

${LEARNING_CMD} ${COMMON_ARGS} --ivf_method ${METHOD} --eval_out "${LOG_FILE}"

# ---------- 2. Baseline L2 ----------
./scripts/clear.sh
METHOD="baseline_l2"
echo "[Experiment] Running with ivf_method: ${METHOD}" | tee -a "${LOG_FILE}"

${LEARNING_CMD} ${COMMON_ARGS} --ivf_method ${METHOD} --eval_out "${LOG_FILE}"

# ---------- 3. ZeroAware ----------
./scripts/clear.sh
METHOD="zeroaware"
echo "[Experiment] Running with ivf_method: ${METHOD}" | tee -a "${LOG_FILE}"

${LEARNING_CMD} ${COMMON_ARGS} --ivf_method ${METHOD} --eval_out "${LOG_FILE}"


echo "========================================"
echo " All experiments finished."
echo " Results saved to: ${LOG_FILE}"
echo "========================================"