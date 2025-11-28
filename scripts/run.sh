#!/usr/bin/env bash
set -e

# parameter
COMMON_ARGS="--nlist 1024 --nprobe 32 --desc_dim 62"

# alias
LEARNING_CMD="python src/ncltpoles_learning.py ${COMMON_ARGS}"
BASELINE_CMD="python src/ncltpoles.py ${COMMON_ARGS}"

# session index
N_SESS=27          # 0..26
IDX_EVAL1_MAP_END=9    # [0,9)  => 2012-01-08 .. 2012-02-19
IDX_EVAL1_LOC_START=9  # [9,13) => 2012-03-17 .. 2012-04-29
IDX_EVAL1_LOC_END=13

IDX_EVAL2_MAP_END=23   # [0,23) => 2012-01-08 .. 2012-11-17
IDX_EVAL2_LOC_START=23 # [23,27)=> 2012-12-01 .. 2013-04-05
IDX_EVAL2_LOC_END=27

# output / eval summary files
LOG_A_LEARN="results_setA_learning.txt"
LOG_A_BASE="results_setA_baseline.txt"
LOG_B_LEARN="results_setB_learning.txt"
LOG_B_BASE="results_setB_baseline.txt"
LOG_C_LEARN="results_setC_learning.txt"
LOG_C_BASE="results_setC_baseline.txt"

# 先把舊的 log 清掉，避免誤混
: > "${LOG_A_LEARN}"
: > "${LOG_A_BASE}"
: > "${LOG_B_LEARN}"
: > "${LOG_B_BASE}"
: > "${LOG_C_LEARN}"
: > "${LOG_C_BASE}"

echo "========================================"
echo " Set A: All sessions (build map + loc)"
echo "========================================"
./scripts/clear.sh

# ---------- Set A: Learning ----------
echo "[Set A][Learning] Build global map + save localmaps + localize on all sessions" | tee -a "${LOG_A_LEARN}"
${LEARNING_CMD} \
  --mode full \
  --session_start 0 \
  --session_end ${N_SESS} \
  --eval_out "${LOG_A_LEARN}"

# ---------- Set A: Baseline ----------
echo "[Set A][Baseline] Build global map + save localmaps + localize on all sessions" | tee -a "${LOG_A_BASE}"
${BASELINE_CMD} \
  --mode full \
  --session_start 0 \
  --session_end ${N_SESS} \
  --eval_out "${LOG_A_BASE}"


echo "========================================"
echo " Set B: Eval1 (map: 01-08~02-19, loc: 03-17~04-29)"
echo "========================================"
./scripts/clear.sh

# ---------- Set B, Part 1: 只用 [0,9) 建 global map ----------
echo "[Set B][Learning] Building global map from sessions[0,${IDX_EVAL1_MAP_END})" | tee -a "${LOG_B_LEARN}"
${LEARNING_CMD} \
  --mode build_map \
  --session_start 0 \
  --session_end ${IDX_EVAL1_MAP_END}

echo "[Set B][Baseline] Building global map from sessions[0,${IDX_EVAL1_MAP_END})" | tee -a "${LOG_B_BASE}"
${BASELINE_CMD} \
  --mode build_map \
  --session_start 0 \
  --session_end ${IDX_EVAL1_MAP_END}

# ---------- Set B, Part 2: 用剛剛的 map，對 [9,13) 做 localization + evaluate ----------
echo "[Set B][Learning] Localizing on sessions[${IDX_EVAL1_LOC_START},${IDX_EVAL1_LOC_END})" | tee -a "${LOG_B_LEARN}"
${LEARNING_CMD} \
  --mode localize \
  --session_start ${IDX_EVAL1_LOC_START} \
  --session_end   ${IDX_EVAL1_LOC_END} \
  --eval_out "${LOG_B_LEARN}"

echo "[Set B][Baseline] Localizing on sessions[${IDX_EVAL1_LOC_START},${IDX_EVAL1_LOC_END})" | tee -a "${LOG_B_BASE}"
${BASELINE_CMD} \
  --mode localize \
  --session_start ${IDX_EVAL1_LOC_START} \
  --session_end   ${IDX_EVAL1_LOC_END} \
  --eval_out "${LOG_B_BASE}"


echo "========================================"
echo " Set C: Eval2 (map: 01-08~11-17, loc: 12-01~04-05)"
echo "========================================"
./scripts/clear.sh

# ---------- Set C, Part 1: 只用 [0,23) 建 global map ----------
echo "[Set C][Learning] Building global map from sessions[0,${IDX_EVAL2_MAP_END})" | tee -a "${LOG_C_LEARN}"
${LEARNING_CMD} \
  --mode build_map \
  --session_start 0 \
  --session_end ${IDX_EVAL2_MAP_END}

echo "[Set C][Baseline] Building global map from sessions[0,${IDX_EVAL2_MAP_END})" | tee -a "${LOG_C_BASE}"
${BASELINE_CMD} \
  --mode build_map \
  --session_start 0 \
  --session_end ${IDX_EVAL2_MAP_END}

# ---------- Set C, Part 2: 用剛剛的 map，對 [23,27) 做 localization + evaluate ----------
echo "[Set C][Learning] Localizing on sessions[${IDX_EVAL2_LOC_START},${IDX_EVAL2_LOC_END})" | tee -a "${LOG_C_LEARN}"
${LEARNING_CMD} \
  --mode localize \
  --session_start ${IDX_EVAL2_LOC_START} \
  --session_end   ${IDX_EVAL2_LOC_END} \
  --eval_out "${LOG_C_LEARN}"

echo "[Set C][Baseline] Localizing on sessions[${IDX_EVAL2_LOC_START},${IDX_EVAL2_LOC_END})" | tee -a "${LOG_C_BASE}"
${BASELINE_CMD} \
  --mode localize \
  --session_start ${IDX_EVAL2_LOC_START} \
  --session_end   ${IDX_EVAL2_LOC_END} \
  --eval_out "${LOG_C_BASE}"

echo "========================================"
echo " All experiments finished."
echo " Results:"
echo "   Set A: ${LOG_A_LEARN} / ${LOG_A_BASE}"
echo "   Set B: ${LOG_B_LEARN} / ${LOG_B_BASE}"
echo "   Set C: ${LOG_C_LEARN} / ${LOG_C_BASE}"
echo "========================================"
