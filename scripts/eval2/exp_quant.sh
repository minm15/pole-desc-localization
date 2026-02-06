#!/usr/bin/env bash
set -e

COMMON_ARGS="--nlist 512 --nprobe 32 --pf 5000 --mode localize --session_start 23 --session_end 27 --quant_method frequency"
LEARNING_CMD="python src/ncltpoles_learning.py"
LOG_FILE="logs/eval2/quant_eval2.log"

: > "${LOG_FILE}"

QUANTS=(4 8 "none")

echo "========================================"
echo " Experiment 1: Quant Variation"
echo " Quants: ${QUANTS[*]}"
echo " Log file: ${LOG_FILE}"
echo "========================================"

cp /home/kaiii/nas/homes/kaiii_data/nclt_desc_globalmap/feature_map_eval2/globalmap_5_0.25_0.08_learning.npz /home/kaiii/gpu_test/pole-desc-localization/nclt/
for q in "${QUANTS[@]}"; do
    ./scripts/clear.sh
    
    if [ "$q" == "none" ]; then
        echo "[Experiment] Running without --quant" | tee -a "${LOG_FILE}"
        ${LEARNING_CMD} ${COMMON_ARGS} --eval_out "${LOG_FILE}"
    else
        echo "[Experiment] Running with --quant ${q}" | tee -a "${LOG_FILE}"
        ${LEARNING_CMD} ${COMMON_ARGS} --quant "${q}" --eval_out "${LOG_FILE}"
    fi
done
