#!/usr/bin/env bash
set -e

COMMON_ARGS="--nlist 512 --nprobe 32 --pf 5000 --mode localize --session_start 23 --session_end 27 --quant 4"
LEARNING_CMD="python src/ncltpoles_learning.py"
LOG_FILE="logs/eval2/quant_method.log"

: > "${LOG_FILE}"

METHOD=("frequency" "uniform" "zscore")

echo "========================================"
echo " Quants: ${METHOD[*]}"
echo " Log file: ${LOG_FILE}"
echo "========================================"

cp /home/kaiii/nas/homes/kaiii_data/nclt_desc_globalmap/feature_map_eval2/globalmap_5_0.25_0.08_learning.npz /home/kaiii/gpu_test/pole-desc-localization/nclt/
for q in "${METHOD[@]}"; do
    ./scripts/clear.sh

    echo "[Experiment] Running quantization method: $q" | tee -a "${LOG_FILE}"
    ${LEARNING_CMD} ${COMMON_ARGS} --eval_out "${LOG_FILE}" --quant_method "${q}"
done
