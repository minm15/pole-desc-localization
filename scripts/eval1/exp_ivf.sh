#!/usr/bin/env bash
set -e

COMMON_ARGS="--nlist 128 --nprobe 8 --pf 5000 --mode localize --session_start 9 --session_end 13 --quant 4 --quant_method frequency"
LEARNING_CMD="python src/ncltpoles_learning.py"
LOG_FILE="logs/eval1/ivf.log"

: > "${LOG_FILE}"

METHODS=("baseline_hamming" "baseline_l2" "zeroaware")

echo "========================================"
echo " Experiment 2: IVF Method Comparison"
echo " Methods: ${METHODS[*]}"
echo " Fixed Quant: 4"
echo " Log file: ${LOG_FILE}"
echo "========================================"

cp /home/kaiii/nas/homes/kaiii_data/nclt_desc_globalmap/feature_map_eval1/globalmap_5_0.25_0.08_learning.npz /home/kaiii/gpu_test/pole-desc-localization/nclt/
for method in "${METHODS[@]}"; do
    ./scripts/clear.sh
    
    echo "[Experiment] Running with ivf_method: ${method}" | tee -a "${LOG_FILE}"
    
    ${LEARNING_CMD} ${COMMON_ARGS} --ivf_method "${method}" --eval_out "${LOG_FILE}"
done
