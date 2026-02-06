#!/usr/bin/env bash
set -e

COMMON_ARGS="--nlist 128 --nprobe 8 --mode localize --session_start 9 --session_end 13"
LEARNING_CMD="python src/ncltpoles_learning.py"
LOG_FILE="logs/eval1/pf.log"

: > "${LOG_FILE}"

PFS=(1000 5000 10000)

echo "========================================"
echo " Experiment 3: Particle Filter Count"
echo " PF Counts: ${PFS[*]}"
echo " Log file: ${LOG_FILE}"
echo "========================================"

cp /home/kaiii/nas/homes/kaiii_data/nclt_desc_globalmap/feature_map_eval1/globalmap_5_0.25_0.08_learning.npz /home/kaiii/gpu_test/pole-desc-localization/nclt/
for pf_count in "${PFS[@]}"; do
    ./scripts/clear.sh
    
    echo "[Experiment] Running with --pf ${pf_count}" | tee -a "${LOG_FILE}"
    
    ${LEARNING_CMD} ${COMMON_ARGS} --pf "${pf_count}" --eval_out "${LOG_FILE}"
done