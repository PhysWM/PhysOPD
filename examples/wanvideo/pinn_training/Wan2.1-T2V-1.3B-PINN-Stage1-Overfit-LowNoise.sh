#!/bin/bash
# =============================================================================
# Stage 1 low-noise overfit diagnostic for Wan2.1-T2V-1.3B PINN
#
# 目标:
#   用 1 个样本 + 固定低噪声 timestep + 短程训练，检查 Stage 1 objective
#   是否至少具备最基本的可拟合性。
#
# 用法:
#   CUDA_VISIBLE_DEVICES=0 \
#   bash examples/wanvideo/pinn_training/Wan2.1-T2V-1.3B-PINN-Stage1-Overfit-LowNoise.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/Wan2.1-T2V-1.3B-PINN-2Stage.sh"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  FIRST_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
  echo "Overfit diagnostic uses a single GPU. Using CUDA_VISIBLE_DEVICES=${FIRST_GPU}."
  CUDA_VISIBLE_DEVICES="${FIRST_GPU}"
fi

OUTPUT_PATH="${OUTPUT_PATH:-./models/train/wan21_stage1_overfit_1sample_lownoise}"
CACHE_ROOT="${CACHE_ROOT:-${OUTPUT_PATH}/runtime_cache}"

export CUDA_VISIBLE_DEVICES
export TRAINING_STAGE="${TRAINING_STAGE:-observable_pretrain}"
export OUTPUT_PATH
export NUM_EPOCHS="${NUM_EPOCHS:-2}"
export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"
export DEBUG_OVERFIT_NUM_SAMPLES="${DEBUG_OVERFIT_NUM_SAMPLES:-1}"
export DEBUG_OVERFIT_DATASET_REPEAT="${DEBUG_OVERFIT_DATASET_REPEAT:-256}"
export DEBUG_FIXED_TIMESTEP_FRACTION="${DEBUG_FIXED_TIMESTEP_FRACTION:-1.0}"
export MAX_STEPS="${MAX_STEPS:-120}"
export HEARTBEAT_LOG_STEPS="${HEARTBEAT_LOG_STEPS:-1}"
export DIAGNOSTIC_METRICS_INTERVAL="${DIAGNOSTIC_METRICS_INTERVAL:-20}"
export CACHE_ROOT
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${CACHE_ROOT}/modelscope}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"

mkdir -p "${MODELSCOPE_CACHE}" "${HF_HOME}" "${OUTPUT_PATH}"

echo "============================================================"
echo "Wan2.1 Stage1 low-noise overfit diagnostic"
echo "  output path            : ${OUTPUT_PATH}"
echo "  CUDA_VISIBLE_DEVICES   : ${CUDA_VISIBLE_DEVICES}"
echo "  overfit samples        : ${DEBUG_OVERFIT_NUM_SAMPLES}"
echo "  overfit repeat         : ${DEBUG_OVERFIT_DATASET_REPEAT}"
echo "  fixed timestep frac    : ${DEBUG_FIXED_TIMESTEP_FRACTION}"
echo "  max steps              : ${MAX_STEPS}"
echo "  MODELSCOPE_CACHE       : ${MODELSCOPE_CACHE}"
echo "  HF_HOME                : ${HF_HOME}"
echo "============================================================"

bash "${BASE_SCRIPT}"
