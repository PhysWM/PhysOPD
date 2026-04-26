#!/bin/bash
#
# Full PhyGenBench MoE active expert count ablation for Wan2.1 PINN.
#
# Defaults run K=0..8 with Granular/Fracture excluded from routing and auto labels.
# Each K writes videos and performance_metrics.jsonl under OUTPUT_ROOT/topk_${K}.
# PhyGenBench evaluation is optional and disabled by default for generation-first runs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dataset-assist-0/algorithm/cong.wang/miniconda3/envs/wan/bin/python}"
CSV="${CSV:-/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/phygenbench_prompts.csv}"
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/models/train/wan21_stage2_fullpinn8/step-18500.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/phygenbench_moe_topk_ablation/fullpinn8_step18500}"
EXCLUDED_EXPERT_NAMES="${EXCLUDED_EXPERT_NAMES:-Granular,Fracture}"
K_VALUES="${K_VALUES:-0 1 2 3 4 5 6 7 8}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
TOTAL="${TOTAL:-160}"

PHYGENBENCH_ROOT="${PHYGENBENCH_ROOT:-/home/dataset-assist-0/algorithm/cong.wang/projects/PhyGenBench}"
PHYGENBENCH_RUN_SH="${PHYGENBENCH_RUN_SH:-${PHYGENBENCH_ROOT}/run.sh}"
RESULT_DIR="${RESULT_DIR:-result/moe_topk_ablation}"
SOURCE_PATTERN="${SOURCE_PATTERN:-}"
if [[ -z "${SOURCE_PATTERN}" ]]; then
    SOURCE_PATTERN='{index:04d}.mp4'
fi
MODEL_PREFIX="${MODEL_PREFIX:-wan21_pinn8_topk}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_EVAL="${RUN_EVAL:-0}"
DRY_RUN_EVAL="${DRY_RUN_EVAL:-0}"
SKIP_EXISTING_EVAL="${SKIP_EXISTING_EVAL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RESET_PERF_METRICS="${RESET_PERF_METRICS:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

echo "MoE top-k ablation"
echo "  CSV: ${CSV}"
echo "  Model: ${MODEL_ID}"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Output root: ${OUTPUT_ROOT}"
echo "  Excluded experts: ${EXCLUDED_EXPERT_NAMES}"
echo "  K values: ${K_VALUES}"
echo "  GPU IDs: ${GPU_IDS}"
echo "  Run eval: ${RUN_EVAL}"
echo "  Skip existing videos: ${SKIP_EXISTING}"
echo "  PhyGenBench: ${PHYGENBENCH_ROOT}"
echo "  Result dir: ${RESULT_DIR}"

mkdir -p "${OUTPUT_ROOT}"

if [[ "${SUMMARY_ONLY}" == "0" ]]; then
    for K in ${K_VALUES}; do
        TOPK_DIR="${OUTPUT_ROOT}/topk_${K}"
        METRICS_PATH="${TOPK_DIR}/performance_metrics.jsonl"
        LOG_DIR="${TOPK_DIR}/logs"
        mkdir -p "${TOPK_DIR}" "${LOG_DIR}"

        echo
        echo "========== K=${K}: generation =========="
        if [[ "${RUN_GENERATION}" != "0" ]]; then
            VIDEO_COUNT="$(find "${TOPK_DIR}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
            if [[ "${SKIP_EXISTING}" != "0" && "${VIDEO_COUNT}" -ge "${TOTAL}" ]]; then
                echo "K=${K}: found ${VIDEO_COUNT}/${TOTAL} mp4 files, skipping generation."
            else
                if [[ "${RESET_PERF_METRICS}" != "0" ]]; then
                    : > "${METRICS_PATH}"
                fi
                env \
                    PYTHON_BIN="${PYTHON_BIN}" \
                    CSV="${CSV}" \
                    OUTPUT_DIR="${TOPK_DIR}" \
                    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
                    MODEL_ID="${MODEL_ID}" \
                    MOE_TOP_K="${K}" \
                    EXCLUDED_EXPERT_NAMES="${EXCLUDED_EXPERT_NAMES}" \
                    PERFORMANCE_METRICS_PATH="${METRICS_PATH}" \
                    SKIP_EXISTING="${SKIP_EXISTING}" \
                    GPU_IDS="${GPU_IDS}" \
                    NUM_GPUS="${NUM_GPUS}" \
                    TOTAL="${TOTAL}" \
                    LOG_DIR="${LOG_DIR}" \
                    bash "${REPO_ROOT}/batch_inference_wan21_8gpu.sh"
            fi
        else
            echo "Generation disabled by RUN_GENERATION=0"
        fi

        if [[ "${RUN_EVAL}" != "0" ]]; then
            echo
            echo "========== K=${K}: PhyGenBench eval =========="
            MODEL_NAME="${MODEL_PREFIX}${K}"
            env \
                VIDEO_DIR="${TOPK_DIR}" \
                MODEL_NAME="${MODEL_NAME}" \
                RESULT_DIR="${RESULT_DIR}" \
                SOURCE_PATTERN="${SOURCE_PATTERN}" \
                DRY_RUN="${DRY_RUN_EVAL}" \
                SKIP_EXISTING="${SKIP_EXISTING_EVAL}" \
                bash "${PHYGENBENCH_RUN_SH}"
        fi
    done
fi

echo
echo "========== Summary =========="
"${PYTHON_BIN}" examples/wanvideo/pinn_inference/summarize_moe_topk_ablation.py \
    --output_root "${OUTPUT_ROOT}" \
    --phygenbench_root "${PHYGENBENCH_ROOT}" \
    --result_dir "${RESULT_DIR}" \
    --model_prefix "${MODEL_PREFIX}" \
    --k_values "${K_VALUES}" \
    --expected_samples "${TOTAL}" \
    --output_prefix "${OUTPUT_ROOT}/moe_topk_ablation_summary"
