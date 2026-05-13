#!/bin/bash
#
# Controlled single-GPU speed benchmark for MoE active expert count ablation.
#
# This is intentionally separate from the full four-GPU generation/eval script:
# it fixes a 32-sample PhyGenBench subset, disables LLM work, runs warmup inside
# the loaded model process, and summarizes measured generation latency only.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dataset-assist-0/algorithm/cong.wang/miniconda3/envs/wan/bin/python}"
PYTHON_ENV_ROOT="$(cd -- "$(dirname "${PYTHON_BIN}")/.." && pwd)"
if [[ -z "${CUDA_HOME:-}" && -x "${PYTHON_ENV_ROOT}/bin/nvcc" ]]; then
    export CUDA_HOME="${PYTHON_ENV_ROOT}"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

SMOKE="${SMOKE:-0}"
FAST="${FAST:-0}"
if [[ "${SMOKE}" != "0" ]]; then
    DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/phygenbench_moe_topk_speed_benchmark/fullpinn8_step18500_smoke"
    DEFAULT_K_VALUES="0 8"
    DEFAULT_REPEATS="1"
    DEFAULT_SPEED_SAMPLE_COUNT="2"
    DEFAULT_SUBSET_QUOTAS="solid_solid=1,solid_fluid=1"
elif [[ "${FAST}" != "0" ]]; then
    DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/phygenbench_moe_topk_speed_benchmark/fullpinn8_step18500_fast8"
    DEFAULT_K_VALUES="0 1 2 3 4 5 6 7 8"
    DEFAULT_REPEATS="1"
    DEFAULT_SPEED_SAMPLE_COUNT="8"
    DEFAULT_SUBSET_QUOTAS="solid_solid=4,solid_fluid=2,fluid_fluid=2"
else
    DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/phygenbench_moe_topk_speed_benchmark/fullpinn8_step18500"
    DEFAULT_K_VALUES="0 1 2 3 4 5 6 7 8"
    DEFAULT_REPEATS="1 2 3"
    DEFAULT_SPEED_SAMPLE_COUNT="32"
    DEFAULT_SUBSET_QUOTAS="solid_solid=14,solid_fluid=10,fluid_fluid=8"
fi

CSV="${CSV:-${REPO_ROOT}/phygenbench_prompts.csv}"
FULL_ABLATION_ROOT="${FULL_ABLATION_ROOT:-${REPO_ROOT}/phygenbench_moe_topk_ablation/fullpinn8_step18500}"
SOURCE_METRICS="${SOURCE_METRICS:-${FULL_ABLATION_ROOT}/topk_4/performance_metrics.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
SUBSET_CSV="${SUBSET_CSV:-${OUTPUT_ROOT}/speed_subset_${DEFAULT_SPEED_SAMPLE_COUNT}.csv}"
SUBSET_QUOTAS="${SUBSET_QUOTAS:-${DEFAULT_SUBSET_QUOTAS}}"
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/models/train/wan21_stage2_fullpinn8/step-18500.pt}"
EXCLUDED_EXPERT_NAMES="${EXCLUDED_EXPERT_NAMES:-Granular,Fracture}"
K_VALUES="${K_VALUES:-${DEFAULT_K_VALUES}}"
REPEATS="${REPEATS:-${DEFAULT_REPEATS}}"
SPEED_SAMPLE_COUNT="${SPEED_SAMPLE_COUNT:-${DEFAULT_SPEED_SAMPLE_COUNT}}"
WARMUP_SAMPLE_IDS="${WARMUP_SAMPLE_IDS:-1}"
GPU_ID="${GPU_ID:-0}"
PREPARE_SUBSET="${PREPARE_SUBSET:-1}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESET_METRICS="${RESET_METRICS:-0}"

NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
SEED="${SEED:-0}"
CFG_SCALE="${CFG_SCALE:-5.0}"
FPS="${FPS:-15}"
QUALITY="${QUALITY:-5}"

echo "MoE top-k speed benchmark"
echo "  CSV: ${CSV}"
echo "  Source metrics: ${SOURCE_METRICS}"
echo "  Output root: ${OUTPUT_ROOT}"
echo "  Subset CSV: ${SUBSET_CSV}"
echo "  Subset quotas: ${SUBSET_QUOTAS}"
echo "  Model: ${MODEL_ID}"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  GPU ID: ${GPU_ID}"
echo "  K values: ${K_VALUES}"
echo "  Repeats: ${REPEATS}"
echo "  Speed samples: ${SPEED_SAMPLE_COUNT}"
echo "  Warmup sample IDs: ${WARMUP_SAMPLE_IDS}"
echo "  Excluded experts: ${EXCLUDED_EXPERT_NAMES}"

mkdir -p "${OUTPUT_ROOT}"

if [[ "${PREPARE_SUBSET}" != "0" ]]; then
    "${PYTHON_BIN}" examples/wanvideo/pinn_inference/prepare_moe_topk_speed_subset.py \
        --csv "${CSV}" \
        --metrics "${SOURCE_METRICS}" \
        --output_csv "${SUBSET_CSV}" \
        --quotas "${SUBSET_QUOTAS}"
fi

valid_record_count() {
    local metrics_path="$1"
    "${PYTHON_BIN}" - "${metrics_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
count = 0
if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("warmup") or record.get("benchmark_phase") == "warmup":
            continue
        if record.get("skipped") or not record.get("success", False):
            continue
        if record.get("generation_seconds") is None:
            continue
        count += 1
print(count)
PY
}

if [[ "${SUMMARY_ONLY}" == "0" && "${RUN_BENCHMARK}" != "0" ]]; then
    for K in ${K_VALUES}; do
        for REPEAT in ${REPEATS}; do
            RUN_DIR="${OUTPUT_ROOT}/topk_${K}/repeat_${REPEAT}"
            VIDEO_DIR="${RUN_DIR}/videos"
            METRICS_PATH="${RUN_DIR}/performance_metrics.jsonl"
            LOG_DIR="${OUTPUT_ROOT}/logs"
            LOG_PATH="${LOG_DIR}/topk_${K}_repeat_${REPEAT}.log"
            mkdir -p "${VIDEO_DIR}" "${LOG_DIR}"

            if [[ "${RESET_METRICS}" != "0" ]]; then
                : > "${METRICS_PATH}"
            fi

            DONE_COUNT="$(valid_record_count "${METRICS_PATH}")"
            if [[ "${SKIP_COMPLETED}" != "0" && "${DONE_COUNT}" -ge "${SPEED_SAMPLE_COUNT}" ]]; then
                echo "K=${K} repeat=${REPEAT}: found ${DONE_COUNT}/${SPEED_SAMPLE_COUNT} measured records, skipping."
                continue
            fi

            echo
            echo "========== K=${K} repeat=${REPEAT}: speed benchmark =========="
            CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" examples/wanvideo/pinn_inference/batch_inference_pinn.py \
                --csv "${SUBSET_CSV}" \
                --output_dir "${VIDEO_DIR}" \
                --checkpoint_path "${CHECKPOINT_PATH}" \
                --model_id "${MODEL_ID}" \
                --negative_prompt "${NEGATIVE_PROMPT}" \
                --height "${HEIGHT}" \
                --width "${WIDTH}" \
                --num_frames "${NUM_FRAMES}" \
                --num_inference_steps "${NUM_INFERENCE_STEPS}" \
                --seed "${SEED}" \
                --cfg_scale "${CFG_SCALE}" \
                --fps "${FPS}" \
                --quality "${QUALITY}" \
                --device cuda \
                --moe_top_k "${K}" \
                --excluded_expert_names "${EXCLUDED_EXPERT_NAMES}" \
                --performance_metrics_path "${METRICS_PATH}" \
                --benchmark_name moe_topk_speed \
                --benchmark_repeat "${REPEAT}" \
                --benchmark_phase measure \
                --benchmark_warmup_sample_ids "${WARMUP_SAMPLE_IDS}" \
                --disable_prompt_refinement \
                --start_id 1 \
                --end_id "${SPEED_SAMPLE_COUNT}" \
                2>&1 | tee "${LOG_PATH}"
        done
    done
fi

echo
echo "========== Speed summary =========="
"${PYTHON_BIN}" examples/wanvideo/pinn_inference/summarize_moe_topk_speed.py \
    --output_root "${OUTPUT_ROOT}" \
    --k_values "${K_VALUES}" \
    --expected_samples "${SPEED_SAMPLE_COUNT}" \
    --expected_repeats "$(echo "${REPEATS}" | wc -w)" \
    --baseline_k 4 \
    --num_frames "${NUM_FRAMES}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --output_prefix "${OUTPUT_ROOT}/moe_topk_speed_summary"
