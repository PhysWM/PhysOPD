#!/bin/bash
#
# 4-GPU pure AnyFlow PhysGenBench inference.
#
# This is the clean AnyFlow baseline:
#   - no PhysicsAdapter
#   - no LLM prompt refinement
#   - no physics label routing
#
# Usage:
#   bash examples/wanvideo/pinn_inference/batch_inference_anyflow_4gpu.sh
#   TOTAL=16 bash examples/wanvideo/pinn_inference/batch_inference_anyflow_4gpu.sh
#   MODEL_PATH=/path/to/AnyFlow-FAR-Wan2.1-1.3B-Diffusers bash ...
#

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dataset-assist-0/algorithm/cong.wang/miniconda3/envs/anyflow/bin/python}"
CSV="${CSV:-/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/phygenbench_prompts.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/phygenbench_anyflow_pure_1p3b_steps4}"
ANYFLOW_ROOT="${ANYFLOW_ROOT:-/home/dataset-assist-0/algorithm/cong.wang/try/AnyFlow}"
MODEL_PATH="${MODEL_PATH:-${ANYFLOW_ROOT}/experiments/pretrained_models/AnyFlow-Wan2.1-T2V-1.3B-Diffusers}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-4}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
SEED="${SEED:-0}"
FPS="${FPS:-16}"
DTYPE="${DTYPE:-bf16}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RESUME="${RESUME:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
TOTAL="${TOTAL:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python not executable: ${PYTHON_BIN}"
    exit 1
fi
if [[ ! -f "${CSV}" ]]; then
    echo "CSV not found: ${CSV}"
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "AnyFlow model path not found: ${MODEL_PATH}"
    exit 1
fi

IFS=',' read -r -a ALL_GPU_IDS <<< "${GPU_IDS}"
if [[ "${NUM_GPUS}" -gt "${#ALL_GPU_IDS[@]}" ]]; then
    echo "NUM_GPUS=${NUM_GPUS} exceeds GPU_IDS count ${#ALL_GPU_IDS[@]} (${GPU_IDS})"
    exit 1
fi
GPU_LIST=("${ALL_GPU_IDS[@]:0:${NUM_GPUS}}")
GPU_COUNT="${#GPU_LIST[@]}"
if [[ "${GPU_COUNT}" -le 0 ]]; then
    echo "No GPU selected."
    exit 1
fi

if [[ -z "${TOTAL}" ]]; then
    TOTAL="$("${PYTHON_BIN}" -c 'import csv, sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"))))' "${CSV}")"
fi
if ! [[ "${TOTAL}" =~ ^[0-9]+$ ]] || [[ "${TOTAL}" -le 0 ]]; then
    echo "TOTAL must be a positive integer, got: ${TOTAL}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
CHUNK=$(( (TOTAL + GPU_COUNT - 1) / GPU_COUNT ))
PERFORMANCE_METRICS_PATH="${PERFORMANCE_METRICS_PATH:-${OUTPUT_DIR}/performance_metrics.jsonl}"

COMMON_ARGS=(
    --csv "${CSV}"
    --output_dir "${OUTPUT_DIR}"
    --performance_metrics_path "${PERFORMANCE_METRICS_PATH}"
    --anyflow_root "${ANYFLOW_ROOT}"
    --model_path "${MODEL_PATH}"
    --negative_prompt "${NEGATIVE_PROMPT}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --num_frames "${NUM_FRAMES}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --seed "${SEED}"
    --fps "${FPS}"
    --dtype "${DTYPE}"
)

if [[ "${SKIP_EXISTING}" != "0" ]]; then
    COMMON_ARGS+=(--skip_existing)
fi
if [[ "${RESUME}" != "0" ]]; then
    COMMON_ARGS+=(--resume)
fi
if [[ "${CONTINUE_ON_ERROR}" != "0" ]]; then
    COMMON_ARGS+=(--continue_on_error)
fi

run_one() {
    local gpu_id="$1"
    local start_id="$2"
    local end_id="$3"

    if [[ "${start_id}" -gt "${end_id}" ]]; then
        return 0
    fi

    echo "[GPU ${gpu_id}] Starting IDs ${start_id}-${end_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" TOKENIZERS_PARALLELISM=false \
        "${PYTHON_BIN}" examples/wanvideo/pinn_inference/batch_inference_anyflow.py \
        "${COMMON_ARGS[@]}" \
        --start_id "${start_id}" \
        --end_id "${end_id}" \
        2>&1 | tee "${LOG_DIR}/gpu${gpu_id}_${start_id}_${end_id}.log"
}

echo "CSV: ${CSV}"
echo "Output: ${OUTPUT_DIR}"
echo "AnyFlow model: ${MODEL_PATH}"
echo "GPU IDs: ${GPU_LIST[*]}"
echo "Total samples: ${TOTAL}"
echo "Chunk size: ${CHUNK}"
echo "Steps: ${NUM_INFERENCE_STEPS}"
echo "Guidance scale: ${GUIDANCE_SCALE}"
echo "Pure AnyFlow: no adapter, no LLM, no routing"

pids=()
for idx in "${!GPU_LIST[@]}"; do
    gpu_id="${GPU_LIST[$idx]}"
    start_id=$(( idx * CHUNK + 1 ))
    end_id=$(( (idx + 1) * CHUNK ))
    if [[ "${end_id}" -gt "${TOTAL}" ]]; then
        end_id="${TOTAL}"
    fi
    if [[ "${start_id}" -gt "${TOTAL}" ]]; then
        break
    fi
    run_one "${gpu_id}" "${start_id}" "${end_id}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if [[ "${status}" -ne 0 ]]; then
    echo "Batch pure AnyFlow inference finished with failures."
    exit "${status}"
fi

echo "Batch pure AnyFlow inference finished successfully."
