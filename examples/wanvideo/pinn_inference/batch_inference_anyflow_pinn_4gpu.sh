#!/bin/bash
#
# 4-GPU PhysGenBench inference for AnyFlow + PILA PhysicsAdapter.
#
# Common usage:
#   bash examples/wanvideo/pinn_inference/batch_inference_anyflow_pinn_4gpu.sh
#   TOTAL=32 bash examples/wanvideo/pinn_inference/batch_inference_anyflow_pinn_4gpu.sh
#   GPU_IDS=0,1,2,3 NUM_INFERENCE_STEPS=8 bash examples/wanvideo/pinn_inference/batch_inference_anyflow_pinn_4gpu.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f ".pinn_api.env" ]]; then
    set -a
    source ".pinn_api.env"
    set +a
fi

PYTHON_BIN="${PYTHON_BIN:-/home/dataset-assist-0/algorithm/cong.wang/miniconda3/envs/anyflow/bin/python}"
CSV="${CSV:-/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/phygenbench_prompts.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/phygenbench_anyflow_pila_14b_steps50}"
ANYFLOW_ROOT="${ANYFLOW_ROOT:-/home/dataset-assist-0/algorithm/cong.wang/try/AnyFlow}"
PILA_ROOT="${PILA_ROOT:-${REPO_ROOT}}"
MODEL_PATH="${MODEL_PATH:-${ANYFLOW_ROOT}/experiments/pretrained_models/AnyFlow-Wan2.1-T2V-14B-Diffusers}"
PINN_CHECKPOINT="${PINN_CHECKPOINT:-${CHECKPOINT_PATH:-${CHECKPOINT:-${REPO_ROOT}/outputs/anyflow_pinn_transfer/step-18500_adapter_slim.pt}}}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
SEED="${SEED:-0}"
FPS="${FPS:-16}"
DTYPE="${DTYPE:-bf16}"
CORRECTION_SCALE="${CORRECTION_SCALE:-1.0}"
MOE_TOP_K="${MOE_TOP_K:-}"
AUTO_LABEL_FROM_PROMPT="${AUTO_LABEL_FROM_PROMPT:-1}"
DISABLE_PROMPT_REFINEMENT="${DISABLE_PROMPT_REFINEMENT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RESUME="${RESUME:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
TOTAL="${TOTAL:-160}"
LLM_MODEL="${LLM_MODEL:-gpt-5.4}"
LLM_BASE_URL="${LLM_BASE_URL:-http://35.220.164.252:3888/v1}"
LLM_API_KEY="${LLM_API_KEY:-sk-8viAj2SPNHZ4W0E4BcKSfdOwXr1xVzpcheUHDIPweBi4EEqB}"
LLM_API_KEY_ENV="${LLM_API_KEY_ENV:-OPENAI_API_KEY}"
LLM_TIMEOUT="${LLM_TIMEOUT:-30}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-2}"
DEFAULT_LABEL="${DEFAULT_LABEL:-Fluid}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python not executable: ${PYTHON_BIN}"
    exit 1
fi
if [[ ! -f "${CSV}" ]]; then
    echo "CSV not found: ${CSV}"
    exit 1
fi
if [[ ! -f "${PINN_CHECKPOINT}" ]]; then
    echo "PINN checkpoint not found: ${PINN_CHECKPOINT}"
    echo "Expected slim checkpoint by default. You can override PINN_CHECKPOINT=/path/to/adapter_slim.pt"
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
    --pila_root "${PILA_ROOT}"
    --model_path "${MODEL_PATH}"
    --pinn_checkpoint "${PINN_CHECKPOINT}"
    --negative_prompt "${NEGATIVE_PROMPT}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --num_frames "${NUM_FRAMES}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --seed "${SEED}"
    --fps "${FPS}"
    --dtype "${DTYPE}"
    --correction_scale "${CORRECTION_SCALE}"
    --llm_model "${LLM_MODEL}"
    --llm_base_url "${LLM_BASE_URL}"
    --llm_api_key_env "${LLM_API_KEY_ENV}"
    --llm_timeout "${LLM_TIMEOUT}"
    --llm_max_retries "${LLM_MAX_RETRIES}"
    --default_label "${DEFAULT_LABEL}"
)

if [[ -n "${LLM_API_KEY}" ]]; then
    COMMON_ARGS+=(--llm_api_key "${LLM_API_KEY}")
fi
if [[ -n "${MOE_TOP_K}" ]]; then
    COMMON_ARGS+=(--moe_top_k "${MOE_TOP_K}")
fi
if [[ "${AUTO_LABEL_FROM_PROMPT}" != "0" ]]; then
    COMMON_ARGS+=(--auto_label_from_prompt)
fi
if [[ "${DISABLE_PROMPT_REFINEMENT}" != "0" ]]; then
    COMMON_ARGS+=(--disable_prompt_refinement)
fi
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
        "${PYTHON_BIN}" examples/wanvideo/pinn_inference/batch_inference_anyflow_pinn.py \
        "${COMMON_ARGS[@]}" \
        --start_id "${start_id}" \
        --end_id "${end_id}" \
        2>&1 | tee "${LOG_DIR}/gpu${gpu_id}_${start_id}_${end_id}.log"
}

echo "CSV: ${CSV}"
echo "Output: ${OUTPUT_DIR}"
echo "AnyFlow model: ${MODEL_PATH}"
echo "PINN checkpoint: ${PINN_CHECKPOINT}"
echo "GPU IDs: ${GPU_LIST[*]}"
echo "Total samples: ${TOTAL}"
echo "Chunk size: ${CHUNK}"
echo "Prompt refinement enabled: $([[ "${DISABLE_PROMPT_REFINEMENT}" == "0" ]] && echo yes || echo no)"
echo "Auto label routing enabled: $([[ "${AUTO_LABEL_FROM_PROMPT}" != "0" ]] && echo yes || echo no)"

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
    echo "Batch AnyFlow + PILA inference finished with failures."
    exit "${status}"
fi

echo "Batch AnyFlow + PILA inference finished successfully."
