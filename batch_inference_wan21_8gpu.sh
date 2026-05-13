#!/bin/bash
#
# 批量读取 CSV 中的 caption，并按顺序分配到多张 GPU 并行生成。
#
# 常用用法:
#   bash batch_inference_wan21_8gpu.sh
#   NUM_GPUS=4 bash batch_inference_wan21_8gpu.sh
#   GPU_IDS=0,2,5 bash batch_inference_wan21_8gpu.sh
#   OUTPUT_DIR=output_videos_test CHECKPOINT_PATH=xxx.pt bash batch_inference_wan21_8gpu.sh
#
# 环境变量:
#   PYTHON_BIN         Python 路径
#   CSV                输入 CSV，默认 videophy_test_public.csv
#   OUTPUT_DIR         输出目录
#   CHECKPOINT_PATH    PINN checkpoint 路径
#   CHECKPOINT         CHECKPOINT_PATH 的兼容别名
#   MODEL_ID           模型 ID
#   MOE_TOP_K          运行时覆盖 MoE active expert count；允许 0
#   EXCLUDED_EXPERT_NAMES  逗号分隔的运行时排除专家，例如 Granular,Fracture
#   PERFORMANCE_METRICS_PATH  每条样本效率 JSONL；默认 OUTPUT_DIR/performance_metrics.jsonl
#   NUM_GPUS           使用前 NUM_GPUS 张卡（基于 GPU_IDS 顺序裁剪）
#   GPU_IDS            使用哪些卡，逗号分隔，例如 0,1,2,3
#   LOG_DIR            若设置，则每张卡输出到对应日志文件
#   TOTAL              手动指定总样本数；默认自动从 CSV 统计
#

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

if [[ -f ".pinn_api.env" ]]; then
    set -a
    source ".pinn_api.env"
    set +a
fi

PYTHON_BIN="${PYTHON_BIN:-/home/dataset-assist-0/algorithm/cong.wang/miniconda3/envs/wan/bin/python}"
PYTHON_ENV_ROOT="$(cd -- "$(dirname "${PYTHON_BIN}")/.." && pwd)"
if [[ -z "${CUDA_HOME:-}" && -x "${PYTHON_ENV_ROOT}/bin/nvcc" ]]; then
    export CUDA_HOME="${PYTHON_ENV_ROOT}"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi
CSV="${CSV:-/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/phygenbench_prompts.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/phygenbench_videos_wan21_pinn_batch2}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT:-/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/models/train/wan21_stage2_fullpinn8/step-18000.pt}}"
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
SEED="${SEED:-${SEED_BASE:-0}}"
CFG_SCALE="${CFG_SCALE:-5.0}"
FPS="${FPS:-15}"
QUALITY="${QUALITY:-5}"
MOE_TOP_K="${MOE_TOP_K:-}"
EXCLUDED_EXPERT_NAMES="${EXCLUDED_EXPERT_NAMES:-}"
PERFORMANCE_METRICS_PATH="${PERFORMANCE_METRICS_PATH:-}"
DEVICE="${DEVICE:-cuda}"
AUTO_LABEL_FROM_PROMPT="${AUTO_LABEL_FROM_PROMPT:-1}"
DISABLE_PROMPT_REFINEMENT="${DISABLE_PROMPT_REFINEMENT:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RESUME="${RESUME:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-}"
LOG_DIR="${LOG_DIR:-}"
TOTAL="${TOTAL:-}"

if [[ ! -f "${CSV}" ]]; then
    echo "CSV not found: ${CSV}"
    exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "PINN checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

IFS=',' read -r -a ALL_GPU_IDS <<< "${GPU_IDS}"

if [[ -n "${NUM_GPUS}" ]]; then
    if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -le 0 ]]; then
        echo "NUM_GPUS must be a positive integer, got: ${NUM_GPUS}"
        exit 1
    fi
    if [[ "${NUM_GPUS}" -gt "${#ALL_GPU_IDS[@]}" ]]; then
        echo "NUM_GPUS=${NUM_GPUS} exceeds available GPU_IDS count ${#ALL_GPU_IDS[@]} (${GPU_IDS})"
        exit 1
    fi
    GPU_LIST=("${ALL_GPU_IDS[@]:0:${NUM_GPUS}}")
else
    GPU_LIST=("${ALL_GPU_IDS[@]}")
fi

GPU_COUNT="${#GPU_LIST[@]}"
if [[ "${GPU_COUNT}" -le 0 ]]; then
    echo "No GPU selected. Check GPU_IDS/NUM_GPUS."
    exit 1
fi

if [[ -z "${TOTAL}" ]]; then
    TOTAL="$("${PYTHON_BIN}" -c 'import csv, sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"))))' "${CSV}")"
fi

if ! [[ "${TOTAL}" =~ ^[0-9]+$ ]] || [[ "${TOTAL}" -le 0 ]]; then
    echo "TOTAL must be a positive integer, got: ${TOTAL}"
    exit 1
fi

CHUNK=$(( (TOTAL + GPU_COUNT - 1) / GPU_COUNT ))

COMMON_ARGS=(
    --csv "${CSV}"
    --output_dir "${OUTPUT_DIR}"
    --checkpoint_path "${CHECKPOINT_PATH}"
    --model_id "${MODEL_ID}"
    --negative_prompt "${NEGATIVE_PROMPT}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --num_frames "${NUM_FRAMES}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --seed "${SEED}"
    --cfg_scale "${CFG_SCALE}"
    --fps "${FPS}"
    --quality "${QUALITY}"
    --device "${DEVICE}"
)

if [[ -n "${MOE_TOP_K}" ]]; then
    COMMON_ARGS+=(--moe_top_k "${MOE_TOP_K}")
fi

if [[ -n "${EXCLUDED_EXPERT_NAMES}" ]]; then
    COMMON_ARGS+=(--excluded_expert_names "${EXCLUDED_EXPERT_NAMES}")
fi

if [[ -n "${PERFORMANCE_METRICS_PATH}" ]]; then
    COMMON_ARGS+=(--performance_metrics_path "${PERFORMANCE_METRICS_PATH}")
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
    shift 3

    if [[ "${start_id}" -gt "${end_id}" ]]; then
        return 0
    fi

    echo "[GPU ${gpu_id}] Starting IDs ${start_id}-${end_id}"

    local cmd=(
        "${PYTHON_BIN}"
        examples/wanvideo/pinn_inference/batch_inference_pinn.py
        "${COMMON_ARGS[@]}"
        --start_id "${start_id}"
        --end_id "${end_id}"
        "$@"
    )

    if [[ -n "${LOG_DIR}" ]]; then
        mkdir -p "${LOG_DIR}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" 2>&1 | tee "${LOG_DIR}/gpu${gpu_id}_${start_id}_${end_id}.log"
    else
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}"
    fi
}

echo "CSV: ${CSV}"
echo "Output: ${OUTPUT_DIR}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Model: ${MODEL_ID}"
echo "MoE top-k: ${MOE_TOP_K:-checkpoint default}"
echo "Excluded experts: ${EXCLUDED_EXPERT_NAMES:-none}"
echo "Performance metrics: ${PERFORMANCE_METRICS_PATH:-${OUTPUT_DIR}/performance_metrics.jsonl}"
echo "GPU IDs: ${GPU_LIST[*]}"
echo "Total samples: ${TOTAL}"
echo "Chunk size: ${CHUNK}"

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

    run_one "${gpu_id}" "${start_id}" "${end_id}" "$@" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if [[ "${status}" -ne 0 ]]; then
    echo "Batch inference finished with failures."
    exit "${status}"
fi

echo "Batch inference finished successfully."
