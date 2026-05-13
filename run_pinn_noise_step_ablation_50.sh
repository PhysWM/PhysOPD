#!/bin/bash
#
# 50-step noise-stage sensitivity ablation.
# Fixed total denoising steps = 50; only the PINN correction-active step range changes.
# LLM prompt refinement is enabled by default to match the main evaluation setting.
#
# Main configs:
#   No PINN, PINN 41-50, PINN 31-50, PINN 21-50, PINN 11-50
# PINN 1-50 is skipped by default because the full baseline is usually already available.
#
# Optional diagnostic same-budget windows:
#   INCLUDE_STAGE_WINDOWS=1 bash run_pinn_noise_step_ablation_50.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

CSV="${CSV:-${REPO_ROOT}/phygenbench_prompts.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/pinn_noise_step_ablation_50_llm_refine}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
DISABLE_PROMPT_REFINEMENT="${DISABLE_PROMPT_REFINEMENT:-0}"

CONFIGS=(
    "no_pinn:none"
    "pinn_41_50:41-50"
    "pinn_31_50:31-50"
    "pinn_21_50:21-50"
    "pinn_11_50:11-50"
)

if [[ "${INCLUDE_FULL_1_50:-0}" != "0" ]]; then
    CONFIGS+=("pinn_1_50:1-50")
fi

if [[ "${INCLUDE_STAGE_WINDOWS:-0}" != "0" ]]; then
    CONFIGS+=(
        "window_1_10:1-10"
        "window_11_20:11-20"
        "window_21_30:21-30"
        "window_31_40:31-40"
        "window_41_50:41-50"
    )
fi

mkdir -p "${OUTPUT_ROOT}"

echo "50-step PINN noise-stage sensitivity ablation"
echo "CSV: ${CSV}"
echo "Output root: ${OUTPUT_ROOT}"
echo "num_inference_steps: ${NUM_INFERENCE_STEPS}"
echo "GPU IDs: ${GPU_IDS}"
echo "NUM_GPUS: ${NUM_GPUS}"
if [[ "${DISABLE_PROMPT_REFINEMENT}" == "0" ]]; then
    echo "LLM prompt refinement: enabled"
else
    echo "LLM prompt refinement: disabled"
fi
echo "Configs: ${#CONFIGS[@]}"

for item in "${CONFIGS[@]}"; do
    name="${item%%:*}"
    range="${item#*:}"
    run_dir="${OUTPUT_ROOT}/${name}"

    echo
    echo "================================================================"
    echo "Running ${name}: PINN_STEP_RANGE=${range}"
    echo "Output: ${run_dir}"
    echo "================================================================"

    CSV="${CSV}" \
    OUTPUT_DIR="${run_dir}" \
    LOG_DIR="${run_dir}/logs" \
    PERFORMANCE_METRICS_PATH="${run_dir}/performance_metrics.jsonl" \
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
    GPU_IDS="${GPU_IDS}" \
    NUM_GPUS="${NUM_GPUS}" \
    DISABLE_PROMPT_REFINEMENT="${DISABLE_PROMPT_REFINEMENT}" \
    PINN_STEP_RANGE="${range}" \
    bash "${REPO_ROOT}/batch_inference_wan21_step_ablation_8gpu.sh"
done

echo
echo "All step-ablation configs finished: ${OUTPUT_ROOT}"
