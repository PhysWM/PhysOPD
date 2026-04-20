#!/bin/bash
# Compare baseline / current / candidate Wan2.1 PINN checkpoints on a fixed prompt set.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
INFER_SCRIPT="${REPO_ROOT}/run_wan2_1_pinn.sh"
COMPARE_REPORT_SCRIPT="${REPO_ROOT}/examples/wanvideo/pinn_inference/build_pinn_compare_report.py"
PYTHON_BIN="${PYTHON_BIN:-/home/dataset-assist-0/algorithm/cong.wang/miniconda3/envs/wan/bin/python}"

COMPARE_OUTPUT_DIR="${COMPARE_OUTPUT_DIR:-${REPO_ROOT}/outputs/wan21_pinn_compare5}"
CURRENT_CHECKPOINT_PATH="${CURRENT_CHECKPOINT_PATH:-${REPO_ROOT}/models/train/pinn_plugin_wan21_t2v_1p3b_explicit_attr_v4/pinn_plugin_final.pt}"
CANDIDATE_CHECKPOINT_PATH="${CANDIDATE_CHECKPOINT_PATH:-${REPO_ROOT}/models/train/pinn_plugin_wan21_t2v_1p3b_moe4_physics_push_v1/pinn_plugin_final.pt}"
SEED="${SEED:-0}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-5.0}"
FPS="${FPS:-15}"
QUALITY="${QUALITY:-5}"
ENABLE_PAIRED_REPORT="${ENABLE_PAIRED_REPORT:-1}"

mkdir -p "${COMPARE_OUTPUT_DIR}"

declare -A PROMPTS=(
  [apple]="An apple falls into the river."
  [burger]="A cheeseburger dropping onto a plate with realistic deformation and bounce."
)

run_variant() {
  local prompt_key="$1"
  local variant_name="$2"
  local enable_pinn="$3"
  local checkpoint_path="$4"
  local output_path="${COMPARE_OUTPUT_DIR}/${prompt_key}_${variant_name}.mp4"

  PROMPT="${PROMPTS[${prompt_key}]}" \
  OUTPUT_PATH="${output_path}" \
  ENABLE_PINN="${enable_pinn}" \
  CHECKPOINT_PATH="${checkpoint_path}" \
  SEED="${SEED}" \
  HEIGHT="${HEIGHT}" \
  WIDTH="${WIDTH}" \
  NUM_FRAMES="${NUM_FRAMES}" \
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  CFG_SCALE="${CFG_SCALE}" \
  FPS="${FPS}" \
  QUALITY="${QUALITY}" \
  bash "${INFER_SCRIPT}"
}

build_paired_report() {
  local prompt_key="$1"
  local variant_name="$2"
  local baseline_video="${COMPARE_OUTPUT_DIR}/${prompt_key}_baseline.mp4"
  local pinn_video="${COMPARE_OUTPUT_DIR}/${prompt_key}_${variant_name}.mp4"
  local pinn_trace="${COMPARE_OUTPUT_DIR}/${prompt_key}_${variant_name}_physics_report_physics_trace.npz"
  local output_prefix="${COMPARE_OUTPUT_DIR}/${prompt_key}_${variant_name}_vs_baseline"

  if [[ "${ENABLE_PAIRED_REPORT}" == "0" ]]; then
    return 0
  fi
  if [[ ! -f "${baseline_video}" ]]; then
    echo "Skipping paired report for ${prompt_key}/${variant_name}: baseline missing at ${baseline_video}"
    return 0
  fi
  if [[ ! -f "${pinn_video}" ]]; then
    echo "Skipping paired report for ${prompt_key}/${variant_name}: PINN video missing at ${pinn_video}"
    return 0
  fi
  if [[ ! -f "${pinn_trace}" ]]; then
    echo "Skipping paired report for ${prompt_key}/${variant_name}: physics trace missing at ${pinn_trace}"
    return 0
  fi

  "${PYTHON_BIN}" "${COMPARE_REPORT_SCRIPT}" \
    --baseline_video "${baseline_video}" \
    --pinn_video "${pinn_video}" \
    --pinn_trace "${pinn_trace}" \
    --output_prefix "${output_prefix}" \
    --fps "${FPS}" \
    --quality "${QUALITY}"
}

for prompt_key in apple burger; do
  run_variant "${prompt_key}" baseline 0 ""
  run_variant "${prompt_key}" current 1 "${CURRENT_CHECKPOINT_PATH}"
  build_paired_report "${prompt_key}" current
  if [[ -f "${CANDIDATE_CHECKPOINT_PATH}" ]]; then
    run_variant "${prompt_key}" candidate 1 "${CANDIDATE_CHECKPOINT_PATH}"
    build_paired_report "${prompt_key}" candidate
  else
    echo "Skipping candidate for ${prompt_key}: checkpoint not found at ${CANDIDATE_CHECKPOINT_PATH}"
  fi
done
