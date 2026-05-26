#!/bin/bash
# =============================================================================
# Three-stage PhysicsAdapter training for AnyFlow-Wan2.1-T2V-1.3B.
#
# This mirrors the staged PILA/Wan recipe, with only the frozen backbone replaced
# by AnyFlow:
#   Stage 1: observable_pretrain
#   Stage 2: encoder_completion
#   Stage 3: full_pinn
#
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   OUTPUT_ROOT=./models/train/anyflow_wan21_1p3b_3stage \
#   bash examples/wanvideo/pinn_training/Wan2.1-T2V-1.3B-AnyFlow-PINN-3Stage.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
SINGLE_STAGE_SCRIPT="${SCRIPT_DIR}/Wan2.1-T2V-1.3B-AnyFlow-PINN-Train.sh"

cd "${REPO_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-./models/train/anyflow_wan21_1p3b_3stage}"
STAGE1_OUTPUT_PATH="${STAGE1_OUTPUT_PATH:-${OUTPUT_ROOT}/stage1_observable}"
ENCODER_OUTPUT_PATH="${ENCODER_OUTPUT_PATH:-${OUTPUT_ROOT}/stage2_encoder_completion}"
FULL_OUTPUT_PATH="${FULL_OUTPUT_PATH:-${OUTPUT_ROOT}/stage3_fullpinn}"

STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-${MAX_STEPS:-5000}}"
ENCODER_MAX_STEPS="${ENCODER_MAX_STEPS:-${MAX_STEPS:-5000}}"
FULL_MAX_STEPS="${FULL_MAX_STEPS:-${MAX_STEPS:-10000}}"
SAVE_STEPS="${SAVE_STEPS:-500}"

FIELD_RECOVERY_PHASE="${FIELD_RECOVERY_PHASE:-psi}"
FIELD_RECOVERY_STEP_SCHEDULE="${FIELD_RECOVERY_STEP_SCHEDULE:-core:0,alpha:1000,T:2000,j:3000,D:4000,psi:4500}"
SECONDARY_FIELD_STRATEGY="${SECONDARY_FIELD_STRATEGY:-u_first_constructor}"

stage_checkpoint() {
  local output_path="$1"
  local preferred_step="$2"
  if [[ -n "${preferred_step}" && -f "${output_path}/step-${preferred_step}.pt" ]]; then
    printf '%s\n' "${output_path}/step-${preferred_step}.pt"
    return 0
  fi
  if [[ -f "${output_path}/final.pt" ]]; then
    printf '%s\n' "${output_path}/final.pt"
    return 0
  fi
  local latest
  latest="$(ls -1 "${output_path}"/step-*.pt 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${latest}" ]]; then
    printf '%s\n' "${latest}"
    return 0
  fi
  echo "No checkpoint found under ${output_path}" >&2
  return 1
}

echo "================ Stage 1: observable_pretrain ================"
TRAINING_STAGE=observable_pretrain \
OUTPUT_PATH="${STAGE1_OUTPUT_PATH}" \
MAX_STEPS="${STAGE1_MAX_STEPS}" \
SAVE_STEPS="${SAVE_STEPS}" \
bash "${SINGLE_STAGE_SCRIPT}"

STAGE1_CKPT="$(stage_checkpoint "${STAGE1_OUTPUT_PATH}" "${STAGE1_MAX_STEPS}")"
echo "Stage 1 checkpoint: ${STAGE1_CKPT}"

echo "================ Stage 2: encoder_completion ================"
TRAINING_STAGE=encoder_completion \
STAGE1_PRETRAINED_ENCODER="${STAGE1_CKPT}" \
OUTPUT_PATH="${ENCODER_OUTPUT_PATH}" \
MAX_STEPS="${ENCODER_MAX_STEPS}" \
SAVE_STEPS="${SAVE_STEPS}" \
FIELD_RECOVERY_PHASE="${FIELD_RECOVERY_PHASE}" \
FIELD_RECOVERY_STEP_SCHEDULE="${FIELD_RECOVERY_STEP_SCHEDULE}" \
SECONDARY_FIELD_STRATEGY="${SECONDARY_FIELD_STRATEGY}" \
bash "${SINGLE_STAGE_SCRIPT}"

ENCODER_CKPT="$(stage_checkpoint "${ENCODER_OUTPUT_PATH}" "${ENCODER_MAX_STEPS}")"
echo "Encoder completion checkpoint: ${ENCODER_CKPT}"

echo "================ Stage 3: full_pinn ================"
TRAINING_STAGE=full_pinn \
STAGE1_PRETRAINED_ENCODER="${ENCODER_CKPT}" \
OUTPUT_PATH="${FULL_OUTPUT_PATH}" \
MAX_STEPS="${FULL_MAX_STEPS}" \
SAVE_STEPS="${SAVE_STEPS}" \
bash "${SINGLE_STAGE_SCRIPT}"

echo "Done. Final full_pinn output: ${FULL_OUTPUT_PATH}"
