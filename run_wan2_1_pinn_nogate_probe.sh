#!/bin/bash
# Short probe for testing whether sigma gating is the primary bottleneck.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
BASE_TRAIN_SCRIPT="${REPO_ROOT}/examples/wanvideo/pinn_training/Wan2.1-T2V-1.3B-PINN.sh"

PINN_CHECKPOINT="${PINN_CHECKPOINT:-${REPO_ROOT}/models/train/pinn_plugin_wan21_t2v_1p3b_moe4/step-6600.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/pinn_plugin_wan21_t2v_1p3b_moe4_nogate_probe}"
PROBE_EXTRA_STEPS="${PROBE_EXTRA_STEPS:-1000}"
RESUME_STEP="${RESUME_STEP:-}"

if [[ -z "${RESUME_STEP}" ]]; then
  checkpoint_name="$(basename -- "${PINN_CHECKPOINT}")"
  if [[ "${checkpoint_name}" =~ ^step-([0-9]+)\.pt$ ]]; then
    RESUME_STEP="${BASH_REMATCH[1]}"
  else
    echo "Unable to infer RESUME_STEP from ${PINN_CHECKPOINT}. Set RESUME_STEP or MAX_STEPS explicitly."
    exit 1
  fi
fi

MAX_STEPS="${MAX_STEPS:-$((RESUME_STEP + PROBE_EXTRA_STEPS))}"

cd "${REPO_ROOT}"

OUTPUT_PATH="${OUTPUT_PATH}" \
PINN_CHECKPOINT="${PINN_CHECKPOINT}" \
PHYSICS_WEIGHT="${PHYSICS_WEIGHT:-0.30}" \
PHYSICS_WEIGHT_TARGET="${PHYSICS_WEIGHT_TARGET:-0.30}" \
PHYSICS_WARMUP_STEPS="${PHYSICS_WARMUP_STEPS:-0}" \
CONDITIONED_PHYSICS_WARMUP_STEPS="${CONDITIONED_PHYSICS_WARMUP_STEPS:-0}" \
STATE_ALIGN_WARMUP_STEPS="${STATE_ALIGN_WARMUP_STEPS:-0}" \
STATE_ALIGN_X_WEIGHT="${STATE_ALIGN_X_WEIGHT:-0.0}" \
STATE_ALIGN_V_WEIGHT="${STATE_ALIGN_V_WEIGHT:-0.0}" \
STATE_ALIGN_V_WEIGHT_TARGET="${STATE_ALIGN_V_WEIGHT_TARGET:-0.0}" \
USE_SIGMA_GATE="${USE_SIGMA_GATE:-0}" \
USE_SIGMA_CONDITIONING="${USE_SIGMA_CONDITIONING:-1}" \
MAX_STEPS="${MAX_STEPS}" \
SAVE_STEPS="${SAVE_STEPS:-200}" \
bash "${BASE_TRAIN_SCRIPT}"
