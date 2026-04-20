#!/bin/bash
# =============================================================================
# Physics-Informed Video Generation Inference Script for Wan2.1-T2V-1.3B
# 物理约束视频生成推理脚本（单 expert 基模版本）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/dataset-assist-0/algorithm/cong.wang/miniconda3/envs/wan/bin/python}"

MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
PROMPT="${PROMPT:-A pineapple falls into the water.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量}"
ENABLE_PINN="${ENABLE_PINN:-1}"
CHECKPOINT_PATH="/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/models/train/wan21_stage2_fullpinn8/step-18500.pt"
OUTPUT_PATH="${OUTPUT_PATH:-./video_wan21_pineapple.mp4}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
SEED="${SEED:-0}"
CFG_SCALE="${CFG_SCALE:-5.0}"
FPS="${FPS:-15}"
QUALITY="${QUALITY:-5}"
AUTO_LABEL_FROM_PROMPT="${AUTO_LABEL_FROM_PROMPT:-1}"
DEVICE="${DEVICE:-cuda}"
OBSERVABLE_INSPECTION_ONLY="${OBSERVABLE_INSPECTION_ONLY:-0}"

cd "${REPO_ROOT}"

if [[ "${ENABLE_PINN}" != "0" && -n "${CHECKPOINT_PATH}" && ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "PINN checkpoint not found: ${CHECKPOINT_PATH}"
  echo "Train the explicit_attribute_bank_v2 adapter first, or override CHECKPOINT_PATH."
  exit 1
fi

INFER_CMD=(
  "${PYTHON_BIN}"
  examples/wanvideo/pinn_inference/inference_pinn.py
  --prompt "${PROMPT}"
  --negative_prompt "${NEGATIVE_PROMPT}"
  --model_id "${MODEL_ID}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_frames "${NUM_FRAMES}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --seed "${SEED}"
  --cfg_scale "${CFG_SCALE}"
  --output "${OUTPUT_PATH}"
  --fps "${FPS}"
  --quality "${QUALITY}"
  --device "${DEVICE}"
  # --disable_motion_weighted_attention
)

if [[ "${ENABLE_PINN}" != "0" && -n "${CHECKPOINT_PATH}" ]]; then
  INFER_CMD+=(--checkpoint_path "${CHECKPOINT_PATH}")
fi

if [[ "${AUTO_LABEL_FROM_PROMPT}" != "0" ]]; then
  INFER_CMD+=(--auto_label_from_prompt)
fi

if [[ "${OBSERVABLE_INSPECTION_ONLY}" != "0" ]]; then
  INFER_CMD+=(--observable_inspection_only)
fi

"${INFER_CMD[@]}"
