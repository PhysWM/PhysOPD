#!/bin/bash
# =============================================================================
# Physics-Informed Video Generation Inference Script for Wan2.1-T2V-1.3B
# 物理约束视频生成推理脚本（单 expert 基模版本）
# =============================================================================

set -euo pipefail

MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
PROMPT="${PROMPT:-water flowing down a rocky slope with physically plausible motion}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量}"
ENABLE_PINN="${ENABLE_PINN:-1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/models/train/pinn_plugin_wan21_t2v_1p3b_moe4/step-6600.pt}"
OUTPUT_PATH="${OUTPUT_PATH:-./video_wan21_t2v_1p3b_pinn.mp4}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
SEED="${SEED:-0}"
CFG_SCALE="${CFG_SCALE:-5.0}"
FPS="${FPS:-15}"
QUALITY="${QUALITY:-5}"
DEVICE="${DEVICE:-cuda}"
METADATA_JSON="${METADATA_JSON:-}"
METADATA_CSV="${METADATA_CSV:-}"
AUTO_LABEL_FROM_PROMPT="${AUTO_LABEL_FROM_PROMPT:-0}"
DISABLE_ATTENTION_OVERLAY="${DISABLE_ATTENTION_OVERLAY:-0}"
DISABLE_MOTION_WEIGHTED_ATTENTION="${DISABLE_MOTION_WEIGHTED_ATTENTION:-0}"
ATTENTION_ALPHA="${ATTENTION_ALPHA:-0.45}"
ATTENTION_MOTION_PERCENTILE="${ATTENTION_MOTION_PERCENTILE:-90.0}"
LLM_MODEL="${LLM_MODEL:-}"
LLM_BASE_URL="${LLM_BASE_URL:-}"
LLM_API_KEY="${LLM_API_KEY:-}"
LLM_API_KEY_ENV="${LLM_API_KEY_ENV:-OPENAI_API_KEY}"
LLM_TIMEOUT="${LLM_TIMEOUT:-30.0}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-2}"

INFER_CMD=(
  python examples/wanvideo/pinn_inference/inference_pinn.py
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
  --attention_alpha "${ATTENTION_ALPHA}"
  --attention_motion_percentile "${ATTENTION_MOTION_PERCENTILE}"
  --device "${DEVICE}"
  --llm_api_key_env "${LLM_API_KEY_ENV}"
  --llm_timeout "${LLM_TIMEOUT}"
  --llm_max_retries "${LLM_MAX_RETRIES}"
)

if [[ "${ENABLE_PINN}" != "0" && -n "${CHECKPOINT_PATH}" ]]; then
  INFER_CMD+=(--checkpoint_path "${CHECKPOINT_PATH}")
fi

if [[ -n "${METADATA_JSON}" ]]; then
  INFER_CMD+=(--metadata_json "${METADATA_JSON}")
fi

if [[ -n "${METADATA_CSV}" ]]; then
  INFER_CMD+=(--metadata_csv "${METADATA_CSV}")
fi

if [[ "${AUTO_LABEL_FROM_PROMPT}" != "0" ]]; then
  INFER_CMD+=(--auto_label_from_prompt)
fi

if [[ "${DISABLE_ATTENTION_OVERLAY}" != "0" ]]; then
  INFER_CMD+=(--disable_attention_overlay)
fi

if [[ "${DISABLE_MOTION_WEIGHTED_ATTENTION}" != "0" ]]; then
  INFER_CMD+=(--disable_motion_weighted_attention)
fi

if [[ -n "${LLM_MODEL}" ]]; then
  INFER_CMD+=(--llm_model "${LLM_MODEL}")
fi

if [[ -n "${LLM_BASE_URL}" ]]; then
  INFER_CMD+=(--llm_base_url "${LLM_BASE_URL}")
fi

if [[ -n "${LLM_API_KEY}" ]]; then
  INFER_CMD+=(--llm_api_key "${LLM_API_KEY}")
fi

"${INFER_CMD[@]}"
