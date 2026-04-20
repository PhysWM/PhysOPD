#!/bin/bash
# =============================================================================
# Strict stage1 observable inspection for Wan2.1 PINN
# 复用 run_wan2_1_pinn.sh 的环境风格，只做训练态 x0_hat 可视化检查
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/dataset-assist-0/algorithm/cong.wang/miniconda3/envs/wan/bin/python}"

CHECKPOINT_PATH="/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/models/train/wan21_stage2_fullpinn/step-2000.pt"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/models/train/wan21_stage2_fullpinn/step-2000_physics_inspection_fixed}"

DATASET_BASE_PATH="${DATASET_BASE_PATH:-/home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-/home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data/metadata_standard.csv}"
MODEL_ID_WITH_ORIGIN_PATHS="${MODEL_ID_WITH_ORIGIN_PATHS:-Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth}"

HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
DEVICE="${DEVICE:-cuda}"
SCAN_COUNT="${SCAN_COUNT:-16}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
SAMPLE_INDICES="${SAMPLE_INDICES:-}"
TIMESTEP_FRACTION="${TIMESTEP_FRACTION:-}"
MIN_TIMESTEP_BOUNDARY="${MIN_TIMESTEP_BOUNDARY:-}"
MAX_TIMESTEP_BOUNDARY="${MAX_TIMESTEP_BOUNDARY:-}"
FLOW_BACKBONE_CKPT="${FLOW_BACKBONE_CKPT:-}"
SEED="${SEED:-1234}"

cd "${REPO_ROOT}"

CMD=(
  "${PYTHON_BIN}"
  examples/wanvideo/pinn_training/inspect_stage1_observable_strict.py
  --checkpoint "${CHECKPOINT_PATH}"
  --dataset_base_path "${DATASET_BASE_PATH}"
  --dataset_metadata_path "${DATASET_METADATA_PATH}"
  --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN_PATHS}"
  --output_dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_frames "${NUM_FRAMES}"
  --scan_count "${SCAN_COUNT}"
  --num_samples "${NUM_SAMPLES}"
  --seed "${SEED}"
)

if [[ -n "${SAMPLE_INDICES}" ]]; then
  CMD+=(--sample_indices "${SAMPLE_INDICES}")
fi

if [[ -n "${TIMESTEP_FRACTION}" ]]; then
  CMD+=(--timestep_fraction "${TIMESTEP_FRACTION}")
fi

if [[ -n "${MIN_TIMESTEP_BOUNDARY}" ]]; then
  CMD+=(--min_timestep_boundary "${MIN_TIMESTEP_BOUNDARY}")
fi

if [[ -n "${MAX_TIMESTEP_BOUNDARY}" ]]; then
  CMD+=(--max_timestep_boundary "${MAX_TIMESTEP_BOUNDARY}")
fi

if [[ -n "${FLOW_BACKBONE_CKPT}" ]]; then
  CMD+=(--flow_backbone_ckpt "${FLOW_BACKBONE_CKPT}")
fi

"${CMD[@]}"
