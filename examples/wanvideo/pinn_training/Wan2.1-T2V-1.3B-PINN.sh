#!/bin/bash
# =============================================================================
# Physics-Informed Video Generation Training Script for Wan2.1-T2V-1.3B
# 物理约束视频生成训练脚本（单 expert 基模版本）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
ACCELERATE_CONFIG_PATH="${SCRIPT_DIR}/accelerate_config_pinn.yaml"
TRAIN_SCRIPT_PATH="${SCRIPT_DIR}/train_pinn.py"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${GPU_COUNT}" -le 0 ]]; then
    echo "No visible GPUs detected. Please set CUDA_VISIBLE_DEVICES explicitly."
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPU_COUNT - 1)))"
  export CUDA_VISIBLE_DEVICES
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${#GPU_IDS[@]}"

export MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-${TORCH_NCCL_ASYNC_ERROR_HANDLING}}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

if pgrep -af "accelerate launch --config_file ${ACCELERATE_CONFIG_PATH}" >/dev/null; then
  echo "Detected an existing PINN accelerate job. Stop old processes first to avoid contention."
  exit 1
fi

DATASET_BASE_PATH="${DATASET_BASE_PATH:-/home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-/home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data/metadata_standard.csv}"
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-2}"
DIAGNOSTIC_METRICS_INTERVAL="${DIAGNOSTIC_METRICS_INTERVAL:-100}"
HEARTBEAT_LOG_STEPS="${HEARTBEAT_LOG_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_TIMESTEP_BOUNDARY="${MAX_TIMESTEP_BOUNDARY:-1.0}"
MIN_TIMESTEP_BOUNDARY="${MIN_TIMESTEP_BOUNDARY:-0.0}"
PHYSICS_WEIGHT="${PHYSICS_WEIGHT:-0.30}"
PHYSICS_WEIGHT_TARGET="${PHYSICS_WEIGHT_TARGET:-0.30}"
PHYSICS_WARMUP_STEPS="${PHYSICS_WARMUP_STEPS:-2000}"
CONDITIONED_PHYSICS_WARMUP_STEPS="${CONDITIONED_PHYSICS_WARMUP_STEPS:-1000}"
OUTPUT_PHYSICS_WEIGHT="${OUTPUT_PHYSICS_WEIGHT:-0.0}"
MOE_TOP_K="${MOE_TOP_K:-4}"
CONDITION_CONSISTENCY_WEIGHT="${CONDITION_CONSISTENCY_WEIGHT:-0.0}"
STATE_ALIGN_WARMUP_STEPS="${STATE_ALIGN_WARMUP_STEPS:-1000}"
STATE_ALIGN_X_WEIGHT="${STATE_ALIGN_X_WEIGHT:-0.0}"
STATE_ALIGN_V_WEIGHT="${STATE_ALIGN_V_WEIGHT:-0.0}"
STATE_ALIGN_V_WEIGHT_TARGET="${STATE_ALIGN_V_WEIGHT_TARGET:-0.0}"
CURRICULUM_TRANSITION_START_STEP="${CURRICULUM_TRANSITION_START_STEP:-1000}"
CURRICULUM_TRANSITION_STEPS="${CURRICULUM_TRANSITION_STEPS:-1000}"
ADAPTER_HIDDEN_DIM="${ADAPTER_HIDDEN_DIM:-128}"
PHYSICS_ATTR_DIM="${PHYSICS_ATTR_DIM:-32}"
EXPERT_PDE_SIGMA_THRESHOLD="${EXPERT_PDE_SIGMA_THRESHOLD:-0.40}"
EXPERT_PDE_SIGMA_THRESHOLD_TARGET="${EXPERT_PDE_SIGMA_THRESHOLD_TARGET:-1.00}"
PHYSICS_STATE_MODE="${PHYSICS_STATE_MODE:-x0_hat}"
USE_SIGMA_GATE="${USE_SIGMA_GATE:-1}"
SIGMA_GATE_CURVE="${SIGMA_GATE_CURVE:-linear}"
USE_SIGMA_CONDITIONING="${USE_SIGMA_CONDITIONING:-1}"
SIGMA_GATE_FLOOR="${SIGMA_GATE_FLOOR:-0.30}"
ABLATION_PRESET="${ABLATION_PRESET:-legacy_direct_bank}"
OBSERVABLE_TARGET_MODE="${OBSERVABLE_TARGET_MODE:-auto}"
SECONDARY_FIELD_STRATEGY="${SECONDARY_FIELD_STRATEGY:-auto}"
ACTIVE_FIELD_SET="${ACTIVE_FIELD_SET:-auto}"
FIELD_ENABLE_SCHEDULE="${FIELD_ENABLE_SCHEDULE:-auto}"
FIELD_RECOVERY_PHASE="${FIELD_RECOVERY_PHASE:-core}"
FIELD_RECOVERY_STEP_SCHEDULE="${FIELD_RECOVERY_STEP_SCHEDULE:-}"
FIELD_RECOVERY_LOSS_RAMP_STEPS="${FIELD_RECOVERY_LOSS_RAMP_STEPS:-100}"
RUN_FULL_PINN_AFTER_RECOVERY="${RUN_FULL_PINN_AFTER_RECOVERY:-0}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-}"
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/pinn_plugin_wan21_t2v_1p3b_explicit_attr_v5}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
PINN_CHECKPOINT="${PINN_CHECKPOINT:-/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/models/train/pinn_plugin_wan21_t2v_1p3b_explicit_attr_v4/pinn_plugin_final.pt}"

if [[ "${USE_SIGMA_GATE}" == "0" ]]; then
  USE_SIGMA_GATE_FLAG="--no_use_sigma_gate"
else
  USE_SIGMA_GATE_FLAG="--use_sigma_gate"
fi

if [[ "${USE_SIGMA_CONDITIONING}" == "0" ]]; then
  USE_SIGMA_CONDITIONING_FLAG="--no_use_sigma_conditioning"
else
  USE_SIGMA_CONDITIONING_FLAG="--use_sigma_conditioning"
fi

ACCELERATE_LAUNCH_ARGS=(
  --config_file "${ACCELERATE_CONFIG_PATH}"
  --num_processes "${NUM_PROCESSES}"
  --num_machines 1
  --machine_rank 0
  --main_process_ip "${MAIN_PROCESS_IP}"
  --main_process_port "${MAIN_PROCESS_PORT}"
)

TRAIN_CMD=(
  accelerate launch
  "${ACCELERATE_LAUNCH_ARGS[@]}"
  "${TRAIN_SCRIPT_PATH}"
  --dataset_base_path "${DATASET_BASE_PATH}"
  --dataset_metadata_path "${DATASET_METADATA_PATH}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_frames "${NUM_FRAMES}"
  --dataset_repeat "${DATASET_REPEAT}"
  --model_id_with_origin_paths "${MODEL_ID}:diffusion_pytorch_model*.safetensors,${MODEL_ID}:models_t5_umt5-xxl-enc-bf16.pth,${MODEL_ID}:Wan2.1_VAE.pth"
  --learning_rate "${LEARNING_RATE}"
  --num_epochs "${NUM_EPOCHS}"
  --output_path "${OUTPUT_PATH}"
  --max_timestep_boundary "${MAX_TIMESTEP_BOUNDARY}"
  --min_timestep_boundary "${MIN_TIMESTEP_BOUNDARY}"
  --physics_weight "${PHYSICS_WEIGHT}"
  --physics_weight_target "${PHYSICS_WEIGHT_TARGET}"
  --physics_warmup_steps "${PHYSICS_WARMUP_STEPS}"
  --conditioned_physics_warmup_steps "${CONDITIONED_PHYSICS_WARMUP_STEPS}"
  --output_physics_weight "${OUTPUT_PHYSICS_WEIGHT}"
  --condition_consistency_weight "${CONDITION_CONSISTENCY_WEIGHT}"
  --state_align_warmup_steps "${STATE_ALIGN_WARMUP_STEPS}"
  --state_align_x_weight "${STATE_ALIGN_X_WEIGHT}"
  --state_align_v_weight "${STATE_ALIGN_V_WEIGHT}"
  --state_align_v_weight_target "${STATE_ALIGN_V_WEIGHT_TARGET}"
  --curriculum_transition_start_step "${CURRICULUM_TRANSITION_START_STEP}"
  --curriculum_transition_steps "${CURRICULUM_TRANSITION_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --adapter_hidden_dim "${ADAPTER_HIDDEN_DIM}"
  --physics_attr_dim "${PHYSICS_ATTR_DIM}"
  --expert_pde_sigma_threshold "${EXPERT_PDE_SIGMA_THRESHOLD}"
  --expert_pde_sigma_threshold_target "${EXPERT_PDE_SIGMA_THRESHOLD_TARGET}"
  --physics_state_mode "${PHYSICS_STATE_MODE}"
  --ablation_preset "${ABLATION_PRESET}"
  --observable_target_mode "${OBSERVABLE_TARGET_MODE}"
  --secondary_field_strategy "${SECONDARY_FIELD_STRATEGY}"
  --active_field_set "${ACTIVE_FIELD_SET}"
  --field_enable_schedule "${FIELD_ENABLE_SCHEDULE}"
  --field_recovery_phase "${FIELD_RECOVERY_PHASE}"
  --field_recovery_step_schedule "${FIELD_RECOVERY_STEP_SCHEDULE}"
  --field_recovery_loss_ramp_steps "${FIELD_RECOVERY_LOSS_RAMP_STEPS}"
  ${USE_SIGMA_GATE_FLAG}
  --sigma_gate_curve "${SIGMA_GATE_CURVE}"
  ${USE_SIGMA_CONDITIONING_FLAG}
  --sigma_gate_floor "${SIGMA_GATE_FLOOR}"
  --disable_adaptive_condition_injection
  --disable_rl_expert_optimization
  --no_use_dual_noise_experts
  --moe_top_k "${MOE_TOP_K}"
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --diagnostic_metrics_interval "${DIAGNOSTIC_METRICS_INTERVAL}"
  --heartbeat_log_steps "${HEARTBEAT_LOG_STEPS}"
  --tensorboard_log_steps 1
  --find_unused_parameters
)

if [[ -n "${PINN_CHECKPOINT}" ]]; then
  TRAIN_CMD+=(--pinn_checkpoint "${PINN_CHECKPOINT}")
fi

if [[ "${RUN_FULL_PINN_AFTER_RECOVERY}" == "1" ]]; then
  TRAIN_CMD+=(--run_full_pinn_after_recovery)
fi

if [[ -n "${MAX_STEPS}" ]]; then
  TRAIN_CMD+=(--max_steps "${MAX_STEPS}")
fi

if [[ -n "${EXTRA_FLAGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_FLAG_ARRAY=( ${EXTRA_FLAGS} )
  TRAIN_CMD+=("${EXTRA_FLAG_ARRAY[@]}")
fi

"${TRAIN_CMD[@]}"
