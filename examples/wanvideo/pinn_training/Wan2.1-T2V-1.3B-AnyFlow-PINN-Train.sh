#!/bin/bash
# Train PILA PhysicsAdapter with the same staged recipe as Wan2.1-T2V-1.3B-PINN-2Stage.sh,
# but replace the frozen DiffSynth Wan backbone with AnyFlow-Wan2.1-T2V-1.3B.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
ACCELERATE_CONFIG_PATH="${ACCELERATE_CONFIG_PATH:-${SCRIPT_DIR}/accelerate_config_anyflow_pinn.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/dataset-assist-0/algorithm/cong.wang/miniconda3/envs/anyflow/bin/python}"

cd "${REPO_ROOT}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_IDS[@]}}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29531}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}:/home/dataset-assist-0/algorithm/cong.wang/try/AnyFlow${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

ANYFLOW_MODEL_PATH="${ANYFLOW_MODEL_PATH:-/home/dataset-assist-0/algorithm/cong.wang/try/AnyFlow/experiments/pretrained_models/AnyFlow-Wan2.1-T2V-1.3B-Diffusers}"
DATASET_BASE_PATH="${DATASET_BASE_PATH:-/home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-/home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data/metadata_standard.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/anyflow_wan21_1p3b_pinn}"
TRAINING_STAGE="${TRAINING_STAGE:-observable_pretrain}"
PHYSICS_WEIGHT="${PHYSICS_WEIGHT:-0.30}"
PHYSICS_WEIGHT_TARGET="${PHYSICS_WEIGHT_TARGET:-0.30}"
PHYSICS_WARMUP_STEPS="${PHYSICS_WARMUP_STEPS:-2000}"
EXPERT_PDE_SIGMA_THRESHOLD="${EXPERT_PDE_SIGMA_THRESHOLD:-0.40}"
EXPERT_PDE_SIGMA_THRESHOLD_TARGET="${EXPERT_PDE_SIGMA_THRESHOLD_TARGET:-1.00}"
CORRECTION_WEIGHT="${CORRECTION_WEIGHT:-0.01}"
OBSERVABLE_TARGET_MODE="${OBSERVABLE_TARGET_MODE:-flow_plus_deformation}"
if [[ "${TRAINING_STAGE}" == "encoder_completion" ]]; then
  SECONDARY_FIELD_STRATEGY="${SECONDARY_FIELD_STRATEGY:-u_first_constructor}"
else
  SECONDARY_FIELD_STRATEGY="${SECONDARY_FIELD_STRATEGY:-direct_bank}"
fi
FIELD_RECOVERY_PHASE="${FIELD_RECOVERY_PHASE:-core}"
FIELD_RECOVERY_STEP_SCHEDULE="${FIELD_RECOVERY_STEP_SCHEDULE:-}"
FIELD_RECOVERY_LOSS_RAMP_STEPS="${FIELD_RECOVERY_LOSS_RAMP_STEPS:-100}"
ENCODER_FREEZE_STEPS="${ENCODER_FREEZE_STEPS:-1000}"
CORE_ABLATION_MODE="${CORE_ABLATION_MODE:-full}"

TRAIN_CMD=(
  "${PYTHON_BIN}" -m accelerate.commands.launch
  --config_file "${ACCELERATE_CONFIG_PATH}"
  --num_processes "${NUM_PROCESSES}"
  --num_machines 1
  --machine_rank 0
  --main_process_ip "${MAIN_PROCESS_IP}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  "${SCRIPT_DIR}/train_anyflow_pinn.py"
  --pila_root "${REPO_ROOT}" \
  --anyflow_root "/home/dataset-assist-0/algorithm/cong.wang/try/AnyFlow" \
  --anyflow_model_path "${ANYFLOW_MODEL_PATH}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --height "${HEIGHT:-480}" \
  --width "${WIDTH:-832}" \
  --num_frames "${NUM_FRAMES:-81}" \
  --batch_size 1 \
  --num_workers "${NUM_WORKERS:-2}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --max_steps "${MAX_STEPS:-2000}" \
  --num_epochs "${NUM_EPOCHS:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --log_steps "${LOG_STEPS:-10}" \
  --adapter_hidden_dim "${ADAPTER_HIDDEN_DIM:-128}" \
  --moe_top_k "${MOE_TOP_K:-4}" \
  --training_stage "${TRAINING_STAGE}" \
  --observable_target_mode "${OBSERVABLE_TARGET_MODE}" \
  --secondary_field_strategy "${SECONDARY_FIELD_STRATEGY}" \
  --field_recovery_phase "${FIELD_RECOVERY_PHASE}" \
  --field_recovery_step_schedule "${FIELD_RECOVERY_STEP_SCHEDULE}" \
  --field_recovery_loss_ramp_steps "${FIELD_RECOVERY_LOSS_RAMP_STEPS}" \
  --encoder_freeze_steps "${ENCODER_FREEZE_STEPS}" \
  --core_ablation_mode "${CORE_ABLATION_MODE}" \
  --physics_weight "${PHYSICS_WEIGHT}" \
  --physics_weight_target "${PHYSICS_WEIGHT_TARGET}" \
  --physics_warmup_steps "${PHYSICS_WARMUP_STEPS}" \
  --expert_pde_sigma_threshold "${EXPERT_PDE_SIGMA_THRESHOLD}" \
  --expert_pde_sigma_threshold_target "${EXPERT_PDE_SIGMA_THRESHOLD_TARGET}" \
  --correction_weight "${CORRECTION_WEIGHT}" \
  --output_path "${OUTPUT_PATH}" \
  --find_unused_parameters \
  --ddp_timeout_seconds "${DDP_TIMEOUT_SECONDS:-3600}"
)

if [[ -n "${MAX_SAMPLES:-}" ]]; then
  TRAIN_CMD+=(--max_samples "${MAX_SAMPLES}")
fi

if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  TRAIN_CMD+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

if [[ -n "${PINN_CHECKPOINT:-}" ]]; then
  TRAIN_CMD+=(--pinn_checkpoint "${PINN_CHECKPOINT}")
fi

if [[ -n "${STAGE1_PRETRAINED_ENCODER:-}" ]]; then
  TRAIN_CMD+=(--stage1_pretrained_encoder "${STAGE1_PRETRAINED_ENCODER}")
fi

if [[ "${FREEZE_U_ENCODER_DURING_RECOVERY:-1}" == "0" ]]; then
  TRAIN_CMD+=(--no_freeze_u_encoder_during_recovery)
fi

if [[ "${ABLATE_DISABLE_MOE:-0}" == "1" ]]; then
  TRAIN_CMD+=(--ablate_disable_moe)
fi

if [[ "${ABLATE_DISABLE_CONDITIONED_PDE:-0}" == "1" ]]; then
  TRAIN_CMD+=(--ablate_disable_conditioned_pde)
fi

if [[ "${ABLATE_DISABLE_AUX_LOSSES:-0}" == "1" ]]; then
  TRAIN_CMD+=(--ablate_disable_aux_losses)
fi

if [[ "${ABLATE_LABEL_ONLY_ROUTER:-0}" == "1" ]]; then
  TRAIN_CMD+=(--ablate_label_only_router)
fi

"${TRAIN_CMD[@]}"
