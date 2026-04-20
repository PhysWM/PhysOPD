#!/bin/bash
# =============================================================================
# Physics-Informed Video Generation Training Script
# 物理约束视频生成训练脚本（多卡并行）
#
# 使用方法:
#   bash examples/wanvideo/pinn_training/Wan2.2-T2V-A14B-PINN.sh
#
# 注意:
#   - 原始 Wan 模型参数完全冻结，只训练 PINN 插件
#   - 使用 DeepSpeed ZeRO Stage 2 多卡并行
#   - 输出仅包含 PINN 插件参数（非常小）
# =============================================================================

set -euo pipefail
# unset PINN_CHECKPOINT
# export PINN_CHECKPOINT="./models/train/pinn_plugin_dual_noise_shared/step-4000.pt"

# ------------------------
# Distributed launch safety
# ------------------------
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
# Set NCCL_DEBUG=INFO and TORCH_DISTRIBUTED_DEBUG=DETAIL only when debugging hangs.
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

if pgrep -af "accelerate launch --config_file examples/wanvideo/pinn_training/accelerate_config_pinn.yaml" >/dev/null; then
  echo "Detected an existing PINN accelerate job. Stop old processes first to avoid contention."
  exit 1
fi

HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-2}"
DIAGNOSTIC_METRICS_INTERVAL="${DIAGNOSTIC_METRICS_INTERVAL:-100}"
HEARTBEAT_LOG_STEPS="${HEARTBEAT_LOG_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
PHYSICS_WEIGHT="${PHYSICS_WEIGHT:-0.03}"
PHYSICS_WEIGHT_TARGET="${PHYSICS_WEIGHT_TARGET:-0.08}"
PHYSICS_WARMUP_STEPS="${PHYSICS_WARMUP_STEPS:-2000}"
MOE_TOP_K="${MOE_TOP_K:-4}"
CONDITION_CONSISTENCY_WEIGHT="${CONDITION_CONSISTENCY_WEIGHT:-0.0}"
STATE_ALIGN_WARMUP_STEPS="${STATE_ALIGN_WARMUP_STEPS:-1000}"
STATE_ALIGN_X_WEIGHT="${STATE_ALIGN_X_WEIGHT:-0.0}"
STATE_ALIGN_V_WEIGHT="${STATE_ALIGN_V_WEIGHT:-0.05}"
STATE_ALIGN_V_WEIGHT_TARGET="${STATE_ALIGN_V_WEIGHT_TARGET:-0.015}"
DUAL_NOISE_EXPERT_BOUNDARY="${DUAL_NOISE_EXPERT_BOUNDARY:-0.417}"
CURRICULUM_TRANSITION_START_STEP="${CURRICULUM_TRANSITION_START_STEP:-1000}"
CURRICULUM_TRANSITION_STEPS="${CURRICULUM_TRANSITION_STEPS:-1000}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
DUAL_NOISE_OUTPUT_PATH="${DUAL_NOISE_OUTPUT_PATH:-./models/train/pinn_plugin_dual_noise_shared_moe4}"

ACCELERATE_LAUNCH_ARGS=(
  --config_file examples/wanvideo/pinn_training/accelerate_config_pinn.yaml
  --num_processes "${NUM_PROCESSES}"
  --num_machines 1
  --machine_rank 0
  --main_process_ip "${MAIN_PROCESS_IP}"
  --main_process_port "${MAIN_PROCESS_PORT}"
)



# 可选：从 checkpoint 恢复训练（设置路径或留空）
PINN_CHECKPOINT="${PINN_CHECKPOINT:-}"  # 例如: ./models/train/pinn_plugin_dual_noise_shared/step-200.pt

# ========================
# 训练 shared adapter across dual noise experts
# boundary is defined in scheduler-index space to stay aligned with the
# previous separate high/low-noise training ranges.
# Important: the load order below is semantic:
#   1) high_noise_model -> dit
#   2) low_noise_model  -> dit2
# ========================

# 构建 checkpoint 参数（如果设置了）
PINN_CHECKPOINT_ARG=""
if [[ -n "${PINN_CHECKPOINT}" ]]; then
  PINN_CHECKPOINT_ARG="--pinn_checkpoint ${PINN_CHECKPOINT}"
fi

# accelerate launch \
#   "${ACCELERATE_LAUNCH_ARGS[@]}" \
#   examples/wanvideo/pinn_training/train_pinn.py \
#   --dataset_base_path /home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data \
#   --dataset_metadata_path /home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data/metadata_standard.csv \
#   --height "${HEIGHT}" \
#   --width "${WIDTH}" \
#   --num_frames "${NUM_FRAMES}" \
#   --dataset_repeat 1 \
#   --model_id_with_origin_paths "Wan-AI/Wan2.2-T2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
#   ${PINN_CHECKPOINT_ARG} \
#   --learning_rate 1e-5 \
#   --num_epochs 3 \
#   --output_path "./models/train/pinn_plugin_low_noise" \
#   --max_timestep_boundary 1 \
#   --min_timestep_boundary 0.417 \
#   --physics_weight 0.05 \
#   --physics_warmup_steps 2000 \
#   --save_steps 200 \
#   --adapter_hidden_dim 64 \
#   --physics_state_mode x0_hat \
#   --use_sigma_gate \
#   --sigma_gate_curve quadratic \
#   --use_sigma_conditioning \
#   --sigma_gate_floor 0.05 \
#   --moe_top_k "${MOE_TOP_K}" \
#   --dataset_num_workers "${DATASET_NUM_WORKERS}" \
#   --diagnostic_metrics_interval "${DIAGNOSTIC_METRICS_INTERVAL}" \
#   --heartbeat_log_steps "${HEARTBEAT_LOG_STEPS}" \
#   --tensorboard_log_steps 1 \
#   --find_unused_parameters \
#   ${EXTRA_FLAGS}


accelerate launch \
  "${ACCELERATE_LAUNCH_ARGS[@]}" \
  examples/wanvideo/pinn_training/train_pinn.py \
  --dataset_base_path /home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data \
  --dataset_metadata_path /home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data/metadata_standard.csv \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_frames "${NUM_FRAMES}" \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-T2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs 1 \
  --output_path "${DUAL_NOISE_OUTPUT_PATH}" \
  --max_timestep_boundary 1.0 \
  --min_timestep_boundary 0 \
  --use_dual_noise_experts \
  --dual_noise_expert_boundary "${DUAL_NOISE_EXPERT_BOUNDARY}" \
  --physics_weight "${PHYSICS_WEIGHT}" \
  --physics_weight_target "${PHYSICS_WEIGHT_TARGET}" \
  --physics_warmup_steps "${PHYSICS_WARMUP_STEPS}" \
  --condition_consistency_weight "${CONDITION_CONSISTENCY_WEIGHT}" \
  --state_align_warmup_steps "${STATE_ALIGN_WARMUP_STEPS}" \
  --state_align_x_weight "${STATE_ALIGN_X_WEIGHT}" \
  --state_align_v_weight "${STATE_ALIGN_V_WEIGHT}" \
  --state_align_v_weight_target "${STATE_ALIGN_V_WEIGHT_TARGET}" \
  --curriculum_transition_start_step "${CURRICULUM_TRANSITION_START_STEP}" \
  --curriculum_transition_steps "${CURRICULUM_TRANSITION_STEPS}" \
  --save_steps 200 \
  --adapter_hidden_dim 64 \
  --physics_state_mode x0_hat \
  --use_sigma_gate \
  --sigma_gate_curve quadratic \
  --use_sigma_conditioning \
  --sigma_gate_floor 0.05 \
  --moe_top_k "${MOE_TOP_K}" \
  --dataset_num_workers "${DATASET_NUM_WORKERS}" \
  --diagnostic_metrics_interval "${DIAGNOSTIC_METRICS_INTERVAL}" \
  --heartbeat_log_steps "${HEARTBEAT_LOG_STEPS}" \
  --tensorboard_log_steps 1 \
  --find_unused_parameters \
  ${EXTRA_FLAGS}


# # ========================
# # 训练 high noise 区间
# # boundary corresponds to timesteps [875, 1000]
# # ========================
# accelerate launch \
#   "${ACCELERATE_LAUNCH_ARGS[@]}" \
#   examples/wanvideo/pinn_training/train_pinn.py \
#   --dataset_base_path /home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data \
#   --dataset_metadata_path /home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data/metadata_new.csv \
#   --height "${HEIGHT}" \
#   --width "${WIDTH}" \
#   --num_frames "${NUM_FRAMES}" \
#   --dataset_repeat 1 \
#   --model_id_with_origin_paths "Wan-AI/Wan2.2-T2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
#   --learning_rate 1e-5 \
#   --num_epochs 2 \
#   --output_path "./models/train/pinn_plugin_high_noise" \
#   --max_timestep_boundary 0.417 \
#   --min_timestep_boundary 0 \
#   --physics_weight 0.1 \
#   --physics_warmup_steps 500 \
#   --adapter_hidden_dim 64 \
#   --physics_state_mode x0_hat \
#   --use_sigma_gate \
#   --sigma_gate_curve quadratic \
#   --use_sigma_conditioning \
#   --sigma_gate_floor 0.05 \
#   --moe_top_k "${MOE_TOP_K}" \
#   --dataset_num_workers "${DATASET_NUM_WORKERS}" \
#   --diagnostic_metrics_interval "${DIAGNOSTIC_METRICS_INTERVAL}" \
#   --heartbeat_log_steps "${HEARTBEAT_LOG_STEPS}" \
#   --tensorboard_log_steps 1 \
#   --find_unused_parameters \
#   ${EXTRA_FLAGS}

# ========================
# Ablation 示例（按需单独执行）
# ========================
# Baseline:
# accelerate launch --config_file examples/wanvideo/pinn_training/accelerate_config_pinn.yaml \
#   examples/wanvideo/pinn_training/train_pinn.py \
#   --dataset_base_path /home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data \
#   --dataset_metadata_path /home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data/metadata_new.csv \
#   --height 480 --width 832 --num_frames 49 --dataset_repeat 1 \
#   --model_id_with_origin_paths "Wan-AI/Wan2.2-T2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
#   --learning_rate 1e-5 --num_epochs 2 \
#   --output_path "./models/train/pinn_plugin_ablation_baseline" \
#   --max_timestep_boundary 0.417 --min_timestep_boundary 0 \
#   --physics_weight 0.1 --physics_warmup_steps 500 \
#   --adapter_hidden_dim 64 --find_unused_parameters
#
# No-MoE:
# ... (same args as baseline) ... \
#   --output_path "./models/train/pinn_plugin_ablation_no_moe" \
#   --ablate_disable_moe
#
# No-conditioned-PDE:
# ... (same args as baseline) ... \
#   --output_path "./models/train/pinn_plugin_ablation_no_conditioned_pde" \
#   --ablate_disable_conditioned_pde
#
# Label-only-router:
# ... (same args as baseline) ... \
#   --output_path "./models/train/pinn_plugin_ablation_label_only" \
#   --ablate_label_only_router
