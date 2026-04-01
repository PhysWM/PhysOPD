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
export PINN_CHECKPOINT="./models/train/pinn_plugin_low_noise/step-15000.pt"

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
NUM_FRAMES="${NUM_FRAMES:-49}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-2}"
DIAGNOSTIC_METRICS_INTERVAL="${DIAGNOSTIC_METRICS_INTERVAL:-100}"
HEARTBEAT_LOG_STEPS="${HEARTBEAT_LOG_STEPS:-1}"
MOE_TOP_K="${MOE_TOP_K:-1}"
MOE_FAST_MODE="${MOE_FAST_MODE:-1}"
MOE_PDE_BRANCHES_PER_SAMPLE="${MOE_PDE_BRANCHES_PER_SAMPLE:-1}"
MOE_WEIGHT_THRESHOLD="${MOE_WEIGHT_THRESHOLD:-0.05}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

if [[ "${MOE_FAST_MODE}" == "0" ]]; then
  MOE_FAST_MODE_FLAG="--no_moe_fast_mode"
else
  MOE_FAST_MODE_FLAG="--moe_fast_mode"
fi

ACCELERATE_LAUNCH_ARGS=(
  --config_file examples/wanvideo/pinn_training/accelerate_config_pinn.yaml
  --num_processes "${NUM_PROCESSES}"
  --num_machines 1
  --machine_rank 0
  --main_process_ip "${MAIN_PROCESS_IP}"
  --main_process_port "${MAIN_PROCESS_PORT}"
)



# 可选：从 checkpoint 恢复训练（设置路径或留空）
PINN_CHECKPOINT="${PINN_CHECKPOINT:-}"  # 例如: ./models/train/pinn_plugin_low_noise/step-200.pt

# ========================
# 训练 low noise 区间
# boundary corresponds to timesteps [0, 875)
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
#   --moe_top_k "${MOE_TOP_K}" \
#   ${MOE_FAST_MODE_FLAG} \
#   --moe_pde_branches_per_sample "${MOE_PDE_BRANCHES_PER_SAMPLE}" \
#   --moe_weight_threshold "${MOE_WEIGHT_THRESHOLD}" \
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
  --model_id_with_origin_paths "Wan-AI/Wan2.2-T2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 3 \
  --output_path "./models/train/pinn_plugin_low_noise" \
  --max_timestep_boundary 1 \
  --min_timestep_boundary 0.417 \
  --physics_weight 0.05 \
  --physics_warmup_steps 2000 \
  --save_steps 200 \
  --adapter_hidden_dim 64 \
  --moe_top_k "${MOE_TOP_K}" \
  ${MOE_FAST_MODE_FLAG} \
  --moe_pde_branches_per_sample "${MOE_PDE_BRANCHES_PER_SAMPLE}" \
  --moe_weight_threshold "${MOE_WEIGHT_THRESHOLD}" \
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
#   --moe_top_k "${MOE_TOP_K}" \
#   ${MOE_FAST_MODE_FLAG} \
#   --moe_pde_branches_per_sample "${MOE_PDE_BRANCHES_PER_SAMPLE}" \
#   --moe_weight_threshold "${MOE_WEIGHT_THRESHOLD}" \
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


