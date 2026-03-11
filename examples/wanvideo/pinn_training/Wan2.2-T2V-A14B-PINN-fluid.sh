#!/bin/bash
# =============================================================================
# PINN Training - Fluid Material Only
# 仅训练流体材质的物理约束
#
# 使用方法:
#   bash examples/wanvideo/pinn_training/Wan2.2-T2V-A14B-PINN-fluid.sh
# =============================================================================

accelerate launch \
  --config_file examples/wanvideo/pinn_training/accelerate_config_pinn.yaml \
  examples/wanvideo/pinn_training/train_pinn.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-T2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --output_path "./models/train/pinn_plugin_fluid" \
  --max_timestep_boundary 0.417 \
  --min_timestep_boundary 0 \
  --physics_weight 0.1 \
  --physics_warmup_steps 500 \
  --material_type fluid \
  --adapter_hidden_dim 64 \
  --find_unused_parameters
