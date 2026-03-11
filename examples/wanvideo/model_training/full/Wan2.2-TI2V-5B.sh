#!/bin/bash
source /home/dataset-assist-0/algorithm/cong.wang/miniconda3/bin/activate
conda activate wan
export HF_HOME="/home/dataset-assist-0/algorithm/cong.wang/cache"
export TORCH_HOME="/home/dataset-assist-0/algorithm/cong.wang/cache/torch"
export XDG_CACHE_HOME="/home/dataset-assist-0/algorithm/cong.wang/cache/xdg_cache"
export HF_ENDPOINT=https://hf-mirror.com
accelerate launch /home/dataset-assist-0/algorithm/cong.wang/projects/DiffSynth-Studio/examples/wanvideo/model_training/train.py \
  --dataset_base_path /home/dataset-assist-0/algorithm/cong.wang/projects/DiffSynth-Studio/data/phys \
  --dataset_metadata_path /home/dataset-assist-0/algorithm/cong.wang/projects/DiffSynth-Studio/data/phys/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/home/dataset-assist-0/algorithm/cong.wang/projects/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_full" \
  --trainable_models "dit" \
  --extra_inputs "input_image"