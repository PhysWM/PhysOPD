#!/bin/bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRAINING_STAGE="${TRAINING_STAGE:-full_pinn}"
export PINN_CHECKPOINT="${PINN_CHECKPOINT:-/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/models/train/wan21_stage2_fullpinn7/step-3000.pt}"
export OUTPUT_PATH="${OUTPUT_PATH:-./models/train/wan21_stage2_fullpinn8}"

# Conservative resumed full_pinn defaults: lower LR and hold the shared encoder longer.
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export ENCODER_FREEZE_STEPS="${ENCODER_FREEZE_STEPS:-3000}"
export ENCODER_LR_SCALE="${ENCODER_LR_SCALE:-0.1}"

bash examples/wanvideo/pinn_training/Wan2.1-T2V-1.3B-PINN-2Stage.sh
