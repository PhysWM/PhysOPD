#!/bin/bash
set -euo pipefail

if [[ -f ".pinn_api.env" ]]; then
    set -a
    source ".pinn_api.env"
    set +a
fi

# OpenAI-compatible API config example:
# export OPENAI_API_KEY=...
# export OPENAI_BASE_URL=https://api.openai.com/v1
# export OPENAI_MODEL=gpt-4.1-mini
export CUDA_VISIBLE_DEVICES=3

# Basic inference with automatic LLM routing labels
/home/dataset-assist-0/algorithm/cong.wang/miniconda3/envs/wan/bin/python examples/wanvideo/pinn_inference/inference_pinn.py \
    --prompt "The eraser rubs against the paper, removing pencil marks." \
    --checkpoint_path /home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/models/train/pinn_plugin_high_noise_moe4/step-3600.pt \
    --auto_label_from_prompt \
    --output video_pinn_eraser.mp4

# Raw n/q metadata inference (recommended if you want to override routing manually)
# You can pass raw fields directly; script will auto-encode to adapter metadata.
# python examples/wanvideo/pinn_inference/inference_pinn.py \
#     --prompt "I put an ice cube in my hot coffee, and it melted." \
#     --checkpoint_path models/train/pinn_plugin_low_noise/pinn_plugin_final.pt \
#     --metadata_json '{
#       "label":"Fluid, Thermal, Phase Change",
#       "n0":"speed around 1.2 to 2.0",
#       "n1":"density 0.5~0.7",
#       "n2":"viscosity 0.01",
#       "q0":"splash, ripple",
#       "q1":"surface tension",
#       "q2":"wake trail",
#       "q3":"no",
#       "q4":"reflection and refraction"
#     }' \
#     --output video_pinn_with_raw_metadata19.mp4

# python examples/wanvideo/pinn_inference/inference_pinn.py \
#   --prompt "Three tennis balls fall onto the ground." \
#   --checkpoint_path models/train/pinn_plugin_low_noise/pinn_plugin_final.pt \
#   --metadata_json '{
#     "label":"collision",
#     "n0":"impact speed around 3.0 to 6.0 m/s",
#     "n1":"ball mass 0.056 to 0.059 kg, ground density around 2400",
#     "n2":"restitution 0.70 to 0.85, friction 0.4 to 0.7",
#     "q0":"falling, impact, bounce",
#     "q1":"ground contact, collision impulse",
#     "q2":"rebound trajectory, rolling after impact",
#     "q3":"yes",
#     "q4":"hard surface, rigid body motion"
#   }' \
#   --output video_pinn_tennis_collision.mp4
