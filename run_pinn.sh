#!/bin/bash
set -euo pipefail

# # Basic inference (no explicit metadata; adapter falls back automatically)
# python examples/wanvideo/pinn_inference/inference_pinn.py \
#     --prompt "Two oranges fall into the lake." \
#     --checkpoint_path models/train/pinn_plugin_low_noise/pinn_plugin_final.pt \
#     --output video_pinn_basic.mp4

# Raw n/q metadata inference (recommended for MoE routing control)
# You can pass raw fields directly; script will auto-encode to adapter metadata.

python examples/wanvideo/pinn_inference/inference_pinn.py \
    --prompt "A volleyball falls into the lake." \
    --checkpoint_path models/train/pinn_plugin_low_noise/pinn_plugin_final.pt \
    --metadata_json '{
      "label":"liquid motion",
      "n0":"speed around 1.2 to 2.0",
      "n1":"density 0.5~0.7",
      "n2":"viscosity 0.01",
      "q0":"splash, ripple",
      "q1":"surface tension",
      "q2":"wake trail",
      "q3":"no",
      "q4":"reflection and refraction"
    }' \
    --output video_pinn_with_raw_metadata6.mp4

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