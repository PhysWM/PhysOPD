#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/models/train/wan21_stage2_fullpinn8/step-18500.pt}"
PYTHON_BIN="${PYTHON_BIN:-/home/dataset-assist-0/algorithm/cong.wang/miniconda3/envs/wan/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/moe_visual_four_examples}"
RUN_BASELINE="${RUN_BASELINE:-1}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
SEED="${SEED:-0}"
CFG_SCALE="${CFG_SCALE:-5.0}"
FPS="${FPS:-15}"
QUALITY="${QUALITY:-5}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTPUT_DIR}"

KEYS=(splash combustion bounce viscous)
LABELS=("Fluid splash" "Combustion / heat" "Elastic collision" "Viscous liquid")
PROMPTS=(
  "An apple falls into a vat of cider, sending up a spray."
  "A small piece of copper is ignited, producing a vivid green-blue flame."
  "A vibrant elastic rubber ball is thrown forcefully toward the ground and bounces on impact."
  "A stream of honey pours slowly into a cup of hot tea, forming thick swirling patterns."
)

records=()
for idx in "${!KEYS[@]}"; do
  key="${KEYS[$idx]}"
  label="${LABELS[$idx]}"
  prompt="${PROMPTS[$idx]}"
  baseline_video="${OUTPUT_DIR}/${key}_baseline.mp4"
  pinn_video="${OUTPUT_DIR}/${key}_pinn.mp4"
  trace_path="${OUTPUT_DIR}/${key}_pinn_physics_report_physics_trace.npz"

  common_args=(
    --prompt "${prompt}"
    --model_id Wan-AI/Wan2.1-T2V-1.3B
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --num_frames "${NUM_FRAMES}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --seed "${SEED}"
    --cfg_scale "${CFG_SCALE}"
    --fps "${FPS}"
    --quality "${QUALITY}"
    --device "${DEVICE}"
  )

  if [[ "${RUN_BASELINE}" != "0" || ! -f "${baseline_video}" ]]; then
    "${PYTHON_BIN}" examples/wanvideo/pinn_inference/inference_pinn.py \
      --output "${baseline_video}" \
      "${common_args[@]}"
  fi
  if [[ ! -f "${baseline_video}" ]]; then
    echo "Baseline video missing: ${baseline_video}"
    exit 1
  fi

  "${PYTHON_BIN}" examples/wanvideo/pinn_inference/inference_pinn.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --output "${pinn_video}" \
    "${common_args[@]}" \
    --export_expert_attention \
    --expert_attention_unweighted \
    --expert_attention_topk 4 \
    --expert_attention_num_frames 1 \
    --auto_label_from_prompt

  if [[ ! -f "${trace_path}" ]]; then
    echo "Physics trace missing: ${trace_path}"
    exit 1
  fi
  records+=("${label}"$'\t'"${prompt}"$'\t'"${baseline_video}"$'\t'"${pinn_video}"$'\t'"${trace_path}")
done

manifest_path="${OUTPUT_DIR}/four_examples_manifest.json"
"${PYTHON_BIN}" - "${manifest_path}" "${records[@]}" <<'PY'
import json
import sys

manifest = []
for record in sys.argv[2:]:
    label, prompt, baseline_video, video, trace = record.split("\t")
    manifest.append({
        "label": label,
        "prompt": prompt,
        "baseline_video": baseline_video,
        "video": video,
        "trace": trace,
    })
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
PY

"${PYTHON_BIN}" examples/wanvideo/pinn_inference/build_pinn_evidence_report.py \
  --examples_json "${manifest_path}" \
  --output_prefix "${OUTPUT_DIR}/wan21_moe_visual" \
  --overlay_alpha 0.48
