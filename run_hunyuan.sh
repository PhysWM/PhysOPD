#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_PATH="/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/phygenbench_prompts.csv"
OUTPUT_DIR="$SCRIPT_DIR/phygenbench_hunyuan"

export HF_HOME="/home/dataset-assist-0/algorithm/cong.wang/cache"
export TORCH_HOME="/home/dataset-assist-0/algorithm/cong.wang/cache/torch"
export XDG_CACHE_HOME="/home/dataset-assist-0/algorithm/cong.wang/cache/xdg_cache"
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=3

mkdir -p "$OUTPUT_DIR"

python - "$CSV_PATH" <<'PY' | while IFS=$'\t' read -r index caption; do
import csv
import sys

csv_path = sys.argv[1]

with open(csv_path, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader, start=1):
        caption = (row.get("caption") or "").strip()
        if caption:
            print(f"{idx}\t{caption}")
PY
    output_path="$(printf "%s/%04d.mp4" "$OUTPUT_DIR" "$index")"
    if [[ -s "$output_path" ]]; then
        echo "[$index] Skip existing: $output_path"
        continue
    fi

    echo "[$index] Generating: $output_path"
    HUNYUAN_PROMPT="$caption" \
    HUNYUAN_OUTPUT="$output_path" \
    python "$SCRIPT_DIR/examples/HunyuanVideo/hunyuanvideo_80G.py"
done
