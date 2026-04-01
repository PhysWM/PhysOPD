#!/bin/bash
#
# 八卡并行批量生成视频
# 将 videophy_test_public.csv 中的 3361 个 caption 按 ID 范围分配到 8 张 GPU
#
# 用法:
#   bash run_batch_8gpu.sh                    # 使用默认参数
#   bash run_batch_8gpu.sh --output_dir xxx   # 自定义输出目录
#
# 环境变量:
#   CSV=xxx.csv          CSV 路径
#   OUTPUT_DIR=xxx       输出目录（默认 output_videos）
#   CHECKPOINT=xxx.pt    PINN checkpoint
#   OPENAI_API_KEY=...   OpenAI-compatible API key
#   OPENAI_BASE_URL=...  OpenAI-compatible base URL
#   OPENAI_MODEL=...     LLM model used for routing labels
#   LOG_DIR=logs         若设置，每卡输出到 logs/gpu0.log 等
#
# 单独跑某张卡（例如卡 0，ID 1-421）:
#   CUDA_VISIBLE_DEVICES=0 python examples/wanvideo/pinn_inference/batch_inference_pinn.py \
#       --start_id 1 --end_id 421 --csv videophy_test_public.csv --output_dir output_videos
#

set -e
cd "$(dirname "$0")"

if [[ -f ".pinn_api.env" ]]; then
    set -a
    source ".pinn_api.env"
    set +a
fi

CSV="${CSV:-videophy_test_public.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-output_videos_pinn_corrected}"
CHECKPOINT="${CHECKPOINT:-models/train/pinn_plugin_low_noise/pinn_plugin_final.pt}"
LOG_DIR="${LOG_DIR:-}"
TOTAL=344
AUTO_LABEL_FROM_PROMPT="${AUTO_LABEL_FROM_PROMPT:-1}"

# 每卡约 420 个，均匀分配
CHUNK=$(( (TOTAL + 7) / 8 ))

run_one() {
    local gpu=$1 start=$2 end=$3
    shift 3
    echo "[GPU $gpu] Starting: IDs $start - $end"
    local log_redirect=""
    local auto_label_flag=""
    [[ -n "$LOG_DIR" ]] && { mkdir -p "$LOG_DIR"; log_redirect="2>&1 | tee $LOG_DIR/gpu${gpu}.log"; }
    if [[ "$AUTO_LABEL_FROM_PROMPT" == "1" ]]; then
        auto_label_flag="--auto_label_from_prompt"
    fi
    eval "CUDA_VISIBLE_DEVICES=$gpu python examples/wanvideo/pinn_inference/batch_inference_pinn.py \
        --csv \"$CSV\" \
        --start_id $start \
        --end_id $end \
        --output_dir \"$OUTPUT_DIR\" \
        --checkpoint_path \"$CHECKPOINT\" \
        --skip_existing \
        $auto_label_flag \
        \"\$@\" $log_redirect"
}

# 后台启动 8 个进程
run_one 0 1 $(( 1 * CHUNK )) "$@" &
run_one 1 $(( 1 * CHUNK + 1 )) $(( 2 * CHUNK )) "$@" &
run_one 2 $(( 2 * CHUNK + 1 )) $(( 3 * CHUNK )) "$@" &
run_one 3 $(( 3 * CHUNK + 1 )) $(( 4 * CHUNK )) "$@" &
run_one 4 $(( 4 * CHUNK + 1 )) $(( 5 * CHUNK )) "$@" &
run_one 5 $(( 5 * CHUNK + 1 )) $(( 6 * CHUNK )) "$@" &
run_one 6 $(( 6 * CHUNK + 1 )) $(( 7 * CHUNK )) "$@" &
run_one 7 $(( 7 * CHUNK + 1 )) $TOTAL "$@" &

wait
echo "All 8 GPU jobs finished."
