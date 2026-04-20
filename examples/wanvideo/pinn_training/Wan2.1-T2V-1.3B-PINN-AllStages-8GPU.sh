#!/bin/bash
# =============================================================================
# Wan2.1-T2V-1.3B PINN all-stage 8-GPU training runner
#
# 默认串行执行:
#   1) Stage 1 / low-noise observable pretrain   : sigma in [0.90, 1.00]
#   2) Stage 1 / mid-noise observable pretrain   : sigma in [0.75, 1.00]
#   3) Stage 1 / wide-noise observable pretrain  : sigma in [0.50, 1.00]
#   4) Stage 2 / full PINN training              : sigma in [0.00, 1.00]
#
# 每一阶段都会自动读取上一阶段输出目录中的 pinn_plugin_final.pt。
#
# 用法:
#   bash examples/wanvideo/pinn_training/Wan2.1-T2V-1.3B-PINN-AllStages-8GPU.sh
#
# 常用覆盖:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   FLOW_BACKBONE_CKPT=/path/to/flow_teacher.pt \
#   BASE_OUTPUT_ROOT=./models/train/wan21_pinn_allstages_8gpu \
#   bash examples/wanvideo/pinn_training/Wan2.1-T2V-1.3B-PINN-AllStages-8GPU.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/Wan2.1-T2V-1.3B-PINN-2Stage.sh"

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-./models/train/wan21_pinn_allstages_8gpu}"
FLOW_BACKBONE_CKPT="${FLOW_BACKBONE_CKPT:-}"
MAIN_PROCESS_PORT_BASE="${MAIN_PROCESS_PORT_BASE:-29511}"

STAGE1_NUM_EPOCHS="${STAGE1_NUM_EPOCHS:-2}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-3000}"
STAGE1_SAVE_STEPS="${STAGE1_SAVE_STEPS:-500}"
STAGE1_LEARNING_RATE="${STAGE1_LEARNING_RATE:-1e-5}"

STAGE2_NUM_EPOCHS="${STAGE2_NUM_EPOCHS:-3}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-}"
STAGE2_SAVE_STEPS="${STAGE2_SAVE_STEPS:-1000}"
STAGE2_LEARNING_RATE="${STAGE2_LEARNING_RATE:-1e-5}"
STAGE2_ENCODER_FREEZE_STEPS="${STAGE2_ENCODER_FREEZE_STEPS:-1000}"
STAGE2_ENCODER_LR_SCALE="${STAGE2_ENCODER_LR_SCALE:-0.3}"
ABLATION_PRESETS="${ABLATION_PRESETS:-legacy_direct_bank,u_only_direct_prho,u_only_ufirst_prho,u_only_ufirst_prho_detach}"

DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-2}"
DIAGNOSTIC_METRICS_INTERVAL="${DIAGNOSTIC_METRICS_INTERVAL:-100}"
HEARTBEAT_LOG_STEPS="${HEARTBEAT_LOG_STEPS:-1}"

STAGE1_LOW_NAME="stage1_phase1_low_noise"
STAGE1_MID_NAME="stage1_phase2_mid_noise"
STAGE1_WIDE_NAME="stage1_phase3_wide_noise"
STAGE2_NAME="stage2_full_pinn"

mkdir -p "${BASE_OUTPUT_ROOT}"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Base script not found: ${BASE_SCRIPT}"
  exit 1
fi

if [[ -z "${FLOW_BACKBONE_CKPT}" ]]; then
  echo "[warn] FLOW_BACKBONE_CKPT is empty. If your Stage 1 setup requires a flow teacher checkpoint, please set it before running."
fi

LAST_STAGE1_CKPT=""
LAST_STAGE2_CKPT=""

preset_observable_target_mode() {
  local preset="$1"
  case "${preset}" in
    legacy_direct_bank) echo "flow_plus_deformation" ;;
    *) echo "flow_only" ;;
  esac
}

preset_secondary_field_strategy() {
  local preset="$1"
  case "${preset}" in
    legacy_direct_bank) echo "legacy_direct_bank" ;;
    u_only_direct_prho) echo "direct_bank" ;;
    u_only_ufirst_prho) echo "u_first_constructor" ;;
    u_only_ufirst_prho_detach) echo "u_first_constructor_detach" ;;
    *)
      echo "Unsupported preset: ${preset}" >&2
      exit 1
      ;;
  esac
}

preset_active_field_set() {
  local preset="$1"
  case "${preset}" in
    legacy_direct_bank) echo "legacy" ;;
    *) echo "u,p,rho" ;;
  esac
}

preset_field_enable_schedule() {
  local preset="$1"
  case "${preset}" in
    legacy_direct_bank) echo "legacy" ;;
    *) echo "fixed_only_u_recovery" ;;
  esac
}

configure_preset_env() {
  local preset="$1"
  export ABLATION_PRESET="${preset}"
  export OBSERVABLE_TARGET_MODE="$(preset_observable_target_mode "${preset}")"
  export SECONDARY_FIELD_STRATEGY="$(preset_secondary_field_strategy "${preset}")"
  export ACTIVE_FIELD_SET="$(preset_active_field_set "${preset}")"
  export FIELD_ENABLE_SCHEDULE="$(preset_field_enable_schedule "${preset}")"
}

run_stage1_phase() {
  local phase_name="$1"
  local min_timestep="$2"
  local max_timestep="$3"
  local resume_ckpt="$4"
  local port="$5"

  local output_path="${BASE_OUTPUT_ROOT}/${phase_name}"
  local final_ckpt="${output_path}/pinn_plugin_final.pt"
  local cache_root="${output_path}/runtime_cache"

  if [[ -f "${final_ckpt}" ]]; then
    echo "[skip] ${phase_name} already finished: ${final_ckpt}"
    LAST_STAGE1_CKPT="${final_ckpt}"
    return 0
  fi

  mkdir -p "${output_path}" "${cache_root}/modelscope" "${cache_root}/huggingface"

  echo "============================================================"
  echo "Running ${phase_name}"
  echo "  stage                : observable_pretrain"
  echo "  ablation preset      : ${ABLATION_PRESET}"
  echo "  output               : ${output_path}"
  echo "  resume checkpoint    : ${resume_ckpt:-<none>}"
  echo "  timestep range       : [${min_timestep}, ${max_timestep}]"
  echo "  gpus                 : ${CUDA_VISIBLE_DEVICES}"
  echo "  main process port    : ${port}"
  echo "============================================================"

  export TRAINING_STAGE="observable_pretrain"
  export OUTPUT_PATH="${output_path}"
  export PINN_CHECKPOINT="${resume_ckpt}"
  export STAGE1_PRETRAINED_ENCODER=""
  export FLOW_BACKBONE_CKPT
  export NUM_EPOCHS="${STAGE1_NUM_EPOCHS}"
  export MAX_STEPS="${STAGE1_MAX_STEPS}"
  export SAVE_STEPS="${STAGE1_SAVE_STEPS}"
  export LEARNING_RATE="${STAGE1_LEARNING_RATE}"
  export MIN_TIMESTEP_BOUNDARY="${min_timestep}"
  export MAX_TIMESTEP_BOUNDARY="${max_timestep}"
  export DATASET_NUM_WORKERS
  export DIAGNOSTIC_METRICS_INTERVAL
  export HEARTBEAT_LOG_STEPS
  export MAIN_PROCESS_PORT="${port}"
  export CACHE_ROOT="${cache_root}"
  export MODELSCOPE_CACHE="${cache_root}/modelscope"
  export HF_HOME="${cache_root}/huggingface"

  export DEBUG_OVERFIT_NUM_SAMPLES=""
  export DEBUG_OVERFIT_DATASET_REPEAT=""
  export DEBUG_FIXED_TIMESTEP_FRACTION=""
  export EXTRA_FLAGS=""

  bash "${BASE_SCRIPT}"

  if [[ ! -f "${final_ckpt}" ]]; then
    echo "Expected final checkpoint was not produced: ${final_ckpt}"
    exit 1
  fi

  LAST_STAGE1_CKPT="${final_ckpt}"
}

run_stage2_full_pinn() {
  local stage1_ckpt="$1"
  local port="$2"
  local stage2_name="${3:-${STAGE2_NAME}}"

  local output_path="${BASE_OUTPUT_ROOT}/${stage2_name}"
  local final_ckpt="${output_path}/pinn_plugin_final.pt"
  local cache_root="${output_path}/runtime_cache"

  if [[ ! -f "${stage1_ckpt}" ]]; then
    echo "Stage 2 requires a valid Stage 1 checkpoint, but got: ${stage1_ckpt}"
    exit 1
  fi

  if [[ -f "${final_ckpt}" ]]; then
    echo "[skip] ${stage2_name} already finished: ${final_ckpt}"
    LAST_STAGE2_CKPT="${final_ckpt}"
    return 0
  fi

  mkdir -p "${output_path}" "${cache_root}/modelscope" "${cache_root}/huggingface"

  echo "============================================================"
  echo "Running ${stage2_name}"
  echo "  stage                : full_pinn"
  echo "  ablation preset      : ${ABLATION_PRESET}"
  echo "  output               : ${output_path}"
  echo "  stage1 init ckpt     : ${stage1_ckpt}"
  echo "  timestep range       : [0.0, 1.0]"
  echo "  gpus                 : ${CUDA_VISIBLE_DEVICES}"
  echo "  main process port    : ${port}"
  echo "============================================================"

  export TRAINING_STAGE="full_pinn"
  export OUTPUT_PATH="${output_path}"
  export PINN_CHECKPOINT=""
  export STAGE1_PRETRAINED_ENCODER="${stage1_ckpt}"
  export FLOW_BACKBONE_CKPT
  export NUM_EPOCHS="${STAGE2_NUM_EPOCHS}"
  export MAX_STEPS="${STAGE2_MAX_STEPS}"
  export SAVE_STEPS="${STAGE2_SAVE_STEPS}"
  export LEARNING_RATE="${STAGE2_LEARNING_RATE}"
  export MIN_TIMESTEP_BOUNDARY="0.0"
  export MAX_TIMESTEP_BOUNDARY="1.0"
  export ENCODER_FREEZE_STEPS="${STAGE2_ENCODER_FREEZE_STEPS}"
  export ENCODER_LR_SCALE="${STAGE2_ENCODER_LR_SCALE}"
  export DATASET_NUM_WORKERS
  export DIAGNOSTIC_METRICS_INTERVAL
  export HEARTBEAT_LOG_STEPS
  export MAIN_PROCESS_PORT="${port}"
  export CACHE_ROOT="${cache_root}"
  export MODELSCOPE_CACHE="${cache_root}/modelscope"
  export HF_HOME="${cache_root}/huggingface"

  export DEBUG_OVERFIT_NUM_SAMPLES=""
  export DEBUG_OVERFIT_DATASET_REPEAT=""
  export DEBUG_FIXED_TIMESTEP_FRACTION=""
  export EXTRA_FLAGS=""

  bash "${BASE_SCRIPT}"

  if [[ ! -f "${final_ckpt}" ]]; then
    echo "Expected final checkpoint was not produced: ${final_ckpt}"
    exit 1
  fi
  LAST_STAGE2_CKPT="${final_ckpt}"
}

echo "============================================================"
echo "Wan2.1 all-stage 8-GPU runner"
echo "  base output root      : ${BASE_OUTPUT_ROOT}"
echo "  CUDA_VISIBLE_DEVICES  : ${CUDA_VISIBLE_DEVICES}"
echo "  flow backbone ckpt    : ${FLOW_BACKBONE_CKPT:-<none>}"
echo "  ablation presets      : ${ABLATION_PRESETS}"
echo "  stage1 steps          : ${STAGE1_MAX_STEPS}"
echo "  stage2 steps          : ${STAGE2_MAX_STEPS:-<epoch-controlled>}"
echo "============================================================"

IFS=',' read -r -a PRESET_LIST <<< "${ABLATION_PRESETS}"
LEGACY_ENABLED=0
U_ONLY_SHARED_PRESET=""
for preset in "${PRESET_LIST[@]}"; do
  preset="$(echo "${preset}" | xargs)"
  if [[ -z "${preset}" ]]; then
    continue
  fi
  if [[ "${preset}" == "legacy_direct_bank" ]]; then
    LEGACY_ENABLED=1
  elif [[ -z "${U_ONLY_SHARED_PRESET}" ]]; then
    U_ONLY_SHARED_PRESET="${preset}"
  fi
done

PORT_OFFSET=0

if [[ "${LEGACY_ENABLED}" == "1" ]]; then
  configure_preset_env "legacy_direct_bank"
  run_stage1_phase "${STAGE1_LOW_NAME}_legacy_direct_bank" "0.90" "1.00" "" "$((MAIN_PROCESS_PORT_BASE + PORT_OFFSET))"
  PORT_OFFSET=$((PORT_OFFSET + 1))
  run_stage1_phase "${STAGE1_MID_NAME}_legacy_direct_bank" "0.75" "1.00" "${LAST_STAGE1_CKPT}" "$((MAIN_PROCESS_PORT_BASE + PORT_OFFSET))"
  PORT_OFFSET=$((PORT_OFFSET + 1))
  run_stage1_phase "${STAGE1_WIDE_NAME}_legacy_direct_bank" "0.50" "1.00" "${LAST_STAGE1_CKPT}" "$((MAIN_PROCESS_PORT_BASE + PORT_OFFSET))"
  PORT_OFFSET=$((PORT_OFFSET + 1))
  run_stage2_full_pinn "${LAST_STAGE1_CKPT}" "$((MAIN_PROCESS_PORT_BASE + PORT_OFFSET))" "${STAGE2_NAME}_legacy_direct_bank"
  PORT_OFFSET=$((PORT_OFFSET + 1))
fi

if [[ -n "${U_ONLY_SHARED_PRESET}" ]]; then
  configure_preset_env "${U_ONLY_SHARED_PRESET}"
  run_stage1_phase "stage1_onlyu_shared_phase1_low_noise" "0.90" "1.00" "" "$((MAIN_PROCESS_PORT_BASE + PORT_OFFSET))"
  PORT_OFFSET=$((PORT_OFFSET + 1))
  run_stage1_phase "stage1_onlyu_shared_phase2_mid_noise" "0.75" "1.00" "${LAST_STAGE1_CKPT}" "$((MAIN_PROCESS_PORT_BASE + PORT_OFFSET))"
  PORT_OFFSET=$((PORT_OFFSET + 1))
  run_stage1_phase "stage1_onlyu_shared_phase3_wide_noise" "0.50" "1.00" "${LAST_STAGE1_CKPT}" "$((MAIN_PROCESS_PORT_BASE + PORT_OFFSET))"
  PORT_OFFSET=$((PORT_OFFSET + 1))

  U_ONLY_STAGE1_CKPT="${LAST_STAGE1_CKPT}"
  for preset in "${PRESET_LIST[@]}"; do
    preset="$(echo "${preset}" | xargs)"
    if [[ -z "${preset}" || "${preset}" == "legacy_direct_bank" ]]; then
      continue
    fi
    configure_preset_env "${preset}"
    run_stage2_full_pinn "${U_ONLY_STAGE1_CKPT}" "$((MAIN_PROCESS_PORT_BASE + PORT_OFFSET))" "${STAGE2_NAME}_${preset}"
    PORT_OFFSET=$((PORT_OFFSET + 1))
  done
fi

echo "============================================================"
echo "All stages completed."
echo "  final Stage 1 ckpt    : ${LAST_STAGE1_CKPT}"
echo "  final Stage 2 ckpt    : ${LAST_STAGE2_CKPT:-<none>}"
echo "============================================================"
