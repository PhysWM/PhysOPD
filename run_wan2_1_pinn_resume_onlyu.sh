#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
BASE_SCRIPT="${REPO_ROOT}/examples/wanvideo/pinn_training/Wan2.1-T2V-1.3B-PINN-2Stage.sh"

cd "${REPO_ROOT}"

PHASE1_RESUME_CKPT="${PHASE1_RESUME_CKPT:-${REPO_ROOT}/models/train/wan21_pinn_allstages_8gpu/stage1_phase1_low_noise/step-2000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./models/train/wan21_pinn_allstages_8gpu_onlyu_resume2000}"

PHASE2_OUTPUT="${PHASE2_OUTPUT:-${OUTPUT_ROOT}/stage1_onlyu_shared_phase2_mid_noise}"
PHASE3_OUTPUT="${PHASE3_OUTPUT:-${OUTPUT_ROOT}/stage1_onlyu_shared_phase3_wide_noise}"
PROGRESSIVE_ENCODER_OUTPUT="${PROGRESSIVE_ENCODER_OUTPUT:-${OUTPUT_ROOT}/encoder_progressive}"
ENCODER_CORE_OUTPUT="${ENCODER_CORE_OUTPUT:-${OUTPUT_ROOT}/encoder_core}"
ENCODER_ALPHA_OUTPUT="${ENCODER_ALPHA_OUTPUT:-${OUTPUT_ROOT}/encoder_alpha}"
ENCODER_T_OUTPUT="${ENCODER_T_OUTPUT:-${OUTPUT_ROOT}/encoder_T}"
ENCODER_J_OUTPUT="${ENCODER_J_OUTPUT:-${OUTPUT_ROOT}/encoder_j}"
ENCODER_D_OUTPUT="${ENCODER_D_OUTPUT:-${OUTPUT_ROOT}/encoder_D}"
ENCODER_PSI_OUTPUT="${ENCODER_PSI_OUTPUT:-${OUTPUT_ROOT}/encoder_psi}"
FULL_PINN_OUTPUT="${FULL_PINN_OUTPUT:-${OUTPUT_ROOT}/full_pinn_from_encoder_psi}"

PHASE2_INPUT_CKPT="${PHASE2_INPUT_CKPT:-${PHASE1_RESUME_CKPT}}"
PHASE3_INPUT_CKPT="${PHASE3_INPUT_CKPT:-${PHASE2_OUTPUT}/pinn_plugin_final.pt}"
PROGRESSIVE_ENCODER_INPUT_CKPT="${PROGRESSIVE_ENCODER_INPUT_CKPT:-}"
ENCODER_CORE_INPUT_CKPT="${ENCODER_CORE_INPUT_CKPT:-${PHASE3_OUTPUT}/pinn_plugin_final.pt}"
ENCODER_ALPHA_INPUT_CKPT="${ENCODER_ALPHA_INPUT_CKPT:-${ENCODER_CORE_OUTPUT}/pinn_plugin_final.pt}"
ENCODER_T_INPUT_CKPT="${ENCODER_T_INPUT_CKPT:-${ENCODER_ALPHA_OUTPUT}/pinn_plugin_final.pt}"
ENCODER_J_INPUT_CKPT="${ENCODER_J_INPUT_CKPT:-${ENCODER_T_OUTPUT}/pinn_plugin_final.pt}"
ENCODER_D_INPUT_CKPT="${ENCODER_D_INPUT_CKPT:-${ENCODER_J_OUTPUT}/pinn_plugin_final.pt}"
ENCODER_PSI_INPUT_CKPT="${ENCODER_PSI_INPUT_CKPT:-${ENCODER_D_OUTPUT}/pinn_plugin_final.pt}"
FULL_PINN_INPUT_CKPT="${FULL_PINN_INPUT_CKPT:-}"

ABLATION_PRESET="${ABLATION_PRESET:-u_only_ufirst_prho}"
OBSERVABLE_TARGET_MODE="${OBSERVABLE_TARGET_MODE:-flow_only}"
SECONDARY_FIELD_STRATEGY="${SECONDARY_FIELD_STRATEGY:-u_first_constructor}"
STAGE1_ACTIVE_FIELD_SET="${STAGE1_ACTIVE_FIELD_SET:-u,p,rho}"
RECOVERY_ACTIVE_FIELD_SET="${RECOVERY_ACTIVE_FIELD_SET:-u,p,rho,d,alpha,T,j,D,psi}"
FIELD_ENABLE_SCHEDULE="${FIELD_ENABLE_SCHEDULE:-fixed_only_u_recovery}"
ENCODER_MODE="${ENCODER_MODE:-progressive}"
FIELD_RECOVERY_STEP_SCHEDULE="${FIELD_RECOVERY_STEP_SCHEDULE:-core:0,alpha+T:800,j+D:1500,psi:2100}"
FIELD_RECOVERY_LOSS_RAMP_STEPS="${FIELD_RECOVERY_LOSS_RAMP_STEPS:-150}"
RUN_FULL_PINN_AFTER_RECOVERY="${RUN_FULL_PINN_AFTER_RECOVERY:-0}"
FREEZE_U_ENCODER_DURING_RECOVERY="${FREEZE_U_ENCODER_DURING_RECOVERY:-1}"
START_STAGE="${START_STAGE:-encoder_progressive}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-encoder_progressive}"

STAGE1_NUM_EPOCHS="${STAGE1_NUM_EPOCHS:-2}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-3000}"
STAGE1_SAVE_STEPS="${STAGE1_SAVE_STEPS:-500}"
STAGE1_LEARNING_RATE="${STAGE1_LEARNING_RATE:-1e-5}"

ENCODER_NUM_EPOCHS="${ENCODER_NUM_EPOCHS:-2}"
ENCODER_MAX_STEPS="${ENCODER_MAX_STEPS:-3000}"
ENCODER_SAVE_STEPS="${ENCODER_SAVE_STEPS:-500}"
ENCODER_LEARNING_RATE="${ENCODER_LEARNING_RATE:-1e-5}"

PROGRESSIVE_ENCODER_NUM_EPOCHS="${PROGRESSIVE_ENCODER_NUM_EPOCHS:-1}"
PROGRESSIVE_ENCODER_MAX_STEPS="${PROGRESSIVE_ENCODER_MAX_STEPS:-2600}"
PROGRESSIVE_ENCODER_SAVE_STEPS="${PROGRESSIVE_ENCODER_SAVE_STEPS:-200}"
PROGRESSIVE_ENCODER_LEARNING_RATE="${PROGRESSIVE_ENCODER_LEARNING_RATE:-1e-5}"

FULL_PINN_NUM_EPOCHS="${FULL_PINN_NUM_EPOCHS:-3}"
FULL_PINN_MAX_STEPS="${FULL_PINN_MAX_STEPS:-}"
FULL_PINN_SAVE_STEPS="${FULL_PINN_SAVE_STEPS:-1000}"
FULL_PINN_LEARNING_RATE="${FULL_PINN_LEARNING_RATE:-1e-5}"
FULL_PINN_ENCODER_FREEZE_STEPS="${FULL_PINN_ENCODER_FREEZE_STEPS:-1000}"
FULL_PINN_ENCODER_LR_SCALE="${FULL_PINN_ENCODER_LR_SCALE:-0.3}"

PHASE2_MIN_TIMESTEP_BOUNDARY="${PHASE2_MIN_TIMESTEP_BOUNDARY:-0.75}"
PHASE2_MAX_TIMESTEP_BOUNDARY="${PHASE2_MAX_TIMESTEP_BOUNDARY:-1.00}"
PHASE3_MIN_TIMESTEP_BOUNDARY="${PHASE3_MIN_TIMESTEP_BOUNDARY:-0.50}"
PHASE3_MAX_TIMESTEP_BOUNDARY="${PHASE3_MAX_TIMESTEP_BOUNDARY:-1.00}"
ENCODER_MIN_TIMESTEP_BOUNDARY="${ENCODER_MIN_TIMESTEP_BOUNDARY:-0.00}"
ENCODER_MAX_TIMESTEP_BOUNDARY="${ENCODER_MAX_TIMESTEP_BOUNDARY:-1.00}"
FULL_PINN_MIN_TIMESTEP_BOUNDARY="${FULL_PINN_MIN_TIMESTEP_BOUNDARY:-0.00}"
FULL_PINN_MAX_TIMESTEP_BOUNDARY="${FULL_PINN_MAX_TIMESTEP_BOUNDARY:-1.00}"

CHECKPOINT_PYTHON="${CHECKPOINT_PYTHON:-python}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
DRY_RUN="${DRY_RUN:-0}"

case "${ENCODER_MODE}" in
  progressive|serial)
    ;;
  *)
    echo "[error] ENCODER_MODE must be progressive or serial, got: ${ENCODER_MODE}" >&2
    exit 1
    ;;
esac

normalize_stage_name() {
  local stage="${1:-}"
  case "${stage}" in
    phase2|stage1_phase2|stage1_phase2_mid_noise)
      echo "phase2"
      ;;
    phase3|stage1_phase3|stage1_phase3_wide_noise)
      echo "phase3"
      ;;
    progressive|encoder_progressive)
      echo "encoder_progressive"
      ;;
    core|encoder_core)
      echo "encoder_core"
      ;;
    alpha|encoder_alpha)
      echo "encoder_alpha"
      ;;
    T|encoder_T)
      echo "encoder_T"
      ;;
    j|encoder_j)
      echo "encoder_j"
      ;;
    D|encoder_D)
      echo "encoder_D"
      ;;
    psi|encoder_psi)
      echo "encoder_psi"
      ;;
    full_pinn)
      echo "full_pinn"
      ;;
    "")
      echo ""
      ;;
    *)
      echo "${stage}"
      ;;
  esac
}

stage_index() {
  local stage
  stage="$(normalize_stage_name "${1:-}")"
  case "${stage}" in
    phase2) echo 1 ;;
    phase3) echo 2 ;;
    encoder_progressive) echo 3 ;;
    encoder_core) echo 4 ;;
    encoder_alpha) echo 5 ;;
    encoder_T) echo 6 ;;
    encoder_j) echo 7 ;;
    encoder_D) echo 8 ;;
    encoder_psi) echo 9 ;;
    full_pinn) echo 10 ;;
    *)
      echo "[error] unknown stage name: ${1:-<empty>}" >&2
      exit 1
      ;;
  esac
}

should_run_stage() {
  local stage_name="$1"
  local stage_idx start_idx stop_idx
  stage_idx="$(stage_index "${stage_name}")"
  start_idx="$(stage_index "${START_STAGE}")"
  if [[ -n "${STOP_AFTER_STAGE}" ]]; then
    stop_idx="$(stage_index "${STOP_AFTER_STAGE}")"
  else
    stop_idx=999
  fi
  [[ "${stage_idx}" -ge "${start_idx}" && "${stage_idx}" -le "${stop_idx}" ]]
}

START_STAGE="$(normalize_stage_name "${START_STAGE}")"
STOP_AFTER_STAGE="$(normalize_stage_name "${STOP_AFTER_STAGE}")"

if [[ "${ENCODER_MODE}" == "progressive" ]]; then
  case "${START_STAGE}" in
    encoder_core|encoder_alpha|encoder_T|encoder_j|encoder_D|encoder_psi)
      START_STAGE="encoder_progressive"
      ;;
  esac
  case "${STOP_AFTER_STAGE}" in
    encoder_core|encoder_alpha|encoder_T|encoder_j|encoder_D|encoder_psi)
      STOP_AFTER_STAGE="encoder_progressive"
      ;;
  esac
fi

require_file() {
  local path="$1"
  local name="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[error] ${name} not found: ${path}" >&2
    exit 1
  fi
}

latest_step_checkpoint() {
  local search_dir="$1"
  local latest_path=""
  local latest_step=-1
  local candidate step_str step_num

  shopt -s nullglob
  for candidate in "${search_dir}"/step-*.pt; do
    step_str="${candidate##*/step-}"
    step_str="${step_str%.pt}"
    if [[ "${step_str}" =~ ^[0-9]+$ ]]; then
      step_num="${step_str}"
      if (( step_num > latest_step )); then
        latest_step="${step_num}"
        latest_path="${candidate}"
      fi
    fi
  done
  shopt -u nullglob

  echo "${latest_path}"
}

validate_stage1_checkpoint() {
  local ckpt_path="$1"
  "${CHECKPOINT_PYTHON}" - "${ckpt_path}" <<'PY'
import io
import pickle
import sys
import zipfile

path = sys.argv[1]

class DummyClass:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})

def dummy_rebuild(*args, **kwargs):
    return None

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name in {
            "_rebuild_tensor_v2",
            "_rebuild_tensor",
            "_rebuild_parameter",
            "_rebuild_sparse_tensor",
        }:
            return dummy_rebuild
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        if module == "__builtin__":
            import builtins
            return getattr(builtins, name)
        import builtins
        if module == "builtins" and hasattr(builtins, name):
            return getattr(builtins, name)
        return DummyClass

    def persistent_load(self, pid):
        return None

with zipfile.ZipFile(path, "r") as zf:
    data_name = None
    for candidate in zf.namelist():
        if candidate.endswith("/data.pkl"):
            data_name = candidate
            break
    if data_name is None:
        raise SystemExit(f"[error] {path}: data.pkl not found in checkpoint archive")
    payload = zf.read(data_name)

obj = SafeUnpickler(io.BytesIO(payload)).load()
if not isinstance(obj, dict):
    raise SystemExit(f"[error] {path}: unexpected checkpoint payload type {type(obj)!r}")

config = obj.get("config")
if not isinstance(config, dict):
    raise SystemExit(f"[error] {path}: missing config dict")

training_stage = config.get("training_stage")
format_version = int(config.get("checkpoint_format_version", 0) or 0)
encoder_stage_state = obj.get("encoder_stage_state_dict")

if training_stage != "observable_pretrain":
    raise SystemExit(
        f"[error] {path}: expected training_stage=observable_pretrain, got {training_stage!r}"
    )
if format_version < 14:
    raise SystemExit(
        f"[error] {path}: checkpoint_format_version={format_version}, require >= 14"
    )
if not isinstance(encoder_stage_state, dict) or not encoder_stage_state:
    raise SystemExit(
        f"[error] {path}: encoder_stage_state_dict missing or empty"
    )

print(
    f"[ok] checkpoint validated: training_stage={training_stage}, "
    f"checkpoint_format_version={format_version}, "
    f"encoder_stage_keys={len(encoder_stage_state)}"
)
PY
}

print_header() {
  echo "========================================"
  echo "Only-U encoder-first recovery plan"
  echo "========================================"
  echo "repo root                 : ${REPO_ROOT}"
  echo "base script               : ${BASE_SCRIPT}"
  echo "phase1 resume checkpoint  : ${PHASE1_RESUME_CKPT}"
  echo "output root               : ${OUTPUT_ROOT}"
  echo "ablation preset           : ${ABLATION_PRESET}"
  echo "observable target mode    : ${OBSERVABLE_TARGET_MODE}"
  echo "secondary field strategy  : ${SECONDARY_FIELD_STRATEGY}"
  echo "stage1 active field set   : ${STAGE1_ACTIVE_FIELD_SET}"
  echo "recovery active field set : ${RECOVERY_ACTIVE_FIELD_SET}"
  echo "field enable schedule     : ${FIELD_ENABLE_SCHEDULE}"
  echo "encoder mode              : ${ENCODER_MODE}"
  if [[ "${ENCODER_MODE}" == "progressive" ]]; then
    echo "progressive schedule      : ${FIELD_RECOVERY_STEP_SCHEDULE}"
    echo "progressive ramp steps    : ${FIELD_RECOVERY_LOSS_RAMP_STEPS}"
    echo "progressive input ckpt    : ${PROGRESSIVE_ENCODER_INPUT_CKPT:-<auto>}"
    echo "progressive output path   : ${PROGRESSIVE_ENCODER_OUTPUT}"
    echo "progressive max steps     : ${PROGRESSIVE_ENCODER_MAX_STEPS}"
  fi
  echo "run full pinn after rec.  : ${RUN_FULL_PINN_AFTER_RECOVERY}"
  echo "freeze u during recovery  : ${FREEZE_U_ENCODER_DURING_RECOVERY}"
  echo "start stage               : ${START_STAGE}"
  if [[ -n "${STOP_AFTER_STAGE}" ]]; then
    echo "stop after stage          : ${STOP_AFTER_STAGE}"
  else
    echo "stop after stage          : <run through end>"
  fi
  echo "skip completed            : ${SKIP_COMPLETED}"
  echo "dry run                   : ${DRY_RUN}"
  if [[ -n "${FLOW_BACKBONE_CKPT:-}" ]]; then
    echo "flow backbone ckpt        : ${FLOW_BACKBONE_CKPT}"
  else
    echo "flow backbone ckpt        : <inherit base script default>"
  fi
  echo "========================================"
}

run_stage() {
  local stage_name="$1"
  local expected_output="$2"
  shift 2

  if [[ "${SKIP_COMPLETED}" == "1" && -f "${expected_output}" ]]; then
    echo "[skip] ${stage_name}: found ${expected_output}"
    return 0
  fi

  echo
  echo "[run] ${stage_name}"
  echo "  expected final checkpoint: ${expected_output}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  dry run only; command not executed."
    printf '  command:'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi

  "$@"

  if [[ ! -f "${expected_output}" ]]; then
    echo "[error] ${stage_name}: expected final checkpoint not found: ${expected_output}" >&2
    exit 1
  fi

  echo "[ok] ${stage_name}: produced ${expected_output}"
}

run_encoder_stage() {
  local stage_name="$1"
  local phase_name="$2"
  local output_path="$3"
  local expected_output="$4"
  local resume_ckpt="$5"
  local stage1_encoder_ckpt="$6"

  if [[ -n "${resume_ckpt}" ]]; then
    require_file "${resume_ckpt}" "${stage_name} resume checkpoint"
  fi
  if [[ -n "${stage1_encoder_ckpt}" ]]; then
    require_file "${stage1_encoder_ckpt}" "${stage_name} stage1 encoder checkpoint"
  fi

  run_stage \
    "${stage_name}" \
    "${expected_output}" \
    env \
      TRAINING_STAGE="encoder_completion" \
      FIELD_RECOVERY_PHASE="${phase_name}" \
      RUN_FULL_PINN_AFTER_RECOVERY="${RUN_FULL_PINN_AFTER_RECOVERY}" \
      FREEZE_U_ENCODER_DURING_RECOVERY="${FREEZE_U_ENCODER_DURING_RECOVERY}" \
      OUTPUT_PATH="${output_path}" \
      PINN_CHECKPOINT="${resume_ckpt}" \
      STAGE1_PRETRAINED_ENCODER="${stage1_encoder_ckpt}" \
      NUM_EPOCHS="${ENCODER_NUM_EPOCHS}" \
      MAX_STEPS="${ENCODER_MAX_STEPS}" \
      SAVE_STEPS="${ENCODER_SAVE_STEPS}" \
      LEARNING_RATE="${ENCODER_LEARNING_RATE}" \
      MIN_TIMESTEP_BOUNDARY="${ENCODER_MIN_TIMESTEP_BOUNDARY}" \
      MAX_TIMESTEP_BOUNDARY="${ENCODER_MAX_TIMESTEP_BOUNDARY}" \
      ABLATION_PRESET="${ABLATION_PRESET}" \
      OBSERVABLE_TARGET_MODE="${OBSERVABLE_TARGET_MODE}" \
      SECONDARY_FIELD_STRATEGY="${SECONDARY_FIELD_STRATEGY}" \
      ACTIVE_FIELD_SET="${RECOVERY_ACTIVE_FIELD_SET}" \
      FIELD_ENABLE_SCHEDULE="${FIELD_ENABLE_SCHEDULE}" \
      FIELD_RECOVERY_STEP_SCHEDULE="" \
      FIELD_RECOVERY_LOSS_RAMP_STEPS="${FIELD_RECOVERY_LOSS_RAMP_STEPS}" \
      FLOW_BACKBONE_CKPT="${FLOW_BACKBONE_CKPT:-}" \
      bash "${BASE_SCRIPT}"
}

run_progressive_encoder_stage() {
  local stage_name="$1"
  local output_path="$2"
  local expected_output="$3"
  local stage1_encoder_ckpt="$4"

  require_file "${stage1_encoder_ckpt}" "${stage_name} stage1 encoder checkpoint"

  run_stage \
    "${stage_name}" \
    "${expected_output}" \
    env \
      TRAINING_STAGE="encoder_completion" \
      FIELD_RECOVERY_PHASE="psi" \
      RUN_FULL_PINN_AFTER_RECOVERY="${RUN_FULL_PINN_AFTER_RECOVERY}" \
      FREEZE_U_ENCODER_DURING_RECOVERY="${FREEZE_U_ENCODER_DURING_RECOVERY}" \
      OUTPUT_PATH="${output_path}" \
      PINN_CHECKPOINT="" \
      STAGE1_PRETRAINED_ENCODER="${stage1_encoder_ckpt}" \
      NUM_EPOCHS="${PROGRESSIVE_ENCODER_NUM_EPOCHS}" \
      MAX_STEPS="${PROGRESSIVE_ENCODER_MAX_STEPS}" \
      SAVE_STEPS="${PROGRESSIVE_ENCODER_SAVE_STEPS}" \
      LEARNING_RATE="${PROGRESSIVE_ENCODER_LEARNING_RATE}" \
      MIN_TIMESTEP_BOUNDARY="${ENCODER_MIN_TIMESTEP_BOUNDARY}" \
      MAX_TIMESTEP_BOUNDARY="${ENCODER_MAX_TIMESTEP_BOUNDARY}" \
      ABLATION_PRESET="${ABLATION_PRESET}" \
      OBSERVABLE_TARGET_MODE="${OBSERVABLE_TARGET_MODE}" \
      SECONDARY_FIELD_STRATEGY="${SECONDARY_FIELD_STRATEGY}" \
      ACTIVE_FIELD_SET="${RECOVERY_ACTIVE_FIELD_SET}" \
      FIELD_ENABLE_SCHEDULE="${FIELD_ENABLE_SCHEDULE}" \
      FIELD_RECOVERY_STEP_SCHEDULE="${FIELD_RECOVERY_STEP_SCHEDULE}" \
      FIELD_RECOVERY_LOSS_RAMP_STEPS="${FIELD_RECOVERY_LOSS_RAMP_STEPS}" \
      FLOW_BACKBONE_CKPT="${FLOW_BACKBONE_CKPT:-}" \
      bash "${BASE_SCRIPT}"
}

require_file "${BASE_SCRIPT}" "base training script"

PHASE2_FINAL="${PHASE2_OUTPUT}/pinn_plugin_final.pt"
PHASE3_FINAL="${PHASE3_OUTPUT}/pinn_plugin_final.pt"
PROGRESSIVE_ENCODER_FINAL="${PROGRESSIVE_ENCODER_OUTPUT}/pinn_plugin_final.pt"
ENCODER_CORE_FINAL="${ENCODER_CORE_OUTPUT}/pinn_plugin_final.pt"
ENCODER_ALPHA_FINAL="${ENCODER_ALPHA_OUTPUT}/pinn_plugin_final.pt"
ENCODER_T_FINAL="${ENCODER_T_OUTPUT}/pinn_plugin_final.pt"
ENCODER_J_FINAL="${ENCODER_J_OUTPUT}/pinn_plugin_final.pt"
ENCODER_D_FINAL="${ENCODER_D_OUTPUT}/pinn_plugin_final.pt"
ENCODER_PSI_FINAL="${ENCODER_PSI_OUTPUT}/pinn_plugin_final.pt"
FULL_PINN_FINAL="${FULL_PINN_OUTPUT}/pinn_plugin_final.pt"

resolve_progressive_encoder_input_ckpt() {
  local latest_ckpt=""
  if [[ -n "${PROGRESSIVE_ENCODER_INPUT_CKPT}" ]]; then
    require_file "${PROGRESSIVE_ENCODER_INPUT_CKPT}" "progressive encoder input checkpoint"
    return
  fi
  if [[ -f "${PHASE3_FINAL}" ]]; then
    PROGRESSIVE_ENCODER_INPUT_CKPT="${PHASE3_FINAL}"
    return
  fi
  latest_ckpt="$(latest_step_checkpoint "${PHASE3_OUTPUT}")"
  if [[ -n "${latest_ckpt}" ]]; then
    PROGRESSIVE_ENCODER_INPUT_CKPT="${latest_ckpt}"
    return
  fi
  PROGRESSIVE_ENCODER_INPUT_CKPT="${ENCODER_CORE_INPUT_CKPT}"
}

resolve_progressive_encoder_input_ckpt

if should_run_stage "phase2"; then
  require_file "${PHASE1_RESUME_CKPT}" "phase1 resume checkpoint"
fi

print_header
if should_run_stage "phase2"; then
  validate_stage1_checkpoint "${PHASE1_RESUME_CKPT}"
fi
if [[ "${ENCODER_MODE}" == "progressive" ]] && should_run_stage "encoder_progressive"; then
  validate_stage1_checkpoint "${PROGRESSIVE_ENCODER_INPUT_CKPT}"
fi

mkdir -p "${OUTPUT_ROOT}"

if should_run_stage "phase2"; then
  run_stage \
    "stage1 phase2 mid-noise" \
    "${PHASE2_FINAL}" \
    env \
      TRAINING_STAGE="observable_pretrain" \
      OUTPUT_PATH="${PHASE2_OUTPUT}" \
      PINN_CHECKPOINT="${PHASE2_INPUT_CKPT}" \
      STAGE1_PRETRAINED_ENCODER="" \
      FREEZE_U_ENCODER_DURING_RECOVERY="${FREEZE_U_ENCODER_DURING_RECOVERY}" \
      NUM_EPOCHS="${STAGE1_NUM_EPOCHS}" \
      MAX_STEPS="${STAGE1_MAX_STEPS}" \
      SAVE_STEPS="${STAGE1_SAVE_STEPS}" \
      LEARNING_RATE="${STAGE1_LEARNING_RATE}" \
      MIN_TIMESTEP_BOUNDARY="${PHASE2_MIN_TIMESTEP_BOUNDARY}" \
      MAX_TIMESTEP_BOUNDARY="${PHASE2_MAX_TIMESTEP_BOUNDARY}" \
      ABLATION_PRESET="${ABLATION_PRESET}" \
      OBSERVABLE_TARGET_MODE="${OBSERVABLE_TARGET_MODE}" \
      SECONDARY_FIELD_STRATEGY="${SECONDARY_FIELD_STRATEGY}" \
      ACTIVE_FIELD_SET="${STAGE1_ACTIVE_FIELD_SET}" \
      FIELD_ENABLE_SCHEDULE="${FIELD_ENABLE_SCHEDULE}" \
      FLOW_BACKBONE_CKPT="${FLOW_BACKBONE_CKPT:-}" \
      bash "${BASE_SCRIPT}"

  if [[ "${DRY_RUN}" != "1" ]]; then
    require_file "${PHASE2_FINAL}" "stage1 phase2 final checkpoint"
  fi
else
  echo "[skip] stage1 phase2 mid-noise: before start stage ${START_STAGE}"
fi

if should_run_stage "phase3"; then
  run_stage \
    "stage1 phase3 wide-noise" \
    "${PHASE3_FINAL}" \
    env \
      TRAINING_STAGE="observable_pretrain" \
      OUTPUT_PATH="${PHASE3_OUTPUT}" \
      PINN_CHECKPOINT="${PHASE3_INPUT_CKPT}" \
      STAGE1_PRETRAINED_ENCODER="" \
      FREEZE_U_ENCODER_DURING_RECOVERY="${FREEZE_U_ENCODER_DURING_RECOVERY}" \
      NUM_EPOCHS="${STAGE1_NUM_EPOCHS}" \
      MAX_STEPS="${STAGE1_MAX_STEPS}" \
      SAVE_STEPS="${STAGE1_SAVE_STEPS}" \
      LEARNING_RATE="${STAGE1_LEARNING_RATE}" \
      MIN_TIMESTEP_BOUNDARY="${PHASE3_MIN_TIMESTEP_BOUNDARY}" \
      MAX_TIMESTEP_BOUNDARY="${PHASE3_MAX_TIMESTEP_BOUNDARY}" \
      ABLATION_PRESET="${ABLATION_PRESET}" \
      OBSERVABLE_TARGET_MODE="${OBSERVABLE_TARGET_MODE}" \
      SECONDARY_FIELD_STRATEGY="${SECONDARY_FIELD_STRATEGY}" \
      ACTIVE_FIELD_SET="${STAGE1_ACTIVE_FIELD_SET}" \
      FIELD_ENABLE_SCHEDULE="${FIELD_ENABLE_SCHEDULE}" \
      FLOW_BACKBONE_CKPT="${FLOW_BACKBONE_CKPT:-}" \
      bash "${BASE_SCRIPT}"

  if [[ "${DRY_RUN}" != "1" ]]; then
    require_file "${PHASE3_FINAL}" "stage1 phase3 final checkpoint"
  fi
else
  echo "[skip] stage1 phase3 wide-noise: before start stage ${START_STAGE}"
fi

if [[ "${ENCODER_MODE}" == "progressive" ]]; then
  if should_run_stage "encoder_progressive"; then
    run_progressive_encoder_stage \
      "encoder_progressive" \
      "${PROGRESSIVE_ENCODER_OUTPUT}" \
      "${PROGRESSIVE_ENCODER_FINAL}" \
      "${PROGRESSIVE_ENCODER_INPUT_CKPT}"
  else
    echo "[skip] encoder_progressive: outside requested stage range"
  fi
else
  if should_run_stage "encoder_core"; then
    run_encoder_stage \
      "encoder_core" \
      "core" \
      "${ENCODER_CORE_OUTPUT}" \
      "${ENCODER_CORE_FINAL}" \
      "" \
      "${ENCODER_CORE_INPUT_CKPT}"
  else
    echo "[skip] encoder_core: outside requested stage range"
  fi

  if should_run_stage "encoder_alpha"; then
    run_encoder_stage \
      "encoder_alpha" \
      "alpha" \
      "${ENCODER_ALPHA_OUTPUT}" \
      "${ENCODER_ALPHA_FINAL}" \
      "${ENCODER_ALPHA_INPUT_CKPT}" \
      ""
  else
    echo "[skip] encoder_alpha: outside requested stage range"
  fi

  if should_run_stage "encoder_T"; then
    run_encoder_stage \
      "encoder_T" \
      "T" \
      "${ENCODER_T_OUTPUT}" \
      "${ENCODER_T_FINAL}" \
      "${ENCODER_T_INPUT_CKPT}" \
      ""
  else
    echo "[skip] encoder_T: outside requested stage range"
  fi

  if should_run_stage "encoder_j"; then
    run_encoder_stage \
      "encoder_j" \
      "j" \
      "${ENCODER_J_OUTPUT}" \
      "${ENCODER_J_FINAL}" \
      "${ENCODER_J_INPUT_CKPT}" \
      ""
  else
    echo "[skip] encoder_j: outside requested stage range"
  fi

  if should_run_stage "encoder_D"; then
    run_encoder_stage \
      "encoder_D" \
      "D" \
      "${ENCODER_D_OUTPUT}" \
      "${ENCODER_D_FINAL}" \
      "${ENCODER_D_INPUT_CKPT}" \
      ""
  else
    echo "[skip] encoder_D: outside requested stage range"
  fi

  if should_run_stage "encoder_psi"; then
    run_encoder_stage \
      "encoder_psi" \
      "psi" \
      "${ENCODER_PSI_OUTPUT}" \
      "${ENCODER_PSI_FINAL}" \
      "${ENCODER_PSI_INPUT_CKPT}" \
      ""
  else
    echo "[skip] encoder_psi: outside requested stage range"
  fi
fi

FULL_PINN_SOURCE_CKPT="${FULL_PINN_INPUT_CKPT}"
if [[ -z "${FULL_PINN_INPUT_CKPT:-}" ]]; then
  if [[ "${ENCODER_MODE}" == "progressive" ]]; then
    FULL_PINN_SOURCE_CKPT="${PROGRESSIVE_ENCODER_FINAL}"
  else
    FULL_PINN_SOURCE_CKPT="${ENCODER_PSI_FINAL}"
  fi
fi

if [[ "${RUN_FULL_PINN_AFTER_RECOVERY}" == "1" ]] && should_run_stage "full_pinn"; then
  run_stage \
    "full_pinn from encoder_psi" \
    "${FULL_PINN_FINAL}" \
    env \
      TRAINING_STAGE="full_pinn" \
      FIELD_RECOVERY_PHASE="psi" \
      RUN_FULL_PINN_AFTER_RECOVERY="1" \
      FREEZE_U_ENCODER_DURING_RECOVERY="${FREEZE_U_ENCODER_DURING_RECOVERY}" \
      OUTPUT_PATH="${FULL_PINN_OUTPUT}" \
      PINN_CHECKPOINT="${FULL_PINN_SOURCE_CKPT}" \
      STAGE1_PRETRAINED_ENCODER="" \
      NUM_EPOCHS="${FULL_PINN_NUM_EPOCHS}" \
      MAX_STEPS="${FULL_PINN_MAX_STEPS}" \
      SAVE_STEPS="${FULL_PINN_SAVE_STEPS}" \
      LEARNING_RATE="${FULL_PINN_LEARNING_RATE}" \
      ENCODER_FREEZE_STEPS="${FULL_PINN_ENCODER_FREEZE_STEPS}" \
      ENCODER_LR_SCALE="${FULL_PINN_ENCODER_LR_SCALE}" \
      MIN_TIMESTEP_BOUNDARY="${FULL_PINN_MIN_TIMESTEP_BOUNDARY}" \
      MAX_TIMESTEP_BOUNDARY="${FULL_PINN_MAX_TIMESTEP_BOUNDARY}" \
      ABLATION_PRESET="${ABLATION_PRESET}" \
      OBSERVABLE_TARGET_MODE="${OBSERVABLE_TARGET_MODE}" \
      SECONDARY_FIELD_STRATEGY="${SECONDARY_FIELD_STRATEGY}" \
      ACTIVE_FIELD_SET="${RECOVERY_ACTIVE_FIELD_SET}" \
      FIELD_ENABLE_SCHEDULE="${FIELD_ENABLE_SCHEDULE}" \
      FLOW_BACKBONE_CKPT="${FLOW_BACKBONE_CKPT:-}" \
      bash "${BASE_SCRIPT}"
elif [[ "${RUN_FULL_PINN_AFTER_RECOVERY}" == "1" ]]; then
  echo "[skip] full_pinn from encoder_psi: outside requested stage range"
fi

echo
echo "[done] only-u encoder-first plan completed."
echo "  phase2 final       : ${PHASE2_FINAL}"
echo "  phase3 final       : ${PHASE3_FINAL}"
if [[ "${ENCODER_MODE}" == "progressive" ]]; then
  echo "  encoder_progressive: ${PROGRESSIVE_ENCODER_FINAL}"
else
  echo "  encoder_core final : ${ENCODER_CORE_FINAL}"
  echo "  encoder_alpha final: ${ENCODER_ALPHA_FINAL}"
  echo "  encoder_T final    : ${ENCODER_T_FINAL}"
  echo "  encoder_j final    : ${ENCODER_J_FINAL}"
  echo "  encoder_D final    : ${ENCODER_D_FINAL}"
  echo "  encoder_psi final  : ${ENCODER_PSI_FINAL}"
fi
if [[ "${RUN_FULL_PINN_AFTER_RECOVERY}" == "1" ]]; then
  echo "  full_pinn final    : ${FULL_PINN_FINAL}"
fi
