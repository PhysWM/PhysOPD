#!/usr/bin/env python3
"""
Train PILA's PhysicsAdapter on top of a frozen AnyFlow-Wan2.1 1.3B backbone.

This is intentionally a separate entry point from train_pinn.py.  The original
training script is coupled to DiffSynth's Wan scheduler/model_fn contract; this
script follows AnyFlow's flow-map contract:

    z_t = scheduler.scale_noise(x, t, eps)
    u_theta = AnyFlowTransformer(z_t, t, r, c)
    x0_hat ~= z_t - sigma(t) * u_theta

Only the PhysicsAdapter is optimized.  The AnyFlow transformer, text encoder,
VAE, and scheduler remain frozen.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


DEFAULT_ANYFLOW_ROOT = "/home/dataset-assist-0/algorithm/cong.wang/try/AnyFlow"
DEFAULT_ANYFLOW_MODEL = (
    "/home/dataset-assist-0/algorithm/cong.wang/try/AnyFlow/"
    "experiments/pretrained_models/AnyFlow-Wan2.1-T2V-1.3B-Diffusers"
)
DEFAULT_DATASET_BASE = (
    "/home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data"
)
DEFAULT_DATASET_META = (
    "/home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data/metadata_standard.csv"
)


PHENOMENON_TO_RESIDUAL_METHOD = {
    "Rigid Body": "rigid_body_residual",
    "Elastic": "elastic_residual",
    "Fluid": "fluid_residual",
    "Compressible Flow": "compressible_flow_residual",
    "Phase Change": "phase_change_residual",
    "Collision/Contact": "collision_contact_residual",
    "Granular": "granular_residual",
    "Fracture": "fracture_residual",
    "Thermal": "thermal_residual",
    "Optical": "optical_residual",
}


def distributed_is_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if distributed_is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if distributed_is_initialized() else 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not distributed_is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank, world_size


def cleanup_distributed():
    if distributed_is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def average_gradients(module):
    if not distributed_is_initialized():
        return
    world_size = float(dist.get_world_size())
    for param in module.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)

FIELD_RECOVERY_PHASES = ("core", "alpha", "T", "j", "D", "psi")
FIELD_RECOVERY_PHASE_TO_INDEX = {name: idx for idx, name in enumerate(FIELD_RECOVERY_PHASES)}
STAGE1_ENCODER_STATE_PREFIXES = (
    "physics_encoder_shared.",
    "shared_attribute_head.",
    "u_head.",
    "d_head.",
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_pila_modules(pila_root):
    """Load adapter/operator files without importing full diffsynth."""
    pila_root = Path(pila_root)
    diffsynth_pkg = types.ModuleType("diffsynth")
    models_pkg = types.ModuleType("diffsynth.models")
    diffsynth_pkg.__path__ = [str(pila_root / "diffsynth")]
    models_pkg.__path__ = [str(pila_root / "diffsynth" / "models")]
    sys.modules.setdefault("diffsynth", diffsynth_pkg)
    sys.modules.setdefault("diffsynth.models", models_pkg)

    contracts = _load_module(
        "diffsynth.models.pinn_contracts",
        str(pila_root / "diffsynth" / "models" / "pinn_contracts.py"),
    )
    operators = _load_module(
        "diffsynth.models.pinn_operators",
        str(pila_root / "diffsynth" / "models" / "pinn_operators.py"),
    )
    adapter = _load_module(
        "diffsynth.models.pinn_adapter",
        str(pila_root / "diffsynth" / "models" / "pinn_adapter.py"),
    )
    return contracts, operators, adapter


def safe_text(value):
    return "" if value is None else str(value).strip()


def hash_to_id(text, modulo):
    if modulo <= 1:
        return 0
    text = safe_text(text).lower()
    if text == "":
        return 0
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], byteorder="big", signed=False) % (modulo - 1)) + 1


def parse_numeric_range(text):
    import re

    matches = re.findall(r"-?\d+(?:\.\d+)?", safe_text(text).lower())
    if not matches:
        return 0.0, 0.0, 0.0, 0.0
    values = [float(x) for x in matches]
    return min(values), max(values), sum(values) / len(values), 1.0


def encode_q_field(text, dim):
    import re

    vec = [0.0 for _ in range(dim)]
    text = safe_text(text).lower()
    if text == "" or dim <= 0:
        return vec
    tokens = re.split(r"[,;|/]| and |\.", text)
    for token in [re.sub(r"\s+", " ", t).strip() for t in tokens if t.strip()]:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:8], byteorder="big", signed=False) % dim
        vec[idx] = 1.0
    return vec


def normalize_label(raw_label, phenomenon_labels, default_label="Rigid Body"):
    label_lookup = {x.casefold(): x for x in phenomenon_labels}
    raw = safe_text(raw_label)
    if raw.casefold() in label_lookup:
        return label_lookup[raw.casefold()]
    compact = raw.replace("_", " ").replace("-", " ").casefold()
    if compact in label_lookup:
        return label_lookup[compact]
    return default_label if default_label in phenomenon_labels else phenomenon_labels[0]


def parse_labels(raw_label, phenomenon_labels, default_label="Rigid Body"):
    labels = []
    for part in safe_text(raw_label).replace(";", ",").split(","):
        label = normalize_label(part, phenomenon_labels, default_label="")
        if label and label in phenomenon_labels and label not in labels:
            labels.append(label)
    if not labels:
        labels = [default_label if default_label in phenomenon_labels else phenomenon_labels[0]]
    return labels


def metadata_from_row(row, phenomenon_labels, n_text_vocab_size=2048, q_dim=64):
    label_keys = ("label", "label_name", "phenomenon", "category", "physics_label")
    raw_label = ""
    for key in label_keys:
        if safe_text(row.get(key)):
            raw_label = row.get(key)
            break
    labels = parse_labels(raw_label, phenomenon_labels)
    label_ids = [phenomenon_labels.index(label) for label in labels]
    label_name = labels[0]
    label_id = label_ids[0]

    n_raw = []
    for key in ("n0", "n1", "n2"):
        n_raw.append(row.get(key, ""))
    if not any(safe_text(x) for x in n_raw):
        for key in ("physical_parameters", "physics_parameters", "parameters"):
            if safe_text(row.get(key)):
                n_raw = [row.get(key, ""), "", ""]
                break
    n_numeric = []
    for value in n_raw:
        n_numeric.extend(parse_numeric_range(value))
    while len(n_numeric) < 12:
        n_numeric.append(0.0)
    n_numeric = n_numeric[:12]

    q_vector = [0.0 for _ in range(q_dim)]
    q_keys = ("q0", "q1", "q2", "q4", "caption", "prompt", "text")
    for key in q_keys:
        q_encoded = encode_q_field(row.get(key, ""), q_dim)
        q_vector = [min(1.0, a + b) for a, b in zip(q_vector, q_encoded)]
    q3 = safe_text(row.get("q3", "")).lower()
    if q3 in {"yes", "true", "1"} and q_dim > 0:
        q_vector[0] = 1.0
    elif q3 in {"no", "false", "0"} and q_dim > 1:
        q_vector[1] = 1.0

    return {
        "label_name": label_name,
        "label_id": label_id,
        "label_ids": label_ids,
        "n_numeric": torch.tensor(n_numeric, dtype=torch.float32),
        "n_text_ids": torch.tensor([hash_to_id(x, n_text_vocab_size) for x in n_raw], dtype=torch.long),
        "q_vector": torch.tensor(q_vector, dtype=torch.float32),
        "parse_success_ratio": torch.tensor(sum(x[3] for x in [parse_numeric_range(v) for v in n_raw]) / 3.0, dtype=torch.float32),
    }


def metadata_to_device(metadata, device):
    result = {}
    for key, value in metadata.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def metadata_to_jsonable(metadata):
    result = {}
    for key, value in metadata.items():
        if torch.is_tensor(value):
            result[key] = value.detach().cpu().tolist()
        else:
            result[key] = value
    return result


def label_only_metadata(metadata):
    if not isinstance(metadata, dict):
        return None
    keys = (
        "label_name",
        "label_id",
        "label_ids",
        "parse_success_ratio",
        "frame_count",
        "frame_delta_t",
        "frame_time_grid",
        "physics_time_source",
    )
    return {key: metadata[key] for key in keys if key in metadata}


class CsvVideoDataset(Dataset):
    def __init__(
        self,
        base_path,
        metadata_path,
        phenomenon_labels,
        height=480,
        width=832,
        num_frames=81,
        max_samples=None,
    ):
        self.base_path = Path(base_path)
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.phenomenon_labels = list(phenomenon_labels)
        with open(metadata_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if max_samples is not None and int(max_samples) > 0:
            rows = rows[: int(max_samples)]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def _resolve_video_path(self, row):
        candidate_keys = (
            "video",
            "video_path",
            "path",
            "file",
            "file_path",
            "mp4",
            "relative_path",
        )
        for key in candidate_keys:
            value = safe_text(row.get(key))
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = self.base_path / path
            if path.exists():
                return path
        for value in row.values():
            value = safe_text(value)
            if value.endswith(".mp4"):
                path = Path(value)
                if not path.is_absolute():
                    path = self.base_path / path
                if path.exists():
                    return path
        raise FileNotFoundError(f"Could not resolve video path from row keys={list(row.keys())}")

    def _prompt_from_row(self, row):
        for key in ("caption", "prompt", "text", "description"):
            value = safe_text(row.get(key))
            if value:
                return value
        return "A physically plausible video."

    def _load_video(self, video_path):
        import decord

        reader = decord.VideoReader(str(video_path), num_threads=1)
        total = len(reader)
        if total <= 0:
            raise RuntimeError(f"Empty video: {video_path}")
        if total >= self.num_frames:
            indices = torch.linspace(0, total - 1, self.num_frames).round().long().tolist()
        else:
            indices = list(range(total)) + [total - 1] * (self.num_frames - total)
        frames = reader.get_batch(indices).asnumpy()
        video = torch.from_numpy(frames).float() / 255.0
        video = video.permute(0, 3, 1, 2).contiguous()
        video = F.interpolate(
            video,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        return video

    def __getitem__(self, index):
        row = self.rows[index]
        video_path = self._resolve_video_path(row)
        return {
            "pixel_values": self._load_video(video_path),
            "prompt": self._prompt_from_row(row),
            "metadata": metadata_from_row(row, self.phenomenon_labels),
            "video_path": str(video_path),
            "index": index,
        }


def collate_batch(items):
    if len(items) != 1:
        raise ValueError(
            "This entry point intentionally uses batch_size=1 because PILA metadata "
            "routing is sample-specific. Use gradient_accumulation_steps for larger effective batch."
        )
    item = items[0]
    return {
        "pixel_values": item["pixel_values"].unsqueeze(0),
        "prompts": [item["prompt"]],
        "metadata": item["metadata"],
        "video_path": item["video_path"],
        "index": item["index"],
    }


def save_checkpoint(path, adapter, optimizer, step, args, contracts):
    if not is_main_process():
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "config": {
            "checkpoint_format_version": 14,
            "training_stage": args.training_stage,
            "backbone_type": "anyflow_wan2.1_t2v_1.3b",
            "anyflow_model_path": args.anyflow_model_path,
            "physics_weight": args.physics_weight,
            "physics_weight_target": args.physics_weight_target,
            "physics_warmup_steps": args.physics_warmup_steps,
            "adapter_hidden_dim": args.adapter_hidden_dim,
            "moe_top_k": args.moe_top_k,
            "core_ablation_mode": args.core_ablation_mode,
            "field_recovery_phase": args.field_recovery_phase,
        },
        "backbone_type": "anyflow_wan2.1_t2v_1.3b",
        "anyflow_model_path": args.anyflow_model_path,
        "adapter_config": {
            "latent_dim": 16,
            "hidden_dim": args.adapter_hidden_dim,
            "physics_attr_dim": int(contracts.PHYSICS_ATTR_DIM),
            "num_phenomena": len(contracts.PHENOMENON_LABELS),
            "moe_top_k": args.moe_top_k,
            "physics_state_mode": "x0_hat",
            "use_sigma_gate": True,
            "sigma_gate_curve": args.sigma_gate_curve,
            "use_sigma_conditioning": True,
            "sigma_gate_floor": args.sigma_gate_floor,
            "core_ablation_mode": args.core_ablation_mode,
        },
        "physics_adapter_state_dict": adapter.state_dict(),
        "encoder_stage_state_dict": {
            key: value.detach().cpu()
            for key, value in adapter.state_dict().items()
            if key.startswith(STAGE1_ENCODER_STATE_PREFIXES)
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "training_state": {
            "current_step": int(step),
            "current_epoch": 0,
        },
        "args": vars(args),
    }
    torch.save(payload, path)


def load_adapter_checkpoint(path, adapter):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("physics_adapter_state_dict", "adapter_state_dict", "state_dict"):
            if isinstance(checkpoint.get(key), dict):
                state = checkpoint[key]
                break
    cleaned = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        clean_key = key
        for prefix in ("module.", "physics_adapter."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        cleaned[clean_key] = value
    return adapter.load_state_dict(cleaned, strict=False)


def strip_known_state_dict_prefixes(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        clean_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "physics_adapter."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    changed = True
        cleaned[clean_key] = value
    return cleaned


def load_stage1_encoder_checkpoint(path, adapter):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("encoder_stage_state_dict"), dict):
            state_dict = checkpoint["encoder_stage_state_dict"]
        elif isinstance(checkpoint.get("physics_adapter_state_dict"), dict):
            state_dict = checkpoint["physics_adapter_state_dict"]
    state_dict = strip_known_state_dict_prefixes(state_dict)
    loaded = []
    for prefix, module, required in (
        ("physics_encoder_shared.", adapter.physics_encoder_shared, True),
        ("shared_attribute_head.", adapter.shared_attribute_head, True),
        ("u_head.", adapter.u_head, False),
        ("d_head.", adapter.d_head, False),
    ):
        module_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not module_state:
            if required:
                raise RuntimeError(f"Stage1 checkpoint is missing required prefix {prefix!r}: {path}")
            continue
        result = module.load_state_dict(module_state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Incompatible stage1 module {prefix}: "
                f"missing={result.missing_keys[:20]}, unexpected={result.unexpected_keys[:20]}"
            )
        loaded.append(prefix.rstrip("."))
    if not loaded:
        raise RuntimeError(f"No stage1 encoder weights were loaded from {path}")
    return loaded


def set_module_trainability(module, requires_grad, training=None):
    if module is None:
        return
    module.requires_grad_(bool(requires_grad))
    module.train(bool(requires_grad) if training is None else bool(training))


def observable_stage_modules(adapter, target_mode):
    modules = [adapter.physics_encoder_shared, adapter.shared_attribute_head, adapter.u_head]
    if str(target_mode) != "flow_only":
        modules.append(adapter.d_head)
    return modules


def observable_head_modules(adapter, target_mode):
    modules = [adapter.u_head]
    if str(target_mode) != "flow_only":
        modules.append(adapter.d_head)
    return modules


def stage2_protected_modules(adapter):
    return [
        adapter.physics_encoder_shared,
        adapter.shared_attribute_head,
        adapter.sigma_condition_proj,
        adapter.n_numeric_proj,
        adapter.n_text_embedding,
        adapter.q_proj,
        adapter.condition_fuse,
        adapter.expert_router,
        adapter.operator_experts,
    ]


def encoder_completion_modules(adapter, args):
    modules = [adapter.prho_constructor]
    if not args.freeze_u_encoder_during_recovery:
        modules[:0] = [adapter.physics_encoder_shared, adapter.shared_attribute_head, adapter.u_head]
    phases_to_train = list(FIELD_RECOVERY_PHASES) if args.field_recovery_step_schedule else [args.field_recovery_phase]
    max_phase = max(FIELD_RECOVERY_PHASE_TO_INDEX.get(str(p), 0) for p in phases_to_train)
    if max_phase >= FIELD_RECOVERY_PHASE_TO_INDEX["alpha"]:
        modules.append(adapter.alpha_head)
    if max_phase >= FIELD_RECOVERY_PHASE_TO_INDEX["T"]:
        modules.append(adapter.T_head)
    if max_phase >= FIELD_RECOVERY_PHASE_TO_INDEX["j"]:
        modules.append(adapter.j_head)
    if max_phase >= FIELD_RECOVERY_PHASE_TO_INDEX["D"]:
        modules.append(adapter.D_head)
    if max_phase >= FIELD_RECOVERY_PHASE_TO_INDEX["psi"]:
        modules.append(adapter.psi_head)
    return modules


def configure_training_stage(adapter, args, current_step=0):
    adapter.train()
    if args.training_stage == "observable_pretrain":
        adapter.requires_grad_(False)
        for module in observable_stage_modules(adapter, args.observable_target_mode):
            set_module_trainability(module, True, training=True)
        return
    if args.training_stage == "encoder_completion":
        adapter.requires_grad_(False)
        for module in encoder_completion_modules(adapter, args):
            set_module_trainability(module, True, training=True)
        set_module_trainability(adapter.d_head, False, training=False)
        return
    adapter.requires_grad_(True)
    for module in observable_head_modules(adapter, args.observable_target_mode):
        set_module_trainability(module, False, training=False)
    should_freeze = int(current_step) < int(args.encoder_freeze_steps)
    for module in stage2_protected_modules(adapter):
        set_module_trainability(module, not should_freeze, training=not should_freeze)


def active_label_names(metadata, phenomenon_labels):
    names = []
    label_ids = metadata.get("label_ids") if isinstance(metadata, dict) else None
    if isinstance(label_ids, torch.Tensor):
        label_ids_iter = label_ids.detach().view(-1).tolist()
    elif isinstance(label_ids, (list, tuple)):
        label_ids_iter = list(label_ids)
    else:
        label_ids_iter = []
    for label_id in label_ids_iter:
        try:
            idx = int(label_id)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(phenomenon_labels):
            names.append(phenomenon_labels[idx])
    if not names and isinstance(metadata, dict):
        label_name = safe_text(metadata.get("label_name"))
        names.extend(parse_labels(label_name, phenomenon_labels))
    return list(dict.fromkeys(names))


def video_time_metadata(batch_size, num_frames, device, dtype):
    frame_delta_t = 1.0 / max(int(num_frames) - 1, 1)
    frame_time_grid = torch.linspace(0.0, 1.0, steps=max(int(num_frames), 1), device=device, dtype=dtype)
    return {
        "frame_count": torch.full((batch_size,), float(num_frames), device=device, dtype=dtype),
        "frame_delta_t": torch.full((batch_size,), float(frame_delta_t), device=device, dtype=dtype),
        "frame_time_grid": frame_time_grid.unsqueeze(0).repeat(batch_size, 1),
        "physics_time_source": "video_frames",
    }


def build_observable_proxy_targets(physics_state):
    with torch.no_grad():
        ref = physics_state.detach().float()
        flow_base = ref[:, :2]
        if flow_base.shape[1] < 2:
            flow_base = flow_base.repeat(1, 2, 1, 1, 1)[:, :2]
        flow_proxy = torch.zeros_like(flow_base)
        if flow_base.shape[2] > 1:
            flow_proxy[:, :, :-1] = flow_base[:, :, 1:] - flow_base[:, :, :-1]
            flow_proxy[:, :, -1:] = flow_proxy[:, :, -2:-1]
        deformation_proxy = flow_base - flow_base.mean(dim=2, keepdim=True)
        energy = flow_proxy.abs().mean(dim=1, keepdim=True)
        flat = energy.flatten(1)
        denom = torch.quantile(flat, 0.90, dim=1).view(-1, 1, 1, 1, 1).clamp_min(1e-6)
        proxy_conf = torch.clamp(energy / denom, 0.05, 1.0).to(dtype=physics_state.dtype)
    return {
        "flow_proxy": flow_proxy.to(device=physics_state.device, dtype=physics_state.dtype),
        "deformation_proxy": deformation_proxy.to(device=physics_state.device, dtype=physics_state.dtype),
        "proxy_conf": proxy_conf,
    }


def observable_alignment_terms(predicted, target, proxy_conf, target_mode="flow_plus_deformation"):
    def match_channels(tensor, channels):
        if tensor.shape[1] == channels:
            return tensor
        if tensor.shape[1] > channels:
            return tensor[:, :channels]
        repeat = math.ceil(float(channels) / float(max(tensor.shape[1], 1)))
        return tensor.repeat(1, repeat, 1, 1, 1)[:, :channels]

    proxy_conf = proxy_conf.to(dtype=predicted["flow"].dtype)
    conf_norm = proxy_conf.sum().clamp_min(1.0)
    flow_target = match_channels(target["flow_proxy"], predicted["flow"].shape[1])
    flow_map = F.smooth_l1_loss(predicted["flow"], flow_target, reduction="none").mean(dim=1, keepdim=True)
    flow_error = (flow_map * proxy_conf).sum() / conf_norm
    deformation_error = torch.zeros_like(flow_error)
    if str(target_mode) != "flow_only":
        deformation_target = match_channels(target["deformation_proxy"], predicted["deformation"].shape[1])
        deformation_map = F.smooth_l1_loss(
            predicted["deformation"],
            deformation_target,
            reduction="none",
        ).mean(dim=1, keepdim=True)
        deformation_error = (deformation_map * proxy_conf).sum() / conf_norm
    return flow_error + deformation_error, {
        "flow_error": flow_error,
        "deformation_error": deformation_error,
    }


def active_recovery_phase(args, step):
    schedule = parse_field_recovery_schedule(args.field_recovery_step_schedule)
    if not schedule:
        return str(args.field_recovery_phase or "core")
    active = "core"
    for phase, start in schedule:
        if int(step) >= int(start):
            active = phase
        else:
            break
    return active


def parse_field_recovery_schedule(text):
    entries = []
    for part in safe_text(text).split(","):
        if not part.strip():
            continue
        name, _, start = part.partition(":")
        name = name.strip()
        if name not in FIELD_RECOVERY_PHASE_TO_INDEX:
            raise ValueError(f"Unknown field recovery phase in schedule: {name}")
        entries.append((name, int(start.strip() or 0)))
    return sorted(entries, key=lambda x: x[1])


def phase_at_least(args, step, target_phase):
    return FIELD_RECOVERY_PHASE_TO_INDEX.get(active_recovery_phase(args, step), -1) >= FIELD_RECOVERY_PHASE_TO_INDEX[target_phase]


def encoder_completion_loss_weight(args, step, owner_phase):
    if not phase_at_least(args, step, owner_phase):
        return 0.0
    schedule = parse_field_recovery_schedule(args.field_recovery_step_schedule)
    if not schedule:
        return 1.0 if str(args.field_recovery_phase) == str(owner_phase) else 0.25
    starts = dict(schedule)
    active = active_recovery_phase(args, step)
    owner_start = int(starts.get(owner_phase, 0))
    active_start = int(starts.get(active, 0))
    base = 1.0 if owner_start == active_start else 0.25
    if owner_start <= 0 or args.field_recovery_loss_ramp_steps <= 0 or owner_start != active_start:
        return base
    return base * min(float(step - owner_start + 1) / float(args.field_recovery_loss_ramp_steps), 1.0)


def compute_pde_loss(adapter, pde_residuals, metadata):
    cache = getattr(adapter, "_cache", {})
    bank = cache.get("fused_attribute_bank_live")
    if bank is None:
        return bank.new_tensor(0.0) if torch.is_tensor(bank) else torch.tensor(0.0), {}
    label_name = metadata.get("label_name", "Rigid Body")
    method_name = PHENOMENON_TO_RESIDUAL_METHOD.get(label_name)
    if method_name is None:
        raise KeyError(f"Unsupported physics label: {label_name}")
    method = getattr(pde_residuals, method_name)
    loss, info = method(bank.float().clamp(-10.0, 10.0), metadata=metadata)
    if not torch.isfinite(loss):
        loss = torch.zeros((), device=bank.device, dtype=bank.dtype)
    return torch.clamp(loss.float(), 0.0, 100.0), dict(info)


def effective_physics_weight(args, step):
    target = args.physics_weight if args.physics_weight_target is None else args.physics_weight_target
    if args.physics_warmup_steps <= 0:
        return float(target)
    alpha = min(max(float(step + 1) / float(args.physics_warmup_steps), 0.0), 1.0)
    return float(args.physics_weight) + alpha * (float(target) - float(args.physics_weight))


def effective_sigma_threshold(args, step):
    if args.physics_warmup_steps <= 0:
        return float(args.expert_pde_sigma_threshold_target)
    alpha = min(max(float(step + 1) / float(args.physics_warmup_steps), 0.0), 1.0)
    return float(args.expert_pde_sigma_threshold) + alpha * (
        float(args.expert_pde_sigma_threshold_target) - float(args.expert_pde_sigma_threshold)
    )


def compute_targeted_expert_pde_loss(adapter, pde_residuals, metadata, phenomenon_labels, sigma, args, step):
    cache = getattr(adapter, "_cache", {})
    shared = cache.get("shared_attribute_bank_live")
    branch_updates = cache.get("branch_attribute_updates_live")
    branch_indices = cache.get("active_expert_indices")
    branch_weights = cache.get("active_expert_weights_live")
    if shared is None or branch_updates is None or branch_indices is None or branch_weights is None:
        return compute_pde_loss(adapter, pde_residuals, metadata)
    sigma_value = float(sigma.detach().float().mean().item()) if torch.is_tensor(sigma) else float(sigma)
    threshold = effective_sigma_threshold(args, step)
    if sigma_value > threshold:
        return shared.new_zeros(()), {
            "physics_mode": "explicit_attribute_bank_v2_expert_disabled",
            "sigma_threshold_effective": threshold,
            "sigma_threshold_pass_samples": 0.0,
        }
    active_names = active_label_names(metadata, phenomenon_labels)
    active_ids = {phenomenon_labels.index(name) for name in active_names if name in phenomenon_labels}
    total = shared.new_zeros(())
    weight_sum = shared.new_zeros(())
    enabled = 0
    for sample_idx in range(shared.shape[0]):
        for slot_idx in range(branch_indices.shape[1]):
            expert_idx = int(branch_indices[sample_idx, slot_idx].detach().item())
            if expert_idx not in active_ids:
                continue
            label_name = phenomenon_labels[expert_idx]
            method = getattr(pde_residuals, PHENOMENON_TO_RESIDUAL_METHOD[label_name])
            candidate_bank = shared[sample_idx:sample_idx + 1] + branch_updates[sample_idx:sample_idx + 1, slot_idx]
            loss_i, _ = method(candidate_bank.float().clamp(-10.0, 10.0), metadata=metadata)
            weight_i = branch_weights[sample_idx, slot_idx].to(device=shared.device, dtype=shared.dtype)
            total = total + weight_i * torch.clamp(loss_i.float(), 0.0, 100.0).to(dtype=shared.dtype)
            weight_sum = weight_sum + weight_i
            enabled += 1
    if enabled == 0:
        return shared.new_zeros(()), {
            "physics_mode": "explicit_attribute_bank_v2_expert_disabled",
            "sigma_threshold_effective": threshold,
            "physics_enabled_samples": 0.0,
        }
    return total / weight_sum.clamp_min(1e-6), {
        "physics_mode": "explicit_attribute_bank_v2_expert",
        "sigma_threshold_effective": threshold,
        "physics_enabled_samples": float(enabled),
        "target_expert_count": float(enabled),
    }


def encoder_completion_loss(adapter, pde_residuals, state_for_physics, sigma, metadata, args, step):
    proxy_targets = build_observable_proxy_targets(state_for_physics)
    stage_outputs = adapter.forward_observable_pretrain(state_for_physics, sigma=sigma)
    loss_obs_raw, obs_errors = observable_alignment_terms(
        stage_outputs["observable_outputs"],
        proxy_targets,
        proxy_targets["proxy_conf"],
        args.observable_target_mode,
    )
    cache = getattr(adapter, "_cache", {})
    shared_bank = cache["shared_attribute_bank_live"]
    physics_feat = cache["physics_feat_live"]
    phase = active_recovery_phase(args, step)
    field_dict, final_bank, field_metrics = adapter.build_physical_field_dict(
        shared_bank,
        physics_feat,
        metadata=metadata,
        field_recovery_phase=phase,
    )
    cache["fused_attribute_bank_live"] = final_bank
    cache["fused_attribute_bank"] = final_bank.detach()
    pde = pde_residuals
    only_u_terms = pde._only_u_fluid_terms(field_dict["u"], field_dict["p"], field_dict["rho"], metadata=metadata)
    local = (
        only_u_terms["mass_residual"]
        + only_u_terms["momentum_residual"]
        + 0.25 * only_u_terms["pressure_smoothness"]
        + 0.25 * (only_u_terms["density_smoothness"] + only_u_terms["density_floor"])
    )
    if "d_phys" in field_dict:
        local = local + 0.5 * pde._temporal_alignment_loss(field_dict["d_phys"], field_dict["u"], metadata=metadata)
    if phase_at_least(args, step, "alpha"):
        alpha = field_dict["alpha_scalar"]
        local = local + encoder_completion_loss_weight(args, step, "alpha") * (
            0.1 * pde._spatial_gradient_energy(alpha, metadata=metadata)
            + 0.1 * pde._weighted_square_mean(torch.relu(-alpha) + torch.relu(alpha - 1.0), metadata=metadata, ref_tensor=alpha)
        )
    if phase_at_least(args, step, "T"):
        temp = field_dict["T_scalar"]
        local = local + encoder_completion_loss_weight(args, step, "T") * (
            0.1 * pde._spatial_gradient_energy(temp, metadata=metadata)
        )
    total = local + 0.1 * loss_obs_raw
    metrics = {
        "loss_obs": float(loss_obs_raw.detach().item()),
        "obs_flow_error": float(obs_errors["flow_error"].detach().item()),
        "obs_deformation_error": float(obs_errors["deformation_error"].detach().item()),
        "encoder_completion_local_loss": float(local.detach().item()),
        "active_field_recovery_phase": phase,
    }
    for key, value in field_metrics.items():
        if torch.is_tensor(value):
            metrics[key] = float(value.detach().float().mean().item())
    return total, metrics


def make_adapter(args, contracts, operators, adapter_mod):
    pde_residuals = operators.MaterialPDEResiduals(
        num_phenomena=len(contracts.PHENOMENON_LABELS),
        q_input_dim=64,
        n_numeric_dim=12,
        strict_metadata_contract=True,
    )
    pde_residuals.eval()
    pde_residuals.requires_grad_(False)
    adapter = adapter_mod.PhysicsAdapter(
        latent_dim=16,
        hidden_dim=args.adapter_hidden_dim,
        physics_attr_dim=contracts.PHYSICS_ATTR_DIM,
        num_phenomena=len(contracts.PHENOMENON_LABELS),
        moe_top_k=args.moe_top_k,
        pde_residuals=pde_residuals,
        physics_state_mode="x0_hat",
        use_sigma_gate=True,
        sigma_gate_curve=args.sigma_gate_curve,
        use_sigma_conditioning=True,
        sigma_gate_floor=args.sigma_gate_floor,
        strict_physical_state_contract=True,
        core_ablation_mode=args.core_ablation_mode,
    )
    adapter.set_ablation_modes(
        use_moe=(not args.ablate_disable_moe and args.core_ablation_mode != "generic_latent_correction"),
        label_only_mode=(args.ablate_label_only_router or args.core_ablation_mode == "wo_learned_expert_routing"),
    )
    adapter.secondary_field_strategy = (
        "direct_bank"
        if args.secondary_field_strategy == "legacy_direct_bank"
        else args.secondary_field_strategy
    )
    if hasattr(pde_residuals, "set_conditioning_enabled"):
        pde_residuals.set_conditioning_enabled(
            enabled=(
                not args.ablate_disable_conditioned_pde
                and args.core_ablation_mode not in {
                    "generic_latent_correction",
                    "wo_explicit_physical_interface",
                    "wo_pde_residuals",
                }
            )
        )
    return adapter, pde_residuals


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pila_root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--anyflow_root", default=DEFAULT_ANYFLOW_ROOT)
    parser.add_argument("--anyflow_model_path", default=DEFAULT_ANYFLOW_MODEL)
    parser.add_argument("--dataset_base_path", default=DEFAULT_DATASET_BASE)
    parser.add_argument("--dataset_metadata_path", default=DEFAULT_DATASET_META)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--output_path", default="./models/train/anyflow_wan21_1p3b_pinn")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--pinn_checkpoint", default="")
    parser.add_argument("--stage1_pretrained_encoder", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter_hidden_dim", type=int, default=128)
    parser.add_argument("--moe_top_k", type=int, default=4)
    parser.add_argument("--sigma_gate_curve", default="linear")
    parser.add_argument("--sigma_gate_floor", type=float, default=0.30)
    parser.add_argument("--core_ablation_mode", default="full")
    parser.add_argument("--training_stage", default="full_pinn", choices=["observable_pretrain", "encoder_completion", "full_pinn"])
    parser.add_argument("--observable_target_mode", default="flow_plus_deformation", choices=["flow_plus_deformation", "flow_only"])
    parser.add_argument(
        "--secondary_field_strategy",
        default="direct_bank",
        choices=["direct_bank", "u_first_constructor", "u_first_constructor_detach", "legacy_direct_bank"],
    )
    parser.add_argument("--field_recovery_phase", default="core", choices=list(FIELD_RECOVERY_PHASES))
    parser.add_argument("--field_recovery_step_schedule", default="")
    parser.add_argument("--field_recovery_loss_ramp_steps", type=int, default=100)
    parser.add_argument("--freeze_u_encoder_during_recovery", action="store_true", default=True)
    parser.add_argument("--no_freeze_u_encoder_during_recovery", dest="freeze_u_encoder_during_recovery", action="store_false")
    parser.add_argument("--encoder_freeze_steps", type=int, default=1000)
    parser.add_argument("--ablate_disable_moe", action="store_true")
    parser.add_argument("--ablate_disable_conditioned_pde", action="store_true")
    parser.add_argument("--ablate_disable_aux_losses", action="store_true")
    parser.add_argument("--ablate_label_only_router", action="store_true")
    parser.add_argument("--physics_weight", type=float, default=0.30)
    parser.add_argument("--physics_weight_target", type=float, default=None)
    parser.add_argument("--physics_warmup_steps", type=int, default=2000)
    parser.add_argument("--expert_pde_sigma_threshold", type=float, default=0.40)
    parser.add_argument("--expert_pde_sigma_threshold_target", type=float, default=1.00)
    parser.add_argument("--correction_weight", type=float, default=0.01)
    parser.add_argument("--flowmap_shift", type=float, default=5.0)
    parser.add_argument("--diffusion_ratio", type=float, default=0.50)
    parser.add_argument("--consistency_ratio", type=float, default=0.25)
    parser.add_argument("--min_sigma", type=float, default=0.02)
    parser.add_argument("--max_sigma", type=float, default=0.98)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Use --batch_size 1 and gradient accumulation for this script.")
    local_rank, world_size = setup_distributed()
    rank = get_rank()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("AnyFlow 1.3B adapter training requires CUDA.")

    sys.path.insert(0, args.anyflow_root)
    from far.pipelines.pipeline_wan_anyflow import WanAnyFlowPipeline

    contracts, operators, adapter_mod = load_pila_modules(args.pila_root)
    adapter, pde_residuals = make_adapter(args, contracts, operators, adapter_mod)
    checkpoint_path = args.pinn_checkpoint or args.resume_checkpoint
    if checkpoint_path:
        result = load_adapter_checkpoint(checkpoint_path, adapter)
        if is_main_process():
            print(f"[resume] {checkpoint_path}: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    if args.training_stage in {"encoder_completion", "full_pinn"} and args.stage1_pretrained_encoder and not checkpoint_path:
        loaded = load_stage1_encoder_checkpoint(args.stage1_pretrained_encoder, adapter)
        if is_main_process():
            print(f"[stage1] loaded encoder scaffold from {args.stage1_pretrained_encoder}: {', '.join(loaded)}")

    adapter.to(device=device, dtype=torch.float32)
    pde_residuals.to(device=device, dtype=torch.float32)
    configure_training_stage(adapter, args, current_step=0)

    if is_main_process():
        print(f"[distributed] world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print(f"[load] AnyFlow pipeline: {args.anyflow_model_path}")
    pipe = WanAnyFlowPipeline.from_pretrained(args.anyflow_model_path, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.transformer.eval().requires_grad_(False)
    pipe.text_encoder.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)
    pipe.scheduler.config.shift = args.flowmap_shift

    dataset = CsvVideoDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        phenomenon_labels=contracts.PHENOMENON_LABELS,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        max_samples=args.max_samples,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    ) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    if not any(param.requires_grad for param in adapter.parameters()):
        raise RuntimeError(f"No trainable PhysicsAdapter parameters for stage={args.training_stage}")
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        with open(output_path / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    global_step = 0
    running = {}
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, desc="anyflow-pinn") if is_main_process() else None
    epoch = 0

    while global_step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if global_step >= args.max_steps:
                break
            configure_training_stage(adapter, args, current_step=global_step)
            videos = batch["pixel_values"].to(device=device, dtype=torch.float32, non_blocking=True)
            metadata = metadata_to_device(batch["metadata"], device)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                videos_for_vae = videos * 2.0 - 1.0
                latents = pipe.encode_latents(videos_for_vae.to(dtype=torch.bfloat16), sample=False)
                latents = rearrange(latents, "b c t h w -> b t c h w").contiguous()
                prompt_embeds, _ = pipe.encode_prompt(
                    prompt=batch["prompts"],
                    negative_prompt=None,
                    do_classifier_free_guidance=False,
                    max_sequence_length=512,
                    device=device,
                    dtype=torch.bfloat16,
                )
                batch_size, latent_frames = latents.shape[0], latents.shape[1]
                noise = torch.randn_like(latents)
                sigma_a = torch.rand(batch_size, device=device, dtype=torch.float32)
                sigma_b = torch.rand(batch_size, device=device, dtype=torch.float32)
                t_sigma = torch.maximum(sigma_a, sigma_b).clamp(args.min_sigma, args.max_sigma)
                r_sigma = torch.minimum(sigma_a, sigma_b).clamp(0.0, args.max_sigma)
                selector = random.random()
                if selector < args.diffusion_ratio:
                    r_sigma = t_sigma
                elif selector < args.diffusion_ratio + args.consistency_ratio:
                    r_sigma = torch.zeros_like(t_sigma)
                t = pipe.scheduler.apply_shift(t_sigma).view(batch_size, 1).repeat(1, latent_frames)
                r = pipe.scheduler.apply_shift(r_sigma).view(batch_size, 1).repeat(1, latent_frames)
                t_steps = (t * pipe.scheduler.config.num_train_timesteps).to(device=device)
                r_steps = (r * pipe.scheduler.config.num_train_timesteps).to(device=device)
                noisy_latents = pipe.scheduler.scale_noise(latents, t_steps, noise)
                target_velocity = noise - latents
                velocity = pipe.transformer(
                    noisy_latents,
                    timestep=t_steps,
                    r_timestep=r_steps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                    is_causal=False,
                )[0]

            sigma_for_adapter = t_sigma.detach().float()
            velocity_bcthw = rearrange(velocity, "b t c h w -> b c t h w").float()
            target_bcthw = rearrange(target_velocity, "b t c h w -> b c t h w").float()
            noisy_bcthw = rearrange(noisy_latents, "b t c h w -> b c t h w").float()
            sigma_map = sigma_for_adapter.view(batch_size, 1, 1, 1, 1)
            state_for_physics = noisy_bcthw - sigma_map * velocity_bcthw

            metadata.update(
                video_time_metadata(
                    batch_size=batch_size,
                    num_frames=state_for_physics.shape[2],
                    device=device,
                    dtype=state_for_physics.dtype,
                )
            )

            if args.training_stage == "observable_pretrain":
                proxy_targets = build_observable_proxy_targets(state_for_physics)
                stage_outputs = adapter.forward_observable_pretrain(
                    state_for_physics,
                    sigma=sigma_for_adapter,
                )
                loss, obs_errors = observable_alignment_terms(
                    stage_outputs["observable_outputs"],
                    proxy_targets,
                    proxy_targets["proxy_conf"],
                    args.observable_target_mode,
                )
                pde_loss = state_for_physics.new_zeros(())
                correction_loss = obs_errors["flow_error"] + obs_errors["deformation_error"]
                fm_adapter_loss = state_for_physics.new_zeros(())
                stage_metrics = {
                    "loss_obs": float(loss.detach().cpu()),
                    "obs_flow_error": float(obs_errors["flow_error"].detach().cpu()),
                    "obs_deformation_error": float(obs_errors["deformation_error"].detach().cpu()),
                    "proxy_conf": float(proxy_targets["proxy_conf"].detach().mean().cpu()),
                }
            elif args.training_stage == "encoder_completion":
                loss, stage_metrics = encoder_completion_loss(
                    adapter,
                    pde_residuals,
                    state_for_physics,
                    sigma_for_adapter,
                    metadata,
                    args,
                    global_step,
                )
                pde_loss = torch.as_tensor(stage_metrics["encoder_completion_local_loss"], device=device, dtype=state_for_physics.dtype)
                correction_loss = state_for_physics.new_zeros(())
                fm_adapter_loss = state_for_physics.new_zeros(())
            else:
                if args.ablate_disable_moe:
                    adapter_metadata = None
                elif args.ablate_label_only_router:
                    adapter_metadata = label_only_metadata(metadata)
                else:
                    adapter_metadata = metadata
                corrected_velocity = adapter(
                    velocity_bcthw,
                    state_for_physics,
                    sigma=sigma_for_adapter,
                    metadata=adapter_metadata,
                )
                pde_loss, pde_info = compute_targeted_expert_pde_loss(
                    adapter,
                    pde_residuals,
                    metadata,
                    contracts.PHENOMENON_LABELS,
                    sigma_for_adapter,
                    args,
                    global_step,
                )
                correction_loss = torch.mean((corrected_velocity - velocity_bcthw) ** 2).clamp(0.0, 100.0)
                fm_adapter_loss = F.mse_loss(corrected_velocity.float(), target_bcthw.float()).clamp(0.0, 100.0)
                physics_weight = effective_physics_weight(args, global_step)
                loss = fm_adapter_loss + physics_weight * pde_loss + args.correction_weight * correction_loss
                stage_metrics = {
                    "loss_fm_adapter": float(fm_adapter_loss.detach().cpu()),
                    "physics_weight_effective": physics_weight,
                    "sigma_threshold_effective": effective_sigma_threshold(args, global_step),
                    **{
                        f"physics_{k}": v
                        for k, v in pde_info.items()
                        if isinstance(v, (int, float, str))
                    },
                }
            loss = loss / max(1, args.gradient_accumulation_steps)
            loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                average_gradients(adapter)
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            raw = {
                "loss": float(loss.detach().cpu()) * max(1, args.gradient_accumulation_steps),
                "pde_loss": float(pde_loss.detach().cpu()),
                "correction_loss": float(correction_loss.detach().cpu()),
                "fm_adapter_loss": float(fm_adapter_loss.detach().cpu()),
                "sigma": float(t_sigma.detach().cpu()[0]),
                "label": metadata.get("label_name", ""),
                "stage": args.training_stage,
            }
            raw.update(stage_metrics)
            for key, value in raw.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    running[key] = running.get(key, 0.95 * float(value)) * 0.95 + 0.05 * float(value)

            global_step += 1
            if progress is not None:
                progress.update(1)
            if is_main_process() and global_step % args.log_steps == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                msg = {
                    "step": global_step,
                    "loss_ema": running.get("loss", raw["loss"]),
                    "pde_ema": running.get("pde_loss", raw["pde_loss"]),
                    "corr_ema": running.get("correction_loss", raw["correction_loss"]),
                    "fm_ema": running.get("fm_adapter_loss", raw["fm_adapter_loss"]),
                    "label": raw["label"],
                    "stage": args.training_stage,
                    "samples_per_sec": global_step / elapsed,
                }
                print("[train] " + json.dumps(msg, ensure_ascii=False), flush=True)
            if is_main_process() and global_step % args.save_steps == 0:
                save_checkpoint(
                    output_path / f"step-{global_step}.pt",
                    adapter,
                    optimizer,
                    global_step,
                    args,
                    contracts,
                )
                with open(output_path / f"step-{global_step}.metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata_to_jsonable(metadata), f, ensure_ascii=False, indent=2)
        epoch += 1

    save_checkpoint(output_path / "final.pt", adapter, optimizer, global_step, args, contracts)
    if progress is not None:
        progress.close()
    if is_main_process():
        print(f"[done] saved final checkpoint to {output_path / 'final.pt'}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
