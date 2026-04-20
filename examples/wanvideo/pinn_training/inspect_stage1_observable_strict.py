#!/usr/bin/env python3
import argparse
import importlib.machinery
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PINN_PATH = REPO_ROOT / "examples" / "wanvideo" / "pinn_training" / "train_pinn.py"
DEFAULT_MODEL_ID_WITH_ORIGIN_PATHS = (
    "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,"
    "Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,"
    "Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth"
)


def register_checkpoint_stubs():
    modules = {}
    for name in (
        "deepspeed",
        "deepspeed.runtime",
        "deepspeed.runtime.fp16",
        "deepspeed.runtime.fp16.loss_scaler",
        "deepspeed.runtime.zero",
        "deepspeed.runtime.zero.config",
        "deepspeed.utils",
        "deepspeed.utils.tensor_fragment",
    ):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = module
        modules[name] = module

    class LossScaler:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class ZeroStageEnum:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class fragment_address:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    modules["deepspeed.runtime.fp16.loss_scaler"].LossScaler = LossScaler
    modules["deepspeed.runtime.zero.config"].ZeroStageEnum = ZeroStageEnum
    modules["deepspeed.utils.tensor_fragment"].fragment_address = fragment_address


def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_train_module():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return load_module_from_path("train_pinn_strict_local", TRAIN_PINN_PATH)


def load_checkpoint(checkpoint_path):
    register_checkpoint_stubs()
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def to_bool(value, default):
    if value is None:
        return bool(default)
    return bool(value)


def make_safe_sample_name(value, fallback):
    text = str(value) if value is not None else ""
    text = os.path.basename(text)
    text = os.path.splitext(text)[0]
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    if not text:
        text = fallback
    return text[:96]


def motion_score(video_frames):
    array = np.stack([np.asarray(frame, dtype=np.float32) for frame in video_frames], axis=0)
    if array.shape[0] <= 1:
        return 0.0
    diff = np.abs(array[1:] - array[:-1]).mean(axis=(1, 2, 3))
    return float(diff.mean())


def flow_to_rgb(flow, max_magnitude):
    flow = np.asarray(flow, dtype=np.float32)
    flow_x = flow[0]
    flow_y = flow[1]
    magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)
    angle = np.arctan2(flow_y, flow_x)
    h = (angle + np.pi) / (2 * np.pi)
    s = np.ones_like(h, dtype=np.float32)
    v = np.clip(magnitude / max(max_magnitude, 1e-6), 0.0, 1.0)

    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).clip(0, 255).astype(np.uint8)


def error_to_rgb(error_map, max_error):
    error_map = np.clip(error_map / max(max_error, 1e-6), 0.0, 1.0)
    r = (255.0 * error_map).astype(np.uint8)
    g = (255.0 * np.sqrt(error_map)).astype(np.uint8)
    b = (255.0 * (1.0 - error_map)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def scalar_to_rgb(field_map, scale=None, positive_only=False):
    field_map = np.asarray(field_map, dtype=np.float32)
    if positive_only:
        resolved_scale = float(scale) if scale is not None else float(np.percentile(field_map, 95))
        resolved_scale = max(resolved_scale, 1e-6)
        normalized = np.clip(field_map / resolved_scale, 0.0, 1.0)
        r = (255.0 * normalized).astype(np.uint8)
        g = (255.0 * np.sqrt(normalized)).astype(np.uint8)
        b = (255.0 * (1.0 - normalized)).astype(np.uint8)
        return np.stack([r, g, b], axis=-1)

    resolved_scale = float(scale) if scale is not None else float(np.percentile(np.abs(field_map), 95))
    resolved_scale = max(resolved_scale, 1e-6)
    normalized = np.clip(field_map / resolved_scale, -1.0, 1.0)
    pos = np.clip(normalized, 0.0, 1.0)
    neg = np.clip(-normalized, 0.0, 1.0)
    r = (255.0 * (1.0 - neg)).astype(np.uint8)
    g = (255.0 * (1.0 - np.maximum(pos, neg))).astype(np.uint8)
    b = (255.0 * (1.0 - pos)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def to_display_frame(frame_tensor):
    frame = frame_tensor.detach().cpu().float()
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = frame.permute(1, 2, 0)
    frame = ((frame + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
    return frame


def resize_rgb(rgb, size):
    image = Image.fromarray(rgb)
    image = image.resize(size, Image.BILINEAR)
    return image


def add_label(image, text):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 18), fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return image


def make_contact_sheet(
    sample_name,
    sample_prompt,
    raw_video,
    x0hat_video,
    raw_flow,
    target_flow,
    pred_flow,
    error_map,
    out_path,
):
    time_steps = target_flow.shape[1]
    columns = np.linspace(0, max(time_steps - 1, 0), 6, dtype=int) if time_steps > 0 else np.array([0] * 6)
    tile_size = (192, 108)
    left_margin = 132
    top_margin = 54
    row_gap = 12
    col_gap = 8
    row_labels = ["Raw Frame", "x0_hat Frame", "Raw Proxy", "Target Proxy", "Pred Flow", "Error"]
    canvas_w = left_margin + len(columns) * tile_size[0] + (len(columns) - 1) * col_gap
    canvas_h = top_margin + len(row_labels) * tile_size[1] + (len(row_labels) - 1) * row_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 22, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), sample_name, fill=(255, 255, 255))
    draw.text((16, 30), sample_prompt[:120], fill=(180, 180, 180))

    max_magnitude = np.percentile(
        np.concatenate(
            [
                np.sqrt((raw_flow ** 2).sum(axis=0)).reshape(-1),
                np.sqrt((target_flow ** 2).sum(axis=0)).reshape(-1),
                np.sqrt((pred_flow ** 2).sum(axis=0)).reshape(-1),
            ]
        ),
        95,
    )
    max_error = np.percentile(error_map.reshape(-1), 95)

    raw_frames_count = raw_video.shape[1]
    x0hat_frames_count = x0hat_video.shape[2]
    for col, t_idx in enumerate(columns):
        raw_t = int(round(t_idx / max(time_steps - 1, 1) * max(raw_frames_count - 1, 0)))
        x0hat_t = int(round(t_idx / max(time_steps - 1, 1) * max(x0hat_frames_count - 1, 0)))
        tiles = [
            to_display_frame(raw_video[0, :, raw_t]),
            to_display_frame(x0hat_video[0, :, x0hat_t]),
            flow_to_rgb(raw_flow[:, t_idx], max_magnitude),
            flow_to_rgb(target_flow[:, t_idx], max_magnitude),
            flow_to_rgb(pred_flow[:, t_idx], max_magnitude),
            error_to_rgb(error_map[t_idx], max_error),
        ]
        x = left_margin + col * (tile_size[0] + col_gap)
        draw.text((x, 30), f"t={t_idx:02d}", fill=(255, 255, 255))
        for row, tile in enumerate(tiles):
            y = top_margin + row * (tile_size[1] + row_gap)
            panel = add_label(resize_rgb(tile, tile_size), "")
            canvas.paste(panel, (x, y))

    for row, label in enumerate(row_labels):
        y = top_margin + row * (tile_size[1] + row_gap) + tile_size[1] // 2 - 8
        draw.text((16, y), label, fill=(255, 255, 255))

    canvas.save(out_path)


def make_video(
    sample_name,
    raw_video,
    x0hat_video,
    raw_flow,
    target_flow,
    pred_flow,
    error_map,
    out_path,
    fps=6,
):
    max_magnitude = np.percentile(
        np.concatenate(
            [
                np.sqrt((raw_flow ** 2).sum(axis=0)).reshape(-1),
                np.sqrt((target_flow ** 2).sum(axis=0)).reshape(-1),
                np.sqrt((pred_flow ** 2).sum(axis=0)).reshape(-1),
            ]
        ),
        95,
    )
    max_error = np.percentile(error_map.reshape(-1), 95)
    frames = []
    raw_frames_count = raw_video.shape[1]
    time_steps = target_flow.shape[1]
    x0hat_frames_count = x0hat_video.shape[2]
    for t_idx in range(time_steps):
        raw_t = int(round(t_idx / max(time_steps - 1, 1) * max(raw_frames_count - 1, 0)))
        x0hat_t = int(round(t_idx / max(time_steps - 1, 1) * max(x0hat_frames_count - 1, 0)))
        panels = [
            add_label(resize_rgb(to_display_frame(raw_video[0, :, raw_t]), (224, 126)), "Raw Frame"),
            add_label(resize_rgb(to_display_frame(x0hat_video[0, :, x0hat_t]), (224, 126)), "x0_hat Frame"),
            add_label(resize_rgb(flow_to_rgb(raw_flow[:, t_idx], max_magnitude), (224, 126)), "Raw Proxy"),
            add_label(resize_rgb(flow_to_rgb(target_flow[:, t_idx], max_magnitude), (224, 126)), "Target Proxy"),
            add_label(resize_rgb(flow_to_rgb(pred_flow[:, t_idx], max_magnitude), (224, 126)), "Pred Flow"),
            add_label(resize_rgb(error_to_rgb(error_map[t_idx], max_error), (224, 126)), "Error"),
        ]
        canvas = Image.new("RGB", (len(panels) * 224, 126 + 24), color=(14, 14, 16))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), f"{sample_name} | t={t_idx:02d}", fill=(255, 255, 255))
        for i, panel in enumerate(panels):
            canvas.paste(panel, (i * 224, 24))
        frames.append(np.asarray(canvas))

    try:
        with imageio.get_writer(
            out_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        return str(out_path)
    except Exception as exc:
        fallback_dir = out_path.with_suffix("")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, frame in enumerate(frames):
            imageio.imwrite(fallback_dir / f"{frame_idx:04d}.png", frame)
        print(
            f"Warning: failed to save mp4 to {out_path} ({exc}). "
            f"Saved PNG sequence to {fallback_dir} instead."
        )
        return str(fallback_dir)


def make_field_contact_sheet(sample_name, sample_prompt, field_rows, out_path):
    if not field_rows:
        return None

    time_steps = max(len(images) for _, images in field_rows)
    columns = np.linspace(0, max(time_steps - 1, 0), 6, dtype=int) if time_steps > 0 else np.array([0] * 6)
    tile_size = (192, 108)
    left_margin = 132
    top_margin = 54
    row_gap = 12
    col_gap = 8
    canvas_w = left_margin + len(columns) * tile_size[0] + (len(columns) - 1) * col_gap
    canvas_h = top_margin + len(field_rows) * tile_size[1] + (len(field_rows) - 1) * row_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 22, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), sample_name, fill=(255, 255, 255))
    draw.text((16, 30), sample_prompt[:120], fill=(180, 180, 180))

    for col, t_idx in enumerate(columns):
        x = left_margin + col * (tile_size[0] + col_gap)
        draw.text((x, 30), f"t={t_idx:02d}", fill=(255, 255, 255))
        for row, (label, frames) in enumerate(field_rows):
            y = top_margin + row * (tile_size[1] + row_gap)
            frame_idx = min(int(t_idx), max(len(frames) - 1, 0))
            panel = add_label(resize_rgb(frames[frame_idx], tile_size), "")
            canvas.paste(panel, (x, y))
            if col == 0:
                draw.text((16, y + tile_size[1] // 2 - 8), label, fill=(255, 255, 255))

    canvas.save(out_path)
    return str(out_path)


def choose_samples(dataset, scan_count, num_samples, explicit_indices):
    if explicit_indices:
        return explicit_indices
    candidates = []
    total = min(scan_count, len(dataset.data))
    for idx in range(total):
        sample = dataset[idx]
        candidates.append((motion_score(sample["video"]), idx))
    candidates.sort(reverse=True)
    return [idx for _, idx in candidates[:num_samples]]


def build_dataset(args, train_module):
    namespace = types.SimpleNamespace(
        dataset_base_path=args.dataset_base_path,
        dataset_metadata_path=args.dataset_metadata_path,
        height=args.height,
        width=args.width,
        max_pixels=args.height * args.width,
        num_frames=args.num_frames,
        data_file_keys="video",
        dataset_repeat=1,
    )
    return train_module.VideoDataset(args=namespace)


def resolve_boundaries(args, checkpoint_config):
    min_boundary = (
        float(args.min_timestep_boundary)
        if args.min_timestep_boundary is not None
        else float(checkpoint_config.get("min_timestep_boundary", 0.0))
    )
    max_boundary = (
        float(args.max_timestep_boundary)
        if args.max_timestep_boundary is not None
        else float(checkpoint_config.get("max_timestep_boundary", 1.0))
    )
    min_boundary = min(max(min_boundary, 0.0), 1.0)
    max_boundary = min(max(max_boundary, 0.0), 1.0)
    if max_boundary < min_boundary:
        min_boundary, max_boundary = max_boundary, min_boundary
    return min_boundary, max_boundary


def summarize_array(array):
    array = np.asarray(array, dtype=np.float32)
    abs_array = np.abs(array)
    return {
        "shape": list(array.shape),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "abs_mean": float(abs_array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p95_abs": float(np.percentile(abs_array, 95)),
    }


def infer_recovery_phase(model, checkpoint_config):
    if hasattr(model, "_active_field_recovery_phase"):
        try:
            return str(model._active_field_recovery_phase())
        except Exception:
            pass
    return str(checkpoint_config.get("field_recovery_phase", "core"))


def build_physical_metadata(model, reference):
    if hasattr(model, "_build_video_time_metadata"):
        return model._build_video_time_metadata(
            batch_size=reference.shape[0],
            num_frames=reference.shape[2],
            device=reference.device,
            dtype=reference.dtype,
        )
    frame_count = max(int(reference.shape[2]), 1)
    frame_delta_t = 1.0 / max(frame_count - 1, 1)
    frame_time_grid = torch.linspace(
        0.0,
        1.0,
        steps=frame_count,
        device=reference.device,
        dtype=reference.dtype,
    )
    if frame_count <= 1:
        frame_time_grid = torch.zeros(1, device=reference.device, dtype=reference.dtype)
    return {
        "frame_count": torch.full((reference.shape[0],), float(frame_count), device=reference.device, dtype=reference.dtype),
        "frame_delta_t": torch.full((reference.shape[0],), float(frame_delta_t), device=reference.device, dtype=reference.dtype),
        "frame_time_grid": frame_time_grid.unsqueeze(0).repeat(reference.shape[0], 1),
        "physics_time_source": "video_frames",
    }


def build_physical_visual_rows(physical_field_dict):
    u = physical_field_dict["u"][0].detach().cpu().float().numpy()
    p_scalar = physical_field_dict["p_scalar"][0, 0].detach().cpu().float().numpy()
    rho_scalar = physical_field_dict["rho_scalar"][0, 0].detach().cpu().float().numpy()
    d_phys = physical_field_dict["d_phys"][0].detach().cpu().float().numpy()
    div_u = physical_field_dict["div_u"][0, 0].detach().cpu().float().numpy()
    curl_u = physical_field_dict["curl_u"][0, 0].detach().cpu().float().numpy()
    eps = physical_field_dict["eps"][0].detach().cpu().float().numpy()
    sigma = physical_field_dict["sigma"][0].detach().cpu().float().numpy()

    u_mag = np.sqrt((u ** 2).sum(axis=0))
    d_mag = np.sqrt((d_phys ** 2).sum(axis=0))
    eps_norm = np.sqrt((eps ** 2).sum(axis=0))
    sigma_norm = np.sqrt((sigma ** 2).sum(axis=0))

    u_scale = float(np.percentile(u_mag.reshape(-1), 95))
    d_scale = float(np.percentile(d_mag.reshape(-1), 95))
    p_scale = float(np.percentile(np.abs(p_scalar).reshape(-1), 95))
    rho_scale = float(np.percentile(rho_scalar.reshape(-1), 95))
    div_scale = float(np.percentile(np.abs(div_u).reshape(-1), 95))
    curl_scale = float(np.percentile(np.abs(curl_u).reshape(-1), 95))
    eps_scale = float(np.percentile(eps_norm.reshape(-1), 95))
    sigma_scale = float(np.percentile(sigma_norm.reshape(-1), 95))

    return [
        ("u", [flow_to_rgb(u[:, t_idx], u_scale) for t_idx in range(u.shape[1])]),
        ("|u|", [scalar_to_rgb(u_mag[t_idx], u_scale, positive_only=True) for t_idx in range(u_mag.shape[0])]),
        ("p", [scalar_to_rgb(p_scalar[t_idx], p_scale, positive_only=False) for t_idx in range(p_scalar.shape[0])]),
        ("rho", [scalar_to_rgb(rho_scalar[t_idx], rho_scale, positive_only=True) for t_idx in range(rho_scalar.shape[0])]),
        ("d", [flow_to_rgb(d_phys[:, t_idx], d_scale) for t_idx in range(d_phys.shape[1])]),
        ("div_u", [scalar_to_rgb(div_u[t_idx], div_scale, positive_only=False) for t_idx in range(div_u.shape[0])]),
        ("curl_u", [scalar_to_rgb(curl_u[t_idx], curl_scale, positive_only=False) for t_idx in range(curl_u.shape[0])]),
        ("||eps||", [scalar_to_rgb(eps_norm[t_idx], eps_scale, positive_only=True) for t_idx in range(eps_norm.shape[0])]),
        ("||sigma||", [scalar_to_rgb(sigma_norm[t_idx], sigma_scale, positive_only=True) for t_idx in range(sigma_norm.shape[0])]),
    ]


def summarize_physical_fields(physical_field_dict, active_phase, physical_metrics):
    summaries = {}
    for field_name in (
        "u",
        "p_scalar",
        "rho_scalar",
        "d_phys",
        "div_u",
        "curl_u",
        "eps",
        "sigma",
        "alpha_scalar",
        "T_scalar",
        "j_phys",
        "D_scalar",
        "psi_scalar",
    ):
        summaries[field_name] = summarize_array(
            physical_field_dict[field_name][0].detach().cpu().float().numpy()
        )

    eps = physical_field_dict["eps"].detach().cpu().float()
    future_zero_fields = {}
    for field_name in ("alpha_scalar", "T_scalar", "j_phys", "D_scalar", "psi_scalar"):
        field_value = physical_field_dict[field_name].detach().cpu().float()
        future_zero_fields[field_name] = float(field_value.abs().mean().item())

    metrics = {}
    for key, value in physical_metrics.items():
        if isinstance(value, torch.Tensor):
            metrics[key] = float(value.detach().cpu().float().item())
        else:
            metrics[key] = float(value)

    return {
        "active_field_recovery_phase": str(active_phase),
        "field_stats": summaries,
        "physical_metrics": metrics,
        "eps_symmetry_error": float((eps[:, 1:2] - eps[:, 2:3]).abs().mean().item()),
        "future_zero_field_abs_mean": future_zero_fields,
    }


def build_runtime_kwargs(args, checkpoint_config):
    return {
        "model_paths": args.model_paths,
        "model_id_with_origin_paths": args.model_id_with_origin_paths,
        "use_gradient_checkpointing": False,
        "use_gradient_checkpointing_offload": False,
        "frozen_model_gradient_checkpointing": False,
        "max_timestep_boundary": resolve_boundaries(args, checkpoint_config)[1],
        "min_timestep_boundary": resolve_boundaries(args, checkpoint_config)[0],
        "use_dual_noise_experts": to_bool(
            checkpoint_config.get("use_dual_noise_experts"),
            False,
        ),
        "dual_noise_expert_boundary": float(
            checkpoint_config.get("dual_noise_expert_boundary", 0.417)
        ),
        "physics_weight": float(checkpoint_config.get("physics_weight", 0.1)),
        "physics_warmup_steps": int(checkpoint_config.get("physics_warmup_steps", 500)),
        "conditioned_physics_warmup_steps": int(
            checkpoint_config.get("conditioned_physics_warmup_steps", 1000)
        ),
        "adapter_hidden_dim": int(checkpoint_config.get("adapter_hidden_dim", 128)),
        "physics_attr_dim": int(checkpoint_config.get("physics_attr_dim", 32)),
        "expert_pde_sigma_threshold": float(
            checkpoint_config.get("expert_pde_sigma_threshold", 0.40)
        ),
        "expert_pde_sigma_threshold_target": float(
            checkpoint_config.get("expert_pde_sigma_threshold_target", 1.00)
        ),
        "training_stage": str(checkpoint_config.get("training_stage", "observable_pretrain")),
        "ablation_preset": str(checkpoint_config.get("ablation_preset", "legacy_direct_bank")),
        "observable_target_mode": str(checkpoint_config.get("observable_target_mode", "auto")),
        "secondary_field_strategy": str(checkpoint_config.get("secondary_field_strategy", "auto")),
        "active_field_set": str(checkpoint_config.get("active_field_set", "auto")),
        "field_enable_schedule": str(checkpoint_config.get("field_enable_schedule", "auto")),
        "field_recovery_phase": str(checkpoint_config.get("field_recovery_phase", "core")),
        "field_recovery_step_schedule": str(checkpoint_config.get("field_recovery_step_schedule", "")),
        "field_recovery_loss_ramp_steps": int(
            checkpoint_config.get("field_recovery_loss_ramp_steps", 100)
        ),
        "run_full_pinn_after_recovery": to_bool(
            checkpoint_config.get("run_full_pinn_after_recovery"),
            False,
        ),
        "pinn_checkpoint": args.checkpoint,
        "stage1_pretrained_encoder": None,
        "flow_backbone_ckpt": (
            args.flow_backbone_ckpt
            if args.flow_backbone_ckpt is not None
            else checkpoint_config.get("flow_backbone_ckpt")
        ),
        "encoder_freeze_steps": int(checkpoint_config.get("encoder_freeze_steps", 1000)),
        "encoder_lr_scale": float(checkpoint_config.get("encoder_lr_scale", 0.3)),
        "expert_balance_weight": float(checkpoint_config.get("expert_balance_weight", 1e-3)),
        "condition_consistency_weight": float(
            checkpoint_config.get("condition_consistency_weight", 1e-2)
        ),
        "moe_top_k": int(checkpoint_config.get("moe_top_k", 4)),
        "ablate_disable_moe": to_bool(checkpoint_config.get("ablate_disable_moe"), False),
        "ablate_disable_conditioned_pde": to_bool(
            checkpoint_config.get("ablate_disable_conditioned_pde"),
            False,
        ),
        "ablate_disable_aux_losses": to_bool(
            checkpoint_config.get("ablate_disable_aux_losses"),
            False,
        ),
        "ablate_label_only_router": to_bool(
            checkpoint_config.get("ablate_label_only_router"),
            False,
        ),
        "diagnostic_metrics_interval": int(
            checkpoint_config.get("diagnostic_metrics_interval", 100)
        ),
        "motion_mask_floor": float(checkpoint_config.get("motion_mask_floor", 0.08)),
        "motion_mask_quantile": float(checkpoint_config.get("motion_mask_quantile", 0.9)),
        "motion_mask_warmup_steps": int(
            checkpoint_config.get("motion_mask_warmup_steps", 300)
        ),
        "physical_mask_transition_steps": int(
            checkpoint_config.get("physical_mask_transition_steps", 1000)
        ),
        "physics_state_mode": str(checkpoint_config.get("physics_state_mode", "x0_hat")),
        "use_sigma_gate": to_bool(checkpoint_config.get("use_sigma_gate"), True),
        "sigma_gate_curve": str(checkpoint_config.get("sigma_gate_curve", "quadratic")),
        "use_sigma_conditioning": to_bool(
            checkpoint_config.get("use_sigma_conditioning"),
            True,
        ),
        "sigma_conditioning_dim": (
            None
            if checkpoint_config.get("sigma_conditioning_dim") is None
            else int(checkpoint_config["sigma_conditioning_dim"])
        ),
        "sigma_gate_floor": float(checkpoint_config.get("sigma_gate_floor", 0.05)),
        "use_adaptive_condition_injection": to_bool(
            checkpoint_config.get("use_adaptive_condition_injection"),
            True,
        ),
        "adaptive_conditioning_dim": (
            None
            if checkpoint_config.get("adaptive_conditioning_dim") is None
            else int(checkpoint_config["adaptive_conditioning_dim"])
        ),
        "adaptive_conditioning_strength": float(
            checkpoint_config.get("adaptive_conditioning_strength", 0.5)
        ),
        "adaptive_conditioning_gate_floor": float(
            checkpoint_config.get("adaptive_conditioning_gate_floor", 0.05)
        ),
        "enable_rl_expert_optimization": to_bool(
            checkpoint_config.get("enable_rl_expert_optimization"),
            True,
        ),
        "rl_reward_decay": float(checkpoint_config.get("rl_reward_decay", 0.95)),
        "rl_hidden_dim": (
            None
            if checkpoint_config.get("rl_hidden_dim") is None
            else int(checkpoint_config["rl_hidden_dim"])
        ),
        "state_align_warmup_steps": int(
            checkpoint_config.get("state_align_warmup_steps", 1000)
        ),
        "state_align_x_weight": float(checkpoint_config.get("state_align_x_weight", 0.0)),
        "state_align_v_weight": float(checkpoint_config.get("state_align_v_weight", 0.0)),
        "curriculum_transition_start_step": int(
            checkpoint_config.get("curriculum_transition_start_step", 1000)
        ),
        "curriculum_transition_steps": int(
            checkpoint_config.get("curriculum_transition_steps", 1000)
        ),
        "physics_weight_target": float(
            checkpoint_config.get(
                "physics_weight_target",
                checkpoint_config.get("physics_weight", 0.1),
            )
        ),
        "output_physics_weight": float(checkpoint_config.get("output_physics_weight", 1.0)),
        "state_align_v_weight_target": float(
            checkpoint_config.get(
                "state_align_v_weight_target",
                checkpoint_config.get("state_align_v_weight", 0.0),
            )
        ),
        "decoded_branch_consistency_weight": float(
            checkpoint_config.get("decoded_branch_consistency_weight", 1e-2)
        ),
        "enable_explainability_reports": False,
    }


def build_module(args, checkpoint_config, train_module):
    runtime_kwargs = build_runtime_kwargs(args, checkpoint_config)
    module = train_module.WanPINNTrainingModule(**runtime_kwargs)
    module.eval()
    torch_dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    module.pipe.device = args.device
    module.pipe.torch_dtype = torch_dtype
    if args.device.startswith("cuda") and hasattr(module.pipe, "enable_cpu_offload"):
        if not getattr(module.pipe, "model_names", None):
            module.pipe.model_names = [
                "text_encoder",
                "image_encoder",
                "dit",
                "dit2",
                "vae",
                "motion_controller",
                "vace",
            ]
        module.pipe.enable_cpu_offload()
    module.physics_adapter = module.physics_adapter.to(device=args.device, dtype=torch_dtype)
    module.physics_adapter.eval()
    module.observable_proxy_extractor = module.observable_proxy_extractor.to(device=args.device)
    module.observable_proxy_extractor.eval()
    module.pipe.scheduler.set_timesteps(1000, training=True)
    return module


def sample_timestep_fraction(args, checkpoint_config):
    if args.timestep_fraction is not None:
        return min(max(float(args.timestep_fraction), 0.0), 1.0)
    min_boundary, max_boundary = resolve_boundaries(args, checkpoint_config)
    return 0.5 * (min_boundary + max_boundary)


def evaluate_sample(sample_idx, sample, sample_relpath, model, args, checkpoint_config, output_dir):
    prompt = str(sample.get("prompt", ""))
    sample_name = make_safe_sample_name(
        sample_relpath,
        fallback=f"sample_{sample_idx:04d}",
    )
    print(f"[sample {sample_idx}] {sample_name}")

    seed = int(args.seed + sample_idx)
    torch.manual_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    with torch.inference_mode():
        inputs = model.forward_preprocess(sample)
        timestep_fraction = sample_timestep_fraction(args, checkpoint_config)
        timestep_index = int(
            round(
                timestep_fraction
                * max(model.pipe.scheduler.num_train_timesteps - 1, 0)
            )
        )
        timestep_id = torch.tensor([timestep_index], dtype=torch.long)
        timestep = model.pipe.scheduler.timesteps[timestep_id].to(
            dtype=model.pipe.torch_dtype,
            device=inputs["latents"].device,
        )

        models = {name: getattr(model.pipe, name) for name in model.pipe.in_iteration_models}
        models, active_noise_regime, active_dit_expert_index = model._select_training_dit_expert(
            timestep_id, models
        )
        if active_noise_regime == "low_noise" and model.pipe.dit2 is not None:
            model.pipe.load_models_to_device(model.pipe.in_iteration_models_2)
        else:
            model.pipe.load_models_to_device(model.pipe.in_iteration_models)

        input_latents = inputs.get("input_latents", inputs["latents"])
        noise = inputs.get("noise", torch.randn_like(inputs["latents"]))
        z_t = model.pipe.scheduler.add_noise(input_latents, noise, timestep)
        v_original = model.pipe.model_fn(
            **models,
            latents=z_t,
            timestep=timestep,
            context=inputs.get("context"),
            clip_feature=inputs.get("clip_feature"),
            y=inputs.get("y"),
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
        )
        sigma = model._scheduler_sigma(
            timestep,
            device=v_original.device,
            dtype=v_original.dtype,
        )
        physics_state_original = model._physics_state_from_prediction(
            z_t,
            v_original,
            sigma,
        )
        proxy_targets = model._build_observable_proxy_targets(physics_state_original)
        stage1_outputs = model.physics_adapter.forward_observable_pretrain(
            physics_state_original,
            sigma=sigma,
        )
        loss_obs, obs_errors = model._observable_alignment_terms(
            stage1_outputs["observable_outputs"],
            proxy_targets,
            proxy_targets["proxy_conf"],
        )

        raw_video = model.pipe.preprocess_video(
            sample["video"],
            torch_dtype=model.pipe.torch_dtype,
            device=model.pipe.device,
        )
        x0hat_video = model._decode_observable_rgb_frames(physics_state_original)
        raw_lowres = F.interpolate(
            raw_video.float(),
            size=x0hat_video.shape[2:],
            mode="trilinear",
            align_corners=False,
        ).to(dtype=model.pipe.torch_dtype)
        raw_proxy = model.observable_proxy_extractor(raw_lowres)["flow_proxy"][0].detach().cpu().float().numpy()
        target_flow = proxy_targets["flow_proxy"][0].detach().cpu().float().numpy()
        pred_flow = stage1_outputs["observable_outputs"]["flow"][0].detach().cpu().float().numpy()
        proxy_conf = proxy_targets["proxy_conf"][0, 0].detach().cpu().float().numpy()
        error = np.sqrt(((pred_flow - target_flow) ** 2).sum(axis=0))
        epe_mean = float(error.mean())
        epe_p95 = float(np.percentile(error, 95))
        pred_mag = np.sqrt((pred_flow ** 2).sum(axis=0))
        target_mag = np.sqrt((target_flow ** 2).sum(axis=0))
        mag_ratio = float(pred_mag.mean() / max(target_mag.mean(), 1e-6))

        active_recovery_phase = infer_recovery_phase(model, checkpoint_config)
        physical_metadata = build_physical_metadata(model, physics_state_original)
        cache = getattr(model.physics_adapter, "_cache", {})
        if not isinstance(cache, dict):
            raise RuntimeError("PhysicsAdapter cache is required for physical field inspection.")
        shared_attribute_bank_live = cache.get("shared_attribute_bank_live")
        physics_feat_live = cache.get("physics_feat_live")
        if shared_attribute_bank_live is None:
            shared_attribute_bank_live = stage1_outputs.get("shared_attribute_bank")
        if shared_attribute_bank_live is None or physics_feat_live is None:
            raise RuntimeError(
                "Physical field inspection requires shared_attribute_bank_live/shared_attribute_bank "
                "and physics_feat_live."
            )
        physical_field_dict, _, physical_metrics = model.physics_adapter.build_physical_field_dict(
            shared_attribute_bank_live,
            physics_feat_live,
            metadata=physical_metadata,
            field_recovery_phase=active_recovery_phase,
        )
        physical_summary = summarize_physical_fields(
            physical_field_dict,
            active_recovery_phase,
            physical_metrics,
        )
        physical_field_rows = build_physical_visual_rows(physical_field_dict)

    sample_dir = output_dir / f"{sample_idx:03d}_{sample_name}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = sample_dir / "flow_sheet.png"
    video_path = sample_dir / "flow.mp4"
    physical_sheet_path = sample_dir / "physical_fields_sheet.png"
    physical_summary_path = sample_dir / "physical_fields_summary.json"
    make_contact_sheet(
        sample_name,
        prompt,
        raw_video.detach().cpu(),
        x0hat_video.detach().cpu(),
        raw_proxy,
        target_flow,
        pred_flow,
        error,
        sheet_path,
    )
    saved_video_path = make_video(
        sample_name,
        raw_video.detach().cpu(),
        x0hat_video.detach().cpu(),
        raw_proxy,
        target_flow,
        pred_flow,
        error,
        video_path,
    )
    make_field_contact_sheet(
        sample_name,
        f"{prompt[:72]} | phase={physical_summary['active_field_recovery_phase']}",
        physical_field_rows,
        physical_sheet_path,
    )
    physical_summary_path.write_text(
        json.dumps(physical_summary, indent=2, ensure_ascii=False)
    )

    model.pipe.load_models_to_device([])
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "sample_idx": int(sample_idx),
        "sample_name": sample_name,
        "prompt": prompt,
        "video_relpath": str(sample_relpath),
        "seed": seed,
        "motion_score": motion_score(sample["video"]),
        "timestep_fraction": float(timestep_fraction),
        "timestep_index": int(timestep_index),
        "sigma_mean": float(sigma.detach().float().mean().item()),
        "noise_regime": active_noise_regime,
        "active_dit_expert_index": int(active_dit_expert_index),
        "loss_obs": float(loss_obs.detach().item()),
        "obs_flow_error": float(obs_errors["flow_error"].detach().item()),
        "obs_deformation_error": float(obs_errors["deformation_error"].detach().item()),
        "flow_epe_mean_vs_target_proxy": epe_mean,
        "flow_epe_p95_vs_target_proxy": epe_p95,
        "pred_target_magnitude_ratio": mag_ratio,
        "proxy_conf_mean": float(proxy_conf.mean()),
        "sheet_path": str(sheet_path),
        "video_path": str(saved_video_path),
        "physical_sheet_path": str(physical_sheet_path),
        "physical_summary_path": str(physical_summary_path),
        "active_field_recovery_phase": physical_summary["active_field_recovery_phase"],
        "rho_mean": float(physical_summary["physical_metrics"]["rho_mean"]),
        "rho_min": float(physical_summary["physical_metrics"]["rho_min"]),
        "p_abs_mean": float(physical_summary["physical_metrics"]["p_abs_mean"]),
        "div_u_abs_mean": float(physical_summary["physical_metrics"]["div_u_abs_mean"]),
        "eps_symmetry_error": float(physical_summary["eps_symmetry_error"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Strict stage1 observable inspection using Wan DiT one-step x0_hat reconstruction."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset_base_path",
        type=str,
        default="/home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data",
    )
    parser.add_argument(
        "--dataset_metadata_path",
        type=str,
        default="/home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data/metadata_standard.csv",
    )
    parser.add_argument("--model_paths", type=str, default=None)
    parser.add_argument(
        "--model_id_with_origin_paths",
        type=str,
        default=DEFAULT_MODEL_ID_WITH_ORIGIN_PATHS,
    )
    parser.add_argument("--flow_backbone_ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--scan_count", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--sample_indices", type=str, default="")
    parser.add_argument("--min_timestep_boundary", type=float, default=None)
    parser.add_argument("--max_timestep_boundary", type=float, default=None)
    parser.add_argument("--timestep_fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else checkpoint_path.parent / f"{checkpoint_path.stem}_strict_x0hat_inspection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_module = load_train_module()
    print("Loading checkpoint:", checkpoint_path)
    ckpt = load_checkpoint(str(checkpoint_path))
    checkpoint_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = build_module(args, checkpoint_config, train_module)
    dataset = build_dataset(args, train_module)
    explicit_indices = (
        [int(item) for item in args.sample_indices.split(",") if item.strip()]
        if args.sample_indices else []
    )
    sample_indices = choose_samples(dataset, args.scan_count, args.num_samples, explicit_indices)
    print("Selected sample indices:", sample_indices)

    results = []
    for sample_idx in sample_indices:
        sample = dataset[sample_idx]
        sample_meta = dataset.data[sample_idx % len(dataset.data)]
        result = evaluate_sample(
            sample_idx,
            sample,
            sample_meta.get("video", f"sample_{sample_idx:04d}"),
            model,
            args,
            checkpoint_config,
            output_dir,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    min_boundary, max_boundary = resolve_boundaries(args, checkpoint_config)
    summary = {
        "checkpoint": str(checkpoint_path),
        "inspection_mode": "strict_x0hat",
        "training_stage": checkpoint_config.get("training_stage"),
        "flow_backbone_ckpt": (
            args.flow_backbone_ckpt
            if args.flow_backbone_ckpt is not None
            else checkpoint_config.get("flow_backbone_ckpt")
        ),
        "model_id_with_origin_paths": args.model_id_with_origin_paths,
        "min_timestep_boundary": min_boundary,
        "max_timestep_boundary": max_boundary,
        "timestep_fraction": sample_timestep_fraction(args, checkpoint_config),
        "sample_indices": sample_indices,
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved summary to:", summary_path)


if __name__ == "__main__":
    main()
