#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
import types
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def register_local_diffsynth_packages():
    if "diffsynth" not in sys.modules:
        pkg = types.ModuleType("diffsynth")
        pkg.__path__ = [str(REPO_ROOT / "diffsynth")]
        sys.modules["diffsynth"] = pkg
    if "diffsynth.models" not in sys.modules:
        pkg = types.ModuleType("diffsynth.models")
        pkg.__path__ = [str(REPO_ROOT / "diffsynth" / "models")]
        sys.modules["diffsynth.models"] = pkg


register_local_diffsynth_packages()
load_module_from_path(
    "diffsynth.models.pinn_contracts",
    REPO_ROOT / "diffsynth" / "models" / "pinn_contracts.py",
)
pinn_adapter_module = load_module_from_path(
    "diffsynth.models.pinn_adapter",
    REPO_ROOT / "diffsynth" / "models" / "pinn_adapter.py",
)
wan_video_vae_module = load_module_from_path(
    "wan_video_vae_local",
    REPO_ROOT / "diffsynth" / "models" / "wan_video_vae.py",
)
flow_match_module = load_module_from_path(
    "flow_match_local",
    REPO_ROOT / "diffsynth" / "schedulers" / "flow_match.py",
)

PhysicsAdapter = pinn_adapter_module.PhysicsAdapter
WanVideoVAE = wan_video_vae_module.WanVideoVAE
FlowMatchScheduler = flow_match_module.FlowMatchScheduler


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


class FrozenDenseFlowTeacher(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        hidden_dim = max(int(hidden_dim), 8)
        self.pair_backbone = nn.Sequential(
            nn.Conv2d(6, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1),
        )
        self.has_learned_weights = False
        self.eval()
        self.requires_grad_(False)

    @staticmethod
    def _grad_x(image):
        return F.pad(image[..., 1:] - image[..., :-1], (0, 1, 0, 0))

    @staticmethod
    def _grad_y(image):
        return F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))

    def _analytic_flow(self, video):
        batch_size, _, num_frames, height, width = video.shape
        if num_frames <= 1:
            return torch.zeros(
                batch_size,
                2,
                1,
                height,
                width,
                device=video.device,
                dtype=video.dtype,
            )
        gray = video.float().mean(dim=1, keepdim=True)
        current = gray[:, :, :-1].permute(0, 2, 1, 3, 4).reshape(-1, 1, height, width)
        future = gray[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(-1, 1, height, width)
        grad_x = self._grad_x(current)
        grad_y = self._grad_y(current)
        grad_t = future - current
        denom = grad_x.square() + grad_y.square() + 1e-4
        flow_x = (-grad_t * grad_x / denom).clamp(-1.0, 1.0)
        flow_y = (-grad_t * grad_y / denom).clamp(-1.0, 1.0)
        pair_flow = torch.cat([flow_x, flow_y], dim=1)
        return pair_flow.view(batch_size, num_frames - 1, 2, height, width).permute(0, 2, 1, 3, 4)

    def forward(self, video):
        pair_flow = self._analytic_flow(video)
        if pair_flow.shape[2] == 0:
            return torch.zeros(
                video.shape[0],
                2,
                video.shape[2],
                video.shape[3],
                video.shape[4],
                device=video.device,
                dtype=video.dtype,
            )
        last_flow = pair_flow[:, :, -1:]
        full_flow = torch.cat([pair_flow, last_flow], dim=2)
        return torch.nan_to_num(full_flow.to(dtype=video.dtype), nan=0.0, posinf=0.0, neginf=0.0)


class ObservableProxyExtractor(nn.Module):
    def __init__(self, flow_teacher):
        super().__init__()
        self.flow_teacher = flow_teacher
        self.eval()
        self.requires_grad_(False)

    @staticmethod
    def _grad_x(tensor):
        return F.pad(tensor[..., 1:] - tensor[..., :-1], (0, 1, 0, 0))

    @staticmethod
    def _grad_y(tensor):
        return F.pad(tensor[..., 1:, :] - tensor[..., :-1, :], (0, 0, 0, 1))

    def forward(self, rgb_video):
        rgb_video = torch.nan_to_num(rgb_video.float(), nan=0.0, posinf=0.0, neginf=0.0)
        flow_proxy = self.flow_teacher(rgb_video).detach()
        flow_x = flow_proxy[:, 0:1]
        flow_y = flow_proxy[:, 1:2]
        du_dx = self._grad_x(flow_x)
        du_dy = self._grad_y(flow_x)
        dv_dx = self._grad_x(flow_y)
        dv_dy = self._grad_y(flow_y)
        deformation_proxy = torch.cat([du_dx, du_dy, dv_dx, dv_dy], dim=1)
        return {
            "flow_proxy": flow_proxy.to(dtype=rgb_video.dtype).detach(),
            "deformation_proxy": deformation_proxy.to(dtype=rgb_video.dtype).detach(),
        }


class SimpleVideoPipe:
    def __init__(self, device, torch_dtype, vae):
        self.device = device
        self.torch_dtype = torch_dtype
        self.vae = vae

    def preprocess_video(self, video_frames, torch_dtype=None, device=None, min_value=-1.0, max_value=1.0):
        target_dtype = torch_dtype or self.torch_dtype
        target_device = device or self.device
        frames = []
        scale = (max_value - min_value) / 255.0
        for frame in video_frames:
            array = np.asarray(frame, dtype=np.float32)
            tensor = torch.from_numpy(array).to(device=target_device, dtype=target_dtype)
            tensor = tensor * scale + min_value
            frames.append(tensor.permute(2, 0, 1))
        return torch.stack(frames, dim=1).unsqueeze(0)


class SimpleVideoDataset:
    def __init__(self, base_path, metadata_path, height, width, num_frames):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.data = self._load_metadata(metadata_path)

    @staticmethod
    def _load_metadata(metadata_path):
        if metadata_path.endswith(".json"):
            with open(metadata_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        if metadata_path.endswith(".jsonl"):
            rows = []
            with open(metadata_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _crop_and_resize(self, image):
        src_w, src_h = image.size
        scale = max(self.width / max(src_w, 1), self.height / max(src_h, 1))
        resized = image.resize(
            (max(int(round(src_w * scale)), 1), max(int(round(src_h * scale)), 1)),
            Image.BILINEAR,
        )
        left = max((resized.width - self.width) // 2, 0)
        top = max((resized.height - self.height) // 2, 0)
        return resized.crop((left, top, left + self.width, top + self.height))

    def _load_video(self, relative_path):
        file_path = os.path.join(self.base_path, relative_path)
        reader = imageio.get_reader(file_path)
        frames = []
        try:
            for frame_idx, frame in enumerate(reader):
                if frame_idx >= self.num_frames:
                    break
                image = Image.fromarray(frame).convert("RGB")
                frames.append(self._crop_and_resize(image))
        finally:
            reader.close()
        if not frames:
            raise RuntimeError(f"No frames loaded from {file_path}")
        return frames

    def __getitem__(self, index):
        row = dict(self.data[index % len(self.data)])
        if "video" not in row:
            raise KeyError(f"Metadata row at index {index} does not contain 'video'")
        row["video_relpath"] = row["video"]
        row["video"] = self._load_video(row["video_relpath"])
        return row

    def __len__(self):
        return len(self.data)


def load_checkpoint(checkpoint_path):
    register_checkpoint_stubs()
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def build_adapter(ckpt, device, torch_dtype):
    config = ckpt["config"]
    adapter = PhysicsAdapter(
        latent_dim=16,
        hidden_dim=int(config["adapter_hidden_dim"]),
        physics_attr_dim=int(config["physics_attr_dim"]),
        num_phenomena=int(config["num_phenomena"]),
        n_numeric_dim=int(config["n_numeric_dim"]),
        q_input_dim=int(config["q_input_dim"]),
        n_text_vocab_size=int(config["n_text_vocab_size"]),
        moe_top_k=int(config["moe_top_k"]),
        physics_state_mode=config["physics_state_mode"],
        use_sigma_gate=bool(config["use_sigma_gate"]),
        sigma_gate_curve=config["sigma_gate_curve"],
        use_sigma_conditioning=bool(config["use_sigma_conditioning"]),
        sigma_conditioning_dim=int(config["sigma_conditioning_dim"]),
        sigma_gate_floor=float(config["sigma_gate_floor"]),
        use_adaptive_condition_injection=bool(config["use_adaptive_condition_injection"]),
        adaptive_conditioning_dim=int(config["adaptive_conditioning_dim"]),
        adaptive_conditioning_strength=float(config["adaptive_conditioning_strength"]),
        adaptive_conditioning_gate_floor=float(config["adaptive_conditioning_gate_floor"]),
        enable_rl_expert_optimization=bool(config["enable_rl_expert_optimization"]),
        rl_hidden_dim=int(config["rl_hidden_dim"]),
        rl_reward_decay=float(config["rl_reward_decay"]),
    ).to(device=device, dtype=torch_dtype)
    missing, unexpected = adapter.load_state_dict(ckpt["physics_adapter_state_dict"], strict=False)
    if missing:
        print("Warning: missing adapter keys:", missing[:20])
    if unexpected:
        print("Warning: unexpected adapter keys:", unexpected[:20])
    adapter.eval()
    adapter.requires_grad_(False)
    return adapter


def load_vae(device, torch_dtype, vae_path):
    state_dict = torch.load(vae_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    vae = WanVideoVAE()
    if not all(str(key).startswith("model.") for key in state_dict.keys()):
        state_dict = vae.state_dict_converter().from_civitai(state_dict)
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    if missing:
        print("Warning: missing VAE keys:", missing[:20])
    if unexpected:
        print("Warning: unexpected VAE keys:", unexpected[:20])
    vae = vae.eval().to(device=device, dtype=torch_dtype)
    return SimpleVideoPipe(device=device, torch_dtype=torch_dtype, vae=vae)


def build_dataset(args):
    return SimpleVideoDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )


def compute_stage_sigma(min_boundary, max_boundary):
    scheduler = FlowMatchScheduler()
    scheduler.set_timesteps(1000, training=True)
    start = max(0, min(int(min_boundary * scheduler.num_train_timesteps), scheduler.num_train_timesteps - 1))
    end = max(start + 1, min(int(max_boundary * scheduler.num_train_timesteps), scheduler.num_train_timesteps))
    sigma_values = scheduler.sigmas[start:end]
    return float(sigma_values.mean().item()), float(sigma_values[0].item()), float(sigma_values[-1].item())


def motion_score(video_frames):
    array = np.stack([np.asarray(frame, dtype=np.float32) for frame in video_frames], axis=0)
    if array.shape[0] <= 1:
        return 0.0
    diff = np.abs(array[1:] - array[:-1]).mean(axis=(1, 2, 3))
    return float(diff.mean())


def make_safe_sample_name(value, fallback):
    text = str(value) if value is not None else ""
    text = os.path.basename(text)
    text = os.path.splitext(text)[0]
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    if not text:
        text = fallback
    return text[:96]


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


def make_contact_sheet(sample_name, sample_prompt, raw_video, raw_flow, recon_flow, pred_flow, error_map, out_path):
    time_steps = raw_flow.shape[1]
    columns = np.linspace(0, time_steps - 1, 6, dtype=int)
    tile_size = (192, 108)
    left_margin = 132
    top_margin = 54
    row_gap = 12
    col_gap = 8
    row_labels = ["Raw Frame", "Raw Proxy", "Recon Proxy", "Pred Flow", "Error"]
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
                np.sqrt((recon_flow ** 2).sum(axis=0)).reshape(-1),
                np.sqrt((pred_flow ** 2).sum(axis=0)).reshape(-1),
            ]
        ),
        95,
    )
    max_error = np.percentile(error_map.reshape(-1), 95)

    raw_frames_count = raw_video.shape[1]
    for col, t_idx in enumerate(columns):
        raw_t = int(round(t_idx / max(time_steps - 1, 1) * max(raw_frames_count - 1, 0)))
        tiles = [
            to_display_frame(raw_video[0, :, raw_t]),
            flow_to_rgb(raw_flow[:, t_idx], max_magnitude),
            flow_to_rgb(recon_flow[:, t_idx], max_magnitude),
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


def make_video(sample_name, raw_video, raw_flow, recon_flow, pred_flow, error_map, out_path, fps=6):
    max_magnitude = np.percentile(
        np.concatenate(
            [
                np.sqrt((raw_flow ** 2).sum(axis=0)).reshape(-1),
                np.sqrt((recon_flow ** 2).sum(axis=0)).reshape(-1),
                np.sqrt((pred_flow ** 2).sum(axis=0)).reshape(-1),
            ]
        ),
        95,
    )
    max_error = np.percentile(error_map.reshape(-1), 95)
    frames = []
    raw_frames_count = raw_video.shape[1]
    time_steps = raw_flow.shape[1]
    for t_idx in range(time_steps):
        raw_t = int(round(t_idx / max(time_steps - 1, 1) * max(raw_frames_count - 1, 0)))
        panels = [
            add_label(resize_rgb(to_display_frame(raw_video[0, :, raw_t]), (224, 126)), "Raw Frame"),
            add_label(resize_rgb(flow_to_rgb(raw_flow[:, t_idx], max_magnitude), (224, 126)), "Raw Proxy"),
            add_label(resize_rgb(flow_to_rgb(recon_flow[:, t_idx], max_magnitude), (224, 126)), "Recon Proxy"),
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


def evaluate_sample(sample_idx, sample, pipe, adapter, observable_extractor, sigma_value, output_dir):
    prompt = str(sample.get("prompt", ""))
    sample_name = make_safe_sample_name(
        sample.get("video_relpath"),
        fallback=f"sample_{sample_idx:04d}",
    )
    print(f"[sample {sample_idx}] {sample_name}")

    with torch.inference_mode():
        raw_video = pipe.preprocess_video(sample["video"], torch_dtype=pipe.torch_dtype, device=pipe.device)
        latents = pipe.vae.encode(raw_video, device=pipe.device, tiled=False).to(dtype=pipe.torch_dtype, device=pipe.device)
        decoded_video = pipe.vae.decode(latents, device=pipe.device, tiled=False).to(dtype=pipe.torch_dtype, device=pipe.device)
        if tuple(decoded_video.shape[2:]) != tuple(latents.shape[2:]):
            decoded_video = F.interpolate(
                decoded_video.float(),
                size=latents.shape[2:],
                mode="trilinear",
                align_corners=False,
            ).to(dtype=pipe.torch_dtype)
        raw_lowres = F.interpolate(
            raw_video.float(),
            size=latents.shape[2:],
            mode="trilinear",
            align_corners=False,
        ).to(dtype=pipe.torch_dtype)
        target_raw = observable_extractor(raw_lowres)["flow_proxy"][0].detach().cpu().float().numpy()
        target_recon = observable_extractor(decoded_video)["flow_proxy"][0].detach().cpu().float().numpy()
        sigma = torch.full((latents.shape[0],), sigma_value, device=pipe.device, dtype=pipe.torch_dtype)
        pred = adapter.forward_observable_pretrain(latents, sigma=sigma)["observable_outputs"]["flow"][0].detach().cpu().float().numpy()

    error = np.sqrt(((pred - target_recon) ** 2).sum(axis=0))
    epe_mean = float(error.mean())
    epe_p95 = float(np.percentile(error, 95))
    target_mag = np.sqrt((target_recon ** 2).sum(axis=0))
    pred_mag = np.sqrt((pred ** 2).sum(axis=0))
    mag_ratio = float(pred_mag.mean() / max(target_mag.mean(), 1e-6))

    contact_path = output_dir / f"{sample_idx:03d}_{sample_name}_sheet.png"
    video_path = output_dir / f"{sample_idx:03d}_{sample_name}_flow.mp4"
    make_contact_sheet(sample_name, prompt, raw_video.detach().cpu(), target_raw, target_recon, pred, error, contact_path)
    saved_video_path = make_video(
        sample_name,
        raw_video.detach().cpu(),
        target_raw,
        target_recon,
        pred,
        error,
        video_path,
    )

    return {
        "sample_idx": int(sample_idx),
        "sample_name": sample_name,
        "prompt": prompt,
        "motion_score": motion_score(sample["video"]),
        "flow_epe_mean_vs_recon_proxy": epe_mean,
        "flow_epe_p95_vs_recon_proxy": epe_p95,
        "pred_target_magnitude_ratio": mag_ratio,
        "sheet_path": str(contact_path),
        "video_path": saved_video_path,
    }


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


def main():
    parser = argparse.ArgumentParser(description="Visual inspect stage1 observable checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_base_path", type=str, default="/home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data")
    parser.add_argument("--dataset_metadata_path", type=str, default="/home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data/metadata_standard.csv")
    parser.add_argument("--vae_path", type=str, default="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--scan_count", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--sample_indices", type=str, default="")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.90)
    parser.add_argument("--max_timestep_boundary", type=float, default=1.00)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else checkpoint_path.parent / f"{checkpoint_path.stem}_inspection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    print("Loading checkpoint:", checkpoint_path)
    ckpt = load_checkpoint(str(checkpoint_path))
    adapter = build_adapter(ckpt, device=args.device, torch_dtype=torch_dtype)
    pipe = load_vae(device=args.device, torch_dtype=torch_dtype, vae_path=args.vae_path)
    observable_extractor = ObservableProxyExtractor(FrozenDenseFlowTeacher(hidden_dim=max(adapter.hidden_dim // 2, 16))).to(args.device)
    observable_extractor.eval()
    observable_extractor.requires_grad_(False)

    sigma_mean, sigma_first, sigma_last = compute_stage_sigma(args.min_timestep_boundary, args.max_timestep_boundary)
    print(
        "Using sigma summary:",
        {
            "sigma_mean": sigma_mean,
            "sigma_first": sigma_first,
            "sigma_last": sigma_last,
        },
    )

    dataset = build_dataset(args)
    explicit_indices = [int(item) for item in args.sample_indices.split(",") if item.strip()] if args.sample_indices else []
    sample_indices = choose_samples(dataset, args.scan_count, args.num_samples, explicit_indices)
    print("Selected sample indices:", sample_indices)

    results = []
    for sample_idx in sample_indices:
        sample = dataset[sample_idx]
        result = evaluate_sample(sample_idx, sample, pipe, adapter, observable_extractor, sigma_mean, output_dir)
        results.append(result)

    summary = {
        "checkpoint": str(checkpoint_path),
        "training_stage": ckpt.get("config", {}).get("training_stage"),
        "flow_backbone_ckpt": ckpt.get("config", {}).get("flow_backbone_ckpt"),
        "sigma_eval_mean": sigma_mean,
        "sigma_eval_first": sigma_first,
        "sigma_eval_last": sigma_last,
        "sample_indices": sample_indices,
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved summary to:", summary_path)
    for item in results:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
