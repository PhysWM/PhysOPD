"""
Physics-Informed Video Generation Inference Script
物理约束视频生成推理脚本

使用方法:
    python inference_pinn.py --prompt "water flowing down" --output video_pinn.mp4
"""
import torch
import sys
import os
import csv
import json
import re
import hashlib
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from diffsynth import save_video
from diffsynth.pipelines.wan_video_pinn import PhysicsInformedWanVideoPipeline
from diffsynth.pipelines.wan_video_new import ModelConfig


PHENOMENON_LABELS = [
    "rigid body motion",
    "collision",
    "liquid motion",
    "gas motion",
    "elastic motion",
    "deformation",
    "melting",
    "solidification",
    "vaporization",
    "liquefaction",
    "combustion",
    "explosion",
    "reflection",
    "refraction",
    "scattering",
    "interference and diffraction",
    "unnatural light source",
]
PHENOMENON_TO_ID = {name: idx for idx, name in enumerate(PHENOMENON_LABELS)}
PHENOMENON_ALIAS = {
    "liquid_motion": "liquid motion",
    "rigid_body_motion": "rigid body motion",
    "elastic_motion": "elastic motion",
    "gas_motion": "gas motion",
    "interference_and_diffraction": "interference and diffraction",
    "unnatural_light_source": "unnatural light source",
}


def _parse_vector(text, cast_type=float):
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        return [cast_type(v) for v in text]
    text = str(text).strip()
    if text == "":
        return None
    items = [it.strip() for it in text.split(",") if it.strip() != ""]
    if not items:
        return None
    return [cast_type(it) for it in items]


def _safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def _normalize_label(label):
    clean = _safe_text(label).lower().replace("_", " ")
    clean = re.sub(r"\s+", " ", clean)
    return PHENOMENON_ALIAS.get(clean.replace(" ", "_"), clean)


def _hash_to_id(text, modulo):
    if modulo <= 1:
        return 0
    text = _safe_text(text).lower()
    if text == "":
        return 0
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    stable_hash = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (stable_hash % (modulo - 1)) + 1


def _parse_numeric_range(text):
    text = _safe_text(text).lower()
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(matches) == 0:
        return 0.0, 0.0, 0.0, 0.0
    values = [float(x) for x in matches]
    min_val = min(values)
    max_val = max(values)
    mean_val = sum(values) / max(len(values), 1)
    return min_val, max_val, mean_val, 1.0


def _encode_q_field(text, dim):
    vec = [0.0 for _ in range(max(dim, 0))]
    text = _safe_text(text).lower()
    if text == "" or dim == 0:
        return vec
    tokens = re.split(r"[,;|/]| and |\.", text)
    tokens = [re.sub(r"\s+", " ", t).strip() for t in tokens if t.strip()]
    for token in tokens:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        stable_hash = int.from_bytes(digest[:8], byteorder="big", signed=False)
        idx = stable_hash % dim
        vec[idx] = 1.0
    return vec


def _is_encoded_metadata(metadata):
    if not isinstance(metadata, dict):
        return False
    encoded_keys = {"label_id", "n_numeric", "n_text_ids", "q_vector"}
    return any(key in metadata for key in encoded_keys)


def encode_raw_metadata(raw_metadata, n_text_vocab_size=2048, q_dim=64):
    if not isinstance(raw_metadata, dict):
        return None
    label_name = _normalize_label(raw_metadata.get("label", raw_metadata.get("label_name", "")))
    label_id = raw_metadata.get("label_id")
    if label_id is None:
        label_id = PHENOMENON_TO_ID.get(label_name, PHENOMENON_TO_ID["liquid motion"])
    label_id = int(label_id)

    n_raw_0 = raw_metadata.get("n0", raw_metadata.get("n1", ""))
    n_raw_1 = raw_metadata.get("n1", raw_metadata.get("n2", ""))
    n_raw_2 = raw_metadata.get("n2", raw_metadata.get("n3", ""))

    n_numeric = []
    for value in (n_raw_0, n_raw_1, n_raw_2):
        n_min, n_max, n_mean, n_valid = _parse_numeric_range(value)
        n_numeric.extend([n_min, n_max, n_mean, n_valid])

    n_text_ids = [
        _hash_to_id(n_raw_0, n_text_vocab_size),
        _hash_to_id(n_raw_1, n_text_vocab_size),
        _hash_to_id(n_raw_2, n_text_vocab_size),
    ]

    q_vector = [0.0 for _ in range(q_dim)]
    for key in ("q0", "q1", "q2", "q4"):
        encoded = _encode_q_field(raw_metadata.get(key, ""), q_dim)
        q_vector = [min(1.0, qv + ev) for qv, ev in zip(q_vector, encoded)]
    q3 = _safe_text(raw_metadata.get("q3", "")).lower()
    if q_dim > 0:
        if q3 in {"yes", "true", "1"}:
            q_vector[0] = 1.0
        elif q3 in {"no", "false", "0"} and q_dim > 1:
            q_vector[1] = 1.0

    encoded = {
        "label_name": label_name,
        "label_id": label_id,
        "n_numeric": n_numeric,
        "n_text_ids": n_text_ids,
        "q_vector": q_vector,
    }
    return encoded


def _load_metadata_source(metadata_json=None, metadata_csv=None):
    if metadata_json is None and metadata_csv is None:
        return None
    if metadata_json is not None:
        if os.path.exists(metadata_json):
            with open(metadata_json, "r", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(metadata_json)
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
        if row is None:
            raise ValueError(f"metadata_csv is empty: {metadata_csv}")
        return row


def load_inference_metadata(metadata_json=None, metadata_csv=None):
    raw_or_encoded = _load_metadata_source(metadata_json=metadata_json, metadata_csv=metadata_csv)
    if raw_or_encoded is None:
        return None, "none"

    if _is_encoded_metadata(raw_or_encoded):
        metadata = dict(raw_or_encoded)
        if "label_id" in metadata:
            metadata["label_id"] = int(metadata["label_id"])
        if isinstance(metadata.get("n_numeric"), str):
            metadata["n_numeric"] = _parse_vector(metadata.get("n_numeric"), cast_type=float)
        if isinstance(metadata.get("n_text_ids"), str):
            metadata["n_text_ids"] = _parse_vector(metadata.get("n_text_ids"), cast_type=int)
        if isinstance(metadata.get("q_vector"), str):
            metadata["q_vector"] = _parse_vector(metadata.get("q_vector"), cast_type=float)
        if not isinstance(metadata, dict) or len(metadata) == 0:
            return None, "none"
        return metadata, "encoded_passthrough"

    metadata = encode_raw_metadata(raw_or_encoded)
    if not isinstance(metadata, dict) or len(metadata) == 0:
        return None, "none"
    return metadata, "raw_encoded"


def inference_pinn(
    prompt="water flowing down",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
    # 模型参数
    model_id="Wan-AI/Wan2.2-T2V-A14B",
    checkpoint_path=None,  # PINN checkpoint路径（可选）
    # 生成参数
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=50,
    seed=0,
    cfg_scale=5.0,
    # 物理参数
    enable_physics_constraint=False,  # 推理时通常不需要物理约束
    # 输出参数
    output_path="video_pinn.mp4",
    fps=15,
    quality=5,
    attention_overlay=True,
    attention_alpha=0.45,
    attention_use_motion_weighted=True,
    attention_motion_percentile=90.0,
    metadata_json=None,
    metadata_csv=None,
    # 设备参数
    device="cuda",
    torch_dtype=torch.bfloat16,
):
    """
    使用 Physics-Informed 模型生成视频
    """
    print("=" * 80)
    print("Physics-Informed Video Generation Inference")
    print("=" * 80)
    
    # 1. 加载模型
    print("\n[1/4] Loading model...")
    pipe = PhysicsInformedWanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
                offload_device="cpu"
            ),
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
                offload_device="cpu"
            ),
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu"
            ),
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="Wan2.1_VAE.pth",
                offload_device="cpu"
            ),
        ],
    )
    pipe.enable_vram_management()
    
    # 2. 加载 PINN plugin（如果有）
    if checkpoint_path:
        print(f"\n[2/4] Loading PINN plugin from {checkpoint_path}...")
        pipe.load_pinn_plugin(checkpoint_path, device=device, enable_tracking=True)
        print("PINN plugin active: every denoising step will apply PhysicsAdapter + PDE检验")
    else:
        print("\n[2/4] No PINN plugin provided, using original model")
    
    # 3. 加载元数据（包含 phenomenon label 用于 MoE 路由）
    pinn_metadata, metadata_mode = load_inference_metadata(
        metadata_json=metadata_json,
        metadata_csv=metadata_csv,
    )
    if pinn_metadata is not None:
        print(f"[3/4] Using PINN metadata for MoE routing (mode={metadata_mode}).")
        phenomenon = pinn_metadata.get("label_name", "liquid motion")
    else:
        print(f"[3/4] No PINN metadata provided, using default phenomenon")
        phenomenon = "liquid motion"

    # 4. 生成视频（同时实时追踪每步的物理场 PDE 残差）
    print(f"\n[4/4] Generating video with real-time physics tracking...")
    print(f"  Prompt: {prompt}")
    print(f"  Phenomenon: {phenomenon}")
    print(f"  Resolution: {height}x{width}, Frames: {num_frames}")
    print(f"  Steps: {num_inference_steps}, CFG Scale: {cfg_scale}")

    pipe.reset_tracking()

    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        seed=seed,
        cfg_scale=cfg_scale,
        tiled=True,
        pinn_metadata=pinn_metadata,
    )
    
    # 5. 保存视频
    save_video(video, output_path, fps=fps, quality=quality)
    print(f"\n✓ Video saved to {output_path}")
    
    # 6. 生成物理场验证报告
    if checkpoint_path and pipe.physics_tracking and len(pipe.physics_tracking.get("steps", [])) > 0:
        report_path = str(Path(output_path).with_suffix("")) + "_physics_report.png"
        
        pipe.save_physics_report(
            report_path,
            video_frames=video,
            attention_overlay=attention_overlay,
            attention_alpha=attention_alpha,
            attention_video_fps=fps,
            attention_use_motion_weighted=attention_use_motion_weighted,
            attention_motion_percentile=attention_motion_percentile,
        )
    
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Physics-Informed Video Generation Inference")
    
    # 生成参数
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, 
                       default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
                       help="Negative prompt")
    
    # 模型参数
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-T2V-A14B", help="Model ID")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="PINN checkpoint path (optional)")
    
    # 视频参数
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale")
    
    # 物理参数（保留兼容性）
    parser.add_argument("--enable_physics", action="store_true", help="(Deprecated) Physics is auto-enabled when checkpoint_path is provided")
    
    # 输出参数
    parser.add_argument("--output", type=str, default="video_pinn.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=15, help="Video FPS")
    parser.add_argument("--quality", type=int, default=5, help="Video quality (1-10)")
    parser.add_argument(
        "--disable_attention_overlay",
        action="store_true",
        help="Disable attention overlay MP4/PNG generation in physics report",
    )
    parser.add_argument(
        "--attention_alpha",
        type=float,
        default=0.45,
        help="Overlay alpha for attention map visualization (0~1)",
    )
    parser.add_argument(
        "--disable_motion_weighted_attention",
        action="store_true",
        help="Use legacy raw_correction attention map instead of motion-weighted map",
    )
    parser.add_argument(
        "--attention_motion_percentile",
        type=float,
        default=90.0,
        help="Percentile threshold for motion prior when building weighted attention (default: 90)",
    )
    parser.add_argument("--metadata_json", type=str, default=None, help="Metadata JSON string or JSON file path")
    parser.add_argument("--metadata_csv", type=str, default=None, help="CSV file path with label_id/label_name/n_numeric/n_text_ids/q_vector")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    inference_pinn(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        model_id=args.model_id,
        checkpoint_path=args.checkpoint_path,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        enable_physics_constraint=args.enable_physics,
        output_path=args.output,
        fps=args.fps,
        quality=args.quality,
        attention_overlay=not args.disable_attention_overlay,
        attention_alpha=args.attention_alpha,
        attention_use_motion_weighted=not args.disable_motion_weighted_attention,
        attention_motion_percentile=args.attention_motion_percentile,
        metadata_json=args.metadata_json,
        metadata_csv=args.metadata_csv,
        device=args.device,
    )
