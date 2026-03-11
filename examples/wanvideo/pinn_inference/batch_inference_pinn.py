"""
批量视频生成脚本 - 根据CSV中的caption按顺序生成视频

支持 ID 范围，便于多卡并行：
    CUDA_VISIBLE_DEVICES=0 python batch_inference_pinn.py --start_id 1 --end_id 421 ...
    CUDA_VISIBLE_DEVICES=1 python batch_inference_pinn.py --start_id 422 --end_id 842 ...

输出命名: 0001.mp4, 0002.mp4, ... (与CSV行号一一对应)
"""
import argparse
import csv
import sys
import os
import json
import re
import hashlib
from pathlib import Path
from typing import Iterable

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
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


def _safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


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


def _normalize_encoded_metadata(metadata):
    normalized = dict(metadata)
    if "label_id" in normalized and _safe_text(normalized["label_id"]) != "":
        normalized["label_id"] = int(normalized["label_id"])
    if isinstance(normalized.get("n_numeric"), str):
        normalized["n_numeric"] = _parse_vector(normalized.get("n_numeric"), cast_type=float)
    if isinstance(normalized.get("n_text_ids"), str):
        normalized["n_text_ids"] = _parse_vector(normalized.get("n_text_ids"), cast_type=int)
    if isinstance(normalized.get("q_vector"), str):
        normalized["q_vector"] = _parse_vector(normalized.get("q_vector"), cast_type=float)
    return normalized


def encode_raw_metadata(raw_metadata, n_text_vocab_size=2048, q_dim=64):
    if not isinstance(raw_metadata, dict):
        return None
    label_name = _normalize_label(raw_metadata.get("label", raw_metadata.get("label_name", "")))
    label_id = raw_metadata.get("label_id")
    if label_id is None or _safe_text(label_id) == "":
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

    return {
        "label_name": label_name,
        "label_id": label_id,
        "n_numeric": n_numeric,
        "n_text_ids": n_text_ids,
        "q_vector": q_vector,
    }


def metadata_from_row(row):
    if not isinstance(row, dict):
        return None, "none"
    if _is_encoded_metadata(row):
        return _normalize_encoded_metadata(row), "encoded_row"
    raw_keys = {"label", "label_name", "n0", "n1", "n2", "n3", "q0", "q1", "q2", "q3", "q4"}
    has_raw = any(_safe_text(row.get(key, "")) != "" for key in raw_keys)
    if not has_raw:
        return None, "none"
    return encode_raw_metadata(row), "raw_row_encoded"


def load_global_metadata(metadata_json=None, metadata_csv=None):
    if metadata_json is None and metadata_csv is None:
        return None, "none"
    if metadata_json is not None:
        if os.path.exists(metadata_json):
            with open(metadata_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            payload = json.loads(metadata_json)
    else:
        with open(metadata_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            payload = next(reader, None)
            if payload is None:
                raise ValueError(f"metadata_csv is empty: {metadata_csv}")
    if _is_encoded_metadata(payload):
        return _normalize_encoded_metadata(payload), "encoded_global"
    return encode_raw_metadata(payload), "raw_global_encoded"


def load_rows_from_csv(csv_path: str) -> list[dict]:
    """读取CSV，返回每行字典（顺序与CSV一致，不含header）。"""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def scan_existing_ids(output_dir: Path) -> set[int]:
    """扫描输出目录内已存在的视频 ID（解析文件名数字部分）"""
    existing_ids: set[int] = set()
    for path in output_dir.glob("*.mp4"):
        stem = path.stem
        if stem.isdigit():
            existing_ids.add(int(stem))
    return existing_ids


def main():
    parser = argparse.ArgumentParser(
        description="批量生成视频 - 从CSV读取caption，支持ID范围多卡并行"
    )
    # CSV 与 范围
    parser.add_argument(
        "--csv",
        type=str,
        default="videophy_test_public.csv",
        help="CSV 文件路径（需包含 caption 列）",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        required=True,
        help="起始 ID（1-based，对应 CSV 第2行）",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        required=True,
        help="结束 ID（1-based，含该行）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_videos",
        help="输出目录",
    )

    # 模型
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/train/pinn_plugin_low_noise/pinn_plugin_final.pt",
        help="PINN checkpoint 路径",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B",
        help="Model ID",
    )

    # 生成参数
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
        help="Negative prompt",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed_base", type=int, default=0, help="种子基准，实际 seed = seed_base + id")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)

    # 设备
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--metadata_json",
        type=str,
        default=None,
        help="Global metadata JSON string or JSON file path (raw or encoded).",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default=None,
        help="Global metadata CSV path (raw or encoded); first row will be used.",
    )
    parser.add_argument("--skip_existing", action="store_true", help="若输出文件已存在则跳过")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：扫描输出目录，仅生成范围内缺失的 ID",
    )

    args = parser.parse_args()

    # 解析路径
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)

    # 读取 caption
    rows = load_rows_from_csv(str(csv_path))
    total = len(rows)
    if total == 0:
        print("Error: No captions found in CSV")
        sys.exit(1)

    # 校验范围
    start_id = max(1, args.start_id)
    end_id = min(total, args.end_id)
    if start_id > end_id:
        print(f"Error: start_id ({start_id}) > end_id ({end_id}) or out of range [1, {total}]")
        sys.exit(1)

    print("=" * 80)
    print(f"Batch PINN Inference: IDs {start_id}-{end_id} / {total} total")
    print(f"CSV: {csv_path}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # 1. 加载模型（只加载一次）
    print("\n[1/3] Loading model...")
    pipe = PhysicsInformedWanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(
                model_id=args.model_id,
                origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id=args.model_id,
                origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id=args.model_id,
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id=args.model_id,
                origin_file_pattern="Wan2.1_VAE.pth",
                offload_device="cpu",
            ),
        ],
    )
    pipe.enable_vram_management()

    print(f"[2/3] Loading PINN plugin from {args.checkpoint_path}...")
    pipe.load_pinn_plugin(args.checkpoint_path, device=args.device)
    global_metadata, global_mode = load_global_metadata(
        metadata_json=args.metadata_json,
        metadata_csv=args.metadata_csv,
    )
    if global_metadata is not None:
        print(f"  Using global PINN metadata (mode={global_mode}).")

    ids_to_process: Iterable[int]
    if args.resume:
        existing_ids = scan_existing_ids(output_dir)
        ids_to_process = [i for i in range(start_id, end_id + 1) if i not in existing_ids]
        print(
            f"[3/3] Resume enabled: existing {len(existing_ids)} files, "
            f"remaining {len(ids_to_process)} in range"
        )
        if not ids_to_process:
            print("All done in range. Nothing to generate.")
            return
    else:
        ids_to_process = range(start_id, end_id + 1)

    print(f"[3/3] Generating videos {start_id}-{end_id}...")
    for vid_id in ids_to_process:
        idx = vid_id - 1  # 0-based index
        row = rows[idx]
        caption = row.get("caption", "").strip()
        if caption.startswith('"') and caption.endswith('"'):
            caption = caption[1:-1]
        if caption == "":
            caption = row.get("prompt", "").strip()
        out_name = f"{vid_id:04d}.mp4"
        out_path = output_dir / out_name

        if args.skip_existing and out_path.exists():
            print(f"  [{vid_id:4d}/{total}] Skip (exists): {out_name}")
            continue

        seed = 0
        pinn_metadata = global_metadata
        metadata_mode = global_mode
        if pinn_metadata is None:
            pinn_metadata, metadata_mode = metadata_from_row(row)
        print(f"  [{vid_id:4d}/{total}] {caption[:60]}{'...' if len(caption) > 60 else ''} -> {out_name}")
        if pinn_metadata is not None:
            print(f"    metadata mode: {metadata_mode}")

        try:
            video = pipe(
                prompt=caption,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                seed=seed,
                cfg_scale=args.cfg_scale,
                tiled=True,
                pinn_metadata=pinn_metadata,
            )
            save_video(video, str(out_path), fps=args.fps, quality=args.quality)
        except Exception as e:
            print(f"  [{vid_id:4d}/{total}] ERROR: {e}")
            continue

    print("\n" + "=" * 80)
    print(f"Done: {start_id}-{end_id}")
    print("=" * 80)


if __name__ == "__main__":
    main()
