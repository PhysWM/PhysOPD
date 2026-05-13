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
from auto_label_utils import (
    PromptPhysicsLabelInferer,
    PromptVideoPromptRefiner,
    build_minimal_label_metadata,
    prompt_preview,
)


PHENOMENON_LABELS = [
    # 10 physics-based categories aligned with Table 1
    "Rigid Body", "Elastic", "Fluid", "Compressible Flow", "Phase Change",
    "Collision/Contact", "Granular", "Fracture", "Thermal", "Optical",
]
PHENOMENON_TO_ID = {name: idx for idx, name in enumerate(PHENOMENON_LABELS)}
PHENOMENON_ALIAS = {}


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
    """Normalize label: strip whitespace and standardize spacing."""
    clean = _safe_text(label)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


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
    encoded_keys = {"label_id", "label_ids", "n_numeric", "n_text_ids", "q_vector"}
    return any(key in metadata for key in encoded_keys)


def encode_raw_metadata(raw_metadata, n_text_vocab_size=2048, q_dim=64):
    if not isinstance(raw_metadata, dict):
        return None
    label_name = _normalize_label(raw_metadata.get("label", raw_metadata.get("label_name", "")))
    label_id = raw_metadata.get("label_id")

    # 支持多标签解析（逗号分隔）
    if label_id is None:
        label_ids = []
        for part in label_name.split(","):
            part = part.strip()
            if part in PHENOMENON_TO_ID:
                label_ids.append(PHENOMENON_TO_ID[part])
        if not label_ids:
            label_ids = [PHENOMENON_TO_ID["Fluid"]]  # 默认标签
        primary_label_id = label_ids[0]
    else:
        # 如果提供了 label_id，可能是单个值或逗号分隔的字符串
        if isinstance(label_id, str):
            label_ids = [int(x.strip()) for x in label_id.split(",") if x.strip()]
            primary_label_id = label_ids[0]
        else:
            label_ids = [int(label_id)]
            primary_label_id = int(label_id)

    label_id = primary_label_id

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
        "label_ids": label_ids,  # 多标签列表
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
        if isinstance(metadata.get("label_ids"), str):
            metadata["label_ids"] = _parse_vector(metadata.get("label_ids"), cast_type=int)
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


def build_wan_model_configs(model_id):
    text_encoder = ModelConfig(
        model_id=model_id,
        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
        offload_device="cpu",
    )
    vae = ModelConfig(
        model_id=model_id,
        origin_file_pattern="Wan2.1_VAE.pth",
        offload_device="cpu",
    )

    if "Wan2.1" in model_id:
        return [
            ModelConfig(
                model_id=model_id,
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
            ),
            text_encoder,
            vae,
        ], "single_expert"

    return [
        ModelConfig(
            model_id=model_id,
            origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
            offload_device="cpu",
        ),
        ModelConfig(
            model_id=model_id,
            origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
            offload_device="cpu",
        ),
        text_encoder,
        vae,
    ], "dual_expert"


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
    export_expert_attention=False,
    expert_attention_topk=4,
    expert_attention_num_frames=2,
    expert_attention_apply_router_weight=True,
    observable_inspection_only=False,
    pinn_step_range=None,
    metadata_json=None,
    metadata_csv=None,
    auto_label_from_prompt=False,
    refine_prompt_with_llm=True,
    llm_model=None,
    llm_base_url=None,
    llm_api_key=None,
    llm_api_key_env="OPENAI_API_KEY",
    llm_timeout=30.0,
    llm_max_retries=2,
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
    model_configs, loader_mode = build_wan_model_configs(model_id)
    print(f"  Model loader mode: {loader_mode}")
    pipe = PhysicsInformedWanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=model_configs,
    )
    pipe.enable_vram_management()
    
    # 2. 加载 PINN plugin（如果有）
    if checkpoint_path:
        print(f"\n[2/4] Loading PINN plugin from {checkpoint_path}...")
        pipe.load_pinn_plugin(
            checkpoint_path,
            device=device,
            enable_tracking=True,
            observable_inspection_only=observable_inspection_only,
            export_expert_attention=export_expert_attention,
            expert_attention_apply_router_weight=expert_attention_apply_router_weight,
            pinn_step_range=pinn_step_range,
        )
        print(f"PINN step range: {pinn_step_range or 'all'}")
        if observable_inspection_only:
            print("PINN observable inspection active: Wan denoising runs normally, encoder diagnostics are recorded without applying adapter correction")
        else:
            print("PINN plugin active: every denoising step will apply PhysicsAdapter + PDE检验")
    else:
        print("\n[2/4] No PINN plugin provided, using original model")
    
    # 3. 使用 LLM 细化 prompt（评测时对粗糙 prompt 更稳，失败则回退原 prompt）
    original_prompt = prompt
    effective_prompt = prompt
    prompt_refinement = None
    if refine_prompt_with_llm:
        refiner = PromptVideoPromptRefiner(
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            api_key_env=llm_api_key_env,
            timeout=llm_timeout,
            max_retries=llm_max_retries,
        )
        prompt_refinement = refiner.refine(prompt)
        effective_prompt = prompt_refinement.get("refined_prompt") or prompt
        if prompt_refinement.get("used_refinement"):
            print(
                "\n[3/5] Refined prompt via LLM: "
                f"original='{prompt_preview(original_prompt, limit=120)}', "
                f"refined='{prompt_preview(effective_prompt, limit=160)}'"
            )
        else:
            print("\n[3/5] Prompt refinement unavailable or unchanged, using original prompt")
            if prompt_refinement.get("error"):
                print(f"       LLM warning: {prompt_refinement['error']}")
    else:
        print("\n[3/5] Prompt refinement disabled, using original prompt")

    # 4. 加载元数据（包含 phenomenon label 用于 MoE 路由）
    pinn_metadata, metadata_mode = load_inference_metadata(
        metadata_json=metadata_json,
        metadata_csv=metadata_csv,
    )
    llm_result = None
    if pinn_metadata is None and auto_label_from_prompt:
        inferer = PromptPhysicsLabelInferer(
            PHENOMENON_LABELS,
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
            api_key_env=llm_api_key_env,
            timeout=llm_timeout,
            max_retries=llm_max_retries,
            default_label="Fluid",
        )
        llm_result = inferer.infer(effective_prompt)
        pinn_metadata = build_minimal_label_metadata(
            llm_result["labels"],
            PHENOMENON_TO_ID,
            default_label="Fluid",
        )
        metadata_mode = "llm_auto_label"
        routing_prompt_source = "refined_prompt" if effective_prompt != original_prompt else "original_prompt"
        print(
            "[4/5] Auto-labeled prompt via LLM: "
            f"{routing_prompt_source}='{prompt_preview(effective_prompt)}', raw_labels={llm_result['raw_labels']}, "
            f"final_labels={llm_result['labels']}, label_ids={pinn_metadata.get('label_ids')}, "
            f"fallback={llm_result['fallback_used']}"
        )
        if llm_result.get("error"):
            print(f"       LLM warning: {llm_result['error']}")
    if pinn_metadata is not None:
        print(f"[4/5] Using PINN metadata for MoE routing (mode={metadata_mode}).")
        phenomenon = pinn_metadata.get("label_name", "Fluid")
    else:
        print(f"[4/5] No PINN metadata provided, using default phenomenon")
        phenomenon = "Fluid"

    # 5. 生成视频（同时实时追踪每步的物理场 PDE 残差）
    print(f"\n[5/5] Generating video with real-time physics tracking...")
    if effective_prompt != original_prompt:
        print(f"  Original prompt: {original_prompt}")
        print(f"  Effective prompt: {effective_prompt}")
    else:
        print(f"  Prompt: {effective_prompt}")
    print(f"  Phenomenon: {phenomenon}")
    print(f"  Resolution: {height}x{width}, Frames: {num_frames}")
    print(f"  Steps: {num_inference_steps}, CFG Scale: {cfg_scale}")

    pipe.reset_tracking()

    video = pipe(
        prompt=effective_prompt,
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
            export_expert_attention=export_expert_attention,
            expert_attention_topk=expert_attention_topk,
            expert_attention_num_frames=expert_attention_num_frames,
            expert_attention_prompt=effective_prompt,
            expert_attention_apply_router_weight=expert_attention_apply_router_weight,
        )
        if observable_inspection_only:
            pipe.save_observable_report(
                str(Path(output_path).with_suffix("")),
                video_frames=video,
                fps=fps,
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
        help="Disable correction attribution overlay MP4/PNG generation in the physics report",
    )
    parser.add_argument(
        "--attention_alpha",
        type=float,
        default=0.45,
        help="Overlay alpha for correction attribution visualization (0~1)",
    )
    parser.add_argument(
        "--disable_motion_weighted_attention",
        action="store_true",
        help="Deprecated compatibility flag; reports now use raw correction attribution as the canonical cause map",
    )
    parser.add_argument(
        "--attention_motion_percentile",
        type=float,
        default=90.0,
        help="Percentile threshold for the auxiliary motion-weighted correction view (default: 90)",
    )
    parser.add_argument(
        "--export_expert_attention",
        action="store_true",
        help="Export per-active-expert spatial correction attribution into trace NPZ and PNG/PDF grid.",
    )
    parser.add_argument(
        "--expert_attention_topk",
        type=int,
        default=4,
        help="Maximum active experts to show in the expert attention grid.",
    )
    parser.add_argument(
        "--expert_attention_num_frames",
        type=int,
        default=2,
        help="Number of evenly sampled frames to show in the expert attention grid.",
    )
    parser.add_argument(
        "--expert_attention_unweighted",
        action="store_true",
        help="Visualize each active expert's raw correction response without multiplying by its router weight.",
    )
    parser.add_argument(
        "--observable_inspection_only",
        action="store_true",
        help="Run full Wan inference but only record encoder observable diagnostics; do not apply adapter correction to v_original.",
    )
    parser.add_argument(
        "--pinn_step_range",
        type=str,
        default=None,
        help="1-indexed inclusive denoising steps where PINN correction is active, e.g. 41-50, 31-50, 1-50, none.",
    )
    parser.add_argument("--metadata_json", type=str, default=None, help="Metadata JSON string or JSON file path")
    parser.add_argument("--metadata_csv", type=str, default=None, help="CSV file path with label_id/label_name/n_numeric/n_text_ids/q_vector")
    parser.add_argument(
        "--auto_label_from_prompt",
        action="store_true",
        help="Call an OpenAI-compatible LLM API to infer routing labels when metadata is absent.",
    )
    parser.add_argument(
        "--disable_prompt_refinement",
        action="store_true",
        help="Skip the best-effort LLM prompt refinement step and use the original prompt directly.",
    )
    parser.add_argument("--llm_model", type=str, default="gpt-5.4", help="LLM model name. Falls back to OPENAI_MODEL or LLM_MODEL.")
    parser.add_argument("--llm_base_url", type=str, default="http://14.103.68.46/v1/chat/completions", help="OpenAI-compatible base URL. Falls back to OPENAI_BASE_URL or LLM_BASE_URL.")
    parser.add_argument("--llm_api_key", type=str, default="sk-0vyxxFvLvTYSH1GQA2YmtS3pUH4kQcOO2h6TRcT0FDy64NxB", help="Optional direct API key. Prefer env vars for security.")
    parser.add_argument("--llm_api_key_env", type=str, default="OPENAI_API_KEY", help="Environment variable name that stores the API key.")
    parser.add_argument("--llm_timeout", type=float, default=30.0, help="LLM request timeout in seconds.")
    parser.add_argument("--llm_max_retries", type=int, default=2, help="Maximum retry count for LLM requests.")
    
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
        export_expert_attention=args.export_expert_attention,
        expert_attention_topk=args.expert_attention_topk,
        expert_attention_num_frames=args.expert_attention_num_frames,
        expert_attention_apply_router_weight=not args.expert_attention_unweighted,
        observable_inspection_only=args.observable_inspection_only,
        pinn_step_range=args.pinn_step_range,
        metadata_json=args.metadata_json,
        metadata_csv=args.metadata_csv,
        auto_label_from_prompt=args.auto_label_from_prompt,
        refine_prompt_with_llm=not args.disable_prompt_refinement,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_api_key_env=args.llm_api_key_env,
        llm_timeout=args.llm_timeout,
        llm_max_retries=args.llm_max_retries,
        device=args.device,
    )
