#!/usr/bin/env python3
"""
Batch PhysGenBench inference for AnyFlow + PILA PhysicsAdapter.

Each process owns one visible GPU, loads the AnyFlow pipeline once, then
iterates over a CSV id range. Outputs are named 0001.mp4, 0002.mp4, ...
to match PhysGenBench-style evaluation scripts.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from diffusers.utils import export_to_video

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from inference_anyflow_pinn import (  # noqa: E402
    DEFAULT_ANYFLOW_ROOT,
    DEFAULT_PINN_CKPT,
    DEFAULT_PILA_ROOT,
    attach_adapter_to_bidirectional_anyflow,
    load_physics_adapter,
    load_pila_adapter_modules,
    resolve_prompt_and_metadata,
)


PHENOMENON_LABELS = [
    "Rigid Body", "Elastic", "Fluid", "Compressible Flow", "Phase Change",
    "Collision/Contact", "Granular", "Fracture", "Thermal", "Optical",
]


def cuda_synchronize(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def reset_cuda_peak_memory(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def cuda_memory_gb(device):
    if not (torch.cuda.is_available() and str(device).startswith("cuda")):
        return None, None
    return (
        float(torch.cuda.max_memory_allocated()) / 1e9,
        float(torch.cuda.max_memory_reserved()) / 1e9,
    )


def append_jsonl(path: Path | None, record: dict[str, Any]):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def extract_prompt(row: dict[str, str]) -> str:
    for key in ("prompt", "caption"):
        value = str(row.get(key, "") or "").strip()
        if value:
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            return value
    return ""


def source_sample_id(row: dict[str, str], fallback_id: int) -> int:
    for key in ("source_sample_id", "original_sample_id", "phygenbench_id", "source_id"):
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        try:
            return int(value)
        except ValueError:
            pass
    return int(fallback_id)


def performance_record_exists(path: Path | None, sample_id: int) -> bool:
    if path is None or not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(record.get("sample_id", -1)) == int(sample_id) and record.get("success") is True:
                    return True
    except OSError:
        return False
    return False


def build_per_sample_args(args, prompt: str):
    values = vars(args).copy()
    values["prompt"] = prompt
    return SimpleNamespace(**values)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/home/dataset-assist-0/algorithm/cong.wang/DiffSynth-Studio/phygenbench_prompts.csv")
    parser.add_argument("--start_id", type=int, default=1)
    parser.add_argument("--end_id", type=int, default=None)
    parser.add_argument("--output_dir", default="outputs/phygenbench_anyflow_pila")
    parser.add_argument("--performance_metrics_path", default=None)
    parser.add_argument("--anyflow_root", default=DEFAULT_ANYFLOW_ROOT)
    parser.add_argument("--pila_root", default=DEFAULT_PILA_ROOT)
    parser.add_argument(
        "--model_path",
        default=f"{DEFAULT_ANYFLOW_ROOT}/experiments/pretrained_models/AnyFlow-Wan2.1-T2V-1.3B-Diffusers",
    )
    parser.add_argument("--pinn_checkpoint", default=DEFAULT_PINN_CKPT)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--metadata_json", default=None)
    parser.add_argument("--auto_label_from_prompt", action="store_true")
    parser.add_argument("--disable_prompt_refinement", action="store_true")
    parser.add_argument("--llm_model", default="gpt-5.4")
    parser.add_argument("--llm_base_url", default="http://35.220.164.252:3888/v1")
    parser.add_argument("--llm_api_key", default="sk-8viAj2SPNHZ4W0E4BcKSfdOwXr1xVzpcheUHDIPweBi4EEqB")
    parser.add_argument("--llm_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--llm_timeout", type=float, default=30.0)
    parser.add_argument("--llm_max_retries", type=int, default=2)
    parser.add_argument("--default_label", default="Fluid")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--correction_scale", type=float, default=1.0)
    parser.add_argument("--moe_top_k", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    performance_path = (
        Path(args.performance_metrics_path)
        if args.performance_metrics_path
        else output_dir / "performance_metrics.jsonl"
    )

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not Path(args.pinn_checkpoint).exists():
        raise FileNotFoundError(f"PINN checkpoint not found: {args.pinn_checkpoint}")
    if "AnyFlow-FAR" in args.model_path:
        raise NotImplementedError("Batch script currently targets bidirectional AnyFlow-Wan models.")

    rows = load_rows(csv_path)
    total = len(rows)
    end_id = total if args.end_id is None else min(args.end_id, total)
    start_id = max(1, args.start_id)
    if start_id > end_id:
        raise ValueError(f"Invalid id range: {start_id}-{end_id}, total={total}")

    sys.path.insert(0, args.anyflow_root)
    from far.pipelines.pipeline_wan_anyflow import WanAnyFlowPipeline

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    contracts, adapter_mod = load_pila_adapter_modules(args.pila_root)
    adapter, adapter_config = load_physics_adapter(
        args.pinn_checkpoint,
        contracts=contracts,
        adapter_mod=adapter_mod,
        device=args.device,
        dtype=dtype,
        moe_top_k_override=args.moe_top_k,
    )

    print("=" * 80)
    print("Batch AnyFlow + PILA PhysGenBench inference")
    print(f"CSV: {csv_path}")
    print(f"IDs: {start_id}-{end_id} / {total}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model_path}")
    print(f"Checkpoint: {args.pinn_checkpoint}")
    print(f"Adapter core: {adapter_config.get('core_ablation_mode')} moe_top_k={adapter.moe_top_k}")
    print(f"Prompt refinement: {not args.disable_prompt_refinement}")
    print(f"Auto label routing: {args.auto_label_from_prompt}")
    print("=" * 80)

    pipe = WanAnyFlowPipeline.from_pretrained(args.model_path).to(args.device, dtype=dtype)

    for sample_id in range(start_id, end_id + 1):
        row = rows[sample_id - 1]
        prompt = extract_prompt(row)
        out_path = output_dir / f"{sample_id:04d}.mp4"
        source_id = source_sample_id(row, sample_id)

        if args.skip_existing and out_path.exists():
            print(f"[{sample_id:4d}/{total}] skip existing: {out_path.name}")
            if not performance_record_exists(performance_path, sample_id):
                append_jsonl(performance_path, {
                    "sample_id": sample_id,
                    "source_sample_id": source_id,
                    "output_path": str(out_path),
                    "skipped": True,
                    "success": True,
                    "error": None,
                })
            continue
        if args.resume and performance_record_exists(performance_path, sample_id):
            print(f"[{sample_id:4d}/{total}] resume skip successful record: {out_path.name}")
            continue

        per_args = build_per_sample_args(args, prompt)
        effective_prompt = prompt
        metadata = None
        metadata_mode = None
        generation_seconds = None
        save_seconds = None
        total_seconds = None
        peak_allocated_gb = None
        peak_reserved_gb = None
        total_start = time.perf_counter()

        try:
            effective_prompt, metadata, metadata_mode = resolve_prompt_and_metadata(
                per_args,
                contracts.PHENOMENON_LABELS,
            )
            print(
                f"[{sample_id:4d}/{total}] {prompt[:70]}"
                f"{'...' if len(prompt) > 70 else ''} -> {out_path.name}"
            )
            print(
                f"    metadata_mode={metadata_mode}, "
                f"label_ids={metadata.get('label_ids')}, label_name={metadata.get('label_name')}"
            )

            attach_adapter_to_bidirectional_anyflow(
                pipe,
                adapter=adapter,
                metadata=metadata,
                correction_scale=args.correction_scale,
            )
            generator = torch.Generator(args.device).manual_seed(args.seed)

            cuda_synchronize(args.device)
            reset_cuda_peak_memory(args.device)
            generation_start = time.perf_counter()
            result = pipe(
                prompt=effective_prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            cuda_synchronize(args.device)
            generation_seconds = time.perf_counter() - generation_start

            save_start = time.perf_counter()
            export_to_video(result.frames[0], output_video_path=str(out_path), fps=args.fps)
            cuda_synchronize(args.device)
            save_seconds = time.perf_counter() - save_start
            total_seconds = time.perf_counter() - total_start
            peak_allocated_gb, peak_reserved_gb = cuda_memory_gb(args.device)

            stats = getattr(pipe, "_pila_adapter_stats", None) or []
            correction_mean = float(sum(stats) / len(stats)) if stats else None
            correction_max = float(max(stats)) if stats else None
            print(
                f"    done: generation={generation_seconds:.2f}s, "
                f"correction_mean={correction_mean}"
            )
            append_jsonl(performance_path, {
                "sample_id": sample_id,
                "source_sample_id": source_id,
                "output_path": str(out_path),
                "skipped": False,
                "success": True,
                "error": None,
                "prompt": prompt,
                "effective_prompt": effective_prompt,
                "metadata_mode": metadata_mode,
                "metadata_label_ids": metadata.get("label_ids") if isinstance(metadata, dict) else None,
                "metadata_label_name": metadata.get("label_name") if isinstance(metadata, dict) else None,
                "model_path": args.model_path,
                "checkpoint_path": args.pinn_checkpoint,
                "moe_top_k": int(adapter.moe_top_k),
                "num_inference_steps": int(args.num_inference_steps),
                "guidance_scale": float(args.guidance_scale),
                "seed": int(args.seed),
                "height": int(args.height),
                "width": int(args.width),
                "num_frames": int(args.num_frames),
                "fps": int(args.fps),
                "gpu_id": os.getenv("CUDA_VISIBLE_DEVICES", ""),
                "cuda_device_index": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
                "generation_seconds": generation_seconds,
                "save_seconds": save_seconds,
                "total_seconds": total_seconds,
                "peak_memory_allocated_gb": peak_allocated_gb,
                "peak_memory_reserved_gb": peak_reserved_gb,
                "adapter_correction_ratio_mean": correction_mean,
                "adapter_correction_ratio_max": correction_max,
            })
        except Exception as exc:
            try:
                cuda_synchronize(args.device)
                peak_allocated_gb, peak_reserved_gb = cuda_memory_gb(args.device)
            except Exception:
                pass
            total_seconds = time.perf_counter() - total_start
            append_jsonl(performance_path, {
                "sample_id": sample_id,
                "source_sample_id": source_id,
                "output_path": str(out_path),
                "skipped": False,
                "success": False,
                "error": str(exc),
                "prompt": prompt,
                "effective_prompt": effective_prompt,
                "metadata_mode": metadata_mode,
                "metadata_label_ids": metadata.get("label_ids") if isinstance(metadata, dict) else None,
                "metadata_label_name": metadata.get("label_name") if isinstance(metadata, dict) else None,
                "model_path": args.model_path,
                "checkpoint_path": args.pinn_checkpoint,
                "gpu_id": os.getenv("CUDA_VISIBLE_DEVICES", ""),
                "cuda_device_index": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
                "total_seconds": total_seconds,
                "peak_memory_allocated_gb": peak_allocated_gb,
                "peak_memory_reserved_gb": peak_reserved_gb,
            })
            print(f"[{sample_id:4d}/{total}] ERROR: {exc}")
            if not args.continue_on_error:
                raise

    print(f"Done: {start_id}-{end_id}")


if __name__ == "__main__":
    main()
