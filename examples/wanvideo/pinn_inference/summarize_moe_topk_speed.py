#!/usr/bin/env python3
"""Summarize controlled MoE top-k speed benchmark metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_k_values(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split() if part.strip()]


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * fraction)
    return ordered[idx]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "q1": None,
            "q3": None,
            "iqr": None,
        }
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p10": quantile(values, 0.10),
        "p90": quantile(values, 0.90),
        "q1": q1,
        "q3": q3,
        "iqr": (q3 - q1) if q1 is not None and q3 is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=Path("phygenbench_moe_topk_speed_benchmark/fullpinn8_step18500"))
    parser.add_argument("--k_values", type=str, default="0 1 2 3 4 5 6 7 8")
    parser.add_argument("--expected_samples", type=int, default=32)
    parser.add_argument("--expected_repeats", type=int, default=3)
    parser.add_argument("--baseline_k", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--output_prefix", type=Path, default=None)
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    output_prefix = args.output_prefix or args.output_root / "moe_topk_speed_summary"
    summaries: list[dict[str, Any]] = []

    for k in k_values:
        metrics_paths = sorted((args.output_root / f"topk_{k}").glob("repeat_*/performance_metrics.jsonl"))
        records: list[dict[str, Any]] = []
        warmup_count = 0
        skipped_count = 0
        error_count = 0
        for path in metrics_paths:
            for record in load_records(path):
                if record.get("warmup") or record.get("benchmark_phase") == "warmup":
                    warmup_count += 1
                    continue
                if record.get("skipped"):
                    skipped_count += 1
                    continue
                if not record.get("success", False):
                    error_count += 1
                    continue
                if record.get("generation_seconds") is None:
                    continue
                records.append(record)

        gen = [float(row["generation_seconds"]) for row in records if row.get("generation_seconds") is not None]
        total = [float(row["total_seconds"]) for row in records if row.get("total_seconds") is not None]
        save = [float(row["save_seconds"]) for row in records if row.get("save_seconds") is not None]
        reserved = [float(row["peak_memory_reserved_gb"]) for row in records if row.get("peak_memory_reserved_gb") is not None]
        allocated = [float(row["peak_memory_allocated_gb"]) for row in records if row.get("peak_memory_allocated_gb") is not None]
        repeats = sorted({row.get("benchmark_repeat") for row in records if row.get("benchmark_repeat") is not None})
        source_ids = sorted({int(row.get("source_sample_id", row.get("sample_id"))) for row in records})
        gen_stats = summarize_values(gen)
        total_stats = summarize_values(total)
        save_stats = summarize_values(save)
        expected_records = args.expected_samples * args.expected_repeats
        status = []
        if len(gen) != expected_records:
            status.append(f"expected_{expected_records}_records_got_{len(gen)}")
        if len(repeats) != args.expected_repeats:
            status.append(f"expected_{args.expected_repeats}_repeats_got_{len(repeats)}")
        if len(source_ids) != args.expected_samples:
            status.append(f"expected_{args.expected_samples}_samples_got_{len(source_ids)}")
        if skipped_count:
            status.append(f"skipped_{skipped_count}")
        if error_count:
            status.append(f"errors_{error_count}")
        if not metrics_paths:
            status.append("missing_metrics")
        summaries.append({
            "K": k,
            "valid_record_count": len(gen),
            "repeat_count": len(repeats),
            "sample_count": len(source_ids),
            "warmup_record_count": warmup_count,
            "mean_generation_seconds": gen_stats["mean"],
            "median_generation_seconds": gen_stats["median"],
            "p10_generation_seconds": gen_stats["p10"],
            "p90_generation_seconds": gen_stats["p90"],
            "iqr_generation_seconds": gen_stats["iqr"],
            "mean_total_seconds": total_stats["mean"],
            "median_total_seconds": total_stats["median"],
            "mean_save_seconds": save_stats["mean"],
            "peak_memory_allocated_gb": max(allocated) if allocated else None,
            "peak_memory_reserved_gb": max(reserved) if reserved else None,
            "videos_per_hour_per_gpu": (3600.0 / gen_stats["mean"]) if gen_stats["mean"] else None,
            "output_fps": (args.num_frames / gen_stats["mean"]) if gen_stats["mean"] else None,
            "frame_steps_per_second": (
                args.num_frames * args.num_inference_steps / gen_stats["mean"]
            ) if gen_stats["mean"] else None,
            "seconds_per_video_step": (gen_stats["mean"] / args.num_inference_steps) if gen_stats["mean"] else None,
            "status": "ok" if not status else ";".join(status),
            "metrics_files": len(metrics_paths),
        })

    baseline = next((row for row in summaries if row["K"] == args.baseline_k), None)
    baseline_mean = baseline.get("mean_generation_seconds") if baseline else None
    baseline_median = baseline.get("median_generation_seconds") if baseline else None
    for row in summaries:
        mean_gen = row.get("mean_generation_seconds")
        median_gen = row.get("median_generation_seconds")
        row["relative_speedup_vs_k4_mean"] = (baseline_mean / mean_gen) if baseline_mean and mean_gen else None
        row["relative_speedup_vs_k4_median"] = (baseline_median / median_gen) if baseline_median and median_gen else None

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    fieldnames = [
        "K",
        "valid_record_count",
        "repeat_count",
        "sample_count",
        "warmup_record_count",
        "mean_generation_seconds",
        "median_generation_seconds",
        "p10_generation_seconds",
        "p90_generation_seconds",
        "iqr_generation_seconds",
        "mean_total_seconds",
        "median_total_seconds",
        "mean_save_seconds",
        "peak_memory_allocated_gb",
        "peak_memory_reserved_gb",
        "videos_per_hour_per_gpu",
        "output_fps",
        "frame_steps_per_second",
        "seconds_per_video_step",
        "relative_speedup_vs_k4_mean",
        "relative_speedup_vs_k4_median",
        "status",
        "metrics_files",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# MoE Top-K Speed Summary\n\n")
        f.write(f"- Output root: `{args.output_root}`\n")
        f.write(f"- Expected measured records per K: `{args.expected_samples} x {args.expected_repeats}`\n")
        f.write(f"- Baseline for relative speedup: `K={args.baseline_k}`\n\n")
        f.write("| K | Valid | Mean Gen (s) | Output FPS | Frame-Steps/s | Videos/h/GPU | Peak Reserved GB | Speedup vs K4 | Status |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in summaries:
            f.write(
                "| {K} | {valid} | {mean} | {fps} | {frame_steps} | {vph} | {mem} | {speedup} | {status} |\n".format(
                    K=row["K"],
                    valid=row["valid_record_count"],
                    mean=fmt(row["mean_generation_seconds"]),
                    fps=fmt(row["output_fps"]),
                    frame_steps=fmt(row["frame_steps_per_second"]),
                    vph=fmt(row["videos_per_hour_per_gpu"]),
                    mem=fmt(row["peak_memory_reserved_gb"]),
                    speedup=fmt(row["relative_speedup_vs_k4_mean"]),
                    status=row["status"],
                )
            )

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
