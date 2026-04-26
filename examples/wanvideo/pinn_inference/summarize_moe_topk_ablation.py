#!/usr/bin/env python3
"""Summarize MoE active expert count ablation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Optional


DEFAULT_K_VALUES = "0,1,2,3,4,5,6,7,8"
SCORE_COLUMNS = ("Mechanics", "Optics", "Thermal", "Material", "Avg")


def parse_k_values(text: str) -> list[int]:
    values = []
    for part in str(text or "").replace(",", " ").split():
        values.append(int(part))
    return values


def numeric(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value:
        return None
    return value


def read_jsonl_by_sample(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = record.get("sample_id")
            if sample_id is None:
                continue
            records[int(sample_id)] = record
    return records


def read_scores(path: Path) -> dict[str, Optional[float]]:
    scores = {key: None for key in SCORE_COLUMNS}
    if not path.exists():
        return scores
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return scores
    table = ((payload.get("scores") or {}).get("paper_table") or {})
    for key in SCORE_COLUMNS:
        scores[key] = numeric(table.get(key))
    return scores


def mean_or_none(values: list[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def median_or_none(values: list[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def fmt(value):
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def resolve_result_dir(phygenbench_root: Path, result_dir: str) -> Path:
    path = Path(result_dir)
    if path.is_absolute():
        return path
    return phygenbench_root / path


def build_rows(args) -> list[dict]:
    output_root = Path(args.output_root).resolve()
    phygenbench_root = Path(args.phygenbench_root).resolve()
    result_dir = resolve_result_dir(phygenbench_root, args.result_dir)
    expected_samples = int(args.expected_samples)
    rows = []
    for k in parse_k_values(args.k_values):
        video_dir = output_root / f"topk_{k}"
        perf_path = video_dir / "performance_metrics.jsonl"
        records = read_jsonl_by_sample(perf_path)
        success_records = [
            record
            for record in records.values()
            if record.get("success") is True and not record.get("skipped")
        ]
        total_seconds = [
            value for value in (numeric(record.get("total_seconds")) for record in success_records)
            if value is not None
        ]
        generation_seconds = [
            value for value in (numeric(record.get("generation_seconds")) for record in success_records)
            if value is not None
        ]
        peak_memory_values = [
            numeric(record.get("peak_memory_reserved_gb"))
            or numeric(record.get("peak_memory_allocated_gb"))
            for record in success_records
        ]
        peak_memory_values = [value for value in peak_memory_values if value is not None]
        video_count = len(list(video_dir.glob("*.mp4"))) if video_dir.exists() else 0
        summary_path = result_dir / f"{args.model_prefix}{k}_closed_summary.json"
        scores = read_scores(summary_path)
        status = []
        if video_count != expected_samples:
            status.append(f"videos={video_count}/{expected_samples}")
        if len(records) != expected_samples:
            status.append(f"perf_records={len(records)}/{expected_samples}")
        if not summary_path.exists():
            status.append("missing_eval_summary")
        elif any(scores[key] is None for key in SCORE_COLUMNS):
            status.append("incomplete_eval_scores")
        row = {
            "K": k,
            "success_sample_count": video_count,
            "performance_record_count": len(records),
            "avg_total_seconds": mean_or_none(total_seconds),
            "median_total_seconds": median_or_none(total_seconds),
            "avg_generation_seconds": mean_or_none(generation_seconds),
            "peak_memory_gb": max(peak_memory_values) if peak_memory_values else None,
            **scores,
            "status": "ok" if not status else "; ".join(status),
            "video_dir": str(video_dir),
            "performance_metrics": str(perf_path),
            "eval_summary": str(summary_path),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "K",
        "success_sample_count",
        "performance_record_count",
        "avg_total_seconds",
        "median_total_seconds",
        "avg_generation_seconds",
        "peak_memory_gb",
        *SCORE_COLUMNS,
        "status",
        "video_dir",
        "performance_metrics",
        "eval_summary",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})


def write_markdown(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "K",
        "Success",
        "Avg Total(s)",
        "Median Total(s)",
        "Avg Gen(s)",
        "Peak Mem(GB)",
        *SCORE_COLUMNS,
        "Status",
    ]
    keys = [
        "K",
        "success_sample_count",
        "avg_total_seconds",
        "median_total_seconds",
        "avg_generation_seconds",
        "peak_memory_gb",
        *SCORE_COLUMNS,
        "status",
    ]
    lines = [
        "# MoE Top-K Ablation Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for key in keys) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--phygenbench_root",
        default="/home/dataset-assist-0/algorithm/cong.wang/projects/PhyGenBench",
    )
    parser.add_argument("--result_dir", default="result/moe_topk_ablation")
    parser.add_argument("--model_prefix", default="wan21_pinn8_topk")
    parser.add_argument("--k_values", default=DEFAULT_K_VALUES)
    parser.add_argument("--expected_samples", type=int, default=160)
    parser.add_argument("--output_prefix", default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_prefix = Path(args.output_prefix).resolve() if args.output_prefix else output_root / "moe_topk_ablation_summary"
    rows = build_rows(args)
    write_csv(output_prefix.with_suffix(".csv"), rows)
    write_markdown(output_prefix.with_suffix(".md"), rows)
    print(f"Wrote {output_prefix.with_suffix('.csv')}")
    print(f"Wrote {output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
