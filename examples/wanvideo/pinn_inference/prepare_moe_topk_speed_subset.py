#!/usr/bin/env python3
"""Prepare a fixed PhyGenBench subset for MoE top-k speed benchmarking."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_QUOTAS = "solid_solid=14,solid_fluid=10,fluid_fluid=8"


def parse_quotas(text: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid quota entry {part!r}; expected name=count.")
        name, value = part.split("=", 1)
        quotas[name.strip()] = int(value.strip())
    if not quotas:
        raise ValueError("At least one quota is required.")
    return quotas


def read_metrics(path: Path) -> dict[int, dict[str, Any]]:
    metrics: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("skipped") or not record.get("success", False):
                continue
            sample_id = int(record["sample_id"])
            metrics[sample_id] = record
    return metrics


def balanced_take(rows: list[dict[str, Any]], quota: int, rng: random.Random) -> list[dict[str, Any]]:
    by_complexity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_complexity[str(row.get("complexity", ""))].append(row)
    for values in by_complexity.values():
        values.sort(key=lambda item: int(item["source_sample_id"]))
        rng.shuffle(values)

    complexities = sorted(by_complexity)
    base = quota // max(len(complexities), 1)
    remainder = quota % max(len(complexities), 1)
    targets = {name: base + (1 if idx < remainder else 0) for idx, name in enumerate(complexities)}

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    shortfall = 0
    for name in complexities:
        take = min(targets[name], len(by_complexity[name]))
        chosen = by_complexity[name][:take]
        selected.extend(chosen)
        selected_ids.update(int(row["source_sample_id"]) for row in chosen)
        shortfall += targets[name] - take

    if shortfall > 0 or len(selected) < quota:
        remaining = [row for row in rows if int(row["source_sample_id"]) not in selected_ids]
        remaining.sort(key=lambda item: int(item["source_sample_id"]))
        rng.shuffle(remaining)
        selected.extend(remaining[: quota - len(selected)])

    if len(selected) != quota:
        raise ValueError(f"Could only select {len(selected)} rows for quota {quota}.")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("phygenbench_prompts.csv"))
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("phygenbench_moe_topk_ablation/fullpinn8_step18500/topk_4/performance_metrics.jsonl"),
        help="Existing full-run metrics used to freeze effective prompts and routing labels.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("phygenbench_moe_topk_speed_benchmark/fullpinn8_step18500/speed_subset_32.csv"),
    )
    parser.add_argument("--quotas", type=str, default=DEFAULT_QUOTAS)
    parser.add_argument("--seed", type=int, default=20260429)
    args = parser.parse_args()

    quotas = parse_quotas(args.quotas)
    metrics = read_metrics(args.metrics)
    rng = random.Random(args.seed)

    rows: list[dict[str, Any]] = []
    with args.csv.open("r", encoding="utf-8", newline="") as f:
        for source_sample_id, row in enumerate(csv.DictReader(f), start=1):
            metric = metrics.get(source_sample_id)
            if metric is None:
                raise ValueError(f"Missing successful metrics row for source sample {source_sample_id}.")
            frozen_prompt = metric.get("effective_prompt") or row.get("caption") or row.get("prompt") or ""
            label_name = metric.get("metadata_label_name") or row.get("label_name") or "Fluid"
            rows.append({
                "source_sample_id": source_sample_id,
                "prompt": frozen_prompt,
                "caption": frozen_prompt,
                "label_name": label_name,
                "states_of_matter": row.get("states_of_matter", ""),
                "complexity": row.get("complexity", ""),
                "majority_sa": row.get("majority_sa", ""),
                "majority_pc": row.get("majority_pc", ""),
                "source_video_url": row.get("video_url", ""),
                "source_caption": row.get("caption", row.get("prompt", "")),
                "frozen_from_moe_top_k": metric.get("moe_top_k"),
            })

    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[str(row["states_of_matter"])].append(row)

    selected: list[dict[str, Any]] = []
    for state, quota in quotas.items():
        candidates = by_state.get(state, [])
        if len(candidates) < quota:
            raise ValueError(f"State {state!r} has {len(candidates)} candidates, need {quota}.")
        selected.extend(balanced_take(candidates, quota, rng))

    rng.shuffle(selected)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_sample_id",
        "prompt",
        "caption",
        "label_name",
        "states_of_matter",
        "complexity",
        "majority_sa",
        "majority_pc",
        "source_video_url",
        "source_caption",
        "frozen_from_moe_top_k",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(f"Wrote {len(selected)} speed samples to {args.output_csv}")
    for state in sorted(quotas):
        state_rows = [row for row in selected if row["states_of_matter"] == state]
        counts: dict[str, int] = defaultdict(int)
        for row in state_rows:
            counts[str(row["complexity"])] += 1
        print(f"  {state}: {len(state_rows)} samples, complexity={dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
