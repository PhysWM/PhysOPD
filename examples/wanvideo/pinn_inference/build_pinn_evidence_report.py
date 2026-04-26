#!/usr/bin/env python3
"""
Build a paper-style PINN evidence report from a generated video and physics trace.

This report intentionally emphasizes where the PINN correction acts in the video.
If per-expert attribution is present, it is summarized as an overlap diagnostic
instead of being framed as spatial expert specialization.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_video_frames(path: Path, indices: np.ndarray) -> dict[int, np.ndarray]:
    reader = imageio.get_reader(str(path))
    frames: dict[int, np.ndarray] = {}
    wanted = {int(idx) for idx in indices.tolist()}
    try:
        for idx in sorted(wanted):
            frames[idx] = np.asarray(reader.get_data(idx))
    finally:
        reader.close()
    return frames


def _video_length(path: Path, fallback: int) -> int:
    reader = imageio.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
        nframes = meta.get("nframes")
        if isinstance(nframes, int) and 0 < nframes < 10**7:
            return nframes
    except Exception:
        pass
    finally:
        reader.close()
    return int(fallback)


def _compose_overlay(frame: np.ndarray, attention: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    frame_f = frame.astype(np.float32)
    att = np.clip(np.asarray(attention, dtype=np.float32), 0.0, 1.0)
    att_rgb = (plt.get_cmap("inferno")(att)[:, :, :3] * 255.0).astype(np.float32)
    local_alpha = float(np.clip(alpha, 0.0, 1.0)) * np.power(att, 1.4)
    blend = frame_f * (1.0 - local_alpha[:, :, None]) + att_rgb * local_alpha[:, :, None]
    return np.clip(blend, 0.0, 255.0).astype(np.uint8)


def _trim_prompt(prompt: str, max_words: int = 28) -> str:
    clean = " ".join(str(prompt or "").split())
    words = clean.split()
    if len(words) > max_words:
        clean = " ".join(words[:max_words]) + "..."
    return textwrap.fill(clean, width=24)


def _justify_prompt(prompt: str, max_words: int = 18, width: int = 25) -> str:
    clean = " ".join(str(prompt or "").split())
    words = clean.split()
    if len(words) > max_words:
        clean = " ".join(words[:max_words]) + "..."
    lines = textwrap.wrap(clean, width=width)
    justified = []
    for line_idx, line in enumerate(lines):
        if line_idx == len(lines) - 1 or " " not in line or line.endswith("..."):
            justified.append(line)
            continue
        parts = line.split()
        gaps = len(parts) - 1
        if gaps <= 0:
            justified.append(line)
            continue
        letters = sum(len(part) for part in parts)
        spaces_needed = max(width - letters, gaps)
        base = spaces_needed // gaps
        extra = spaces_needed % gaps
        pieces = []
        for idx, part in enumerate(parts[:-1]):
            pieces.append(part)
            pieces.append(" " * (base + (1 if idx < extra else 0)))
        pieces.append(parts[-1])
        justified.append("".join(pieces))
    return "\n".join(justified)


def _wrap_prompt_left(prompt: str, max_words: int = 16, width: int = 22) -> str:
    clean = " ".join(str(prompt or "").split())
    words = clean.split()
    if len(words) > max_words:
        clean = " ".join(words[:max_words]) + "..."
    return textwrap.fill(clean, width=width)


def _plot_correction_report(
    video_path: Path,
    trace: dict[str, np.ndarray],
    prompt: str,
    output_prefix: Path,
    num_frames: int,
    alpha: float,
    baseline_video_path: Path | None = None,
    frame_indices: list[int] | None = None,
) -> None:
    correction = np.asarray(trace["correction_attribution_video"], dtype=np.float32)
    if correction.ndim != 3:
        raise ValueError("correction_attribution_video must have shape [F,H,W]")

    n_total = correction.shape[0]
    indices = _select_correction_frames(
        correction,
        num_frames=num_frames,
        frame_indices=frame_indices,
    )
    n_rows = int(indices.shape[0])
    frames = _read_video_frames(video_path, indices)
    baseline_frames = None
    if baseline_video_path is not None:
        baseline_frames = _read_video_frames(baseline_video_path, indices)

    sample = frames[int(indices[0])]
    frame_h, frame_w = sample.shape[:2]
    aspect = frame_h / max(float(frame_w), 1.0)
    cell_w = 2.15
    cell_h = cell_w * aspect
    prompt_w = 1.7
    weights_w = 1.75
    has_baseline = baseline_frames is not None
    image_cols = 3 if has_baseline else 2
    fig_w = prompt_w + cell_w * image_cols + weights_w
    fig_h = 0.25 + cell_h * n_rows

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    grid = fig.add_gridspec(
        n_rows,
        image_cols + 2,
        width_ratios=[prompt_w] + [cell_w] * image_cols + [weights_w],
        left=0.02,
        right=0.995,
        top=0.88,
        bottom=0.035,
        wspace=0.018,
        hspace=0.02,
    )

    prompt_ax = fig.add_subplot(grid[:, 0])
    prompt_ax.axis("off")
    prompt_ax.text(
        0.97,
        0.5,
        _trim_prompt(prompt),
        ha="right",
        va="center",
        fontsize=9.0,
        family="serif",
        linespacing=1.18,
    )

    titles = ["No PINN", "PINN correction", "PINN output"] if has_baseline else ["PINN correction", "PINN output"]
    for row, frame_idx in enumerate(indices):
        frame_idx = int(frame_idx)
        frame = frames[frame_idx]
        att = np.clip(correction[frame_idx], 0.0, 1.0)
        panels = []
        if has_baseline:
            panels.append((baseline_frames[frame_idx], None))
        panels.extend([
            (att, "inferno"),
            (frame, None),
        ])
        for col, (image, cmap) in enumerate(panels, start=1):
            ax = fig.add_subplot(grid[row, col])
            if cmap is None:
                ax.imshow(image)
            else:
                ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col - 1], fontsize=10.5, fontweight="bold", family="serif", pad=5)
            for spine in ax.spines.values():
                spine.set_color("white")
                spine.set_linewidth(1.4)

    weights_ax = fig.add_subplot(grid[:, image_cols + 1])
    _plot_expert_weights_panel(weights_ax, trace)

    png_path = output_prefix.with_name(output_prefix.name + "_correction_evidence.png")
    pdf_path = output_prefix.with_name(output_prefix.name + "_correction_evidence.pdf")
    fig.savefig(png_path, dpi=260, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print(f"[Evidence] correction figure -> {png_path}")
    print(f"[Evidence] correction PDF -> {pdf_path}")


def _plot_multi_example_report(
    examples: list[dict[str, str]],
    output_prefix: Path,
    alpha: float,
) -> None:
    rows = []
    for example in examples:
        video_path = Path(example["video"])
        baseline_path = Path(example["baseline_video"])
        trace_path = Path(example["trace"])
        prompt = str(example.get("prompt", ""))
        label = str(example.get("label", ""))
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        if not baseline_path.exists():
            raise FileNotFoundError(baseline_path)
        if not trace_path.exists():
            raise FileNotFoundError(trace_path)
        with np.load(trace_path) as loaded:
            trace = {key: loaded[key] for key in loaded.files}
        correction = np.asarray(trace["correction_attribution_video"], dtype=np.float32)
        frame_idx = int(_select_correction_frames(correction, num_frames=1)[0])
        frames = _read_video_frames(video_path, np.asarray([frame_idx], dtype=int))
        baseline_frames = _read_video_frames(baseline_path, np.asarray([frame_idx], dtype=int))
        rows.append(
            {
                "label": label,
                "prompt": prompt,
                "frame_idx": frame_idx,
                "baseline": baseline_frames[frame_idx],
                "pinn": frames[frame_idx],
                "correction": np.clip(correction[frame_idx], 0.0, 1.0),
                "trace": trace,
            }
        )

    if not rows:
        raise ValueError("No examples provided.")

    sample = rows[0]["pinn"]
    frame_h, frame_w = sample.shape[:2]
    aspect = frame_h / max(float(frame_w), 1.0)
    n_rows = len(rows)
    prompt_w = 1.72
    cell_w = 1.95
    weights_w = 2.95
    cell_h = cell_w * aspect
    fig_w = prompt_w + cell_w * 3 + weights_w
    fig_h = 0.42 + cell_h * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    grid = fig.add_gridspec(
        n_rows,
        5,
        width_ratios=[prompt_w, cell_w, cell_w, cell_w, weights_w],
        left=0.018,
        right=0.995,
        top=0.91,
        bottom=0.035,
        wspace=0.018,
        hspace=0.06,
    )

    titles = ["No PINN", "PINN correction", "PINN output"]

    for row_idx, row in enumerate(rows):
        prompt_ax = fig.add_subplot(grid[row_idx, 0])
        prompt_ax.axis("off")
        prompt = row["prompt"]
        prompt_ax.text(
            0.05,
            0.5,
            _wrap_prompt_left(prompt, max_words=16, width=22),
            ha="left",
            va="center",
            fontsize=8.2,
            family="serif",
            linespacing=1.12,
        )

        panels = [
            (row["baseline"], None),
            (row["correction"], "inferno"),
            (row["pinn"], None),
        ]
        for col, (image, cmap) in enumerate(panels, start=1):
            ax = fig.add_subplot(grid[row_idx, col])
            if cmap is None:
                ax.imshow(image)
            else:
                ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(titles[col - 1], fontsize=10.5, fontweight="bold", family="serif", pad=5)
            for spine in ax.spines.values():
                spine.set_color("white")
                spine.set_linewidth(1.2)

        weights_ax = fig.add_subplot(grid[row_idx, 4])
        _plot_expert_weights_panel(
            weights_ax,
            row["trace"],
            title=(row_idx == 0),
            compact=True,
            show_axis=False,
        )

    png_path = output_prefix.with_name(output_prefix.name + "_four_examples.png")
    pdf_path = output_prefix.with_name(output_prefix.name + "_four_examples.pdf")
    fig.savefig(png_path, dpi=260, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print(f"[Evidence] four-example figure -> {png_path}")
    print(f"[Evidence] four-example PDF -> {pdf_path}")


def _select_correction_frames(
    correction: np.ndarray,
    num_frames: int,
    frame_indices: list[int] | None = None,
) -> np.ndarray:
    n_total = int(correction.shape[0])
    if n_total <= 0:
        return np.asarray([], dtype=int)
    if frame_indices:
        clipped = sorted({min(max(int(idx), 0), n_total - 1) for idx in frame_indices})
        return np.asarray(clipped, dtype=int)

    n_select = min(max(1, int(num_frames)), n_total)
    if n_select >= n_total:
        return np.arange(n_total, dtype=int)

    maps = np.asarray(correction, dtype=np.float32)
    flat = maps.reshape(n_total, -1)
    # Top-percentile mean is more robust than whole-frame mean for sparse corrections.
    k = max(1, int(flat.shape[1] * 0.02))
    topk = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:]
    energy = topk.mean(axis=1)

    chosen: list[int] = []
    candidate_order = np.argsort(energy)[::-1]
    min_gap = max(1, n_total // (n_select * 3))
    for idx in candidate_order:
        idx_i = int(idx)
        if all(abs(idx_i - prev) >= min_gap for prev in chosen):
            chosen.append(idx_i)
            if len(chosen) >= n_select:
                break
    for idx in candidate_order:
        if len(chosen) >= n_select:
            break
        idx_i = int(idx)
        if idx_i not in chosen:
            chosen.append(idx_i)
    return np.asarray(sorted(chosen), dtype=int)


def _plot_expert_weights_panel(
    ax,
    trace: dict[str, np.ndarray],
    title: bool = True,
    compact: bool = False,
    show_axis: bool = True,
) -> None:
    names = trace.get("expert_names")
    weights = trace.get("expert_weights")
    if names is None or weights is None:
        ax.axis("off")
        if title:
            ax.set_title("Expert weights", fontsize=10.5, fontweight="bold", family="serif", pad=5)
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=9, family="serif")
        return

    names_arr = np.asarray(names).astype(str).reshape(-1)
    weights_arr = np.asarray(weights, dtype=np.float32).reshape(-1)
    n = min(len(names_arr), len(weights_arr))
    if n <= 0:
        ax.axis("off")
        return
    names_arr = names_arr[:n]
    weights_arr = weights_arr[:n]
    order = np.argsort(weights_arr)[::-1]
    names_arr = names_arr[order]
    weights_arr = weights_arr[order]

    if compact and n > 4:
        names_arr = names_arr[:4]
        weights_arr = weights_arr[:4]
        n = 4
    if compact and not show_axis:
        ax.axis("off")
        if title:
            ax.set_title("Expert weights", fontsize=10.5, fontweight="bold", family="serif", pad=5)
        display_names = []
        for name in names_arr:
            name = str(name)
            display_names.append(name[:12] + "..." if len(name) > 14 else name)
        max_width = max(float(weights_arr.max()), 1.0)
        y_positions = np.linspace(0.80, 0.20, n)
        for name, val, y_pos in zip(display_names, weights_arr, y_positions):
            val = float(val)
            ax.text(
                0.02,
                y_pos,
                name,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=6.0,
                family="serif",
            )
            bar_x = 0.38
            bar_w = 0.46 * val / max_width
            ax.add_patch(
                plt.Rectangle(
                    (bar_x, y_pos - 0.025),
                    bar_w,
                    0.05,
                    transform=ax.transAxes,
                    facecolor="#4c78a8",
                    edgecolor="none",
                    clip_on=False,
                )
            )
            ax.text(
                min(bar_x + bar_w + 0.025, 0.84),
                y_pos,
                f"{val:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=6.0,
                family="serif",
            )
        ax.add_patch(
            plt.Rectangle(
                (0.0, 0.02),
                0.96,
                0.94,
                transform=ax.transAxes,
                facecolor="none",
                edgecolor=(0.0, 0.0, 0.0, 0.08),
                linewidth=0.6,
                clip_on=False,
            )
        )
        return

    y = np.arange(n)
    display_names = []
    for name in names_arr:
        if len(name) > 18:
            display_names.append(name[:16] + "...")
        else:
            display_names.append(str(name))
    ax.barh(y, weights_arr, color="#4c78a8", height=0.26 if compact else 0.28, left=0.0)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim(-0.02, max(1.0, float(weights_arr.max()) * 1.18))
    if title:
        ax.set_title("Expert weights", fontsize=10.5, fontweight="bold", family="serif", pad=5)
    if show_axis:
        ax.tick_params(axis="x", labelsize=6.5 if compact else 7, length=2, pad=1)
        ax.grid(axis="x", alpha=0.12, linewidth=0.5)
    else:
        ax.set_xticks([])
        ax.grid(axis="x", alpha=0.08, linewidth=0.45)
    for idx, (name, val) in enumerate(zip(display_names, weights_arr)):
        ax.text(
            0.0,
            idx - (0.24 if compact else 0.32),
            name,
            va="center",
            ha="left",
            fontsize=5.6 if compact else 7.3,
            family="serif",
        )
        ax.text(
            min(float(val) + 0.015, 0.98),
            idx,
            f"{val:.2f}",
            va="center",
            fontsize=5.8 if compact else 7.2,
        )
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    if show_axis:
        ax.spines["bottom"].set_alpha(0.3)
    else:
        ax.spines["bottom"].set_visible(False)


def _expert_overlap_metrics(trace: dict[str, np.ndarray], keep_percentile: float) -> dict[str, object]:
    if "expert_attribution_video" not in trace:
        return {"has_expert_attribution": False}
    expert = np.asarray(trace["expert_attribution_video"], dtype=np.float32)
    if expert.ndim != 4 or expert.shape[0] < 2:
        return {"has_expert_attribution": False}

    names = trace.get("expert_names")
    if names is None:
        names_list = [f"Expert {idx}" for idx in range(expert.shape[0])]
    else:
        names_list = [str(name) for name in np.asarray(names).reshape(-1)]
    while len(names_list) < expert.shape[0]:
        names_list.append(f"Expert {len(names_list)}")

    flat = expert.reshape(expert.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    cosine = (flat / norms) @ (flat / norms).T

    masks = []
    for idx in range(expert.shape[0]):
        threshold = np.percentile(expert[idx], keep_percentile)
        masks.append(expert[idx] >= threshold)
    masks_arr = np.stack(masks, axis=0)
    iou = np.eye(expert.shape[0], dtype=np.float32)
    for i in range(expert.shape[0]):
        for j in range(i + 1, expert.shape[0]):
            inter = np.logical_and(masks_arr[i], masks_arr[j]).sum()
            union = np.logical_or(masks_arr[i], masks_arr[j]).sum()
            val = float(inter / max(float(union), 1.0))
            iou[i, j] = val
            iou[j, i] = val

    offdiag = ~np.eye(expert.shape[0], dtype=bool)
    return {
        "has_expert_attribution": True,
        "expert_names": names_list[: expert.shape[0]],
        "mean_pairwise_cosine": float(np.mean(cosine[offdiag])),
        "mean_pairwise_iou_at_percentile": float(np.mean(iou[offdiag])),
        "keep_percentile": float(keep_percentile),
        "cosine_matrix": cosine.tolist(),
        "iou_matrix": iou.tolist(),
    }


def _plot_expert_overlap(metrics: dict[str, object], output_prefix: Path) -> None:
    if not metrics.get("has_expert_attribution"):
        return
    names = [str(name) for name in metrics["expert_names"]]
    cosine = np.asarray(metrics["cosine_matrix"], dtype=np.float32)
    iou = np.asarray(metrics["iou_matrix"], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), facecolor="white")
    for ax, matrix, title in [
        (axes[0], cosine, "Expert attribution cosine"),
        (axes[1], iou, f"Top-region IoU p{metrics['keep_percentile']:.0f}"),
    ]:
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10.5, fontweight="bold", family="serif")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.tight_layout(pad=0.7)
    png_path = output_prefix.with_name(output_prefix.name + "_expert_overlap_diagnostic.png")
    pdf_path = output_prefix.with_name(output_prefix.name + "_expert_overlap_diagnostic.pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[Evidence] expert overlap diagnostic -> {png_path}")
    print(f"[Evidence] expert overlap diagnostic PDF -> {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PINN correction evidence report.")
    parser.add_argument(
        "--examples_json",
        default=None,
        type=Path,
        help="Optional JSON list for a four-example summary figure.",
    )
    parser.add_argument("--video", default=None, type=Path)
    parser.add_argument("--baseline_video", default=None, type=Path)
    parser.add_argument("--trace", default=None, type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output_prefix", required=True, type=Path)
    parser.add_argument("--num_frames", type=int, default=2)
    parser.add_argument(
        "--frame_indices",
        default="",
        help="Optional comma-separated frame indices. Defaults to frames with strongest correction energy.",
    )
    parser.add_argument("--overlay_alpha", type=float, default=0.48)
    parser.add_argument("--expert_iou_percentile", type=float, default=90.0)
    args = parser.parse_args()

    if args.examples_json is not None:
        if not args.examples_json.exists():
            raise FileNotFoundError(args.examples_json)
        with args.examples_json.open("r", encoding="utf-8") as f:
            examples = json.load(f)
        if not isinstance(examples, list):
            raise ValueError("--examples_json must contain a JSON list.")
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        _plot_multi_example_report(
            examples=examples,
            output_prefix=args.output_prefix,
            alpha=args.overlay_alpha,
        )
        return

    if args.video is None:
        raise ValueError("--video is required unless --examples_json is provided.")
    if args.trace is None:
        raise ValueError("--trace is required unless --examples_json is provided.")
    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if args.baseline_video is not None and not args.baseline_video.exists():
        raise FileNotFoundError(args.baseline_video)
    if not args.trace.exists():
        raise FileNotFoundError(args.trace)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    with np.load(args.trace) as loaded:
        trace = {key: loaded[key] for key in loaded.files}
    if "correction_attribution_video" not in trace:
        raise ValueError(f"Missing correction_attribution_video in trace: {args.trace}")
    frame_indices = None
    if args.frame_indices.strip():
        frame_indices = [
            int(part.strip())
            for part in args.frame_indices.split(",")
            if part.strip() != ""
        ]

    _plot_correction_report(
        video_path=args.video,
        trace=trace,
        prompt=args.prompt,
        output_prefix=args.output_prefix,
        num_frames=args.num_frames,
        alpha=args.overlay_alpha,
        baseline_video_path=args.baseline_video,
        frame_indices=frame_indices,
    )
    metrics = _expert_overlap_metrics(trace, keep_percentile=args.expert_iou_percentile)
    metrics_path = args.output_prefix.with_name(args.output_prefix.name + "_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[Evidence] metrics -> {metrics_path}")
    _plot_expert_overlap(metrics, args.output_prefix)


if __name__ == "__main__":
    main()
