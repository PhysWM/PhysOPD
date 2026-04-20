"""
Build paired baseline-vs-PINN compare reports.

This script treats the PINN trace sidecar as the cause view
(`correction_attribution_video`) and the final RGB delta as the effect view.
"""

import argparse
import base64
import html
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")


def _load_video_helpers():
    module_name = "diffsynth.data.video"
    module_path = REPO_ROOT / "diffsynth" / "data" / "video.py"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_VIDEO_HELPERS = None


def _get_video_helpers():
    global _VIDEO_HELPERS
    if _VIDEO_HELPERS is None:
        _VIDEO_HELPERS = _load_video_helpers()
    return _VIDEO_HELPERS


def video_helpers_available():
    try:
        _get_video_helpers()
    except ModuleNotFoundError:
        return False
    return True


def _to_uint8_frame(frame):
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB frame with shape [H, W, 3], got {array.shape}")
    return array


def _load_video_frames(video_path):
    video = _get_video_helpers().LowMemoryVideo(str(video_path))
    frames = [_to_uint8_frame(video[idx]) for idx in range(len(video))]
    return frames


def _resize_map_frame(map_2d, width, height):
    array = np.asarray(map_2d, dtype=np.float32)
    image = Image.fromarray(np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8))
    image = image.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _resize_map_video(map_video, n_frames, height, width):
    maps = np.asarray(map_video, dtype=np.float32)
    if maps.ndim != 3:
        raise ValueError(f"Expected map video with shape [F, H, W], got {maps.shape}")
    if maps.shape[0] == 0:
        return np.zeros((n_frames, height, width), dtype=np.float32)
    if maps.shape[0] == n_frames:
        frame_indices = np.arange(n_frames, dtype=np.int32)
    else:
        frame_indices = np.rint(np.linspace(0, maps.shape[0] - 1, num=n_frames)).astype(np.int32)
    resized = []
    for idx in frame_indices:
        resized.append(_resize_map_frame(maps[idx], width=width, height=height))
    return np.stack(resized, axis=0).astype(np.float32)


def _normalize_global(volume, percentile=99.0):
    volume = np.asarray(volume, dtype=np.float32)
    if volume.size == 0:
        return volume
    denom = float(np.percentile(volume, percentile))
    if denom <= 1e-8:
        denom = 1.0
    return np.clip(volume / denom, 0.0, 1.0).astype(np.float32)


def sparsify_maps(map_video, keep_percentile=90.0):
    maps = np.asarray(map_video, dtype=np.float32).copy()
    if maps.ndim != 3:
        raise ValueError(f"Expected [F, H, W], got {maps.shape}")
    for idx in range(maps.shape[0]):
        thresh = float(np.percentile(maps[idx], keep_percentile))
        maps[idx] = np.clip(
            maps[idx] * np.clip((maps[idx] - thresh) / (1.0 - thresh + 1e-8), 0.0, 1.0),
            0.0,
            1.0,
        )
    return maps.astype(np.float32)


def compute_effect_delta_video(
    baseline_frames,
    pinn_frames,
    effect_percentile=99.0,
    sparsify_percentile=90.0,
):
    if len(baseline_frames) != len(pinn_frames):
        raise ValueError(
            f"Baseline/PINN frame count mismatch: {len(baseline_frames)} vs {len(pinn_frames)}"
        )
    deltas = []
    for base_frame, pinn_frame in zip(baseline_frames, pinn_frames):
        base = _to_uint8_frame(base_frame).astype(np.float32)
        pinn = _to_uint8_frame(pinn_frame).astype(np.float32)
        if base.shape != pinn.shape:
            raise ValueError(f"Baseline/PINN frame shape mismatch: {base.shape} vs {pinn.shape}")
        delta = np.mean(np.abs(pinn - base), axis=2) / 255.0
        deltas.append(delta.astype(np.float32))
    delta_video = np.stack(deltas, axis=0)
    delta_video = _normalize_global(delta_video, percentile=effect_percentile)
    delta_video = sparsify_maps(delta_video, keep_percentile=sparsify_percentile)
    return delta_video


def compute_cause_effect_overlap(cause_maps, effect_maps, eps=1e-8):
    cause = np.asarray(cause_maps, dtype=np.float32)
    effect = np.asarray(effect_maps, dtype=np.float32)
    if cause.shape != effect.shape:
        raise ValueError(f"Cause/effect shape mismatch: {cause.shape} vs {effect.shape}")
    intersection = np.minimum(cause, effect).sum(axis=(1, 2))
    union = np.maximum(cause, effect).sum(axis=(1, 2)) + eps
    return (intersection / union).astype(np.float32)


def _heatmap_rgb(map_2d, cmap_name="inferno"):
    from matplotlib import colormaps

    map_2d = np.clip(np.asarray(map_2d, dtype=np.float32), 0.0, 1.0)
    heat = colormaps.get_cmap(cmap_name)(map_2d)[..., :3]
    return np.clip(heat * 255.0, 0.0, 255.0).astype(np.uint8)


def overlay_heatmap(frame, map_2d, alpha=0.45, cmap_name="inferno"):
    frame_rgb = _to_uint8_frame(frame).astype(np.float32)
    heat = _heatmap_rgb(map_2d, cmap_name=cmap_name).astype(np.float32)
    local_alpha = float(max(0.0, min(1.0, alpha))) * np.power(np.clip(map_2d, 0.0, 1.0), 1.5)
    local_alpha = local_alpha[..., None]
    return np.clip(frame_rgb * (1.0 - local_alpha) + heat * local_alpha, 0.0, 255.0).astype(np.uint8)


def _label_image(frame, label):
    image = Image.fromarray(_to_uint8_frame(frame))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    pad = 6
    draw.rectangle(
        (8, 8, 8 + text_width + 2 * pad, 8 + text_height + 2 * pad),
        fill=(0, 0, 0),
    )
    draw.text((8 + pad, 8 + pad), label, fill=(255, 255, 255), font=font)
    return np.asarray(image, dtype=np.uint8)


def _tile_four_frames(top_left, top_right, bottom_left, bottom_right):
    top_left = _to_uint8_frame(top_left)
    top_right = _to_uint8_frame(top_right)
    bottom_left = _to_uint8_frame(bottom_left)
    bottom_right = _to_uint8_frame(bottom_right)
    if top_left.shape != top_right.shape or top_left.shape != bottom_left.shape or top_left.shape != bottom_right.shape:
        raise ValueError("All tiled panels must share the same frame shape")
    top = np.concatenate([top_left, top_right], axis=1)
    bottom = np.concatenate([bottom_left, bottom_right], axis=1)
    return np.concatenate([top, bottom], axis=0)


def build_overview_frames(baseline_frames, pinn_frames, cause_maps, effect_maps, alpha=0.45):
    overview = []
    for idx in range(len(baseline_frames)):
        baseline = _label_image(baseline_frames[idx], "Baseline")
        pinn = _label_image(pinn_frames[idx], "With PINN")
        cause = _label_image(
            overlay_heatmap(pinn_frames[idx], cause_maps[idx], alpha=alpha),
            "Correction Attribution Overlay",
        )
        effect = _label_image(
            overlay_heatmap(pinn_frames[idx], effect_maps[idx], alpha=alpha),
            "Visible Change Overlay",
        )
        overview.append(Image.fromarray(_tile_four_frames(baseline, pinn, cause, effect)))
    return overview


def _sample_indices(n_frames, sample_count):
    if n_frames <= 0:
        return []
    sample_count = max(1, min(sample_count, n_frames))
    return np.rint(np.linspace(0, n_frames - 1, num=sample_count)).astype(int).tolist()


def _image_data_uri(frame):
    image = Image.fromarray(_to_uint8_frame(frame))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _curve_svg(values, width=900, height=220, stroke="#ff6b3d"):
    values = [float(v) for v in values]
    if not values:
        return "<svg></svg>"
    xs = np.linspace(20.0, width - 20.0, num=len(values))
    ys = (1.0 - np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)) * (height - 40.0) + 20.0
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    y_mid = height / 2.0
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#111418" rx="6" ry="6"/>'
        f'<line x1="20" y1="{y_mid:.2f}" x2="{width - 20}" y2="{y_mid:.2f}" stroke="#303842" stroke-width="1"/>'
        f'<polyline fill="none" stroke="{stroke}" stroke-width="3" points="{points}"/>'
        "</svg>"
    )


def _build_html_report(
    output_prefix,
    summary,
    sample_rows,
    overlap_values,
):
    style = """
body { font-family: Arial, sans-serif; background: #0f1115; color: #f1f3f5; margin: 24px; }
h1, h2 { margin: 0 0 16px 0; }
.cards { display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
.card { background: #171a20; padding: 16px; border: 1px solid #2a3038; border-radius: 6px; }
.label { color: #9aa4b2; font-size: 12px; text-transform: uppercase; }
.value { font-size: 28px; margin-top: 6px; }
.samples { display: grid; gap: 18px; }
.sample { background: #171a20; border: 1px solid #2a3038; border-radius: 6px; padding: 12px; }
.sample-grid { display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 10px; }
.sample-grid img { width: 100%; height: auto; border-radius: 4px; display: block; }
.sample-grid div { font-size: 13px; color: #c7d0da; }
.frame-title { margin-bottom: 10px; color: #c7d0da; }
"""
    cards_html = "".join(
        [
            f'<div class="card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{value:.4f}</div></div>'
            for label, value in (
                ("mean_cause_strength", summary["mean_cause_strength"]),
                ("mean_visible_change", summary["mean_visible_change"]),
                ("cause_effect_overlap", summary["cause_effect_overlap"]),
            )
        ]
    )
    sample_html = []
    for row in sample_rows:
        sample_html.append(
            '<div class="sample">'
            f'<div class="frame-title">Frame {row["frame_index"]}</div>'
            '<div class="sample-grid">'
            f'<div><img src="{row["baseline_uri"]}" alt="Baseline"><div>Baseline</div></div>'
            f'<div><img src="{row["pinn_uri"]}" alt="With PINN"><div>With PINN</div></div>'
            f'<div><img src="{row["cause_uri"]}" alt="Cause"><div>Cause</div></div>'
            f'<div><img src="{row["effect_uri"]}" alt="Effect"><div>Effect</div></div>'
            "</div>"
            "</div>"
        )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(Path(output_prefix).name)} Paired Report</title>"
        f"<style>{style}</style></head><body>"
        f"<h1>{html.escape(Path(output_prefix).name)} Paired Report</h1>"
        "<div class='cards'>"
        f"{cards_html}"
        "</div>"
        "<h2>Overlap Curve</h2>"
        f"{_curve_svg(overlap_values)}"
        "<h2>Sampled Frames</h2>"
        "<div class='samples'>"
        f"{''.join(sample_html)}"
        "</div>"
        "</body></html>"
    )


def build_paired_report(
    baseline_video,
    pinn_video,
    pinn_trace,
    output_prefix,
    fps=15,
    quality=5,
    overlay_alpha=0.45,
    sample_frames=6,
    effect_percentile=99.0,
    effect_sparsify_percentile=90.0,
):
    baseline_path = Path(baseline_video)
    pinn_path = Path(pinn_video)
    trace_path = Path(pinn_trace)
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    baseline_frames = _load_video_frames(baseline_path)
    pinn_frames = _load_video_frames(pinn_path)
    if len(baseline_frames) != len(pinn_frames):
        raise ValueError(
            f"Baseline/PINN frame count mismatch: {len(baseline_frames)} vs {len(pinn_frames)}"
        )
    if not baseline_frames:
        raise ValueError("Input videos are empty")
    frame_height, frame_width = baseline_frames[0].shape[:2]

    trace = np.load(trace_path)
    if "correction_attribution_video" not in trace:
        raise ValueError(f"Missing correction_attribution_video in trace: {trace_path}")
    cause_maps = _resize_map_video(
        trace["correction_attribution_video"],
        n_frames=len(baseline_frames),
        height=frame_height,
        width=frame_width,
    )
    cause_maps = np.clip(cause_maps, 0.0, 1.0).astype(np.float32)

    effect_maps = compute_effect_delta_video(
        baseline_frames,
        pinn_frames,
        effect_percentile=effect_percentile,
        sparsify_percentile=effect_sparsify_percentile,
    )
    overlap = compute_cause_effect_overlap(cause_maps, effect_maps)

    overview_frames = build_overview_frames(
        baseline_frames=baseline_frames,
        pinn_frames=pinn_frames,
        cause_maps=cause_maps,
        effect_maps=effect_maps,
        alpha=overlay_alpha,
    )
    overview_path = output_prefix.with_name(output_prefix.name + "_paired_overview.mp4")
    _get_video_helpers().save_video(
        overview_frames,
        str(overview_path),
        fps=int(max(1, fps)),
        quality=int(quality),
    )

    sample_rows = []
    for frame_idx in _sample_indices(len(baseline_frames), sample_frames):
        sample_rows.append(
            {
                "frame_index": int(frame_idx),
                "baseline_uri": _image_data_uri(baseline_frames[frame_idx]),
                "pinn_uri": _image_data_uri(pinn_frames[frame_idx]),
                "cause_uri": _image_data_uri(
                    overlay_heatmap(pinn_frames[frame_idx], cause_maps[frame_idx], alpha=overlay_alpha)
                ),
                "effect_uri": _image_data_uri(
                    overlay_heatmap(pinn_frames[frame_idx], effect_maps[frame_idx], alpha=overlay_alpha)
                ),
            }
        )

    summary = {
        "mean_cause_strength": float(np.mean(cause_maps)),
        "mean_visible_change": float(np.mean(effect_maps)),
        "cause_effect_overlap": float(np.mean(overlap)),
        "frame_count": int(len(baseline_frames)),
        "frame_height": int(frame_height),
        "frame_width": int(frame_width),
    }
    json_payload = {
        "summary": summary,
        "frame_metrics": [
            {
                "frame_index": int(idx),
                "cause_strength": float(np.mean(cause_maps[idx])),
                "visible_change": float(np.mean(effect_maps[idx])),
                "cause_effect_overlap": float(overlap[idx]),
            }
            for idx in range(len(overlap))
        ],
        "inputs": {
            "baseline_video": str(baseline_path),
            "pinn_video": str(pinn_path),
            "pinn_trace": str(trace_path),
        },
    }

    html_path = output_prefix.with_name(output_prefix.name + "_paired_report.html")
    html_path.write_text(
        _build_html_report(
            output_prefix=output_prefix,
            summary=summary,
            sample_rows=sample_rows,
            overlap_values=overlap.tolist(),
        ),
        encoding="utf-8",
    )

    json_path = output_prefix.with_name(output_prefix.name + "_paired_report.json")
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    return {
        "overview_path": str(overview_path),
        "html_path": str(html_path),
        "json_path": str(json_path),
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Build paired baseline-vs-PINN correction reports")
    parser.add_argument("--baseline_video", type=str, required=True, help="Baseline video path")
    parser.add_argument("--pinn_video", type=str, required=True, help="PINN video path")
    parser.add_argument("--pinn_trace", type=str, required=True, help="PINN trace npz path")
    parser.add_argument("--output_prefix", type=str, required=True, help="Output prefix")
    parser.add_argument("--fps", type=int, default=15, help="Overview video FPS")
    parser.add_argument("--quality", type=int, default=5, help="Overview video quality")
    parser.add_argument("--overlay_alpha", type=float, default=0.45, help="Overlay alpha")
    parser.add_argument("--sample_frames", type=int, default=6, help="Number of sampled frames in HTML")
    parser.add_argument("--effect_percentile", type=float, default=99.0, help="Global effect normalization percentile")
    parser.add_argument(
        "--effect_sparsify_percentile",
        type=float,
        default=90.0,
        help="Per-frame effect sparsify percentile",
    )
    args = parser.parse_args()

    outputs = build_paired_report(
        baseline_video=args.baseline_video,
        pinn_video=args.pinn_video,
        pinn_trace=args.pinn_trace,
        output_prefix=args.output_prefix,
        fps=args.fps,
        quality=args.quality,
        overlay_alpha=args.overlay_alpha,
        sample_frames=args.sample_frames,
        effect_percentile=args.effect_percentile,
        effect_sparsify_percentile=args.effect_sparsify_percentile,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
