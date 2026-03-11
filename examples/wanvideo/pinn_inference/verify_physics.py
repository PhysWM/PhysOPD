"""
物理场验证与可视化工具
Physics Field Verification & Visualization

用途：验证 PINN 生成的物理场是否正确，提供两个层面的验证：
  1. Latent 空间：在去噪过程中追踪 PDE 残差（散度、涡量），对比有无 PINN
  2. Pixel 空间：从生成视频中提取稠密光流，可视化散度/涡量热力图

使用方法：
  # 从已有视频提取光流做物理验证
  python verify_physics.py --mode flow --video video_pinn.mp4 --output verify_output/

  # 推理时同步记录 latent 物理场（需提供 checkpoint）
  python verify_physics.py --mode latent --prompt "water flowing" \\
      --checkpoint models/train/pinn_plugin_low_noise/pinn_plugin_final.pt \\
      --output verify_output/

  # A/B 对比：有无 PINN 的物理指标对比
  python verify_physics.py --mode compare --video_pinn video_pinn.mp4 \\
      --video_base video_base.mp4 --output verify_output/
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Pixel-space optical flow analysis
# ─────────────────────────────────────────────────────────────────────────────

def extract_optical_flow(video_path: str) -> list[np.ndarray]:
    """用 Farneback 算法提取视频相邻帧之间的稠密光流。
    Returns: list of (H, W, 2) flow arrays (u, v)"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    flows = []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i + 1],
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )  # (H, W, 2)
        flows.append(flow)
    return flows, frames


def compute_divergence_2d(flow: np.ndarray) -> np.ndarray:
    """计算光流的散度 ∇·v = du/dx + dv/dy。
    对不可压缩流体，散度应趋近于 0。
    flow: (H, W, 2)  →  返回 (H, W)"""
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    du_dx = np.gradient(u, axis=1)
    dv_dy = np.gradient(v, axis=0)
    return du_dx + dv_dy


def compute_vorticity_2d(flow: np.ndarray) -> np.ndarray:
    """计算光流的涡量（旋度的 z 分量）ω = dv/dx - du/dy。
    涡量可视化揭示旋转结构（涡旋中心）。
    flow: (H, W, 2)  →  返回 (H, W)"""
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    dv_dx = np.gradient(v, axis=1)
    du_dy = np.gradient(u, axis=0)
    return dv_dx - du_dy


def compute_flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """光流大小（速度幅值）。"""
    return np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """将光流转换为 HSV 色轮可视化（方向→色调，幅值→亮度）。"""
    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[:, :, 0] = ang * 180 / np.pi / 2
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def visualize_flow_physics(video_path: str, output_dir: str, tag: str = ""):
    """主入口：从视频提取光流并可视化物理场。"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"[Flow Physics] {video_path}")
    print(f"{'='*60}")

    flows, frames = extract_optical_flow(video_path)
    n = len(flows)
    if n == 0:
        print("  ERROR: No optical flow extracted (video too short?)")
        return {}

    # 计算每帧的物理统计量
    div_means, div_stds = [], []
    vor_means, vor_stds = [], []
    mag_means = []

    for flow in flows:
        div = compute_divergence_2d(flow)
        vor = compute_vorticity_2d(flow)
        mag = compute_flow_magnitude(flow)
        div_means.append(np.mean(np.abs(div)))
        div_stds.append(np.std(div))
        vor_means.append(np.mean(np.abs(vor)))
        vor_stds.append(np.std(vor))
        mag_means.append(np.mean(mag))

    # ── 图1：关键帧光流物理场可视化（取第1、1/4、1/2、3/4帧）
    sample_indices = sorted(set([
        0,
        max(1, n // 4),
        max(1, n // 2),
        max(1, 3 * n // 4),
        n - 1,
    ]))
    sample_indices = [i for i in sample_indices if i < n]

    fig, axes = plt.subplots(len(sample_indices), 5, figsize=(20, 4 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"Optical Flow Physics Field Analysis{' - ' + tag if tag else ''}", fontsize=14)

    for row, fi in enumerate(sample_indices):
        flow = flows[fi]
        div = compute_divergence_2d(flow)
        vor = compute_vorticity_2d(flow)
        mag = compute_flow_magnitude(flow)
        rgb_flow = flow_to_rgb(flow)

        ax_titles = ["Frame", "Optical Flow (HSV)", "Magnitude", "Divergence ∇·v", "Vorticity ω"]
        datas = [
            cv2.cvtColor(cv2.imread(video_path) if False else
                         cv2.cvtColor(frames[fi], cv2.COLOR_GRAY2RGB), cv2.COLOR_BGR2RGB),
            rgb_flow,
            mag,
            div,
            vor,
        ]
        cmaps = [None, None, "hot", "RdBu_r", "RdBu_r"]

        # 重新获取正确的帧（灰度→RGB）
        frame_rgb = cv2.cvtColor(frames[fi], cv2.COLOR_GRAY2RGB)
        datas[0] = frame_rgb

        for col, (title, data, cmap) in enumerate(zip(ax_titles, datas, cmaps)):
            ax = axes[row, col]
            if cmap is None:
                ax.imshow(data)
            else:
                vmax = np.percentile(np.abs(data), 95) + 1e-6
                ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax)
                sm = ScalarMappable(cmap=cmap, norm=Normalize(-vmax, vmax))
                sm.set_array([])
                plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

            if row == 0:
                ax.set_title(title, fontsize=11)
            ax.set_ylabel(f"Frame {fi}", fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"flow_field{('_'+tag) if tag else ''}.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved flow field visualization → {fig_path}")

    # ── 图2：物理量随时间变化曲线
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle(f"Physics Metrics Over Time{' - ' + tag if tag else ''}", fontsize=13)
    t = list(range(n))

    axes2[0].plot(t, div_means, "b-o", markersize=3, label="|div| mean")
    axes2[0].fill_between(t, np.array(div_means) - np.array(div_stds),
                           np.array(div_means) + np.array(div_stds), alpha=0.2)
    axes2[0].set_title("Divergence |∇·v| (↓ = more incompressible)")
    axes2[0].set_xlabel("Frame pair index")
    axes2[0].set_ylabel("|∇·v| mean")
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)

    axes2[1].plot(t, vor_means, "r-o", markersize=3, label="|vorticity| mean")
    axes2[1].fill_between(t, np.array(vor_means) - np.array(vor_stds),
                           np.array(vor_means) + np.array(vor_stds), alpha=0.2, color="red")
    axes2[1].set_title("Vorticity |ω| (turbulent structures)")
    axes2[1].set_xlabel("Frame pair index")
    axes2[1].set_ylabel("|ω| mean")
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)

    axes2[2].plot(t, mag_means, "g-o", markersize=3, label="flow speed mean")
    axes2[2].set_title("Flow Speed (motion magnitude)")
    axes2[2].set_xlabel("Frame pair index")
    axes2[2].set_ylabel("speed (px/frame)")
    axes2[2].legend()
    axes2[2].grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(output_dir, f"physics_curve{('_'+tag) if tag else ''}.png")
    plt.savefig(curve_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved physics curve → {curve_path}")

    stats = {
        "tag": tag,
        "num_frames": n,
        "div_mean": float(np.mean(div_means)),
        "div_std": float(np.mean(div_stds)),
        "vor_mean": float(np.mean(vor_means)),
        "mag_mean": float(np.mean(mag_means)),
    }
    print(f"\n  [Summary - {tag or 'video'}]")
    print(f"    Average |divergence|  = {stats['div_mean']:.4f}  (理想: 0，数值越小越符合不可压缩约束)")
    print(f"    Average |vorticity|   = {stats['vor_mean']:.4f}  (涡旋强度)")
    print(f"    Average flow speed    = {stats['mag_mean']:.4f} px/frame")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Latent-space PDE residual tracking during inference
# ─────────────────────────────────────────────────────────────────────────────

def compute_latent_divergence(v: torch.Tensor) -> float:
    """计算 latent 速度场的散度 L2 均值 (B, C, T, H, W)。"""
    dv_dh = v[:, :, :, 1:, :] - v[:, :, :, :-1, :]
    dv_dh = F.pad(dv_dh, (0, 0, 0, 1))
    dv_dw = v[:, :, :, :, 1:] - v[:, :, :, :, :-1]
    dv_dw = F.pad(dv_dw, (0, 1))
    div = dv_dh.mean(dim=1, keepdim=True) + dv_dw.mean(dim=1, keepdim=True)
    return div.pow(2).mean().item()


def compute_latent_vorticity(v: torch.Tensor) -> float:
    """计算 latent 速度场的涡量强度。"""
    C = v.shape[1]
    half_C = max(C // 2, 1)
    dv_dh = v[:, :, :, 1:, :] - v[:, :, :, :-1, :]
    dv_dw = v[:, :, :, :, 1:] - v[:, :, :, :, :-1]
    dv_dh = dv_dh[:, :, :, :, :-1]
    dv_dw = dv_dw[:, :, :-1 if dv_dw.shape[2] > 1 else None, :-1]
    # align sizes
    min_h = min(dv_dh.shape[3], dv_dw.shape[3])
    min_w = min(dv_dh.shape[4], dv_dw.shape[4])
    dv_dh = dv_dh[:, :, :, :min_h, :min_w]
    dv_dw = dv_dw[:, :, :, :min_h, :min_w]
    half_C_dh = max(dv_dh.shape[1] // 2, 1)
    curl = (dv_dw[:, :half_C_dh].mean(dim=1) - dv_dh[:, half_C_dh:].mean(dim=1))
    return curl.pow(2).mean().item()


def run_inference_with_physics_tracking(
    prompt: str,
    checkpoint_path: str,
    output_dir: str,
    model_id: str = "Wan-AI/Wan2.2-T2V-A14B",
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    seed: int = 0,
    device: str = "cuda",
):
    """
    运行推理并在每个去噪步骤记录 latent 物理场 PDE 残差。
    对比有无 PINN 的残差，画图说明 PINN 的作用。
    """
    from diffsynth import save_video
    from diffsynth.pipelines.wan_video_pinn import PhysicsInformedWanVideoPipeline
    from diffsynth.pipelines.wan_video_new import ModelConfig

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[Latent Physics Tracking] prompt: {prompt[:60]}")
    print(f"{'='*60}")

    # ── 加载模型
    pipe = PhysicsInformedWanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id=model_id,
                        origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
                        offload_device="cpu"),
            ModelConfig(model_id=model_id,
                        origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
                        offload_device="cpu"),
            ModelConfig(model_id=model_id,
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                        offload_device="cpu"),
            ModelConfig(model_id=model_id,
                        origin_file_pattern="Wan2.1_VAE.pth",
                        offload_device="cpu"),
        ],
    )
    pipe.enable_vram_management()

    # ── Hook：记录每步的 v_original 和 v_corrected
    tracking = {
        "steps": [],
        "div_original": [],
        "div_corrected": [],
        "vor_original": [],
        "vor_corrected": [],
        "scale_history": [],
        "correction_ratio": [],
    }

    # 先加载 PINN plugin
    pipe.load_pinn_plugin(checkpoint_path, device=device)

    # 再包一层记录 hook
    original_model_fn = pipe.model_fn
    step_counter = [0]

    def model_fn_tracking(**kwargs):
        v_out = original_model_fn(**kwargs)
        # NOTE: original_model_fn 已经是 adapter-wrapped 版本；
        # 我们需要在 adapter 内部拿到 v_original 和 v_corrected
        # 此处通过重新 hook adapter 获取（见下方）
        return v_out

    # 重新 hook：更细粒度地拿到 before/after adapter
    if pipe.physics_adapter is not None:
        original_adapter_forward = pipe.physics_adapter.forward
        adapter = pipe.physics_adapter
        base_model_fn = pipe.model_fn  # 已经是 wrapped 版本

        # 解包：找到 unwrapped 的原始 model_fn
        # 我们需要重新 wrap 以同时记录 before/after
        # 策略：再次包装，在 adapter 调用前后记录

        # 保存 adapter 的原始 forward
        orig_adapter_fwd = adapter.forward.__func__  # unbound

        def patched_adapter_forward(self_adapter, v_original, z_t, metadata=None, material_id=None):
            v_corrected = orig_adapter_fwd(
                self_adapter,
                v_original,
                z_t,
                metadata=metadata,
                material_id=material_id,
            )

            with torch.no_grad():
                step = step_counter[0]
                tracking["steps"].append(step)
                tracking["div_original"].append(compute_latent_divergence(v_original.float()))
                tracking["div_corrected"].append(compute_latent_divergence(v_corrected.float()))
                tracking["vor_original"].append(compute_latent_vorticity(v_original.float()))
                tracking["vor_corrected"].append(compute_latent_vorticity(v_corrected.float()))
                tracking["scale_history"].append(self_adapter.scale.item())
                diff = (v_corrected - v_original).abs().mean().item()
                ratio = diff / (v_original.abs().mean().item() + 1e-10)
                tracking["correction_ratio"].append(ratio)
                step_counter[0] += 1

            return v_corrected

        import types
        adapter.forward = types.MethodType(patched_adapter_forward, adapter)

    # ── 生成视频（PINN 版）
    print("\n[1/2] Generating PINN video (with physics correction)...")
    video_pinn = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        seed=seed,
        cfg_scale=5.0,
        tiled=True,
    )
    pinn_path = os.path.join(output_dir, "video_pinn.mp4")
    save_video(video_pinn, pinn_path, fps=15, quality=5)
    print(f"  Saved → {pinn_path}")

    # ── 可视化 latent 物理场随去噪步变化
    if tracking["steps"]:
        _plot_latent_residuals(tracking, output_dir)

    return tracking, pinn_path


def _plot_latent_residuals(tracking: dict, output_dir: str):
    """画 latent 空间物理残差随去噪步的变化曲线。"""
    steps = tracking["steps"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Latent-Space PDE Residuals During Denoising\n"
                 "(lower = more physics-consistent velocity prediction)", fontsize=13)

    ax = axes[0, 0]
    ax.plot(steps, tracking["div_original"], "b--o", markersize=4, label="Baseline (no PINN)", alpha=0.8)
    ax.plot(steps, tracking["div_corrected"], "r-o", markersize=4, label="PINN corrected", alpha=0.8)
    ax.set_title("Divergence² mean (↓ = more incompressible)")
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("|∇·v|² mean (latent)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log" if min(tracking["div_original"] + tracking["div_corrected"]) > 0 else "linear")

    ax = axes[0, 1]
    ax.plot(steps, tracking["vor_original"], "b--o", markersize=4, label="Baseline", alpha=0.8)
    ax.plot(steps, tracking["vor_corrected"], "r-o", markersize=4, label="PINN corrected", alpha=0.8)
    ax.set_title("Vorticity² mean (rotational energy)")
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("|ω|² mean (latent)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, tracking["scale_history"], "g-o", markersize=4)
    ax.set_title("adapter.scale over denoising steps\n(= learned correction strength)")
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("scale value")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, [r * 100 for r in tracking["correction_ratio"]], "m-o", markersize=4)
    ax.set_title("PINN correction ratio: |Δv| / |v_orig| (%)")
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("correction (%)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "latent_residuals.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved latent residuals plot → {path}")

    # 打印统计摘要
    n = len(steps)
    if n > 0:
        print(f"\n  [Latent Physics Summary]")
        print(f"    Divergence  baseline avg = {np.mean(tracking['div_original']):.6f}")
        print(f"    Divergence  PINN avg     = {np.mean(tracking['div_corrected']):.6f}")
        reduce_pct = (1 - np.mean(tracking['div_corrected']) /
                      (np.mean(tracking['div_original']) + 1e-10)) * 100
        print(f"    Divergence reduction     = {reduce_pct:.2f}%  (正值 = PINN 确实减少了物理违约)")
        print(f"    adapter.scale avg        = {np.mean(tracking['scale_history']):.6f}")
        print(f"    correction ratio avg     = {np.mean(tracking['correction_ratio'])*100:.4f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: A/B Comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_videos(video_pinn: str, video_base: str, output_dir: str):
    """对比两个视频的光流物理指标，画对比图。"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"[A/B Comparison]")
    print(f"  PINN video : {video_pinn}")
    print(f"  Base video : {video_base}")
    print(f"{'='*60}")

    stats_pinn = visualize_flow_physics(video_pinn, output_dir, tag="PINN")
    stats_base = visualize_flow_physics(video_base, output_dir, tag="Baseline")

    if not stats_pinn or not stats_base:
        print("ERROR: Could not compute stats for one or both videos")
        return

    # 对比条形图
    metrics = ["div_mean", "vor_mean", "mag_mean"]
    labels  = ["Divergence |∇·v| ↓", "Vorticity |ω|", "Flow Speed (px/fr)"]
    colors  = ["steelblue", "tomato", "seagreen"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Physics Metrics: PINN vs Baseline", fontsize=14)

    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        vals = [stats_base[metric], stats_pinn[metric]]
        bars = ax.bar(["Baseline", "PINN"], vals, color=["lightgray", color], edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=10)
        ax.set_title(label)
        ax.set_ylabel("value")
        ax.grid(True, axis="y", alpha=0.3)

        # 标注相对变化
        if stats_base[metric] > 1e-10:
            delta = (stats_pinn[metric] - stats_base[metric]) / stats_base[metric] * 100
            color_txt = "green" if (metric == "div_mean" and delta < 0) else "black"
            ax.text(0.5, 0.95, f"Δ = {delta:+.1f}%", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10,
                    color=color_txt, fontweight="bold")

    plt.tight_layout()
    cmp_path = os.path.join(output_dir, "ab_comparison.png")
    plt.savefig(cmp_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved A/B comparison → {cmp_path}")

    # 打印摘要
    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │           Physics Metrics Summary (A/B)             │")
    print(f"  ├──────────────────┬────────────┬────────────┬────────┤")
    print(f"  │ Metric           │  Baseline  │    PINN    │  Δ%    │")
    print(f"  ├──────────────────┼────────────┼────────────┼────────┤")
    for metric, label in zip(metrics, ["Divergence", "Vorticity ", "Flow Speed"]):
        b = stats_base[metric]
        p = stats_pinn[metric]
        d = (p - b) / (b + 1e-10) * 100
        arrow = "↓" if d < 0 else "↑"
        print(f"  │ {label:16s} │  {b:8.4f}  │  {p:8.4f}  │{d:+6.1f}{arrow} │")
    print(f"  └──────────────────┴────────────┴────────────┴────────┘")
    print(f"\n  解读：")
    print(f"    - Divergence ↓ : 速度场更接近不可压缩（流体物理约束更好）")
    print(f"    - Vorticity     : 涡旋结构对比（流体应有合理的涡旋分布）")
    print(f"    - Flow Speed    : 整体运动幅度对比")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PINN Physics Field Verification & Visualization")
    parser.add_argument("--mode", choices=["flow", "latent", "compare"], default="flow",
                        help="验证模式: flow=像素空间光流分析 | latent=去噪过程中的latent残差追踪 | compare=A/B视频对比")

    # flow / compare 模式
    parser.add_argument("--video", type=str, help="输入视频路径（flow 模式）")
    parser.add_argument("--video_pinn", type=str, help="PINN 生成视频路径（compare 模式）")
    parser.add_argument("--video_base", type=str, help="基线视频路径（compare 模式）")

    # latent 模式
    parser.add_argument("--prompt", type=str, default="water flowing down",
                        help="生成 prompt（latent 模式）")
    parser.add_argument("--checkpoint", type=str,
                        default="models/train/pinn_plugin_low_noise/pinn_plugin_final.pt",
                        help="PINN checkpoint 路径（latent 模式）")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-T2V-A14B")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--output", type=str, default="verify_output",
                        help="可视化输出目录")

    args = parser.parse_args()

    if args.mode == "flow":
        if not args.video:
            parser.error("--video is required for flow mode")
        visualize_flow_physics(args.video, args.output)

    elif args.mode == "latent":
        ckpt = args.checkpoint
        if not Path(ckpt).is_absolute():
            ckpt = str(project_root / ckpt)
        run_inference_with_physics_tracking(
            prompt=args.prompt,
            checkpoint_path=ckpt,
            output_dir=args.output,
            model_id=args.model_id,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            device=args.device,
        )

    elif args.mode == "compare":
        if not args.video_pinn or not args.video_base:
            parser.error("--video_pinn and --video_base are required for compare mode")
        compare_videos(args.video_pinn, args.video_base, args.output)


if __name__ == "__main__":
    main()
