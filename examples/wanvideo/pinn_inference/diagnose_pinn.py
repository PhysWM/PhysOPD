"""
PINN Plugin Diagnostic Script
诊断 PINN 插件是否真正起作用
"""
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def diagnose(checkpoint_path):
    print("=" * 80)
    print("PINN Plugin Diagnostic")
    print("=" * 80)
    
    # 1. 加载 checkpoint
    print(f"\nLoading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # 2. 检查 config
    if 'config' in ckpt:
        print(f"\n[Config]")
        for k, v in ckpt['config'].items():
            print(f"  {k}: {v}")
    
    # 3. 检查 PhysicsAdapter 参数
    if 'physics_adapter_state_dict' in ckpt:
        print(f"\n[PhysicsAdapter Parameters]")
        adapter_sd = ckpt['physics_adapter_state_dict']
        print(f"  Total keys: {len(adapter_sd)}")
        
        for name, param in adapter_sd.items():
            is_zero = (param.abs().max().item() == 0)
            print(f"  {name:50s} | shape={str(list(param.shape)):20s} | "
                  f"min={param.min().item():+.6f} | max={param.max().item():+.6f} | "
                  f"mean={param.mean().item():+.6f} | std={param.std().item():.6f} | "
                  f"{'*** ALL ZERO ***' if is_zero else 'OK'}")
        
        # 重点检查 scale 参数
        if 'scale' in adapter_sd:
            scale_val = adapter_sd['scale'].item()
            print(f"\n  >>> CRITICAL: scale = {scale_val:.8f}")
            if scale_val == 0.0:
                print(f"  >>> WARNING: scale is exactly 0.0!")
                print(f"  >>> This means PhysicsAdapter correction is completely disabled!")
                print(f"  >>> The adapter output = v_original + 0 * correction = v_original")
            elif abs(scale_val) < 1e-6:
                print(f"  >>> WARNING: scale is very small ({scale_val:.2e})")
                print(f"  >>> The adapter correction is negligible")
            else:
                print(f"  >>> OK: scale is non-zero, adapter is active")
        else:
            print(f"\n  >>> WARNING: 'scale' not found in adapter state dict!")
    else:
        print(f"\n  WARNING: No physics_adapter_state_dict in checkpoint!")
    
    # 4. 检查 PDE Residuals 参数
    if 'pde_residuals_state_dict' in ckpt:
        print(f"\n[PDE Residuals Parameters]")
        pde_sd = ckpt['pde_residuals_state_dict']
        for name, param in pde_sd.items():
            print(f"  {name:30s} | value={param.item():.8f}")
    
    # 5. 总结
    print(f"\n{'=' * 80}")
    print("Summary:")
    
    adapter_sd = ckpt.get('physics_adapter_state_dict', {})
    all_zero_count = sum(1 for p in adapter_sd.values() if p.abs().max().item() == 0)
    total_count = len(adapter_sd)
    
    if total_count == 0:
        print("  NO adapter parameters found!")
    elif all_zero_count == total_count:
        print("  ALL parameters are zero - adapter has NOT been trained!")
    elif all_zero_count > 0:
        print(f"  {all_zero_count}/{total_count} parameters are zero")
    else:
        print(f"  All {total_count} parameters are non-zero - adapter has been trained")
    
    scale_val = adapter_sd.get('scale', torch.tensor(0.0)).item()
    if scale_val == 0.0:
        print("  scale = 0.0 --> adapter correction is INACTIVE (no effect)")
    else:
        print(f"  scale = {scale_val:.8f} --> adapter correction is ACTIVE")
    
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str, help="Path to PINN plugin checkpoint")
    args = parser.parse_args()
    diagnose(args.checkpoint_path)
