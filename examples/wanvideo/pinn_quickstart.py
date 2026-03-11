"""
PINN Video Generation - Quick Start
快速开始脚本

运行方式:
    python pinn_quickstart.py
"""
import torch
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from diffsynth import save_video
from diffsynth.pipelines.wan_video_pinn import PhysicsInformedWanVideoPipeline
from diffsynth.pipelines.wan_video_new import ModelConfig


def main():
    print("=" * 80)
    print("PINN for Video Generation - Quick Start Demo")
    print("=" * 80)
    
    # 1. 加载模型
    print("\n[1/3] Loading model (this may take a few minutes)...")
    pipe = PhysicsInformedWanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.2-T2V-A14B",
                origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
                offload_device="cpu"
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-T2V-A14B",
                origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
                offload_device="cpu"
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-T2V-A14B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu"
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-T2V-A14B",
                origin_file_pattern="Wan2.1_VAE.pth",
                offload_device="cpu"
            ),
        ],
    )
    pipe.enable_vram_management()
    print("✓ Model loaded successfully")
    
    # 2. 生成示例视频
    print("\n[2/3] Generating videos with physics-aware prompts...")
    
    examples = [
        {
            'prompt': 'water flowing down from a waterfall',
            'output': 'pinn_demo_water.mp4',
            'expected_material': 'fluid'
        },
        {
            'prompt': 'a red ball falling and bouncing on the ground',
            'output': 'pinn_demo_ball.mp4',
            'expected_material': 'rigid'
        },
        {
            'prompt': 'a white cloth waving in the wind',
            'output': 'pinn_demo_cloth.mp4',
            'expected_material': 'elastic'
        },
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n  [{i}/{len(examples)}] Generating: {example['prompt']}")
        
        # 识别材质
        material = pipe.material_classifier.classify(example['prompt'])
        print(f"      Detected material: {material} (expected: {example['expected_material']})")
        
        # 生成视频
        video = pipe(
            prompt=example['prompt'],
            negative_prompt="",
            seed=42,
            height=480,
            width=832,
            num_frames=81,
            num_inference_steps=30,  # 快速演示用较少步数
            tiled=True,
        )
        
        # 保存
        save_video(video, example['output'], fps=15, quality=5)
        print(f"      ✓ Saved to {example['output']}")
    
    # 3. 完成
    print("\n[3/3] Demo completed!")
    print("\n" + "=" * 80)
    print("What's next?")
    print("-" * 80)
    print("1. Check the generated videos")
    print("2. Try training PINN plugin:")
    print("   cd pinn_training")
    print("   python train_pinn.py --help")
    print("3. Read PINN_README.md for detailed documentation")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
