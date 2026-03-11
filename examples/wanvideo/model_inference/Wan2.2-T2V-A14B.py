import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

# Text-to-video
video = pipe(
    prompt="The video captures a serene mountain landscape featuring a tranquil lake nestled at its base. The scene is dominated by a majestic mountain range in the background, with snow-capped peaks partially obscured by low-hanging clouds. The foreground is filled with a calm, reflective lake that mirrors the surrounding trees and mountains, creating a symmetrical and peaceful visual effect. The trees lining the shore are dense and lush, primarily coniferous, adding to the natural beauty of the setting. The sky above is a soft gradient of blue and pink hues, suggesting either early morning or late afternoon light. There are no visible animals or characters in the video; it focuses entirely on the natural elements. The camera remains stationary throughout the sequence, allowing viewers to fully absorb the stillness and tranquility of the scene. The overall atmosphere is one of quiet contemplation and natural splendor.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
)
save_video(video, "video6.mp4", fps=15, quality=5)
