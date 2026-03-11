import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

# # Text-to-video
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     num_frames=121,
# )
# save_video(video, "video1.mp4", fps=15, quality=5)

# Image-to-video
# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=["data/examples/wan/cat_fightning.jpg"]
# )

input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data/img/scene12.png").resize((512, 512))
video = pipe(
    prompt="阳台上的衣服随风舞动",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=8, tiled=True,
    height=512, width=512,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "scene12_llllllllll.mp4", fps=15, quality=5)

input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data/img/scene12.png").resize((512, 512))
video = pipe(
    prompt="阳台上的衣服随风舞动",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=9, tiled=True,
    height=512, width=512,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "scene12_lllllllllll.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data/img/scene23_fixed.png").resize((512, 512))
# video = pipe(
#     prompt="With a cake on the plate, yellow jam is spread on the cake.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=512, width=512,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "scene23_fixed.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data/img/scene24_fixed.png").resize((512, 512))
# video = pipe(
#     prompt="With a field for playing frisbee in the scene, the frisbee flew down to the ground and wobbles back and forth.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=512, width=512,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "scene24_fixed.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data/img/scene25_fixed.png").resize((512, 512))
# video = pipe(
#     prompt="With the background of a tidy desk, a computer mouse moves smoothly on the non-slip mouse pad.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=512, width=512,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "scene25_fixed.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data/img/scene29_fixed.png").resize((512, 512))
# video = pipe(
#     prompt="With a tidy and well-maintained floor in the scene, a bouncy ball rebounds off the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=512, width=512,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "scene29_fixed.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data/img/scene30_1_fixed.png").resize((512, 512))
# video = pipe(
#     prompt="With a simple, modern room with minimalist furniture and decor, the blue bottle standing on the ground fell and rolled on the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=512, width=512,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "scene30_1_fixed.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data/img/scene31_fixed.png").resize((512, 512))
# video = pipe(
#     prompt="With a wooden table in the scene, the chocolate on the table melted.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=512, width=512,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "scene31_fixed.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data/img/scene33_fixed.png").resize((512, 512))
# video = pipe(
#     prompt="With an old stone bridge spans a clear stream with gently flowing water in the scene, a boat float on the river.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=512, width=512,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "scene33_fixed.mp4", fps=15, quality=5)
