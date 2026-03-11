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


input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0001.png").resize((1248, 704))
video = pipe(
    prompt="A chair is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0001.mp4", fps=15, quality=5)

input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0002.png").resize((1248, 704))
video = pipe(
    prompt="A chair is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0002.mp4", fps=15, quality=5)


input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0003.png").resize((1248, 704))
video = pipe(
    prompt="A bottle is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0003.mp4", fps=15, quality=5)


input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0004.png").resize((1248, 704))
video = pipe(
    prompt="A bottle is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0004.mp4", fps=15, quality=5)

input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0005.png").resize((1248, 704))
video = pipe(
    prompt="A cup is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0005.mp4", fps=15, quality=5)


input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0006.png").resize((1248, 704))
video = pipe(
    prompt="A cup is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0006.mp4", fps=15, quality=5)


input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0007.png").resize((1248, 704))
video = pipe(
    prompt="A hammer is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0007.mp4", fps=15, quality=5)

input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0008.png").resize((1248, 704))
video = pipe(
    prompt="A ball is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0008.mp4", fps=15, quality=5)

input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0009.png").resize((1248, 704))
video = pipe(
    prompt="A wheel is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0009.mp4", fps=15, quality=5)

input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0010.png").resize((1248, 704))
video = pipe(
    prompt="A cushion is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0010.mp4", fps=15, quality=5)

input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0011.png").resize((1248, 704))
video = pipe(
    prompt="A basket is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0011.mp4", fps=15, quality=5)

input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0012.png").resize((1248, 704))
video = pipe(
    prompt="A basket is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0012.mp4", fps=15, quality=5)

input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0013.png").resize((1248, 704))
video = pipe(
    prompt="A lamp is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0013.mp4", fps=15, quality=5)

input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0014.png").resize((1248, 704))
video = pipe(
    prompt="A chair is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0014.mp4", fps=15, quality=5)

input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0015.png").resize((1248, 704))
video = pipe(
    prompt="A chair is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0015.mp4", fps=15, quality=5)

input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0016.png").resize((1248, 704))
video = pipe(
    prompt="A hammer is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0016.mp4", fps=15, quality=5)

input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data_iclr/img/0017.png").resize((1248, 704))
video = pipe(
    prompt="A ball is falling onto the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "iclr/0017.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/cvpr/CogVideo/eval/images/0018.png").resize((1248, 704))
# video = pipe(
#     prompt="a badminton ball falls onto the ground",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "cvpr/0018.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/cvpr/CogVideo/eval/images/0019.png").resize((1248, 704))
# video = pipe(
#     prompt="a glass of water falls onto the ground",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "cvpr/0019.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/cvpr/CogVideo/eval/images/0020.png").resize((1248, 704))
# video = pipe(
#     prompt="A towel fell onto the table.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "cvpr/0020.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/cvpr/CogVideo/eval/images/0021.png").resize((1248, 704))
# video = pipe(
#     prompt="a toy rabbit falls onto the ground and bounds back up",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "cvpr/0021.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/cvpr/CogVideo/eval/images/0022.png").resize((1248, 704))
# video = pipe(
#     prompt="a soccer ball falls onto the soccer field and bounds up",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "cvpr/0022.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/cvpr/CogVideo/eval/images/0023.png").resize((1248, 704))
# video = pipe(
#     prompt="A pack of tissues fell from the table to the floor.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "cvpr/0023.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/cvpr/CogVideo/eval/images/0024.png").resize((1248, 704))
# video = pipe(
#     prompt="A flower fell to the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "cvpr/0024.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/cvpr/CogVideo/eval/images/0025.png").resize((1248, 704))
# video = pipe(
#     prompt="A cushion fell to the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "cvpr/0025.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/cvpr/CogVideo/eval/images/0026.png").resize((1248, 704))
# video = pipe(
#     prompt="A cushion fell to the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "cvpr/0026.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/cvpr/CogVideo/eval/images/0027.png").resize((1248, 704))
# video = pipe(
#     prompt="A hat fell to the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "cvpr/0027.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/cvpr/CogVideo/eval/images/0028.png").resize((1248, 704))
# video = pipe(
#     prompt="A baseball cap fell to the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "cvpr/0028.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/cvpr/CogVideo/eval/images/0029.png").resize((1248, 704))
# video = pipe(
#     prompt="A wooden basin fell to the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "cvpr/0029.mp4", fps=15, quality=5)

input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/cvpr/CogVideo/eval/images/0030.png").resize((1248, 704))
video = pipe(
    prompt="A wooden basin fell to the ground.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
)
save_video(video, "cvpr/0030.mp4", fps=15, quality=5)

# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/cvpr/CogVideo/eval/images/0031.png").resize((1248, 704))
# video = pipe(
#     prompt="A doll fell to the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "cvpr/0031.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/projects/PhysGen3D/data/img/scene30_fixed.png").resize((1248, 704))
# video = pipe(
#     prompt="With a simple, modern room with minimalist furniture and decor, the blue bottle standing on the ground fell and rolled on the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "scene30_fixed.mp4", fps=15, quality=5)


