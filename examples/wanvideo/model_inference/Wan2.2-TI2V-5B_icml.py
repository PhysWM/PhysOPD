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

prompt_path = "/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/prompt.txt"
prompt_list = open(prompt_path, "r").readlines()

for i in range(1, 34):
    # 读取对应行的prompt（索引从0开始，所以用i-1）
    prompt = prompt_list[i - 1].strip()
    
    img_path = f"/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/{i:03d}.png"
    input_image = Image.open(img_path).resize((1248, 704))
    
    print(f"Processing {i:03d}: {prompt}")
    
    video = pipe(
        prompt=prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
        seed=0, tiled=True,
        height=704, width=1248,
        input_image=input_image,
        num_frames=121,
    )
    save_video(video, f"icml/{i:03d}.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/001.png").resize((1248, 704))
# video = pipe(
#     prompt="A ball falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/001.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/003.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy dog falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/003.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/004.png").resize((1248, 704))
# video = pipe(
#     prompt="A yellow toy dog falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/004.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/005.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy goose falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/005.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/006.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy rooster falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/006.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/007.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy doll falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/007.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/008.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy pig falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/008.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/009.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy tiger falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/009.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/010.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy lion falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/010.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/011.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy deer falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/011.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)


# input_image = Image.open("/home/dataset-assist-0/algorithm/cong.wang/4dgen/ours/Phys4D/evaluation_dataset/with_bg/images_cropped/002.png").resize((1248, 704))
# video = pipe(
#     prompt="A toy duck falls onto the ground.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，变化的镜头，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     input_image=input_image,
#     num_frames=121,
# )
# save_video(video, "icml/002.mp4", fps=15, quality=5)





