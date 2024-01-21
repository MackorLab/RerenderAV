from diffusers import DiffusionPipeline, ControlNetModel, AutoencoderKL, DDIMScheduler
from diffusers.utils import export_to_gif
import numpy as np
import torch
import cv2
from PIL import Image

def video_to_frame(video_path: str, interval: int):
    vidcap = cv2.VideoCapture(video_path)
    success = True

    count = 0
    res = []
    while success:
        count += 1
        success, image = vidcap.read()
        if count % interval != 1:
            continue
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res.append(image)

    vidcap.release()
    return res

input_video_path = 'dance512.mp4'
input_interval = 2
frames = video_to_frame(
    input_video_path, input_interval)

control_frames = []

for frame in frames:
    np_image = cv2.Canny(frame, 50, 100)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    canny_image = Image.fromarray(np_image)
    control_frames.append(canny_image)

controlnet = ControlNetModel.from_pretrained(
    "controlnet/control_v11p_sd15_canny",
    torch_dtype=torch.float16
)
pipe = DiffusionPipeline.from_pretrained(
    "model/mistoonAnimev20_ema",
    controlnet=controlnet,
    custom_pipeline='custom_pipeline/rerender_a_video.py',
    torch_dtype=torch.float16
)
pipe.vae = AutoencoderKL.from_single_file(
    "vae/vae-ft-mse-840000-ema-pruned.safetensors",
    torch_dtype=torch.float16
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

generator = torch.manual_seed(0)
frames = [Image.fromarray(frame) for frame in frames]
output_frames = pipe(
    prompt="anime style, high quality, best quality, man with black hair, wearing sunglasses, black sweater, dancing",
    negative_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
    frames=frames,
    control_frames=control_frames,
    num_inference_steps=20,
    strength=0.6,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.7,
    generator=generator,
    warp_start=0.0,
    warp_end=0.1,
    mask_start=0.5,
    mask_end=0.8,
    mask_strength=0.1
).frames

export_to_gif(output_frames, "result.gif")
