import numpy as np
import argparse

from skimage import io
from diffusers import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from models.pipeline_ddpm_text_encoder import DDPMPipeline
from models.unet_2d import UNet2DModel

from utils.process import *

def min_max_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def parse_args():
    parser = argparse.ArgumentParser(description="DDPM Text-to-Image Generation")
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default="Nucleus of BPAE",
        help="The prompt or description for generating the image (e.g., 'CCPs of COS-7')."
    )
    parser.add_argument(
        "--check_point",
        type=str,
        default="Pretrained FluoGen model path",
        help="Checkpoint to load the model from."
    )
    parser.add_argument(
        "--ddpm_num_steps",
        type=int,
        default=1000,
        help="Number of DDPM training timesteps."
    )
    parser.add_argument(
        "--ddpm_num_inference_steps",
        type=int,
        default=20,
        help="Number of DDPM inference timesteps."
    )
    parser.add_argument(
        "--ddpm_beta_schedule",
        type=str,
        default="linear",
        help="Beta schedule for DDPM."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for inference (e.g., 'cuda:0' or 'cpu')."
    )
    return parser.parse_args()


def convert_to_rgb(image):
    return np.repeat(image, 3, axis=-1)


args = parse_args()
ddpm_num_steps = args.ddpm_num_steps
ddpm_num_inference_steps = args.ddpm_num_inference_steps
ddpm_beta_schedule = args.ddpm_beta_schedule
ddpm_timestep_spacing = "trailing"
prediction_type = "v_prediction"
pretrained_model_name_or_path = "stable-diffusion-v1-5"
unet_path = args.check_point
device=args.device

noise_scheduler = DDPMScheduler(
    num_train_timesteps=ddpm_num_steps,
    beta_schedule=ddpm_beta_schedule,
    prediction_type=prediction_type,
    rescale_betas_zero_snr=False,
    timestep_spacing=ddpm_timestep_spacing,
)

unet = UNet2DModel.from_pretrained(
    unet_path, 
    subfolder="unet"
)

text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", revision=None, variant=None
).to(device)
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer", revision=None
)

pipe = DDPMPipeline(
    unet=unet,
    scheduler=noise_scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer
)
pipe.safety_checker = lambda images, clip_input: (images, None)
pipe.to(device)

validation_prompts = args.validation_prompts

pipe.set_progress_bar_config(disable=False)
image = pipe(
    validation_prompts,
    generator=None,
    num_inference_steps=ddpm_num_inference_steps,
    output_type="np",
).images[0]

image = convert_to_rgb((min_max_norm(image) * 255).astype("uint8"))
io.imsave(f"validation_output/Syn_out/temp.tif", image)
