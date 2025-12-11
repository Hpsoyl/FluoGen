import numpy as np
import argparse
import os
import cv2

from tqdm import tqdm
from skimage import io
from diffusers.schedulers import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from models.pipeline_ddpm_text_encoder import DDPMPipeline
from models.unet_2d import UNet2DModel

from utils.process import *

def min_max_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def parse_args():
    parser = argparse.ArgumentParser(description="DDPM Text-to-Image Generation")
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
        default=50,
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
        default="cuda:3",
        help="Device to use for inference (e.g., 'cuda:0' or 'cpu')."
    )
    return parser.parse_args()


def convert_to_rgb(image):
    return np.repeat(image, 3, axis=-1)


def get_folder(prompt: str) -> str:
    prompt_lower = prompt.lower()

    if "dna" in prompt_lower:
        return "dap"
    elif "er" in prompt_lower:
        return "erdak"
    elif "mitochondria" in prompt_lower:
        return "mc151"
    elif "lysosomes" in prompt_lower:
        return "h4b4"
    elif "golgi" in prompt_lower:
        if "gpp130" in prompt_lower:
            return "gpp130"
        else:
            return "giant"
    elif "endosomes" in prompt_lower:
        return "tfr"
    elif "microtubules" in prompt_lower:
        return "tubol"
    elif "nucleolus" in prompt_lower:
        return "nucle"
    elif "actin" in prompt_lower:
        return "phal"
    else:    
        print(f"unknown prompt: {prompt}")
        return "unknown"


args = parse_args()
prompts = [
    "dna of hela",
    "er of hela",
    "mitochondria of hela",
    "lysosomes of hela",
    "golgi of hela, gpp130",
    "golgi of hela, giant",
    "endosomes of hela",
    "microtubules of hela",
    "nucleolus of hela",
    "actin of hela"
]

check_point = args.check_point
ddpm_num_steps = args.ddpm_num_steps
ddpm_num_inference_steps = args.ddpm_num_inference_steps
ddpm_beta_schedule = args.ddpm_beta_schedule
ddpm_timestep_spacing = "trailing"
prediction_type = "v_prediction"
pretrained_model_name_or_path = "stable-diffusion-v1-5"
unet_path = check_point
device = args.device

noise_scheduler = DDPMScheduler(
    num_train_timesteps=ddpm_num_steps,
    beta_schedule=ddpm_beta_schedule,
    prediction_type=prediction_type,
    rescale_betas_zero_snr=True,
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

for prompt in prompts:
    folder_name = get_folder(prompt)
    os.makedirs(f"validation_output/HeLa_fewshot/20/{folder_name}", exist_ok=True)

    for i in tqdm(range(1, 1001), desc=f"Generating images for \"{prompt}\""):
        pipe.set_progress_bar_config(disable=True)
        image = pipe(
            prompt.lower(),
            generator=None,
            num_inference_steps=ddpm_num_inference_steps,
            output_type="np",
        ).images[0]
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = (min_max_norm(image) * 255).astype("uint8")
        io.imsave(f"validation_output/HeLa_fewshot/20/{folder_name}/{i+1:05d}.png", image, check_contrast=False)
