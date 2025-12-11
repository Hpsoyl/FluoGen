import os
import re
import glob
import torch
import logging
import imageio
import argparse
import warnings
import accelerate
import torchvision.transforms.functional as TF

from skimage import io
from torchvision import transforms
from transformers import AutoTokenizer
from transformers import CLIPTextModel
from diffusers.utils.torch_utils import randn_tensor
from models.controlnet import ControlNetModel
from models.unet_2d_condition import UNet2DConditionModel
from models.pipeline_controlnet import DDPMControlnetPipeline
from diffusers import DDPMScheduler, DDIMScheduler
from utils.process import *
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate.state import AcceleratorState
from transformers.utils import ContextManagers


os.environ["DIFFUSERS_NO_LOGS"] = "true"
logging.basicConfig(level=logging.ERROR)


def tansform_condition(example, weight_dtype, device):
    image = example.astype('float32') / 255
    tensor = torch.from_numpy(image).unsqueeze(0).to(dtype=weight_dtype, device=device)
    return tensor


def get_validation_prompt(base_name):
    text_rules = {
        'MICE': 'mouse brain tissues',
        'FISH': 'zebrafish embryos',
        'BPAE_B': 'nucleus of BPAE',
        'BPAE_R': 'mitochondria of BPAE',
        'BPAE_G': 'F-actin of BPAE'
    }

    for key, text in text_rules.items():
        if key in base_name:
            return text
    
    warnings.warn(f'No match for prompt in base_name: {base_name}', UserWarning)
    return 'unknown'
    

def log_validation(args, weight_dtype, device):
    psnr_max, ssim_max, nrmse_max = 0, 0, 1
    unet = UNet2DConditionModel.from_pretrained(
        args.unet_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model_path,
        subfolder="controlnet",
    ).to(dtype=weight_dtype, device=device)
    scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type=args.prediction_type,
        rescale_betas_zero_snr=args.rescale_betas_zero_snr,
        timestep_spacing=args.ddpm_timestep_spacing
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.CLIP_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
    
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.CLIP_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    ).to(dtype=weight_dtype, device=device)
        
    pipeline = DDPMControlnetPipeline(
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    ).to(dtype=weight_dtype, device=device)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    pipeline.set_progress_bar_config(disable=True)

    images_path = glob.glob(f'{args.validation_path}/*.png')
    os.makedirs(args.validation_output_dir, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    for iidx, image_path in enumerate(images_path):
        print('\r[%d/%d] Reconstructing %s' % (iidx + 1, len(images_path), image_path), end='')
        basename = os.path.basename(image_path)

        validation_image = imageio.v2.imread(image_path)
        validation_image = tansform_condition(validation_image, weight_dtype, device)
        validation_prompt = get_validation_prompt(basename)

        with torch.autocast("cuda"):
            image = pipeline(
                prompt=validation_prompt,
                image_cond=validation_image,
                num_inference_steps=args.ddpm_num_inference_steps,
                generator=generator,
                output_type="np"
            ).images[0]

        image = (image * 255).astype("uint8").squeeze()      
        io.imsave(f'{args.validation_output_dir}/{basename[:-4]}.png', image)
    print()
    del pipeline
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_model_path",
                        type=str,
                        default="Pretrained FluoGen model path"
    )
    parser.add_argument("--controlnet_model_path",
                        type=str,
                        default="FluoGen control branch model path"
    )
    parser.add_argument(
        "--CLIP_path",
        type=str,
        default="stable-diffusion-v1-5",
        help="Path to pretrained CLIP model"
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--rescale_betas_zero_snr", action="store_false", help="Whether or not to rescale betas to zero snr."
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="v_prediction",
        choices=["epsilon", "sample", "v_prediction"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )

    parser.add_argument("--validation_path", type=str, nargs="+", default="FMD/test_mix/raw")
    parser.add_argument("--validation_output_dir", type=str, default="validation_output/FMD")

    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=9)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument("--ddpm_timestep_spacing", type=str, default="trailing")

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    device = "cuda:0"
    weight_dtype = torch.float16

    args = parser.parse_args()
    image_validate = log_validation(
        args,
        weight_dtype,
        device,
    )
    
