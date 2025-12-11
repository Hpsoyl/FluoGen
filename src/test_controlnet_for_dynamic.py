import os
import torch
import imageio
import argparse
from skimage import io
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler
from models.controlnet import ControlNetModel
from models.unet_2d_condition import UNet2DConditionModel
from models.pipeline_controlnet import DDPMControlnetPipeline
from transformers.utils import ContextManagers
from tqdm import tqdm

def transform_condition(example, weight_dtype, device):
    image = example.astype("float32")
    tensor = torch.from_numpy(image).unsqueeze(0).to(dtype=weight_dtype, device=device)
    return tensor

def log_validation(args, weight_dtype, device):
    unet = UNet2DConditionModel.from_pretrained(
        args.unet_model_path, subfolder="unet"
    ).to(dtype=weight_dtype, device=device)
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model_path, subfolder="controlnet"
    ).to(dtype=weight_dtype, device=device)
    scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type=args.prediction_type,
        rescale_betas_zero_snr=args.rescale_betas_zero_snr,
        timestep_spacing=args.ddpm_timestep_spacing,
    )
    tokenizer = CLIPTokenizer.from_pretrained(args.CLIP_path, subfolder="tokenizer", revision=args.revision)

    with ContextManagers([]):
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
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed else None
    
    base_path = args.train_data_path
    output_dir = args.validation_output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    cell_types = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    total_images = sum(len([f for f in os.listdir(os.path.join(base_path, d)) if f.endswith(".tif")]) for d in cell_types)
    
    with tqdm(total=total_images * 5, desc="Generating Images") as pbar:
        for cell_type in cell_types:
            cell_folder = os.path.join(base_path, cell_type)
            prompt = f"nuclei of {cell_type}"
            
            for tif_file in os.listdir(cell_folder):
                if not tif_file.endswith(".tif"):
                    continue
                
                image_path = os.path.join(cell_folder, tif_file)
                image = imageio.v2.imread(image_path)
                image = transform_condition(image, weight_dtype, device)
                
                filename = os.path.splitext(tif_file)[0]
                
                for i in range(6, 11):
                    save_path = os.path.join(output_dir, f"gen_0{i}_{filename}.tif")
                    if os.path.exists(save_path):
                        pbar.update(1)
                        continue

                    with torch.autocast("cuda"):
                        gen_image = pipeline(
                            prompt=prompt,
                            image_cond=image,
                            num_inference_steps=args.ddpm_num_inference_steps,
                            generator=generator,
                            output_type="np",
                        ).images[0]
                    
                    gen_image = (gen_image * 65535).astype("uint16")
                    save_path = os.path.join(output_dir, f"gen_0{i}_{filename}.tif")
                    io.imsave(save_path, gen_image, check_contrast=False)
                    pbar.update(1)
    
    del pipeline
    torch.cuda.empty_cache()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_model_path",
                        type=str,
                        default="Pretrained FluoGen foundation model path"
    )
    parser.add_argument(
        "--CLIP_path",
        type=str,
        default="stable-diffusion-v1-5",
        help="Path to pretrained CLIP model"
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
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
    parser.add_argument("--train_data_path", type=str, default="DynamicNuclearNet-segmentation-v1_0/train/y")
    parser.add_argument("--controlnet_model_path",
                        type=str,
                        default="FluoGen control branch model path"
    )
    parser.add_argument('--validation_output_dir', type=str, default="validation_output/DynamicNuclearNet-Seg/")
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=10)
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


    weight_dtype = torch.float16
    device = "cuda:7"

    args = parser.parse_args()
    image_validate = log_validation(
        args,
        weight_dtype,
        device,
    )
    
