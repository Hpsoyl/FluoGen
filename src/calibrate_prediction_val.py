import os
import glob
import torch
import imageio
import accelerate
import tifffile

from torchvision import transforms
from diffusers import DDPMScheduler
from accelerate.state import AcceleratorState
from transformers.utils import ContextManagers
from transformers import CLIPTextModel, CLIPTokenizer 

from models.controlnet import ControlNetModel
from models.unet_2d_uncertainty import UNet2DUncertainty
from models.pipeline_uncertainty import DDPMUncertaintyPipeline

CLIP_path="/data0/syhong/stable-diffusion-v1-5"
specimen = "FMD"
device = "cuda:5"

# CCPs
if specimen == "CCPs":
    unet_model_path="model_output/BIDM_uncertainty_BioSR_CCPs/checkpoint-24000"
    controlnet_model_path="model_output/BIDM_controlnet_CCPs_SIM_SR_temp_3_3/checkpoint-98000"
    validation_prompt = "CCPs of COS-7"
# F-actin
elif specimen == "F-actin":
    unet_model_path="model_output/BIDM_uncertainty_BioSR_Factin/checkpoint-25000"
    controlnet_model_path="model_output/BIDM_SR_Factin/checkpoint-35000"
    validation_prompt = "Factin of COS-7"
# ER
elif specimen == "ER":
    unet_model_path="model_output/BIDM_uncertainty_BioSR_ER/checkpoint-24000"
    controlnet_model_path="model_output/BIDM_controlnet_ER_SIM_SR/checkpoint-30000"
    validation_prompt = "ER of COS-7"
# Microtubules
elif specimen == "Microtubules":
    unet_model_path="model_output/BIDM_uncertainty_BioSR_CCPs/checkpoint-24000"
    controlnet_model_path="model_output/BIDM_controlnet_Microtubules_SIM_SR/checkpoint-72500"
    validation_prompt = "Microtubules of COS-7"
elif specimen == "FMD":
    unet_model_path="/data0/syhong/BioDiff/model_output/BIDM_uncertainty_FMD/checkpoint-8000"
    controlnet_model_path="/data0/syhong/BioDiff/model_output/BIDM_denosing_FMD_raw_wo_19thFOV/checkpoint-10000"
    validation_prompt = ""

ddpm_num_steps = 1000
ddpm_beta_schedule = "linear"
prediction_type = "v_prediction"
rescale_betas_zero_snr = True
ddpm_timestep_spacing = "trailing"
revision = None
variant = None

weight_dtype = torch.float16

unet = UNet2DUncertainty.from_pretrained(
    unet_model_path,
    subfolder="unet",
).to(dtype=weight_dtype, device=device)
controlnet = ControlNetModel.from_pretrained(
    controlnet_model_path,
    subfolder="controlnet",
).to(dtype=weight_dtype, device=device)
scheduler = DDPMScheduler(
    num_train_timesteps=ddpm_num_steps,
    beta_schedule=ddpm_beta_schedule,
    prediction_type=prediction_type,
    rescale_betas_zero_snr=rescale_betas_zero_snr,
    timestep_spacing=ddpm_timestep_spacing
)
tokenizer = CLIPTokenizer.from_pretrained(
    CLIP_path, subfolder="tokenizer", revision=revision
)

def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

# Load text encoder
with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    text_encoder = CLIPTextModel.from_pretrained(
        CLIP_path, subfolder="text_encoder", revision=revision, variant=variant
    ).to(dtype=weight_dtype, device=device)

pipeline = DDPMUncertaintyPipeline(
    unet=unet,
    controlnet=controlnet,
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
).to(dtype=weight_dtype, device=device)


def tansform_condition(example, weight_dtype, device, specimen):
    if specimen == "FMD":
        image = example.astype('float32') / 255
        tensor = torch.from_numpy(image).unsqueeze(0).to(dtype=weight_dtype, device=device)
    else:
        image = example.astype('float32') / 65535
        tensor = torch.from_numpy(image).unsqueeze(0).to(dtype=weight_dtype, device=device)
        resize = transforms.Resize((512, 512))
        tensor = resize(tensor)
    return tensor


if __name__ == "__main__":
    unet.eval()

    ddpm_num_inference_steps = 10
    pipeline.set_progress_bar_config(disable=True)

    # validation_path = f"/lab310/enhance/BioDiffuse_Base/bioSR_512/{specimen}/validate/validate_9"
    validation_path = f"/lab310/enhance/BioDiffuse_Base/FMD/test_mix/raw/"
    validation_output_dir = f"/data0/syhong/BioDiff/validation_output/Uncertainty_{specimen}_prediction_val"
    validation_prompt = validation_prompt

    os.makedirs(validation_output_dir, exist_ok=True)
    images_path = glob.glob(os.path.join(validation_path, "*.png"))

    for iidx, image_path in enumerate(images_path):
        print('\r[%d/%d] Reconstructing %s' % (iidx + 1, len(images_path), image_path), end='')

        basename = os.path.basename(image_path)
        image_path = [image_path]

        for validation_image in image_path:
            validation_image = imageio.v2.imread(validation_image)
            validation_image = tansform_condition(validation_image, weight_dtype, device, specimen)
            
            with torch.autocast("cuda"):
                image = pipeline(
                    prompt=validation_prompt,
                    image_cond=validation_image,
                    num_inference_steps=ddpm_num_inference_steps,
                    generator=None,
                    output_type="np"
                ).images
            tifffile.imwrite(f'{validation_output_dir}/{basename[:-4]}.tif', image, )
                
    del pipeline
    torch.cuda.empty_cache()


