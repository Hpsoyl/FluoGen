from typing import List, Optional, Tuple, Union
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.image_processor import PipelineImageInput
from transformers import CLIPTextModel, CLIPTokenizer
from models.controlnet import ControlNetModel
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from models.pipeline_ddpm_text_encoder import DDPMPipeline

import torch
import pdb
import skimage
import numpy as np

class DDPMUncertaintyPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet,
        scheduler,
        controlnet,
        text_encoder: CLIPTextModel | None = None,
        tokenizer: CLIPTokenizer | None = None
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            controlnet=controlnet,
        )

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        image_cond: PipelineImageInput = None,
        generator: torch.Generator | torch.List[torch.Generator] | None = None,
        num_inference_steps: int = 1000,
        output_type: str | None = "pil",
        return_dict: bool = True,
        prompt: Optional[str] = None,
    ) -> ImagePipelineOutput | torch.Tuple:
        text_inputs = self.tokenizer(
            prompt.lower(),
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        encoder_hidden_states = self.text_encoder(text_input_ids, return_dict=False)[0]

        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)
        image_out = []
        for i in range(0, 3):
            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
                image = image.to(self.device)
            else:
                image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

            # set step values
            self.scheduler.set_timesteps(num_inference_steps)
            # denoising loop
            for t in self.progress_bar(self.scheduler.timesteps):
                # 1. controlnet output
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    sample=image,
                    timestep=t,
                    encoder_hidden_states = encoder_hidden_states,
                    controlnet_cond=image_cond,
                    return_dict=False,
                )
                # 2. predict noise model_output
                model_output = self.unet(
                    sample=image,
                    timestep=t,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    encoder_hidden_states = encoder_hidden_states,
                    return_dict=False,
                )[0]

                # 3. compute previous image: x_t -> x_t-1
                image = self.scheduler.step(model_output[:,i,:,:], t, image, generator=generator).prev_sample
            # loop done!
            image = (image / 2 + 0.5)
            image = image.cpu().numpy().squeeze()
            image_out.append(image)
        image_uncertainty = np.stack((image_out[2],image_out[1],image_out[0]), axis=0)
        if not return_dict:
            return (image_uncertainty,)

        return ImagePipelineOutput(images=image_uncertainty)
    
