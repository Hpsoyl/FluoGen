from contextlib import nullcontext
import pdb
import argparse
import inspect
import logging
import math
import numpy as np
import os
import shutil
import sys

os.environ['HF_DATASETS_OFFLINE'] = '1'

from tqdm import tqdm
from datetime import timedelta, datetime
from pathlib import Path

import accelerate
import datasets
import skimage
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration
from huggingface_hub import create_repo, upload_folder
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available

from utils.data_loader import creat_bio_dataset_and_dataloader

from models.pipeline_ddpm_text_encoder import DDPMPipeline
from models.unet_2d import UNet2DModel
from transformers import PreTrainedModel


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.31.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def log_validation(
    text_encoder,
    tokenizer,
    unet,
    args,
    accelerator,
    weight_dtype,
    global_step,
    noise_scheduler,
    is_final_validation=False,
):
    logger.info("Running validation... ")
    unet = accelerator.unwrap_model(unet)

    pipeline = DDPMPipeline(
        unet=unet,
        scheduler=noise_scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer
    )
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)
    images = []
    for i in range(len(args.validation_prompts)):
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        with autocast_ctx:
            image = pipeline(
                args.validation_prompts[i],
                generator=generator,
                num_inference_steps=args.ddpm_num_inference_steps,
                output_type="np",
            ).images[0]
        images.append(image)
    
    # denormalize the images and save to tensorboard
    images_processed = [(img * 255).round().astype("uint8") for img in images]

    if args.logger == "tensorboard":
        if is_accelerate_version(">=", "0.17.0.dev0"):
            tracker = accelerator.get_tracker("tensorboard", unwrap=True)
        else:
            tracker = accelerator.get_tracker("tensorboard")
        np_images = np.stack([np.asarray(img) for img in images_processed])
        tracker.add_images("test_samples", np_images.transpose(0, 3, 1, 2), global_step)
    elif args.logger == "wandb":
        # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
        accelerator.get_tracker("wandb").log(
            {"test_samples": [wandb.Image(img) for img in images_processed], "global_step": global_step},
            step=global_step,
        )

    del pipeline
    torch.cuda.empty_cache()

    return images


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
                "Run validation every X steps. Validation consists of running the prompt"
                " `args.validation_prompt` multiple times: `args.num_validation_images`"
                " and logging the images."
            ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample", "v_prediction"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=20)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    parser.add_argument(
        "--rescale_betas_zero_snr", action="store_true", help="Whether or not to rescale betas to zero snr."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
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
    parser.add_argument("--ddpm_timestep_spacing", type=str, default="trailing")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=120))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    def register_nan_hooks(model):
        for name, module in model.named_modules():
            def hook_fn(module, input, output, name=name):
                if isinstance(output, torch.Tensor):
                    if torch.isnan(output).any():
                        print(f"[Rank {accelerator.process_index}] NaN detected at layer: {name}")
                elif isinstance(output, (tuple, list)):
                    for i, out in enumerate(output):
                        if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                            print(f"[Rank {accelerator.process_index}] NaN detected at layer: {name}[{i}]")
            module.register_forward_hook(hook_fn)

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        # def save_model_hook(models, weights, output_dir):
        #     if accelerator.is_main_process:
        #         if args.use_ema:
        #             ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

        #         for i, model in enumerate(models):
        #             model_to_save = model
        #             weight = weights.pop()

        #             if isinstance(model_to_save, UNet2DModel):
        #                 save_path = os.path.join(output_dir, "unet")
        #             elif isinstance(model_to_save, CLIPTextModel):
        #                 save_path = os.path.join(output_dir, "text_encoder")
        #             else:
        #                 print(f"[WARNING] Unknown model type: {type(model_to_save)}")
        #                 continue

        #             model_to_save.save_pretrained(save_path)
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                if isinstance(model, UNet2DModel):
                    # load diffusers style into model
                    load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif isinstance(model, PreTrainedModel):
                    print(f"[INFO] Skipping register_to_config for transformer model: {type(model)}")
                    

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        training_settings = [
            "/data0/syhong/BioDiff/src/train_DDPM_text_encoder.py",
            "/data0/syhong/BioDiff/src/utils/data_loader_text_encoder.py",
            "/data0/syhong/BioDiff/src/train_foundation_model.sh",
        ]
        training_settings_save_path = os.path.join(args.output_dir, "training_settings")
        os.makedirs(training_settings_save_path, exist_ok=True)
        for training_setting in training_settings:
            shutil.copy(src=training_setting, dst=training_settings_save_path)

    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
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
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )

    encoder_size = tokenizer.model_max_length * text_encoder.config.hidden_size

    # Initialize the model
    if args.model_config_name_or_path is None:
        unet = UNet2DModel(
            sample_size=args.resolution,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            use_prompt=True,
            encoder_size=encoder_size
        )
    else:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        unet = UNet2DModel.from_config(config)

    # Freeze text_encoder and set unet to trainable
    text_encoder.requires_grad_(False)
    unet.train()

    # Create EMA for the model.
    if args.use_ema:
        ema_unet = EMAModel(
            unet.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=unet.config,
        )
    
    weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    #     args.mixed_precision = accelerator.mixed_precision
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16
    #     args.mixed_precision = accelerator.mixed_precision

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
            rescale_betas_zero_snr=args.rescale_betas_zero_snr,
            timestep_spacing=args.ddpm_timestep_spacing,
        )
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps, 
            beta_schedule=args.ddpm_beta_schedule,
            rescale_betas_zero_snr=args.rescale_betas_zero_snr
        )


    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # DataLoaders creation:
    dataset, train_dataloader = creat_bio_dataset_and_dataloader(args, tokenizer)
    
    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
    num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
    num_training_steps_for_scheduler = (
        args.num_epochs * num_update_steps_per_epoch * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    if num_training_steps_for_scheduler != max_train_steps * accelerator.num_processes:
        logger.warning(
            f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
            f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
            f"This inconsistency may result in the learning rate scheduler not functioning properly."
        )
    # Afterwards we recalculate our number of training epochs
    args.num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    # Train!    
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(args.resume_from_checkpoint)
            
            # initial_global_step = 0

            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # reassign learning rate to raw learning rate
    # for param_group in optimizer.param_groups:
    #     param_group['initial_lr'] = args.learning_rate
    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    #     num_training_steps=num_training_steps_for_scheduler,
    # )
    # lr_scheduler = accelerator.prepare(lr_scheduler)

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # register_nan_hooks(unet)
    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                clean_images = batch["pixel_values"].to(weight_dtype)
                noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device)
                bsz = clean_images.shape[0]
                
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
                ).long()

                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                model_output = unet(noisy_images, timesteps, encoder_hidden_states).sample
                model_output = torch.nan_to_num(model_output, nan=1.0)
                
                if args.prediction_type == "epsilon":
                    loss = F.mse_loss(model_output.float(), noise.float())
                elif args.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    # use SNR weighting from distillation paper
                    loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                    loss = loss.mean()
                elif args.prediction_type == "v_prediction":
                    velocity = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                    loss = F.mse_loss(model_output.float(), velocity.float())
                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")
                
                if torch.isnan(loss):
                    rank = accelerator.process_index
                    print(f"[Rank {rank}] NaN detected!")
                    print(f"[Rank {rank}] file_path: {batch['file_name']}")
                    print(f"[Rank {rank}] T: {timesteps}")
                    print(f"[Rank {rank}] model output is nan: {torch.isnan(model_output).any()}")
                    print(f"[Rank {rank}] model input is nan: {torch.isnan(noisy_images).any()}")
                    accelerator.wait_for_everyone()
                    raise RuntimeError("NaN detected, aborting training cleanly")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                # pdb.set_trace()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        log_validation(
                            text_encoder,
                            tokenizer,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            noise_scheduler,
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.use_ema:
                logs["ema_decay"] = ema_unet.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    # Generate sample images for visual inspection
    if accelerator.is_main_process:
        # save the model
        unet = accelerator.unwrap_model(unet)

        if args.use_ema:
            ema_unet.store(unet.parameters())
            ema_unet.copy_to(unet.parameters())

        pipeline = DDPMPipeline(
            unet=unet,
            scheduler=noise_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer
        )
        pipeline.save_pretrained(args.output_dir)

        if args.use_ema:
            ema_unet.restore(unet.parameters())

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
