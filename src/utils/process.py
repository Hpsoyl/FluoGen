import os
import mrcfile
import random
import torch
import numpy as np
from skimage.exposure import is_low_contrast
from PIL import Image
import cv2
from torchvision import transforms
import pdb
from skimage import io


PROBABILITY_BIOSR = 1.0
PROBABILITY_BPAEC = 1.0
PROBABILITY_DEEPBACS_MREB = 1.0
PROBABILITY_DEEPBACS_H_NS_MSCARLET = 1.0
PROBABILITY_DEEPBACS_SIM = 1.0
PROBABILITY_FMD = 1.0
PROBABILITY_CACO2 = 1.0
PROBABILITY_HELA_SEG= 1.0
PROBABILITY_DARA_BOWL = 1.0
PROBABILITY_FLUO = 1.0
PROBABILITY_DYNNUCNET_SEG = 1.0
PROBABILITY_DYNNUCNET_TRA = 1.0
PROBABILITY_IDR035 = 1.0
PROBABILITY_IDR036 = 1.0
PROBABILITY_IDR050 = 1.0
PROBABILITY_IDR119 = 1.0
PROBABILITY_IDR123 = 0.0
PROBABILITY_IDR133 = 1.0
PROBABILITY_CPJUMP1 = 1.0

PROBABILITY_BBBC_HT29 = 1.0
PROBABILITY_BBBC_MARCOPHAGES = 1.0
PROBABILITY_BBBC_HEPA = 1.0
PROBABILITY_IDR088 = 1.0
PROBABILITY_IDR003 = 1.0
PROBABILITY_IDR094 = 1.0
PROBABILITY_IDR037 = 1.0
PROBABILITY_HPA = 1.0
PROBABILITY_RXRX1 = 1.0
PROBABILITY_RXRX19A = 1.0


class RandomStateManager:
    def __init__(self):
        self.shared_seed = None
    
    def set_seed(self, seed=None):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        self.shared_seed = seed
    
    def get_state(self):
        if self.shared_seed is None:
            self.set_seed()
        random.seed(self.shared_seed)
        return random.getstate()

random_state_manager = RandomStateManager()


def convert_gray(image: np.ndarray):
    gray = Image.fromarray(image).convert("L")
    return np.array(gray)


def get_jsonl_files_dict(target_path):
    return {
        jsonl_file: os.path.join(target_path, jsonl_file)
        for jsonl_file in os.listdir(target_path)
        if jsonl_file.endswith('.jsonl') and os.path.isfile(os.path.join(target_path, jsonl_file))
    }


# random crop and flip the image
def random_flip(image: np.ndarray, flip_lr=False, flip_ud=False):
    if flip_lr:
        image = np.fliplr(image)
    if flip_ud:
        image = np.flipud(image)
    return image


def random_crop(image: np.ndarray, size=512, crop_pos=None):
    h, w = image.shape
    if crop_pos is None:
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
    else:
        top, left = crop_pos

    cropped_image = image[top:top + size, left:left + size]
    return cropped_image, (top, left)


def resize_image(image: np.ndarray, size=512):
    resized_image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
    return resized_image


def random_crop_and_flip(image, size=512, rng_state=None) -> np.ndarray:
    if rng_state is not None:
        random.setstate(rng_state)
    
    do_flip_lr = random.random() < 0.5
    do_flip_ud = random.random() < 0.5
    
    h, w = image.shape
    shorter_size = min(h, w)

    if h==w:
        if shorter_size>size:
            if random.randint(0, 1) == 0:
                # crop -> flip
                crop_pos = (random.randint(0, h - size), random.randint(0, w - size))
                image = random_crop(image, size, crop_pos)[0]
                image = random_flip(image, do_flip_lr, do_flip_ud)
            else:
                # resieze -> flip
                image = resize_image(image)
                image = random_flip(image, do_flip_lr, do_flip_ud)
            return image.copy()
        elif shorter_size==size:
            # flip
            return random_flip(image, do_flip_lr, do_flip_ud)
        elif shorter_size<size:
            # resize -> flip
            image = resize_image(image)
            image = random_flip(image, do_flip_lr, do_flip_ud)
            return image.copy()
        
    elif h!=w:
        if shorter_size>size or shorter_size==size:
            if random.randint(0, 1) == 0:
                # crop to shorter size -> resize_image -> flip
                crop_pos = (random.randint(0, h - shorter_size), random.randint(0, w - shorter_size))
                image = random_crop(image, shorter_size, crop_pos)[0]
                image = resize_image(image)
                image = random_flip(image, do_flip_lr, do_flip_ud)
            else:
                # crop to size -> flip
                crop_pos = (random.randint(0, h - size), random.randint(0, w - size))
                image = random_crop(image, size, crop_pos)[0]
                image = random_flip(image, do_flip_lr, do_flip_ud)
            return image.copy()
        elif shorter_size<size:
            crop_pos = (random.randint(0, h - shorter_size), random.randint(0, w - shorter_size))
            image = random_crop(image, shorter_size, crop_pos)[0]
            image = resize_image(image)
            image = random_flip(image, do_flip_lr, do_flip_ud)
            return image.copy()


# convert np ndarray to tensor and normlize it to the range diffusion models need
def convert_to_diffusion_tensor(example):
    tensors = (torch.from_numpy(example).unsqueeze(0)*2.0-1.0).to(torch.float32)
    tensors = torch.clip(tensors, -1.0, 1.0)
    return tensors


def prctile_norm(
        x: np.ndarray,
        min_prc=0,
        max_prc=100
    ):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


def is_low_contrast(
        array: np.ndarray,
        fraction_threshold=0.02,
        lower_percentile=0,
        upper_percentile=99,
    ):
    array = array.squeeze()

    lower_limit = np.percentile(array, lower_percentile)
    upper_limit = np.percentile(array, upper_percentile)
    dlimits = [array.min(), array.max()]
    
    if dlimits[1] == dlimits[0]:
        return True

    ratio = (upper_limit - lower_limit) / (dlimits[1] - dlimits[0])

    return ratio < fraction_threshold

    
def creat_one_hot(
        num_datasets,
        dataset_idx,
        height=512,
        width=512
    ):
    one_hot = torch.full((num_datasets, height, width), -1, dtype=torch.float32)
    one_hot[dataset_idx] = 1.0

    return one_hot


def tokenize_captions(example, tokenizer, is_train=True):
    captions = []
    for caption in [example["text"]]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return inputs.input_ids


def tansform_condition(example, weight_dtype, device, args, seed=None):
    random_state_manager.set_seed(seed)
    rng_state = random_state_manager.get_state()

    if args.dataset_name == "BioSR":
        image = example.astype('float32') / 65535
        tensor = torch.from_numpy(image).to(dtype=weight_dtype, device=device)
        resize = transforms.Resize((512, 512))
        tensor = resize(tensor)
        
    elif args.dataset_name == "FMD":
        image = example.astype('float32') / 255
        tensor = torch.from_numpy(image).unsqueeze(0).to(dtype=weight_dtype, device=device)

    elif args.dataset_name == "DynamicNet":
        image = example.astype('float32')
        tensor = torch.from_numpy(image).unsqueeze(0).to(dtype=weight_dtype, device=device)

    elif args.dataset_name == "2018_Data_Science_Bowl":
        image = random_crop_and_flip(example, rng_state=rng_state).astype('float32')
        tensor = torch.from_numpy(image).unsqueeze(0).to(dtype=weight_dtype, device=device)

    return tensor


def transform_gt(example, weight_dtype, device, args, seed=None):
    random_state_manager.set_seed(seed)
    rng_state = random_state_manager.get_state()

    if args.dataset_name == "BioSR":
        image = example.astype('float32') / 65535
        tensor = (torch.from_numpy(image).unsqueeze(0)*2.0-1.0).to(dtype=weight_dtype, device=device)
        tensor = torch.clip(tensor, -1.0, 1.0)

    elif args.dataset_name == "FMD":
        image = example.astype('float32') / 255
        tensor = (torch.from_numpy(image).unsqueeze(0)*2.0-1.0).to(dtype=weight_dtype, device=device)
        tensor = torch.clip(tensor, -1.0, 1.0)

    elif args.dataset_name == "DynamicNet":
        image = example.astype('float32')
        tensor = (torch.from_numpy(image).unsqueeze(0)*2.0-1.0).to(dtype=weight_dtype, device=device)

    elif args.dataset_name == "2018_Data_Science_Bowl":
        image = random_crop_and_flip(example[:,:,0], rng_state=rng_state).astype('float32') / 255
        tensor = (torch.from_numpy(image).unsqueeze(0)*2.0-1.0).to(dtype=weight_dtype, device=device)

        assert image.shape[0] == image.shape[1] == 512

    return tensor

