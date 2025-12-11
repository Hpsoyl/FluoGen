import os
import sys

os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "600"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import imageio
import skimage
import numpy as np
import random
import torch
import time

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image
from utils.process import *
from transformers import CLIPTokenizer
from tqdm import tqdm

cache_dir = "/data0/syhong/BioDiff/cache"
data_dir = "/data0/syhong/BioDiff/dataset/"
root_path = '/data0/syhong/BioDiff/dataset'

pretrained_model_name_or_path = "/data0/syhong/stable-diffusion-v1-5"

train_path = os.path.join(root_path, '1_foundation_model')


# Datasets transform functions
def transform_BioSR(examples):
    """
    dataset: BioSR
    bits: 16bit
    size: 1004*1004
    min, max value: 0, 65535
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = random_crop_and_flip(image) / 65535.0
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_BPAEC(examples):
    """
    dataset: BPAEC
    bits: 8bit
    size: 1024*1024
    min, max value: 0, 255
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = random_crop_and_flip(image) / 255.0
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_DeepBacs_denoising_MreB(examples):
    """
    dataset: DeepBacs_denoising_MreB
    bits: 16bit
    size: 512*512
    min, max value: 0, 40142
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = random_crop_and_flip(image) / 65535.0
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_DeepBacs_denoising_H_NS_mScarlet_I(examples):
    """
    dataset: DeepBacs_denoising_H_NS_mScarlet_I
    bits: 8bit
    size: 512*512
    min, max value: 0, 255
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = random_crop_and_flip(image) / 255.0
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_DeepBacs_SIM(examples):
    """
    dataset: DeepBacs_SIM
    bits: 32bit float
    size: 1024*1024
    min, max value: 0.0, 55867.24609375
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = random_crop_and_flip(image) / 65535.0
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_FMD(examples):
    """
    dataset: FMD
    bits: 8bit
    size: 512*512
    min, max value: 0, 255
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = random_crop_and_flip(image) / 255.0
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_caco2(examples):
    """
    dataset: caco2
    bits: 8bit
    size: 512*512
    min, max value: 1, 255
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = random_crop_and_flip(image) / 255.0
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_hela_seg(examples):
    """
    dataset: HeLa10Class2DImages_16bit_tiff
    bits: 16bit
    size: 512*382
    min, max value: 0, 2669
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = prctile_norm(random_crop_and_flip(image))
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_data_bowl(examples):
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)[:, :, 0]
        image = random_crop_and_flip(image) / 255.0
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_Fluo(examples):
    """
    dataset: Fluo_xxxx
    bits: 16bit
    size: various
    min, max value:
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = prctile_norm(random_crop_and_flip(image))
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_DynNucNet(examples):
    """
    dataset: DynamicNuclearNet-segmentation / tracking
    bits: 16bit
    size: various
    min, max value: 6, 65535; 0, 65535
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        if "HeLa" in image_path:
            image = prctile_norm(random_crop_and_flip(image))
        else:
            image = random_crop_and_flip(image) / 65535
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_idr035(examples):
    """
    dataset: idr035
    bits: 16bit
    size: 1280*1024
    min, max value: 
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = prctile_norm(random_crop_and_flip(image))
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}



def transform_idr036(examples):
    """
    dataset: idr036
    bits: 16bit
    size: 696*520
    min, max value: 
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = prctile_norm(random_crop_and_flip(image))
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_idr050(examples):
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = prctile_norm(random_crop_and_flip(image))
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_idr119(examples):
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = prctile_norm(random_crop_and_flip(image))
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_idr123(examples):
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = random_crop_and_flip(image) / 65535.0
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_idr133(examples):
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = random_crop_and_flip(image) / 65535.0
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_CPJUMP1(examples):
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = random_crop_and_flip(image) / 65535.0
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_BBBC_HT29(examples):
    """
    dataset: BBBC HT29
    bits: 16bit
    size: 512*512
    min, max value: <5000
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = prctile_norm(random_crop_and_flip(image))
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_BBBC_macrophages(examples):
    """
    dataset: BBBC macrophages
    bits: 8 bit RGB
    size: 1388*1040*3
    min, max value: 0-255
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = random_crop_and_flip(convert_gray(image)) / 255.0
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_BBBC_hepa(examples):
    """
    dataset: BBBC hepa
    bits: 8 bit
    size: 1392*1040
    min, max value: 0-255
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = prctile_norm(random_crop_and_flip(image))
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_idr088(examples):
    """
    dataset: idr088
    bits: 16 bit
    size: 1250*1050
    min, max value: <30000
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = random_crop_and_flip(image) / 65535.0
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_idr003(examples):
    """
    dataset: idr003
    bits: 16 bit
    size: 672*512
    min, max value: 30000 - 35000
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = prctile_norm(random_crop_and_flip(image))
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_idr094(examples):
    """
    dataset: idr094
    bits: 16 bit
    size: 1080*1080
    min, max value: <10000
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = prctile_norm(random_crop_and_flip(image))
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_idr037(examples):
    """
    dataset: idr037
    bits: 16 bit
    size: 1360*1024
    min, max value: <20000
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = prctile_norm(random_crop_and_flip(image))
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}



def transform_HPA(examples):
    """
    dataset: HPA
    bits: 8 bit RGB
    size: 2048*2048*3
    min, max value: 0-255
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = prctile_norm(resize_image(convert_gray(image)))
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_rxrx1(examples):
    """
    dataset: RxRx1
    bits: 8 bit
    size: 512*512
    min, max value: 0-255
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = random_crop_and_flip(image) / 255.0
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


def transform_rxrx19a(examples):
    """
    dataset: RxRx19a
    bits: 8 bit
    size: 1024*1024
    min, max value: 0-255
    """
    arrays = []
    input_ids = []
    for image_path in examples["file_name"]:
        image = imageio.v2.imread(image_path)
        image = random_crop_and_flip(image) / 255.0
        arrays.append(image)
    for input_id in examples["text"]:
        input_ids.append(input_id)
    return {"pixel_values": arrays, "text": input_ids}


# get data jsonl files path
data_files = get_jsonl_files_dict(train_path)

# load datasets and set their corresponding transform
train_BioSR = load_dataset("json", data_files={"train": data_files["BioSR.jsonl"]}, cache_dir=cache_dir)
train_BPAEC = load_dataset("json", data_files={"train": data_files["BPAEC.jsonl"]}, cache_dir=cache_dir)
train_DeepBacs_denoising_MreB = load_dataset("json", data_files={"train": data_files["DeepBacs_denoising_MreB.jsonl"]}, cache_dir=cache_dir)
train_DeepBacs_denoising_H_NS_mScarlet_I = load_dataset("json", data_files={"train": data_files["DeepBacs_denoising_H-NS-mScarlet-I.jsonl"]}, cache_dir=cache_dir)
train_DeepBacs_SIM = load_dataset("json", data_files={"train": data_files["DeepBacs_SIM.jsonl"]}, cache_dir=cache_dir)
train_FMD = load_dataset("json", data_files={"train": data_files["FMD.jsonl"]}, cache_dir=cache_dir)
train_caco2 = load_dataset("json", data_files={"train": data_files["caco2.jsonl"]}, cache_dir=cache_dir)
train_hela_seg = load_dataset("json", data_files={"train": data_files["HeLa10Class.jsonl"]}, cache_dir=cache_dir)
train_data_bowl = load_dataset("json", data_files={"train": data_files["2018_Data_Science_Bowl.jsonl"]}, cache_dir=cache_dir)
train_Fluo = load_dataset("json", data_files={"train": data_files["Fluo.jsonl"]}, cache_dir=cache_dir)
train_DynNucNet_seg = load_dataset("json", data_files={"train": data_files["DynNucNet-seg.jsonl"]}, cache_dir=cache_dir)
train_DynNucNet_tra = load_dataset("json", data_files={"train": data_files["DynNucNet-tra.jsonl"]}, cache_dir=cache_dir)
train_idr035 = load_dataset("json", data_files={"train": data_files["idr035.jsonl"]}, cache_dir=cache_dir)
train_idr036 = load_dataset("json", data_files={"train": data_files["idr036.jsonl"]}, cache_dir=cache_dir)
train_idr050 = load_dataset("json", data_files={"train": data_files["idr050.jsonl"]}, cache_dir=cache_dir)
train_idr119 = load_dataset("json", data_files={"train": data_files["idr119.jsonl"]}, cache_dir=cache_dir)
train_idr123 = load_dataset("json", data_files={"train": data_files["idr123.jsonl"]}, cache_dir=cache_dir)
train_idr133 = load_dataset("json", data_files={"train": data_files["idr133.jsonl"]}, cache_dir=cache_dir)
train_CPJUMP1 = load_dataset("json", data_files={"train": data_files["CPJUMP1.jsonl"]}, cache_dir=cache_dir)

train_BBBC_HT29 = load_dataset("json", data_files={"train": data_files["BBBC_ht29.jsonl"]}, cache_dir=cache_dir)
train_BBBC_macrophages = load_dataset("json", data_files={"train": data_files["BBBC_macrophages.jsonl"]}, cache_dir=cache_dir)
train_BBBC_hepa = load_dataset("json", data_files={"train": data_files["BBBC_hepa_fibro_dna.jsonl"]}, cache_dir=cache_dir)
train_idr088 = load_dataset("json", data_files={"train": data_files["idr088.jsonl"]}, cache_dir=cache_dir)
train_idr003 = load_dataset("json", data_files={"train": data_files["idr003.jsonl"]}, cache_dir=cache_dir)
train_idr094 = load_dataset("json", data_files={"train": data_files["idr094.jsonl"]}, cache_dir=cache_dir)
train_idr037 = load_dataset("json", data_files={"train": data_files["idr037.jsonl"]}, cache_dir=cache_dir)
train_HPA = load_dataset("json", data_files={"train": data_files["HPA.jsonl"]}, cache_dir=cache_dir)
train_rxrx1 = load_dataset("json", data_files={"train": data_files["rxrx1.jsonl"]}, cache_dir=cache_dir)
train_rxrx19a = load_dataset("json", data_files={"train": data_files["rxrx19a.jsonl"]}, cache_dir=cache_dir)


train_BioSR.set_transform(transform_BioSR)
train_BPAEC.set_transform(transform_BPAEC)
train_DeepBacs_denoising_MreB.set_transform(transform_DeepBacs_denoising_MreB)
train_DeepBacs_denoising_H_NS_mScarlet_I.set_transform(transform_DeepBacs_denoising_H_NS_mScarlet_I)
train_DeepBacs_SIM.set_transform(transform_DeepBacs_SIM)
train_FMD.set_transform(transform_FMD)
train_caco2.set_transform(transform_caco2)
train_hela_seg.set_transform(transform_hela_seg)
train_data_bowl.set_transform(transform_data_bowl)
train_Fluo.set_transform(transform_Fluo)
train_DynNucNet_seg.set_transform(transform_DynNucNet)
train_DynNucNet_tra.set_transform(transform_DynNucNet)
train_idr035.set_transform(transform_idr035)
train_idr036.set_transform(transform_idr036)
train_idr050.set_transform(transform_idr050)
train_idr119.set_transform(transform_idr119)
train_idr123.set_transform(transform_idr123)
train_idr133.set_transform(transform_idr133)
train_CPJUMP1.set_transform(transform_CPJUMP1)

train_BBBC_HT29.set_transform(transform_BBBC_HT29)
train_BBBC_macrophages.set_transform(transform_BBBC_macrophages)
train_BBBC_hepa.set_transform(transform_BBBC_hepa)
train_idr088.set_transform(transform_idr088)
train_idr003.set_transform(transform_idr003)
train_idr094.set_transform(transform_idr094)
train_idr037.set_transform(transform_idr037)
train_HPA.set_transform(transform_HPA)
train_rxrx1.set_transform(transform_rxrx1)
train_rxrx19a.set_transform(transform_rxrx19a)


datasets = [
    train_BioSR["train"],
    train_BPAEC["train"],
    train_DeepBacs_denoising_MreB["train"],
    train_DeepBacs_denoising_H_NS_mScarlet_I["train"],
    train_DeepBacs_SIM["train"],
    train_FMD["train"],
    train_caco2["train"],
    train_hela_seg["train"],
    train_data_bowl["train"],
    train_Fluo["train"],
    train_DynNucNet_seg["train"],
    train_DynNucNet_tra["train"],
    train_idr035["train"],
    train_idr036["train"],
    train_idr050["train"],
    train_idr119["train"],
    train_idr123["train"],
    train_idr133["train"],
    train_CPJUMP1["train"],
    train_BBBC_HT29["train"],
    train_BBBC_macrophages["train"],
    train_BBBC_hepa["train"],
    train_idr088["train"],
    train_idr003["train"],
    train_idr094["train"],
    train_idr037["train"],
    train_HPA["train"],
    train_rxrx1["train"],
    train_rxrx19a["train"],
]

datasets_choice_weights = [
    PROBABILITY_BIOSR,
    PROBABILITY_BPAEC,
    PROBABILITY_DEEPBACS_MREB,
    PROBABILITY_DEEPBACS_H_NS_MSCARLET,
    PROBABILITY_DEEPBACS_SIM,
    PROBABILITY_FMD,
    PROBABILITY_CACO2,
    PROBABILITY_HELA_SEG,
    PROBABILITY_DARA_BOWL,
    PROBABILITY_FLUO,
    PROBABILITY_DYNNUCNET_SEG,
    PROBABILITY_DYNNUCNET_TRA,
    PROBABILITY_IDR035,
    PROBABILITY_IDR036,
    PROBABILITY_IDR050,
    PROBABILITY_IDR119,
    PROBABILITY_IDR123,
    PROBABILITY_IDR133,
    PROBABILITY_CPJUMP1,
    PROBABILITY_BBBC_HT29,
    PROBABILITY_BBBC_MARCOPHAGES,
    PROBABILITY_BBBC_HEPA,
    PROBABILITY_IDR088,
    PROBABILITY_IDR003,
    PROBABILITY_IDR094,
    PROBABILITY_IDR037,
    PROBABILITY_HPA,
    PROBABILITY_RXRX1,
    PROBABILITY_RXRX19A,
]


# create random choice dataset
class RandomDataset(Dataset):
    def __init__(
        self,
        datasets,
        tokenizer,
        weights=None
    ):
        self.datasets = datasets
        self.weights = weights if weights is not None else [1] * len(datasets)
        self.tokenizer = tokenizer
        self.cnt = 0

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        dataset_choice = None
        self.cnt+=1
        while dataset_choice is None or is_low_contrast(image, fraction_threshold=0):
            dataset_choice = random.choices(self.datasets, weights=self.weights, k=1)[0]
            dataset_len = len(dataset_choice)
            idx = idx % dataset_len

            success = False
            while not success:
                idx = idx % dataset_len  # 确保 idx 在合法范围内
                try:
                    batch = dataset_choice[idx]
                    image = batch["pixel_values"]
                    success = True
                except Exception:
                    idx = random.randint(0, dataset_len - 1)
                

            if self.tokenizer:
                # random choice to use the true prompt or empty prompt
                use_emptyt_prompt = np.random.binomial(1, 0.2)
                if use_emptyt_prompt:
                    batch["text"] = ""
                    input_ids = tokenize_captions(batch, self.tokenizer).squeeze()
                else:
                    batch["text"] = batch["text"].lower()
                    input_ids = tokenize_captions(batch, self.tokenizer).squeeze()

        image = convert_to_diffusion_tensor(batch["pixel_values"])
        if self.tokenizer:
            return {"pixel_values": image, "input_ids": input_ids}
        else:
            return {"pixel_values": image}


def creat_bio_dataset_and_dataloader(args, tokenizer=None):
    batch_size = args.train_batch_size

    bio_image_dataset = RandomDataset(datasets, weights=datasets_choice_weights, tokenizer=tokenizer)
    bio_image_dataloader = DataLoader(bio_image_dataset, 
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=8,
                           )

    return bio_image_dataset, bio_image_dataloader


if __name__ == "__main__":
    batch_size = 1
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer"
    )

    bio_image_dataset = RandomDataset(datasets, tokenizer=tokenizer, weights=datasets_choice_weights)
    bio_image_dataloader = DataLoader(bio_image_dataset, 
                                      batch_size=batch_size,
                                      num_workers=32,
                                      shuffle=False,
    )

    for step, batch in enumerate(tqdm(bio_image_dataloader, desc="Processing batches")):
        print(step)
        
    print("done")

