# FluoGen: An Open-Source Generative Foundation Model for Fluorescence Microscopy Image Enhancement and Analysis

Codes, representative pre-trained models, test data for FluoGen
<div align="center">

✨ [**Method overview**](#-Method-overview) **|** 🚩 [**Paper**](#-Paper) **|** 🔧 [**Install**](#Install)  **|** 🏰 [**Model Download**](#-Model-Download) **|** ⚡ [**Inference**](#-Inference) **|** 💻 [**Training**](#-Training) **|** 🎨 [**Dataset**](#-Dataset)  **|** &#x1F308; [**Results**](#-Results)

</div>

## ✨ Method overview

<p align="center">
<img src="assert\Fig1.png" width='600'>
</p>
First, we curated a large-scale, heterogeneous dataset comprising 3.5 million high-quality fluorescence microscopy images, paired with textual annotations covering over 3,500 cell types and subcellular components. Second, we developed FluoGen, a diffusion-based generative foundation model that operates directly in pixel space to preserve high-frequency biological details without latent compression. Crucially, we reformulated the learning objective from standard noise prediction to velocity prediction, a strategic design that effectively eliminates the brightness bias inherent in conventional diffusion models and enables the learning of robust, generalizable biological priors.
<p align="center">
<img src="assert\Fig2.png" width='600'>
</p>
Third, we designed a trainable conditional control branch to adapt the frozen FluoGen backbone for diverse downstream tasks, ranging from image enhancement (denoising, super-resolution) to high-level analysis (segmentation, classification). To ensure reliability in scientific imaging, we integrated a distribution-free conformalized quantile regression framework, enabling the model to output calibrated pixel-wise uncertainty maps that rigorously quantify potential hallucinations or errors. We demonstrate that FluoGen serves as both a superior backbone and a data-efficient sample generator, allowing state-of-the-art models to achieve high performance using as little as 30-50 training samples (or roughly 2% of standard datasets).

## 🚩 Paper
This repository is for FluoGen introduced in the following paper:

[Huaian Chen, Shiyao Hong, Yuxuan Gu, et al. "FluoGen: An Open-Source Generative Foundation Model for Fluorescence Microscopy Image Enhancement and Analysis" ***bioRxiv 2025.xx.xx.xxxxxx*** (2025)](网址) 

## 🔧 Install
### Our environment
  - Ubuntu 20.04.4
  - CUDA 11.4
  - Python 3.11.11
  - Pytorch 2.3.0
  - NVIDIA GPU (GeForce RTX 3090) 

### Quick installation
We recommend using **Miniconda** or **Anaconda** to manage the environment.

**Step 0.**
Install Anaconda from the [official website](https://www.anaconda.com/download/) if you haven't already.

**Step 1.**
Clone this repository using the following command.
   ```bash
    git clone https://github.com/Hpsoyl/FluoGen
    cd FluoGen
  ```

**Step 2.**
Create a virtual environment with Python 3.11.
   ```bash
    conda create -n FluoGen python=3.11 -y
    conda activate FluoGen
  ```

**Step 3.**
Install PyTorch with CUDA support.
   ```bash
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
  ```
**💡 Note:** We use PyTorch 2.3.0. For RTX 3090, we recommend the CUDA 11.8 version for best compatibility.

**Step 4.**
Install the remaining dependencies.
   ```bash
    pip install -r requirements.txt
  ```
**💡 Note:** Although your system driver might be CUDA 11.4 (as in our environment), the installation command in Step 3 installs a local CUDA 11.8 toolkit strictly for PyTorch, which is compatible with the RTX 3090.

## 🏰 Model Download
| Model  |Download                |
|:--------- | :------------------------------------------- |
| Foundation Model  |  [Coming soon]                                              |
| FMD     |    [Coming soon]  
| BioSR    |    [Coming soon]  

Download the pre-trained models and place them into `./model_output`.


## ⚡ Inference

### 🌐 Online Web Demo
For a quick start without local installation, you can try our interactive web demo on Hugging Face Spaces. It supports Text-to-Image generation and basic downstream tasks directly in your browser. Try it here: https://huggingface.co/spaces/FluoGen-Group/FluoGen

<p align="center">
<img src="assert\Fig8.png" width='800'>
</p>

### 🖥️ Local Inference
#### 1. Prepare models
Ensure you have downloaded our pre-trained checkpoints (Foundation Model & ControlNet weights) and placed them in the ./model_output/ folder (or specify your own path).

#### 2. Text-to-Image Generation
To generate synthetic fluorescence microscopy images using the pre-trained Foundation Model:
```bash
python src/test_DDPM.py
```

#### 3. Downstream Tasks (Control Branch)
For tasks like Super-Resolution (SR), Denoising, or Segmentation, run the corresponding task-specific script. For example:

* Super-Resolution (BioSR):
```bash
python src/test_controlnet_for_BioSR.py
```
* Denoising (FMD):
```bash
python src/test_controlnet_for_FMD.py
```
* Segmentation (DSB):
```bash
python src/test_controlnet_for_dsb.py
```


## 💻 Training 
FluoGen training consists of three stages: Foundation Model Pre-training, Control Branch Fine-tuning, and Uncertainty Calibration.

### 1. Foundation Model Pre-training
Train the diffusion backbone on large-scale datasets (defined in `dataset/1_foundation_model/*.jsonl`). 
* **Configuration:** Adjust parameters in src/train_foundation_model.sh (e.g., batch size, learning rate and model paths)

**Start Training:**
```bash
bash train_foundation_model.sh
```

### 2. Control Branch Fine-tuning
Adapt the frozen backbone to specific downstream tasks (e.g., Denoising, SR) by training the ControlNet branch.
* **Prepare Data:** Ensure the task-specific .jsonl files (e.g., for BioSR or FMD) are ready.

**Start Training:** 
```bash
bash train_control_branch.sh
```

### 3. Uncertainty Evaluatio
To enable reliable confidence estimation (Risk-Controlling Prediction Sets, RCPS), we train and calibrate the quantile estimators.
* **Configuration:** Edit src/models/utils/config.yml to set quantile levels (e.g., alpha for bounds).
  
**Step A: Train Quantile Estimators**
Train the model to predict upper/lower bounds.
```bash
bash src/train_uncertainty.sh
```
**Step B: Calibration**
Minimize the Expected Calibration Error (ECE) and calibrate the intervals using the validation set.
```bash
python src/calibrate_model.py
```
*This step generates the final calibrated confidence maps for your predictions.*

## 🎨 Dataset
We curated a massive, heterogeneous fluorescence microscopy dataset comprising approximately 3.5 million high-quality images (totaling ~5.6 TB). This dataset aggregates data from major public repositories, including **the Broad Bioimage Benchmark Collection (BBBC), Image Data Resource (IDR), RxRx, and the Human Protein Atlas (HPA)**. It spans over 100 distinct cell lines and covers 3,566 categories of cell types and subcellular components, providing a robust foundation for learning generalizable biological representations.

To ensure high fidelity, the collected raw data underwent rigorous manual curation and preprocessing. Low-quality samples (e.g., low-SNR, blurry, or low-contrast images) were strictly excluded. Retained images were standardized to a resolution of 512×512 and annotated with structured textual prompts (e.g., *"[Organism] of [Cell Line]"*) to support text-conditioned generation. Detailed summaries, source references, and download links for the subsets used are extensively listed in **Supplementary Tables 8, 9, and 10** of our paper.

We provide a comprehensive data preprocessing pipeline to facilitate custom training. Specifically, users can utilize the scripts located in `dataset/scripts/` to automatically scan raw image directories and generate the required `.jsonl` annotation files.

**💡 Note:** Training the Foundation Model from scratch is computationally intensive due to the massive scale of the dataset (3.5M images). We highly recommend using our pre-trained weights as a backbone. Users can then efficiently fine-tune only the Control Branch using a small amount of their own data (e.g., 30-50 images) to achieve state-of-the-art performance on specific downstream tasks.

### 📄 Custom Dataset JSONL Format
To fine-tune the Control Branch on your own data, you need to prepare a .jsonl file. Each line in the file represents a single training sample and must strictly follow this JSON format:
```josn
{"image": "/abs/path/to/GT/img.tif", "conditioning_image": "/abs/path/to/input/img.tif", "text": "F-actin of COS-7"}
```
**Key Definitions:**
* `image`: The path to the Ground Truth (Target) image.
  * For Enhancement: The high-quality reference (e.g., High-Resolution or Clean image).
  * For Segmentation: The original microscopy image.
* `conditioning_image`: The path to the Control Input image.
  * For Enhancement: The low-quality input (e.g., Low-Resolution or Noisy image).
  * For Segmentation/Generation: The segmentation mask or label map.
* `text`: The textual prompt describing the biological content (e.g., "Mitochondria of HeLa").

  **💡 Note:** We recommend using absolute paths to avoid FileNotFoundError. You can use our provided scripts in dataset/scripts/ to automatically generate these files from your folders.

## &#x1F308; Results

 ### 1. Synthesis results of FluoGen.

<p align="center">
<img src="assert\Fig3.png" width='600'>
</p>

### 2. Comparison with other SOTA model in super-resolution and denosing.

<p align="center">
<img src="assert\Fig4.png" width='600'>
</p>

<p align="center">
<img src="assert\Fig5.png" width='600'>
</p>

### 3. Serving as a sample generator: reducing the data requirements of downstream analysis tasks.

<p align="center">
<img src="assert\Fig6.png" width='600'>
</p>

### 4. Serving as a performance enhancer: breaking through performance ceilings for downstream models.

<p align="center">
<img src="assert\Fig7.png" width='600'>
</p>
