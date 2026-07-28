# Mask-Conditioned DDPM Generation

This module trains a **mask-conditioned Denoising Diffusion Probabilistic Model (DDPM)** to synthesize medical images for downstream segmentation experiments.

## Overview

The pipeline has three stages:

1. **Prepare paired data** — images and masks organized as ImageFolder with label subfolders
2. **Train DDPM** — learn to generate images conditioned on segmentation masks
3. **Sample & evaluate** — produce synthetic datasets and optionally compute FID

Generated data can be placed under `data/` and consumed by the segmentation trainers in the project root.

## Data Layout

Training expects ImageFolder structure:

```
path/to/images/
  0/
    img_001.png
  1/
    img_002.png

path/to/masks/
  0/
    img_001.png
  1/
    img_002.png
```

For ISIC-style data, you can prepare folders like:

```
data/isic2018_gen/
  images/0/ ...
  masks/0/ ...
```

Use `generation/resize_images.py` to normalize resolution before training.

## Training

Run from the **project root**:

```bash
python -m generation.train \
  --run-name isic18_ddpm \
  --image-path data/isic2018_gen/images \
  --mask-path data/isic2018_gen/masks \
  --num-classes 2 \
  --batch-size 4 \
  --image-size 256 \
  --gpu 0
```

Checkpoints are saved to `generation/checkpoints/<run-name>/`.
Preview grids are saved to `generation/outputs/<run-name>/`.

## Sampling

Generate synthetic image-mask pairs from trained checkpoints:

```bash
python -m generation.sample \
  --run-name isic18_ddpm \
  --mask-path data/isic2018/val/masks \
  --output data/synthetic/isic18_ddpm \
  --num-classes 2 \
  --batch-size 4 \
  --image-size 256 \
  --channels 3 \
  --gpu 0
```

Output structure:

```
data/synthetic/isic18_ddpm/
  0_0/
    0_0_img.png
    0_0_mask.png
  1_1/
    ...
```

These folders can be merged into federated or centralized segmentation datasets (e.g. `sd900combine`).

## FID Evaluation

Install optional dependency: `pip install pytorch-fid scipy==1.11.1`

```bash
# Single folder pair
python -m generation.eval_fid path/to/real path/to/generated

# Multiple subfolders with averaging
python -m generation.eval_fid path/to/real_root path/to/generated_root --recursive
```

## Utilities

```bash
# Resize an image tree
python -m generation.resize_images input_dir output_dir --size 256
```

## Model

- **UNet_mask**: 4-channel input (mask + noisy RGB), 3-channel noise prediction
- **EMA**: exponential moving average for stable sampling
- **Classifier-free guidance**: 10% label dropout during training (configurable via `--cfg-dropout`)

## Integration with Segmentation

Typical workflow:

```
Real images + masks
        ↓
  generation.train
        ↓
  generation.sample  →  synthetic image/mask pairs
        ↓
  merge into data/   →  train.py / train_federated.py
```

For federated experiments, synthetic data is commonly combined with real client data under paths such as `data/sd900_syn_all_local_relay_diff/`.

## Files

| File | Description |
|------|-------------|
| `train.py` | DDPM training |
| `sample.py` | Mask-conditioned sampling |
| `eval_fid.py` | FID evaluation |
| `resize_images.py` | Batch resize utility |
| `diffusion.py` | Noise schedule and sampler |
| `modules.py` | UNet_mask and EMA |
| `dataset.py` | Paired image/mask loaders |

Traditional Chinese documentation: [README_zh-TW.md](README_zh-TW.md)
