# Lapin

[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-222?style=flat&logo=github)](https://kanhaojun.github.io/lapin/)

Lapin is a research framework for **industrial image segmentation** with both **centralized** and **federated** training pipelines. It also includes a **mask-conditioned DDPM** module for synthesizing training data before segmentation. The unified codebase supports training, validation, and comparison of multiple segmentation architectures on industrial surface-defect datasets such as SD900 (Saliency900).

## Highlights

- Unified training and evaluation for 13 segmentation architectures
- Federated learning with FedAvg, FedProx, and SCAFFOLD
- Mask-conditioned DDPM for synthetic data augmentation
- Standard segmentation metrics: IoU, Dice, accuracy, sensitivity, specificity
- TensorBoard logging and optional Weights & Biases integration
- Modular configuration for models, datasets, and optimizers

## Supported Models

| Model | Key |
|-------|-----|
| UNet | `unet` |
| VMUNet | `vmunet` |
| VMUNet V2 | `vmunet-v2` |
| HVMUNet | `hvmunet` |
| U²-Net | `u2net` |
| UNet++ | `unetpp` |
| UNet+++ | `unetppp` |
| TransUNet | `tunet` |
| ResUNet | `resunet` |
| ResUNet++ | `resunetpp` |
| Attention UNet | `attu` |
| R2U-Net | `r2u` |
| R2AttU-Net | `attr2u` |

## Project Structure

```
lapin/
├── train.py                 # Centralized segmentation training
├── train_federated.py       # Federated segmentation training
├── generation/              # Mask-conditioned DDPM data synthesis
│   ├── train.py             # Train diffusion model
│   ├── sample.py            # Generate synthetic pairs
│   └── eval_fid.py          # FID evaluation
├── engine.py                # Training / validation loops
├── utils.py                 # Optimizers, losses, transforms
├── configs/config.py        # Configuration builder
├── models/                  # Segmentation model implementations
├── datasets/dataset.py      # Segmentation dataset loader
├── federated/               # Federated data split & SCAFFOLD helpers
├── data/                    # Dataset directory (user-provided)
├── pretrained/              # External checkpoints (user-provided)
└── results/                 # Experiment outputs
```

## Installation

```bash
git clone https://github.com/your-org/lapin.git
cd lapin

conda create -n lapin python=3.8
conda activate lapin
pip install -r requirements.txt
```

For VMUNet-based models, install Mamba dependencies separately:

```bash
pip install triton==2.0.0
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
```

See `pretrained/README.md` for required checkpoint files.

## Dataset Preparation

Download or prepare the SD900 industrial defect dataset and place it under `data/` as described in [data/README.md](data/README.md).

Example layout:

```
data/sdsaliency900/
  train/images/
  train/masks/
  val/images/
  val/masks/
```

## Centralized Training

Train a model on the full dataset:

```bash
python train.py --model unet --dataset sd900 --gpu 0 --epochs 300
python train.py --model vmunet --dataset sd900 --gpu 0 --epochs 300 --wandb
python train.py --model hvmunet --dataset sd900 --gpu 0 --batch-size 2
```

Arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Architecture key | `unet` |
| `--dataset` | `sd900` | `sd900` |
| `--gpu` | CUDA device id | `0` |
| `--epochs` | Training epochs | `300` |
| `--batch-size` | Batch size | model default |
| `--lr` | Learning rate | `0.001` |
| `--wandb` | Enable W&B logging | off |

Checkpoints and logs are saved under `results/`.

## Synthetic Data Generation

Before segmentation training, you can augment datasets with a mask-conditioned DDPM under `generation/`. See [generation/README.md](generation/README.md) for full details.

```bash
# 1. Train DDPM on paired images and masks
python -m generation.train \
  --run-name sd900_ddpm \
  --image-path data/sd900_gen/images \
  --mask-path data/sd900_gen/masks \
  --num-classes 2 --batch-size 4 --image-size 256 --gpu 0

# 2. Sample synthetic image-mask pairs
python -m generation.sample \
  --run-name sd900_ddpm \
  --mask-path data/sdsaliency900/val/masks \
  --output data/synthetic/sd900_ddpm \
  --num-classes 2 --batch-size 4 --gpu 0

# 3. (Optional) Evaluate generation quality
python -m generation.eval_fid data/sdsaliency900/val/images data/synthetic/sd900_ddpm/0_0
```

Typical pipeline: **generate synthetic data → merge into `data/` → run segmentation training**.

## Federated Training

Train across simulated factory clients with FedAvg, FedProx, or SCAFFOLD:

```bash
# FedAvg on SD900
python train_federated.py --model unet --dataset sd900combine --method fedavg --gpu 0

# FedProx with non-IID split
python train_federated.py --model unet --dataset sd900combine --method fedprox --iid --gpu 0

# SCAFFOLD with combined real and synthetic industrial data
python train_federated.py --model unet --dataset sd900combine --method scaffold --num-clients 23 --gpu 0
```

Arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--method` | `fedavg`, `fedprox`, `scaffold` | `fedavg` |
| `--num-clients` | Number of clients | `10` |
| `--frac` | Client participation ratio per round | `0.2` |
| `--mu` | FedProx proximal coefficient | `0.01` |
| `--iid` | Use IID data partition | off (non-IID) |

## Metrics

Validation reports pixel-level metrics derived from the confusion matrix:

- **IoU** (mIoU for binary segmentation)
- **Dice** (F1 / DSC)
- **Accuracy**
- **Sensitivity** (recall)
- **Specificity**

## Acknowledgments

This project builds upon open-source implementations of UNet variants, Vision Mamba, TransUNet, and federated learning baselines. We thank the authors of VM-UNet, U²-Net, TransUNet, and related segmentation works for their publicly available code.

## License

See [LICENSE](LICENSE).

For the Traditional Chinese documentation, see [README_zh-TW.md](README_zh-TW.md).
