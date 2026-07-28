# Pretrained weights

Some models require external pretrained checkpoints:

| Model | File | Source |
|-------|------|--------|
| VMUNet / VMUNet-V2 | `vmamba_small_e238_ema.pth` | [VMamba](https://github.com/MzeroMiko/VMamba) |
| TransUNet (TUNet) | `vit_checkpoint/imagenet21k/R50+ViT-B_16.npz` | [TransUNet](https://github.com/Beckschen/TransUNet) |

Recommended layout:

```
pretrained/
  vmamba_small_e238_ema.pth
  vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

UNet can initialize from ImageNet-pretrained encoders via `load_from()` when available.

Weight files are not included in this repository because of size and licensing constraints.
