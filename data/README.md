# Dataset layout

Place downloaded or prepared datasets under `data/` using the following structure.

## SD900 (Saliency900) — Industrial Surface Defect

Primary dataset for industrial surface-defect segmentation:

```
data/sdsaliency900/
  train/
    images/
      *.png
    masks/
      *.png
  val/
    images/
      *.png
    masks/
      *.png
```

Each image is a product-surface capture; masks annotate defect regions (e.g. scratches, stains, dents).

## Federated industrial dataset (optional)

For federated experiments with combined real and synthetic data across multiple factory clients:

```
data/sdsaliency900/
  train/images/
  train/masks/
  val/images/
  val/masks/

data/sd900_syn_all_local_relay_diff/
  train/images/
  train/masks/
  val/images/
  val/masks/
```

Use `--dataset sd900combine` in `train_federated.py` to merge both sources.

## Download / preparation

- SD900 (Saliency900): prepare from your industrial inspection pipeline or internal defect-annotation release.
- Typical train/val split ratio is 7:3; ensure image and mask filenames match between `images/` and `masks/` folders.

After placing the files, verify that image and mask filenames match between the `images/` and `masks/` folders.

## Synthetic data (generation module)

The `generation/` module produces synthetic image-mask pairs via a mask-conditioned DDPM. Typical layout for **training the generator**:

```
data/sd900_gen/
  images/
    0/
      *.png
    1/
      *.png
  masks/
    0/
      *.png
    1/
      *.png
```

Sampled outputs are usually written to:

```
data/synthetic/<run-name>/
  0_0/
    *_img.png
    *_mask.png
```

For federated segmentation, synthetic data can be merged with real client data under paths such as `data/sd900_syn_all_local_relay_diff/`. See [generation/README.md](../generation/README.md).
