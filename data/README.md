# Dataset layout

Place downloaded datasets under `data/` using the following structure.

## ISIC 2017 / ISIC 2018

```
data/isic2018/
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

The same layout applies to `data/isic2017/`.

## Federated industrial dataset (optional)

For federated experiments with combined real and synthetic data:

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

## Download

- ISIC challenge data: https://challenge.isic-archive.com/data
- Preprocessed ISIC17/ISIC18 splits (7:3) are commonly shared by segmentation benchmark releases.

After placing the files, verify that image and mask filenames match between the `images/` and `masks/` folders.

## Synthetic data (generation module)

The `generation/` module produces synthetic image-mask pairs via a mask-conditioned DDPM. Typical layout for **training the generator**:

```
data/isic2018_gen/
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
