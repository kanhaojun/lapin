import os

import numpy as np
import torch
import torchvision
from PIL import Image


def save_image_grid(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).cpu().numpy()
    if ndarr.dtype != np.uint8:
        ndarr = (ndarr * 255).astype(np.uint8)
    Image.fromarray(ndarr).save(path)


def ensure_run_dirs(run_name, root='generation'):
    for sub in ('checkpoints', 'outputs'):
        os.makedirs(os.path.join(root, sub, run_name), exist_ok=True)


def resolve_checkpoint(checkpoint_dir, preferred=None):
    if preferred and os.path.exists(preferred):
        return preferred
    latest = os.path.join(checkpoint_dir, 'ema_ckpt_latest.pt')
    if os.path.exists(latest):
        return latest
    candidates = sorted(
        f for f in os.listdir(checkpoint_dir)
        if f.startswith('ema_ckpt_') and f.endswith('.pt')
    )
    if not candidates:
        raise FileNotFoundError(f'No EMA checkpoint found in {checkpoint_dir}')
    return os.path.join(checkpoint_dir, candidates[-1])
