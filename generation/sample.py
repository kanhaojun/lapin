#!/usr/bin/env python3
"""Generate synthetic image-mask pairs from a trained DDPM."""

import argparse
import logging
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from generation.dataset import build_mask_loader
from generation.diffusion import Diffusion
from generation.modules import UNet_mask
from generation.utils import resolve_checkpoint

logging.basicConfig(
    format='%(asctime)s - %(levelname)s: %(message)s',
    level=logging.INFO,
    datefmt='%I:%M:%S',
)


def parse_args():
    parser = argparse.ArgumentParser(description='Sample images from a trained DDPM.')
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--mask-path', type=str, required=True, help='MaskFolder root for conditioning')
    parser.add_argument('--output', type=str, required=True, help='Directory to save generated pairs')
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--channels', type=int, default=3, help='1 for grayscale, 3 for RGB output')
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional explicit checkpoint path')
    parser.add_argument('--gpu', type=str, default='0')
    return parser.parse_args()


def save_pair(image, mask, image_path, mask_path, channels):
    if channels > 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(image_path, image)
    cv2.imwrite(mask_path, (mask * 255 * 50).astype(np.uint8))


def sample(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataloader = build_mask_loader(args)
    model = UNet_mask(num_classes=args.num_classes).to(device)

    checkpoint_dir = os.path.join('generation', 'checkpoints', args.run_name)
    checkpoint_path = resolve_checkpoint(checkpoint_dir, args.checkpoint)
    logging.info('Loading checkpoint: %s', checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)

    diffusion = Diffusion(img_size=args.image_size, device=device)
    os.makedirs(args.output, exist_ok=True)

    for class_id in range(args.num_classes):
        class_dir = os.path.join(args.output, f'{class_id}_{class_id}')
        os.makedirs(class_dir, exist_ok=True)
        logging.info('Generating class %d into %s', class_id, class_dir)

        for batch_idx, (masks, labels, names) in enumerate(tqdm(dataloader)):
            masks = masks.to(device)
            labels = torch.full((args.batch_size,), class_id, dtype=torch.long, device=device)
            samples = diffusion.sample(model, args.batch_size, masks, labels).cpu().numpy()
            mask_np = masks.squeeze(1).cpu().numpy()

            if args.channels > 1:
                samples = np.transpose(samples, (0, 2, 3, 1))

            for i in range(samples.shape[0]):
                img_path = os.path.join(class_dir, f'{batch_idx}_{i}_img.png')
                msk_path = os.path.join(class_dir, f'{batch_idx}_{i}_mask.png')
                save_pair(samples[i], mask_np[i], img_path, msk_path, args.channels)


if __name__ == '__main__':
    sample(parse_args())
