#!/usr/bin/env python3
"""Train a mask-conditioned DDPM for synthetic medical image generation."""

import argparse
import copy
import logging
import os

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from generation.dataset import build_train_loader
from generation.diffusion import Diffusion
from generation import lr_scheduler
from generation.modules import EMA, UNet_mask
from generation.utils import ensure_run_dirs, save_image_grid

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')

logging.basicConfig(
    format='%(asctime)s - %(levelname)s: %(message)s',
    level=logging.INFO,
    datefmt='%I:%M:%S',
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train mask-conditioned DDPM.')
    parser.add_argument('--run-name', type=str, required=True, help='Experiment name')
    parser.add_argument('--image-path', type=str, required=True, help='ImageFolder root')
    parser.add_argument('--mask-path', type=str, required=True, help='Mask root with label subfolders')
    parser.add_argument('--mask-suffix', type=str, default='', help='Optional mask filename suffix')
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save-every', type=int, default=5, help='Save preview every N epochs')
    return parser.parse_args()


def estimate_epochs(dataloader_len, batch_size, num_classes):
    return max(1, int(6.8e6 * num_classes / (dataloader_len * batch_size)))


def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ensure_run_dirs(args.run_name)
    checkpoint_dir = os.path.join('generation', 'checkpoints', args.run_name)
    output_dir = os.path.join('generation', 'outputs', args.run_name)

    dataloader = build_train_loader(args)
    model = UNet_mask(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler.lr_lambda)
    criterion = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    epochs = estimate_epochs(len(dataloader), args.batch_size, args.num_classes)
    logging.info('Dataset size=%d, planned epochs=%d', len(dataloader.dataset), epochs)

    preview_masks = None
    for epoch in range(epochs):
        logging.info('Starting epoch %d/%d', epoch + 1, epochs)
        pbar = tqdm(dataloader)
        for images, masks, labels in pbar:
            images = images.to(device) * 2 - 1
            masks = masks.to(device)
            labels = labels.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, masks, t, None)
            loss = criterion(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.step_ema(ema_model, model)
            pbar.set_postfix(MSE=loss.item())
            preview_masks = masks[:args.num_classes]

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'ckpt_latest.pt'))
        torch.save(ema_model.state_dict(), os.path.join(checkpoint_dir, 'ema_ckpt_latest.pt'))
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optim_latest.pt'))

        if epoch % args.save_every == 0 and preview_masks is not None:
            labels = torch.arange(args.num_classes).long().to(device)
            samples = diffusion.sample(
                ema_model,
                n=len(labels),
                masks=preview_masks,
                labels=labels,
            )
            save_image_grid(preview_masks, os.path.join(output_dir, f'{epoch}_mask.png'))
            save_image_grid(samples, os.path.join(output_dir, f'{epoch}_ema.png'))
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'ckpt_{epoch}.pt'))
            torch.save(ema_model.state_dict(), os.path.join(checkpoint_dir, f'ema_ckpt_{epoch}.pt'))


if __name__ == '__main__':
    train(parse_args())
