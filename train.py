#!/usr/bin/env python3
"""Centralized training entry point for medical image segmentation."""

import argparse
import os
import sys
import warnings

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from configs.config import build_config
from datasets.dataset import NPY_datasets
from engine import test_one_epoch, train_one_epoch, val_one_epoch
from models.registry import build_model
from utils import (
    cal_params_flops,
    get_logger,
    get_optimizer,
    get_scheduler,
    log_config_info,
    set_seed,
)

warnings.filterwarnings('ignore')

SUPPORTED_MODELS = [
    'unet', 'vmunet', 'vmunet-v2', 'hvmunet', 'u2net', 'unetpp', 'unetppp',
    'tunet', 'resunet', 'resunetpp', 'attu', 'r2u', 'attr2u',
]
SUPPORTED_DATASETS = ['isic18', 'isic17']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a segmentation model on a centralized dataset.',
    )
    parser.add_argument('--model', default='unet', choices=SUPPORTED_MODELS)
    parser.add_argument('--dataset', default='isic18', choices=SUPPORTED_DATASETS)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true')
    return parser.parse_args()


def setup_logging(config):
    os.makedirs(config.work_dir, exist_ok=True)
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs_dir = os.path.join(config.work_dir, 'outputs')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    logger = get_logger('train', log_dir)
    writer = SummaryWriter(os.path.join(config.work_dir, 'summary'))
    return logger, writer, checkpoint_dir


def main():
    args = parse_args()
    config = build_config(
        network=args.model,
        dataset=args.dataset,
        mode='centralized',
        gpu_id=args.gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    config.seed = args.seed

    wandb = None
    if args.wandb:
        import wandb as wandb_module
        wandb = wandb_module
        wandb.init(
            project='lapin',
            config={
                'learning_rate': config.lr,
                'architecture': args.model,
                'dataset': args.dataset,
                'epochs': config.epochs,
            },
        )

    logger, writer, checkpoint_dir = setup_logging(config)
    log_config_info(config, logger)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    train_dataset = NPY_datasets(config.data_path, config, train=True)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=True,
    )

    model = build_model(config.network, config.model_config).cuda()
    cal_params_flops(model, config.input_size_h, logger)

    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    min_loss = float('inf')
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['min_loss']
        min_epoch = checkpoint['min_epoch']
        logger.info(
            f'Resumed from {resume_model} at epoch {checkpoint["epoch"]}, '
            f'min_loss={min_loss:.4f}'
        )

    step = 0
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        step, _ = train_one_epoch(
            train_loader, model, criterion, optimizer, scheduler,
            epoch, step, logger, config, writer, wandb, args,
        )
        loss = val_one_epoch(
            val_loader, model, criterion, epoch, logger, config, wandb, args,
        )

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save({
            'epoch': epoch,
            'min_loss': min_loss,
            'min_epoch': min_epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, resume_model)

    best_path = os.path.join(checkpoint_dir, 'best.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location='cpu'))
        test_one_epoch(
            val_loader, model, criterion, logger, config, wandb, args,
        )
        os.rename(
            best_path,
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth'),
        )

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
