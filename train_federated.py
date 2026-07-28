#!/usr/bin/env python3
"""Federated learning entry point for industrial image segmentation."""

import argparse
import copy
import os
import warnings

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from configs.config import build_config
from datasets.dataset import NPY_datasets
from engine import (
    test_img,
    train_one_epoch_fedavg,
    train_one_epoch_fedprox,
    train_one_epoch_scaffold,
    val_one_epoch,
)
from federated.data import average_weights, build_client_dataloaders
from federated.scaffold import compute_delta_c, initialize_c, update_c
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

SUPPORTED_MODELS = ['unet', 'vmunet', 'vmunet-v2', 'hvmunet']
SUPPORTED_DATASETS = ['sd900', 'sd900combine', 'isic18', 'isic17']
SUPPORTED_METHODS = ['fedavg', 'fedprox', 'scaffold']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a segmentation model with federated learning.',
    )
    parser.add_argument('--model', default='unet', choices=SUPPORTED_MODELS)
    parser.add_argument('--dataset', default='sd900combine', choices=SUPPORTED_DATASETS)
    parser.add_argument('--method', default='fedavg', choices=SUPPORTED_METHODS)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-clients', type=int, default=10)
    parser.add_argument('--frac', type=float, default=0.2)
    parser.add_argument('--mu', type=float, default=0.01, help='FedProx proximal term')
    parser.add_argument('--iid', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    return parser.parse_args()


def setup_logging(config):
    os.makedirs(config.work_dir, exist_ok=True)
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = SummaryWriter(os.path.join(config.work_dir, 'summary'))
    return logger, writer, checkpoint_dir


def select_clients(epoch, num_clients, frac):
    m = max(int(frac * num_clients), 1)
    loop_index = max(int(1 / frac), 1)
    begin_index = (epoch % loop_index) * m
    end_index = begin_index + m
    return list(range(num_clients))[begin_index:end_index]


def train_client_fedavg(client, global_model, client_loader, criterion, config,
                        optimizer_cfg, epoch, step, logger, writer, wandb, args):
    local_model = copy.deepcopy(global_model)
    optimizer = get_optimizer(optimizer_cfg, local_model)
    return train_one_epoch_fedavg(
        client_loader, local_model, criterion, optimizer, None,
        epoch, step, logger, config, writer, wandb, args,
    )


def train_client_fedprox(client, global_model, client_loader, criterion, config,
                         optimizer_cfg, epoch, step, logger, writer, wandb, args, mu):
    local_model = copy.deepcopy(global_model)
    optimizer = get_optimizer(optimizer_cfg, local_model)
    return train_one_epoch_fedprox(
        client_loader, local_model, global_model, criterion, optimizer, None,
        epoch, step, logger, config, writer, mu, wandb, args,
    )


def train_client_scaffold(client, global_model, client_loader, criterion, config,
                          optimizer_cfg, epoch, step, logger, writer, wandb, args,
                          c_global, c_locals):
    local_model = copy.deepcopy(global_model)
    optimizer = get_optimizer(optimizer_cfg, local_model)
    scheduler = get_scheduler(config, optimizer)
    steps = len(client_loader) * config.n_minibatch
    c_local = c_locals[client]
    state_params_diff = [c_l - c_g for c_l, c_g in zip(c_local, c_global)]
    local_model, local_loss = train_one_epoch_scaffold(
        client_loader, local_model, criterion, optimizer, scheduler,
        epoch, step, logger, config, writer, state_params_diff, wandb, args,
    )
    c_delta = compute_delta_c(c_local, c_global, config.learning_rate, steps)
    return local_model, local_loss, c_delta


def main():
    args = parse_args()
    config = build_config(
        network=args.model,
        dataset=args.dataset,
        mode='federated',
        fed_method=args.method,
        gpu_id=args.gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_clients=args.num_clients,
        mu=args.mu,
    )

    wandb = None
    if args.wandb:
        import wandb as wandb_module
        wandb = wandb_module
        wandb.init(
            project='lapin',
            config={
                'method': args.method,
                'architecture': args.model,
                'dataset': args.dataset,
                'epochs': config.epochs,
                'num_clients': args.num_clients,
                'iid': args.iid,
            },
        )

    logger, writer, checkpoint_dir = setup_logging(config)
    log_config_info(config, logger)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    train_datasets = [NPY_datasets(config.data_path, config, train=True)]
    if config.data_path_aux:
        train_datasets.append(NPY_datasets(config.data_path_aux, config, train=True))

    client_dataloaders = build_client_dataloaders(
        train_datasets,
        config.num_clients,
        config.batch_size,
        config.num_workers,
        iid=args.iid,
        verbose=config.num_users_info,
    )

    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=True,
    )

    global_model = build_model(config.network, config.model_config).cuda()
    cal_params_flops(global_model, config.input_size_h, logger)

    criterion = config.criterion
    optimizer = get_optimizer(config, global_model)
    scheduler = get_scheduler(config, optimizer)

    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    min_loss = float('inf')
    start_epoch = 1
    min_epoch = 1
    step = 0

    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model, map_location='cpu')
        global_model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['min_loss']
        min_epoch = checkpoint['min_epoch']
        logger.info(f'Resumed federated run from epoch {checkpoint["epoch"]}')

    c_global = None
    c_locals = None
    if args.method == 'scaffold':
        c_global = initialize_c(global_model)
        c_locals = [initialize_c(global_model) for _ in range(config.num_clients)]

    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        logger.info(f'Round {epoch}, global lr={current_lr:.6f}')

        selected = select_clients(epoch, config.num_clients, args.frac)
        local_weights = []
        local_losses = []
        c_deltas = []

        for client in selected:
            logger.info(f'Training client {client + 1}/{config.num_clients}')
            config.lr = current_lr

            if args.method == 'fedavg':
                local_model, local_loss = train_client_fedavg(
                    client, global_model, client_dataloaders[client], criterion,
                    config, config, epoch, step, logger, writer, wandb, args,
                )
            elif args.method == 'fedprox':
                local_model, local_loss = train_client_fedprox(
                    client, global_model, client_dataloaders[client], criterion,
                    config, config, epoch, step, logger, writer, wandb, args, config.mu,
                )
            else:
                local_model, local_loss, c_delta = train_client_scaffold(
                    client, global_model, client_dataloaders[client], criterion,
                    config, config, epoch, step, logger, writer, wandb, args,
                    c_global, c_locals,
                )
                c_deltas.append(c_delta)

            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_losses.append(local_loss)

        avg_loss = sum(local_losses) / len(local_losses)
        logger.info(f'Round {epoch} average client loss: {avg_loss:.4f}')
        writer.add_scalar(f'{args.method}/round_avg_loss', avg_loss, epoch)
        if args.wandb:
            wandb.log({'round': epoch, 'average_loss': avg_loss})

        global_model.load_state_dict(average_weights(local_weights))

        if args.method == 'scaffold':
            c_global = update_c(c_locals[selected[0]], c_global, c_deltas[0], 1 / len(selected))
            for i in range(1, len(selected)):
                c_global = update_c(c_locals[selected[i]], c_global, c_deltas[i], 1 / len(selected))
            for i, client_idx in enumerate(selected):
                c_locals[client_idx] = [
                    c_g + c_d for c_g, c_d in zip(c_global, c_deltas[i])
                ]

        test_img(val_loader, global_model, criterion, epoch, logger, config, args)
        loss = val_one_epoch(
            val_loader, global_model, criterion, epoch, logger, config, wandb, args,
        )

        if loss < min_loss:
            torch.save(global_model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save({
            'epoch': epoch,
            'min_loss': min_loss,
            'min_epoch': min_epoch,
            'loss': loss,
            'model_state_dict': global_model.state_dict(),
        }, resume_model)

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
