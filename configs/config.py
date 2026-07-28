from datetime import datetime
from types import SimpleNamespace

from torchvision import transforms

from models.registry import MODEL_CONFIGS, build_criterion
from utils import (
    myNormalize,
    myRandomHorizontalFlip,
    myRandomRotation,
    myRandomVerticalFlip,
    myResize,
    myToTensor,
)

DATASET_PATHS = {
    'sd900': {'data_path': './data/sdsaliency900/'},
    'sd900combine': {
        'data_path': './data/sdsaliency900/',
        'data_path_aux': './data/sd900_syn_all_local_relay_diff/',
    },
    'isic18': {'data_path': './data/isic2018/'},
    'isic17': {'data_path': './data/isic2017/'},
}

OPTIMIZER_DEFAULTS = {
    'opt': 'AdamW',
    'lr': 0.001,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'weight_decay': 1e-2,
    'amsgrad': False,
    'sch': 'CosineAnnealingLR',
    'T_max': 50,
    'eta_min': 0.00001,
    'last_epoch': -1,
}


def build_config(
    network='unet',
    dataset='sd900',
    mode='centralized',
    fed_method='fedavg',
    gpu_id='0',
    epochs=300,
    batch_size=None,
    num_clients=10,
    mu=0.01,
    learning_rate=None,
    n_minibatch=1,
):
    if network not in MODEL_CONFIGS:
        raise ValueError(f'Unsupported network: {network}')
    if dataset not in DATASET_PATHS:
        raise ValueError(f'Unsupported dataset: {dataset}')

    model_info = MODEL_CONFIGS[network]
    dataset_info = DATASET_PATHS[dataset]
    criterion_name = model_info.get('criterion', 'bce_dice')

    input_size_h = 256
    input_size_w = 256
    train_transformer = transforms.Compose([
        myNormalize(dataset, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w),
    ])
    test_transformer = transforms.Compose([
        myNormalize(dataset, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w),
    ])

    run_tag = f'{network}_{dataset}'
    if mode == 'federated':
        run_tag = f'fed_{fed_method}_{run_tag}'

    config = SimpleNamespace(
        network=network,
        model_config=dict(model_info['model_config']),
        datasets=dataset,
        data_path=dataset_info['data_path'],
        data_path_aux=dataset_info.get('data_path_aux'),
        criterion=build_criterion(criterion_name),
        num_classes=1,
        input_size_h=input_size_h,
        input_size_w=input_size_w,
        input_channels=3,
        distributed=False,
        local_rank=-1,
        num_workers=0,
        seed=42,
        amp=False,
        gpu_id=gpu_id,
        batch_size=batch_size or model_info.get('batch_size', 2),
        epochs=epochs,
        print_interval=20,
        val_interval=30,
        save_interval=100,
        threshold=0.5,
        train_transformer=train_transformer,
        test_transformer=test_transformer,
        work_dir=(
            f'results/{run_tag}_'
            f'{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
        ),
        mode=mode,
        fed_method=fed_method,
        num_clients=num_clients,
        num_users_info=False,
        mu=mu,
        learning_rate=learning_rate or OPTIMIZER_DEFAULTS['lr'],
        n_minibatch=n_minibatch,
        **OPTIMIZER_DEFAULTS,
    )
    if learning_rate is not None:
        config.lr = learning_rate
    return config
