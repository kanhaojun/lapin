import numpy as np
import torch
import torch.nn as nn

from models.attunet_r2unet import AttU_Net, R2AttU_Net, R2U_Net
from models.hvmunet import HVMUNet
from models.resunet import ResUnet
from models.resunetplus import ResUnetPlusPlus
from models.tunet import TUNet, CONFIGS as CONFIGS_TUNet
from models.u2net import U2NET
from models.unet_model import UNET
from models.unetpp.unet2plus import UNet_2Plus
from models.unetppp.unet3plus import UNet_3Plus
from models.vmunet.vmunet import VMUNet
from models.vmunet.vmunet_v2 import VMUNetV2
from utils import BceDiceLoss, BceDiceLoss2, DiceLoss


class U2NetLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super().__init__()
        self.bce = nn.BCELoss(reduction='mean')
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        d0, d1, d2, d3, d4, d5, d6 = pred
        losses = []
        for output in (d0, d1, d2, d3, d4, d5, d6):
            losses.append(self.bce(output, target) + self.dice(output, target))
        return losses[0] + 0.5 * sum(losses[1:])


MODEL_CONFIGS = {
    'unet': {
        'model_config': {'num_classes': 1, 'input_channels': 3},
        'batch_size': 2,
    },
    'vmunet': {
        'model_config': {
            'num_classes': 1,
            'input_channels': 3,
            'depths': [2, 2, 2, 2],
            'depths_decoder': [2, 2, 2, 1],
            'drop_path_rate': 0.2,
            'load_ckpt_path': './pretrained/vmamba_small_e238_ema.pth',
        },
        'batch_size': 3,
    },
    'vmunet-v2': {
        'model_config': {
            'num_classes': 1,
            'input_channels': 3,
            'depths': [2, 2, 9, 2],
            'depths_decoder': [2, 2, 2, 1],
            'drop_path_rate': 0.2,
            'load_ckpt_path': './pretrained/vmamba_small_e238_ema.pth',
            'deep_supervision': True,
        },
        'batch_size': 2,
    },
    'hvmunet': {
        'model_config': {
            'num_classes': 1,
            'input_channels': 3,
            'c_list': [8, 16, 32, 64, 128, 256],
            'split_att': 'fc',
            'bridge': True,
            'drop_path_rate': 0.4,
        },
        'batch_size': 2,
    },
    'u2net': {
        'model_config': {'num_classes': 1, 'input_channels': 3},
        'batch_size': 2,
        'criterion': 'u2net',
    },
    'unetpp': {
        'model_config': {'num_classes': 1, 'input_channels': 3},
        'batch_size': 2,
    },
    'unetppp': {
        'model_config': {'num_classes': 1, 'input_channels': 3},
        'batch_size': 2,
    },
    'tunet': {
        'model_config': {'num_classes': 1, 'input_channels': 3},
        'batch_size': 2,
        'criterion': 'bce_dice2',
    },
    'resunet': {
        'model_config': {'num_classes': 1, 'input_channels': 3},
        'batch_size': 2,
    },
    'resunetpp': {
        'model_config': {'num_classes': 1, 'input_channels': 3},
        'batch_size': 2,
    },
    'attu': {
        'model_config': {'num_classes': 1, 'input_channels': 3},
        'batch_size': 2,
    },
    'r2u': {
        'model_config': {'num_classes': 1, 'input_channels': 3, 't': 3},
        'batch_size': 2,
    },
    'attr2u': {
        'model_config': {'num_classes': 1, 'input_channels': 3, 't': 3},
        'batch_size': 2,
    },
}


def build_criterion(name):
    if name == 'u2net':
        return U2NetLoss(wb=1, wd=1)
    if name == 'bce_dice2':
        return BceDiceLoss2(wb=1, wd=1)
    return BceDiceLoss(wb=1, wd=1)


def build_model(network, model_config):
    if network == 'unet':
        model = UNET(
            n_channels=model_config['input_channels'],
            n_classes=model_config['num_classes'],
            bilinear=False,
        )
        model.load_from()
        return model

    if network == 'vmunet':
        model = VMUNet(
            num_classes=model_config['num_classes'],
            input_channels=model_config['input_channels'],
            depths=model_config['depths'],
            depths_decoder=model_config['depths_decoder'],
            drop_path_rate=model_config['drop_path_rate'],
            load_ckpt_path=model_config['load_ckpt_path'],
        )
        model.load_from()
        return model

    if network == 'vmunet-v2':
        model = VMUNetV2(
            num_classes=model_config['num_classes'],
            input_channels=model_config['input_channels'],
            depths=model_config['depths'],
            depths_decoder=model_config['depths_decoder'],
            drop_path_rate=model_config['drop_path_rate'],
            load_ckpt_path=model_config['load_ckpt_path'],
            deep_supervision=model_config['deep_supervision'],
        )
        model.load_from()
        return model

    if network == 'hvmunet':
        return HVMUNet(
            num_classes=model_config['num_classes'],
            input_channels=model_config['input_channels'],
            c_list=model_config['c_list'],
            split_att=model_config['split_att'],
            bridge=model_config['bridge'],
            drop_path_rate=model_config['drop_path_rate'],
        )

    if network == 'u2net':
        return U2NET(
            in_ch=model_config['input_channels'],
            out_ch=model_config['num_classes'],
        )

    if network == 'unetpp':
        return UNet_2Plus(
            in_channels=model_config['input_channels'],
            n_classes=model_config['num_classes'],
        )

    if network == 'unetppp':
        return UNet_3Plus(
            in_channels=model_config['input_channels'],
            n_classes=model_config['num_classes'],
        )

    if network == 'tunet':
        config_vit = CONFIGS_TUNet['R50-ViT-B_16']
        model = TUNet(
            num_classes=model_config['num_classes'],
            input_channels=model_config['input_channels'],
        )
        model.load_from(weights=np.load(config_vit.pretrained_path))
        return model

    if network == 'resunet':
        return ResUnet(channel=model_config['input_channels'])

    if network == 'resunetpp':
        return ResUnetPlusPlus(channel=model_config['input_channels'])

    if network == 'attu':
        return AttU_Net(
            img_ch=model_config['input_channels'],
            output_ch=model_config['num_classes'],
        )

    if network == 'r2u':
        return R2U_Net(
            img_ch=model_config['input_channels'],
            output_ch=model_config['num_classes'],
            t=model_config.get('t', 3),
        )

    if network == 'attr2u':
        return R2AttU_Net(
            img_ch=model_config['input_channels'],
            output_ch=model_config['num_classes'],
            t=model_config.get('t', 3),
        )

    raise ValueError(f'Unsupported network: {network}')
