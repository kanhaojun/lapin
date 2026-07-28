import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImageMaskDataset(Dataset):
    """ImageFolder-style dataset with paired masks under label subfolders."""

    def __init__(self, image_dir, mask_dir, transform=None, mask_suffix=''):
        self.transform = transform
        self.mask_suffix = mask_suffix
        self.image_folder = torchvision.datasets.ImageFolder(image_dir, transform=None)
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        img_path, label = self.image_folder.imgs[idx]
        mask_name = os.path.basename(img_path)
        if self.mask_suffix:
            base, ext = os.path.splitext(mask_name)
            mask_name = f'{base}{self.mask_suffix}{ext}'
        mask_path = os.path.join(self.mask_dir, str(label), mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask, label


class MaskDataset(Dataset):
    """Mask-only dataset for mask-conditioned sampling."""

    def __init__(self, mask_dir, transform=None):
        self.transform = transform
        self.mask_folder = torchvision.datasets.ImageFolder(mask_dir, transform=None)

    def __len__(self):
        return len(self.mask_folder)

    def __getitem__(self, idx):
        mask_path, label = self.mask_folder.imgs[idx]
        mask = Image.open(mask_path)
        name = os.path.basename(mask_path)
        if self.transform:
            mask = self.transform(mask)
        return mask, label, name


def build_transform(image_size):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
    ])


def build_train_loader(args):
    transform = build_transform(args.image_size)
    dataset = ImageMaskDataset(
        args.image_path,
        args.mask_path,
        transform=transform,
        mask_suffix=args.mask_suffix,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


def build_mask_loader(args):
    transform = build_transform(args.image_size)
    dataset = MaskDataset(args.mask_path, transform=transform)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
