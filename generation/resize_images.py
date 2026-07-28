#!/usr/bin/env python3
"""Resize an image tree to a fixed resolution while preserving folder layout."""

import argparse
import os

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Batch resize images.')
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--size', type=int, default=256)
    return parser.parse_args()


def resize_tree(input_dir, output_dir, size):
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue
            src = os.path.join(root, filename)
            rel = os.path.relpath(src, input_dir)
            dst = os.path.join(output_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with Image.open(src) as img:
                img.convert('RGB').resize((size, size), Image.LANCZOS).save(dst)


if __name__ == '__main__':
    args = parse_args()
    resize_tree(args.input_dir, args.output_dir, args.size)
    print(f'Resized images saved to {args.output_dir}')
