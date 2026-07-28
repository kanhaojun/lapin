#!/usr/bin/env python3
"""Compute FID between real and generated image folders."""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Compute FID score(s).')
    parser.add_argument('real_dir', help='Real images folder or root with subfolders')
    parser.add_argument('generated_dir', help='Generated images folder or root with subfolders')
    parser.add_argument('--recursive', action='store_true', help='Compute FID per subfolder and average')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def compute_fid(real_path, generated_path, batch_size, device):
    from pytorch_fid import fid_score
    return fid_score.calculate_fid_given_paths(
        [real_path, generated_path],
        batch_size=batch_size,
        device=device,
        dims=2048,
    )


def list_subdirs(path):
    return sorted(f.path for f in os.scandir(path) if f.is_dir())


def main():
    args = parse_args()
    if not args.recursive:
        fid = compute_fid(args.real_dir, args.generated_dir, args.batch_size, args.device)
        print(f'FID: {fid:.4f}')
        return

    real_subdirs = list_subdirs(args.real_dir)
    generated_subdirs = list_subdirs(args.generated_dir)
    if len(real_subdirs) != len(generated_subdirs):
        print('Warning: subfolder counts differ; pairing by order.', file=sys.stderr)

    fid_values = []
    for idx, (real_sub, gen_sub) in enumerate(zip(real_subdirs, generated_subdirs)):
        fid = compute_fid(real_sub, gen_sub, args.batch_size, args.device)
        fid_values.append(fid)
        print(f'Subfolder {idx + 1}: {fid:.4f}')

    print(f'Average FID: {sum(fid_values) / len(fid_values):.4f}')


if __name__ == '__main__':
    main()
