from core.dataset.transforms import DataTransform_Test as DataTransform
from torch.utils.data import DataLoader
from core.dataset.utils import create_loader_for_infer as create_data_loader
import torch
from core.model.utils import build_model
from collections import defaultdict
from core.dataset.utils import save_reconstructions
from mmcv import Config
from tqdm import tqdm
import numpy as np
import argparse
import pathlib
import sys

def load_model(cfg):
    model = build_model(cfg)
    if cfg.data_parallel:
        model = torch.nn.DataParallel(model)
    checkpoint = torch.load(cfg.infer_cfg.ckpt)
    model.load_state_dict(checkpoint['model'])
    return model


def infer(cfg, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (input, mean, std, fnames, slices) in tqdm(data_loader):
            input = input.unsqueeze(1).to(cfg.device)
            recons = model(input).to('cpu').squeeze(1)
            for i in range(recons.shape[0]):
                recons[i] = recons[i] * std[i] + mean[i]
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions


def main(cfg):
    data_loader = create_data_loader(cfg)
    model = load_model(cfg)
    reconstructions = infer(cfg, model, data_loader)
    save_reconstructions(reconstructions, cfg.out_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--mask-kspace', action='store_true',
                        help='Whether to apply a mask (set to True for val data and False '
                             'for test data')
    parser.add_argument('--data-path', default=None, type=pathlib.Path)
    parser.add_argument('--ckpt', type=pathlib.Path, help='Path to the model')
    parser.add_argument('--out-dir', type=pathlib.Path, help='Path to save the reconstructions to')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    cfg.mask_kspace = args.mask_kspace

    if args.data_path:
        cfg.infer_cfg.data_path = args.data_path
    else:
        cfg.infer_cfg.data_path = pathlib.Path(cfg.infer_cfg.data_path)
    
    if args.ckpt:
        cfg.infer_cfg.ckpt = args.ckpt
    else:
        cfg.infer_cfg.ckpt = pathlib.Path(cfg.infer_cfg.ckpt)

    if args.out_dir:
        cfg.infer_cfg.out_dir = args.out_dir
    else:
        cfg.infer_cfg.out_dir = pathlib.Path(cfg.infer_cfg.out_dir)
    if args.device:
        cfg.infer_cfg.device
    main(cfg)