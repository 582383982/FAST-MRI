from core.train_settings import build_optimizer
from core.dataset.utils import create_data_loaders
from core.model.utils import build_model
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from mmcv import Config
from tqdm import tqdm
import numpy as np
import torchvision
import argparse
import pathlib
import torch
import logging
import time
import shutil
def save_model(cfg, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'cfg': cfg,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

def load_model(cfg):
    checkpoint = torch.load(cfg.train_cfg.ckpt)
    load_cfg = checkpoint['cfg']
    model = build_model(load_cfg)
    if load_cfg.train_cfg.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optimizer(load_cfg.train_cfg.optimizer, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer

def train_epoch(cfg, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    with tqdm(total=len(data_loader), postfix=[dict(loss=0, avg_loss=0)]) as t:
        for iter, data in enumerate(data_loader):
            input, target, mean, std, norm = data
            input = input.unsqueeze(1).to(cfg.device)
            target = target.to(cfg.device)

            output = model(input).squeeze(1)
            loss = F.l1_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
            writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

            t.postfix[0]["loss"] = '%.4f' % (loss.item())
            t.postfix[0]["avg_loss"] = '%.4f' % (avg_loss)
            t.update()
            start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(cfg, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    
    with torch.no_grad():
        with tqdm(total=len(data_loader), postfix=[dict(avg_loss=0)]) as t:
            for iter, data in enumerate(data_loader):
                input, target, mean, std, norm = data
                input = input.unsqueeze(1).to(cfg.device)
                target = target.to(cfg.device)
                output = model(input).squeeze(1)

                mean = mean.unsqueeze(1).unsqueeze(2).to(cfg.device)
                std = std.unsqueeze(1).unsqueeze(2).to(cfg.device)
                target = target * std + mean
                output = output * std + mean

                norm = norm.unsqueeze(1).unsqueeze(2).to(cfg.device)
                loss = F.mse_loss(output / norm, target / norm, size_average=False)
                losses.append(loss.item())
                t.postfix[0]["avg_loss"] = '%.4f' % (np.mean(losses))
                t.update()
            writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
    return np.mean(losses), time.perf_counter() - start


def visualize(cfg, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, mean, std, norm = data
            input = input.unsqueeze(1).to(cfg.device)
            target = target.unsqueeze(1).to(cfg.device)
            output = model(input)
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target - output), 'Error')
            break


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--cfg', help='train config file path',
                        default='configs/baseline_unet.py')
    # parser.add_argument('--work_dir', help='the dir to save logs and models')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    # prepare log
    log_dir = pathlib.Path(cfg.exp_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir/'summary')

    # model
    if cfg.train_cfg.resume:
        checkpoint, model, optimizer = load_model(cfg)
        cfg = checkpoint['cfg']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(cfg)
        if cfg.train_cfg.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optimizer(
            cfg.train_cfg.optimizer, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, **cfg.train_cfg.lr_scheduler)

    # prepare data
    train_loader, dev_loader, display_loader = create_data_loaders(cfg)

    for epoch in range(start_epoch, cfg.train_cfg.num_epochs):
        scheduler.step(epoch)
        train_loss, train_time = train_epoch(
            cfg, epoch, model, train_loader, optimizer, writer)
        dev_loss, dev_time = evaluate(cfg, epoch, model, dev_loader, writer)
        visualize(cfg, epoch, model, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(cfg, log_dir, epoch, model,
                   optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{cfg.train_cfg.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


if __name__ == '__main__':
    main()
