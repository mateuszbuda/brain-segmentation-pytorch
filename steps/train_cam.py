import importlib
import logging
import os
from statistics import mean

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import Logger
from utils import get_ap_score, log_loss_summary
from voc12 import dataloader

mlsm_loss = nn.MultiLabelSoftMarginLoss(reduction='mean')


def multi_margin_loss_fusion(class_preds, labels_v):
    if type(class_preds) == tuple:
        losses = [mlsm_loss(c, labels_v) for c in class_preds]
        return losses, sum(losses)
    else:
        loss = mlsm_loss(class_preds, labels_v)
        return [loss], loss


def log_losses(logger, step, phase, losses_dict, ap_scores, total_cnt):
    losses = losses_dict[phase]

    log_loss_summary(logger, losses[0], step, tag=f'{phase}_total_loss')
    for idx, tr_loss in enumerate(losses):
        if idx == 0:
            continue
        if sum(losses[idx]) != 0:
            log_loss_summary(logger, losses[idx], step, tag=f'{phase}_block{idx}_loss')
    log_loss_summary(logger, float(ap_scores[0]) / total_cnt, step, tag=f'{phase}_mAP')
    for idx, ap_score in enumerate(ap_scores):
        if idx == 0:
            continue
        if ap_scores[idx] != 0:
            log_loss_summary(logger, float(ap_scores[idx]) / total_cnt, step, tag=f'{phase}_block{idx}_mAP')


@hydra.main(config_path='../conf', config_name="tests/train_cam/unet")
def run_app(cfg: DictConfig) -> None:
    os.makedirs(cfg.weights, exist_ok=True)
    os.makedirs(cfg.logs, exist_ok=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    loader_train, loader_valid = data_loaders(cfg)
    loaders = {"train": loader_train, "valid": loader_valid}
    model = getattr(importlib.import_module(cfg.model), 'Net')(
        in_ch=3,
        mid_ch=cfg.mid_ch,
        out_ch=cfg.out_ch,
        num_classes=cfg.num_classes,
        share_classifier=cfg.share_classifier)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    logger = Logger(cfg.logs)
    step = 0
    losses_dict = {
        'train': [[] for _ in range(7)],
        'valid': [[] for _ in range(7)]
    }
    best_val_loss = 10 * 10
    counter = 0
    for epoch in tqdm(range(cfg.epochs), total=cfg.epochs):
        print(f'Epoch: {epoch}')
        for phase in ["train", "valid"]:
            total_cnt = 0
            ap_scores = [0. for _ in range(7)]
            if phase == "train":
                model.train()
            else:
                model.eval()
            loader = loaders[phase]
            for i, data in tqdm(enumerate(loader), total=len(loader.dataset) // loader.batch_size):
                if phase == "train":
                    step += 1
                img = data['img']
                label = data['label']
                img, label = img.to(device), label.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    class_preds = model(img)
                    losses, loss = multi_margin_loss_fusion(class_preds, label)
                    for idx, l in enumerate(losses):
                        losses_dict[phase][idx].append(l.item())

                    with torch.set_grad_enabled(False):
                        total_cnt += label.size(0)
                        if type(class_preds) == tuple:
                            for idx, c in enumerate(class_preds):
                                ap_scores[idx] += get_ap_score(label.cpu().detach().numpy(),
                                                               torch.sigmoid(c).cpu().detach().numpy())
                        else:
                            ap_scores[0] += get_ap_score(label.cpu().detach().numpy(),
                                                         torch.sigmoid(class_preds).cpu().detach().numpy())
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    log_losses(logger, step, phase, losses_dict, ap_scores, total_cnt)
                    losses_dict.update({phase: [[] for _ in range(7)]})
            if phase == "valid":
                mean_loss = mean(losses_dict[phase][0])
                if mean_loss < best_val_loss:
                    logging.info(f'Validation Loss decreased from {best_val_loss} to {mean_loss}')
                    best_val_loss = mean_loss
                    torch.save(model.state_dict(), os.path.join(cfg.weights, "best_model.pt"))
                    counter = 0
                else:
                    counter += 1
                    logging.info(f'Validation Loss is not decreasing for the {counter} times')
                log_losses(logger, step, phase, losses_dict, ap_scores, total_cnt)
                losses_dict.update({phase: [[] for _ in range(7)]})
        if counter >= cfg.get('early_stopping', 15):
            break
        torch.save(model.state_dict(), os.path.join(cfg.weights, "model.pt"))


def data_loaders(cfg):
    dataset_train = dataloader.VOC12ClassificationDataset(cfg.train_list, voc12_root=cfg.voc12_root,
                                                          resize_long=(320, 640), hor_flip=True,
                                                          crop_size=cfg.crop_size, crop_method="random")
    dataset_valid = dataloader.VOC12ClassificationDataset(cfg.val_list, voc12_root=cfg.voc12_root,
                                                          crop_size=cfg.crop_size)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=cfg.batch_size,
        drop_last=False,
        num_workers=cfg.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


if __name__ == '__main__':
    run_app()
