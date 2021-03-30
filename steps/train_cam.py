import importlib
import os

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import Logger
from utils import get_ap_score, log_loss_summary, makedirs
from voc12 import dataloader

mlsm_loss = nn.MultiLabelSoftMarginLoss(size_average=True)


def muti_margin_loss_fusion(c0, c1, c2, c3, c4, c5, c6, labels_v):
    loss0 = mlsm_loss(c0, labels_v)
    loss1 = mlsm_loss(c1, labels_v)
    loss2 = mlsm_loss(c2, labels_v)
    loss3 = mlsm_loss(c3, labels_v)
    loss4 = mlsm_loss(c4, labels_v)
    loss5 = mlsm_loss(c5, labels_v)
    loss6 = mlsm_loss(c6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss


@hydra.main(config_path='../conf', config_name="classifier/train_cam")
def run_app(cfg: DictConfig) -> None:
    makedirs(cfg)
    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    loader_train, loader_valid = data_loaders(cfg)
    loaders = {"train": loader_train, "valid": loader_valid}
    model = getattr(importlib.import_module(cfg.model), 'Net')(in_ch=3, mid_ch=cfg.mid_ch, out_ch=1,
                                                               share_classifier=cfg.share_classifier)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    logger = Logger(cfg.logs)
    loss_train, loss_valid, cls_loss_train, cls_loss_valid = [], [], [], []
    step = 0

    for epoch in tqdm(range(cfg.epochs), total=cfg.epochs):
        print(f'Epoch: {epoch}')
        for phase in ["train", "valid"]:
            total_cnt, running_ap0, running_ap1, running_ap2, running_ap3, running_ap4, running_ap5, running_ap6 = 0, 0., 0., 0., 0., 0., 0., 0.
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
                    c0, c1, c2, c3, c4, c5, c6 = model(img)
                    cls_loss2, loss = muti_margin_loss_fusion(c0, c1, c2, c3, c4, c5, c6, label)

                    with torch.set_grad_enabled(False):
                        total_cnt += label.size(0)
                        running_ap0 += get_ap_score(label.cpu().detach().numpy(),
                                                    torch.sigmoid(c0).cpu().detach().numpy())
                        running_ap1 += get_ap_score(label.cpu().detach().numpy(),
                                                    torch.sigmoid(c1).cpu().detach().numpy())
                        running_ap2 += get_ap_score(label.cpu().detach().numpy(),
                                                    torch.sigmoid(c2).cpu().detach().numpy())
                        running_ap3 += get_ap_score(label.cpu().detach().numpy(),
                                                    torch.sigmoid(c3).cpu().detach().numpy())
                        running_ap4 += get_ap_score(label.cpu().detach().numpy(),
                                                    torch.sigmoid(c4).cpu().detach().numpy())
                        running_ap5 += get_ap_score(label.cpu().detach().numpy(),
                                                    torch.sigmoid(c5).cpu().detach().numpy())
                        running_ap6 += get_ap_score(label.cpu().detach().numpy(),
                                                    torch.sigmoid(c6).cpu().detach().numpy())

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        cls_loss_valid.append(cls_loss2.item())
                    if phase == "train":
                        loss_train.append(loss.item())
                        cls_loss_train.append(cls_loss2.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    log_loss_summary(logger, loss_train, step, tag='total_loss')
                    log_loss_summary(logger, cls_loss_train, step, tag="loss0")
                    log_loss_summary(logger, [float(running_ap0) / total_cnt], step, tag="mAP0")
                    log_loss_summary(logger, [float(running_ap1) / total_cnt], step, tag="mAP1")
                    log_loss_summary(logger, [float(running_ap2) / total_cnt], step, tag="mAP2")
                    log_loss_summary(logger, [float(running_ap3) / total_cnt], step, tag="mAP3")
                    log_loss_summary(logger, [float(running_ap4) / total_cnt], step, tag="mAP4")
                    log_loss_summary(logger, [float(running_ap5) / total_cnt], step, tag="mAP5")
                    log_loss_summary(logger, [float(running_ap6) / total_cnt], step, tag="mAP6")

            if phase == "valid":
                log_loss_summary(logger, loss_valid, step, tag="val_total_loss")
                log_loss_summary(logger, cls_loss_valid, step, tag="val_loss0")
                log_loss_summary(logger, [float(running_ap0) / total_cnt], step, tag="val_mAP0")
                log_loss_summary(logger, [float(running_ap1) / total_cnt], step, tag="val_mAP1")
                log_loss_summary(logger, [float(running_ap2) / total_cnt], step, tag="val_mAP2")
                log_loss_summary(logger, [float(running_ap3) / total_cnt], step, tag="val_mAP3")
                log_loss_summary(logger, [float(running_ap4) / total_cnt], step, tag="val_mAP4")
                log_loss_summary(logger, [float(running_ap5) / total_cnt], step, tag="val_mAP5")
                log_loss_summary(logger, [float(running_ap6) / total_cnt], step, tag="val_mAP6")

                torch.save(model.state_dict(), os.path.join(cfg.weights, "model.pt"))
            loss_valid = []
            cls_loss_valid = []
            print('\nmAP', float(running_ap0) / total_cnt)


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
