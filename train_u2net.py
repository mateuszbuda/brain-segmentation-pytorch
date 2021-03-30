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
from models.u2netp import U2NETP
from utils import get_ap_score, log_images
from voc12 import dataloader
from voc12.dataloader import TorchvisionNormalize


def loss_multipliers(epoch, max_epoch):
    mid_epoch = max_epoch // 2
    if epoch > mid_epoch:
        return 0.5, 0.5
    start = 0.5
    end = 0.9
    steps_cnt = 10
    step = (end - start) / steps_cnt
    ranges = np.arange(start, end, step).tolist()[::-1]
    first = ranges[epoch // steps_cnt]
    second = 1 - first
    return first, second


bce_loss = nn.BCELoss(size_average=True)
mlsm_loss = nn.MultiLabelSoftMarginLoss(size_average=True)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    labels_v = labels_v.unsqueeze(1).float()
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss


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


@hydra.main(config_path='./conf', config_name="train_u2netp")
def run_app(cfg: DictConfig) -> None:
    makedirs(cfg)
    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    loader_train, loader_valid = data_loaders(cfg)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = U2NETP(in_ch=3, out_ch=1, threshold=cfg.threshold)
    unet.to(device)

    optimizer = optim.Adam(unet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    logger = Logger(cfg.logs)
    loss_train, loss_valid, cls_loss_train, seg_loss_valid, seg_loss_train, cls_loss_valid = [], [], [], [], [], []
    step = 0

    for epoch in tqdm(range(cfg.epochs), total=cfg.epochs):
        total_cnt, running_ap = 0, 0.
        print(f'Epoch: {epoch}')
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1
                img = data['img']
                cls_label = data['cls_label']
                seg_label = data['seg_label']
                img, cls_label, seg_label = img.to(device), cls_label.to(device), seg_label.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    seg_label_pred, cls_label_pred, bottleneck = unet(img)
                    seg_label_pseudo = []
                    bottleneck1, bottleneck2, bottleneck3, bottleneck4, bottleneck5, bottleneck6 = bottleneck
                    for l1, l2, l3, l4, l5, l6, j, k in zip(bottleneck1, bottleneck2, bottleneck3, bottleneck4,
                                                            bottleneck5, bottleneck6, cls_label, seg_label):
                        seg_label_pseudo.append(unet.create_seg_label_from_cls_labels(l1, l2, l3, l4, l5, l6, j, k))

                    seg_label_pseudo = torch.cat(seg_label_pseudo, axis=0).to(device)
                    d0, d1, d2, d3, d4, d5, d6 = seg_label_pred
                    c0, c1, c2, c3, c4, c5, c6 = cls_label_pred
                    seg_loss2, seg_loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, seg_label_pseudo)
                    cls_loss2, cls_loss = muti_margin_loss_fusion(c0, c1, c2, c3, c4, c5, c6, cls_label)

                    with torch.set_grad_enabled(False):
                        outputs = torch.sigmoid(c0)
                        total_cnt += cls_label.size(0)
                        running_ap += get_ap_score(cls_label.cpu().detach().numpy(),
                                                   outputs.cpu().detach().numpy())
                    lambda1, lambda2 = loss_multipliers(epoch, cfg.epochs)
                    loss = lambda1 * cls_loss + lambda2 * seg_loss
                    if phase == "valid":
                        loss_valid.append(loss.item())
                        seg_loss_valid.append(seg_loss.item())
                        cls_loss_valid.append(cls_loss.item())
                        if (epoch % cfg.vis_freq == 0) or (epoch == cfg.epochs - 1):
                            if i * cfg.batch_size < cfg.vis_images:
                                tag = "image/{}".format(i)
                                num_images = cfg.vis_images - i * cfg.batch_size
                                logger.image_list_summary(
                                    tag,
                                    log_images(img, seg_label_pseudo, seg_label_pred[0], cfg.threshold)[
                                    :num_images],
                                    step,
                                )

                    if phase == "train":
                        loss_train.append(loss.item())
                        cls_loss_train.append(cls_loss.item())
                        seg_loss_train.append(seg_loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    log_loss_summary(logger, loss_train, step, tag='loss')
                    log_loss_summary(logger, [lambda1], step, tag='lamdba1')
                    log_loss_summary(logger, [lambda2], step, tag='lamdba2')
                    log_loss_summary(logger, cls_loss_train, step, tag="cls_loss")
                    log_loss_summary(logger, seg_loss_train, step, tag="seg_loss")
                    log_loss_summary(logger, [float(running_ap) / total_cnt], step, tag="mAP")

                    loss_train = []
                    cls_loss_train = []
                    seg_loss_train = []

            if phase == "valid":
                log_loss_summary(logger, loss_valid, step, tag="val_loss")
                log_loss_summary(logger, cls_loss_valid, step, tag="val_cls_loss")
                log_loss_summary(logger, seg_loss_valid, step, tag="val_seg_loss")
                log_loss_summary(logger, [float(running_ap) / total_cnt], step, tag="val_mAP")

                torch.save(unet.state_dict(), os.path.join(cfg.weights, "u2netp.pt"))
                loss_valid = []
                cls_loss_valid = []
                seg_loss_valid = []
            print('\nmAP', float(running_ap) / total_cnt)


def data_loaders(cfg):
    dataset_train = dataloader.VOC12SegmentationDataset(
        cfg.train_list, cfg.label_dir, cfg.crop_size, cfg.voc12_root,
        rescale=None, hor_flip=False,
        img_normal=TorchvisionNormalize(),
        crop_method='random', resize_long=(320, 640)
    )
    dataset_valid = dataloader.VOC12SegmentationDataset(cfg.val_list, cfg.label_dir, cfg.crop_size,
                                                        img_normal=TorchvisionNormalize(),
                                                        voc12_root=cfg.voc12_root)

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


def log_loss_summary(logger, loss, step, tag):
    logger.scalar_summary(tag, np.mean(loss), step)


def makedirs(cfg):
    os.makedirs(cfg.weights, exist_ok=True)
    os.makedirs(cfg.logs, exist_ok=True)


if __name__ == '__main__':
    run_app()
