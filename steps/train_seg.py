import importlib
import logging
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
from steps.train_seg_unet import loss_multipliers
from utils import get_ap_score, log_images
from voc12 import dataloader
from voc12.dataloader import TorchvisionNormalize

ce_loss = nn.CrossEntropyLoss(size_average=True)
mlsm_loss = nn.MultiLabelSoftMarginLoss(size_average=True)


def muti_ce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    labels_v = labels_v.long()
    weights = torch.sum(labels_v, axis=[1, 2])
    weights[weights != 0] = 1
    d0 = d0 * weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    d1 = d1 * weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    d2 = d2 * weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    d3 = d3 * weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    d4 = d4 * weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    d5 = d5 * weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    d6 = d6 * weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)

    labels_v = labels_v.long()
    loss0 = ce_loss(d0, labels_v)
    loss1 = ce_loss(d1, labels_v)
    loss2 = ce_loss(d2, labels_v)
    loss3 = ce_loss(d3, labels_v)
    loss4 = ce_loss(d4, labels_v)
    loss5 = ce_loss(d5, labels_v)
    loss6 = ce_loss(d6, labels_v)

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


@hydra.main(config_path='../conf', config_name="train_seg_finetune")
def run_app(cfg: DictConfig) -> None:
    makedirs(cfg)
    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    loader_train, loader_valid, dataset_train, dataset_valid = data_loaders(cfg)
    loaders = {"train": loader_train, "valid": loader_valid}

    model = getattr(importlib.import_module(cfg.model), 'Segmentation')(in_ch=3, mid_ch=cfg.mid_ch, out_ch=21,
                                                                        share_classifier=cfg.share_classifier)
    if cfg.get('finetune', False):
        logging.info(f'Finetuning from {cfg.finetune_from}')
        model.load_state_dict(torch.load(cfg.finetune_from, map_location='cpu'), strict=True)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    logger = Logger(cfg.logs)
    loss_train, loss_valid, cls_loss_train, seg_loss_valid, seg_loss_train, cls_loss_valid = [], [], [], [], [], []
    step = 0

    for epoch in tqdm(range(10, cfg.epochs), total=cfg.epochs):
        print(f'Epoch: {epoch}')
        for phase in ["train", "valid"]:
            total_cnt, running_ap = 0, 0.
            if phase == "train":
                model.train()
            else:
                model.eval()
            loader = loaders[phase]
            for i, batch in tqdm(enumerate(loader), total=len(loader.dataset) // loader.batch_size):
                if phase == "train":
                    step += 1
                img = batch['img']
                cls_label = batch['label']
                seg_label = batch['seg_label']
                img, cls_label, seg_label = img.to(device), cls_label.to(device), seg_label.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    cls_pred, seg_pred = model(img)
                    c0, c1, c2, c3, c4, c5, c6 = cls_pred
                    d0, d1, d2, d3, d4, d5, d6 = seg_pred

                    seg_loss2, seg_loss = muti_ce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, seg_label)
                    cls_loss2, cls_loss = muti_margin_loss_fusion(c0, c1, c2, c3, c4, c5, c6, cls_label)

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
                                    log_images(img, seg_label, d0)[:num_images],
                                    step,
                                )

                    if phase == "train":
                        loss_train.append(loss.item())
                        cls_loss_train.append(cls_loss.item())
                        seg_loss_train.append(seg_loss.item())
                        loss.backward()
                        optimizer.step()
                    with torch.set_grad_enabled(False):
                        outputs = torch.sigmoid(c0)
                        total_cnt += cls_label.size(0)
                        running_ap += get_ap_score(cls_label.cpu().detach().numpy(),
                                                   outputs.cpu().detach().numpy())

                if phase == "train" and (step + 1) % 10 == 0:
                    log_loss_summary(logger, loss_train, step, tag='loss')
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

                torch.save(model.state_dict(), os.path.join(cfg.weights, "u2netp.pt"))
                loss_valid = []
                cls_loss_valid = []
                seg_loss_valid = []
            print('\nmAP', float(running_ap) / total_cnt)
            logging.info(f'\nmAP {float(running_ap) / total_cnt} Epoch {epoch}')
        _, lambda2 = loss_multipliers(epoch + 1, cfg.epochs)
        if lambda2 > 0:
            model.eval()
            print('\nCreating Pseudo Labels for training')
            for data in tqdm(dataset_train, total=len(dataset_train)):
                idx = data['idx']
                img_i = [torch.from_numpy(img_ii).unsqueeze(0).to(device) for img_ii in data['img']]
                pseudo_label_i = model.generate_pseudo_label(img_i, data['label'], data['size'],
                                                             cam_order=cfg.cam_order)
                loader_train.dataset.update_cam(idx, pseudo_label_i.cpu().numpy())
            print('\nCreating Pseudo Labels for validation')
            for data in tqdm(dataset_valid, total=len(dataset_valid)):
                idx = data['idx']
                img_i = [torch.from_numpy(img_ii).unsqueeze(0).to(device) for img_ii in data['img']]
                pseudo_label_i = model.generate_pseudo_label(img_i, data['label'], data['size'],
                                                             cam_order=cfg.cam_order)
                loader_valid.dataset.update_cam(idx, pseudo_label_i.cpu().numpy())


def data_loaders(cfg):
    dataset_train = dataloader.VOC12PseudoSegmentationDataset(
        cfg.train_list, crop_size=cfg.crop_size, voc12_root=cfg.voc12_root,
        rescale=None, hor_flip=True,
        crop_method='random', resize_long=(320, 640),
        cam_eval_thres=cfg.cam_eval_thres
    )

    dataset_valid = dataloader.VOC12PseudoSegmentationDataset(cfg.val_list, crop_size=cfg.crop_size,
                                                              img_normal=TorchvisionNormalize(),
                                                              voc12_root=cfg.voc12_root,
                                                              cam_eval_thres=cfg.cam_eval_thres)

    train_dataset = dataloader.VOC12ClassificationDatasetMSF(cfg.train_list, voc12_root=cfg.voc12_root,
                                                             scales=[1.0, 0.5, 1.5, 2.0])
    valid_dataset = dataloader.VOC12ClassificationDatasetMSF(cfg.val_list, voc12_root=cfg.voc12_root,
                                                             scales=[1.0, 0.5, 1.5, 2.0])

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

    return loader_train, loader_valid, train_dataset, valid_dataset


def log_loss_summary(logger, loss, step, tag):
    logger.scalar_summary(tag, np.mean(loss), step)


def makedirs(cfg):
    os.makedirs(cfg.weights, exist_ok=True)
    os.makedirs(cfg.logs, exist_ok=True)


if __name__ == '__main__':
    run_app()
