import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import Logger
from models.unet import UNetClassifier
from train import log_loss_summary, makedirs
from utils import get_ap_score
from voc12 import dataloader
from voc12.dataloader import TorchvisionNormalize


@hydra.main(config_path='./conf', config_name="train_cls")
def run_app(cfg: DictConfig) -> None:
    makedirs(cfg)
    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    loader_train, loader_valid = data_loaders(cfg)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNetClassifier(in_channels=3, out_channels=20, threshold=cfg.threshold, init_features=cfg.init_features)
    unet.to(device)

    optimizer = optim.AdamW(unet.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logger = Logger(cfg.logs)
    loss_train, loss_valid = [], []
    step = 0
    for epoch in tqdm(range(cfg.epochs), total=cfg.epochs):
        learning_rates = []
        total_cnt, running_ap = 0., 0
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
                img, cls_label = img.to(device), cls_label.to(device)
                optimizer.zero_grad()
                for param_group in optimizer.param_groups:
                    learning_rates.append(param_group['lr'])
                with torch.set_grad_enabled(phase == "train"):
                    cls_label_pred = unet(img)
                    loss = F.multilabel_soft_margin_loss(cls_label_pred, cls_label)

                    with torch.set_grad_enabled(False):
                        outputs = torch.sigmoid(cls_label_pred)
                        total_cnt += cls_label.size(0)
                        running_ap += get_ap_score(cls_label.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                    if phase == "valid":
                        loss_valid.append(loss.item())

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    log_loss_summary(logger, loss_train, step, tag='loss')
                    log_loss_summary(logger, [float(running_ap) / total_cnt], step, tag="mAP")
                    loss_train = []

            if phase == "valid":
                log_loss_summary(logger, loss_valid, step, tag="val_loss")
                log_loss_summary(logger, [float(running_ap) / total_cnt], step, tag="val_mAP")

                torch.save(unet.state_dict(), os.path.join(cfg.weights, "unet.pt"))
                loss_valid = []
            log_loss_summary(logger, learning_rates, step, tag="lr")
            print('\nmAP', float(running_ap) / total_cnt)


def data_loaders(cfg):
    dataset_train = dataloader.VOC12SegmentationDataset(
        cfg.train_list, cfg.label_dir, cfg.crop_size, cfg.voc12_root,
        rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
        crop_method='random', resize_long=(320, 640)
    )
    dataset_valid = dataloader.VOC12SegmentationDataset(cfg.val_list, cfg.label_dir, cfg.crop_size,
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


if __name__ == '__main__':
    run_app()
