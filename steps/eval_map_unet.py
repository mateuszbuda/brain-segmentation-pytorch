import importlib
import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_ap_score
from voc12 import dataloader
from voc12.dataloader import TorchvisionNormalize


@hydra.main(config_path='../conf', config_name="eval_map_unet")
def run_app(cfg: DictConfig) -> None:
    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    loader_train, loader_valid = data_loaders(cfg)
    loaders = {"train": loader_train, "valid": loader_valid}

    model = getattr(importlib.import_module(cfg.model), 'Net')(in_channels=3, init_features=cfg.mid_ch, out_channels=21)
    state_dict = torch.load(cfg.weights, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    for phase in ["train", "valid"]:
        total_cnt, running_ap = 0, 0.
        model.eval()
        loader = loaders[phase]
        for i, batch in tqdm(enumerate(loader), total=len(loader.dataset) // loader.batch_size):
            img = batch['img']
            cls_label = batch['label']
            img, cls_label, = img.to(device), cls_label.to(device)

            with torch.set_grad_enabled(False):
                cls_pred = model(img)
                if type(cls_pred) == tuple:
                    cls_pred = cls_pred[0]

                with torch.set_grad_enabled(False):
                    outputs = torch.sigmoid(cls_pred)
                    total_cnt += cls_label.size(0)
                    running_ap += get_ap_score(cls_label.cpu().detach().numpy(),
                                               outputs.cpu().detach().numpy())

        logging.info(f'mAP {phase}, {float(running_ap) / total_cnt}')


def data_loaders(cfg):
    dataset_train = dataloader.VOC12PseudoSegmentationDataset(
        cfg.train_list, crop_size=cfg.crop_size, voc12_root=cfg.voc12_root,
        rescale=None, hor_flip=True,
        crop_method='random', resize_long=(320, 640)
    )

    dataset_valid = dataloader.VOC12PseudoSegmentationDataset(cfg.val_list, crop_size=cfg.crop_size,
                                                              img_normal=TorchvisionNormalize(),
                                                              voc12_root=cfg.voc12_root)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
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
