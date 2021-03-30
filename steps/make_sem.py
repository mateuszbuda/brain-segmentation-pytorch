import importlib
import os

import hydra
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from voc12 import dataloader


@hydra.main(config_path='../conf', config_name="make_sem")
def run_app(cfg: DictConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    data_loader = data_loaders(cfg)
    model = getattr(importlib.import_module(cfg.model), 'Segmentation')(in_ch=3, mid_ch=cfg.mid_ch, out_ch=21,
                                                               share_classifier=cfg.share_classifier)
    model.load_state_dict(torch.load(cfg.weights, map_location='cpu'), strict=True)
    model.to(device)
    model.eval()

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        img_name = data['name'][0]
        imgs = data['img']

        with torch.set_grad_enabled(False):
            cls_pred, seg_pred = model(imgs.to(device))
            d0, d1, d2, d3, d4, d5, d6 = seg_pred
            sem_seg = torch.argmax(d0, 1)[0].cpu().numpy()
            imageio.imsave(os.path.join(cfg.output_dir, img_name + '.png'), sem_seg.astype(np.uint8))


def data_loaders(cfg):
    dataset = dataloader.VOC12ClassificationDataset(cfg.infer_list, voc12_root=cfg.voc12_root)

    loader = DataLoader(dataset, drop_last=False, num_workers=0)

    return loader


if __name__ == '__main__':
    run_app()
