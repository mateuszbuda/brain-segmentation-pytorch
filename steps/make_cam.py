import importlib
import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from voc12 import dataloader


def extract_valid_cams(cams, size, label, cfg, idx, img_name):
    idx = str(idx)
    strided_cam = torch.sum(
        torch.stack(
            [F.interpolate(o, size, mode='bilinear', align_corners=False)[0] for o in cams]),
        0)

    valid_cat = torch.nonzero(label, as_tuple=False)[:, 0]

    strided_cam = strided_cam[valid_cat]
    strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5
    output_folder = os.path.join(cfg.output_dir, idx)
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, img_name + '.npy'),
            {"keys": valid_cat.cpu().numpy(), "high_res": strided_cam.cpu().numpy()})


@hydra.main(config_path='../conf', config_name="tests/make_cam/unet")
def run_app(cfg: DictConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    data_loader = data_loaders(cfg)
    model = getattr(importlib.import_module(cfg.model), 'CAM')(
        in_ch=3,
        mid_ch=cfg.mid_ch,
        out_ch=cfg.out_ch,
        num_classes=cfg.num_classes,
        share_classifier=cfg.share_classifier)
    model.load_state_dict(torch.load(cfg.weights, map_location='cpu'), strict=True)
    model.to(device)
    model.eval()

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        img_name = data['name'][0]
        label = data['label'][0]
        imgs = data['img']
        size = data['size']
        label = label.to(device)

        with torch.set_grad_enabled(False):
            cams = [model(img[0].to(device)) for img in imgs]
            if type(cams[0]) == tuple:
                cams = list(zip(*cams))
                for idx, selected_cams in enumerate(cams):
                    extract_valid_cams(selected_cams, size, label, cfg, idx, img_name)
            else:
                extract_valid_cams(cams, size, label, cfg, 0, img_name)


def data_loaders(cfg):
    dataset = dataloader.VOC12ClassificationDatasetMSF(cfg.infer_list, voc12_root=cfg.voc12_root,
                                                       scales=[1.0, 0.5, 1.5, 2.0])

    loader = DataLoader(dataset, drop_last=False, num_workers=0)

    return loader


if __name__ == '__main__':
    run_app()
