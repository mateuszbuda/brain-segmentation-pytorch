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

from misc.imutils import crf_inference_label
from voc12 import dataloader
from voc12.dataloader import get_img_path


def extract_valid_cams(img, cams, size, label, cfg, idx, img_name):
    idx = str(idx)
    cams = torch.sum(
        torch.stack(
            [F.interpolate(o, size, mode='bilinear', align_corners=False)[0] for o in cams]),
        0)

    valid_cat = torch.nonzero(label, as_tuple=False)[:, 0]

    cams = cams[valid_cat]
    cams /= F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5
    cams = cams.cpu().numpy()

    keys = np.pad(valid_cat.cpu().numpy() + 1, (1, 0), mode='constant')

    # 1. find confident fg & bg
    fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=cfg.conf_fg_thres)
    fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
    pred = crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
    fg_conf = keys[pred]

    bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=cfg.conf_bg_thres)
    bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
    pred = crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
    bg_conf = keys[pred]

    # 2. combine confident fg & bg
    conf = fg_conf.copy()
    conf[fg_conf == 0] = 255
    conf[bg_conf + fg_conf == 0] = 0

    output_folder = os.path.join(cfg.output_dir, idx)
    os.makedirs(output_folder, exist_ok=True)

    imageio.imwrite(os.path.join(output_folder, img_name + '.png'), conf.astype(np.uint8))


@hydra.main(config_path='../conf', config_name="tests/make_cam_crf/unet")
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
        img = np.asarray(imageio.imread(get_img_path(img_name, cfg.voc12_root)))

        with torch.set_grad_enabled(False):
            cams = [model(img[0].to(device)) for img in imgs]
            if type(cams[0]) == tuple:
                cams = list(zip(*cams))
                for idx, selected_cams in enumerate(cams):
                    extract_valid_cams(img, selected_cams, size, label, cfg, idx, img_name)
            else:
                extract_valid_cams(img, cams, size, label, cfg, 0, img_name)


def data_loaders(cfg):
    dataset = dataloader.VOC12ClassificationDatasetMSF(cfg.infer_list, voc12_root=cfg.voc12_root,
                                                       scales=[1.0, 0.5, 1.5, 2.0])

    loader = DataLoader(dataset, drop_last=False, num_workers=0)

    return loader


if __name__ == '__main__':
    run_app()
