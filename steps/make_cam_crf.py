import os

import hydra
import imageio
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from misc.imutils import crf_inference_label
from voc12 import dataloader
from voc12.dataloader import get_img_path


def extract_valid_cams(img, cams, keys, cfg, idx, img_name):
    idx = str(idx)
    keys = np.pad(keys + 1, (1, 0), mode='constant')

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


@hydra.main(config_path='../conf', config_name="unet/make_cam_crf")
def run_app(cfg: DictConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    data_loader = data_loaders(cfg)

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        img_name = data['name'][0]
        img = np.asarray(imageio.imread(get_img_path(img_name, cfg.voc12_root)))

        for subdir in os.listdir(cfg.cam_out_dir):
            folder = os.path.join(cfg.cam_out_dir, subdir)

            cam_dict = np.load(os.path.join(folder, img_name + '.npy'), allow_pickle=True).item()
            cams = cam_dict['high_res']
            keys = cam_dict['keys']

            extract_valid_cams(img, cams, keys, cfg, subdir, img_name)


def data_loaders(cfg):
    dataset = dataloader.VOC12ImageDataset(cfg.infer_list, voc12_root=cfg.voc12_root, to_torch=False)

    loader = DataLoader(dataset, drop_last=False, num_workers=0)

    return loader


if __name__ == '__main__':
    run_app()
