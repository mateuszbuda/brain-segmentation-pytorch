import logging
import os

import hydra
import numpy as np
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path='../conf', config_name="tests/eval_cam")
def run_app(cfg: DictConfig) -> None:
    dataset = VOCSemanticSegmentationDataset(split=cfg.infer_set, data_dir=cfg.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    for subdir in os.listdir(cfg.cam_out_dir):
        folder = os.path.join(cfg.cam_out_dir, subdir)
        preds = []
        for id in tqdm(dataset.ids):
            cam_dict = np.load(os.path.join(folder, id + '.npy'), allow_pickle=True).item()
            cams = cam_dict['high_res']
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=cfg.cam_eval_thres)
            keys = np.pad(cam_dict['keys'].cpu() + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            preds.append(cls_labels.copy())

        confusion = calc_semantic_segmentation_confusion(preds, labels)

        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou = gtjresj / denominator

        print(f'iou: {iou}\tmiou: {np.nanmean(iou)}\tFolder: {folder}')
        logging.info(f'iou: {iou}\tmiou: {np.nanmean(iou)}\tFolder: {folder}')


if __name__ == '__main__':
    run_app()
