import os

import hydra
import imageio
import numpy as np
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(config_path='../conf', config_name="eval_sem")
def run_app(cfg: DictConfig) -> None:
    dataset = VOCSemanticSegmentationDataset(split=cfg.infer_set, data_dir=cfg.voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    preds = []
    for id in tqdm(dataset.ids):
        cls_labels = imageio.imread(os.path.join(cfg.output_dir, id + '.png')).astype(np.uint8)
        cls_labels[cls_labels == 255] = 0
        preds.append(cls_labels.copy())

    confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator

    print(fp[0], fn[0])
    print(np.mean(fp[1:]), np.mean(fn[1:]))

    print({'iou': iou, 'miou': np.nanmean(iou)})


if __name__ == '__main__':
    run_app()
