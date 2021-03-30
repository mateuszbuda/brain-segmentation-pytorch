import argparse

import numpy as np

from voc12 import dataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", default='train_aug.txt', type=str)
    parser.add_argument("--val_list", default='val.txt', type=str)
    parser.add_argument("--out", default="cls_labels.npy", type=str)
    parser.add_argument("--voc12_root", default="../../../Dataset/VOC2012", type=str)
    args = parser.parse_args()

    train_name_list = dataloader.load_img_name_list(args.train_list)
    val_name_list = dataloader.load_img_name_list(args.val_list)

    train_val_name_list = np.concatenate([train_name_list, val_name_list], axis=0)
    label_list = dataloader.load_image_label_list_from_xml(train_val_name_list, args.voc12_root)

    total_label = np.zeros(20)

    d = dict()
    for img_name, label in zip(train_val_name_list, label_list):
        d[img_name] = label
        total_label += label

    print(total_label)
    np.save(args.out, d)
