import os.path
import pathlib

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from misc import imutils
from misc.imutils import crf_inference_label

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
IGNORE = 255

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

N_CAT = len(CAT_LIST)

CAT_NAME_TO_NUM = dict(zip(CAT_LIST, range(len(CAT_LIST))))

cls_labels_dict = np.load(f'{os.path.dirname(pathlib.Path(__file__).absolute())}/cls_labels.npy',
                          allow_pickle=True).item()


def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]


def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    elem_list = minidom.parse(
        os.path.join(voc12_root, ANNOT_FOLDER_NAME, decode_int_filename(img_name) + '.xml')).getElementsByTagName(
        'name')

    multi_cls_lab = np.zeros((N_CAT), np.float32)

    for elem in elem_list:
        cat_name = elem.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab


def load_image_label_list_from_xml(img_name_list, voc12_root):
    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]


def load_image_label_list_from_npy(img_name_list):
    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])


def get_img_path(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')


def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.int32)

    return img_name_list


class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img = imutils.random_lr_flip(img)

        if self.crop_size:
            if self.crop_method == "random":
                img = imutils.random_crop(img, self.crop_size, 0)
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)

        if self.to_torch:
            img = imutils.HWC_to_CHW(img)

        return {'name': name_str, 'img': img, 'idx': idx}


class VOC12ClassificationDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, voc12_root,
                         resize_long, rescale, img_normal, hor_flip,
                         crop_size, crop_method)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        out['label'] = torch.from_numpy(self.label_list[idx])
        return out


class VOC12ClassificationDatasetMSF(VOC12ClassificationDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 img_normal=TorchvisionNormalize(),
                 scales=(1.0,)):
        self.scales = scales

        super().__init__(img_name_list_path, voc12_root, img_normal=img_normal)
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = imageio.imread(get_img_path(name_str, self.voc12_root))

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        out = {"name": name_str, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": torch.from_numpy(self.label_list[idx]), 'idx': idx}
        return out


class VOC12SegmentationDataset(Dataset):

    def __init__(self, img_name_list_path, label_dir, crop_size, voc12_root,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_method='random'):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root

        self.label_dir = label_dir

        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = imageio.imread(get_img_path(name_str, self.voc12_root))
        label = imageio.imread(os.path.join(self.label_dir, name_str + '.png'))

        img = np.asarray(img)

        if self.rescale:
            img, label = imutils.random_scale((img, label), scale_range=self.rescale, order=(3, 0))

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img, label = imutils.random_lr_flip((img, label))

        if self.crop_method == "random":
            img, label = imutils.random_crop((img, label), self.crop_size, (0, 255))
        else:
            img = imutils.top_left_crop(img, self.crop_size, 0)
            label = imutils.top_left_crop(label, self.crop_size, 255)

        img = imutils.HWC_to_CHW(img)

        return {'name': name, 'img': img, 'label': label, 'idx': idx}



class VOC12PseudoSegmentationDataset(VOC12ClassificationDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, superpixel_dir=None, cam_eval_thres=None):
        super().__init__(img_name_list_path, voc12_root,
                         resize_long, rescale, img_normal, hor_flip,
                         crop_size, crop_method)
        self.superpixel_dir = superpixel_dir
        self.cam_results = [None for _ in self.img_name_list]
        self.cam_eval_thres = cam_eval_thres

    def update_cam(self, idx, cam):
        self.cam_results[idx] = cam

    def generate_label(self, idx, img):
        fg_conf_cam, keys = self.cam_results[idx]
        pred = crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
        conf = keys[pred]

        return conf.astype(np.uint8)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))
        label = torch.from_numpy(self.label_list[idx])
        if self.cam_results[idx] is not None:
            seg_label = self.generate_label(idx, img)
        else:
            seg_label = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)

        if self.resize_long:
            img, seg_label = imutils.random_resize_long((img, seg_label), self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img, seg_label = imutils.random_scale((img, seg_label), scale_range=self.rescale, order=(3, 0))

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img, seg_label = imutils.random_lr_flip((img, seg_label))

        if self.crop_size:
            if self.crop_method == "random":
                img, seg_label = imutils.random_crop((img, seg_label), self.crop_size, (0, 0))
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)
                seg_label = imutils.top_left_crop(seg_label, self.crop_size, 0)

        if self.to_torch:
            img = imutils.HWC_to_CHW(img)

        return {'name': name_str, 'img': img, 'idx': idx, 'seg_label': seg_label, 'label': label}

    # def __getitem__(self, idx):
    #     out = super().__getitem__(idx)
    #     name = self.img_name_list[idx]
    #     name_str = decode_int_filename(name)
    #
    #     img = imageio.imread(get_img_path(name_str, self.voc12_root))
    #
    #     cls_label = torch.from_numpy(self.label_list[idx])
    #
    #     img = np.asarray(img)
    #
    #     if self.resize_long:
    #         img, seg_label = imutils.random_resize_long((img, seg_label), self.resize_long[0], self.resize_long[1])
    #
    #     if self.rescale:
    #         img, seg_label = imutils.random_scale((img, seg_label), scale_range=self.rescale, order=(3, 0))
    #
    #     if self.img_normal:
    #         img = self.img_normal(img)
    #
    #     if self.hor_flip:
    #         img, seg_label = imutils.random_lr_flip((img, seg_label))
    #
    #     if self.crop_method == "random":
    #         img, seg_label = imutils.random_crop((img, seg_label), self.crop_size, (0, 0))
    #     else:
    #         img = imutils.top_left_crop(img, self.crop_size, 0)
    #         seg_label = imutils.top_left_crop(seg_label, self.crop_size, 0)
    #
    #     img = imutils.HWC_to_CHW(img)
    #
    #     return {'name': name, 'img': img, 'cls_label': cls_label, 'seg_label': seg_label}


if __name__ == '__main__':
    for i in VOC12ClassificationDataset('./dev.txt',
                                        '../../data/raw/VOCdevkit/VOC2012/',
                                        resize_long=(320, 640),
                                        hor_flip=True, crop_size=512, crop_method="random"):
        print(i['img'].shape)
        break
    img_name_list_path = './dev.txt'
    voc12_root = '../../data/raw/VOCdevkit/VOC2012/'
    label_dir = '../../result/irnet/val/cv/felzenszwalb'
    hor_flip = True,
    crop_size = 512
    crop_method = "random",
    rescale = (0.5, 1.5)
    for i in VOC12PseudoSegmentationDataset(
            img_name_list_path, label_dir, crop_size, voc12_root,
            rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
            crop_method='random', resize_long=(320, 640)):
        print(i)
