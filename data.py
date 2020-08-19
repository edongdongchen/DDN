import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset

"""
Dataloader used in supre-resolution (SR) examples (training/testing) in the paper
@inproceedings{chen2020decomposition,
	author = {Chen, Dongdong and Davies, Mike E},
	title = {Deep Decomposition Learning for Inverse Imaging Problems},
	booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
	year = {2020}
}
"""

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def CVDB(dataset_name='Set1', batch_size=5, shuffle=True, crop_size=(64, 64)):#, rand_crops=False):
    imgs_path = os.path.join('dataset', dataset_name)
    transform_data = transforms.Compose([transforms.ToTensor()])

    def transform_center_crop(img):
        if crop_size is not None:
            crop = transforms.CenterCrop(crop_size)
            x_pil = crop(img)
        else:
            factor = 1
            (w, h) = img.size
            (h, w) = (np.floor(factor * h), np.floor(factor * w))
            crop = transforms.CenterCrop((h - np.mod(h, 32), w - np.mod(w, 32)))
            x_pil = crop(img)

        ar = np.array(x_pil)

        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]
        x = rgb2ycbcr(ar.transpose(1, 2, 0), only_y=False)
        x = transform_data(x)
        return x  # (HR)

    dataset = datasets.ImageFolder(imgs_path,
                                   transform=transform_center_crop,
                                   target_transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return dataloader


def CVDB_Y(dataset_name='BSDS300', batch_size=5, shuffle=True, crop_size=(64, 64)):
    imgs_path = os.path.join('dataset', dataset_name)

    def transform_random_10_crops(img):
        (w, h) = img.size
        crops_torch = []
        for i in range(5):
            if i == 0:
                crop_pil = transforms.CenterCrop(crop_size)(img)
            else:
                cx = np.random.randint(0, w - crop_size[0] - 1, 1)
                cy = np.random.randint(0, h - crop_size[1] - 1, 1)
                crop_pil = transforms.functional.crop(img, cx[0], cy[0], crop_size[0], crop_size[1])

            crop_np = np.array(crop_pil)
            if len(crop_np.shape) == 3:
                crop_np = crop_np.transpose(2, 0, 1)
            else:
                crop_np = crop_np[None, ...]
            crop_ycbcr = rgb2ycbcr(crop_np.transpose(1, 2, 0), only_y=True)
            crop_torch = transforms.ToTensor()(crop_ycbcr)
            crops_torch.append(crop_torch)
        img_crops = torch.stack([crop_torch for crop_torch in crops_torch])
        return img_crops

    dataset = datasets.ImageFolder(imgs_path,
                                   transform=transform_random_10_crops,
                                   target_transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return dataloader