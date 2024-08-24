# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import numpy as np
import torch
import torch.nn as nn


def strong_transform(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target


def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks(labels, target_depths, pseudo_labels, iter):
    class_masks = []
    for i in range(len(labels)):
        label = labels[i]
        target_depth = target_depths[i]
        pseudo_label = pseudo_labels[i]
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        if iter > 120000:
            class_masks.append(generate_class_mask_filter(label, classes, target_depth, pseudo_label).unsqueeze(0))
        else:
            class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks

def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def generate_class_mask_filter(label, classes, target_depth, pseudo_label):
    # print(torch.unique(label))
    # print(classes.unsqueeze(1).unsqueeze(2).shape)
    # print(torch.unique(pseudo_label))
    pseudo_label = pseudo_label.unsqueeze(0)
    #print(classes)
    clses = classes
    pseudo_label, clses = torch.broadcast_tensors(pseudo_label,
                                                  clses.unsqueeze(1).unsqueeze(2))
    filtered_classes = []
    fc = classes
    filtered_label = label

    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))

    cls = []
    # Calculate the number of pixels of a class appear at a certain depth in the target domain
    c1, c2, c3, c4 = 0, 1, 10, 100
    td1 = (target_depth > c1) & (target_depth <= c2)
    td2 = (target_depth > c2) & (target_depth <= c3)
    td3 = (target_depth > c3) & (target_depth <= c4)
    td4 = (target_depth > c4)

    for i in range(len(classes)):
        m = pseudo_label[i].eq(clses[i])
        msum = m.sum()
        #print(m.sum()) # for this certain class to be paste: the number of pixels in the whole picture
        classi_at_number_td1 = td1 & m
        classi_at_number_td1 = classi_at_number_td1.sum()
        classi_at_number_td2 = td2 & m
        classi_at_number_td2 = classi_at_number_td2.sum()
        classi_at_number_td3 = td3 & m
        classi_at_number_td3 = classi_at_number_td3.sum()
        classi_at_number_td4 = td4 & m
        classi_at_number_td4 = classi_at_number_td4.sum()
        # print(classi_at_number_td1)
        # print(classi_at_number_td2)
        # print(classi_at_number_td3)
        # print(classi_at_number_td4)
        # print('-----------')

        source_m = label[i].eq(classes[i])
        #print(source_m.sum())
        classi_pasted_at_number_td1 = td1 & source_m
        classi_pasted_at_number_td1 = classi_pasted_at_number_td1.sum()
        classi_pasted_at_number_td2 = td2 & source_m
        classi_pasted_at_number_td2 = classi_pasted_at_number_td2.sum()
        classi_pasted_at_number_td3 = td3 & source_m
        classi_pasted_at_number_td3 = classi_pasted_at_number_td3.sum()
        classi_pasted_at_number_td4 = td4 & source_m
        classi_pasted_at_number_td4 = classi_pasted_at_number_td4.sum()
        # print(classi_pasted_at_number_td1)
        # print(classi_pasted_at_number_td2)
        # print(classi_pasted_at_number_td3)
        # print(classi_pasted_at_number_td4)
        # Calculate the difference at each interval
        mn = classi_pasted_at_number_td1
        t = '1'
        if classi_pasted_at_number_td2 > classi_pasted_at_number_td1:
            mn = classi_pasted_at_number_td2
            t = '2'
        if classi_pasted_at_number_td3 > classi_pasted_at_number_td2:
            mn = classi_pasted_at_number_td3
            t = '3'
        if classi_pasted_at_number_td4 > classi_pasted_at_number_td3:
            mn = classi_pasted_at_number_td4
            t = '4'
        #print(mn)
        interval = []
        if t in 'classi_at_number_td1':
            diff = abs(classi_pasted_at_number_td1-classi_at_number_td1)
            interval = classi_pasted_at_number_td1
            interval_t = classi_at_number_td1
            #print(diff)
        if t in 'classi_at_number_td2':
            diff = abs(classi_pasted_at_number_td2-classi_at_number_td2)
            interval = classi_pasted_at_number_td2
            interval_t = classi_at_number_td2
            #print(diff)
        if t in 'classi_at_number_td3':
            diff = abs(classi_pasted_at_number_td3-classi_at_number_td3)
            interval = classi_pasted_at_number_td3
            interval_t = classi_at_number_td3
            #print(diff)
        if t in 'classi_at_number_td4':
            diff = abs(classi_pasted_at_number_td4-classi_at_number_td4)
            interval = classi_pasted_at_number_td4
            interval_t = classi_at_number_td4
            #print(diff)
        xx = 10
        if interval < 10000:
            if diff < 5000:
                filtered_classes.append(fc[i].item())
            elif msum < 3000:
                filtered_classes.append(fc[i].item())
            else:
                xx=10
        else:
            if interval_t > 5000:
                filtered_classes.append(fc[i].item())
            elif msum < 5000:
                filtered_classes.append(fc[i].item())
            else:
                xx=10
    filtered_classes = torch.tensor(filtered_classes).cuda()

        #print('---------')
        # diff1 = abs(classi_pasted_at_number_td1 - classi_at_number_td1)
        # diff2 = abs(classi_pasted_at_number_td2 - classi_at_number_td2)
        # diff3 = abs(classi_pasted_at_number_td3 - classi_at_number_td3)
        # diff4 = abs(classi_pasted_at_number_td4 - classi_at_number_td4)
        # print(diff1)
        # print(diff2)
        # print(diff3)
        # print(diff4)
    #print(filtered_classes)
    filtered_label, filtered_classes = torch.broadcast_tensors(filtered_label,
                                             filtered_classes.unsqueeze(1).unsqueeze(2))
    class_mask = filtered_label.eq(filtered_classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target
