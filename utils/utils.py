import math
import random

import torch
import numpy as np
from medpy.metric.binary import hd95
from torch.optim.lr_scheduler import _LRScheduler

def extract_patch(image, label, patch_size):
    while True:
        patch_idx = [random.randint(0, img - patch) for img, patch in zip(image.shape, patch_size)]
    
        img_patch = image[patch_idx[0]:patch_idx[0] + patch_size[0], patch_idx[1]:patch_idx[1] + patch_size[1], patch_idx[2]:patch_idx[2] + patch_size[2]]
        label_patch = label[patch_idx[0]:patch_idx[0] + patch_size[0], patch_idx[1]:patch_idx[1] + patch_size[1], patch_idx[2]:patch_idx[2] + patch_size[2]]
        if np.random.rand() <= 1/2:
            break
        elif np.count_nonzero(label_patch) > 0:
            break
    if len(img_patch.shape) == 3:
        img_patch = np.expand_dims(img_patch, axis=0)
    if len(label_patch.shape) == 3:
        label_patch = np.expand_dims(label_patch, axis=0)

    return img_patch, label_patch

def extract_patch2ch(image, label, patch_size):
    while True:
        patch_idx = [random.randint(0, img - patch) for img, patch in zip(image.shape[-3:], patch_size)]
    
        img_patch = image[...,patch_idx[0]:patch_idx[0] + patch_size[0], patch_idx[1]:patch_idx[1] + patch_size[1], patch_idx[2]:patch_idx[2] + patch_size[2]]
        label_patch = label[...,patch_idx[0]:patch_idx[0] + patch_size[0], patch_idx[1]:patch_idx[1] + patch_size[1], patch_idx[2]:patch_idx[2] + patch_size[2]]
        if np.random.rand() <= 1/2:
            break
        elif np.count_nonzero(label_patch) > 0:
            break
    if len(img_patch.shape) == 3:
        img_patch = np.expand_dims(img_patch, axis=0)
    if len(label_patch.shape) == 3:
        label_patch = np.expand_dims(label_patch, axis=0)

    return img_patch, label_patch

def test_img2ch(img, label):
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    if len(label.shape) == 3:
        label = np.expand_dims(label, axis=0)
    return torch.FloatTensor(img), torch.FloatTensor(label)

def diceCoeff(preds, gts, labels, loss=False):
    '''
    pred = torch.zeros_like(gts)
    gt = torch.zeros_like(gts)
    for label in labels:
        if loss:
            pred = torch.add(pred, preds[:,label,:,:,:])
        else:
            pred[preds == label] = 1
        gt[gts == label] = 1
    if (torch.sum(gt) == 0.0):
        gt = 1 - gt
        pred = 1 - pred
        return 2 * torch.sum(gt * pred) / (torch.sum(pred) + torch.sum(gt) + 1e-6 )
    else:
    '''

    if (torch.sum(gts) == 0.0):
        gts = 1 - gts
        preds = 1 - preds
        return 2 * torch.sum(gts * preds) / (torch.sum(preds) + torch.sum(gts) + 1e-6 )
    else:
        return 2 * torch.sum(gts * preds) / (torch.sum(preds) + torch.sum(gts) + 1e-6 )