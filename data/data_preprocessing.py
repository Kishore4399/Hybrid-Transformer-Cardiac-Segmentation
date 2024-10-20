import os
import math
import time
import numpy as np
import nibabel as nib
import h5py
import nrrd
import torch
import torchio as tio
import torch.nn.functional as F
from torchio import RandomElasticDeformation

import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

def normalizeCardiac(img, seg=False):
    #img = img_nii.get_fdata()
    if seg:
        return np.array(img, dtype = np.int32)
    else:
        img = np.array(img, dtype = np.float32)
        img = (img - np.min(img)) / (np.max(img)-np.min(img))

    return img

def normalizeCardiac2ch(img, seg=False):
    #img = img_nii.get_fdata()
    if seg:
        return np.array(img, dtype = np.int32)
    else:
        img = np.array(img, dtype = np.float32)
        for i in range(img.shape[0]):
            img[i,:] = (img[i,:] - np.min(img[i,:])) / (np.max(img[i,:]) - np.min(img[i,:]))

    return img

def preprocessingCardiac(data_path, need_header=False):
    image_path = data_path[0]
    image_name = data_path[1]
    label_name = image_name.replace("ct.nrrd", "label.nrrd")

    img_ct, _  = nrrd.read(os.path.join(image_path, 'images',image_name))
    label, label_header  = nrrd.read(os.path.join(image_path, 'labels',label_name))
    #canonical_img = nib.as_closest_canonical(img)
    img_numpy = normalizeCardiac(img_ct)
    seg_numpy = normalizeCardiac(label, seg=True)
    logging.info(f"input shape:{img_numpy.shape}, label shape:{seg_numpy.shape}")
    if not need_header:
        return img_numpy, seg_numpy
    else:
        return img_numpy, seg_numpy, label_header

def preprocessingCardiac2ch(data_path):
    image_path = data_path[0]
    image_name = data_path[1]
    label_name = image_name.replace("ct.nrrd", "label.nrrd")

    img_ct, _  = nrrd.read(os.path.join(image_path, 'images',image_name))
    label, _  = nrrd.read(os.path.join(image_path, 'labels',label_name))
    #canonical_img = nib.as_closest_canonical(img)
    img_numpy = normalizeCardiac2ch(img_ct)
    seg_numpy = normalizeCardiac2ch(label, seg=True)
    logging.info(f"input shape:{img_numpy.shape}, label shape:{seg_numpy.shape}")
    return img_numpy, seg_numpy

def preprocessingCardiac2ch_pred(data_path):
    image_path = data_path[0]
    image_name = data_path[1]

    img_ct, header  = nrrd.read(os.path.join(image_path, 'images',image_name))
    #canonical_img = nib.as_closest_canonical(img)
    img_numpy = normalizeCardiac2ch(img_ct)
    logging.info(f"input shape:{img_numpy.shape}")
    return img_numpy, header

def preprocessingCardiac2ch_eval(data_path):
    image_path = data_path[0]
    image_name = data_path[1]
    label_name = image_name.replace("ct.nrrd", "label.nrrd")

    img_ct, _  = nrrd.read(os.path.join(image_path, 'images',image_name))
    label, header  = nrrd.read(os.path.join(image_path, 'labels',label_name))
    #canonical_img = nib.as_closest_canonical(img)
    img_numpy = normalizeCardiac2ch(img_ct)
    seg_numpy = normalizeCardiac2ch(label, seg=True)
    logging.info(f"input shape:{img_numpy.shape}, label shape:{seg_numpy.shape}")
    return img_numpy, seg_numpy, header

def data_augmentaion(img_numpy, seg_numpy):    
    ##Elastic Deformation, probability of 0.3
    '''
    if np.random.rand() >= 0.7:
        concat = np.concatenate((t1_numpy, t2_numpy, t1ce_numpy, flair_numpy, seg_numpy), axis=0)
        t1_numpy[0,:,:,:], t2_numpy[0,:,:,:], t1ce_numpy[0,:,:,:], flair_numpy[0,:,:,:], seg_numpy[0,:,:,:] = random_elastic_deformation(concat)
    '''
    ##Augmentation(scale, rotation, translation), probability of 0.2
    rot = [0,0,0]
    if np.random.rand() <= 0.2:
        rot = []
        for i in range (3):
            rot.append(np.random.uniform(-30, 30))

    scale = [1, 1, 1]
    if np.random.rand() <= 0.2:
        scale = []
        for i in range (3):
            scale.append(np.random.uniform(0.7, 1.4))
    trans = [0,0,0]
    img_numpy = random_affine(img_numpy, scale, rot, trans, 0)
    seg_numpy = random_affine(seg_numpy, scale, rot, trans, 1)
    
    #Gaussian noise
    if np.random.rand() <= 0.15:
        img_numpy = random_noise(img_numpy)

    #Gaussian blur
    if np.random.rand() <= 0.1:
        img_numpy = random_blur(img_numpy)

    #Brightness
    if np.random.rand() <= 0.15:
        img_numpy = random_brightness(img_numpy)
    
    #Contrast
    if np.random.rand() <= 0.15:
        img_numpy = random_contrast(img_numpy)

    '''
    ##Random Jitter, probability of 0.2
    if np.random.rand() >= 0.8:
        t1_numpy = random_jitter(t1_numpy)
        t2_numpy = random_jitter(t2_numpy)
        t1ce_numpy = random_jitter(t1ce_numpy)
        flair_numpy = random_jitter(flair_numpy)
    '''
    ##Gamma augmentation, probability of 0.3
    if np.random.rand() <= 0.15:
        img_numpy = random_gamma(img_numpy)


    #Flip
    if np.random.rand() <= 0.5:
        img_numpy = random_flip(img_numpy, axes=0)
        seg_numpy = random_flip(seg_numpy, axes=0)
    if np.random.rand() <= 0.5:
        img_numpy = random_flip(img_numpy, axes=1)
        seg_numpy = random_flip(seg_numpy, axes=1)
    if np.random.rand() <= 0.5:
        img_numpy = random_flip(img_numpy, axes=2)
        seg_numpy = random_flip(seg_numpy, axes=2)

    return img_numpy, seg_numpy

def data_augmentaion2ch(img_numpy, seg_numpy):

    img_numpy_1ch = img_numpy[0:1,:]
    img_numpy_2ch = img_numpy[1:2,:]
    
    ##Elastic Deformation, probability of 0.3
    '''
    if np.random.rand() >= 0.7:
        concat = np.concatenate((t1_numpy, t2_numpy, t1ce_numpy, flair_numpy, seg_numpy), axis=0)
        t1_numpy[0,:,:,:], t2_numpy[0,:,:,:], t1ce_numpy[0,:,:,:], flair_numpy[0,:,:,:], seg_numpy[0,:,:,:] = random_elastic_deformation(concat)
    '''
    ##Augmentation(scale, rotation, translation), probability of 0.2
    rot = [0,0,0]
    if np.random.rand() <= 0.2:
        rot = []
        for i in range (3):
            rot.append(np.random.uniform(-30, 30))

    scale = [1, 1, 1]
    if np.random.rand() <= 0.2:
        scale = []
        for i in range (3):
            scale.append(np.random.uniform(0.7, 1.4))
    trans = [0,0,0]
    img_numpy_1ch = random_affine(img_numpy_1ch, scale, rot, trans, 0)
    img_numpy_2ch = random_affine(img_numpy_2ch, scale, rot, trans, 1)
    seg_numpy = random_affine(seg_numpy, scale, rot, trans, 1)
    
    #Gaussian noise
    if np.random.rand() <= 0.15:
        img_numpy_1ch = random_noise(img_numpy_1ch)

    #Gaussian blur
    if np.random.rand() <= 0.1:
        img_numpy_1ch = random_blur(img_numpy_1ch)

    #Brightness
    if np.random.rand() <= 0.15:
        img_numpy_1ch = random_brightness(img_numpy_1ch)
    
    #Contrast
    if np.random.rand() <= 0.15:
        img_numpy_1ch = random_contrast(img_numpy_1ch)

    '''
    ##Random Jitter, probability of 0.2
    if np.random.rand() >= 0.8:
        t1_numpy = random_jitter(t1_numpy)
        t2_numpy = random_jitter(t2_numpy)
        t1ce_numpy = random_jitter(t1ce_numpy)
        flair_numpy = random_jitter(flair_numpy)
    '''
    ##Gamma augmentation, probability of 0.3
    if np.random.rand() <= 0.15:
        img_numpy_1ch = random_gamma(img_numpy_1ch)


    #Flip
    if np.random.rand() <= 0.5:
        img_numpy_1ch = random_flip(img_numpy_1ch, axes=0)
        img_numpy_2ch = random_flip(img_numpy_2ch, axes=0)
        seg_numpy = random_flip(seg_numpy, axes=0)
    if np.random.rand() <= 0.5:
        img_numpy_1ch = random_flip(img_numpy_1ch, axes=1)
        img_numpy_2ch = random_flip(img_numpy_2ch, axes=1)
        seg_numpy = random_flip(seg_numpy, axes=1)
    if np.random.rand() <= 0.5:
        img_numpy_1ch = random_flip(img_numpy_1ch, axes=2)
        img_numpy_2ch = random_flip(img_numpy_2ch, axes=2)
        seg_numpy = random_flip(seg_numpy, axes=2)

    return np.concatenate((img_numpy_1ch, img_numpy_2ch), 0), seg_numpy

def random_affine(img, scale, rot, trans, label = 0): #Scale+Rotation+Translation at once
    mode = 'nearest' if label else 'linear'
    transform = tio.RandomAffine(
        scales = (scale[0], scale[0], scale[1], scale[1], scale[2], scale[2]), #scale (scale1, scale2) for each axis
        degrees = (rot[0], rot[0], rot[1], rot[1], rot[2], rot[2]), #rotate (rot1, rot2) for each axis
        translation = (trans[0], trans[0], trans[1], trans[1], trans[2], trans[2]), #translate (trans1, trans2) for each axis in mm unit
        image_interpolation = mode,
    )
    return transform(img)

def random_flip(img, axes=(0,1,2), p=1.0):
    transform = tio.RandomFlip(axes = axes, flip_probability = p) #Index or tuple of indices of the spatial dimensions along which the image might be flipped. If they are integers, they must be in (0, 1, 2).
    return transform(img)

def random_blur(img, std = (0.5,1.5)):
    transform = tio.RandomBlur(std = std) #If two values (a,b) are provided, then Gaussian kernels used to blur the image along each axis, where sigma ~ U(a,b) 
    img = transform(img)
    return img

def random_noise(img, mean = 0, std = (0, 0.1)):
    transform = tio.RandomNoise(mean = mean, std = std) # Mean of the Gaussian distribution from which the noise is sampled. Standard Deviation of the Gaussian distribution from which the noise is sampled.
    img = transform(img)
    return img

def random_gamma(img, gamma = (-0.35, 0.4)): #Randomly change contrast of an image by raising its values to the power gamma
    transform = tio.RandomGamma(log_gamma = gamma) #Tuple (a,b) to compute the exponent gamma = e^(beta), where beta ~ U(a,b)
    img = transform(img)
    return img

def random_brightness(img):
    img = img * np.random.uniform(0.7,1.3)
    return img

def random_contrast(img):
    max_intensity = np.max(img)
    min_intensity = np.min(img)
    img = img * np.random.uniform(0.65,1.5)
    img = np.clip(img, min_intensity, max_intensity)
    return img