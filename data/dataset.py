import os

from torch.utils.data import Dataset
import numpy as np
from utils.utils import *
from .data_preprocessing import *

class CardiacDataset2ch(Dataset):
    """Cardiac Dataset""" 
    def __init__(self,
                 dataset_path,
                 patch_size,
                 subset='train',
                 crossvalid= False,
                 valid_fold=1,
                 pretask=False,
                 ):

        self.idx = 1
        self.patch_size = patch_size
        self.hdf5 = 'hdf5' in dataset_path
        self.subset = subset
        self.save_to_mem = True
        if subset == 'test':
            subset = 'valid'
            self.save_to_mem = False
        self.crossvalid = crossvalid
        self.pretask=pretask
        if self.crossvalid:
            if subset == 'train':
                img_names = [d for d in os.listdir(os.path.join(dataset_path,'images')) if not f"case{valid_fold}_ct" in d]
                img_names.sort()
                self.data_path = [[dataset_path, name] for name in img_names]

            elif subset == 'valid' or subset == 'test':
                img_names = [d for d in os.listdir(os.path.join(dataset_path,'images')) if f"case{valid_fold}" in d]
                img_names.sort()
                self.data_path = [[dataset_path, name] for name in img_names]
        else:
            self.dataset_path = os.path.join(dataset_path, subset)
            img_names = [d for d in os.listdir(os.path.join(self.dataset_path,'images')) if d.startswith('case')]
            img_names.sort()
            
            # N x 2
            self.data_path = [[self.dataset_path, name] for name in img_names]
        
        if self.save_to_mem:
            self.img = np.zeros((len(self)), object)
            self.label = np.zeros((len(self)), object)

            for n in range(len(self)):
                img, label = preprocessingCardiac2ch(self.data_path[n])
                self.img[n] = img
                self.label[n] = label

        
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        if self.save_to_mem:
            img, label = self.img[idx], self.label[idx]
        else:
            img, label = preprocessingCardiac2ch(self.data_path[idx])

        if self.subset == 'test':
            return test_img2ch(img, label)
        else:
            img_patch, label_patch = extract_patch2ch(img, label, self.patch_size)
            if self.subset == 'train':
                img_patch, label_patch = data_augmentaion2ch(img_patch, label_patch)
            return torch.FloatTensor(img_patch), torch.FloatTensor(label_patch)
        

class CardiacDataset(Dataset):
    """Cardiac Dataset""" 
    def __init__(self,
                 dataset_path,
                 patch_size,
                 subset='train',
                 crossvalid= False,
                 valid_fold=1,
                 pretask=False,
                 ):

        self.idx = 1
        self.patch_size = patch_size
        self.hdf5 = 'hdf5' in dataset_path
        self.subset = subset
        self.save_to_mem = True
        if subset == 'test':
            subset = 'valid'
            self.save_to_mem = False
        self.crossvalid = crossvalid
        self.pretask=pretask
        
        if self.crossvalid:
            if subset == 'train':
                img_names = [d for d in os.listdir(os.path.join(dataset_path,'images')) if not f"case{valid_fold}_ct" in d]
                img_names.sort()
                self.data_path = [[dataset_path, name] for name in img_names]
            elif subset == 'valid' or subset == 'test':
                img_names = [d for d in os.listdir(os.path.join(dataset_path,'images')) if f"case{valid_fold}_ct" in d]
                img_names.sort()
                self.data_path = [[dataset_path, name] for name in img_names]
        else:
            self.dataset_path = os.path.join(dataset_path, subset)
            img_names = [d for d in os.listdir(os.path.join(self.dataset_path,'images')) if d.startswith('case')]
            img_names.sort()
            # N x 2
            self.data_path = [[self.dataset_path, name] for name in img_names]
        
        if self.save_to_mem:
            self.img = np.zeros((len(self)), object)
            self.label = np.zeros((len(self)), object)

            for n in range(len(self)):
                img, label = preprocessingCardiac(self.data_path[n])
                self.img[n] = img
                self.label[n] = label

        
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        if self.save_to_mem:
            img, label = self.img[idx], self.label[idx]
        else:
            img, label = preprocessingCardiac(self.data_path[idx])

        if self.subset == 'test':
            return test_img2ch(img, label)
        else:
            img_patch, label_patch = extract_patch(img, label, self.patch_size)
            if self.subset == 'train':
                img_patch, label_patch = data_augmentaion(img_patch, label_patch)
            return torch.FloatTensor(img_patch), torch.FloatTensor(label_patch)