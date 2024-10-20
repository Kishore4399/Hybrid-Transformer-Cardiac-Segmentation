import logging
import os
import glob
import random, math
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
#import cc3d
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _triple
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom
from skimage.exposure import match_histograms

from network.networks import *
from data.dataset import *

from monai.metrics import DiceMetric, HausdorffDistanceMetric

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

def set_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def makeTensor(numpy, device):
    return torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(numpy).to(device), 0), 0)

def attn_map(att_mat):
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)
    #att_mat, _ = torch.min(att_mat, dim=1)
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # Attention from the output token to the input space.
    return joint_attentions[-1]

def get_gaussian(patch_size, sigma_scale=1. / 8):
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def eval(model, device, path, name, patch_size, model_name, config, tta, resize, trained_size, output_dir):
    with torch.no_grad():
        model.eval()

        img, label, header  = preprocessingCardiac([path, name], need_header=True)
        # histtogram match for case19
        if "case19" in name:
            ref_path = path.replace("case19", "case1")
            ref_name = name.replace("case19", "case1")
            ref_img, ref_label, ref_header  = preprocessingCardiac([ref_path, ref_name], need_header=True)
            img = match_histograms(img, ref_img)
        if resize:
            img = zoom(img, tuple(trained_size/i for i in img.shape))
        input_img, label = makeTensor(img, device), makeTensor(label, device)
        #input_img = torch.cat((t1_img, t2_img, t1ce_img), dim=1)

        pred = torch.zeros(input_img.shape[0], 1, input_img.shape[2], input_img.shape[3], input_img.shape[4]).to(device)
        cnt = torch.zeros(input_img.shape[0], 1, input_img.shape[2], input_img.shape[3], input_img.shape[4]).to(device)
        size = input_img.shape[2:5]
        stride = [p // 2 for p in patch_size]
        importance_map = torch.Tensor(get_gaussian(patch_size)).to(device)
        sm = nn.Sigmoid()
        x = 0
        for i in range(1 + math.ceil((size[0] - patch_size[0]) / stride[0])):
            y = 0
            for j in range(1 + math.ceil((size[1] - patch_size[1]) / stride[1])):
                z = 0
                for k in range(1 + math.ceil((size[2] - patch_size[2]) / stride[2])):
                    input_patch = input_img[:,:,x: x + patch_size[0], y: y + patch_size[1], z: z + patch_size[2]]
                    # TTA
                    if tta:
                        flip_dims = []
                        img_patch = torch.zeros(8, input_patch.shape[1], input_patch.shape[2], input_patch.shape[3], input_patch.shape[4])
                        for n in range(8):
                            flip_dim = []
                            if n // 4 == 1:
                                flip_dim.append(2)
                            if (n % 4) // 2  == 1:
                                flip_dim.append(3)
                            if n % 4 % 2 == 1:
                                flip_dim.append(4)
                            flip_dims.append(flip_dim)
                            img_patch[n:n+1,:,:,:,:] = torch.flip(input_patch, flip_dim)
                        if 'HFTrans' in model_name:
                            segs, *_ = model(img_patch)
                        else:
                            segs= model(img_patch)
                        for n in range(8):
                            seg = torch.flip(segs[n:n+1,:,:,:,:], flip_dims[n])
                            pred[:,:,x: x + patch_size[0], y: y + patch_size[1], z: z + patch_size[2]] += seg #* importance_map
                            cnt[:,:,x: x + patch_size[0], y: y + patch_size[1], z: z + patch_size[2]] += 1
                    else:
                        if 'HFTrans' in model_name:
                            seg, *_ = model(input_patch)
                        else:
                            seg= model(input_patch)
                        pred[:,:,x: x + patch_size[0], y: y + patch_size[1], z: z + patch_size[2]] += seg #* importance_map
                        cnt[:,:,x: x + patch_size[0], y: y + patch_size[1], z: z + patch_size[2]] += 1
                    z += stride[2]
                    if z + patch_size[2] > size[2]:
                        z = size[2] - patch_size[2]
                y += stride[1]
                if y + patch_size[1] > size[1]:
                    y = size[1] - patch_size[1]
            x += stride[0]
            if x + patch_size[0] > size[0]:
                x = size[0] - patch_size[0]

        pred = pred / cnt
        pred = sm(pred)

        pred[pred >= 0.5] = 1.0
        pred[pred <= 0.5] = 0.0
        if resize:
            pred = zoom(pred.cpu().numpy(), (1,1,label.shape[2]/trained_size, label.shape[3]/trained_size, label.shape[4]/trained_size), order=0)
            pred = torch.FloatTensor(pred).to(device)

        diceMetric = DiceMetric(include_background=True, reduction="mean")
        HD95Metric = HausdorffDistanceMetric(percentile=95, include_background=True, reduction="mean")
        dice = diceMetric(y=label, y_pred=pred)
        HD95 = HD95Metric(y=label, y_pred=pred)
   
        pred = torch.squeeze(pred).detach().cpu().numpy()
     
        return dice.item(), HD95[0].item()

if __name__ == "__main__":
    # Version of Pytorch
    logging.info("Pytorch Version:%s" % torch.__version__)

    parser = argparse.ArgumentParser(description='BraTS')

    parser.add_argument('--dataset', type=str, required=True,
                        help='path of processed dataset')
    parser.add_argument('--patch-size', '--list', nargs='+', required=True, metavar='N',
                        help='3D patch-size x y z')
    parser.add_argument('--extention', type=str, required=True, metavar='N',
                        help='file extention format')
    parser.add_argument('--model', type=str, required=True, metavar='N',
                        help='model name')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 0)')

    parser.add_argument('--gpu', type=str, default='0', metavar='N',
                        help='Select the GPU (defualt 0)')
    parser.add_argument('--weight', type=str, default='./weights/mood_best_valid.pth',
                        help='trained weights of model for eval')
    
    parser.add_argument('--crossvalid', action='store_true',
                        help='Training using crossfold')
    parser.add_argument('--fold', type=int, default=1, metavar='N',
                        help='Valid fold num (1~5)')
    parser.add_argument('--vis', action='store_true',
                        help='Attention Map Visualization')
    parser.add_argument('--tta', action='store_true',
                        help='Test Time Augmentation')
    parser.add_argument('--resize', action='store_true',
                        help='resizing the image')
    parser.add_argument('--trained-size', type=int, default='0', metavar='N',
                        help='trained size of model)')
    parser.add_argument('--cc3d', action='store_true',
                        help='cc3d')

    args = parser.parse_args()
    args.patch_size = list(map(int, args.patch_size))
    fold = args.fold
    patch_size = args.patch_size
    batch_size = 1

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Torch use the device: %s" % device)
    weights = sorted(glob.glob(args.weight+"*.pth"))
    meanDices = []
    meanHD95s = []
    for i, weight in enumerate(weights):
        # Create model and check if we want to resume training
        if args.model == 'unet':
            config_vit = CONFIGS['BTS']
            model = SimpleUnet(num_channels=64, num_inputs=1, num_outputs=1).to(device)
        elif args.model == 'HFTrans':
            config_vit = CONFIGS['BTS']
            model = HFTrans(config_vit, img_size=patch_size, input_channels=1, num_classes=1, vis=False).to(device)
        elif args.model == 'HFTrans2':
            config_vit = CONFIGS['HFTrans2']
            model = HFTrans2(config_vit, img_size=patch_size, input_channels=2, num_classes=1, vis=False).to(device)
        else:
            print(f'invalid model name {args.model}')
            exit(0)
        model = nn.DataParallel(model)

        # Load the weight 
        if('latest_checkpoints_' in weight):
            checkpoint = torch.load(weight)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(torch.load(weight))

        # Start Training 
        
        #writer = SummaryWriter('runs2/eval/'+args.weight.split('/')[-1]+'/'+str(time.time()))

        dices = []
        hd95s = []

        set_seed(args.seed)

        output_dir = './eval/' + args.weight.split('/')[-1].split('.')[0]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        test_imgs = [d for d in os.listdir(os.path.join(args.dataset,'images')) if f"case{fold}_" not in d]
        test_imgs.sort()
        test_path = [[args.dataset, name] for name in test_imgs]

        for path, name in test_path:
            dice, hd95 = eval(model, device, path, name, patch_size, args.model, config_vit, args.tta, args.resize, args.trained_size, output_dir)
            dices.append(dice)
            hd95s.append(hd95)
        
        meanDice = sum(dices) / len(dices) * 100
        meanHD95 = sum(hd95s) / len(hd95s)
        meanDices.append(meanDice)
        meanHD95s.append(meanHD95)
        print(f'Round {i}  ########################################')
        print(args.weight)
        print(f'Average Dice Score: {sum(dices) / len(dices) * 100}')
        print(f'Average Hausdorff Distacne95: {sum(hd95s) / len(hd95s)}')
        print('########################################')
    
    result_df = pd.DataFrame(data=np.vstack([meanDices, meanHD95s]).T, columns=['mean Dice', 'mean HD95'])
    result_df.to_csv(f"MP_case{args.fold}.csv")
    print(f"mean Dice: {np.mean(meanDices)}, std Dice: {np.std(meanDices)}")
    print(f"mean HD95: {np.mean(meanHD95s)}, std HD95: {np.std(meanHD95s)}")

