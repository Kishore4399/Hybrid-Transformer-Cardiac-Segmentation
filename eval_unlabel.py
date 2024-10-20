import logging
import os
import time
import random, math
import argparse
from tqdm import tqdm

import numpy as np

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
    return torch.unsqueeze(torch.FloatTensor(numpy).to(device), 0)

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

def normalizeCardiac2ch(img, seg=False):
    #img = img_nii.get_fdata()
    if seg:
        return np.array(img, dtype = np.int32)
    else:
        img = np.array(img, dtype = np.float32)
        img = (img - np.min(img)) / np.max(img)

    return img

def preprocessingCardiac2ch(data_path):
    image_path = data_path[0]
    image_name = data_path[1]
    label_name = image_name.replace("ct.nrrd", "label.nrrd")

    img_ct, _  = nrrd.read(os.path.join(image_path, 'train', 'images',image_name))
    label, header  = nrrd.read(os.path.join(image_path, 'train', 'labels',label_name))
    #canonical_img = nib.as_closest_canonical(img)
    img_numpy = normalizeCardiac2ch(img_ct)
    seg_numpy = normalizeCardiac2ch(label, seg=True)
    
    stacked_img = np.stack([img_numpy, seg_numpy], axis=0)
    logging.info(f"input shape:{img_numpy.shape}, label shape:{seg_numpy.shape}")
    return stacked_img, seg_numpy, header

def eval(model, device, path, name, patch_size, model_name, config, tta, resize, trained_size, output_dir):
    with torch.no_grad():
        model.eval()

        img, label, header  = preprocessingCardiac2ch([path,name])
        if resize:
            img = zoom(img, (1, trained_size/img.shape[1], trained_size/img.shape[2], trained_size/img.shape[3]))
        input_img, label = makeTensor(img, device), torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(label).to(device), 0), 0)

        pred = torch.zeros(input_img.shape[0], 1, input_img.shape[2], input_img.shape[3], input_img.shape[4]).to(device)
        cnt = torch.zeros(input_img.shape[0], 1, input_img.shape[2], input_img.shape[3], input_img.shape[4]).to(device)
        size = input_img.shape[2:5]
        stride = [p // 2 for p in patch_size]
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
                            seg, *_ = model(img_patch)
                        else:
                            seg= model(img_patch)
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
        pred = torch.squeeze(pred)
        nrrd.write(os.path.join(output_dir,name.replace("ct.nrrd", "pred.nrrd")), np.array(pred.to('cpu')), header=header)
        
        #logging.info(f"Dice Score of EN {dice_EN.item() * 100:.4f}, Dice Score of TC: {dice_TC.item() * 100:.4f}, Dice Score of WT: {dice_WT.item() * 100:.4f}")
        #logging.info(f"HD95 of EN {hd95_EN:.4f}, HD95 of TC: {hd95_TC:.4f}, HD95 of WT: {hd95_WT:.4f}")


        return 0

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
    if('latest_checkpoints_' in args.weight):
        checkpoint = torch.load(args.weight)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(args.weight))

    # Start Training 
    
    #writer = SummaryWriter('runs2/eval/'+args.weight.split('/')[-1]+'/'+str(time.time()))

    dices = []
    hd95s = []

    set_seed(args.seed)

    output_dir = './eval/' + args.weight.split('/')[-1].split('.')[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    test_imgs = [d for d in os.listdir("/home/shqin/cardiac_seg_HFTrans2/dataset/stageOne/train/images")]
    test_imgs.sort()
    test_path = [[args.dataset, name] for name in test_imgs]

    with tqdm(total=len(test_path), desc="generating....") as pbar:
        for path, name in test_path:
            eval(model, device, path, name, patch_size, args.model, config_vit, args.tta, args.resize, args.trained_size, output_dir)
            pbar.update(1)
    # print('########################################')
    # print(args.dataset)
    # print(args.weight)
    # print(f'Average Dice Score: {sum(dices) / len(dices) * 100}')
    # print(f'Average Hausdorff Distacne95: {sum(hd95s) / len(hd95s)}')
    # print('########################################')

