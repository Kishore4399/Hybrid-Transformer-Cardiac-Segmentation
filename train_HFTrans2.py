import logging
import os
import time
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import zoom

from tensorboardX import SummaryWriter

from network.networks import *
from data.dataset import *

# from cardiac_seg.network.networks import SimpleUnet, HFTrans

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

def train(model, device, img_patch, label_patch, optimizer, writer, network):
    model.train()
    optimizer.zero_grad()

    img_patch, label_patch = img_patch.to(device), label_patch.to(device)
    input_patch = img_patch
    #input_patch = torch.cat((t1_patch, t2_patch, t1ce_patch), dim=1)
    if 'HFTrans' in network:
        seg, *_ = model(input_patch)
    else:
        seg = model(input_patch)

    ce = nn.BCEWithLogitsLoss()
    ce_loss = ce(seg, label_patch[:,0:1,...])

    sig = nn.Sigmoid()
    seg_sig = sig(seg.clone())
    dice = diceCoeff(seg_sig, label_patch[:,0:1,...], [1], loss=True)
    '''
    # laplacian kernel
    weights = torch.tensor([[[0., 0., 0.],
                            [0., -1., 0.],
                            [0., 0., 0.]],
                            [[0., -1., 0.],
                            [-1., 6., -1.],
                            [0., -1., 0.]],
                            [[0., 0., 0.],
                            [0., -1., 0.],
                            [0., 0., 0.]]])
    weights = weights.view(1,1,3,3,3).to(device)
    weights = weights / 2
    laplacian = F.conv3d(seg_sig, weights, padding=1).pow(2).sum().sqrt() / seg_sig.shape[0]

    # total variation
    # B x C x X x Y x Z (B x 1 x 160 x 160 x 160)
    # idx: 0 ~ 159, value: 0 ~ 1 (due to sigmoid)
    # seg[1:] 1~159 
    # seg[:-1] 0~158
    tv_x = (seg_sig[:,:,1:,:-1,:-1] - seg_sig[:,:,:-1,:-1,:-1]).pow(2)
    tv_y = (seg_sig[:,:,:-1,1:,:-1] - seg_sig[:,:,:-1,:-1,:-1]).pow(2)
    tv_z = (seg_sig[:,:,:-1,:-1,1:] - seg_sig[:,:,:-1,:-1,:-1]).pow(2) 
    tv = tv_x + tv_y + tv_z
    tv = (tv + 1e-8).sqrt().sum() / seg_sig.shape[0]
    '''
    total_loss = 1.0 * ce_loss + (1 - dice) #+ 1e-6 * laplacian # 1e-9 * tv #
    #total_loss = (1 - dice_EN) + (1 - dice_TC) + (1 - dice_WT)
    #logging.info(f"Train total Loss: {total_loss.item():.6f}, Cross Entrophy Loss: {ce_loss.item():.6f}, Dice Score: {dice.item() * 100:.4f}, laplacian: {laplacian.item():.4f}, tv: {tv.item():.4f}")
    logging.info(f"Train total Loss: {total_loss.item():.6f}, Cross Entrophy Loss: {ce_loss.item():.6f}, Dice Score: {dice.item() * 100:.4f}")

    # optimize the parameters
    total_loss.backward()
    optimizer.step()
    #scheduler.step()
    return total_loss.item()

def valid(model, device, image_patch, label_patch, network, cnt):
    with torch.no_grad():
        model.eval()

        image_patch, label_patch = image_patch.to(device), label_patch.to(device)
        input_patch = image_patch
        #input_patch = torch.cat((t1_patch, t2_patch, t1ce_patch), dim=1)

        if 'HFTrans' in network:
            seg, *_ = model(input_patch)
        else:
            seg = model(input_patch)
        ce = nn.BCEWithLogitsLoss()
        ce_loss = ce(seg, label_patch[:,0:1,...])

        sig = nn.Sigmoid()
        seg_sig = sig(seg.clone())

        dice = diceCoeff(seg_sig, label_patch[:,0:1,...], [1], loss=True)

        # laplacian kernel
        weights = torch.tensor([[[0., 0., 0.],
                                [0., -1., 0.],
                                [0., 0., 0.]],
                                [[0., -1., 0.],
                                [-1., 6., -1.],
                                [0., -1., 0.]],
                                [[0., 0., 0.],
                                [0., -1., 0.],
                                [0., 0., 0.]]])
        weights = weights.view(1,1,3,3,3).to(device)
        laplacian = F.conv3d(seg_sig, weights, padding=1).pow(2).sum().sqrt() / img_patch.shape[0]

        # total variation
        tv_x = (seg_sig[:,:,1:,:,:] - seg_sig[:,:,:-1,:,:]).pow(2).sum().sqrt() / img_patch.shape[0]
        tv_y = (seg_sig[:,:,:,1:,:] - seg_sig[:,:,:,:-1,:]).pow(2).sum().sqrt() / img_patch.shape[0]
        tv_z = (seg_sig[:,:,:,:,1:] - seg_sig[:,:,:,:,:-1]).pow(2).sum().sqrt() / img_patch.shape[0]
        tv = tv_x + tv_y + tv_z

        total_loss =  (1 - dice)
        logging.info(f"Valid Total Loss: {total_loss.item():.6f}, Cross Entrophy Loss: {ce_loss.item():.6f}, Dice Score: {dice.item() * 100:.4f}, laplacian: {laplacian.item():.4f}, tv: {tv.item():.4f}")
        # tensorboard.
        writer.add_image('valid/ct_patch', image_patch[0,0:1,:,:,image_patch.shape[4] // 2], cnt)
        writer.add_image('valid/pred_patch', seg_sig[0,0:1,:,:,image_patch.shape[4] // 2], cnt)
        writer.add_image('valid/label_patch', label_patch[0,0:1,:,:,image_patch.shape[4] // 2], cnt)
        writer.add_scalar('valid/dice', dice.item() * 100, cnt)
        writer.add_scalar('valid/laplacian', laplacian.item() * 100, cnt)

        return total_loss.item()

if __name__ == "__main__":
    # Version of Pytorch
    logging.info("Pytorch Version:%s" % torch.__version__)

    parser = argparse.ArgumentParser(description='BraTS')

    parser.add_argument('--dataset', type=str, required=True,
                        help='path of processed dataset')
    parser.add_argument('--batch-size', type=int, required=True, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--patch-size', '--list', nargs='+', required=True, metavar='N',
                        help='3D patch-size x y z')
    parser.add_argument('--epoches', type=int, required=True, metavar='N',
                        help='number of epoches to train (default: 1000)')
    parser.add_argument('--identifier', type=str, required=True, metavar='N',
                        help='Select the identifier for file name')
    parser.add_argument('--extention', type=str, required=True, metavar='N',
                        help='file extention format')
    parser.add_argument('--model', type=str, required=True, metavar='N',
                        help='model name')

    parser.add_argument('--weights', type=str, default='./weights',
                        help='path of training weight')
    parser.add_argument('--pre_path', type=str, default='./weights',
                        help='path of training weight')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints',
                        help='path of training snapshot')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 0)')

    parser.add_argument('--gpu', type=str, default='0', metavar='N',
                        help='Select the GPU (defualt 0)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='number of iterations to log (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight', type=str, default='./weights/mood_best_valid.pth',
                        help='trained weights of model for resume')

    parser.add_argument('--crossvalid', action='store_true',
                        help='Training using crossfold')
    parser.add_argument('--fold', type=int, default=1, metavar='N',
                        help='Valid fold num (1~5)')
    parser.add_argument('--resume', action='store_true',
                        help='resume training by loading last snapshot')
    parser.add_argument('--save-to-mem', action='store_true',
                        help='Save the training images to memory for speedup')

    args = parser.parse_args()
    args.patch_size = list(map(int, args.patch_size))
    batch_size = args.batch_size
    patch_size = args.patch_size

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    logging.info(args.gpu)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Torch use the device: %s" % device)

    # Create model and check if we want to resume training
    
    if args.model == 'unet':
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

    train_dataset = CardiacDataset2ch(args.dataset, patch_size = patch_size, subset='train', crossvalid=args.crossvalid, valid_fold=args.fold)
    valid_dataset = CardiacDataset2ch(args.dataset, patch_size = patch_size, subset='valid', crossvalid=args.crossvalid, valid_fold=args.fold)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, generator=generator)
    train_iterator = iter(train_loader)

    total_iteration = args.epoches * len(train_loader)
    train_interval = args.log_interval * len(train_loader) 

    logging.info(f"total iter: {total_iteration}")

    # optimizer
    #optimizer = optim.Adam(model.parameters(), lr=args.lr )
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, nesterov=True)
    #scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=(args.epoches // 3)*len(train_loader), T_mult=1, eta_max=args.lr,  T_up=5*len(train_loader), gamma=0.316)
    #scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.epoches *len(train_loader), T_mult=1, eta_max=args.lr,  T_up= 5 *len(train_loader), gamma=0.316)
    iteration = 1
    best_train_loss, best_valid_loss = float('inf'), float('inf')

    if args.resume:
        logging.info("Resume Training: Load states from latest checkpoint.")
        checkpoint = torch.load(os.path.join(args.checkpoints, 'latest_checkpoints_' + args.identifier +'.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        iteration = checkpoint['iteration']
        best_train_loss = checkpoint['best_train_loss']
        best_valid_loss = checkpoint['best_valid_loss']
                
    # Start Training
    writer = SummaryWriter('runs2/'+args.identifier+'/'+str(time.time()))
    
    epoch_train_loss = []
    epoch_valid_loss = []

    set_seed(args.seed)
    start_time = time.time()
    cnt = 0
    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.weights, exist_ok=True)

    while iteration <= total_iteration:
        try:
            img_patch, label_patch = next(train_iterator)

        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            train_iterator = iter(train_loader)
            img_patch, label_patch = next(train_iterator)
            # polynomial lr schedule
            optimizer.param_groups[0]['lr'] = args.lr * (1 - iteration / total_iteration)**0.9

        t_segloss = train(model, device, img_patch, label_patch, optimizer, writer, args.model)

        epoch_train_loss.append(t_segloss)

        if (iteration % train_interval == 0):
            avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)

            logging.info(f'Iter {iteration / train_interval}-{total_iteration / train_interval}: \t Loss: {avg_train_loss:.6f}\t')

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                logging.info(f'--- Saving model at Avg Train Loss:{avg_train_loss:.6f}  ---')
                torch.save(model.state_dict(), os.path.join(args.weights, 'best_train_' + args.identifier +'.pth'))

            # validation process
            valid_iterator = iter(valid_loader)
            for i in range(len(valid_loader)):
                img_patch, label_patch = next(valid_iterator)
            
                v_segloss = valid(model, device, img_patch, label_patch, args.model, cnt)
                cnt +=1

                epoch_valid_loss.append(v_segloss)

            avg_valid_loss = sum(epoch_valid_loss) / (len(epoch_valid_loss) + 1e-6)

            logging.info(f'Iter {iteration / train_interval}-{total_iteration / train_interval} eval: \t Loss: {avg_valid_loss:.6f}\t')

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                logging.info(f'--- Saving model at Avg Valid Loss:{avg_valid_loss:.6f}  ---')
                torch.save(model.state_dict(), os.path.join(args.weights, 'best_valid_' + args.identifier +'.pth'))

            writer.add_scalar('valid/total_loss', avg_valid_loss, iteration)
            # save snapshot for resume training
            logging.info('--- Saving snapshot ---')
            torch.save({
                'iteration': iteration+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                #'scheduler_state_dict': scheduler.state_dict(),
                'best_train_loss': best_train_loss,
                'best_valid_loss': best_valid_loss,
            },
                os.path.join(args.checkpoints, 'latest_checkpoints_' + args.identifier +'.pth'))

            # Save the checkpoints for each 1000 epoches
            if not os.path.exists(args.checkpoints):
                os.makedirs(args.checkpoints)
                
            if(iteration % (len(train_loader)*1000) == 0):
                torch.save({
                    'iteration': iteration+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    #'scheduler_state_dict': scheduler.state_dict(),
                    'best_train_loss': best_train_loss,
                    'best_valid_loss': best_valid_loss,
                    },
                    os.path.join(args.checkpoints, 'latest_checkpoints_' + args.identifier + '_' + str(int(iteration / len(train_loader)))+'.pth'))
            
            logging.info(f"--- {time.time() - start_time} seconds ---")

            epoch_train_loss = []
            epoch_valid_loss = []

            start_time = time.time()
        iteration += 1
            
    writer.close()


