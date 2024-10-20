#!/bin/bash

model=HFTrans
seed=0
fold_num=34
weight='/home/shqin/cardiac_seg_HFTrans2/checkpoints/latest_checkpoints_HFTrans_stageTwoCT_case'$fold_num'_5000.pth'

python ./eval_ch1.py \
--dataset /home/shqin/cardiac_seg_HFTrans2/dataset/ct2smooth \
--patch-size 128 128 128 \
--fold $fold_num \
--gpu 1 \
--seed $seed \
--model $model \
--weight $weight \
--tta \
--extention nrrd \
--trained-size 160 \
--resize

