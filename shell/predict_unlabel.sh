#!/bin/bash

model=HFTrans2
seed=0
fold_num=100
weight='/home/shqin/cardiac_seg_HFTrans2/checkpoints/latest_checkpoints_HFTrans2_stageTwo_case'$fold_num'_5000_5000.pth'

python ./eval_unlabel.py \
--dataset /home/shqin/cardiac_seg_HFTrans2/dataset/stageOne \
--fold $fold_num \
--patch-size 160 160 160 \
--gpu 3 \
--seed $seed \
--model $model \
--weight $weight \
--tta \
--extention nrrd \
--trained-size 160 \
--resize


