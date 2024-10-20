#!/bin/bash

model=HFTrans
seed=0
fold_num=5
weight='/home/shqin/cardiac_seg_HFTrans2/checkpoints/latest_checkpoints_HFTrans_stageTwoCT_case'$fold_num'_2500_round'

python ./eval_CT.py \
--dataset /home/shqin/cardiac_seg_HFTrans2/dataset/ct2smooth \
--fold $fold_num \
--patch-size 128 128 128 \
--gpu 1 \
--seed $seed \
--model $model \
--weight $weight \
--extention nrrd \
--tta \
--trained-size 160 \
--resize


