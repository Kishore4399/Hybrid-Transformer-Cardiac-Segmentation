#!/bin/bash

model=HFTrans2
seed=0
weight=/home/shqin/cardiac_seg_HFTrans2/checkpoints/latest_checkpoints_HFTrans2case2_5000_5000.pth

python ./eval.py \
--dataset /home/shqin/cardiac_seg_HFTrans2/dataset/2channel/case1/valid/ \
--patch-size 160 160 160 \
--gpu 1 \
--seed $seed \
--model $model \
--weight $weight \
--tta \
--extention nrrd \
--trained-size 160 \
--resize
