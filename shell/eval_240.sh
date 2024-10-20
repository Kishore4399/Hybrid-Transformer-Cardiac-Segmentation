#!/bin/bash

model=HFTrans
seed=0
weight=./checkpoints/latest_checkpoints_HFTrans_240_10000.pth

python ./eval.py \
--dataset /home/jcho/dataset/cardiac/valid \
--patch-size 160 160 160 \
--gpu 3 \
--seed $seed \
--model $model \
--weight $weight \
--tta \
--extention nrrd \
--trained-size 240 \
--resize


