#!/bin/bash

model=unet
seed=0
weight=./checkpoints/latest_checkpoints_unet_10000.pth

python ./eval.py \
--dataset /home/jcho/dataset/cardiac/valid \
--patch-size 160 160 160 \
--gpu 2 \
--seed $seed \
--model $model \
--weight $weight \
--tta \
--extention nrrd


