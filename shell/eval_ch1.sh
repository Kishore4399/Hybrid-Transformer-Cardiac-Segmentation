#!/bin/bash

model=HFTrans
seed=0
weight=./checkpoints/latest_checkpoints_HFTranschannel1_case1_5000.pth

python ./eval_ch1.py \
--dataset /home/shqin/dataset/1channel/case1/valid/ \
--patch-size 160 160 160 \
--gpu 3 \
--seed $seed \
--model $model \
--weight $weight \
--tta \
--extention nrrd \
--trained-size 160 \
--resize


