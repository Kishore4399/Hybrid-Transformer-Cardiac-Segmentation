#!/bin/bash

model=HFTrans2 # model selection (HFTrans or unet or HFTrans2)
epoch=5000
seed=0 
identifier=$model''case1_$epoch
log_interval=100

#nohup
# /jihoon//dataset/cardiac240 \
#python ./train_JihoonOld.py \
python /home/shqin/HFTrans2/train_HFTrans2.py \
--dataset /home/shqin/dataset/case1/ \
--patch-size 160 160 160 \
--batch-size 2 \
--epoches $epoch \
--log-interval $log_interval \
--lr 1e-3 \
--gpu 4 \
--seed $seed \
--model $model \
--identifier $identifier \
--extention nrrd \
> /home/shqin/HFTrans2/log/$identifier''_FX.out

