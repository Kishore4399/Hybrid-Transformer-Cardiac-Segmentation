#!/bin/bash

model=HFTrans # model selection (HFTrans or unet or HFTrans2)
epoch=2000
seed=0 
identifier=$model''stageOne_$epoch
log_interval=100

#nohup
# /jihoon//dataset/cardiac240 \
#python ./train_JihoonOld.py \
python train_HFTrans.py \
--dataset /home/shqin/cardiac_seg_HFTrans2/dataset/stageOne/ \
--patch-size 128 128 128 \
--batch-size 2 \
--epoches $epoch \
--log-interval $log_interval \
--lr 1e-3 \
--gpu 0 \
--seed $seed \
--model $model \
--identifier $identifier \
--extention nrrd \
> ./log/$identifier''_FX.out

