#!/bin/bash

model=HFTrans2 # model selection (HFTrans or unet or HFTrans2)
gpu_=3
epoch=2500
val_case=34  # 1, 5, 16, 19, 34
log_interval=100

# Loop over different values of train_round
for train_round in {1..10}; do  # Replace 5 with the maximum value of train_round you want
    seed=$train_round
    
    identifier=${model}_stageTwo_case${val_case}_${epoch}_round${train_round}

    python train_HFTrans2.py \
    --dataset /home/shqin/cardiac_seg_HFTrans2/dataset/2channel \
    --crossvalid \
    --fold $val_case \
    --patch-size 160 160 160 \
    --batch-size 2 \
    --epoches $epoch \
    --log-interval $log_interval \
    --lr 1e-3 \
    --gpu $gpu_ \
    --seed $seed \
    --model $model \
    --identifier $identifier \
    --extention nrrd \
    > ./log/${identifier}_FX.out

done


