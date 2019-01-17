#!/usr/bin/env bash

source ~/.virtualenvs/av3.6/bin/activate

python3 train.py \
 --name 1 \
 --dataroot ./input2 \
 --model cycle_gan \
 --no_dropout \
 --padding_type "zero" \
 --which_model_netG 'resnet_12blocks' \
 --lambda_A 15.0 \
 --lambda_feat_AfB 0.3 \
 --identity 0.1 \
 --skip_gen_connection \
 --batchSize 1 \
 --no_flip \
 --resize_or_crop 'resize_and_crop' \
 --max_dataset_size 1843 \
 --display_id 0 \
 --print_freq 500 \
 --display_freq 1000 \
 --norm "instance" \
 --data_description "1843\ segmented\ and\ scaled\ photos\ and\ not\ augmented\ avatars\ 256x256" \
 --use_shuffle_conv true
