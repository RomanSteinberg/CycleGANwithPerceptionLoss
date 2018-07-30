#!/usr/bin/env bash
pip install visdom dominate

python3 train.py --name 20 --dataroot ./input --model cycle_gan --lambda_feat_AfB 0.1 --lambda_feat_BfA 0.1 --no_dropout --batchSize 1 \
 --no_flip --resize_or_crop 'resize_and_crop' --max_dataset_size 7500 \
 --display_id 0 --print_freq 1000 --display_freq 7500
