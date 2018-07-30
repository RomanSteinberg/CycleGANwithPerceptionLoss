#!/usr/bin/env bash
pip install visdom
pip install dominate

python3 test.py --name 16 --dataroot ./input --model cycle_gan --no_dropout --display_id 0 --which_epoch "5, 20, 5" \
 --resize_or_crop 'resize'
