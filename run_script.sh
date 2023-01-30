#!/usr/bin/env bash

python train.py --flagfile configs/cifar10c_1pct.cfg --flagfile configs/v_ours.cfg --data_dir /data --log_path /data/log --wandb --exp v+selecmix --gpu 0