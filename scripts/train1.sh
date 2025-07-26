#!/bin/bash

##################### CIFAR-10 Mismatch Ratio Experiments #####################

# CIFAR-10 mismatch ratio 0.3
## stage 1
python train_stage1.py \
 --dataset cifar10 \
 --mismatch-ratio 0.3 \
 --arch wideresnet \
 --seed 3 \
 --batch-size 64 \
 --expand-labels \
 --total-steps 50000 \
 --eval-step 500 \
 --use-ema \
 --out results/cifar10_ratio0.3/stage1

## stage 2
python train_stage2.py \
--dataset cifar10 \
--mismatch-ratio 0.3 \
--arch wideresnet \
--seed 3 \
--batch-size 64 \
--expand-labels \
--mu 5 \
--total-steps 200000 \
--eval-step 1000 \
--use-ema \
--resume results/cifar10_ratio0.3/stage1/checkpoint.pth.tar \
--out results/cifar10_ratio0.3/stage2



