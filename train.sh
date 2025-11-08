#!/bin/bash

set -e

python train.py \
    --threads 12 \
    --gpu 0 \
    --dataset lidarcap \
    --bs 2 \
    --eval_bs 2 \
    --epochs 123 \
    --lr 0.0001 \
    --resume output/run_202511071055
