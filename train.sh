#!/bin/bash

set -e

python train.py \
    --threads 12 \
    --gpu 0 \
    --dataset lidarcap \
    --bs 2 \
    --eval_bs 3 \
    --epochs 236 \
    --lr 0.0001 \
    # --resume best-valid-loss.pth
