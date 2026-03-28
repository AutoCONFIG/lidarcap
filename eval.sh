#!/bin/bash
python train.py --gpu 7 --dataset lidarcap_39 --eval_bs 4 --threads 4 --epochs 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --dataset lidarcap_7 --eval_bs 2 --threads 4 --epochs 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --dataset lidarcap_24 --eval_bs 8 --threads 4 --epochs 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --dataset lidarcap_29 --eval_bs 8 --threads 4 --epochs 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --dataset lidarcap_41 --eval_bs 8 --threads 4 --epochs 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --dataset kitti_15 --eval_bs 8 --threads 4 --epochs 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --dataset kitti_35 --eval_bs 8 --threads 4 --epochs 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --dataset kitti_53 --eval_bs 8 --threads 4 --epochs 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --dataset kitti_56 --eval_bs 8 --threads 4 --epochs 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --dataset kitti_57 --eval_bs 8 --threads 4 --epochs 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --dataset kitti_84 --eval_bs 8 --threads 4 --epochs 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug
# python train.py --gpu 7 --dataset waymo --eval_bs 8 --threads 4 --epochs 10000 --regress $3 --train_step second --ckpt_path output/$1/model/best-train-loss.pth --eval --visual --debug


OUTPUT_DIR="${OUTPUT_DIR:-./exp/$2}"
mkdir -p "$OUTPUT_DIR"
mv ./visual/$1/* "$OUTPUT_DIR/" 2>/dev/null || true
mv ./eval/$1/* "$OUTPUT_DIR/" 2>/dev/null || true