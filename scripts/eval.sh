#!/bin/bash
#
# 评估STM-LiDARCap模型
# 用法: ./scripts/eval.sh [选项]
#

set -e

# 默认参数
MODEL=""
GPU="0"
SEQS=""  # 留空表示评估所有序列

# 显示帮助信息
show_help() {
    echo "===================================================================="
    echo "评估STM-LiDARCap模型"
    echo "===================================================================="
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model PATH      模型权重路径 (必需)"
    echo "  -g, --gpu GPU         GPU ID (默认: 0)"
    echo "  -s, --seqs NUMS       测试序列索引，逗号分隔 (默认: 全部)"
    echo "  -h, --help            显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 评估指定模型"
    echo "  $0 -m output/exp1/best_model.pth"
    echo ""
    echo "  # 指定GPU和序列"
    echo "  $0 -m output/exp1/best_model.pth -g 1 -s 7,24,29"
    echo ""
    echo "输出:"
    echo "  评估结果包含 MPJPE, PA-MPJPE 等指标"
    echo ""
    echo "===================================================================="
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU="$2"
            shift 2
            ;;
        -s|--seqs)
            SEQS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "错误: 未知参数 $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$MODEL" ]; then
    echo "错误: 必须指定模型路径"
    echo ""
    show_help
    exit 1
fi

# 检查模型文件
if [ ! -f "$MODEL" ]; then
    echo "错误: 模型文件不存在: $MODEL"
    exit 1
fi

# 显示信息
echo "===================================================================="
echo "开始评估"
echo "===================================================================="
echo "模型路径: $MODEL"
echo "GPU: $GPU"
[ -n "$SEQS" ] && echo "测试序列: $SEQS"
echo "===================================================================="
echo ""

# 运行评估
cd "$(dirname "$0")/.."

if [ -n "$SEQS" ]; then
    # 指定序列评估
    IFS=',' read -ra SEQ_ARRAY <<< "$SEQS"
    for seq in "${SEQ_ARRAY[@]}"; do
        echo "评估序列 $seq..."
        CUDA_VISIBLE_DEVICES=$GPU python -c "
from eval import eval
from tools import util
import metric
import torch

model_name = '$MODEL'
seq_idx = $seq

pred_poses = util.get_pred_poses(model_name, seq_idx)
gt_poses = util.get_gt_poses(seq_idx)
pred_poses = pred_poses[:len(gt_poses)]

print(f'序列 {seq_idx} 结果:')
metric.output_metric(pred_poses, gt_poses)
"
        echo ""
    done
else
    # 评估所有默认序列
    CUDA_VISIBLE_DEVICES=$GPU python eval.py "$MODEL"
fi

echo ""
echo "===================================================================="
echo "评估完成！"
echo "===================================================================="
