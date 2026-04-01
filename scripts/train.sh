#!/bin/bash
#
# 训练STM-LiDARCap模型
# 用法: ./scripts/train.sh [选项]
#

set -e

# 默认参数
CONFIG="configs/base.yaml"
EXP_NAME=""
GPU="0"
RESUME=""
EPOCHS=""
BATCH_SIZE=""
LR=""

# 显示帮助信息
show_help() {
    echo "===================================================================="
    echo "训练STM-LiDARCap模型"
    echo "===================================================================="
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -c, --config PATH     配置文件路径 (默认: configs/base.yaml)"
    echo "  -n, --name NAME       实验名称 (必需)"
    echo "  -g, --gpu GPU         GPU ID (默认: 0, 多GPU: 0,1,2)"
    echo "  -r, --resume PATH     恢复训练的检查点路径"
    echo "  -e, --epochs NUM      训练轮数"
    echo "  -b, --batch NUM       批次大小"
    echo "  -l, --lr RATE         学习率"
    echo "  -h, --help            显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 基本训练"
    echo "  $0 -n exp1"
    echo ""
    echo "  # 指定GPU和配置"
    echo "  $0 -n exp2 -g 0,1 -c configs/base.yaml"
    echo ""
    echo "  # 恢复训练"
    echo "  $0 -n exp1 -r output/exp1/checkpoint_epoch50.pth"
    echo ""
    echo "  # 自定义参数"
    echo "  $0 -n exp3 -e 100 -b 32 -l 0.0001"
    echo ""
    echo "输出:"
    echo "  模型和日志保存在: output/<实验名称>/"
    echo ""
    echo "===================================================================="
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -n|--name)
            EXP_NAME="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--lr)
            LR="$2"
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
if [ -z "$EXP_NAME" ]; then
    echo "错误: 必须指定实验名称"
    echo ""
    show_help
    exit 1
fi

# 检查配置文件
if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

# 构建命令
CMD="CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --name $EXP_NAME"

if [ -n "$RESUME" ]; then
    if [ ! -f "$RESUME" ]; then
        echo "错误: 检查点文件不存在: $RESUME"
        exit 1
    fi
    CMD="$CMD --resume $RESUME"
fi

if [ -n "$EPOCHS" ]; then
    CMD="$CMD --epochs $EPOCHS"
fi

if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
fi

if [ -n "$LR" ]; then
    CMD="$CMD --lr $LR"
fi

# 显示信息
echo "===================================================================="
echo "开始训练"
echo "===================================================================="
echo "实验名称: $EXP_NAME"
echo "配置文件: $CONFIG"
echo "GPU: $GPU"
[ -n "$RESUME" ] && echo "恢复检查点: $RESUME"
[ -n "$EPOCHS" ] && echo "训练轮数: $EPOCHS"
[ -n "$BATCH_SIZE" ] && echo "批次大小: $BATCH_SIZE"
[ -n "$LR" ] && echo "学习率: $LR"
echo "===================================================================="
echo ""
echo "执行命令: $CMD"
echo ""

# 运行训练
cd "$(dirname "$0")/.."
$CMD

echo ""
echo "===================================================================="
echo "训练完成！"
echo "模型保存在: output/$EXP_NAME/"
echo "===================================================================="
