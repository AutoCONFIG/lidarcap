#!/bin/bash

set -e

# ============================================================================
# LidarCap 训练脚本 - 快速启动配置
# ============================================================================
# 说明：直接运行 bash train.sh 即可启动训练
#      在下方配置区域修改参数即可
#      训练策略参数（学习率、早停、梯度裁剪等）请在 base.yaml 中配置
# ============================================================================

# ============================================================================
# 📋 训练配置区域（修改此处参数）
# ============================================================================

# ==================== 必需参数 ====================
# 数据集名称
DATASET="lidarcap"

# ==================== 基本训练参数 ====================
# GPU ID (多个GPU用空格分隔，如 "0 1"，-1 表示使用CPU)
GPU="0"

# 训练批次大小
BS=16

# 评估批次大小
EVAL_BS=16

# 线程数
THREADS=4

# 训练轮数
EPOCHS=200

# 输出目录
OUTPUT_DIR="output"

# 日志间隔
LOG_INTERVAL=100

# ==================== 模式配置 ====================
# 调试模式（true/false）
DEBUG="false"

# 评估模式（true/false）
EVAL_MODE="false"

# 可视化模式（true/false）
VISUAL="false"

# 恢复训练路径（留空表示新建训练）
RESUME_PATH=""

# 检查点路径（用于评估/可视化，留空则自动选择）
CKPT_PATH=""

# ============================================================================
# 🚀 启动训练
# ============================================================================

CMD="python train.py"

# 必需参数
CMD="$CMD --dataset $DATASET"

# 基本训练参数
CMD="$CMD --gpu $GPU"
CMD="$CMD --bs $BS"
CMD="$CMD --eval_bs $EVAL_BS"
CMD="$CMD --threads $THREADS"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --log_interval $LOG_INTERVAL"

# 模式参数
if [ "$DEBUG" = "true" ]; then
    CMD="$CMD --debug"
fi

if [ "$EVAL_MODE" = "true" ]; then
    CMD="$CMD --eval"
fi

if [ "$VISUAL" = "true" ]; then
    CMD="$CMD --visual"
fi

# 恢复训练
if [ -n "$RESUME_PATH" ]; then
    CMD="$CMD --resume $RESUME_PATH"
fi

# 检查点路径
if [ -n "$CKPT_PATH" ]; then
    CMD="$CMD --ckpt_path $CKPT_PATH"
fi

# ============================================================================
# 执行命令
# ============================================================================
echo "=========================================="
echo "🚀 LidarCap 训练启动"
echo "=========================================="
echo ""
echo "配置信息："
echo "  数据集: $DATASET"
echo "  GPU: $GPU"
echo "  批次大小: $BS"
echo "  评估批次大小: $EVAL_BS"
echo "  线程数: $THREADS"
echo "  训练轮数: $EPOCHS"
echo "  输出目录: $OUTPUT_DIR"
echo ""
echo "运行模式："
echo "  调试: $DEBUG"
echo "  评估: $EVAL_MODE"
echo "  可视化: $VISUAL"
if [ -n "$RESUME_PATH" ]; then
    echo "  恢复路径: $RESUME_PATH"
fi
if [ -n "$CKPT_PATH" ]; then
    echo "  检查点: $CKPT_PATH"
fi
echo ""
echo "=========================================="
echo ""
echo "执行命令："
echo "$CMD"
echo ""
echo "=========================================="
echo ""

eval $CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 训练完成！"
else
    echo "❌ 训练失败，退出码: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
