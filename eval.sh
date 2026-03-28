#!/bin/bash

set -e

# ============================================================================
# LidarCap 评估脚本 - 从配置文件读取所有参数
# ============================================================================

EXPERIMENT_NAME="${1:-default}"
OUTPUT_DIR="${OUTPUT_DIR:-./exp/$EXPERIMENT_NAME}"

echo "=========================================="
echo "LidarCap 评估启动"
echo "=========================================="
echo ""
echo "实验名称: $EXPERIMENT_NAME"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 执行评估
python train.py --config-dir config --eval --visual

# 整理输出文件
mkdir -p "$OUTPUT_DIR"
mv ./visual/* "$OUTPUT_DIR/" 2>/dev/null || true
mv ./eval/* "$OUTPUT_DIR/" 2>/dev/null || true

echo ""
echo "=========================================="
echo "评估完成!"
echo "输出文件保存在: $OUTPUT_DIR"
echo "=========================================="