#!/bin/bash
# =============================================================================
# 数据集可视化启动脚本
#
# 用法:
#   bash scripts/vis_dataset.sh <mode> [options]
#
# 模式:
#   single    - 可视化单帧 (点云+关节+统计)
#   sequence  - 可视化整个序列 (16帧骨架轨迹)
#   animation - 生成骨架动画GIF
#   stats     - 查看数据集统计信息
#
# 示例:
#   bash scripts/vis_dataset.sh single              # 默认: 训练集第0个样本第0帧
#   bash scripts/vis_dataset.sh single --test       # 测试集
#   bash scripts/vis_dataset.sh single --index 5    # 第5个样本
#   bash scripts/vis_dataset.sh sequence --count 3  # 可视化3个序列
#   bash scripts/vis_dataset.sh animation           # 生成动画
#   bash scripts/vis_dataset.sh stats               # 统计信息
# =============================================================================

set -e

# 默认参数
MODE=${1:-single}
SPLIT="train"
INDEX=0
FRAME=0
START=0
COUNT=5
OUTPUT="output/vis_dataset"
PRELOAD=""

# 解析参数
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            SPLIT="test"
            shift
            ;;
        --train)
            SPLIT="train"
            shift
            ;;
        --index)
            INDEX=$2
            shift 2
            ;;
        --frame)
            FRAME=$2
            shift 2
            ;;
        --start)
            START=$2
            shift 2
            ;;
        --count)
            COUNT=$2
            shift 2
            ;;
        --output)
            OUTPUT=$2
            shift 2
            ;;
        --preload)
            PRELOAD="--preload"
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 创建输出目录
mkdir -p ${OUTPUT}

echo "=============================================="
echo "数据集可视化"
echo "=============================================="
echo "模式: ${MODE}"
echo "数据集: ${SPLIT}"
echo "输出目录: ${OUTPUT}"
echo "=============================================="

# 运行可视化
python vis/dataset.py \
    --mode ${MODE} \
    --split ${SPLIT} \
    --index ${INDEX} \
    --frame ${FRAME} \
    --start ${START} \
    --count ${COUNT} \
    --output ${OUTPUT} \
    ${PRELOAD}

echo "=============================================="
echo "可视化完成! 结果保存在: ${OUTPUT}"
echo "=============================================="
