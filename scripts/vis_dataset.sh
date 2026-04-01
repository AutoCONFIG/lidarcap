#!/bin/bash
# =============================================================================
# 数据集可视化启动脚本 - 直接修改下方参数后运行
# =============================================================================

# ======================== 在这里修改参数 ========================
MODE="interactive"  # single / sequence / animation / interactive / stats
SPLIT="train"       # train / test
INDEX=2             # 样本索引
FRAME=0             # 帧索引 (single模式)
START=0             # 起始索引 (sequence/animation模式)
COUNT=5             # 数量 (sequence模式)
OUTPUT="output/vis_dataset"
# PRELOAD="--preload"  # 取消注释以预加载整个数据集
# ================================================================

mkdir -p ${OUTPUT}

python vis/dataset.py \
    --mode ${MODE} \
    --split ${SPLIT} \
    --index ${INDEX} \
    --frame ${FRAME} \
    --start ${START} \
    --count ${COUNT} \
    --output ${OUTPUT} \
    ${PRELOAD}
