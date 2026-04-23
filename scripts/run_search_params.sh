#!/bin/bash
# ============================================================
# 投影参数自动搜索预处理工具 (GPU加速 v2)
# ============================================================
# 修改下方参数后直接运行即可
# ============================================================

cd /media/yun/de2a43ce-446c-4a62-99b3-8ddc6ea1ef87/lidarcap

# ============================================================
# 参数配置区 - 根据需要修改
# ============================================================

# 数据集路径
DATA_ROOT="/media/yun/de2a43ce-446c-4a62-99b3-8ddc6ea1ef87/lidarhuman26M"

# 输出目录 (留空则默认输出到数据集目录下的 projection_params/)
OUTPUT_DIR=""

# 要处理的序列 (留空则处理全部，多个序列用逗号分隔，如: "34,35,36")
SEQUENCES="24,27"

# ===== 第一帧搜索范围 (大范围搜索) =====
# 注意: 程序会自动估计质心偏移，这里的范围是相对于质心偏移的搜索窗口
FIRST_DX_MIN=-30
FIRST_DX_MAX=30
FIRST_DY_MIN=-30
FIRST_DY_MAX=30
FIRST_SCALE_MIN=0.75
FIRST_SCALE_MAX=1.25

# ===== 后续帧微调范围 (基于前一帧结果) =====
REFINE_DX_MIN=-10
REFINE_DX_MAX=10
REFINE_DY_MIN=-10
REFINE_DY_MAX=10
REFINE_SCALE_MIN=0.93
REFINE_SCALE_MAX=1.07

# ===== 搜索步长 (越小越精细，但越慢) =====
STEP=0.1           # 平移步长 (像素)
SCALE_STEP=0.1    # 缩放步长

# ============================================================
# 执行命令 (一般无需修改)
# ============================================================

CMD="python tools/auto_search_projection_params.py \
    --data_root $DATA_ROOT \
    --first_dx $FIRST_DX_MIN $FIRST_DX_MAX \
    --first_dy $FIRST_DY_MIN $FIRST_DY_MAX \
    --first_scale $FIRST_SCALE_MIN $FIRST_SCALE_MAX \
    --refine_dx $REFINE_DX_MIN $REFINE_DX_MAX \
    --refine_dy $REFINE_DY_MIN $REFINE_DY_MAX \
    --refine_scale $REFINE_SCALE_MIN $REFINE_SCALE_MAX \
    --step $STEP \
    --scale_step $SCALE_STEP"

# 添加可选参数
if [ -n "$SEQUENCES" ]; then
    CMD="$CMD --seqs $SEQUENCES"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

echo "============================================================"
echo "投影参数自动搜索 (GPU加速 v2)"
echo "============================================================"
echo "数据集: $DATA_ROOT"
echo "序列: ${SEQUENCES:-全部}"
echo "第一帧范围: dx=[$FIRST_DX_MIN,$FIRST_DX_MAX], dy=[$FIRST_DY_MIN,$FIRST_DY_MAX], scale=[$FIRST_SCALE_MIN,$FIRST_SCALE_MAX]"
echo "微调范围: dx=[$REFINE_DX_MIN,$REFINE_DX_MAX], dy=[$REFINE_DY_MIN,$REFINE_DY_MAX], scale=[$REFINE_SCALE_MIN,$REFINE_SCALE_MAX]"
echo "步长: step=$STEP, scale_step=$SCALE_STEP"
echo "============================================================"
echo ""

# 执行
eval $CMD
