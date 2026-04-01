#!/bin/bash
#
# 可视化Transformer和Mamba的中间结果
# 用法: ./scripts/vis_intermediate.sh [选项]
#

set -e

# 默认参数
MODEL=""
SEQ=0
OUTPUT_DIR="vis_results"

# 显示帮助信息
show_help() {
    echo "===================================================================="
    echo "可视化Transformer和Mamba中间结果"
    echo "===================================================================="
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model PATH      模型权重路径 (必需)"
    echo "  -s, --seq NUM         测试序列索引 (默认: 0)"
    echo "  -o, --output DIR      输出目录 (默认: vis_results)"
    echo "  -h, --help            显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -m output/best_model.pth -s 0"
    echo "  $0 --model output/checkpoint.pth --seq 7 --output my_vis"
    echo ""
    echo "输出文件:"
    echo "  - *_spatial.png      空间特征可视化 (Transformer效果)"
    echo "  - *_temporal.png     时序特征可视化 (Mamba效果)"
    echo "  - *_comparison.png   两者对比分析 (协同效果)"
    echo "  - *_skeleton.gif     骨架动画"
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
        -s|--seq)
            SEQ="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
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

# 检查模型文件是否存在
if [ ! -f "$MODEL" ]; then
    echo "错误: 模型文件不存在: $MODEL"
    exit 1
fi

# 运行可视化
echo "===================================================================="
echo "开始可视化"
echo "===================================================================="
echo "模型路径: $MODEL"
echo "序列索引: $SEQ"
echo "输出目录: $OUTPUT_DIR"
echo "===================================================================="
echo ""

cd "$(dirname "$0")/.."
python vis/intermediate.py \
    --model "$MODEL" \
    --seq "$SEQ" \
    --output "$OUTPUT_DIR"

echo ""
echo "===================================================================="
echo "可视化完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "===================================================================="
