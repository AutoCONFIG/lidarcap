#!/bin/bash

# LiDARCap 数据集可视化工具启动脚本
# 用于快速配置参数并启动可视化

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
DATASET_PATH=""
DATASET_ID=""
FRAME_IDX=0
START_FRAME=0
END_FRAME=""
MODE="both"
OUTPUT_DIR="./visualization"
VISUALIZER="open3d"
DEVICE="cuda"
TOOL="single"  # single 或 sequence

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISUALIZATION_DIR="$SCRIPT_DIR/tools/visualization"

# 显示帮助信息
show_help() {
    echo -e "${GREEN}LiDARCap 数据集可视化工具${NC}"
    echo ""
    echo "用法: ./visualize.sh [选项]"
    echo ""
    echo "选项:"
    echo "  -p, --path PATH         HDF5数据集路径 (必填)"
    echo "  -i, --id ID             数据集ID (如 0001)"
    echo "  -f, --frame INDEX       单帧模式: 帧索引 (默认: 0)"
    echo "  -s, --start INDEX       序列模式: 起始帧 (默认: 0)"
    echo "  -e, --end INDEX         序列模式: 结束帧 (默认: 全部)"
    echo "  -m, --mode MODE         可视化模式: pointcloud/smlp/both (默认: both)"
    echo "  -o, --output DIR        输出目录 (默认: ./visualization)"
    echo "  -t, --tool TOOL         工具类型: single/sequence (默认: single)"
    echo "  -v, --visualizer TYPE   可视化库: open3d/matplotlib (默认: open3d)"
    echo "  -d, --device DEVICE     计算设备: cuda/cpu (默认: cuda)"
    echo "  -h, --help              显示帮助信息"
    echo ""
    echo "示例:"
    echo "  # 可视化单帧"
    echo "  ./visualize.sh -p data/0001.hdf5 -i 0001 -f 10"
    echo ""
    echo "  # 可视化序列"
    echo "  ./visualize.sh -p data/0001.hdf5 -t sequence -s 0 -e 100"
    echo ""
    echo "  # 保存为文件而不是显示"
    echo "  ./visualize.sh -p data/0001.hdf5 -o ./output"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--path)
            DATASET_PATH="$2"
            shift 2
            ;;
        -i|--id)
            DATASET_ID="$2"
            shift 2
            ;;
        -f|--frame)
            FRAME_IDX="$2"
            shift 2
            ;;
        -s|--start)
            START_FRAME="$2"
            shift 2
            ;;
        -e|--end)
            END_FRAME="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -t|--tool)
            TOOL="$2"
            shift 2
            ;;
        -v|--visualizer)
            VISUALIZER="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 检查必要参数
if [ -z "$DATASET_PATH" ]; then
    echo -e "${RED}错误: 请指定数据集路径 (-p)${NC}"
    show_help
    exit 1
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo -e "${RED}错误: 文件不存在: $DATASET_PATH${NC}"
    exit 1
fi

# 如果没有指定数据集ID，从文件名推断
if [ -z "$DATASET_ID" ]; then
    DATASET_ID=$(basename "$DATASET_PATH" .hdf5)
    echo -e "${YELLOW}自动推断数据集ID: $DATASET_ID${NC}"
fi

# 检查可视化工具目录是否存在
if [ ! -d "$VISUALIZATION_DIR" ]; then
    echo -e "${RED}错误: 可视化工具目录不存在: $VISUALIZATION_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}====================================${NC}"
echo -e "${GREEN}LiDARCap 数据集可视化工具${NC}"
echo -e "${BLUE}====================================${NC}"
echo "数据集路径: $DATASET_PATH"
echo "数据集ID: $DATASET_ID"
echo "工具类型: $TOOL"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 根据工具类型执行
if [ "$TOOL" = "single" ]; then
    echo -e "${GREEN}启动单帧可视化工具...${NC}"
    echo "帧索引: $FRAME_IDX"
    echo "可视化模式: $MODE"
    echo ""
    
    cd "$SCRIPT_DIR"
    python tools/visualization/visualize_dataset.py \
        --dataset_path "$DATASET_PATH" \
        --dataset_id "$DATASET_ID" \
        --frame_idx "$FRAME_IDX" \
        --mode "$MODE" \
        --visualizer "$VISUALIZER" \
        ${OUTPUT_DIR:+--save_dir "$OUTPUT_DIR"}

elif [ "$TOOL" = "sequence" ]; then
    echo -e "${GREEN}启动序列可视化工具...${NC}"
    echo "起始帧: $START_FRAME"
    [ -n "$END_FRAME" ] && echo "结束帧: $END_FRAME"
    echo ""
    
    cd "$SCRIPT_DIR"
    if [ -n "$END_FRAME" ]; then
        python tools/visualization/visualize_sequence.py \
            --dataset_path "$DATASET_PATH" \
            --dataset_id "$DATASET_ID" \
            --start_frame "$START_FRAME" \
            --end_frame "$END_FRAME" \
            --output_dir "$OUTPUT_DIR" \
            --device "$DEVICE"
    else
        python tools/visualization/visualize_sequence.py \
            --dataset_path "$DATASET_PATH" \
            --dataset_id "$DATASET_ID" \
            --start_frame "$START_FRAME" \
            --output_dir "$OUTPUT_DIR" \
            --device "$DEVICE"
    fi
else
    echo -e "${RED}错误: 未知的工具类型: $TOOL${NC}"
    echo "支持的工具类型: single, sequence"
    exit 1
fi

echo ""
echo -e "${GREEN}可视化完成!${NC}"

# 如果是保存模式，提示用户
if [ -n "$OUTPUT_DIR" ] && [ "$TOOL" = "sequence" ]; then
    echo "输出文件保存在: $OUTPUT_DIR"
    
    # 询问是否生成视频
    if command -v ffmpeg &> /dev/null; then
        echo ""
        read -p "是否使用ffmpeg生成视频? (y/n): " gen_video
        if [ "$gen_video" = "y" ] || [ "$gen_video" = "Y" ]; then
            video_output="$OUTPUT_DIR/${DATASET_ID}_sequence.mp4"
            ffmpeg -framerate 10 -pattern_type glob -i "$OUTPUT_DIR/*.png" \
                   -c:v libx264 -pix_fmt yuv420p "$video_output"
            echo "视频已保存到: $video_output"
        fi
    fi
fi
