#!/bin/bash

###############################################################################
# LidarCap 数据集缓存制作脚本
###############################################################################

# 默认配置
DATA_DIR="${DATA_DIR:-/mnt/66bc2970-1c9c-4d90-a36c-2c7ecd0299d6/datasets/lidarhuman26M}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}}"
SEQLEN="${SEQLEN:-16}"
NPOINTS="${NPOINTS:-512}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"

# 压缩设置 (设为空字符串禁用)
COMPRESS="${COMPRESS:---compress}"

# 显示帮助信息
show_help() {
    cat << EOF
使用方法: $0 [选项]

选项:
    -d, --data-dir PATH    原始数据目录 (默认: $DATA_DIR)
    -o, --output-dir PATH   输出目录 (默认: $OUTPUT_DIR)
    -s, --seqlen N          序列长度 (默认: $SEQLEN)
    -n, --npoints N         点云点数 (默认: $NPOINTS)
    -c, --chunk-size N      分块大小 (默认: $CHUNK_SIZE)
    --no-compress           禁用压缩
    -h, --help              显示此帮助信息

环境变量:
    DATA_DIR                原始数据目录
    OUTPUT_DIR              输出目录
    SEQLEN                  序列长度
    NPOINTS                 点云点数
    CHUNK_SIZE              分块大小
    COMPRESS                压缩选项 (空=禁用, --compress=启用)

示例:
    # 使用默认配置
    $0

    # 指定数据目录和输出目录
    $0 -d /path/to/data -o ./mycache

    # 调整参数
    $0 -d /path/to/data -s 32 -n 1024

    # 禁用压缩
    $0 --no-compress

    # 使用环境变量
    DATA_DIR=/path/to/data OUTPUT_DIR=./cache $0
EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--seqlen)
            SEQLEN="$2"
            shift 2
            ;;
        -n|--npoints)
            NPOINTS="$2"
            shift 2
            ;;
        -c|--chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --no-compress)
            COMPRESS=""
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "错误: 未知选项 $1"
            echo "使用 $0 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查Python脚本是否存在
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/generate.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "错误: 找不到Python脚本 $PYTHON_SCRIPT"
    exit 1
fi

# 构建命令
CMD="python $PYTHON_SCRIPT --data-dir $DATA_DIR --output-dir $OUTPUT_DIR --seqlen $SEQLEN --npoints $NPOINTS --chunk-size $CHUNK_SIZE"

# 添加压缩选项
if [[ -n "$COMPRESS" ]]; then
    CMD="$CMD --compress"
fi

# 显示配置信息
echo "=========================================="
echo "  LidarCap 数据集缓存制作"
echo "=========================================="
echo ""
echo "配置信息:"
echo "  数据目录:   $DATA_DIR"
echo "  输出目录:   $OUTPUT_DIR"
echo "  序列长度:   $SEQLEN"
echo "  点云点数:   $NPOINTS"
echo "  分块大小:   $CHUNK_SIZE"
echo "  压缩:       $([ -n "$COMPRESS" ] && echo "启用" || echo "禁用")"
echo ""
echo "执行命令:"
echo "  $CMD"
echo ""
echo "=========================================="
echo ""

# 执行命令
$CMD

# 检查执行结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "  缓存制作完成!"
    echo "=========================================="
    echo ""
    echo "输出文件:"
    echo "  训练集: $OUTPUT_DIR/lidarcap_train.hdf5"
    echo "  测试集: $OUTPUT_DIR/lidarcap_test.hdf5"
    echo ""
    echo "更新 base.yaml 配置:"
    echo "  TrainDataset:"
    echo "    dataset_path: '$OUTPUT_DIR/lidarcap_train.hdf5'"
    echo ""
    echo "  TestDataset:"
    echo "    dataset_path: '$OUTPUT_DIR/lidarcap_test.hdf5'"
    echo ""
else
    echo ""
    echo "错误: 缓存制作失败"
    exit 1
fi
