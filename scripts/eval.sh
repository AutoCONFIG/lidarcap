#!/bin/bash
#
# 评估STM-LiDARCap模型
#

set -e

MODEL=""
GPU=""

show_help() {
    cat << EOF
评估STM-LiDARCap模型

用法: ./scripts/eval.sh -m <模型路径> [-g GPU]

选项:
  -m, --model PATH   模型权重路径 (必需)
  -g, --gpu GPU      GPU ID (默认: 从配置读取)
  -h, --help         显示帮助

示例:
  ./scripts/eval.sh -m output/run_202604011059/model/best-valid-loss.pth
  ./scripts/eval.sh -m output/run_202604011059/model/best-valid-loss.pth -g 1
EOF
}

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

if [ -z "$MODEL" ]; then
    echo "错误: 必须指定模型路径"
    show_help
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "错误: 模型文件不存在: $MODEL"
    exit 1
fi

cd "$(dirname "$0")/.."

if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU
fi

python eval.py "$MODEL"
